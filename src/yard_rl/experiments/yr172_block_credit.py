"""YR-172 — 보상을 **블록별 공로 배분**으로 쪼갠다 (학습이 되게 하는 마지막 축).

문제(YR-170 진단): `phi_pre` 가 터미널 전역 스칼라 1개라 **같은 epoch 의 21블록
결정이 전부 같은 보상**을 받는다. "3번 블록이 판 게 나빴고 7번 블록이 안 판 게
좋았다"를 advantage 가 원리적으로 구분하지 못한다(행동 귀속 분산 3.8e-6).

여기서는 각 결정에 **그 블록 자신의 이후 비용 변화**를 준다:

    r2go(블록 b, 시각 t) = −[ (φ_b(끝) − φ_b(t)) + (b 가 t 이후 판 건의 주행비) ] / SCALE

- φ_b = 그 블록의 검열 v2 실현비용(`phi_v2`). Φ_total = Σ_b φ_b + 주행 + 기사대기
  이므로 **합하면 전역과 같은 잣대**다(단위 통일 유지).
- 주행비는 **판 블록에 부과**한다 — 안 그러면 "옮기면 내 비용만 준다"가 된다.

**떠넘기기 위험은 남는다** — 수신 블록이 나빠지는 몫은 그 블록의 보상에만 잡히고
판 블록은 못 느낀다. 실측(YR-164)에서 수신 손실이 소스 이득의 1.3배였으므로
이 위험은 실재한다. 그래서 **관측 항목에 수신 블록 비용 추이를 넣고**, 떠넘기기가
나타나면 다음 축(소스+수신 묶음 귀속)으로 간다. 한 번에 한 축씩.

무대·비용식·집행 정책은 건드리지 않는다 — **학습기 쪽만** 바뀐다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.baselines import _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.profiles import build_h21_profile
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                          ScheduledAnnouncer, TerminalStreamParams,
                                          admission_epochs, build_diurnal,
                                          ensure_time_ledger)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr139_blockq_v4_ppo import phi_v2
from .yr149_load_cells import _sim_from
from .yr151_transfer_ppo import (AdoptedExecFleet, load_adopted_execution_head,
                                 load_kf)
from .yr170_sell_ppo_diurnal import (GRAD_CLIP, MINIBATCH, RET_SCALE, SPACE_ONLY,
                                     SpaceOnly, TRAIN_SEEDS)

OUT = Path("outputs/reports/yr172_block_credit")
SPEC = ".claude/docs/dashboard-task-specs/YR-172-block-credit-reward.md"


class BlockPhiRecorder:
    """epoch 마다 **블록별** φ 를 기록. 전역 합만 남기던 구판을 대체한다.

    비용은 어차피 블록별로 계산해 더하던 것이라(Φ = Σφ_b + …) **추가 계산이 없다** —
    합치기 전 값을 버리지 않고 남기는 것뿐이다.
    """

    def __init__(self):
        self.by_t: dict[float, dict[str, float]] = {}

    def review(self, mbt, t: float) -> None:
        self.by_t[round(t, 6)] = {b: phi_v2(sim, t) for b, sim in mbt.blocks.items()}


def _route_index(layout, ledger) -> dict:
    """블록별 **접미합** 색인 — 결정마다 원장을 다시 훑지 않게 한다.

    구판은 결정 1건마다 원장 전체를 순회해 O(결정수 × 원장크기)였다
    (약 15,000 × 15,000 = 2.25억 회). 여기서는 블록별로 한 번 정렬하고
    뒤에서부터 누적해, 조회를 이분탐색 1회로 만든다.
    """
    from bisect import bisect_left
    by_src: dict[str, list[tuple[float, float]]] = {}
    for e in ledger:
        if (e.get("decision") == "SELL" and e.get("axis") == "SPACE"
                and e.get("dst")):
            src = e["src"]
            c = max(0.0, layout.gate_to_block_s(e["dst"])
                    - layout.gate_to_block_s(src)) / 3600.0
            by_src.setdefault(src, []).append((e["t"], c))
    idx = {}
    for src, rows in by_src.items():
        rows.sort()
        ts = [t for t, _ in rows]
        suf = [0.0] * (len(rows) + 1)
        for i in range(len(rows) - 1, -1, -1):
            suf[i] = suf[i + 1] + rows[i][1]
        idx[src] = (ts, suf, bisect_left)
    return idx


def _route_charge(idx: dict, src: str, after_t: float) -> float:
    """블록 src 가 after_t 이후 확정한 공간 판매의 주행 추가분(비용시간)."""
    got = idx.get(src)
    if got is None:
        return 0.0
    ts, suf, bisect_left = got
    return suf[bisect_left(ts, after_t - 1e-9)]


def build_batch_block(trail: list[dict], phi_by_t: dict, phi_final: dict,
                      ledger: list[dict], layout, *,
                      ret_scale: float = RET_SCALE) -> list[dict]:
    """결정별 (총수익 R, advantage) — **그 블록 자신의** 이후 비용으로 채점한다."""
    idx = _route_index(layout, ledger)
    out = []
    for tr in trail:
        b = tr["src"]
        pre = phi_by_t.get(round(tr["t"], 6))
        if pre is None or b not in pre:
            raise RuntimeError(f"블록 φ 기록에 epoch t={tr['t']} 없음 — 배치 실격")
        cost = (phi_final[b] - pre[b]) + _route_charge(idx, b, tr["t"])
        r2go = -cost / ret_scale
        out.append({**tr, "ret": r2go, "adv": r2go - tr["value"], "resolver": None})
    return out


def run_episode_block(seed: int, policy, kf) -> dict:
    """5차 계약 1 에피소드 — 블록별 φ 기록판. 무대·집행은 yr170 과 동일."""
    obs, prof = OBS_24H, build_h21_profile()
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    built = build_diurnal(prof, seed, obs=obs, layout=layout, params=params,
                          background_seed=seed)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=built["sim_end_s"])
    orch = UnifiedSellOrchestrator(policy, layout, kf, dry_run=False)
    rec = BlockPhiRecorder()
    fleet = AdoptedExecFleet(*load_adopted_execution_head(),
                            config=ADOPTED_C0_GUARD)
    gens: dict[int, CandidateGenerator] = {}

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=ADOPTED_C0_GUARD))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, fleet.get(sim).decide(sim, dp, gb))

    def review(m, t):
        ann.review(m, t)      # 투입
        rec.review(m, t)      # Φ 기록(행동 직전) — 블록별
        orch.review(m, t)     # 판매 확정

    mbt.run(exec_policy, review_fn=review)

    n_rows = sum(len(s.time_ledger.records) for s in mbt.blocks.values())
    n_skip = sum(1 for e in ann.ledger if e["event"] in ("SKIP", "SKIP_TAIL"))
    if n_rows != ann.n_admitted or n_skip or ann.n_admitted != len(built["schedule"]):
        raise RuntimeError(f"장부 불일치 — 등록 {n_rows}/투입 {ann.n_admitted}/"
                           f"계획 {len(built['schedule'])}/누락 {n_skip} (실격)")

    end = obs.observe_s
    phi_final = {b: phi_v2(sim, end) for b, sim in mbt.blocks.items()}
    return {"phi_by_t": rec.by_t, "phi_final_by_block": phi_final,
            "phi_final": sum(phi_final.values()) + mbt.route_cost_s / 3600.0,
            "sell_ledger": orch.ledger, "layout": layout,
            "n_space": orch.n_space, "n_time": orch.n_time,
            "admitted": ann.n_admitted}


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    seed, sd_a, sd_c, pol_seed = args
    a, c = TransferActor(), TransferCritic()
    a.load_state_dict(sd_a)
    c.load_state_dict(sd_c)
    pol = PpoSellPolicy(a, c, mode="live", sample=True, seed=pol_seed,
                        layout=terminal_layout())
    ep = run_episode_block(seed, SpaceOnly(pol) if SPACE_ONLY else pol, load_kf())
    batch = build_batch_block(pol.trail, ep["phi_by_t"], ep["phi_final_by_block"],
                              ep["sell_ledger"], ep["layout"])
    batch = [{k: (v.detach().clone() if hasattr(v, "detach") else v)
              for k, v in b.items()} for b in batch]
    return {"batch": batch, "phi_final": ep["phi_final"],
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "phi_final_by_block": ep["phi_final_by_block"]}


def train(ts: int, *, n_iter: int, eps_per_iter: int) -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from statistics import pstdev
    from ..integrated.repro import code_dirty, repro_stamp
    from ..v1.ppo_policy import TransferActor, TransferCritic
    from .yr151_transfer_ppo import LR, ppo_update
    _mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    torch.manual_seed(ts)
    actor, critic = TransferActor(), TransferCritic()
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    out = OUT / f"ppo_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    hist: list[dict] = []
    for it in range(n_iter):
        sd_a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        sd_c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        jobs = [(ts + it * eps_per_iter + e, sd_a, sd_c, ts + it * 100 + e)
                for e in range(eps_per_iter)]
        with ProcessPoolExecutor(max_workers=eps_per_iter) as pool:
            eps = list(pool.map(_worker, jobs))
        batch_all = [b for ep in eps for b in ep["batch"]]
        stats = ppo_update(actor, critic, opt_a, opt_c, batch_all,
                           minibatch=MINIBATCH, seed=ts + it, grad_clip=GRAD_CLIP)
        # ★관측: 블록별 비용의 흩어짐 — 떠넘기기가 생기면 커진다
        spread = [pstdev(ep["phi_final_by_block"].values()) for ep in eps]
        hist.append({"iter": it, **stats,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / len(eps),
                     "n_space_mean": sum(e["n_space"] for e in eps) / len(eps),
                     "n_time_mean": sum(e["n_time"] for e in eps) / len(eps),
                     "block_cost_spread": round(sum(spread) / len(spread), 3),
                     "n_batch": len(batch_all)})
        (out / "train.json").write_text(json.dumps(
            {"history": hist, "in_progress": True, "n_iter_target": n_iter},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   out / "net.pt")
    stamp = repro_stamp(
        experiment="YR-172 판매 PPO — 블록별 공로 배분 보상",
        seeds={"train": [ts]},
        params={"N_ITER": n_iter, "EPS_PER_ITER": eps_per_iter,
                "RET_SCALE": RET_SCALE, "MINIBATCH": MINIBATCH,
                "GRAD_CLIP": GRAD_CLIP, "SPACE_ONLY": SPACE_ONLY,
                "reward": "블록별 φ_b 증분 + 판 블록에 주행비 부과 (전역 합 = Φ 유지)"},
        prereg=SPEC)
    (out / "train.json").write_text(json.dumps(
        {"history": hist, "in_progress": False, "code_dirty": bool(code_dirty()),
         "stamp": stamp}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    a = ap.parse_args()
    p = train(TRAIN_SEEDS[a.seed_idx], n_iter=a.n_iter, eps_per_iter=a.eps_per_iter)
    print(json.dumps({"out": str(p)}, ensure_ascii=False))
    print("DONE")
