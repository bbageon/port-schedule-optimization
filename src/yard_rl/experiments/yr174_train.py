"""YR-174 학습 — **거래별 실현 손익(차이 보상)** 으로 SELL PPO 를 돌린다.

■ 이번 판에서 바꾸는 것은 **보상 하나**다 (단일축)
무대·집행 정책·resolver·행동 공간은 그대로 두고, 보상만 교체한다.

  YR-170 전역:   모든 결정 = −(하루 총비용 변화)      → 행동 신호 0.0004%
  YR-172 블록별: 결정 = −(자기 블록 비용)             → 떠넘기기 구조 + 측정 결함으로 보류
  **YR-174 거래별: 판매한 결정 = 그 거래의 실현 손익 D_i, 나머지는 0**

BUY 견적망은 **아직 연결하지 않는다** — 학습·검증 전이라 출력이 무의미하고, 넣으면
"보상 교체" 와 "견적 교체" 두 축이 섞여 원인을 못 가린다(설계안 단일축 원칙).
배정은 지금까지와 같은 resolver 계산식이 한다.

■ 시간 판매를 막지 않는다 (사용자 질문: "시간 판매가 잘 되는지")
YR-172 의 `SpaceOnly` 는 정책 지명만 걸러 resolver 가 붙이는 TIME 좌표를 못 막았다
(sell_review.py:352 "반입·반출 공통"). 여기서는 **두 축을 다 열고**, 각 축을 **자기
실현 손익으로** 채점한다 — 시간 이연은 "원래 시각에 안 와서 덜 밀린 몫 − 기사 외부
대기" 로 계산되므로, 손해면 정책이 음의 보상을 받는다. 막는 대신 **가르친다**.

■ 눈금 안전장치
매 iteration 배치를 `assert_scale_sane` 으로 검사한다 — 목표가 과대/과소하거나 보상이
전부 0(조인 실패)이면 **학습을 시작하지 않고 실격**. 첫 회차 `scale_report` 를 기록에
남긴다. 전역 판에서 눈금이 2만 7천 배 어긋난 채 16시간 돈 실패를 구조적으로 막는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.terminal_stream import OBS_24H
from ..integrated.yard_layout import terminal_layout
from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
from .yr151_transfer_ppo import load_kf
from .yr170_sell_ppo_diurnal import GRAD_CLIP, MINIBATCH, TRAIN_SEEDS, run_episode_diurnal
from .yr174_txn_reward import (RET_SCALE, TransactionLog, assert_scale_sane,
                               build_batch_txn, realized_credit, scale_report)

OUT = Path("outputs/reports/yr174_txn_reward")
SPEC = ".claude/docs/dashboard-task-specs/YR-174-buy-market-closed-loop.md"


def _worker(args) -> dict:
    """에피소드 1개 + 거래별 보상 부착까지 자식 프로세스에서 끝낸다."""
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                            TransferCritic)
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    seed, sd_a, sd_c, pol_seed = args
    a, c = TransferActor(), TransferCritic()
    a.load_state_dict(sd_a)
    c.load_state_dict(sd_c)
    pol = PpoSellPolicy(a, c, mode="live", sample=True, seed=pol_seed,
                        layout=terminal_layout())
    ep = run_episode_diurnal(seed, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             _return_mbt=True)
    mbt = ep.pop("_mbt")
    layout = terminal_layout()
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    txns = TransactionLog().collect(ep["sell_ledger"])
    credits = realized_credit(mbt, txns, layout, l_t=SLA_ANCHOR_S + sla)
    batch = build_batch_txn(pol.trail, txns, credits)
    batch = [{k: (v.detach().clone() if hasattr(v, "detach") else v)
              for k, v in b.items()} for b in batch]
    return {"batch": batch, "phi_final": ep["phi_final"],
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "n_txn": len(txns),
            "credit_sum": round(sum(credits.values()), 4),
            "credit_pos": sum(1 for v in credits.values() if v > 0),
            "credit_neg": sum(1 for v in credits.values() if v < 0)}


def train(ts: int, *, n_iter: int, eps_per_iter: int) -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.transfer_head import TransferActor, TransferCritic
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
    first_scale = None
    for it in range(n_iter):
        sd_a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        sd_c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        jobs = [(ts + it * eps_per_iter + e, sd_a, sd_c, ts + it * 100 + e)
                for e in range(eps_per_iter)]
        with ProcessPoolExecutor(max_workers=eps_per_iter) as pool:
            eps = list(pool.map(_worker, jobs))
        batch_all = [b for ep in eps for b in ep["batch"]]
        # ★눈금 검사 — 이상하면 여기서 멈춘다(다 돌고 나서 알던 실패를 시작 전에)
        rep = assert_scale_sane(batch_all)
        if first_scale is None:
            first_scale = rep
        stats = ppo_update(actor, critic, opt_a, opt_c, batch_all,
                           minibatch=MINIBATCH, seed=ts + it, grad_clip=GRAD_CLIP)
        n = len(eps)
        hist.append({"iter": it, **stats,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / n,
                     "n_space_mean": sum(e["n_space"] for e in eps) / n,
                     "n_time_mean": sum(e["n_time"] for e in eps) / n,
                     "n_txn_mean": sum(e["n_txn"] for e in eps) / n,
                     "credit_sum_mean": sum(e["credit_sum"] for e in eps) / n,
                     "credit_pos_mean": sum(e["credit_pos"] for e in eps) / n,
                     "credit_neg_mean": sum(e["credit_neg"] for e in eps) / n,
                     "scale": rep, "n_batch": len(batch_all)})
        (out / "train.json").write_text(json.dumps(
            {"history": hist, "in_progress": True, "n_iter_target": n_iter,
             "first_scale_report": first_scale},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   out / "net.pt")
    stamp = repro_stamp(
        experiment="YR-174 SELL PPO — 거래별 실현 손익(차이 보상)",
        seeds={"train": [ts]},
        params={"N_ITER": n_iter, "EPS_PER_ITER": eps_per_iter,
                "RET_SCALE": RET_SCALE, "MINIBATCH": MINIBATCH,
                "GRAD_CLIP": GRAD_CLIP,
                "reward": "거래별 실현 D_i = R_src − B_dst − 주행 − 시간변경 "
                          "(KEEP·거절 = 0). BUY 견적망 미연결(단일축).",
                "axes": "공간·시간 둘 다 열림 — 각 축을 자기 실현 손익으로 채점",
                "observation": OBS_24H.as_dict()},
        prereg=SPEC)
    (out / "train.json").write_text(json.dumps(
        {"history": hist, "in_progress": False, "code_dirty": bool(code_dirty()),
         "first_scale_report": first_scale, "stamp": stamp},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="에피소드 1회 + 보상 부착까지만 (학습 없음)")
    a = ap.parse_args()
    if a.smoke:
        import torch
        torch.manual_seed(0)
        from ..integrated.transfer_head import (TransferActor, TransferCritic)
        sd_a = TransferActor().state_dict()
        sd_c = TransferCritic().state_dict()
        r = _worker((TRAIN_SEEDS[a.seed_idx], sd_a, sd_c, 0))
        batch = r.pop("batch")
        r["scale"] = scale_report(batch)
        print(json.dumps(r, ensure_ascii=False))
    else:
        p = train(TRAIN_SEEDS[a.seed_idx], n_iter=a.n_iter,
                  eps_per_iter=a.eps_per_iter)
        print(json.dumps({"out": str(p)}, ensure_ascii=False))
    print("DONE")
