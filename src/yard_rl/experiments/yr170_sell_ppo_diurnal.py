"""YR-170 — 5차 계약(24시간 이중 피크) 무대에서 **판매 PPO 학습**.

사용자 결정 2026-08-11: *"학습부터 해서 학습이 안 됐다가 나와야 디버깅을 하는 것"*.
그동안 0B 의 논리("손으로 만든 견적이 좋은 판매를 못 고르면 학습해도 안 된다")를
전제로 견적을 진단해 왔으나, **학습기는 견적을 보상으로 쓰지 않는다** —
보상 = `−(Φ_final − Φ(t_k⁻)) + V(s_end)` 로 **실현 터미널 총비용**에서 나온다
(yr151_transfer_ppo.build_batch). 견적의 부정확은 학습을 막을 이유가 아니라
학습이 할 일이 남았다는 뜻이다. 그래서 무대만 5차로 옮겨 **먼저 돌린다**.

바뀌는 것은 **무대뿐**이다 — 학습 루프·보상·PPO 갱신·재현 스탬프는
`yr151_transfer_ppo.train_one(episode_fn=...)` 을 그대로 공유한다.

  4차(고정 WIP)                     → 5차(도착률 24h 이중 피크)
  build_fixed_wip + hotspot         → build_diurnal (명단 사전 확정)
  WipAdmissionController(pool)      → ScheduledAnnouncer (개방 루프)
  ObservationContract() 5h          → OBS_24H (워밍업 2h + 측정 22h, 배수 2h)
  _sim_from(scn)                    → _sim_from(scn, prof) + ensure_time_ledger

성능 주장 없음 — 이 파일은 **학습이 되는지 보는 실행**이다. 판정은 별도 사전등록.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import ADOPTED_C0_GUARD, LEGACY_DEFAULT
from ..integrated.profiles import build_h21_profile
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                          ScheduledAnnouncer, TerminalStreamParams,
                                          admission_epochs, build_diurnal,
                                          ensure_time_ledger)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr151_transfer_ppo import (AdoptedExecFleet, PhiRecorder, _Chain,
                                 build_joint_transitions, load_adopted_execution_head,
                                 phi_terminal, train_one)

OUT = Path("outputs/reports/yr170_sell_ppo_diurnal")
SPEC = ".claude/docs/dashboard-task-specs/YR-170-sell-ppo-diurnal.md"
TRAIN_SEEDS = (8_400_000, 8_500_000, 8_600_000)   # 자격·관찰·파일럿 대역과 교집합 0


def run_episode_diurnal(seed: int, policy, kf, *,
                        obs=None, exec_head: str = "adopted",
                        exec_fleet=None, exec_config=None) -> dict:
    """5차 계약 1 에피소드 — 4차 `run_episode` 와 **반환 형태 동일**(학습 루프 공유).

    무대만 다르다: 도착 명단 사전 확정 + 개방 루프 투입 + 24시간 이중 피크.
    투입이 정책과 무관하므로 두 정책이 같은 트럭·같은 시각을 받는다.
    """
    if exec_head not in ("adopted", "sf"):
        raise ValueError(f"exec_head 는 'adopted'|'sf' 만 허용: {exec_head!r}")
    if exec_head == "adopted":
        if exec_config is None:
            raise ValueError("adopted execution requires explicit exec_config")
        if exec_config != ADOPTED_C0_GUARD:
            raise ValueError("performance harness requires ADOPTED_C0_GUARD")
    else:
        exec_config = LEGACY_DEFAULT

    obs = obs or OBS_24H
    prof, layout = build_h21_profile(), terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    built = build_diurnal(prof, seed, obs=obs, layout=layout, params=params,
                          background_seed=seed)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=built["sim_end_s"])
    orch = UnifiedSellOrchestrator(policy, layout, kf,
                                   dry_run=(getattr(policy, "mode", None) == "shadow"))
    rec = PhiRecorder()

    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}
    if exec_head == "adopted":
        if exec_fleet is None:
            exec_fleet = AdoptedExecFleet(
                *load_adopted_execution_head(), config=exec_config)
        elif exec_fleet.config != exec_config:
            raise ValueError("exec_fleet and exec_config do not match")

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), CandidateGenerator(config=exec_config))
            gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
            _apply(sim, exec_fleet.get(sim).decide(sim, dp, gb))   # 예외 = 즉시 실격
    else:
        pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), CandidateGenerator(config=exec_config))
            gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
            try:
                _apply(sim, pol.decide(sim, dp, gb))
            except Exception:
                exc["n"] += 1
                _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    # Φ 는 **행동 직전** 기록: 투입 → Φ → 판매 (4차 계약과 같은 순서 계약)
    mbt.run(exec_policy, review_fn=_Chain(ann, rec, orch).review)

    # 장부 보존 — 5차 계약판(채움 개념 없음): 등록 = 투입, 누락 0
    n_rows = sum(len(s.time_ledger.records) for s in mbt.blocks.values()
                 if getattr(s, "time_ledger", None) is not None)
    n_skip = sum(1 for e in ann.ledger if e["event"] in ("SKIP", "SKIP_TAIL"))
    if n_rows != ann.n_admitted or n_skip or ann.n_admitted != len(built["schedule"]):
        raise RuntimeError(
            f"장부 불일치 — 등록 {n_rows} / 투입 {ann.n_admitted} / "
            f"계획 {len(built['schedule'])} / 누락 {n_skip} (에피소드 실격)")

    phi_final = phi_terminal(mbt, obs.observe_s)
    from ..integrated.transfer_head import critic_input
    end_inputs = {b: critic_input(mbt, b, obs.observe_s, 0, layout=layout)
                  for b in mbt.blocks}
    return {"phi": rec.samples, "phi_final": phi_final, "sell_ledger": orch.ledger,
            "exec_policy_config": exec_config.as_dict(),
            "end_inputs": end_inputs,
            "joint": build_joint_transitions(policy.trail, orch.ledger, rec.samples,
                                             phi_final, obs.observe_s),
            "n_space": orch.n_space, "n_time": orch.n_time,
            "policy_exceptions": exc["n"], "admitted": ann.n_admitted}


class KeepAllTrail:
    """기준선 K — 아무것도 팔지 않는다. 학습 루프와 같은 인터페이스(trail 보유)."""

    mode = "live"

    def __init__(self):
        self.trail: list[dict] = []

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        return None


def baseline(seed: int, *, exec_head: str = "adopted") -> dict:
    """전건 KEEP 의 Φ — 학습 결과를 읽을 기준점(성능 판정 아님, 참조값)."""
    from .yr151_transfer_ppo import load_kf
    ep = run_episode_diurnal(
        seed, KeepAllTrail(), load_kf(), exec_head=exec_head,
        exec_config=(ADOPTED_C0_GUARD if exec_head == "adopted" else None))
    return {"seed": seed, "exec_head": exec_head,
            "phi_final": round(ep["phi_final"], 4),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "admitted": ep["admitted"]}


def train(ts: int, *, n_iter: int, eps_per_iter: int) -> Path:
    return train_one(ts, out_root=OUT, episode_fn=run_episode_diurnal,
                     n_iter=n_iter, eps_per_iter=eps_per_iter,
                     experiment="YR-170 판매 PPO — 5차 계약 24시간 이중 피크 무대",
                     prereg=SPEC)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="에피소드 1회만 돌려 배선 확인 (학습 없음)")
    ap.add_argument("--baseline", action="store_true",
                    help="전건 KEEP 기준선 Φ (참조값 — 판정 아님)")
    a = ap.parse_args()
    if a.baseline:
        r = baseline(TRAIN_SEEDS[a.seed_idx])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"baseline_s{r['seed']}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(r, ensure_ascii=False))
    elif a.smoke:
        import torch
        from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                                TransferCritic)
        from .yr151_transfer_ppo import load_kf
        torch.manual_seed(0)
        pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="live",
                            sample=True, seed=0, layout=terminal_layout())
        ep = run_episode_diurnal(TRAIN_SEEDS[0], pol, load_kf(),
                                 exec_config=ADOPTED_C0_GUARD)
        OUT.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"phi_final": round(ep["phi_final"], 4),
                          "n_phi_samples": len(ep["phi"]),
                          "n_joint": len(ep["joint"]),
                          "n_space": ep["n_space"], "n_time": ep["n_time"],
                          "admitted": ep["admitted"],
                          "n_decisions": len(pol.trail)}, ensure_ascii=False))
    else:
        p = train(TRAIN_SEEDS[a.seed_idx], n_iter=a.n_iter,
                  eps_per_iter=a.eps_per_iter)
        print(json.dumps({"out": str(p)}, ensure_ascii=False))
    print("DONE")
