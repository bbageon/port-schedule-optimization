"""YR-171-C 학습 — 48칸 시간 좌표 + BUY 견적망으로 SELL PPO 를 돌린다.

■ 무엇이 바뀌나 (YR-174 대비 **행동 공간 하나**)
보상은 YR-174 의 거래별 실현 손익 그대로다. 바뀌는 것은 정책이 고를 수 있는 세계다.

  YR-174: 시간 좌표 = **+15분 한 칸**        → "미룰까 말까" 만 배울 수 있다
  YR-171-C: 시간 좌표 = **오늘의 48칸**      → "**언제** 오라고 할까" 를 배운다

전제(선행 완료): 하루 공개 예약 장부(171-A) 없이는 미래 칸이 0.4~5.6% 만 채워져
"먼 슬롯일수록 한가하다"를 배운다. `UnifiedSellOrchestrator` 가 장부 부재 시 실격.

■ 팔 구성 (같은 시드·같은 날, 행동 공간만 다름)
  · `fixed15`  구 계약 재현 — 48칸을 닫고 +15분 한 칸 (YR-174 조건)
  · `slots48`  48칸 + **결정론 계산식** 견적
  · `slots48_buy` 48칸 + **BUY 견적망** 견적
세 팔을 갈라야 "48칸이 이득인가"와 "견적망이 이득인가"가 분리된다(단일축).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr171c_slots")
SPEC = ".claude/docs/dashboard-task-specs/YR-171-time-sale-slot-contract.md"
BUY_NET = Path("outputs/reports/yr171b_estimator/buy_net.pt")
ARMS = ("fixed15", "slots48", "slots48_buy")


def _load_buy():
    import torch
    from ..integrated.buy_estimator import JOB_DIM, BuyEstimator
    from ..integrated.slot_plan import N_FEATURES
    net = BuyEstimator(slot_dim=N_FEATURES, job_dim=JOB_DIM)
    st = torch.load(BUY_NET, map_location="cpu", weights_only=True)
    net.load_state_dict(st["buy"])
    net.eval()
    return net


def _arm_kwargs(arm: str) -> dict:
    if arm == "fixed15":
        return {"time_slots": False, "buy_net": None}
    if arm == "slots48":
        return {"time_slots": True, "buy_net": None}
    if arm == "slots48_buy":
        return {"time_slots": True, "buy_net": _load_buy()}
    raise ValueError(f"모르는 팔: {arm}")


def _worker(args) -> dict:
    """에피소드 1개 + 거래별 보상 부착 (YR-174 `_worker` 와 같은 계약)."""
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                            TransferCritic)
    from ..integrated.yard_layout import terminal_layout
    from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import run_episode_diurnal
    from .yr174_txn_reward import (TransactionLog, build_batch_txn,
                                   realized_credit)
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    seed, sd_a, sd_c, pol_seed, arm = args
    a, c = TransferActor(), TransferCritic()
    a.load_state_dict(sd_a)
    c.load_state_dict(sd_c)
    pol = PpoSellPolicy(a, c, mode="live", sample=True, seed=pol_seed,
                        layout=terminal_layout())
    ep = run_episode_diurnal(seed, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, _return_mbt=True,
                             **_arm_kwargs(arm))
    mbt = ep.pop("_mbt")
    layout = terminal_layout()
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    txns = TransactionLog().collect(ep["sell_ledger"])
    credits = realized_credit(mbt, txns, layout, l_t=SLA_ANCHOR_S + sla)
    batch = build_batch_txn(pol.trail, txns, credits)
    batch = [{k: (v.detach().clone() if hasattr(v, "detach") else v)
              for k, v in b.items()} for b in batch]
    # 어느 칸을 골랐는지 — 48칸이 실제로 쓰이는지 보는 유일한 창
    slots = [r.get("slot") for r in ep["sell_ledger"]
             if r.get("axis") == "TIME" and r.get("decision") == "DEFER"]
    defers = [r.get("defer_s") for r in ep["sell_ledger"]
              if r.get("axis") == "TIME" and r.get("decision") == "DEFER"
              and r.get("defer_s") is not None]
    return {"batch": batch, "phi_final": ep["phi_final"],
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "n_txn": len(txns),
            "credit_sum": round(sum(credits.values()), 4),
            "credit_pos": sum(1 for v in credits.values() if v > 0),
            "credit_neg": sum(1 for v in credits.values() if v < 0),
            "slot_hist": {str(s): slots.count(s) for s in sorted(set(slots))
                          if s is not None},
            "defer_mean_s": (sum(defers) / len(defers)) if defers else None,
            "defer_max_s": max(defers) if defers else None}


def train(ts: int, arm: str, *, n_iter: int, eps_per_iter: int) -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    from ..integrated.transfer_head import TransferActor, TransferCritic
    from .yr151_transfer_ppo import LR, ppo_update
    from .yr170_sell_ppo_diurnal import GRAD_CLIP, MINIBATCH
    from .yr174_txn_reward import RET_SCALE, assert_scale_sane
    _mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    torch.manual_seed(ts)
    actor, critic = TransferActor(), TransferCritic()
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    out = OUT / f"{arm}_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    hist, first_scale = [], None
    for it in range(n_iter):
        sd_a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        sd_c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        jobs = [(ts + it * eps_per_iter + e, sd_a, sd_c, ts + it * 100 + e, arm)
                for e in range(eps_per_iter)]
        with ProcessPoolExecutor(max_workers=eps_per_iter) as pool:
            eps = list(pool.map(_worker, jobs))
        batch_all = [b for ep in eps for b in ep["batch"]]
        rep = assert_scale_sane(batch_all)
        first_scale = first_scale or rep
        stats = ppo_update(actor, critic, opt_a, opt_c, batch_all,
                           minibatch=MINIBATCH, seed=ts + it, grad_clip=GRAD_CLIP)
        n = len(eps)
        merged: dict[str, int] = {}
        for e in eps:
            for k, v in e["slot_hist"].items():
                merged[k] = merged.get(k, 0) + v
        hist.append({"iter": it, **stats,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / n,
                     "n_space_mean": sum(e["n_space"] for e in eps) / n,
                     "n_time_mean": sum(e["n_time"] for e in eps) / n,
                     "credit_sum_mean": sum(e["credit_sum"] for e in eps) / n,
                     "defer_mean_s": ([e["defer_mean_s"] for e in eps
                                       if e["defer_mean_s"] is not None] or [None])[0],
                     "n_slots_used": len(merged), "slot_hist": merged,
                     "scale": rep, "n_batch": len(batch_all)})
        (out / "train.json").write_text(json.dumps(
            {"arm": arm, "history": hist, "in_progress": True,
             "n_iter_target": n_iter, "first_scale_report": first_scale},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   out / "net.pt")
    stamp = repro_stamp(
        experiment=f"YR-171-C SELL PPO — 시간 좌표 48칸 ({arm})",
        seeds={"train": [ts]},
        params={"arm": arm, "N_ITER": n_iter, "EPS_PER_ITER": eps_per_iter,
                "RET_SCALE": RET_SCALE, "MINIBATCH": MINIBATCH,
                "GRAD_CLIP": GRAD_CLIP, "day_plan_public": True,
                "reward": "YR-174 거래별 실현 손익 (변경 없음 — 단일축은 행동공간)",
                "observation": OBS_24H.as_dict()},
        prereg=SPEC)
    (out / "train.json").write_text(json.dumps(
        {"arm": arm, "history": hist, "in_progress": False,
         "code_dirty": bool(code_dirty()), "first_scale_report": first_scale,
         "stamp": stamp}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, default="slots48")
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    from .yr170_sell_ppo_diurnal import TRAIN_SEEDS
    if a.smoke:
        import torch
        from ..integrated.transfer_head import TransferActor, TransferCritic
        torch.manual_seed(0)
        r = _worker((TRAIN_SEEDS[a.seed_idx], TransferActor().state_dict(),
                     TransferCritic().state_dict(), 0, a.arm))
        r.pop("batch")
        print(json.dumps({"arm": a.arm, **r}, ensure_ascii=False))
    else:
        p = train(TRAIN_SEEDS[a.seed_idx], a.arm,
                  n_iter=a.n_iter, eps_per_iter=a.eps_per_iter)
        print(json.dumps({"out": str(p)}, ensure_ascii=False))
    print("DONE")
