"""YR-185 ②③ — 보상 희소·critic 붕괴를 고치고 한 팔만 재학습한다.

■ 왜 ②③을 묶나 (원인이 하나다)
  표본의 **91% 가 보상 정확히 0** 이다(설계상 KEEP·resolver 거절은 실제 상태가
  KEEP 과 같으므로 0 — YR-174). 그래서 critic 은 **"항상 0"이 MSE 최적해**에
  가까워 그쪽으로 붕괴했다(rms 0.0567 → 0.0199, 목표는 0.133 → 0.108 유지).
  기준선이 사라져 `adv ≈ ret` 가 되고 PPO 가 REINFORCE 로 퇴화한다.
  0 을 줄이면 둘이 동시에 완화된다. 한 축이다.

■ 무엇을 바꾸나 — 딱 하나
  0 보상 표본을 확률 `ZERO_KEEP` 로 **하위표집**한다. 그리고 **정책과 가치에
  다른 가중을 건다** — 이 비대칭이 핵심이다.

  · **정책**: `1/p` 로 되돌린다. 정책경사는 편향되면 안 된다. 0 표본도
    배치 평균을 통해 신호를 나르므로 버리면 안 되고 **솎고 되돌리는** 것이다.
  · **가치**: 되돌리지 **않는다**. 되돌리면 목표 분포가 원래대로 복원돼
    "항상 0"이 다시 MSE 최적해가 되고 **③의 목적이 소멸한다**. 기준선은
    행동에 의존하지만 않으면 편향돼도 유효하므로, 목표 분포를 일부러
    재조정해 조건수를 개선한다.

  (구현 중 발견: 처음엔 양쪽에 같은 가중을 걸었는데, 그러면 가중이
   하위표집을 정확히 상쇄해 critic 붕괴가 그대로 남는다.)

■ 바꾸지 않는 것 (원인 분리)
  보상 정의·무대·행동공간(고정 +15분)·집행(ADOPTED_C0_GUARD)·학습률·클립·
  엔트로피·미니배치·grad clip·회차 수. **오직 표본 구성 하나만** 다르다.
  대조군은 YR-171-C `fixed15` (같은 시드 8,400,000·같은 10회차).

■ 사전 동결 (이 파일 커밋 시점에 확정)
  · `ZERO_KEEP = 0.15` — 0 비율 **91% → ~60%**(실측). 더 낮추면 조건수는
    좋아지나 정책경사 가중 분산이 커진다. 첫 시도이므로 과튜닝하지 않고
    **결과를 보고 바꾸지 않는다**.
  · 판정은 **학습 곡선이 안 팔기 기준선을 넘는가** 하나다. 넘으면 RL 트랙
    계속, 못 넘으면 서사 B(메커니즘 주인공)로 확정한다(종합문서 §7).
  · 성능 주장 금지 — 이것은 "고치면 되나"를 보는 진단판이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr185_retrain")
SPEC = ".claude/docs/dashboard-task-specs/YR-185-training-setup-audit.md"
ARM = "fixed15"                 # 한 팔만 — 곡선이 0 을 넘는지만 본다
TRAIN_SEED = 8_400_000          # YR-171-C 와 같은 시드 → 직접 대조
N_ITER, EPS_PER_ITER = 10, 4    # 회차도 동일 (④는 이번 범위 밖)
ZERO_KEEP = 0.15                # ★동결 — 0 보상 표본 유지 확률


def resample_zeros(batch: list[dict], *, keep: float, seed: int) -> tuple[list[dict], dict]:
    """0 보상 표본을 `keep` 확률로 솎고 `1/keep` 가중을 준다.

    버리는 게 아니라 솎는 것이다 — 0 표본도 배치 평균을 통해 신호를 나르므로
    없애면 편향된다. 가중으로 기대 기울기를 되돌린다.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    out, n_zero_in, n_zero_out = [], 0, 0
    for b in batch:
        if abs(float(b["ret"])) > 1e-12:
            out.append({**b, "w": 1.0, "wv": 1.0})
            continue
        n_zero_in += 1
        if float(torch.rand(1, generator=g).item()) < keep:
            # 정책은 되돌리고(1/p), 가치는 되돌리지 않는다(1.0) — 비대칭이 핵심
            out.append({**b, "w": 1.0 / keep, "wv": 1.0})
            n_zero_out += 1
    nz = sum(1 for b in out if abs(float(b["ret"])) > 1e-12)
    rep = {"n_in": len(batch), "n_out": len(out),
           "n_zero_in": n_zero_in, "n_zero_out": n_zero_out,
           "zero_share_in": round(1 - (len(batch) - n_zero_in) / max(1, len(batch)), 4),
           "zero_share_out": round(n_zero_out / max(1, len(out)), 4),
           "nonzero_share_out": round(nz / max(1, len(out)), 4),
           "keep": keep}
    return out, rep


def train() -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    from ..integrated.transfer_head import TransferActor, TransferCritic
    from .yr151_transfer_ppo import LR, ppo_update
    from .yr170_sell_ppo_diurnal import GRAD_CLIP, MINIBATCH
    from .yr171c_train import _worker
    from .yr174_txn_reward import RET_SCALE, scale_report
    _mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    torch.manual_seed(TRAIN_SEED)
    actor, critic = TransferActor(), TransferCritic()
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    OUT.mkdir(parents=True, exist_ok=True)

    hist = []
    for it in range(N_ITER):
        sd_a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        sd_c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        # 에피소드 시드 계약을 YR-171-C 와 **완전히 동일**하게 둔다 (같은 40일).
        jobs = [(TRAIN_SEED + it * EPS_PER_ITER + e, sd_a, sd_c,
                 TRAIN_SEED + it * 100 + e, ARM) for e in range(EPS_PER_ITER)]
        with ProcessPoolExecutor(max_workers=EPS_PER_ITER) as pool:
            eps = list(pool.map(_worker, jobs))
        raw = [b for ep in eps for b in ep["batch"]]
        before = scale_report(raw)
        batch, rep = resample_zeros(raw, keep=ZERO_KEEP, seed=TRAIN_SEED + it)
        after = scale_report(batch)
        stats = ppo_update(actor, critic, opt_a, opt_c, batch,
                           minibatch=MINIBATCH, seed=TRAIN_SEED + it,
                           grad_clip=GRAD_CLIP)
        n = len(eps)
        hist.append({"iter": it, **stats,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / n,
                     "n_space_mean": sum(e["n_space"] for e in eps) / n,
                     "n_time_mean": sum(e["n_time"] for e in eps) / n,
                     "credit_sum_mean": sum(e["credit_sum"] for e in eps) / n,
                     "resample": rep,
                     "scale_before": before, "scale_after": after})
        (OUT / "train.json").write_text(json.dumps(
            {"arm": ARM, "zero_keep": ZERO_KEEP, "history": hist,
             "in_progress": it < N_ITER - 1}, ensure_ascii=False,
            indent=1, default=str), encoding="utf-8")
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   OUT / "net.pt")

    payload = {"experiment": "YR-185 ②③ 보상 희소·critic 붕괴 교정 후 재학습",
               "kind": "diagnostic",
               "note": "성능 판정 아님 — 곡선이 안 팔기 기준선을 넘는지만 본다.",
               "arm": ARM, "train_seed": TRAIN_SEED,
               "zero_keep": ZERO_KEEP, "history": hist, "in_progress": False,
               "control": "YR-171-C fixed15 s8400000 (같은 시드·회차·무대)",
               "changed_axis": "표본 구성만 — 0 보상 하위표집 + 1/p 가중",
               "code_dirty": bool(code_dirty()),
               "stamp": repro_stamp(
                   experiment="YR-185 ②③ 재학습", seeds={"train": [TRAIN_SEED]},
                   params={"arm": ARM, "N_ITER": N_ITER,
                           "EPS_PER_ITER": EPS_PER_ITER, "ZERO_KEEP": ZERO_KEEP,
                           "RET_SCALE": RET_SCALE, "MINIBATCH": MINIBATCH,
                           "GRAD_CLIP": GRAD_CLIP,
                           "observation": OBS_24H.as_dict()},
                   prereg=SPEC)}
    p = OUT / "train.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = train()
    d = json.loads(p.read_text(encoding="utf-8"))
    ctrl = json.loads(Path("outputs/reports/yr171c_slots/fixed15_s8400000/"
                           "train.json").read_text(encoding="utf-8"))["history"]
    print(f"0 보상 유지율 {d['zero_keep']}  |  대조군 = YR-171-C fixed15 s8400000")
    print(f"{'iter':>4} {'Φ(교정)':>10} {'Φ(대조)':>10} {'0비율':>7} "
          f"{'critic rms':>10} {'비율':>6} {'공간':>7} {'시간':>7}")
    for h, c in zip(d["history"], ctrl):
        s = h["scale_after"]
        print(f"{h['iter']:>4} {h['phi_final_mean']:>10.1f} "
              f"{c['phi_final_mean']:>10.1f} "
              f"{1 - s['nonzero_share']:>6.1%} {s['value_rms']:>10.4f} "
              f"{s['ratio']:>6.2f} {h['n_space_mean']:>7.0f} {h['n_time_mean']:>7.0f}")
    print("DONE", p)
