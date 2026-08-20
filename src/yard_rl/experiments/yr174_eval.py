"""YR-174 평가 — 학습한 정책이 **"아예 안 파는 것"보다 나은가**.

■ 왜 학습 곡선만으로는 판정할 수 없나
학습 중 Φ 는 2,907.8 → 2,488.0 으로 내려갔다. 그러나 이 출발점은 **난수 초기화
정책이 하루 2,658건을 마구 이연시킨 상태**다. 거기서 내려온 것은 "해롭던 짓을
그만둔 것" 일 수 있고, 그것만으로는 **판매 기구 자체가 이득인지** 알 수 없다.

판정에 필요한 기준선은 하나뿐이다:

    K (전건 KEEP) — 아무것도 팔지 않는다.  학습한 정책이 이보다 낮아야 이득이다.

■ 왜 축 분리 진단(2,346.2)을 그대로 쓰지 않나
그 진단은 `exec_head="sf"`, `exec_config=None` 로 돌았고 YR-174 학습은
`exec_head="adopted"`, `exec_config=ADOPTED_C0_GUARD` 로 돌았다. **집행 정책이
다르면 Φ 수준 자체가 다르다.** 같은 조건에서 다시 잰다.

■ 설계
  · 평가일(에피소드 시드)은 **학습에 쓰지 않은 날**(학습은 ts+0..ts+39 만 사용)
  · 모든 팔이 **같은 날**을 본다 — 짝지어 비교라 날씨 차이가 상쇄된다
  · 정책은 **결정론**(sample=False) — 판정에 표본 잡음을 넣지 않는다
  · 팔: K(전건 KEEP) · P0(학습 전 난수 초기화) · P9×6(시드별 학습 완료)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr174_txn_reward")
SPEC = ".claude/docs/dashboard-task-specs/YR-174-buy-market-closed-loop.md"
N_EVAL_DAYS = 6         # 6일 × 8팔 = 48 에피소드 = 24코어 2배치
EVAL_SEED0 = 9_000_000          # 학습 구간(8.4M~8.9M + 39) 밖 — 미사용일


def eval_seeds() -> list[int]:
    return [EVAL_SEED0 + i * 1000 for i in range(N_EVAL_DAYS)]


def _worker(args) -> dict:
    """(팔 이름, 가중치 파일 또는 None, 평가일) 하나를 돌린다."""
    import torch
    import torch.multiprocessing as _mp
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    arm, ckpt, day = args
    if arm == "K":
        pol = KeepAllTrail()
    else:
        a, c = TransferActor(), TransferCritic()
        if ckpt is not None:
            st = torch.load(ckpt, map_location="cpu", weights_only=True)
            a.load_state_dict(st["actor"])
            c.load_state_dict(st["critic"])
        else:
            torch.manual_seed(0)          # P0 = 학습 전 난수 초기화(고정)
            a, c = TransferActor(), TransferCritic()
        pol = PpoSellPolicy(a, c, mode="live", sample=False,
                            layout=terminal_layout())
    ep = run_episode_diurnal(day, pol, load_kf(), exec_config=ADOPTED_C0_GUARD)
    return {"arm": arm, "day": day, "phi_final": round(ep["phi_final"], 4),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "admitted": ep["admitted"]}


def run() -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean, pstdev

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    from .yr170_sell_ppo_diurnal import TRAIN_SEEDS
    _mp.set_sharing_strategy("file_system")

    days = eval_seeds()
    jobs: list[tuple] = [("K", None, d) for d in days]
    jobs += [("P0", None, d) for d in days]
    nets = []
    for ts in TRAIN_SEEDS:
        p = OUT / f"ppo_s{ts}" / "net.pt"
        if p.exists():
            nets.append((f"P9_s{ts}", str(p)))
            jobs += [(f"P9_s{ts}", str(p), d) for d in days]
    if not nets:
        raise RuntimeError("학습 가중치(net.pt)가 없다 — 학습을 먼저 완주할 것")

    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, jobs))

    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    kphi = {r["day"]: r["phi_final"] for r in by_arm["K"]}

    summary = []
    for arm in ["K", "P0"] + [n for n, _ in nets]:
        g = sorted(by_arm[arm], key=lambda r: r["day"])
        d = [r["phi_final"] - kphi[r["day"]] for r in g]      # 짝지어 차이
        se = pstdev(d) / (len(d) - 1) ** 0.5 if len(d) > 1 else float("nan")
        summary.append({
            "arm": arm, "n_days": len(g),
            "phi_mean": round(fmean(r["phi_final"] for r in g), 2),
            "phi_sd": round(pstdev(r["phi_final"] for r in g), 2),
            "vs_keep_mean": round(fmean(d), 2),
            "vs_keep_se": round(se, 2),
            "vs_keep_t": round(fmean(d) / se, 2) if se else None,
            "n_worse_than_keep": sum(1 for v in d if v > 0),
            "n_space_mean": round(fmean(r["n_space"] for r in g), 1),
            "n_time_mean": round(fmean(r["n_time"] for r in g), 1)})

    # 학습 완료 팔 6개를 하나로 — 시드간 일관성
    p9 = [s for s in summary if s["arm"].startswith("P9")]
    pooled = {
        "arm": "P9_pooled", "n_seeds": len(p9),
        "vs_keep_mean": round(fmean(s["vs_keep_mean"] for s in p9), 2),
        "vs_keep_sd_across_seeds": round(pstdev(s["vs_keep_mean"] for s in p9), 2),
        "n_seeds_better_than_keep": sum(1 for s in p9 if s["vs_keep_mean"] < 0),
        "n_time_mean": round(fmean(s["n_time_mean"] for s in p9), 1),
        "n_space_mean": round(fmean(s["n_space_mean"] for s in p9), 1)}

    res = {"experiment": "YR-174 평가 — 학습 정책 vs 전건 KEEP",
           "eval_days": days, "n_eval_days": N_EVAL_DAYS,
           "note": "vs_keep 양수 = 기준선(안 팔기)보다 나쁨. 같은 날 짝지어 비교.",
           "summary": summary, "pooled": pooled, "rows": rows,
           "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(
               experiment="YR-174 평가 — 학습 정책 vs 전건 KEEP",
               seeds={"eval_days": days},
               params={"exec_head": "adopted", "exec_config": "ADOPTED_C0_GUARD",
                       "sample": False, "observation": OBS_24H.as_dict()},
               prereg=SPEC)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "eval_vs_keep.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    for s in d["summary"]:
        print(f"{s['arm']:>14} Φ {s['phi_mean']:>9.2f}  vs_KEEP {s['vs_keep_mean']:>+9.2f} "
              f"± {s['vs_keep_se']:>6.2f} (t={s['vs_keep_t']})  "
              f"공간 {s['n_space_mean']:>7.1f}  시간 {s['n_time_mean']:>7.1f}")
    print("POOLED", json.dumps(d["pooled"], ensure_ascii=False))
    print("DONE", p)
