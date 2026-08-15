"""YR-171-B ① 정답지 수집 — BUY 견적망 지도학습용 (입력·라벨 경로 분리).

■ 한 에피소드가 만드는 것
  · **입력**: 하루 중 여러 기준시각에서 찍은 계획표 `terminal_slot_plan` (21블록×48칸×9)
             + 그 시각의 후보 작업 특징 `job_features`  ← **공개 정보만**
  · **라벨**: 그 트럭이 실제로 겪은 부담 `burden_label.realized_burden`  ← **사후 장부**

두 경로는 **섞이지 않는다**. 입력은 그 시각에 알 수 있는 것만, 라벨은 하루가 끝난 뒤
장부에서. 견적망이 배우는 것은 "지금 보이는 계획으로 앞으로의 부담을 맞히기"다.

■ 왜 이 형태인가
견적망 출력은 (작업 N, 블록 B, 슬롯 48) 이다. 학습에 필요한 것은 그 격자 위의 점
(작업 j, 블록 b, 슬롯 t) → 부담이다. 트럭 한 대가 블록 b 의 슬롯 t 에 실제로 도착해
부담을 남기면, 그 점 하나가 관측된다. 하루 3,600대 = 3,600점이고, 트럭은 하루 종일
모든 블록에 오므로 격자가 자연히 덮인다(**판매된 건만 보이는 선택 편향 없음**).

■ 정보 경계
계획표 기준시각은 **그 트럭의 공개 ETA 보다 앞**이어야 한다 — 이미 도착한 트럭의
부담을 그 도착을 보고 맞히는 것은 예측이 아니다. `lead_slots` 로 강제한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr171b_estimator")
SPEC = ".claude/docs/dashboard-task-specs/YR-171-time-sale-slot-contract.md"
SNAP_HOURS = tuple(range(1, 23))       # 1~22시 매시 정각 — 하루를 고르게 덮는다
LEAD_SLOTS = 1                          # 최소 선행: 계획표 시각 + 1슬롯 이후의 트럭만


class _PlanSnaps:
    """관측 전용 — 지정 시각에 계획표를 통째로 찍는다(입력 경로)."""

    def __init__(self, times):
        self.want = sorted(times)
        self.snaps: list[dict] = []
        self._i = 0

    def review(self, mbt, t: float) -> None:
        if self._i >= len(self.want) or t < self.want[self._i]:
            return
        self._i += 1
        from ..integrated.slot_plan import terminal_slot_plan
        plan = terminal_slot_plan(mbt, t)          # 정규화판 = 망 입력 형식
        self.snaps.append({"t": t, "blocks": sorted(plan),
                           "plan": [plan[b] for b in sorted(plan)]})


def collect(seed: int) -> dict:
    """에피소드 1개 → {snapshots, labels}. 판매는 하지 않는다(정답지는 판매와 무관)."""
    from ..integrated.burden_label import burden_stats, terminal_burden
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.slot_plan import SLOT_S
    from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal

    snaps = _PlanSnaps([h * 3600.0 for h in SNAP_HOURS])
    ep = run_episode_diurnal(seed, KeepAllTrail(), load_kf(),
                             exec_config=ADOPTED_C0_GUARD, day_plan_public=True,
                             _return_mbt=True, _extra_review=snaps.review)
    mbt = ep.pop("_mbt")
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    labels = terminal_burden(mbt, l_t=SLA_ANCHOR_S + sla)

    # 라벨에 그 트럭의 **작업 특징**을 붙인다 — 어느 기준시각에서 본 것인지와 함께.
    # 기준시각은 "그 트럭 도착보다 LEAD_SLOTS 앞선 스냅샷 중 가장 늦은 것"이다.
    snap_ts = [s["t"] for s in snaps.snaps]
    rows = []
    for lab in labels:
        cutoff = lab["block_arrival_s"] - LEAD_SLOTS * SLOT_S
        cand = [i for i, t in enumerate(snap_ts) if t <= cutoff]
        if not cand:
            continue                       # 하루 초반 — 예측할 계획표가 아직 없다
        si = cand[-1]
        rows.append({**lab, "snap_idx": si, "snap_t": snap_ts[si],
                     "lead_s": lab["block_arrival_s"] - snap_ts[si]})
    return {"seed": seed, "snapshots": snaps.snaps, "labels": rows,
            "n_labels_all": len(labels), "n_labels_usable": len(rows),
            "stats": burden_stats(labels)}


def run(seeds: list[int]) -> Path:
    from concurrent.futures import ProcessPoolExecutor

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    _mp.set_sharing_strategy("file_system")
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(seeds), 24)) as pool:
        eps = list(pool.map(collect, seeds))
    for e in eps:
        (OUT / f"data_s{e['seed']}.json").write_text(
            json.dumps(e, ensure_ascii=False), encoding="utf-8")
    meta = {"experiment": "YR-171-B 정답지 수집", "kind": "dataset",
            "data_status": "시뮬레이션 산출물이지 실제 터미널 운영 기록이 아니다"
                           " (YR-173 §데이터의 지위).",
            "seeds": seeds, "snap_hours": list(SNAP_HOURS),
            "lead_slots": LEAD_SLOTS,
            "n_labels_usable": sum(e["n_labels_usable"] for e in eps),
            "per_seed": [{k: e[k] for k in ("seed", "n_labels_all",
                                            "n_labels_usable", "stats")}
                         for e in eps],
            "code_dirty": bool(code_dirty()),
            "stamp": repro_stamp(experiment="YR-171-B 정답지 수집",
                                 seeds={"episodes": seeds},
                                 params={"snap_hours": list(SNAP_HOURS),
                                         "lead_slots": LEAD_SLOTS,
                                         "observation": OBS_24H.as_dict()},
                                 prereg=SPEC)}
    p = OUT / "dataset_meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9_100_000)
    a = ap.parse_args()
    p = run([a.seed0 + i * 1000 for i in range(a.n_days)])
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"정답지 {d['n_labels_usable']:,}행 ({len(d['seeds'])}일)")
    for s in d["per_seed"]:
        st = s["stats"]
        print(f"  시드 {s['seed']}  사용 {s['n_labels_usable']:>5}/{s['n_labels_all']:>5}  "
              f"부담 평균 {st['burden_mean']:.3f} (자기 {st['own_share']:.0%} · "
              f"뒤차 {st['blocked_share']:.0%} · 본선 {st['vessel_share']:.0%})  "
              f"칸 {st['n_slots_covered']}/48 · 블록 {st['n_blocks_covered']}/21")
    print("DONE", p)
