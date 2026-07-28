"""YR-109 / YR-106-b 게이트 A·B — 본선 마감 물리 정합의 효과 측정 (재현 가능판).

■ 왜 이 파일이 새로 생겼나 (게이트 B)
YR-109 가 board evidence 로 박제한 `legacy_vs_achievable.json` 을 만든 스크립트가
**저장소에 없었다**. 표를 재생산할 수 없는 evidence 는 연구 기록으로 성립하지 않는다.
이 파일이 그 구멍을 메우고, 앞으로의 판정런이 지켜야 할 3가지를 함께 구현한다:
  ① CLI (모듈 상수 토글 금지 — 커밋된 코드로 재현),
  ② 재현 스탬프(절대 시드·파라미터 전문·git 커밋·프로파일),
  ③ 시드별 원자료(채널 13항 분해·guard 지표·본선 물리 상태) 박제.

■ 측정 내용
같은 시드 짝에서 `vessel_deadline_achievable` OFF/ON 을 돌려 ①구조적 최소초과 ②채널별
분산·필요 표본수 ③본선 채널 방향이 어떻게 달라지는지 비교한다. arm 구성은 YR-105
(BASE/VGUARD)를 그대로 쓰되 **판정이 아니라 계측**이 목적이다.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from ..integrated.evalkit import CHANNELS, paired
from ..integrated.profiles import build_calibrated_profile
from ..integrated.repro import repro_stamp, vessel_physics_rows
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from . import yr105_conditional_transfer as y105

OUT = Path("outputs/reports/yr109_deadline_physics")
DELTA = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0, "total": 10.0}
SD_CONF = 0.80          # 필요 표본수는 파일럿 sd 의 80% 상측한계로 (보수적)


def _physics(cell, seed, achievable) -> list[dict]:
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
                            time_contract_v2=True, gate_block_contract=True,
                            vessel_deadline_achievable=achievable)
    return vessel_physics_rows(generate_terminal_scenario(prof, seed, p))


def run_mode(achievable: bool, band: str, n_seeds: int) -> dict:
    """OFF/ON 한쪽을 n_seeds 시드로 실행 — arm 은 YR-105 의 BASE/VGUARD."""
    prev = y105.ACHIEVABLE_DEADLINE
    y105.ACHIEVABLE_DEADLINE = achievable          # 하네스 재사용 (CLI 로만 켜진다)
    try:
        rows = []
        base_a, base_b = y105.BANDS[band]
        for i in range(n_seeds):
            b = y105.run_arm(i, band, vessel_guard=False)
            v = y105.run_arm(i, band, vessel_guard=True)
            rows.append({
                "i": i, "seed_A": base_a + i, "seed_B": base_b + i,
                "base": b, "vguard": v,
                "physics_A": _physics(y105.CELL_A, base_a + i, achievable),
                "physics_B": _physics(y105.CELL_B, base_b + i, achievable)})
            print(f"[{'ON ' if achievable else 'OFF'} s{i}] BASE={b['total']:.2f} "
                  f"VG={v['total']:.2f} d={v['total'] - b['total']:+.2f}", flush=True)
    finally:
        y105.ACHIEVABLE_DEADLINE = prev

    floors = [p["structural_min_overrun_s"] for r in rows
              for p in r["physics_A"] + r["physics_B"]]
    chans = {}
    for ch in list(CHANNELS) + ["total"]:
        d = [r["vguard"]["chan"][ch] - r["base"]["chan"][ch] for r in rows]
        p = paired(d, delta_interest=DELTA[ch], sd_conf=SD_CONF)
        chans[ch] = p.as_dict()
    vessel_share = _share(rows, "vessel")
    return {"achievable": achievable, "n": len(rows), "rows": rows, "channels": chans,
            "structural_floor_s": {"max": round(max(floors), 1) if floors else None,
                                   "mean": round(sum(floors) / len(floors), 1) if floors else None,
                                   "all_zero": all(f == 0.0 for f in floors)},
            "vessel_share_of_base": vessel_share,
            "base_total_mean": round(sum(r["base"]["total"] for r in rows) / len(rows), 3)}


def _share(rows, ch) -> float:
    tot = sum(r["base"]["chan"]["total"] for r in rows)
    return round(sum(r["base"]["chan"][ch] for r in rows) / tot, 4) if tot else 0.0


def run(band: str, n_seeds: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    off = run_mode(False, band, n_seeds)
    on = run_mode(True, band, n_seeds)
    base_a, base_b = y105.BANDS[band]
    res = {
        "repro": repro_stamp(
            experiment="YR-109/106-b 게이트A — 마감 물리 정합 계측",
            seeds={"A": [base_a + i for i in range(n_seeds)],
                   "B": [base_b + i for i in range(n_seeds)]},
            params={"cell_A": calibrated_load_params(y105.CELL_A[0],
                                                     vessel_deadline_mult=y105.CELL_A[1]),
                    "cell_B": calibrated_load_params(y105.CELL_B[0],
                                                     vessel_deadline_mult=y105.CELL_B[1])},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="판정이 아니라 계측 — ①구조적 최소초과 0 여부 ②채널별 필요 표본수 "
                   "③본선 채널 방향. 유의판정은 하지 않는다(n 부족).",
            extra={"delta_interest": DELTA, "sd_conf": SD_CONF, "band": band}),
        "legacy": off, "achievable": on,
    }
    (OUT / f"physics_{band}_n{n_seeds}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n=== YR-109 계측 [{band}] n={n_seeds} ===")
    for name, m in (("legacy", off), ("achievable", on)):
        f = m["structural_floor_s"]
        print(f"{name:11s} 구조초과 최대 {f['max']}s (전부 0: {f['all_zero']}) · "
              f"본선비중 {m['vessel_share_of_base']:.1%} · BASE 총비용 {m['base_total_mean']}")
        for ch in ("truck", "vessel", "total"):
            c = m["channels"][ch]
            print(f"    {ch:7s} {c['mean']:+8.3f} CI[{c['ci'][0]:+7.2f},{c['ci'][1]:+7.2f}] "
                  f"MDE {c['mde80']:6.2f} 필요n(δ={DELTA[ch]:g}) {c['required_n_for_delta']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="select", choices=list(y105.BANDS))
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    run(a.band, a.seeds)
    print("DONE")
