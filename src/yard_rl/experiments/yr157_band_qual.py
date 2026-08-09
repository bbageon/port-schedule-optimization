"""YR-157 — 부하 대역 재정의 **자격 시험** (사용자 확정 2026-08-09: 몰아주기 주축 6셀).

■ 동결 설계 (결과 열람 전 — 유도는 이론·문헌만, 재자격 수치 역산 금지)
  · 주축 = **hotspot 집중도** w ∈ {1(균등), 3, 5}: 트럭 행선지 배분 p 에서 hotspot
    4블록에 w 배 가중. 총량은 현실 정합 대역 유지 — 부축 L ∈ {100, 150}.
  · 유도 근거: YC 1기 20~30 작업/h(PEMA 2~3분/작업·본선 cadence 2.4분 정합), 본선
    스트림 블록의 트럭 몫 15~35/h. L 의 내부 일꾼 = L×15/(15+30) (턴타임 15분·예고
    30분). w=5 에서 hotspot 블록 유입 ≈ 27/h → 이용률 0.77~1.8 (BUSY~OVERLOADED),
    나머지 블록은 CLEAR — 세 구간 공존이 목표(이송 연구의 무대).
  · hotspot 4곳은 **시드 추첨**(`hotspot_rotation` — 특정 블록 운 제거), 본선 배치와
    독립 스트림. 배경은 전 셀 공통(`background_seed`).
■ 자격 검사 = YR-150 W1~W9 그대로 (같은 검사를 새 대역에서). 상태 분류는 사후 관찰.
■ 성능 주장 없음 — 이 하네스는 환경 자격이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.repro import code_dirty
from ..integrated.terminal_stream import (ObservationContract,
                                          TerminalStreamParams, hotspot_rotation)
from ..integrated.yard_layout import terminal_layout
from .yr150_h21_pilot import _git, _sha256
from .yr150_h21_wip_pilot import run_cell

OUT = Path("outputs/reports/yr157_band_qual")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-157-h21-load-band-extension.md")
SEED = 6_600_000                          # YR-157 전용 시드 대역 (미사용 대역)
WEIGHTS = (1.0, 3.0, 5.0)                 # 집중도 축 (동결)
LOADS = (100, 150)                        # 유지 대수 축 (동결 — 현실 정합 대역)
N_HOTSPOT = 4                             # hotspot 블록 수 (동결)


def cell_seed(w: float, load: int) -> int:
    return SEED + int(w * 10) * 1000 + load


def run_one(w: float, load: int, obs: ObservationContract) -> dict:
    layout = terminal_layout()
    seed = cell_seed(w, load)
    hs: tuple[str, ...] = ()
    if w > 1.0:
        hs = hotspot_rotation(layout, seed, N_HOTSPOT)
    params = TerminalStreamParams(load_4h=load, hotspot_blocks=hs, hotspot_weight=w)
    cell = run_cell(load, obs, params=params, seed=seed, background_seed=SEED)
    # hotspot 사후 관측(판정 아님) — 몰린 블록의 최대 내부 대수(집중이 실제 생겼는가).
    cell.update({
        "hotspot_weight": w, "hotspot_blocks": list(hs), "cell_seed": seed,
        "hotspot_wip_peak": {
            b: max(s["wip_by_block"].get(b, 0) for s in cell["snapshots"])
            for b in hs},
    })
    return cell


def run() -> dict:
    obs = ObservationContract()
    cells = [run_one(w, load, obs) for w in WEIGHTS for load in LOADS]
    checks = {
        "W1_wip_maintained": all(c["wip_maintained_pm5pct"] for c in cells),
        "W2_ledger_conserved": all(c["ledger_conserved"] for c in cells),
        "W3_deterministic_build": all(
            c["internal_checks"]["deterministic_replay"] for c in cells),
        "W4_vessels_spread": all(
            len(set(c["vessel_blocks"])) >= 5
            and max(c["vessel_starts_s"]) >= 0.7 * obs.observe_s for c in cells),
        "W5_block_level_snapshots": all(
            all(len(s["wip_by_block"]) == 21 for s in c["snapshots"]) for c in cells),
        "W6_profile_is_h21_yt": all(c["profile"]["transfer_kind"] == "YT" for c in cells),
        "W7_internal_checks_pass": all(all(c["internal_checks"].values())
                                       for c in cells),
        "W8_pool_not_exhausted": all(c["pool_exhausted_at"] is None for c in cells),
        "W9_flow_fallback_zero": all(c["flow_fallbacks_used"] == 0 for c in cells),
        "no_policy_exceptions": all(c["policy_exceptions"] == 0 for c in cells),
    }
    verdict = {
        "qualification_all_pass": all(checks.values()),
        "checks": checks,
        "states": {f"w{c['hotspot_weight']}-L{c['wip_target']}":
                   c["classification"]["state"] for c in cells},
        "note": "YR-157 6셀 자격 — 성능 주장 없음. 상태 분류는 사후 관찰이며 판정 "
                "임계가 아니다. 성능은 처리량+시간당 비용 공동 판정만 허용.",
    }
    dirty = bool(code_dirty())
    res = {"task": "YR-157", "structure": "H-21", "design": "hotspot-primary-6cell",
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "remote_ref": "origin/master",
                       "remote_head": _git("rev-parse", "origin/master"),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "params": {"WEIGHTS": list(WEIGHTS), "LOADS": list(LOADS),
                                  "N_HOTSPOT": N_HOTSPOT,
                                  "observation": obs.as_dict()},
                       "seeds": {"cells": [cell_seed(w, l)
                                           for w in WEIGHTS for l in LOADS],
                                 "background": SEED}},
           "verdict": verdict, "cells": cells}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "band_qual.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "band_qual.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        obs = ObservationContract(warmup_s=1800.0, measure_s=7200.0, snapshot_s=300.0)
        c = run_one(5.0, 100, obs)
        print(json.dumps({k: v for k, v in c.items()
                          if k not in ("snapshots", "hourly", "vessel_placement",
                                       "flow_fallback_jobs")},
                         ensure_ascii=False, indent=1))
    elif a.run:
        run()
    print("DONE")
