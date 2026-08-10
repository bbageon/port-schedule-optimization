"""YR-158 — 야드크레인 서비스시간 **런 실측** 계측 (현실성 재제출 선결, 2026-08-10).

앵커 crane_service_time([120,180]s — PEMA IP15 등 외부 문헌)의 시뮬 대응값은
엔진 유도치(자기참조 금지)가 아니라 **실행 런의 실측 분포**여야 한다. 엔진이
작업별 service_start/service_end 를 기록하므로 외부트럭 작업의 (완료−시작)을
전수 수집한다.

★단위 정합(1차 계측에서 발견·정정): 문헌 앵커(PEMA "2~3분/사이클")는 **컨테이너
1리프트(move)** 기준인데, 트럭 1건 처리시간에는 재취급(blocker 파내기) 리프트가
포함된다 — 장치율 0.65 에선 재취급이 흔해 1건 전체 평균(~268초)이 대역을 넘지만
이는 단위 불일치다. 정본 비교값 = **1건 처리시간 ÷ (1 + 재취급 수)** 의 셀별 평균
(엔진이 작업별 rehandle_count 를 기록). 1건 전체 평균은 참고로 함께 보고한다.
계측 셀 = 확정 무대 2곳(주 w3-L150·대조 w1-L150, rep0 시드·장치율 0.65 정본).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, quantiles

from ..integrated.terminal_stream import ObservationContract
from .yr150_h21_pilot import _git, _sha256
from .yr150_h21_wip_pilot import run_cell
from .yr157_band_qual import (N_HOTSPOT, WEIGHTS, cell_seed, hotspot_seed)
from ..integrated.terminal_stream import (TerminalStreamParams, hotspot_rotation)
from ..integrated.yard_layout import terminal_layout

OUT = Path("outputs/reports/yr158_crane_service_probe")
CELLS = ((3.0, 150), (1.0, 150))          # 확정 무대: 주 w3-L150 · 대조 w1-L150


def measure_cell(w: float, load: int) -> dict:
    """run_cell 을 재사용하되 sims 의 작업별 서비스시간을 전수 수집한다."""
    from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                        _apply, _wait_of)
    from ..integrated.candidates import CandidateGenerator
    from ..integrated.multiblock import MultiBlockTerminal
    from ..integrated.profiles import build_h21_profile
    from ..integrated.sell_review import ANNOUNCE_LEAD_S
    from ..integrated.terminal_stream import (WipAdmissionController,
                                              admission_epochs, build_fixed_wip)
    from .yr088_joint_rl import LEVEL
    from .yr149_load_cells import _sim_from
    from .yr157_band_qual import SEED as BAND_SEED

    obs = ObservationContract()
    layout = terminal_layout()
    seed = cell_seed(w, load)
    hs = hotspot_rotation(layout, hotspot_seed(w), N_HOTSPOT) if w > 1.0 else ()
    params = TerminalStreamParams(load_4h=load, hotspot_blocks=hs, hotspot_weight=w)
    built = build_fixed_wip(build_h21_profile(), seed, wip_target=load, obs=obs,
                            layout=layout, params=params, background_seed=BAND_SEED)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    ctrl = WipAdmissionController(built["pool"], wip_target=load,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=obs.observe_s)
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    mbt.run(policy, review_fn=ctrl.review)
    durs, per_move, moves = [], [], 0
    for sim in mbt.blocks.values():
        for j in sim.jobs.values():
            if (j.is_external_truck
                    and getattr(j, "service_start", None) is not None
                    and getattr(j, "service_end", None) is not None):
                d = j.service_end - j.service_start
                if d > 0:
                    n_moves = 1 + int(getattr(j, "rehandle_count", 0) or 0)
                    durs.append(d)
                    per_move.append(d / n_moves)
                    moves += n_moves
    qs = quantiles(per_move, n=10) if len(per_move) >= 10 else []
    return {"cell": f"w{w}-L{load}", "seed": seed, "n_jobs": len(durs),
            "n_moves": moves,
            "mean_per_move_s": round(fmean(per_move), 2) if per_move else None,
            "p10_per_move_s": round(qs[0], 2) if qs else None,
            "p90_per_move_s": round(qs[-1], 2) if qs else None,
            "mean_per_job_s": round(fmean(durs), 2) if durs else None}


def run() -> dict:
    cells = [measure_cell(w, load) for w, load in CELLS]
    means = [c["mean_per_move_s"] for c in cells
             if c["mean_per_move_s"] is not None]
    res = {"task": "YR-158-crane-service-probe",
           "unit_note": "문헌 앵커(PEMA 2~3분/사이클)는 1리프트(move) 기준 — 정본 "
                        "비교값 = 1건 처리시간÷(1+재취급 수)의 셀별 평균. 1건 전체 "
                        "평균(mean_per_job_s)은 재취급 포함 참고값.",
           "simulated_mean_range_s": [min(means), max(means)] if means else None,
           "cells": cells,
           "runtime": {"commit": _git("rev-parse", "HEAD"),
                       "observation": ObservationContract().as_dict()}}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "crane_service.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "crane_service.json.sha256").write_text(_sha256(p) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("simulated_mean_range_s", "cells")},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    print("DONE")
