"""YR-158 — 야드크레인 서비스시간 **런 실측** 계측 (현실성 재제출 선결, 2026-08-10).

앵커 crane_service_time([120,180]s — PEMA IP15 등 외부 문헌)의 시뮬 대응값은
엔진 유도치(자기참조 금지)가 아니라 **실행 런의 실측 분포**여야 한다. 엔진이
작업별 service_start/service_end 를 기록하므로 외부트럭 작업의 (완료−시작)을
전수 수집한다.

문헌 앵커는 "평균 서비스시간"의 대역이므로 비교 단위도 **셀별 평균**이다 —
per-작업 최소/최대(재취급 짧은 건·긴 건)를 대역과 직접 비교하면 단위가 어긋난다.
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
    durs = []
    for sim in mbt.blocks.values():
        for j in sim.jobs.values():
            if (j.is_external_truck
                    and getattr(j, "service_start", None) is not None
                    and getattr(j, "service_end", None) is not None):
                d = j.service_end - j.service_start
                if d > 0:
                    durs.append(d)
    qs = quantiles(durs, n=10) if len(durs) >= 10 else []
    return {"cell": f"w{w}-L{load}", "seed": seed, "n_jobs": len(durs),
            "mean_s": round(fmean(durs), 2) if durs else None,
            "p10_s": round(qs[0], 2) if qs else None,
            "p90_s": round(qs[-1], 2) if qs else None,
            "min_s": round(min(durs), 2) if durs else None,
            "max_s": round(max(durs), 2) if durs else None}


def run() -> dict:
    cells = [measure_cell(w, load) for w, load in CELLS]
    means = [c["mean_s"] for c in cells if c["mean_s"] is not None]
    res = {"task": "YR-158-crane-service-probe",
           "unit_note": "문헌 앵커(평균 대역)와 단위 일치를 위해 비교값 = 셀별 평균. "
                        "분포(p10/p90/min/max)는 참고 보고.",
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
