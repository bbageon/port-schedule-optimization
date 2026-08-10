"""YR-153 — 현실성(scenario_validity) 게이트 **재제출** (33차 감사 ③, 2026-08-09).

저장된 공식 게이트의 현실성 축은 구 1단계 제출본(구 앵커 [120,300]·유입량 계약)이라
낡았다. 최신 정본으로 재제출한다 — **통과 조작이 아니라 정직 갱신**이다:
  · gate_to_block_time: 정본 결정([180,420] — 사용자 2026-08-09) vs 시뮬 190~410 → 정합
  · vessel_workload: 유도 앵커 [145,170] vs 계획 작업률 150.0(슬롯 5%~95% 해석) → 정합
  · 미수집 3종(crane_service_time·initial_yard_occupancy·truck_arrival_rate)은 그대로
    미수집 → **게이트는 여전히 FAIL 이 예상**되고 그 사실을 갱신 기록한다.
내부타당성·흐름 검사는 YR-157 재자격(18런·pairing 정정본)의 W 검사에서 가져온다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.terminal_stream import TerminalStreamParams, ObservationContract
from ..integrated.yard_layout import terminal_layout
from .gate_harness import AnchorEvidence, judge_scenario_validity
from .yr150_h21_pilot import _git, _sha256

ANCHORS = Path("configs/anchors/external_anchors_v1.json")
BAND = Path("outputs/reports/yr157_band_qual/band_qual.json")
CRANE = Path("outputs/reports/yr158_crane_service_probe/crane_service.json")
OUT = Path("outputs/reports/yr153_research_gates/scenario_resubmit_wip.json")


def planned_vessel_rate() -> float:
    """계획 본선 작업률(moves/h) — 슬롯 공식의 해석적 값(시드 무관: 슬롯 집합 동일)."""
    p = TerminalStreamParams(load_4h=100)
    obs = ObservationContract()
    n, cad = p.vessels_total, p.sts_move_interval_s
    total = 0.0
    for k in range(n):
        start = obs.observe_s * (0.05 + 0.90 * k / (n - 1))
        total += min(p.vessel_moves, (obs.observe_s - start) / cad)
    return total / (obs.observe_s / 3600.0)


def run() -> dict:
    reg = json.loads(ANCHORS.read_text(encoding="utf-8"))
    layout = terminal_layout()
    lo, hi = layout.gate_time_range_s()
    rate = planned_vessel_rate()
    sha = _sha256(ANCHORS)
    anchors = {}
    g = reg["anchors"]["gate_to_block_time"]
    anchors["gate_to_block_time"] = AnchorEvidence(
        observed_min=float(g["observed_range"][0]),
        observed_max=float(g["observed_range"][1]),
        simulated_min=lo, simulated_max=hi, unit=g["unit"],
        source_path=str(ANCHORS.as_posix()), source_sha256=sha)
    v = reg["anchors"]["vessel_workload"]
    anchors["vessel_workload"] = AnchorEvidence(
        observed_min=float(v["observed_range"][0]),
        observed_max=float(v["observed_range"][1]),
        simulated_min=rate, simulated_max=rate, unit=v["unit"],
        source_path=str(ANCHORS.as_posix()), source_sha256=sha)

    band = json.loads(BAND.read_text(encoding="utf-8"))

    # ★2026-08-10 확장 — 5종 전부 제출 (미수집 3종 등록 + 재보정·계측 완료 후):
    # 장치율 = H-21 계약값 0.65(사용자 결정 — 전 셀 동일 계약값이라 점 범위)
    o = reg["anchors"]["initial_yard_occupancy"]
    fill = TerminalStreamParams(load_4h=100).fill_ratio
    anchors["initial_yard_occupancy"] = AnchorEvidence(
        observed_min=float(o["observed_range"][0]),
        observed_max=float(o["observed_range"][1]),
        simulated_min=fill, simulated_max=fill, unit=o["unit"],
        source_path=str(ANCHORS.as_posix()), source_sha256=sha)
    # 게이트 처리율(재정의 지표) = 자격 18런의 관측창 처리율 실측 대역
    t = reg["anchors"]["truck_arrival_rate"]
    ths = [c["throughput_per_h"] for c in band["cells"]]
    anchors["truck_arrival_rate"] = AnchorEvidence(
        observed_min=float(t["observed_range"][0]),
        observed_max=float(t["observed_range"][1]),
        simulated_min=min(ths), simulated_max=max(ths), unit=t["unit"],
        source_path=str(ANCHORS.as_posix()), source_sha256=sha)
    # 크레인 서비스시간 = 확정 무대 2셀의 **런 실측 평균** 대역(자기참조 금지 —
    # 엔진 유도치가 아니라 실행 결과 계측. 단위 = 문헌과 같은 '평균')
    cs = json.loads(CRANE.read_text(encoding="utf-8"))
    lo_m, hi_m = cs["simulated_mean_range_s"]
    c = reg["anchors"]["crane_service_time"]
    anchors["crane_service_time"] = AnchorEvidence(
        observed_min=float(c["observed_range"][0]),
        observed_max=float(c["observed_range"][1]),
        simulated_min=float(lo_m), simulated_max=float(hi_m), unit=c["unit"],
        source_path=str(ANCHORS.as_posix()), source_sha256=sha)
    cells = band["cells"]
    internal = {k: all(c["internal_checks"][k] for c in cells)
                for k in cells[0]["internal_checks"]}
    checks = band["verdict"]["checks"]
    # W 검사 → 흐름 계약 매핑 (근거를 evidence 에 명시)
    flow = {"continuous_arrivals": bool(checks["W1_wip_maintained"]),
            "warmup_excluded": True,                       # 판정창 계약(워밍업 제외)
            "fixed_measurement_window": True,              # ObservationContract 동결
            "load_state_classified": True,                 # CLEAR/BUSY/OVERLOADED 사후 분류
            "flow_balance_consistent_with_classification":
                bool(checks["W2_ledger_conserved"])}

    outcome = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=False, root=".")

    res = {
        "runtime": {"commit": _git("rev-parse", "HEAD"),
                    "source_qualification": str(BAND.as_posix()),
                    "source_sha256": _sha256(BAND)},
        "scenario_gate": {
            "outcome": {"status": outcome.status.value,
                        "summary": outcome.summary,
                        "reasons": list(outcome.reasons),
                        "evidence": outcome.evidence},
            "anchor_registry_sha256": sha,
            "anchors_unavailable": sorted(reg.get("unavailable", {})),
            "crane_service_measurement": {
                "path": str(CRANE.as_posix()), "sha256": _sha256(CRANE),
                "note": "시뮬측 값 = 확정 무대 2셀 런 실측(사이클당 평균 — 재취급 "
                        "리프트로 나눠 문헌 단위와 정합. 1건 전체 평균은 참고 보고)"},
            "flow_mapping_note": "continuous_arrivals=W1(교체 투입 유지)·flow_balance="
                                 "W2(장부 보존) — YR-157 18런(pairing 정정본) W 검사에서 유도",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(json.dumps({"status": outcome.status.value,
                      "reasons": list(outcome.reasons)}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    print("DONE")
