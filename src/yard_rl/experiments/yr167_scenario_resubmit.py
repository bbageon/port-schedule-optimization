"""YR-167 — 현실성(scenario_validity) 게이트 **5차 계약 재제출** (40차 감사).

저장된 공식 게이트의 현실성 PASS 는 **4차 계약**(고정 재공량·YR-157 band) 자료다.
YR-168 은 5차 계약(도착률·24시간 이중 피크)에서 돌리므로, 실험하지 않는 무대의
증거로 현실성을 주장하는 상태가 된다 — 39차 감사가 지적한 "코드↔대시보드 불일치"와
같은 종류다. **인가 통과를 위해서가 아니라 정직을 위해** 5차 자료로 갱신한다
(실제 인가 차단 사유는 board 표류이며 게이트 재발행으로 닫힌다).

앵커 5종의 시뮬측 값 출처 — 전부 5차 무대 실측:
  · gate_to_block_time  : 레이아웃 정본 범위(계약값, 계약 불변)
  · vessel_workload     : YR-167 자격의 **실현** 작업률(계획값 아님)
  · initial_yard_occupancy: 계약값 0.65 (점 범위)
  · truck_arrival_rate  : 측정창 게이트 처리율 — 앵커 정의가 "24h 평균 환산(피크 아님)"
                          이므로 **평균 처리율**로 대조한다. 순간 시간대별 도착
                          (22~461대/h)은 설계상 앵커 범위를 넘으며, 그 근거는
                          앵커가 아니라 사전등록 §A 의 이중 피크 문헌이다 — 병기 보고.
  · crane_service_time  : YR-168 관찰 모드의 **5차 무대 실측** 사이클당 평균
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.terminal_stream import DIURNAL_DAY_TOTAL, TerminalStreamParams
from ..integrated.yard_layout import terminal_layout
from .gate_harness import AnchorEvidence, judge_scenario_validity
from .yr150_h21_pilot import _git, _sha256

ANCHORS = Path("configs/anchors/external_anchors_v1.json")
QUAL = Path("outputs/reports/yr167_diurnal_qual/diurnal_qual.json")
OBSERVE = Path("outputs/reports/yr168_observe/observe.json")
PREREG = Path(".claude/docs/strategy-history/"
              "2026-08-11-24시간-이중피크-상수-유도-사전등록.md")
OUT = Path("outputs/reports/yr153_research_gates/scenario_resubmit_diurnal.json")


def run() -> dict:
    reg = json.loads(ANCHORS.read_text(encoding="utf-8"))
    qual = json.loads(QUAL.read_text(encoding="utf-8"))
    obsv = json.loads(OBSERVE.read_text(encoding="utf-8"))
    cells, sha = qual["cells"], _sha256(ANCHORS)
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    anchors: dict[str, AnchorEvidence] = {}

    def _ev(key: str, lo: float, hi: float) -> None:
        a = reg["anchors"][key]
        anchors[key] = AnchorEvidence(
            observed_min=float(a["observed_range"][0]),
            observed_max=float(a["observed_range"][1]),
            simulated_min=lo, simulated_max=hi, unit=a["unit"],
            source_path=str(ANCHORS.as_posix()), source_sha256=sha)

    g_lo, g_hi = layout.gate_time_range_s()
    _ev("gate_to_block_time", g_lo, g_hi)
    rates = [c["W4_detail"]["realized"]["realized_moves_per_h"] for c in cells]
    _ev("vessel_workload", min(rates), max(rates))
    _ev("initial_yard_occupancy", params.fill_ratio, params.fill_ratio)
    th = obsv["summary"]["gate_throughput_per_h"]
    _ev("truck_arrival_rate", min(th), max(th))
    cs = [c["crane_service_s"]["mean"] for c in obsv["cells"]]
    _ev("crane_service_time", min(cs), max(cs))

    # 내부타당성 — 자격 W7 의 bool 항목 전부 (5차 판에서 추가된 항목 포함)
    internal = {k: all(bool(c["W7_internal"][k]) for c in cells)
                for k, v in cells[0]["W7_internal"].items() if isinstance(v, bool)}
    internal["ledger_conservation"] = all(c["W2_ledger_conserved"] for c in cells)
    internal["deterministic_replay"] = bool(
        qual["verdict"]["checks"].get("W3_deterministic_run"))
    flow = {
        # 도착이 명단대로 전건 실현됐는가 (5차는 '유지'가 아니라 '명단 준수'가 계약)
        "continuous_arrivals": all(c["W1p_schedule_honored"] for c in cells),
        "warmup_excluded": True,               # 측정창 [2h,24h) — 부수 관측도 코호트 기준
        "fixed_measurement_window": True,      # OBS_24H 동결
        "load_state_classified": True,         # 시간대별 도착·체류·재공 곡선 보고
        "flow_balance_consistent_with_classification":
            all(c["W8p_conservation"]["ok"] for c in cells),   # 재고 보존 항등식
    }
    outcome = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=False, root=".")

    hourly = cells[0]["obs_hourly_arrivals"]
    res = {
        "runtime": {"commit": _git("rev-parse", "HEAD"),
                    "contract": "diurnal_24h",
                    "source_qualification": str(QUAL.as_posix()),
                    "source_qualification_sha256": _sha256(QUAL),
                    "source_observation": str(OBSERVE.as_posix()),
                    "source_observation_sha256": _sha256(OBSERVE),
                    "prereg": str(PREREG.as_posix()),
                    "prereg_sha256": _sha256(PREREG)},
        "scenario_gate": {
            "outcome": {"status": outcome.status.value, "summary": outcome.summary,
                        "reasons": list(outcome.reasons), "evidence": outcome.evidence},
            "anchor_registry_sha256": sha,
            "anchors_unavailable": sorted(reg.get("unavailable", {})),
            "instantaneous_arrival_range_per_h": [min(hourly), max(hourly)],
            "arrival_anchor_note":
                "truck_arrival_rate 앵커는 원장 정의상 '24h 평균 환산(피크 아님)'이라 "
                "평균 처리율로 대조했다. 순간 시간대별 도착은 설계상 앵커 상단을 넘으며"
                "(야간 22 ~ 피크 461대/h), 그 근거는 앵커가 아니라 사전등록 §A 의 "
                "주야 이중 피크 문헌이다 — 앵커 통과로 피크 현실성을 주장하지 않는다.",
            "flow_mapping_note":
                "continuous_arrivals=W1'(명단 준수·계획=투입=등록=3,600)·"
                "flow_balance=W8'(재고 보존 항등식). 4차의 W1(재공량 유지)·W2 매핑을 "
                "5차 계약 검사로 교체했다.",
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
