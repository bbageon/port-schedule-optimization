"""공식 용어 스키마와 성과지표가 코드와 맞는가 (사용자 지시 2026-09-22).

■ 트럭 작업 — 여섯 칸
    Truck ETA(통지) → 게이트 인 → 블록 도착 → **작업 시작** → 상차 완료 → 게이트 아웃

  ⚠️ "작업 시작" 만 **터미널이 전송하지 않는다**(실데이터에도 없다). 시뮬레이터 내부
  관측이라 **정책 입력으로 쓰면 정보 경계를 넘는다** — 보고·분해 전용이다.

■ 본선 작업 — ETA · ETB · ETW · ETC · ETD
  ETW·ETC·ETD 는 있고 **ETA·ETB 는 없다**. 정박 대기·접이안은 야드 정책이 어찌할 수
  없어 "구조적 유휴" 로 묶어 Φ 에서 뺐기 때문이다(설계 결정 — 빠뜨린 게 아니다).

■ 성과지표 = **턴타임 = 게이트 아웃 − 게이트 인**
"""
from __future__ import annotations

import pytest

from yard_rl.v4.schema.lifecycle import STAGE_FIELD, TRANSMITTED, Stage, turn_time_s
from yard_rl.v4.schema.record import ExecutionRecord
from yard_rl.v4.world.integrated.multiblock import JobRecord, TerminalLedger
from yard_rl.v4.world.integrated.vessel import VesselPlan


def test_truck_stage_names_are_official():
    """트럭 다섯 단계가 공식 용어 이름을 쓴다 — 문자 코드(A/B/C/O) 가 아니다."""
    assert [STAGE_FIELD[s] for s in Stage] == [
        "copino_notice_s", "gate_in_s", "block_in_s", "job_done_s", "gate_out_s"]


def test_service_start_is_recorded_but_not_transmitted():
    """★작업 시작은 **기록하되 전송 단계가 아니다** — 정보 경계."""
    assert hasattr(ExecutionRecord("d"), "service_start_s")
    assert hasattr(JobRecord("j", "Y01", "Y01", "GATE_IN"), "service_start_s")
    # 전송 단계 넷에 작업 시작이 끼면 정책이 그것을 볼 자격이 생긴다 — 막는다
    assert [STAGE_FIELD[s] for s in TRANSMITTED] == [
        "gate_in_s", "block_in_s", "job_done_s", "gate_out_s"]


def test_turn_time_is_gate_out_minus_gate_in():
    """★성과지표 정의 — 턴타임 = 게이트 아웃 − 게이트 인."""
    r = ExecutionRecord("d")
    r.gate_in_s, r.gate_out_s = 100.0, 1_720.0
    assert turn_time_s(r) == 1_620.0                    # 27분


def test_turn_time_splits_into_four_parts():
    """★턴타임을 진입·대기·작업·반출로 가른다 — 어디서 길어졌나를 보려고."""
    led = TerminalLedger()
    rec = JobRecord("j1", "Y01", "Y01", "GATE_IN")
    # 게이트인 0 → 블록 360 → 작업시작 1,000 → 완료 1,600 → 게이트아웃 2,000
    (rec.gate_in_s, rec.block_in_s, rec.service_start_s,
     rec.job_done_s, rec.gate_out_s) = 0.0, 360.0, 1_000.0, 1_600.0, 2_000.0
    led.register(rec)
    p = led.turn_time_parts_s()
    assert p["n"] == 1
    assert (p["진입"], p["대기"], p["작업"], p["반출"]) == (360.0, 640.0, 600.0, 400.0)
    # ★토막의 합은 턴타임과 정확히 같아야 한다
    assert p["턴타임"] == pytest.approx(rec.gate_out_s - rec.gate_in_s)


def test_incomplete_trucks_are_excluded_from_parts():
    """끊긴 트럭은 토막 계산에서 뺀다 — 섞으면 합이 턴타임과 안 맞는다."""
    led = TerminalLedger()
    ok = JobRecord("ok", "Y01", "Y01", "GATE_IN")
    (ok.gate_in_s, ok.block_in_s, ok.service_start_s,
     ok.job_done_s, ok.gate_out_s) = 0.0, 100.0, 200.0, 300.0, 400.0
    half = JobRecord("half", "Y01", "Y01", "GATE_IN")
    half.gate_in_s, half.block_in_s = 0.0, 100.0        # 작업 시작 전에 창이 끝났다
    led.register(ok); led.register(half)
    assert led.turn_time_parts_s()["n"] == 1


def test_vessel_plan_carries_etw_etc_etd():
    """본선 — ETW·ETC·ETD 는 있다 (ETA·ETB 는 설계상 없다)."""
    f = VesselPlan.__dataclass_fields__
    assert "planned_start_s" in f           # ETW
    assert "planned_completion_s" in f      # ETC
    assert "etd_s" in f                     # ETD
    assert not any(k in f for k in ("eta_s", "etb_s")), (
        "ETA·ETB 가 생겼다 — 구조적 유휴를 Φ 에 넣을지 먼저 결정해야 한다")
