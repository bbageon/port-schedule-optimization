"""도메인 완전성 — 모든 최종 도메인 + 결측/가정 표기 (YR-035 수용기준 자동회귀)."""
from yard_rl.contract import CandidateKind, build_minimal_transition


def test_multi_yc_present():
    """다중 YC 2기 + neighbor(MULTI_YC) 채널 채움."""
    rec = build_minimal_transition()
    cranes = {o.crane_id for o in rec.observations}
    assert cranes == {"YC-A", "YC-B"}
    assert rec.observations[0].features.known_of("neighbor_load_gap") is True
    assert rec.state.features.known_of("load_imbalance") is True


def test_external_truck_in_and_out():
    """외부트럭 반입·반출(도착완료·미도착 PRE_ADVICE) 모두 표현."""
    rec = build_minimal_transition()
    reals = [c for o in rec.observations for c in o.candidates.real_items]
    ext = [c for c in reals if c.features.value_of("is_external") == 1.0]
    assert len(ext) >= 3
    # 도착완료: cum_wait known / 미도착: predicted_arrival_gap known + eta_confidence 결측
    arrived = [c for c in ext if c.features.known_of("cum_wait_s")]
    pre_adv = [c for c in ext if c.features.known_of("predicted_arrival_gap_s")]
    assert arrived and pre_adv
    assert any(not c.features.known_of("eta_confidence") for c in pre_adv)  # 결측 케이스


def test_vessel_risk_and_symptom():
    from yard_rl.contract import VesselUrgencyMode
    rec = build_minimal_transition()
    modes = {v.mode for v in rec.state.vessels}
    assert VesselUrgencyMode.RISK in modes
    assert VesselUrgencyMode.SYMPTOM in modes


def test_transfer_equipment_waits():
    """STS·AGV/SC/YT 이송장비 누적대기."""
    rec = build_minimal_transition()
    assert rec.state.features.value_of("sts_wait_accum_s") > 0
    assert rec.state.features.value_of("transfer_wait_accum_s") > 0


def test_lane_graph_connected():
    """레인 2개 + 연결."""
    rec = build_minimal_transition()
    assert rec.state.lane_graph.lane_ids == ("L1", "L2")
    assert ("L1", "L2") in rec.state.lane_graph.edges
    lanes = {c.lane_id for o in rec.observations for c in o.candidates.real_items
             if c.lane_id}
    assert {"L1", "L2"} <= lanes


def test_all_candidate_kinds():
    rec = build_minimal_transition()
    kinds = {c.kind for o in rec.observations for c in o.candidates.real_items}
    assert kinds == {CandidateKind.SERVE, CandidateKind.PRE_REHANDLE,
                     CandidateKind.REPOSITION, CandidateKind.WAIT}


def test_missing_and_assumed_markers_present():
    """수용기준: 결측·가정 표기가 실제로 존재 + 가정 프로파일 표시."""
    rec = build_minimal_transition()
    assert len(rec.audit.missing_fields) > 0
    assert len(rec.audit.assumed_fields) > 0
    assert rec.state.profile_assumed is True
    # 가정 근거: imputed 채널 (vessel remaining_service_time_s)
    v2 = rec.state.vessels[1]
    assert v2.features.channel("remaining_service_time_s")[2] is True  # assumed=True
