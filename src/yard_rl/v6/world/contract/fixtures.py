"""최소 통합 fixture — 전 최종 도메인 + 결측/가정 표기 (YR-035).

Python builder 로 둔다: TOK 게이팅으로 known/assumed 를 값과 함께 산출해야 하고
`validate_all` 통과 객체를 반환해야 하므로 정적 YAML 로는 부적합. 담는 엔티티:
- 다중 YC 2기(YC-A/YC-B), 공유 본선 후보 1건(joint 중복검사 대상)
- 외부트럭 반출 2건(도착완료·PRE_ADVICE 미도착[eta_confidence 결측]) + 반입 1건
- 본선 양하 = RISK(완료시각 확보) · 선적 = SYMPTOM(완료시각 결측)
- STS·이송장비 누적대기 / 레인 2개+연결 / PRE_REHANDLE·REPOSITION·WAIT 후보
- 결측 케이스(eta_confidence·완료시각)·가정 케이스(remaining_service_time_s imputed·profile_assumed)
반환 전 canonical 왕복 + `validate_all` 통과를 보장한다.
"""
from __future__ import annotations

import hashlib

from ..domain.enums import ControlScope, InformationLevel
from .candidate import Candidate, CandidateSet, padding_candidate
from .cost import make_cost
from .schema import COST_TERMS, SCHEMA_VERSION, CandidateKind
from .serialize import dumps, loads
from .state import (Assignment, GlobalState, JointAction, LaneGraph,
                    LocalObservation)
from .transition import TransitionAudit, TransitionRecord
from .validate import validate_all
from .vectors import build_feature_vector
from .vessel import CompletionBasis, VesselUrgency, resolve_mode

_NOW = 3600.0
_END = 28800.0        # 8h 운영구간
_LEVEL = InformationLevel.PRE_ADVICE
_KINDS = list(CandidateKind)


def _kind_idx(kind: CandidateKind) -> float:
    return _KINDS.index(kind) / (len(_KINDS) - 1)


def _cand_fv(kind, *, is_external=False, is_vessel=False, cum_wait_s=None,
             long_wait_excess_s=None, arrival_realized_at=None,
             predicted_arrival_gap_s=None, eta_confidence=None, deadline_slack_s=None,
             reach_s=0.0, service_s=0.0, handling=0.0, blockers=0.0, rehandle_s=0.0,
             end_bay=0.0, lane_local=0.0, interference_s=0.0, resequence=0.0,
             vessel_risk_delta=None):
    raw = {
        "action_kind_idx": _kind_idx(kind),
        "is_external": 1.0 if is_external else 0.0,
        "is_vessel": 1.0 if is_vessel else 0.0,
        "cum_wait_s": cum_wait_s,
        "long_wait_excess_s": long_wait_excess_s,
        "predicted_arrival_gap_s": predicted_arrival_gap_s,
        "eta_confidence": eta_confidence,
        "deadline_slack_s": deadline_slack_s,
        "reach_s": reach_s,
        "expected_service_time_s": service_s,
        "expected_handling_count": handling,
        "blocker_count": blockers,
        "expected_rehandle_time_s": rehandle_s,
        "end_bay": end_bay,
        "lane_congestion_local": lane_local,
        "interference_penalty_s": interference_s,
        "resequence_count": resequence,
        "vessel_risk_delta": vessel_risk_delta,
        "contention_risk": 0.0,   # v3 COORD — minimal fixture 는 경합 없음
    }
    realized = {}
    if arrival_realized_at is not None:   # 도착시점 파생 필드의 실현시각
        realized["cum_wait_s"] = arrival_realized_at
        realized["long_wait_excess_s"] = arrival_realized_at
    return build_feature_vector("candidate", raw, now=_NOW, info_level=_LEVEL,
                                realized_at=realized)


def _vessel_fv(*, slack_s=None, risk=None, delay_symptom_score=None, remaining_moves=0.0,
               remaining_service_time_s=None, sts_wait_s=0.0, transfer_wait_s=0.0,
               expected_delay_s=None):
    raw = {"slack_s": slack_s, "risk": risk, "delay_symptom_score": delay_symptom_score,
           "remaining_moves": remaining_moves,
           "remaining_service_time_s": remaining_service_time_s,
           "sts_wait_s": sts_wait_s, "transfer_wait_s": transfer_wait_s,
           "expected_delay_s": expected_delay_s}
    return build_feature_vector("vessel", raw, now=_NOW, info_level=_LEVEL)


def _yc_fv(*, crane_bay, own_queue_len, own_oldest_wait_s, neighbor_load_gap,
           neighbor_min_gap_bay, recent_throughput):
    raw = {"crane_bay": crane_bay, "trolley_row": 0.0, "available_in_s": 0.0,
           "is_loaded": 0.0, "last_move_dir": 1.0, "recent_throughput": recent_throughput,
           "recent_empty_travel_s": 120.0, "assigned_load": 0.0,
           "own_queue_len": own_queue_len, "own_oldest_wait_s": own_oldest_wait_s,
           "neighbor_load_gap": neighbor_load_gap, "neighbor_min_gap_bay": neighbor_min_gap_bay,
           # v3 COORD — minimal fixture: 상대는 SERVE 실행 중 (채널 직렬화 검증용)
           "neighbor_busy_kind": 0.0, "neighbor_busy_target_bay": 18.0,
           "neighbor_available_in_s": 90.0, "recent_yield_count": 1.0}
    return build_feature_vector("yc", raw, now=_NOW, info_level=_LEVEL)


def _queue_fv(*, cand_count, wait_max_s, wait_mean_s, over_sla_count, outbound_share):
    raw = {"cand_count": cand_count, "service_min_s": 60.0, "service_mean_s": 180.0,
           "service_max_s": 420.0, "reach_min_s": 20.0, "reach_mean_s": 90.0,
           "wait_max_s": wait_max_s, "wait_mean_s": wait_mean_s,
           "outbound_share": outbound_share, "short_service_share": 0.5,
           "vessel_urgency_max": 0.7, "lane_cong_mean": 0.4, "over_sla_count": over_sla_count}
    return build_feature_vector("queue", raw, now=_NOW, info_level=_LEVEL)


def _global_fv():
    raw = {"time_frac": _NOW / _END, "shift_idx": 1.0, "vessel_count": 2.0,
           "lane_congestion_mean": 0.4, "lane_congestion_max": 0.7,
           "sts_wait_accum_s": 210.0, "transfer_wait_accum_s": 140.0,
           "backlog_external": 5.0, "backlog_vessel": 2.0, "crane_count": 2.0,
           "load_imbalance": 0.3}
    return build_feature_vector("global", raw, now=_NOW, info_level=_LEVEL)


# ------------------------------------------------------------- 후보 집합
def _yc_a_candidates() -> CandidateSet:
    items = (
        # 0: 도착완료 외부 반출 — mandatory(SLA 초과), lane L1
        Candidate(0, CandidateKind.SERVE,
                  _cand_fv(CandidateKind.SERVE, is_external=True, cum_wait_s=2000.0,
                           long_wait_excess_s=200.0, arrival_realized_at=_NOW - 2000.0,
                           reach_s=45.0, service_s=180.0, handling=1.0, blockers=0.0,
                           end_bay=5.0, lane_local=0.5, resequence=0.0),
                  mandatory=True, ref_job_id="J-OUT-1", resolver_token="JOUT1",
                  lane_id="L1"),
        # 1: 미도착 PRE_ADVICE 외부 반출 — eta_confidence 결측(known=0), cum_wait known=0
        Candidate(1, CandidateKind.SERVE,
                  _cand_fv(CandidateKind.SERVE, is_external=True,
                           arrival_realized_at=_NOW + 600.0, cum_wait_s=0.0,
                           predicted_arrival_gap_s=600.0, eta_confidence=None,
                           reach_s=80.0, service_s=200.0, handling=2.0, blockers=1.0,
                           rehandle_s=150.0, end_bay=12.0, lane_local=0.6),
                  ref_job_id="J-OUT-2", resolver_token="JOUT2", lane_id="L2"),
        # 2: 본선 양하 연계(RISK) — 공유 후보(YC-A·YC-B 자격), lane 없음
        Candidate(2, CandidateKind.SERVE,
                  _cand_fv(CandidateKind.SERVE, is_vessel=True, deadline_slack_s=-300.0,
                           reach_s=120.0, service_s=240.0, handling=1.0, blockers=0.0,
                           end_bay=8.0, vessel_risk_delta=0.15),
                  ref_job_id="J-VES-1", resolver_token="JVES1",
                  eligible_crane_ids=("YC-A", "YC-B")),
        # 3: 도착 전 재조작 선처리 — J-OUT-2 와 동일 gap(+600, 미도착)
        Candidate(3, CandidateKind.PRE_REHANDLE,
                  _cand_fv(CandidateKind.PRE_REHANDLE, is_external=True,
                           predicted_arrival_gap_s=600.0, reach_s=60.0, service_s=0.0,
                           handling=2.0, blockers=2.0, rehandle_s=180.0, end_bay=12.0),
                  ref_job_id="J-OUT-2"),
        # 4: 위치조정 (ref_job 없음)
        Candidate(4, CandidateKind.REPOSITION,
                  _cand_fv(CandidateKind.REPOSITION, reach_s=90.0, end_bay=18.0)),
        # 5: 양보/대기
        Candidate(5, CandidateKind.WAIT, _cand_fv(CandidateKind.WAIT)),
        # 6: 실행 불가 후보 (재조작 슬롯 고갈) — feasible=False + mask_reason
        Candidate(6, CandidateKind.SERVE,
                  _cand_fv(CandidateKind.SERVE, is_external=True, cum_wait_s=300.0,
                           long_wait_excess_s=0.0, arrival_realized_at=_NOW - 300.0,
                           reach_s=70.0, service_s=260.0, handling=3.0, blockers=3.0,
                           rehandle_s=400.0, end_bay=15.0, lane_local=0.5),
                  ref_job_id="J-OUT-3"),
        # 7: 배치 패딩 슬롯 (pad=False, 전 채널 known=0)
        padding_candidate(7),
    )
    pad = (True, True, True, True, True, True, True, False)
    feas = (True, True, True, True, True, True, False, False)
    reason = (None, None, None, None, None, None, "REHANDLE_NO_SLOT", "PAD")
    return CandidateSet("YC-A", items, pad, feas, reason,
                        _queue_fv(cand_count=7.0, wait_max_s=2000.0, wait_mean_s=760.0,
                                  over_sla_count=1.0, outbound_share=0.7))


def _yc_b_candidates() -> CandidateSet:
    items = (
        # 0: 본선 양하 공유 후보 (YC-A 와 동일 resolver_token — joint 중복검사 대상)
        Candidate(0, CandidateKind.SERVE,
                  _cand_fv(CandidateKind.SERVE, is_vessel=True, deadline_slack_s=-300.0,
                           reach_s=140.0, service_s=240.0, handling=1.0, blockers=0.0,
                           end_bay=8.0, vessel_risk_delta=0.15),
                  ref_job_id="J-VES-1", resolver_token="JVES1",
                  eligible_crane_ids=("YC-A", "YC-B")),
        # 1: 외부 반입 처리 — lane L2 (fixture 선택 대상)
        Candidate(1, CandidateKind.SERVE,
                  _cand_fv(CandidateKind.SERVE, is_external=True, cum_wait_s=0.0,
                           long_wait_excess_s=0.0, arrival_realized_at=_NOW - 120.0,
                           reach_s=55.0, service_s=150.0, handling=1.0, blockers=0.0,
                           end_bay=28.0, lane_local=0.4),
                  ref_job_id="J-IN-1", resolver_token="JIN1", lane_id="L2"),
        # 2: 연착 트럭 선제 재조작 — ETA 경과·미도착 = 음수 gap (v2/YR-050 부호 보존 커버)
        Candidate(2, CandidateKind.PRE_REHANDLE,
                  _cand_fv(CandidateKind.PRE_REHANDLE, is_external=True,
                           predicted_arrival_gap_s=-180.0, reach_s=75.0, service_s=90.0,
                           handling=1.0, blockers=1.0, rehandle_s=90.0, end_bay=30.0),
                  ref_job_id="J-OUT-4"),
        # 3: 위치조정
        Candidate(3, CandidateKind.REPOSITION,
                  _cand_fv(CandidateKind.REPOSITION, reach_s=110.0, end_bay=34.0)),
        # 4: 양보/대기
        Candidate(4, CandidateKind.WAIT, _cand_fv(CandidateKind.WAIT)),
    )
    pad = (True, True, True, True, True)
    feas = (True, True, True, True, True)
    reason = (None, None, None, None, None)
    return CandidateSet("YC-B", items, pad, feas, reason,
                        _queue_fv(cand_count=5.0, wait_max_s=800.0, wait_mean_s=300.0,
                                  over_sla_count=0.0, outbound_share=0.25))


# --------------------------------------------------------------- 조립
def _vessels() -> tuple[VesselUrgency, ...]:
    # V1: 완료시각 확보(PLAN_COMPUTED) → RISK
    mode1, asm1 = resolve_mode(planned_completion_s=_NOW + 1800.0,
                               basis=CompletionBasis.PLAN_COMPUTED)
    v1 = VesselUrgency("VES-DISCH-1", mode1, CompletionBasis.PLAN_COMPUTED, asm1,
                       _vessel_fv(slack_s=-300.0, risk=0.7, remaining_moves=40.0,
                                  remaining_service_time_s=1500.0, sts_wait_s=120.0,
                                  transfer_wait_s=80.0, expected_delay_s=300.0))
    # V2: 완료시각 결측 → SYMPTOM · remaining_service_time_s 는 assumed_default imputed
    mode2, asm2 = resolve_mode(planned_completion_s=None, basis=None)
    v2 = VesselUrgency("VES-LOAD-2", mode2, None, asm2,
                       _vessel_fv(delay_symptom_score=0.4, remaining_moves=25.0,
                                  remaining_service_time_s=None, sts_wait_s=90.0,
                                  transfer_wait_s=60.0))
    return (v1, v2)


def _global_state() -> GlobalState:
    return GlobalState(
        schema_version=SCHEMA_VERSION, episode_id="EP-MIN-001", decision_index=7,
        now_s=_NOW, info_level=_LEVEL.value, control_scope=ControlScope.PLUS_PRE_REHANDLE.value,
        profile_assumed=True, features=_global_fv(), vessels=_vessels(),
        lane_graph=LaneGraph(("L1", "L2"), (("L1", "L2"),)))


def _observations() -> tuple[LocalObservation, ...]:
    return (
        LocalObservation(SCHEMA_VERSION, "YC-A", _NOW,
                         _yc_fv(crane_bay=5.0, own_queue_len=3.0, own_oldest_wait_s=2000.0,
                                neighbor_load_gap=0.2, neighbor_min_gap_bay=15.0,
                                recent_throughput=8.0),
                         _yc_a_candidates()),
        LocalObservation(SCHEMA_VERSION, "YC-B", _NOW,
                         _yc_fv(crane_bay=30.0, own_queue_len=2.0, own_oldest_wait_s=800.0,
                                neighbor_load_gap=-0.2, neighbor_min_gap_bay=15.0,
                                recent_throughput=6.0),
                         _yc_b_candidates()),
    )


def _cost():
    raw = {"truck_wait": 120.0, "long_wait": 30.0, "crane_travel": 200.0,
           "empty_travel": 50.0, "rehandle": 2.0, "sts_wait": 60.0, "transfer_wait": 40.0,
           "vessel_delay": 300.0, "depart_delay": 0.0, "lane_cong": 15.0,
           "interference": 5.0, "resequence": 1.0, "imbalance": 0.2}
    scale = {"truck_wait": 600.0, "long_wait": 1800.0, "crane_travel": 1000.0,
             "empty_travel": 1000.0, "rehandle": 5.0, "sts_wait": 600.0,
             "transfer_wait": 600.0, "vessel_delay": 600.0, "depart_delay": 600.0,
             "lane_cong": 100.0, "interference": 100.0, "resequence": 10.0, "imbalance": 1.0}
    weight = {k: 1.0 for k in COST_TERMS}
    return make_cost(interval_start_s=_NOW, interval_end_s=_NOW + 300.0,
                     raw=raw, scale=scale, weight=weight, lambda_vessel=3.0)


def _collect_audit(state: GlobalState, obs) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing: list[str] = []
    assumed: list[str] = []

    def scan(path, fv):
        for nm, kn, asm in zip(fv.names, fv.known, fv.assumed):
            if not kn:
                missing.append(f"{path}.{nm}")
            if asm:
                assumed.append(f"{path}.{nm}")

    scan("state", state.features)
    for i, v in enumerate(state.vessels):
        scan(f"state.vessels[{i}]", v.features)
    for o in obs:
        scan(f"obs[{o.crane_id}]", o.features)
        scan(f"obs[{o.crane_id}].queue", o.candidates.queue_summary)
        for c, pad in zip(o.candidates.items, o.candidates.pad_mask):
            if pad:
                scan(f"obs[{o.crane_id}].cand[{c.candidate_id}]", c.features)
    return tuple(sorted(missing)), tuple(sorted(assumed))


def build_minimal_transition() -> TransitionRecord:
    state = _global_state()
    obs = _observations()
    joint = JointAction(SCHEMA_VERSION, _NOW, (
        Assignment("YC-A", 0, CandidateKind.SERVE, "local_argmin"),
        Assignment("YC-B", 1, CandidateKind.SERVE, "central_resolver"),
    ))
    missing, assumed = _collect_audit(state, obs)
    audit = TransitionAudit(
        built_at_now_s=_NOW, info_level=_LEVEL.value, ablation_off=(),
        missing_fields=missing, assumed_fields=assumed, forbidden_touched=(),
        event_stream_hash=hashlib.sha1(b"itc-v1-minimal").hexdigest()[:16])
    rec = TransitionRecord(
        schema_version=SCHEMA_VERSION, episode_id="EP-MIN-001", decision_index=7,
        dt_s=300.0, state=state, observations=obs, joint_action=joint, cost=_cost(),
        next_state=state, next_observations=obs, terminal=False, audit=audit)
    validate_all(rec)
    canon = loads(dumps(rec))   # canonical float 형태로 정규화 (round-trip 기준)
    validate_all(canon)
    return canon
