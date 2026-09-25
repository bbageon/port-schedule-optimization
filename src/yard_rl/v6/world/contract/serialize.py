"""전이 계약 직렬화 — dict/JSON 왕복 (YR-035).

규약:
- dumps(rec) = json.dumps(to_dict(rec), sort_keys, ensure_ascii=False) — recorder.py 관습 계승.
- 모든 float 은 _canon(round 6·-0 정규화)으로 정규화 → 재직렬화 bit-안정. inf/nan 은 거부.
- 결측은 오직 known=0·value=0.0 으로 표현하며 sentinel float 을 직렬화하지 않는다.
- tuple↔list, Enum↔.value. ID 감사필드(ref_job_id·resolver_token·lane_id·episode_id·
  event_stream_hash)는 텐서 채널이 아닌 dict 필드로 무손실 왕복 → candidate_id 역참조 복원.
- 불변식: loads(dumps(rec)) == rec, dumps(loads(dumps(rec))) == dumps(rec).
"""
from __future__ import annotations

import json

from ..domain.validators import ValidationError
from .candidate import Candidate, CandidateSet
from .cost import CostBreakdown
from .schema import SCHEMA_VERSION, CandidateKind
from .state import (Assignment, GlobalState, JointAction, LaneGraph,
                    LocalObservation)
from .transition import TransitionAudit, TransitionRecord
from .vectors import FeatureVector, _canon
from .vessel import CompletionBasis, VesselUrgency, VesselUrgencyMode


# ------------------------------------------------------------------ helpers
def _floats(xs) -> list[float]:
    return [_canon(x) for x in xs]


def _bools(xs) -> list[bool]:
    return [bool(x) for x in xs]


# ---------------------------------------------------------------- FeatureVector
def _fv_to(fv: FeatureVector) -> dict:
    return {"schema_version": fv.schema_version, "group": fv.group,
            "names": list(fv.names), "value": _floats(fv.value),
            "known": _bools(fv.known), "assumed": _bools(fv.assumed)}


def _fv_from(d: dict) -> FeatureVector:
    return FeatureVector(
        schema_version=d["schema_version"], group=d["group"],
        names=tuple(d["names"]), value=tuple(_floats(d["value"])),
        known=tuple(bool(x) for x in d["known"]),
        assumed=tuple(bool(x) for x in d["assumed"]))


# --------------------------------------------------------------------- vessel
def _vessel_to(v: VesselUrgency) -> dict:
    return {"vessel_id": v.vessel_id, "mode": v.mode.value,
            "completion_basis": v.completion_basis.value if v.completion_basis else None,
            "assumed": bool(v.assumed), "features": _fv_to(v.features)}


def _vessel_from(d: dict) -> VesselUrgency:
    cb = d["completion_basis"]
    return VesselUrgency(
        vessel_id=d["vessel_id"], mode=VesselUrgencyMode(d["mode"]),
        completion_basis=CompletionBasis(cb) if cb is not None else None,
        assumed=bool(d["assumed"]), features=_fv_from(d["features"]))


# ------------------------------------------------------------------ candidate
def _cand_to(c: Candidate) -> dict:
    return {"candidate_id": int(c.candidate_id), "kind": c.kind.value,
            "features": _fv_to(c.features), "mandatory": bool(c.mandatory),
            "ref_job_id": c.ref_job_id, "resolver_token": c.resolver_token,
            "eligible_crane_ids": list(c.eligible_crane_ids), "lane_id": c.lane_id}


def _cand_from(d: dict) -> Candidate:
    return Candidate(
        candidate_id=int(d["candidate_id"]), kind=CandidateKind(d["kind"]),
        features=_fv_from(d["features"]), mandatory=bool(d["mandatory"]),
        ref_job_id=d["ref_job_id"], resolver_token=d["resolver_token"],
        eligible_crane_ids=tuple(d["eligible_crane_ids"]), lane_id=d["lane_id"])


def _cset_to(cs: CandidateSet) -> dict:
    return {"crane_id": cs.crane_id, "items": [_cand_to(c) for c in cs.items],
            "pad_mask": _bools(cs.pad_mask), "feasible_mask": _bools(cs.feasible_mask),
            "mask_reason": list(cs.mask_reason), "queue_summary": _fv_to(cs.queue_summary)}


def _cset_from(d: dict) -> CandidateSet:
    return CandidateSet(
        crane_id=d["crane_id"], items=tuple(_cand_from(x) for x in d["items"]),
        pad_mask=tuple(bool(x) for x in d["pad_mask"]),
        feasible_mask=tuple(bool(x) for x in d["feasible_mask"]),
        mask_reason=tuple(d["mask_reason"]),
        queue_summary=_fv_from(d["queue_summary"]))


# ------------------------------------------------------------------- state
def _lane_to(g: LaneGraph) -> dict:
    return {"lane_ids": list(g.lane_ids), "edges": [list(e) for e in g.edges]}


def _lane_from(d: dict) -> LaneGraph:
    return LaneGraph(lane_ids=tuple(d["lane_ids"]),
                     edges=tuple((e[0], e[1]) for e in d["edges"]))


def _gs_to(s: GlobalState) -> dict:
    return {"schema_version": s.schema_version, "episode_id": s.episode_id,
            "decision_index": int(s.decision_index), "now_s": _canon(s.now_s),
            "info_level": s.info_level, "control_scope": s.control_scope,
            "profile_assumed": bool(s.profile_assumed), "features": _fv_to(s.features),
            "vessels": [_vessel_to(v) for v in s.vessels],
            "lane_graph": _lane_to(s.lane_graph)}


def _gs_from(d: dict) -> GlobalState:
    return GlobalState(
        schema_version=d["schema_version"], episode_id=d["episode_id"],
        decision_index=int(d["decision_index"]), now_s=_canon(d["now_s"]),
        info_level=d["info_level"], control_scope=d["control_scope"],
        profile_assumed=bool(d["profile_assumed"]), features=_fv_from(d["features"]),
        vessels=tuple(_vessel_from(v) for v in d["vessels"]),
        lane_graph=_lane_from(d["lane_graph"]))


def _obs_to(o: LocalObservation) -> dict:
    return {"schema_version": o.schema_version, "crane_id": o.crane_id,
            "now_s": _canon(o.now_s), "features": _fv_to(o.features),
            "candidates": _cset_to(o.candidates)}


def _obs_from(d: dict) -> LocalObservation:
    return LocalObservation(
        schema_version=d["schema_version"], crane_id=d["crane_id"],
        now_s=_canon(d["now_s"]), features=_fv_from(d["features"]),
        candidates=_cset_from(d["candidates"]))


def _assign_to(a: Assignment) -> dict:
    return {"crane_id": a.crane_id,
            "candidate_id": None if a.candidate_id is None else int(a.candidate_id),
            "kind": a.kind.value, "resolved_by": a.resolved_by}


def _assign_from(d: dict) -> Assignment:
    cid = d["candidate_id"]
    return Assignment(crane_id=d["crane_id"],
                      candidate_id=None if cid is None else int(cid),
                      kind=CandidateKind(d["kind"]), resolved_by=d["resolved_by"])


def _joint_to(j: JointAction) -> dict:
    return {"schema_version": j.schema_version, "now_s": _canon(j.now_s),
            "assignments": [_assign_to(a) for a in j.assignments]}


def _joint_from(d: dict) -> JointAction:
    return JointAction(schema_version=d["schema_version"], now_s=_canon(d["now_s"]),
                       assignments=tuple(_assign_from(a) for a in d["assignments"]))


# -------------------------------------------------------------------- cost
def _cost_to(c: CostBreakdown) -> dict:
    return {"schema_version": c.schema_version,
            "interval_start_s": _canon(c.interval_start_s),
            "interval_end_s": _canon(c.interval_end_s),
            "raw": {k: _canon(v) for k, v in c.raw.items()},
            "scale": {k: _canon(v) for k, v in c.scale.items()},
            "weight": {k: _canon(v) for k, v in c.weight.items()},
            "lambda_vessel": _canon(c.lambda_vessel),
            "total_normalized": _canon(c.total_normalized),
            "reward": _canon(c.reward), "assumed": bool(c.assumed)}


def _cost_from(d: dict) -> CostBreakdown:
    return CostBreakdown(
        schema_version=d["schema_version"],
        interval_start_s=_canon(d["interval_start_s"]),
        interval_end_s=_canon(d["interval_end_s"]),
        raw={k: _canon(v) for k, v in d["raw"].items()},
        scale={k: _canon(v) for k, v in d["scale"].items()},
        weight={k: _canon(v) for k, v in d["weight"].items()},
        lambda_vessel=_canon(d["lambda_vessel"]),
        total_normalized=_canon(d["total_normalized"]),
        reward=_canon(d["reward"]), assumed=bool(d["assumed"]))


# -------------------------------------------------------------------- audit
def _audit_to(a: TransitionAudit) -> dict:
    return {"built_at_now_s": _canon(a.built_at_now_s), "info_level": a.info_level,
            "ablation_off": list(a.ablation_off), "missing_fields": list(a.missing_fields),
            "assumed_fields": list(a.assumed_fields),
            "forbidden_touched": list(a.forbidden_touched),
            "event_stream_hash": a.event_stream_hash}


def _audit_from(d: dict) -> TransitionAudit:
    return TransitionAudit(
        built_at_now_s=_canon(d["built_at_now_s"]), info_level=d["info_level"],
        ablation_off=tuple(d["ablation_off"]), missing_fields=tuple(d["missing_fields"]),
        assumed_fields=tuple(d["assumed_fields"]),
        forbidden_touched=tuple(d["forbidden_touched"]),
        event_stream_hash=d["event_stream_hash"])


# --------------------------------------------------------------- transition
def to_dict(rec: TransitionRecord) -> dict:
    return {
        "schema_version": rec.schema_version, "episode_id": rec.episode_id,
        "decision_index": int(rec.decision_index), "dt_s": _canon(rec.dt_s),
        "state": _gs_to(rec.state),
        "observations": [_obs_to(o) for o in rec.observations],
        "joint_action": _joint_to(rec.joint_action), "cost": _cost_to(rec.cost),
        "next_state": None if rec.next_state is None else _gs_to(rec.next_state),
        "next_observations": [_obs_to(o) for o in rec.next_observations],
        "terminal": bool(rec.terminal), "audit": _audit_to(rec.audit),
    }


def from_dict(d: dict) -> TransitionRecord:
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            "SCHEMA_MISMATCH",
            f"계약 버전 불일치: {d.get('schema_version')} != {SCHEMA_VERSION}")
    ns = d["next_state"]
    return TransitionRecord(
        schema_version=d["schema_version"], episode_id=d["episode_id"],
        decision_index=int(d["decision_index"]), dt_s=_canon(d["dt_s"]),
        state=_gs_from(d["state"]),
        observations=tuple(_obs_from(o) for o in d["observations"]),
        joint_action=_joint_from(d["joint_action"]), cost=_cost_from(d["cost"]),
        next_state=None if ns is None else _gs_from(ns),
        next_observations=tuple(_obs_from(o) for o in d["next_observations"]),
        terminal=bool(d["terminal"]), audit=_audit_from(d["audit"]))


def dumps(rec: TransitionRecord) -> str:
    return json.dumps(to_dict(rec), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def loads(s: str) -> TransitionRecord:
    return from_dict(json.loads(s))
