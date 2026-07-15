"""계약 수준 검증 전수 — 스키마·누출·단위·결측·vessel·joint·cost (YR-035).

검사 순서 = task 계획 순서(스키마→누출→단위→결측→구조). 오류는 조용히 통과시키지 않고
`ValidationError`/`LeakageError` 로 던진다 (domain/validators.py 관습 재사용).
"""
from __future__ import annotations

import math

from ..domain.enums import InformationLevel
from ..domain.validators import ValidationError
from .candidate import CandidateSet
from .cost import CostBreakdown
from .leakage import LeakageError, assert_no_forbidden, visible_toks
from .schema import (COST_TERMS, SCHEMA, SCHEMA_VERSION, CandidateKind,
                     FieldSource, TimeOfKnowledge, Unit)
from .transition import TransitionRecord
from .vectors import FeatureVector
from .vessel import VesselUrgency, VesselUrgencyMode, _ASSUMED_BASES

_TOL = 1e-6


# ---------------------------------------------------- FeatureVector 순회
def _iter_vectors(rec: TransitionRecord):
    """(path, FeatureVector, is_pad) 스트림 — state·vessels·obs(yc/queue/candidate)·next_*.

    is_pad=True 인 후보 슬롯은 구조적 패딩(전 채널 known=0)이므로 필드 단위 검사(단위·결측·
    누출)를 건너뛴다 — 필수 필드가 결측으로 오탐되는 것을 막고, all-known=0 은 validate_candidates
    가 별도로 강제한다.
    """
    def from_state(tag, st):
        if st is None:
            return
        yield f"{tag}.features", st.features, False
        for i, v in enumerate(st.vessels):
            yield f"{tag}.vessels[{i}].features", v.features, False

    def from_obs(tag, obs):
        for o in obs:
            yield f"{tag}[{o.crane_id}].features", o.features, False
            yield f"{tag}[{o.crane_id}].queue_summary", o.candidates.queue_summary, False
            for c, pad in zip(o.candidates.items, o.candidates.pad_mask):
                yield f"{tag}[{o.crane_id}].cand[{c.candidate_id}]", c.features, not pad

    yield from from_state("state", rec.state)
    yield from from_obs("obs", rec.observations)
    yield from from_state("next_state", rec.next_state)
    yield from from_obs("next_obs", rec.next_observations)


# ------------------------------------------------------------- 개별 검사
def validate_schema_fv(path: str, fv: FeatureVector) -> None:
    if fv.schema_version != SCHEMA_VERSION:
        raise ValidationError("SCHEMA_MISMATCH", f"{path}: {fv.schema_version}")
    want = SCHEMA.names(fv.group)
    if fv.names != want:
        raise ValidationError("FIELD_ORDER", f"{path}: names 순서/집합 불일치")
    n = len(want)
    if not (len(fv.value) == len(fv.known) == len(fv.assumed) == n):
        raise ValidationError("LENGTH_MISMATCH", f"{path}: 채널 길이 불일치")


def validate_missing_fv(path: str, fv: FeatureVector) -> None:
    for nm, val, kn in zip(fv.names, fv.value, fv.known):
        sp = SCHEMA.spec(fv.group, nm)
        if not kn:
            if val != 0.0:
                raise ValidationError("STALE_MISSING", f"{path}.{nm}: known=0 인데 value={val}")
            if not sp.nullable:
                raise ValidationError("REQUIRED_MISSING", f"{path}.{nm}: 필수 필드 결측")


def validate_assumed_fv(path: str, fv: FeatureVector) -> None:
    for nm, kn, asm in zip(fv.names, fv.known, fv.assumed):
        if not asm:
            continue
        sp = SCHEMA.spec(fv.group, nm)
        if not kn:
            raise ValidationError("ASSUMED_NO_BASIS", f"{path}.{nm}: assumed=1 인데 known=0")
        if sp.assumed_default is None:
            raise ValidationError("ASSUMED_NO_BASIS",
                                  f"{path}.{nm}: assumed_default 없는 필드의 assumed=1")


def validate_units_fv(path: str, fv: FeatureVector) -> None:
    for nm, val in zip(fv.names, fv.value):
        sp = SCHEMA.spec(fv.group, nm)
        if sp.unit not in Unit:
            raise ValidationError("UNIT_RANGE", f"{path}.{nm}: 미지원 단위")
        if sp.unit == Unit.RATIO_0_1 and not (-_TOL <= val <= 1 + _TOL):
            raise ValidationError("UNIT_RANGE", f"{path}.{nm}: ratio {val} ∉ [0,1]")
        if sp.unit == Unit.BOOL01 and min(abs(val), abs(val - 1.0)) > _TOL:
            raise ValidationError("UNIT_RANGE", f"{path}.{nm}: bool {val} ∉ {{0,1}}")
        if sp.clip_lo is not None and val < sp.clip_lo - _TOL:
            raise ValidationError("UNIT_RANGE", f"{path}.{nm}: {val} < clip_lo {sp.clip_lo}")
        if sp.clip_hi is not None and val > sp.clip_hi + _TOL:
            raise ValidationError("UNIT_RANGE", f"{path}.{nm}: {val} > clip_hi {sp.clip_hi}")


def validate_leakage_fv(path: str, fv: FeatureVector, level: InformationLevel) -> None:
    """저장 레코드 불변식: known=1 채널의 TOK 가 실험 상한 안이고 금지원천이 아니어야 함."""
    ceiling = visible_toks(level)
    for nm, kn in zip(fv.names, fv.known):
        if not kn:
            continue
        sp = SCHEMA.spec(fv.group, nm)
        if sp.tok == TimeOfKnowledge.NEVER or sp.source == FieldSource.GROUND_TRUTH:
            raise LeakageError("LEAKAGE", f"{path}.{nm}: 금지원천 known=1")
        if sp.tok not in ceiling:
            raise LeakageError(
                "LEAKAGE",
                f"{path}.{nm}: tok={sp.tok.value} 가 정보수준 {level.value} 상한 초과")


# --------------------------------------------------------------- vessel
def validate_vessel(path: str, v: VesselUrgency) -> None:
    f = v.features
    known = {nm: kn for nm, kn in zip(f.names, f.known)}
    risk_known = known.get("risk", False)
    slack_known = known.get("slack_s", False)
    delay_known = known.get("expected_delay_s", False)
    symptom_known = known.get("delay_symptom_score", False)
    if v.mode == VesselUrgencyMode.SYMPTOM:
        if v.completion_basis is not None:
            raise ValidationError("VESSEL_MODE", f"{path}: SYMPTOM 인데 완료근거 존재")
        if risk_known or slack_known or delay_known:
            raise ValidationError("VESSEL_MODE", f"{path}: SYMPTOM 인데 risk/slack/delay known=1")
        if not symptom_known:
            raise ValidationError("VESSEL_MODE", f"{path}: SYMPTOM 인데 징후점수 known=0")
    elif v.mode == VesselUrgencyMode.RISK:
        if v.completion_basis is None:
            raise ValidationError("VESSEL_MODE", f"{path}: RISK 인데 완료근거 없음")
        if not risk_known:
            raise ValidationError("VESSEL_MODE", f"{path}: RISK 인데 risk known=0")
        if symptom_known:  # §7.10 위험도·징후는 상호배타 (SYMPTOM 분기와 대칭)
            raise ValidationError("VESSEL_MODE", f"{path}: RISK 인데 징후점수 known=1 (혼용)")
        if v.assumed != (v.completion_basis in _ASSUMED_BASES):
            raise ValidationError("VESSEL_MODE", f"{path}: assumed 플래그가 근거와 불일치")
    else:  # NONE
        if risk_known or slack_known or symptom_known or delay_known:
            raise ValidationError("VESSEL_MODE", f"{path}: NONE 인데 위험/징후 채널 known=1")


# ------------------------------------------------------------ candidate
def validate_candidates(cs: CandidateSet) -> None:
    n = len(cs.items)
    if not (len(cs.pad_mask) == len(cs.feasible_mask) == len(cs.mask_reason) == n):
        raise ValidationError("LENGTH_MISMATCH", f"{cs.crane_id}: 후보 마스크 길이 불일치")
    for i, (c, pad, feas, reason) in enumerate(
            zip(cs.items, cs.pad_mask, cs.feasible_mask, cs.mask_reason)):
        if c.candidate_id != i:
            raise ValidationError("CANDIDATE_ID", f"{cs.crane_id}[{i}]: id={c.candidate_id}")
        if feas and not pad:
            raise ValidationError("INFEASIBLE_PAD", f"{cs.crane_id}[{i}]: feasible⊄pad")
        if feas and reason is not None:
            raise ValidationError("MASK_REASON", f"{cs.crane_id}[{i}]: feasible 인데 사유 존재")
        if (not feas) and reason is None:
            raise ValidationError("MASK_REASON", f"{cs.crane_id}[{i}]: 불가인데 사유 없음")
        if not pad and (any(c.features.known) or any(v != 0.0 for v in c.features.value)):
            raise ValidationError("PAD_NONZERO", f"{cs.crane_id}[{i}]: 패딩 자리 known=1 또는 value≠0")


# ---------------------------------------------------------------- joint
def validate_joint(rec: TransitionRecord) -> None:
    by_crane = {o.crane_id: o.candidates for o in rec.observations}
    seen_crane: set[str] = set()
    seen_token: dict[str, str] = {}
    seen_lane: dict[str, str] = {}
    for a in rec.joint_action.assignments:
        # §8.6 Joint Action = [a_YC1..a_YCn]: 크레인당 정확히 1개 동작
        if a.crane_id in seen_crane:
            raise ValidationError("DUP_CRANE", f"{a.crane_id}: 동일 크레인 이중배정")
        seen_crane.add(a.crane_id)
        if a.candidate_id is None:
            # None ⟺ WAIT/no-op (양보) — 중복·레인 검사만 제외하되 불변식은 강제
            if a.kind != CandidateKind.WAIT:
                raise ValidationError("KIND_MISMATCH",
                                      f"{a.crane_id}: candidate_id=None 인데 kind={a.kind.value}")
            if a.crane_id not in by_crane:
                raise ValidationError("BAD_ASSIGN", f"{a.crane_id}: 미존재 크레인")
            continue
        cs = by_crane.get(a.crane_id)
        if cs is None or not (0 <= a.candidate_id < len(cs.items)):
            raise ValidationError("BAD_ASSIGN", f"{a.crane_id}: candidate_id={a.candidate_id}")
        cand = cs.items[a.candidate_id]
        if not cs.feasible_mask[a.candidate_id]:
            raise ValidationError("INFEASIBLE_SELECTION",
                                  f"{a.crane_id}: 불가 후보 {a.candidate_id} 선택")
        if a.kind != cand.kind:
            raise ValidationError("KIND_MISMATCH", f"{a.crane_id}: kind 불일치")
        if cand.eligible_crane_ids and a.crane_id not in cand.eligible_crane_ids:
            raise ValidationError("INELIGIBLE", f"{a.crane_id}: 자격 밖 후보 선택")
        if cand.resolver_token is not None:
            other = seen_token.get(cand.resolver_token)
            if other is not None:
                raise ValidationError("DUP_JOB",
                                      f"{cand.resolver_token}: {other}·{a.crane_id} 중복배정")
            seen_token[cand.resolver_token] = a.crane_id
        if cand.lane_id is not None:
            other = seen_lane.get(cand.lane_id)
            if other is not None:
                raise ValidationError("LANE_CONFLICT",
                                      f"lane {cand.lane_id}: {other}·{a.crane_id} 동시점유")
            seen_lane[cand.lane_id] = a.crane_id


# ----------------------------------------------------------------- cost
def validate_cost(c: CostBreakdown) -> None:
    terms = set(COST_TERMS)
    for name, d in (("raw", c.raw), ("scale", c.scale), ("weight", c.weight)):
        if set(d) != terms:
            raise ValidationError("COST_TERMS", f"{name} 키가 13항과 불일치")
    # 유한성은 항등식보다 먼저 — inf/nan 이면 abs(inf-inf)=nan 으로 항등식을 조용히 통과함.
    # 세 dict 를 개별 순회 (동일 COST_TERMS 키라 {**merge} 는 raw/scale 를 weight 로 덮음).
    for name, d in (("raw", c.raw), ("scale", c.scale), ("weight", c.weight)):
        for k, v in d.items():
            if not math.isfinite(v):
                raise ValidationError("COST_NONFINITE", f"{name}.{k}: inf/nan")
    for nm, v in (("lambda_vessel", c.lambda_vessel),
                  ("total_normalized", c.total_normalized), ("reward", c.reward)):
        if not math.isfinite(v):
            raise ValidationError("COST_NONFINITE", f"{nm}: inf/nan")
    if any(s <= 0 for s in c.scale.values()):
        raise ValidationError("ZERO_SCALE", "scale 은 전부 양수여야 함")
    total = sum(c.contributions().values())
    if abs(total - c.total_normalized) > _TOL:
        raise ValidationError("COST_IDENTITY", f"Σcontrib {total} != total {c.total_normalized}")
    if abs(c.reward + c.total_normalized) > _TOL:
        raise ValidationError("COST_IDENTITY", f"reward {c.reward} != -total")


# ------------------------------------------------------------- 전체
def validate_all(rec: TransitionRecord) -> None:
    if rec.schema_version != SCHEMA_VERSION:
        raise ValidationError("SCHEMA_MISMATCH", rec.schema_version)
    level = InformationLevel(rec.state.info_level)
    specs_cache = {g: SCHEMA.group_specs(g) for g in SCHEMA.groups()}
    for path, fv, is_pad in _iter_vectors(rec):
        validate_schema_fv(path, fv)
        assert_no_forbidden(fv.names, fv.known, specs_cache[fv.group], where=f"{path}.")
        if is_pad:
            continue   # 패딩 슬롯: all-known=0 은 validate_candidates 가 강제
        validate_units_fv(path, fv)
        validate_missing_fv(path, fv)
        validate_assumed_fv(path, fv)
        validate_leakage_fv(path, fv, level)
    for st, tag in ((rec.state, "state"), (rec.next_state, "next_state")):
        if st is not None:
            for i, v in enumerate(st.vessels):
                validate_vessel(f"{tag}.vessels[{i}]", v)
    for o in (*rec.observations, *rec.next_observations):
        validate_candidates(o.candidates)
    validate_joint(rec)
    validate_cost(rec.cost)
    if rec.audit.forbidden_touched:
        raise LeakageError("FORBIDDEN_TOUCHED", str(rec.audit.forbidden_touched))
