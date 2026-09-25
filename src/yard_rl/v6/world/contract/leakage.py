"""정보누출 게이팅 — info_filter.py 의 계약 일반화 (YR-035).

envs/info_filter.py 는 외부트럭 Job 단위 가시성을 판정한다. 여기서는 그 규칙을
**필드 단위**(FieldSpec.tok × 실험 정보수준 × 실현시점)로 일반화한다.

두 방어선:
1. 스키마 부재: NEVER·GROUND_TRUTH 필드는 `_SPECS` 에 아예 없다 (물리적 차단).
2. field_visible / assert_no_forbidden: 값 생성·검증 단계에서 미실현·상한초과·금지원천을 재차단.
"""
from __future__ import annotations

from ..domain.enums import InformationLevel
from .schema import FieldSource, FieldSpec, TimeOfKnowledge


class LeakageError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


_A = TimeOfKnowledge
# 실험 정보수준별 허용 TOK 상한 (info_filter.py 의 BLOCK_ARRIVAL⊂GATE_IN⊂PRE_ADVICE 계승).
# 야드 내부(본선·이송·레인) PLANNED/ALWAYS 정보는 전 레벨에서 가시 — info_filter.is_visible
# 의 non-external 분기(release 이후 상시 공개)와 정합.
_LEVEL_CEILING: dict[InformationLevel, frozenset[TimeOfKnowledge]] = {
    InformationLevel.BLOCK_ARRIVAL: frozenset({_A.ALWAYS, _A.PLANNED, _A.BLOCK_ARRIVAL}),
    InformationLevel.GATE_IN: frozenset({_A.ALWAYS, _A.PLANNED, _A.BLOCK_ARRIVAL, _A.GATE_IN}),
    InformationLevel.PRE_ADVICE: frozenset(
        {_A.ALWAYS, _A.PLANNED, _A.BLOCK_ARRIVAL, _A.GATE_IN, _A.PRE_ADVICE}),
}


def visible_toks(info_level: InformationLevel) -> frozenset[TimeOfKnowledge]:
    try:
        return _LEVEL_CEILING[info_level]
    except KeyError:
        raise ValueError(f"미지원 정보수준 {info_level}")


def field_visible(spec: FieldSpec, *, now: float, realized_at: float | None,
                  info_level: InformationLevel) -> bool:
    """이 필드 값을 지금(now) 정책이 알 수 있는가.

    - NEVER·GROUND_TRUTH: 항상 불가 (계약 진입 금지 필드).
    - 실험 정보수준 상한(ceiling)을 넘는 source: 불가 (예: BLOCK_ARRIVAL 레벨의 ETA).
    - realized_at 이 미래(now 초과): 아직 실현되지 않은 값 (미도착 트럭의 누적대기 등).
    """
    if spec.tok == TimeOfKnowledge.NEVER or spec.source == FieldSource.GROUND_TRUTH:
        return False
    if spec.tok not in visible_toks(info_level):
        return False
    if realized_at is not None and realized_at > now:
        return False
    return True


def assert_no_forbidden(names: tuple[str, ...], known: tuple[bool, ...],
                        specs: tuple[FieldSpec, ...], *, where: str = "") -> None:
    """FeatureVector 방어선: NEVER·GROUND_TRUTH 채널이 known=1 이면 예외.

    `_SPECS` 는 이런 필드를 포함하지 않으므로 정상 경로에서는 절대 발화하지 않지만,
    스키마에 실수로 금지필드가 추가돼도 값이 노출되면 즉시 잡힌다.
    """
    for nm, kn, sp in zip(names, known, specs):
        if not kn:
            continue
        if sp.tok == TimeOfKnowledge.NEVER or sp.source == FieldSource.GROUND_TRUTH:
            raise LeakageError(
                "FORBIDDEN_FIELD",
                f"{where}{nm}: source={sp.source.value} tok={sp.tok.value} 는 known=1 불가")
