"""FeatureVector — 스키마에 정렬된 value/known/assumed 3채널 (YR-035).

한 group 의 모든 필드를 `SCHEMA.names(group)` 순서로 담는다. 결측·미래정보·ablation-off 는
inf/nan 이 아니라 **known=0·value=0.0** 으로 중화한다 (JSON 왕복 안전 + 텐서 진입 시 마스크 곱).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.enums import InformationLevel
from .leakage import field_visible
from .schema import FLOAT_DECIMALS, SCHEMA, FieldSpec


def _canon(x: float) -> float:
    """직렬화 idempotent 를 위한 float 정규화. inf/nan 은 결측이지 값이 아니다."""
    xf = float(x)
    if not math.isfinite(xf):
        raise ValueError("inf/nan 금지 — 결측은 known=0 으로 표현한다")
    r = round(xf, FLOAT_DECIMALS)
    return 0.0 if r == 0.0 else r   # -0.0 정규화 (직렬화 문자열 안정)


def _clip(x: float, spec: FieldSpec) -> float:
    if spec.clip_lo is not None and x < spec.clip_lo:
        x = spec.clip_lo
    if spec.clip_hi is not None and x > spec.clip_hi:
        x = spec.clip_hi
    return x


@dataclass(frozen=True)
class FeatureVector:
    schema_version: str
    group: str
    names: tuple[str, ...]          # == SCHEMA.names(group)
    value: tuple[float, ...]
    known: tuple[bool, ...]         # 미래정보·결측·ablation = False
    assumed: tuple[bool, ...]       # 가정·imputation = True

    def channel(self, name: str) -> tuple[float, bool, bool]:
        i = self.names.index(name)
        return self.value[i], self.known[i], self.assumed[i]

    def known_of(self, name: str) -> bool:
        return self.known[self.names.index(name)]

    def value_of(self, name: str) -> float:
        return self.value[self.names.index(name)]


def build_feature_vector(
    group: str,
    raw: dict[str, float | None],
    *,
    now: float,
    info_level: InformationLevel,
    realized_at: dict[str, float] | None = None,
    ablation_off: frozenset | set | tuple = (),
) -> FeatureVector:
    """raw{name→값|None} 를 SCHEMA(group) 순서로 정렬하고 TOK/source/ablation 게이팅.

    규칙 (렌즈 병합):
    1. field_visible=False (미래·상한초과·금지원천) → known=0·value=0.0.
    2. ablation 그룹 off → 강제 known=0·value=0.0.
    3. 가시 + raw 값 존재 → clip 후 known=1·assumed=0.
    4. 가시 + raw None + assumed_default 존재 → default·clip, known=1·assumed=1 (imputation).
    5. 가시 + raw None + default 없음 → known=0·value=0.0 (결측; nullable=False 면 validate 가 잡음).
    6. inf/nan → known=0·value=0.0 으로 중화 (직렬화 시 _canon 이 재차 차단).
    """
    realized_at = realized_at or {}
    off = set(ab.value if hasattr(ab, "value") else ab for ab in ablation_off)
    specs = SCHEMA.group_specs(group)
    names: list[str] = []
    values: list[float] = []
    known: list[bool] = []
    assumed: list[bool] = []
    for sp in specs:
        names.append(sp.name)
        visible = field_visible(sp, now=now, realized_at=realized_at.get(sp.name),
                                info_level=info_level)
        if sp.ablation.value in off:
            visible = False
        v, kn, asm = 0.0, False, False
        if visible:
            rv = raw.get(sp.name)
            if rv is not None and math.isfinite(float(rv)):
                v, kn, asm = _clip(float(rv), sp), True, False
            elif rv is None and sp.assumed_default is not None:
                v, kn, asm = _clip(float(sp.assumed_default), sp), True, True
            # else: 결측 → 0/False/False 유지
        values.append(round(v, FLOAT_DECIMALS))
        known.append(kn)
        assumed.append(asm)
    return FeatureVector(
        schema_version=SCHEMA.version, group=group, names=tuple(names),
        value=tuple(values), known=tuple(known), assumed=tuple(assumed))


def zero_feature_vector(group: str) -> FeatureVector:
    """패딩·no-op 자리용 전(全) known=0 벡터 (candidate padding 등)."""
    names = SCHEMA.names(group)
    n = len(names)
    return FeatureVector(schema_version=SCHEMA.version, group=group, names=names,
                         value=(0.0,) * n, known=(False,) * n, assumed=(False,) * n)
