"""계약 객체 → 학습 텐서 인코딩 (YR-039 §2). torch 미의존 (list 기반).

FeatureVector 3채널 계약: 그룹별 `x = [value ⊙ known ‖ known]` — 결측은
value=0·known=0 으로 이미 중화돼 있고, known 지시자를 함께 주어 망이 결측을
구분하게 한다. 차원은 SCHEMA_VERSION 의 fv.names 길이에서 유도.

YR-059 상태 정규화 (적용전략 §4): StateNorm 을 주면 value/norm_ref 로 나눠 O(1)
범위로 맞추고 ±clip 에서 자른다 — **scale-only** (중심이동 없음: 결측=0 규약 보존,
0/ref==0). running 통계 금지(골든 결정성) — 기준값은 스키마 assumed + fitted 동결
override 뿐이다. 저장 레코드(계약 값)는 불변 — 정규화는 학습 텐서에서만.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import SCHEMA, CandidateKind, GlobalState, LocalObservation


@dataclass(frozen=True)
class StateNorm:
    """필드별 동결 정규화 기준 — refs["group.name"] 이 스키마 assumed norm_ref 를 override.

    basis: "assumed"(스키마 초기값만) | "fitted_baseline_p90" 등 — provenance 문서화 의무.
    """

    refs: dict = field(default_factory=dict)
    clip: float = 5.0
    basis: str = "assumed"

    def ref_row(self, group: str) -> tuple[float, ...]:
        return tuple(float(self.refs.get(f"{group}.{sp.name}", sp.norm_ref))
                     for sp in SCHEMA.group_specs(group))


def fv_to_vec(fv, norm: StateNorm | None = None) -> list[float]:
    if norm is None:
        vals = [v * (1.0 if k else 0.0) for v, k in zip(fv.value, fv.known)]
    else:
        c = norm.clip
        vals = [max(-c, min(c, v / r)) * (1.0 if k else 0.0)
                for v, r, k in zip(fv.value, norm.ref_row(fv.group), fv.known)]
    return vals + [1.0 if k else 0.0 for k in fv.known]


@dataclass(frozen=True)
class DecisionEncoding:
    """크레인 1기의 결정 시점 인코딩 — candidate_id 순서는 CandidateSet.items 그대로."""

    crane_id: str
    g: tuple[float, ...]                 # global 2·Fg
    yc: tuple[float, ...]                # yc 2·Fy
    queue: tuple[float, ...]             # queue summary 2·Fq
    cand: tuple[tuple[float, ...], ...]  # [K_max][2·Fc]
    selectable: tuple[bool, ...]         # feasible ∧ 비패딩
    # 학습·backup 행동집합. YR-043: WAIT 를 실제 학습 행동으로 복구 — resolver 가 WAIT 를
    # pair 에 포함해 선택·회귀 표적을 받으므로 backup argmin 에서 배제할 이유가 사라졌다
    # (배제 근거였던 "표적 못 받는 표류값" 전제 소멸). actionable = selectable.
    actionable: tuple[bool, ...]
    candidate_ids: tuple[int, ...]
    wait_pos: int | None = None          # WAIT 후보의 행 위치 (replay 표본 매핑용, YR-043)

    @property
    def k_max(self) -> int:
        return len(self.cand)


def encode_observation(state: GlobalState, ob: LocalObservation,
                       norm: StateNorm | None = None) -> DecisionEncoding:
    cs = ob.candidates
    selectable = tuple(bool(p and f) for p, f in zip(cs.pad_mask, cs.feasible_mask))
    if not any(selectable):
        raise ValueError(f"{ob.crane_id}: 선택 가능 후보 없음 — 결정 계약 위반")
    actionable = selectable            # YR-043: WAIT 포함 (Hold/Yield 는 실제 행동)
    wait_pos = next((i for i, (s, c) in enumerate(zip(selectable, cs.items))
                     if s and c.kind == CandidateKind.WAIT), None)
    return DecisionEncoding(
        crane_id=ob.crane_id,
        g=tuple(fv_to_vec(state.features, norm)),
        yc=tuple(fv_to_vec(ob.features, norm)),
        queue=tuple(fv_to_vec(cs.queue_summary, norm)),
        cand=tuple(tuple(fv_to_vec(c.features, norm)) for c in cs.items),
        selectable=selectable,
        actionable=actionable,
        candidate_ids=tuple(c.candidate_id for c in cs.items),
        wait_pos=wait_pos)


def encoding_dims(enc: DecisionEncoding) -> tuple[int, int, int, int]:
    """(Fg2, Fy2, Fq2, Fc2) — 망 생성 시 스키마에서 유도 (하드코딩 금지)."""
    return len(enc.g), len(enc.yc), len(enc.queue), len(enc.cand[0])
