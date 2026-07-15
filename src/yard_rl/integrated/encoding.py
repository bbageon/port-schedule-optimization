"""계약 객체 → 학습 텐서 인코딩 (YR-039 §2). torch 미의존 (list 기반).

FeatureVector 3채널 계약: 그룹별 `x = [value ⊙ known ‖ known]` — 결측은
value=0·known=0 으로 이미 중화돼 있고, known 지시자를 함께 주어 망이 결측을
구분하게 한다. 차원은 SCHEMA_VERSION 의 fv.names 길이에서 유도.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contract import CandidateKind, GlobalState, LocalObservation


def fv_to_vec(fv) -> list[float]:
    vals = [v * (1.0 if k else 0.0) for v, k in zip(fv.value, fv.known)]
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
    # 학습·backup 행동집합 (리뷰 HIGH): WAIT 는 resolver 가 pair 에서 배제해
    # 절대 회귀 표적을 받지 않음 → backup argmin 에 포함하면 표류값이 표적을
    # 하향 오염. actionable = selectable ∧ (kind != WAIT).
    actionable: tuple[bool, ...]
    candidate_ids: tuple[int, ...]

    @property
    def k_max(self) -> int:
        return len(self.cand)


def encode_observation(state: GlobalState, ob: LocalObservation) -> DecisionEncoding:
    cs = ob.candidates
    selectable = tuple(bool(p and f) for p, f in zip(cs.pad_mask, cs.feasible_mask))
    if not any(selectable):
        raise ValueError(f"{ob.crane_id}: 선택 가능 후보 없음 — 결정 계약 위반")
    actionable = tuple(s and c.kind != CandidateKind.WAIT
                       for s, c in zip(selectable, cs.items))
    return DecisionEncoding(
        crane_id=ob.crane_id,
        g=tuple(fv_to_vec(state.features)),
        yc=tuple(fv_to_vec(ob.features)),
        queue=tuple(fv_to_vec(cs.queue_summary)),
        cand=tuple(tuple(fv_to_vec(c.features)) for c in cs.items),
        selectable=selectable,
        actionable=actionable,
        candidate_ids=tuple(c.candidate_id for c in cs.items))


def encoding_dims(enc: DecisionEncoding) -> tuple[int, int, int, int]:
    """(Fg2, Fy2, Fq2, Fc2) — 망 생성 시 스키마에서 유도 (하드코딩 금지)."""
    return len(enc.g), len(enc.yc), len(enc.queue), len(enc.cand[0])
