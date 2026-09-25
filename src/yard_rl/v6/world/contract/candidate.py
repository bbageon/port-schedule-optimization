"""동적 후보와 마스크 계약 — 최종전략 §8 (YR-035).

Action 은 규칙번호가 아니라 각 YC 의 실행 가능한 작업(가변 후보) 선택이다. 계약은 후보를
저장·검증만 한다 (생성기·resolver 는 YR-037). 세 마스크를 분리한다:
- pad_mask     : 구조적 패딩 (배치 텐서화 대비, True=실후보)
- feasible_mask: §8.5 정책 실행가능 (feasible ⊆ pad)
- FeatureVector.known : 필드 결측 (마스크 3번째 층, vectors.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CandidateKind
from .vectors import FeatureVector, zero_feature_vector


@dataclass(frozen=True)
class Candidate:
    candidate_id: int               # 결정 내 안정 로컬 인덱스 (items 위치와 일치, Job ID 아님)
    kind: CandidateKind
    features: FeatureVector          # group="candidate" (net 입력)
    mandatory: bool = False          # SLA 임박 보존 — pruning 금지 (YR-037/YR-029)
    ref_job_id: str | None = None    # 감사 back-ref (SERVE/PRE_REHANDLE 만; net 입력 금지)
    resolver_token: str | None = None  # 익명 동일성 토큰 (§8.6 중복검사; 신원 암기 불가)
    eligible_crane_ids: tuple[str, ...] = ()   # §8.6 수행가능 YC (빈 tuple = 소유 YC 전용)
    lane_id: str | None = None       # §7.6 사용 레인 (joint 레인충돌 검사용)


@dataclass(frozen=True)
class CandidateSet:
    crane_id: str
    items: tuple[Candidate, ...]     # 가변 K (Top-K 아님, mandatory 보존)
    pad_mask: tuple[bool, ...]       # True=실후보, False=배치 패딩
    feasible_mask: tuple[bool, ...]  # §8.5 정책 mask, feasible ⊆ pad
    mask_reason: tuple[str | None, ...]  # feasible=False 사유코드 (feasible=True 면 None)
    queue_summary: FeatureVector = field(  # group="queue", permutation-invariant (YR-031-b)
        default_factory=lambda: zero_feature_vector("queue"))

    @property
    def real_items(self) -> tuple[Candidate, ...]:
        return tuple(c for c, p in zip(self.items, self.pad_mask) if p)


def padding_candidate(candidate_id: int) -> Candidate:
    """배치 고정 K 를 맞추기 위한 zero 패딩 후보 (pad_mask=False 와 함께 사용)."""
    return Candidate(candidate_id=candidate_id, kind=CandidateKind.WAIT,
                     features=zero_feature_vector("candidate"))
