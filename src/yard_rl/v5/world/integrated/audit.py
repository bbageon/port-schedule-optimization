"""공동배정 감사 로그 — 계약 밖 side-channel (YR-037, 최종전략 §13.3).

계약 TransitionAudit(feature provenance 전용)에 넣지 않는다 (SCHEMA_VERSION bump·golden 전면
재생성 회피). 엔진 resolution_log 로 노출 — 운영자 근거·테스트·설명용. 결정론적.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..contract.schema import CandidateKind


@dataclass(frozen=True)
class CandidateVerdict:
    crane_id: str
    candidate_id: int
    token: str | None
    kind: CandidateKind
    feasible: bool
    reason: str | None            # == CandidateSet.mask_reason (단일 소스)
    mandatory: bool


@dataclass(frozen=True)
class CraneResolution:
    crane_id: str
    action: CandidateKind
    chosen_candidate_id: int | None   # None ⟺ WAIT
    chosen_token: str | None
    yield_reason: str | None          # NO_FEASIBLE | LOST_CONTENTION
    tiebreak_key: tuple               # 재현·설명
    rejected: tuple                   # CandidateVerdict 정렬


@dataclass(frozen=True)
class JointResolution:
    now_s: float
    crane_ids: tuple[str, ...]
    resolutions: tuple                # CraneResolution, crane_id 정렬
    mandatory_deferred: tuple[str, ...]   # 미수용 mandatory token (정렬)
    contested: tuple                  # (token, (경쟁 크레인들,)) 정렬

    def digest(self) -> str:
        parts = [f"{round(self.now_s, 6)}"]
        for r in self.resolutions:
            parts.append(f"{r.crane_id}:{r.action.value}:{r.chosen_token or '-'}")
        parts.append("md=" + ",".join(self.mandatory_deferred))
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def resolution_stream_hash(log) -> str:
    blob = "|".join(r.digest() for r in log)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
