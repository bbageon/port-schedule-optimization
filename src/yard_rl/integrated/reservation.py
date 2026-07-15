"""다중 YC 자원예약 — 4종 lock 단일 소유자 (YR-036, 최종전략 §8.6).

이중배정(token)·슬롯·corridor(비통과·안전거리)·레인을 여기서만 배타 관리한다. lane.py 는
혼잡 적분만. 예약=dispatch, 해제=완료 이벤트(부동시간 암묵해제 금지 — 비결정 방지).
두 겹 방어: candidates_for 가 1차로 걸러도 commit 의 reserve() 가 임의 joint 를 재검증한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..sim.constraints import ConstraintViolation


@dataclass(frozen=True)
class Corridor:
    """작업 중 크레인이 점유하는 gantry(bay) 축 구간 [lo, hi]."""

    lo: float
    hi: float

    def overlaps(self, other: "Corridor", gap: float) -> bool:
        return not (self.hi + gap <= other.lo or other.hi + gap <= self.lo)


@dataclass(frozen=True)
class Reservation:
    crane_id: str
    job_token: str | None
    corridor: Corridor
    slots: frozenset
    lane_id: str | None
    release_at: float


@dataclass
class ReservationTable:
    safety_gap_bay: float
    _by_crane: dict[str, Reservation] = field(default_factory=dict)
    _tokens: dict[str, str] = field(default_factory=dict)   # token -> crane_id

    def active(self) -> tuple[Reservation, ...]:
        return tuple(self._by_crane[c] for c in sorted(self._by_crane))

    def job_taken(self, token: str | None) -> str | None:
        return None if token is None else self._tokens.get(token)

    def reserved_slots(self) -> frozenset:
        out: set = set()
        for r in self._by_crane.values():
            out |= r.slots
        return frozenset(out)

    def lane_owner(self, lane_id: str | None) -> str | None:
        if lane_id is None:
            return None
        for cid in sorted(self._by_crane):
            if self._by_crane[cid].lane_id == lane_id:
                return cid
        return None

    def corridor_conflict(self, crane_id: str, corridor: Corridor) -> str | None:
        for cid in sorted(self._by_crane):
            if cid == crane_id:
                continue
            if self._by_crane[cid].corridor.overlaps(corridor, self.safety_gap_bay):
                return cid
        return None

    def reject_reason(self, r: Reservation) -> str | None:
        """5-lock 순차 판정 코드 (None=성공). 생성기·can_reserve·reserve 가 공유 —
        1·2·3차 방어가 동일 소스라 divergence 0 (YR-037)."""
        if r.crane_id in self._by_crane:
            return "DOUBLE_RESERVE"
        if self.job_taken(r.job_token) is not None:
            return "DUP_JOB"
        if self.lane_owner(r.lane_id) is not None:
            return "LANE_CONFLICT"
        if self.corridor_conflict(r.crane_id, r.corridor) is not None:
            return "CRANE_INTERFERENCE"
        if self.reserved_slots() & r.slots:
            return "SLOT_CONFLICT"
        return None

    def reserve(self, r: Reservation) -> None:
        """2차 방어선 — 임의 joint 입력도 검증 (DUP_JOB·LANE_CONFLICT·CRANE_INTERFERENCE)."""
        reason = self.reject_reason(r)
        if reason is not None:
            raise ConstraintViolation(reason, f"{r.crane_id}: {reason} (token={r.job_token} lane={r.lane_id})")
        self._by_crane[r.crane_id] = r
        if r.job_token is not None:
            self._tokens[r.job_token] = r.crane_id

    def can_reserve(self, r: Reservation) -> bool:
        """예약 성공 여부만 판정 (실제 예약 없이 — 순차 전파·dry_run 용)."""
        return self.reject_reason(r) is None

    def release(self, crane_id: str) -> None:
        r = self._by_crane.pop(crane_id, None)
        if r is not None and r.job_token is not None:
            self._tokens.pop(r.job_token, None)

    def orphan_count(self) -> int:
        return len(self._by_crane)
