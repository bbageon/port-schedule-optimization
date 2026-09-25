"""이송장비 fleet (YT/AGV/SC) — 최소 큐 모델 (YR-036, 최종전략 §7.8·15).

per-box 신원 추적 없이 집계 카운터만(대수·대기) → 결정론. 본선↔야드 이송이 밀리면
transfer_wait 가 창발하고, 버퍼가 막히면 STS 가 대기(sts_wait, vessel.py 소유)한다.
미확보 분포(n_units·move_time)는 assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_EPS = 1e-9


@dataclass
class TransferFleet:
    fleet_id: str
    kind: str
    n_units: int
    move_time_s: float
    busy_until: list[float] = field(default_factory=list)     # len == n_units
    pending: list[tuple[float, str]] = field(default_factory=list)  # (요청시각, vessel_id) FIFO
    transfer_wait_accum_s: float = 0.0

    def __post_init__(self):
        if not self.busy_until:
            self.busy_until = [0.0] * self.n_units

    def _free_index(self, now: float) -> int | None:
        for i in range(self.n_units):
            if self.busy_until[i] <= now + _EPS:
                return i
        return None

    def request(self, now: float, vessel_id: str) -> float | None:
        """이송 요청. 유닛 여유 시 배차하고 도착시각 반환, 아니면 pending 큐."""
        i = self._free_index(now)
        if i is None:
            self.pending.append((now, vessel_id))
            return None
        self.busy_until[i] = now + self.move_time_s
        return now + self.move_time_s

    def dispatch_pending(self, now: float) -> tuple[float, str] | None:
        """유닛이 free 된 시점에 대기요청 재배차 — (도착시각, vessel_id) 또는 None."""
        if not self.pending:
            return None
        i = self._free_index(now)
        if i is None:
            return None
        _, vid = self.pending.pop(0)
        self.busy_until[i] = now + self.move_time_s
        return now + self.move_time_s, vid

    def waiting_count(self) -> int:
        return len(self.pending)

    def integrate(self, t0: float, t1: float) -> None:
        dt = t1 - t0
        if dt > 0:
            self.transfer_wait_accum_s += len(self.pending) * dt
