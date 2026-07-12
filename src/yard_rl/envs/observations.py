"""집계 bucket 상태 인코더 (Tabular PoC) — 구현계획 02 §5.1.

bucket 경계는 임의값이 아니라 train 데이터(FIFO Baseline 관측)의 분위수로
결정하고 이후 고정한다. 공개된(visible) 후보만 사용 — 미래정보 누출 금지.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.models import CraneState, Job, TerminalProfile
from ..sim.stack import YardStacks
from .rules import blockers_of, reach_s

StateKey = tuple[int, int, int, int, int, int, int, int, int]

# 미래도착 feature 의 고정 경계 (assumed — 전 실험수준 공통, 분위수 fit 대상 아님)
_IMMINENT_GAP_BOUNDS = [300.0, 900.0, 1800.0]


def _bucket(x: float, bounds: list[float]) -> int:
    for i, b in enumerate(bounds):
        if x <= b:
            return i
    return len(bounds)


@dataclass
class BucketConfig:
    """분위수 기반 경계. 기본값은 train fit 전 임시(assumed) — fit 후 고정."""

    queue_len: list[float] = field(default_factory=lambda: [1, 3, 6])
    oldest_wait_s: list[float] = field(default_factory=lambda: [300.0, 900.0, 1800.0])
    reach_s: list[float] = field(default_factory=lambda: [60.0, 150.0, 300.0])
    fitted: bool = False

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BucketConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def fit(cls, queue_lens: list[float], oldest_waits: list[float],
            reaches: list[float]) -> "BucketConfig":
        def quartiles(xs: list[float]) -> list[float]:
            xs = sorted(xs)
            if not xs:
                return [0.0, 0.0, 0.0]
            return [xs[int(len(xs) * q)] for q in (0.25, 0.5, 0.75)]

        return cls(queue_len=quartiles(queue_lens), oldest_wait_s=quartiles(oldest_waits),
                   reach_s=quartiles(reaches), fitted=True)


class ObservationEncoder:
    def __init__(self, profile: TerminalProfile, buckets: BucketConfig):
        self.profile = profile
        self.buckets = buckets

    def raw_features(self, candidates: list[Job], crane: CraneState,
                     stacks: YardStacks, now: float, horizon_s: float,
                     future_arrivals: list[float] | None = None) -> dict:
        """future_arrivals: 아직 도착하지 않았지만 공개된 작업의 (수준별) 도착예상.
        Exp-1 은 항상 빈 리스트 — 해당 feature 가 상수로 고정된다."""
        ext_waiting = [j for j in candidates if j.is_external_truck]
        oldest = max((now - j.actual_block_arrival for j in ext_waiting), default=0.0)
        nearest = min((reach_s(j, crane, stacks, self.profile) for j in candidates),
                      default=0.0)
        min_blk = min((blockers_of(j, stacks) for j in candidates), default=0)
        vessels = [j for j in candidates if j.is_vessel_linked and j.deadline is not None]
        if not vessels:
            vessel_urgency = 0
        else:
            slack = min(j.deadline - now for j in vessels)
            vessel_urgency = 2 if slack < self.profile.long_wait_sla_s else 1
        fut = future_arrivals or []
        horizon = self.profile.decision_horizon_s
        imminent = [t for t in fut if t - now <= horizon]
        next_gap = min((max(0.0, t - now) for t in fut), default=float("inf"))
        return {
            "time_frac": min(0.999, max(0.0, now / horizon_s)),
            "crane_bay": crane.position_bay,
            "queue_len": float(len(ext_waiting)),
            "oldest_wait_s": oldest,
            "nearest_reach_s": nearest,
            "min_blockers": min_blk,
            "vessel_urgency": vessel_urgency,
            "imminent_count": len(imminent),
            "next_arrival_gap_s": next_gap,
        }

    def encode(self, candidates: list[Job], crane: CraneState, stacks: YardStacks,
               now: float, horizon_s: float,
               future_arrivals: list[float] | None = None) -> StateKey:
        f = self.raw_features(candidates, crane, stacks, now, horizon_s, future_arrivals)
        geom = self.profile.block
        zone = min(3, int((f["crane_bay"] - 1) / max(1, geom.bay_count) * 4))
        return (
            int(f["time_frac"] * 4),                              # 0..3 shift 사분면
            zone,                                                 # 0..3 크레인 위치
            _bucket(f["queue_len"], self.buckets.queue_len),      # 0..3
            _bucket(f["oldest_wait_s"], self.buckets.oldest_wait_s),
            _bucket(f["nearest_reach_s"], self.buckets.reach_s),
            min(3, f["min_blockers"]),                            # 0/1/2/3+
            f["vessel_urgency"],                                  # 0..2
            min(3, f["imminent_count"]),                          # 0/1/2/3+ (Exp-2/3)
            _bucket(f["next_arrival_gap_s"], _IMMINENT_GAP_BOUNDS),  # 0..3 (없음=3)
        )
