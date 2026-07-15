"""구간 비용 적분 — 13항 raw delta (YR-036, 최종전략 §10).

sim 은 raw 물리 delta 만 산출한다. scale/weight/λ 실수치는 assumed placeholder(YR-038 위임).
- truck_wait/long_wait: 엔진이 재사용 KpiTracker 의 정확 적분(queue_area·tail_area) 증분을 accrue.
- rate 항(sts_wait/transfer_wait/lane_cong/interference/imbalance): advance 에서 rate×dt.
- delta 항(crane_travel/empty_travel/rehandle/vessel_delay/depart_delay/resequence): 이벤트 임펄스.
각 accrual 이 정확히 한 구간에 귀속 → Σ cut()[k] == episode_raw()[k] (중복·누락 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.schema import COST_TERMS

# 피스와이즈 상수율로 시간적분하는 항 (엔진이 이벤트마다 set_rate)
RATE_TERMS: frozenset = frozenset(
    {"sts_wait", "transfer_wait", "lane_cong", "interference", "imbalance"})

# assumed placeholder (YR-038 이 costcfg 만 교체) — won_cost_v1 관습과 정합
ASSUMED_SCALE: dict[str, float] = {
    "truck_wait": 600.0, "long_wait": 1800.0, "crane_travel": 1000.0,
    "empty_travel": 1000.0, "rehandle": 5.0, "sts_wait": 600.0, "transfer_wait": 600.0,
    "vessel_delay": 600.0, "depart_delay": 600.0, "lane_cong": 100.0,
    "interference": 100.0, "resequence": 10.0, "imbalance": 1.0}
ASSUMED_WEIGHT: dict[str, float] = {t: 1.0 for t in COST_TERMS}


def assumed_lambda_vessel(risk_max: float) -> float:
    """§10.6 동적 본선계수 — 위험도(0~1)별 초기 후보값 (민감도 대상, assumed)."""
    if risk_max >= 0.8:
        return 6.0
    if risk_max >= 0.6:
        return 4.0
    if risk_max >= 0.4:
        return 2.5
    if risk_max >= 0.2:
        return 1.5
    return 1.0


@dataclass
class CostAccumulator:
    _rate: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in RATE_TERMS})
    _pending: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in COST_TERMS})
    _episode: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in COST_TERMS})

    def set_rate(self, term: str, rate: float) -> None:
        if term not in RATE_TERMS:
            raise ValueError(f"{term} 은 rate 항 아님")
        self._rate[term] = max(0.0, float(rate))

    def advance(self, t0: float, t1: float) -> None:
        dt = t1 - t0
        if dt <= 0:
            return
        for term, rate in self._rate.items():
            self.accrue(term, rate * dt)

    def accrue(self, term: str, amount: float) -> None:
        """임의 항에 raw 증분 누적 (한 구간·한 소스에서만 호출 — 이중계상 금지)."""
        if term not in self._pending:
            raise ValueError(f"미지원 비용항 {term}")
        if amount < 0:
            raise ValueError(f"{term} 음수 증분 {amount}")
        self._pending[term] += float(amount)
        self._episode[term] += float(amount)

    def cut(self) -> dict[str, float]:
        """현재 결정구간 raw 를 반환하고 리셋 (_episode 보존)."""
        out = dict(self._pending)
        self._pending = {t: 0.0 for t in COST_TERMS}
        return out

    def episode_raw(self) -> dict[str, float]:
        return dict(self._episode)
