"""정규화 구간 비용·보상 계약 — 최종전략 §10 (YR-035).

한 결정 구간 [start, end] 의 13항 비용을 self-contained 로 담는다. 계약은 항목 이름·순서·
단위·항등식만 버전 고정하고, 실수치(scale/weight/λ_vessel)는 assumed config 로 위임(YR-038).

정보흐름 안전: 비용은 transition 이후 산출되어 정책 입력으로 되먹지 않는다. 따라서 실현지연
등 GROUND_TRUTH 를 비용 계산에는 쓸 수 있으나 state/observation/candidate 에는 넣지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import COST_TERMS, FLOAT_DECIMALS, SCHEMA_VERSION, VESSEL_FAMILY


def _q(x: float) -> float:
    """직렬화와 동일한 6자리 양자화 — 파생 total 과 저장 입력의 반올림 기준을 일치."""
    return round(float(x), FLOAT_DECIMALS)


@dataclass(frozen=True)
class CostBreakdown:
    schema_version: str
    interval_start_s: float
    interval_end_s: float
    raw: dict[str, float]            # 13항 물리단위 delta (keys == set(COST_TERMS))
    scale: dict[str, float]          # 정규화 기준값 (train FIFO fit; scale>0)
    weight: dict[str, float]         # assumed 가중치
    lambda_vessel: float             # §10.6 동적 본선계수 (본선 계열에만 곱)
    total_normalized: float          # = Σ contributions
    reward: float                    # = -total_normalized (파생, 검증용)
    assumed: bool = True

    def contributions(self) -> dict[str, float]:
        """항목별 정규화 기여 = λ(항이 본선계열이면) × weight × raw / scale (§10.2·10.3·10.6)."""
        out: dict[str, float] = {}
        for k in COST_TERMS:
            lam = self.lambda_vessel if k in VESSEL_FAMILY else 1.0
            out[k] = lam * self.weight[k] * self.raw[k] / self.scale[k]
        return out


def make_cost(*, interval_start_s: float, interval_end_s: float,
              raw: dict[str, float], scale: dict[str, float], weight: dict[str, float],
              lambda_vessel: float, assumed: bool = True) -> CostBreakdown:
    """항등식(total=Σcontrib, reward=-total)을 자동 충족하는 CostBreakdown 생성.

    입력(raw/scale/weight/λ)과 파생(total/reward)을 모두 6자리로 양자화해 직렬화 왕복
    후에도 loads(dumps(rec))==rec 와 validate_cost 항등식이 유지되게 한다 (직렬화가
    입력만 독립 반올림하면 재적재 시 total 재계산과 어긋나는 결함 방지, YR-035 리뷰).
    """
    raw = {k: _q(v) for k, v in raw.items()}
    scale = {k: _q(v) for k, v in scale.items()}
    weight = {k: _q(v) for k, v in weight.items()}
    lam = _q(lambda_vessel)
    tmp = CostBreakdown(
        schema_version=SCHEMA_VERSION, interval_start_s=_q(interval_start_s),
        interval_end_s=_q(interval_end_s), raw=raw, scale=scale, weight=weight,
        lambda_vessel=lam, total_normalized=0.0, reward=0.0, assumed=assumed)
    total = _q(sum(tmp.contributions().values()))
    return CostBreakdown(
        schema_version=SCHEMA_VERSION, interval_start_s=_q(interval_start_s),
        interval_end_s=_q(interval_end_s), raw=raw, scale=scale, weight=weight,
        lambda_vessel=lam, total_normalized=total, reward=_q(-total), assumed=assumed)
