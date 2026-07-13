"""Deterministic paired-bootstrap statistics for direct-job experiments.

The raw difference is always ``alternative - baseline``.  ``improvement``
normalizes that difference by metric direction, so a positive value always
means that the alternative is better.  Percent changes use the baseline as
their denominator and are undefined (``None``) when that denominator is zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import random
from statistics import fmean
from typing import Any, Sequence


class MetricDirection(str, Enum):
    """Whether a lower or a higher metric value is preferable."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @property
    def improvement_sign(self) -> float:
        return -1.0 if self is MetricDirection.MINIMIZE else 1.0


class SummaryStatistic(str, Enum):
    """Statistic applied to the paired seed-level metric values."""

    MEAN = "mean"
    P95 = "p95"


@dataclass(frozen=True)
class MetricSpec:
    """Metric identity, preferred direction, and across-seed summary."""

    name: str
    direction: MetricDirection
    statistic: SummaryStatistic = SummaryStatistic.MEAN

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name must not be empty")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("direction must be MetricDirection")
        if not isinstance(self.statistic, SummaryStatistic):
            raise TypeError("statistic must be SummaryStatistic")


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float = 0.95


@dataclass(frozen=True)
class PairedChange:
    """One common-random-number seed's baseline/alternative comparison."""

    seed: int
    baseline: float
    alternative: float
    difference: float
    percent_change: float | None
    improvement: float
    improvement_percent: float | None


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Point estimates and percentile CIs from paired index resampling."""

    metric: MetricSpec
    pairs: tuple[PairedChange, ...]
    baseline: float
    alternative: float
    difference: float
    percent_change: float | None
    improvement: float
    improvement_percent: float | None
    difference_ci: ConfidenceInterval
    percent_change_ci: ConfidenceInterval | None
    improvement_ci: ConfidenceInterval
    improvement_percent_ci: ConfidenceInterval | None
    n_resamples: int
    bootstrap_seed: int

    @property
    def n(self) -> int:
        return len(self.pairs)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["metric"]["direction"] = self.metric.direction.value
        payload["metric"]["statistic"] = self.metric.statistic.value
        payload["n"] = self.n
        return payload


def safe_percent_change(
    baseline: float,
    alternative: float,
    *,
    zero_atol: float = 0.0,
) -> float | None:
    """Return ``100 * (alternative - baseline) / baseline`` safely."""

    baseline_value = _finite_number(baseline, "baseline")
    alternative_value = _finite_number(alternative, "alternative")
    tolerance = _valid_zero_atol(zero_atol)
    if abs(baseline_value) <= tolerance:
        return None
    return 100.0 * (alternative_value - baseline_value) / baseline_value


def paired_changes(
    baseline: Sequence[float],
    alternative: Sequence[float],
    *,
    direction: MetricDirection,
    seeds: Sequence[int] | None = None,
    zero_atol: float = 0.0,
) -> tuple[PairedChange, ...]:
    """Compute changes for each seed while preserving the supplied pairing."""

    base, alt = _paired_values(baseline, alternative, min_size=1)
    direction = _valid_direction(direction)
    pair_seeds = _valid_seeds(seeds, len(base))
    tolerance = _valid_zero_atol(zero_atol)
    sign = direction.improvement_sign
    changes: list[PairedChange] = []
    for seed, base_value, alt_value in zip(pair_seeds, base, alt):
        difference = alt_value - base_value
        percent = safe_percent_change(
            base_value, alt_value, zero_atol=tolerance
        )
        changes.append(
            PairedChange(
                seed=seed,
                baseline=base_value,
                alternative=alt_value,
                difference=difference,
                percent_change=percent,
                improvement=sign * difference,
                improvement_percent=None if percent is None else sign * percent,
            )
        )
    return tuple(changes)


def paired_bootstrap(
    baseline: Sequence[float],
    alternative: Sequence[float],
    *,
    metric: MetricSpec,
    seeds: Sequence[int] | None = None,
    seed: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    zero_atol: float = 0.0,
) -> PairedBootstrapResult:
    """Compare paired seed-level values with a deterministic percentile CI.

    Each bootstrap draw samples indices once and applies those same indices to
    baseline and alternative values.  A percent CI is returned only when every
    resampled baseline summary has a non-zero denominator.
    """

    base, alt = _paired_values(baseline, alternative, min_size=2)
    if not isinstance(metric, MetricSpec):
        raise TypeError("metric must be MetricSpec")
    bootstrap_seed = _valid_seed(seed)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    confidence_value = _finite_number(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    tolerance = _valid_zero_atol(zero_atol)
    pairs = paired_changes(
        base,
        alt,
        direction=metric.direction,
        seeds=seeds,
        zero_atol=tolerance,
    )

    baseline_point = _summarize(base, metric.statistic)
    alternative_point = _summarize(alt, metric.statistic)
    difference_point = alternative_point - baseline_point
    percent_point = safe_percent_change(
        baseline_point, alternative_point, zero_atol=tolerance
    )
    sign = metric.direction.improvement_sign

    rng = random.Random(bootstrap_seed)
    difference_samples: list[float] = []
    percent_samples: list[float] = []
    percent_is_defined = True
    size = len(base)
    for _ in range(n_resamples):
        indices = [rng.randrange(size) for _ in range(size)]
        sampled_base = [base[index] for index in indices]
        sampled_alt = [alt[index] for index in indices]
        base_summary = _summarize(sampled_base, metric.statistic)
        alt_summary = _summarize(sampled_alt, metric.statistic)
        difference_samples.append(alt_summary - base_summary)
        percent = safe_percent_change(
            base_summary, alt_summary, zero_atol=tolerance
        )
        if percent is None:
            percent_is_defined = False
        else:
            percent_samples.append(percent)

    difference_ci = _percentile_ci(difference_samples, confidence_value)
    improvement_samples = [sign * value for value in difference_samples]
    improvement_ci = _percentile_ci(improvement_samples, confidence_value)
    percent_ci = (
        _percentile_ci(percent_samples, confidence_value)
        if percent_is_defined
        else None
    )
    improvement_percent_ci = (
        _percentile_ci(
            [sign * value for value in percent_samples], confidence_value
        )
        if percent_is_defined
        else None
    )
    return PairedBootstrapResult(
        metric=metric,
        pairs=pairs,
        baseline=baseline_point,
        alternative=alternative_point,
        difference=difference_point,
        percent_change=percent_point,
        improvement=sign * difference_point,
        improvement_percent=(
            None if percent_point is None else sign * percent_point
        ),
        difference_ci=difference_ci,
        percent_change_ci=percent_ci,
        improvement_ci=improvement_ci,
        improvement_percent_ci=improvement_percent_ci,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _paired_values(
    baseline: Sequence[float],
    alternative: Sequence[float],
    *,
    min_size: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        base_raw, alt_raw = tuple(baseline), tuple(alternative)
    except TypeError as exc:
        raise TypeError("baseline and alternative must be sequences") from exc
    if len(base_raw) != len(alt_raw):
        raise ValueError("paired inputs must have equal lengths")
    if len(base_raw) < min_size:
        raise ValueError(f"paired inputs require at least {min_size} values")
    base = tuple(
        _finite_number(value, f"baseline[{index}]")
        for index, value in enumerate(base_raw)
    )
    alt = tuple(
        _finite_number(value, f"alternative[{index}]")
        for index, value in enumerate(alt_raw)
    )
    return base, alt


def _valid_direction(direction: MetricDirection) -> MetricDirection:
    if not isinstance(direction, MetricDirection):
        raise TypeError("direction must be MetricDirection")
    return direction


def _valid_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def _valid_seeds(seeds: Sequence[int] | None, size: int) -> tuple[int, ...]:
    if seeds is None:
        return tuple(range(size))
    try:
        values = tuple(seeds)
    except TypeError as exc:
        raise TypeError("seeds must be a sequence") from exc
    if len(values) != size:
        raise ValueError("seeds and paired inputs must have equal lengths")
    validated = tuple(_valid_seed(seed) for seed in values)
    if len(set(validated)) != len(validated):
        raise ValueError("seeds must be unique")
    return validated


def _valid_zero_atol(zero_atol: float) -> float:
    value = _finite_number(zero_atol, "zero_atol")
    if value < 0.0:
        raise ValueError("zero_atol must be non-negative")
    return value


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _summarize(values: Sequence[float], statistic: SummaryStatistic) -> float:
    if statistic is SummaryStatistic.MEAN:
        return fmean(values)
    if statistic is SummaryStatistic.P95:
        return _quantile(values, 0.95)
    raise ValueError(f"unsupported statistic: {statistic!r}")


def _percentile_ci(
    values: Sequence[float], confidence: float
) -> ConfidenceInterval:
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        lower=_quantile(values, tail),
        upper=_quantile(values, 1.0 - tail),
        confidence=confidence,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated sample quantile (type-7 convention)."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight
