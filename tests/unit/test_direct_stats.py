"""Unit tests for deterministic paired-bootstrap direct-job statistics."""
import math

import pytest

from yard_rl.experiments.direct_stats import (
    MetricDirection,
    MetricSpec,
    SummaryStatistic,
    paired_bootstrap,
    paired_changes,
    safe_percent_change,
)


def test_paired_seed_changes_use_baseline_denominator_and_metric_direction():
    changes = paired_changes(
        [10.0, 0.0],
        [8.0, 2.0],
        direction=MetricDirection.MINIMIZE,
        seeds=[101, 102],
    )

    assert changes[0].seed == 101
    assert changes[0].difference == -2.0
    assert changes[0].percent_change == -20.0
    assert changes[0].improvement == 2.0
    assert changes[0].improvement_percent == 20.0
    assert changes[1].percent_change is None
    assert changes[1].improvement_percent is None
    assert safe_percent_change(0.0, 1.0) is None


def test_maximize_direction_makes_positive_change_a_positive_improvement():
    change = paired_changes(
        [40.0],
        [50.0],
        direction=MetricDirection.MAXIMIZE,
    )[0]

    assert change.difference == change.improvement == 10.0
    assert change.percent_change == change.improvement_percent == 25.0


def test_bootstrap_is_seed_reproducible_and_preserves_pairs():
    metric = MetricSpec("mean_wait_min", MetricDirection.MINIMIZE)
    baseline = [1.0, 10.0, 100.0, 1_000.0]
    alternative = [3.0, 12.0, 102.0, 1_002.0]

    first = paired_bootstrap(
        baseline, alternative, metric=metric, seed=71, n_resamples=500
    )
    second = paired_bootstrap(
        baseline, alternative, metric=metric, seed=71, n_resamples=500
    )

    assert first == second
    # Resampling common indices keeps the constant within-pair difference exact.
    assert first.difference == 2.0
    assert first.difference_ci.lower == pytest.approx(2.0)
    assert first.difference_ci.upper == pytest.approx(2.0)
    assert first.improvement_ci.lower == pytest.approx(-2.0)
    assert first.improvement_ci.upper == pytest.approx(-2.0)


def test_different_bootstrap_seeds_can_change_non_degenerate_interval():
    metric = MetricSpec("queue_area_h", MetricDirection.MINIMIZE)
    kwargs = dict(
        baseline=[1.0, 2.0, 4.0, 8.0, 16.0],
        alternative=[2.0, 1.0, 5.0, 6.0, 20.0],
        metric=metric,
        n_resamples=37,
    )

    first = paired_bootstrap(**kwargs, seed=1)
    second = paired_bootstrap(**kwargs, seed=2)

    assert first.difference_ci != second.difference_ci


def test_p95_summary_and_serializable_direction_are_explicit():
    result = paired_bootstrap(
        [1.0, 2.0, 3.0, 100.0],
        [1.0, 2.0, 3.0, 80.0],
        metric=MetricSpec(
            "p95_wait_min", MetricDirection.MINIMIZE, SummaryStatistic.P95
        ),
        seeds=[10, 11, 12, 13],
        seed=5,
        n_resamples=100,
    )
    payload = result.as_dict()

    assert result.alternative < result.baseline
    assert result.improvement > 0.0
    assert payload["metric"] == {
        "name": "p95_wait_min",
        "direction": "minimize",
        "statistic": "p95",
    }
    assert payload["n"] == 4
    assert payload["pairs"][0]["seed"] == 10


def test_zero_bootstrap_baseline_suppresses_unsafe_percent_interval():
    result = paired_bootstrap(
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
        metric=MetricSpec("rare_rate", MetricDirection.MINIMIZE),
        seed=3,
        n_resamples=100,
    )

    # Some resamples contain only zero baselines, so no conditional percent CI.
    assert result.percent_change is not None
    assert result.percent_change_ci is None
    assert result.improvement_percent_ci is None


@pytest.mark.parametrize(
    ("baseline", "alternative", "message"),
    [
        ([], [], "at least 2"),
        ([1.0], [1.0, 2.0], "equal lengths"),
        ([1.0, math.nan], [1.0, 2.0], "finite"),
        ([1.0, 2.0], [1.0, math.inf], "finite"),
    ],
)
def test_bootstrap_rejects_invalid_paired_inputs(baseline, alternative, message):
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(
            baseline,
            alternative,
            metric=MetricSpec("wait", MetricDirection.MINIMIZE),
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_resamples": 0}, ValueError, "at least 1"),
        ({"n_resamples": 2.5}, TypeError, "integer"),
        ({"confidence": 1.0}, ValueError, "between 0 and 1"),
        ({"seed": True}, TypeError, "integer"),
        ({"zero_atol": -1.0}, ValueError, "non-negative"),
        ({"seeds": [7, 7]}, ValueError, "unique"),
        ({"seeds": [7]}, ValueError, "equal lengths"),
    ],
)
def test_bootstrap_parameter_validation(kwargs, error, message):
    with pytest.raises(error, match=message):
        paired_bootstrap(
            [1.0, 2.0],
            [0.5, 1.5],
            metric=MetricSpec("wait", MetricDirection.MINIMIZE),
            **kwargs,
        )


def test_default_resample_count_is_ten_thousand():
    result = paired_bootstrap(
        [1.0, 2.0],
        [0.5, 1.5],
        metric=MetricSpec("wait", MetricDirection.MINIMIZE),
        seed=9,
    )

    assert result.n_resamples == 10_000
