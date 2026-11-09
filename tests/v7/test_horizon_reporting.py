"""Positive reduction must mean improvement, including exported labels/tables."""
from pathlib import Path
import pytest


@pytest.fixture
def reports(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v7'))
    import report_horizon_sweep as report
    import finalize_horizon_sweep as final
    monkeypatch.setattr(report, 'OUT', tmp_path)
    monkeypatch.setattr(final, 'OUT', tmp_path)
    values = {}
    factor = dict(initial=1., k1=.8, k2=1.1, k3=.9, k6=.95, sf_spt=.85)
    for arm in report.ARMS:
        for seed in ((0,) if arm == 'sf_spt' else report.SEEDS):
            for load in report.LOADS:
                for day in range(3):
                    v = dict(cost=100*factor[arm], vessel_unfinished=0,
                             vessel_idle_s=0, vessel_delay_s=0,
                             VESSEL_unfinished_26=0)
                    for end in ('24', '26'):
                        for m in report.METRICS:
                            v[f'{m}_mean_s_{end}'] = (120+day)*factor[arm]
                            v[f'{m}_n_{end}'] = load
                            v[f'{m}_unfinished_{end}'] = 0
                            v[f'{m}_not_ready_{end}'] = 0
                    values[(arm, seed, load, day, 0)] = v
    training = [dict(k=k, totals=dict(labeled=4, zero_raw_gap=0, sign_flip_vs_k3=0, truncated=0),
                    effective_horizon_s=dict(min=k*3600, max=k*3600, mean=k*3600)) for k in report.KS]
    summary = report.summarize(values, training, dict(runs=147, all_order_rows=1000))
    return report, final, summary


def test_reduction_and_statistical_label_have_same_direction(reports):
    _, _, summary = reports
    get = lambda k: next(r for r in summary['comparisons'] if
        r['k'] == k and r['control'] == 'initial' and r['metric'] == 'TRUCK_mean_s_24')
    assert get(1)['reduction_percent']['mean'] == pytest.approx(20)
    assert get(2)['reduction_percent']['mean'] == pytest.approx(-10)
    assert get(1)['reduction_percent']['label'] == '유의(개선)'
    assert get(2)['reduction_percent']['label'] == '유의(악화)'


def test_all_reported_numbers_match_source_after_rounding(reports):
    report, final, summary = reports
    report.write_report(summary)
    result = final.report_claims(summary)
    assert result.status.value == 'PASS'
