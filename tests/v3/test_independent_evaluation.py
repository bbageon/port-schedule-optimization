import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/v3'
sys.path.insert(0, str(SCRIPTS))
from independent_eval_checks import ARMS, SEEDS, paired_summary


def rows():
    return [dict(seed=s, arm=a, cost_28d_krw=cost, cost_30d_krw=cost+20,
        requested=100, completed=100, vessel_all_completed=True,
        requested_identity_sha256=str(s))
        for s in SEEDS for a, cost in zip(ARMS, (100., 80., 90.))]


def test_monthly_statistics_use_twenty_pairs_and_correct_direction():
    result = paired_summary(rows(), bootstrap_samples=100)
    assert result['months'] == 20 and result['policy_runs'] == 60
    main = result['comparisons']['28d:NO_REALLOC-RL']
    assert main['mean_saving_krw'] == 20
    assert main['mean_saving_interval_krw'] == [20, 20]
    assert main['ratio_of_sums_saving_pct'] == 20
    assert main['wins'] == 20 and main['interval_level'] == .975
    assert result['comparisons']['28d:RL_TIME-RL']['mean_saving_krw'] == 10
    assert not result['claim_eligible']


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'different_requests', 'nonfinite'])
def test_partial_or_unpaired_runs_cannot_be_reported_as_complete(problem):
    values = rows()
    if problem == 'missing':
        values.pop()
    elif problem == 'duplicate':
        values.append(values[0])
    elif problem == 'different_requests':
        values[0]['requested_identity_sha256'] = 'wrong'
    else:
        values[0]['cost_28d_krw'] = float('nan')
    with pytest.raises(ValueError):
        paired_summary(values, bootstrap_samples=10)


def test_unfinished_work_is_retained_and_prevents_completion_claim():
    values = rows()
    values[1]['completed'] = 99
    result = paired_summary(values, bootstrap_samples=10)
    assert not result['all_requested_work_completed']
    assert result['months'] == 20


def test_actual_engine_inputs_must_match_before_world_runs():
    from yard_rl.v3.stage.month import plan_days
    from yard_rl.v3.stage.month_run import run_month
    with pytest.raises(ValueError, match='Frozen monthly input mismatch'):
        run_month(seed=9_900_722, days=plan_days(9_900_722, (5,)), arm='NO_REALLOC',
                  admission_mode='PRESERVE', expected_input={'schedule_sha256': 'bad'})


def test_existing_daily_artifacts_recompute_exact_queues():
    from audit_daily_observation import audit
    folder = Path(__file__).resolve().parents[2] / 'outputs/reports/yr317_v3_daily_observation/run-e9838a3/on_RL'
    if not folder.exists():
        pytest.skip('Saved diagnostic artifacts not in this checkout')
    result = audit(folder)
    assert result['passed'] and result['checks']['queues_match_request_events']


@pytest.mark.parametrize('state', ['running', 'failed'])
def test_supply_run_must_finish_before_independent_start(tmp_path, state):
    from independent_eval_checks import save, supply_preflight
    save(tmp_path / 'progress.json', {'state': state})
    with pytest.raises(ValueError, match='not complete'):
        supply_preflight(tmp_path, 'checkpoint')


def test_missing_supply_summary_does_not_allow_start(tmp_path):
    from independent_eval_checks import save, supply_preflight
    save(tmp_path / 'progress.json', {'state': 'completed'})
    with pytest.raises(FileNotFoundError):
        supply_preflight(tmp_path, 'checkpoint')


def test_supply_completion_with_failed_checks_does_not_allow_start(tmp_path):
    from independent_eval_checks import save, supply_preflight
    save(tmp_path / 'progress.json', {'state': 'completed'})
    save(tmp_path / 'full/summary.json', {'passed': False})
    with pytest.raises(ValueError, match='failed'):
        supply_preflight(tmp_path, 'checkpoint')


def test_probe_completion_is_not_confused_with_record_validity():
    from independent_eval_checks import smoke_summary
    values = [dict(month=dict(seed=9_900_722, arm=arm),
        audit=dict(passed=True, all_trucks_completed=True, all_vessels_completed=arm != 'NO_REALLOC'))
        for arm in ARMS]
    result = smoke_summary(values)
    assert result['passed'] and not result['all_work_completed']
    assert result['independent_runs'] == 0
    values[0]['audit']['passed'] = False
    assert not smoke_summary(values)['passed']
    assert not smoke_summary(values[1:])['passed']
