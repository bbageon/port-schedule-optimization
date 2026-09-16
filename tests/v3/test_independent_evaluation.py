import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

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


def test_queue_at_review_includes_truck_dispatched_at_the_same_timestamp():
    from audit_daily_observation import queue_audit
    from yard_rl.v3.stage.daily_observation import snapshot
    from yard_rl.v3.world.domain.enums import JobFlow, JobStatus
    from yard_rl.v3.world.domain.models import Job
    job = Job('same-time', JobFlow.GATE_OUT, 0, 0, 100, status=JobStatus.WAITING)
    world = NS(blocks={'Y01': NS(jobs={job.job_id:job},
        kpis=NS(queue_area_s=0), fleet=NS(all=lambda: []))})
    before = snapshot(world, 300)
    job.service_start, job.status = 300, JobStatus.RUNNING
    after = snapshot(world, 301)
    final_events = [dict(final_block='Y01', block_in_s=100, service_start_s=300)]
    out = queue_audit([before, after], final_events)
    assert out['passed'] and out['same_time_starts'] == 1
    before['blocks']['Y01']['truck_queue'] = 0  # Wrong post-dispatch count at the pre-dispatch sample.
    assert not queue_audit([before, after], final_events)['passed']
    before['blocks']['Y01']['truck_queue'] = 2  # Genuine excess queue must still fail.
    assert not queue_audit([before, after], final_events)['passed']


def test_final_events_with_missing_arrival_or_early_service_are_not_hidden():
    from audit_daily_observation import queue_audit
    states=[dict(at_s=300, blocks={'Y01':dict(truck_queue=1)})]
    assert not queue_audit(states, [dict(final_block='Y01',block_in_s=301,service_start_s=400)])['passed']
    assert not queue_audit(states, [dict(final_block='Y01',block_in_s=100,service_start_s=299)])['passed']


def test_failed_worker_does_not_kill_other_active_workers_or_launch_new_ones(tmp_path, monkeypatch):
    import run_independent_evaluation as runner
    from independent_eval_checks import save, read
    monkeypatch.setattr(runner, 'resources', lambda cfg: (2, 100_000))
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    launched = []
    class Process:
        def __init__(self, cmd, **kwargs):
            self.seed, self.arm = int(cmd[cmd.index('--seed')+1]), cmd[cmd.index('--arm')+1]
            self.pid, self.polls, self.killed = 100+len(launched), 0, False
            launched.append(self)
        def poll(self):
            self.polls += 1
            if self.arm == 'NO_REALLOC': return 1
            if self.polls == 1: return None
            save(tmp_path/'months'/str(self.seed)/self.arm/'completion.json', {'completed':True})
            return 0
        def terminate(self): self.killed = True
        def kill(self): self.killed = True
        def wait(self, timeout): return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    args=NS(out=tmp_path, config=tmp_path/'config',workspace=tmp_path,recover_from=None)
    with pytest.raises(RuntimeError, match='all other active jobs allowed to finish'):
        runner.run_jobs(args, {'worker_budget_gib':3}, [(1,'NO_REALLOC'),(1,'RL'),(2,'RL')])
    assert len(launched)==2 and not any(p.killed for p in launched)
    outcome=read(tmp_path/'months-outcomes.json')
    assert len(outcome['completed'])==1 and len(outcome['failed'])==1 and outcome['pending']==[[2,'RL']]


def test_recovery_refuses_a_changed_model_or_run_contract():
    from recover_independent_evaluation import require_compatible
    contract={'settings':{'seed':1,'arm':'RL','days':[]},'runtime':{'version':'fixed'}}
    repro={'checkpoint_sha256':'fixed','code':{'git_dirty':False}}
    manifest={'contract':contract,'repro':repro}
    result={'seed':1,'arm':'RL','plan':[],'repro':repro}
    require_compatible(manifest,result,contract,'fixed')
    changed=deepcopy(contract);changed['settings']['seed']=2
    with pytest.raises(ValueError,match='settings/model/engine differ'):
        require_compatible(manifest,result,changed,'fixed')
    with pytest.raises(ValueError,match='provenance differs'):
        require_compatible(manifest,result,contract,'changed')
