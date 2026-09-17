from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v3'))
from independent_eval_checks import ARMS, SEEDS, month_row, paired_summary, read, save, sha
from block_only_comparison import ALL_ARMS, _validated_run, additional_summary, merge_runs


def _make_run(folder, seed, arm):
    folder.mkdir(parents=True)
    plan = [dict(index=i, load=2+i % 2, seed=seed+i, t0=i*86400., n_days=30) for i in range(30)]
    daily_cost = {'NO_REALLOC': 100., 'RL': 80., 'RL_TIME': 90., 'RL_SPACE': 95.}[arm]
    # Preserve a loss and unfinished work, rather than cherry-picking successful months.
    if seed == SEEDS[0] and arm == 'RL_SPACE':
        daily_cost = 110.
    days = [dict(index=i, load=p['load'], train=0 < i < 29, phi_krw=daily_cost,
                 n_time=int(arm in ('RL', 'RL_TIME'))) for i, p in enumerate(plan)]
    inputs = {k: str(seed)+k for k in ('schedule_sha256', 'initial_scenarios_sha256', 'vessels_sha256')}
    settings = dict(seed=seed, arm=arm, days=plan, n_days=30, expected_input=inputs,
                    seller_net={'state_sha256':'seller'}, buyer_net={'state_sha256':'buyer'},
                    horizon_s=10800, supply_mode='COUNT_BALANCED')
    repro = dict(checkpoint_sha256='checkpoint', runner_sha256='runner', profile_id=None,
                 code=dict(git_dirty=False, git_head='old' if arm == 'NO_REALLOC' else 'new'))
    manifest = dict(contract=dict(schema='contract', label=arm, settings=settings,
        runtime=dict(engine='fixed', python='3.12')), repro=repro)
    requests = [dict(requested_arrival_s=i*100., final_reserved_arrival_s=i*100.) for i in range(75)]
    with gzip.open(folder/'requests.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.writelines(json.dumps(r)+'\n' for r in requests)
    for name in ('container-links.jsonl.gz', 'operating-state.jsonl.gz', 'daily-final.jsonl'):
        (folder/name).write_bytes(b'saved raw evidence\n')
    result = dict(seed=seed, arm=arm, label=arm, repro=repro, plan=plan, days=days,
        space=int(arm in ('RL', 'RL_SPACE')), time=sum(d['n_time'] for d in days),
        policy_exceptions=0, rollout_calls=0, elapsed_s=20,
        recording_checks={'fixed_model':True}, requested_identity_sha256=str(seed),
        request_summary=dict(requested=75, states={'COMPLETED':74, 'CENSORED':1},
            all_requests_admitted=True, unbound_jobs_at_end=1),
        vessel_work_summary=dict(unadmitted_moves=0, outstanding_yard_jobs=2,
            all_requested_work_completed=False),
        request_ledger_sha256=sha(folder/'requests.jsonl.gz'),
        container_links_sha256=sha(folder/'container-links.jsonl.gz'),
        daily_artifacts={name:sha(folder/name) for name in ('operating-state.jsonl.gz', 'daily-final.jsonl')})
    save(folder/'manifest.json', manifest)
    save(folder/'result.json', result)
    _refresh(folder)


def _refresh(folder):
    result, manifest = read(folder/'result.json'), read(folder/'manifest.json')
    okay = dict(passed=True, checks={'recording':True})
    save(folder/'independent-audit.json', dict(**okay, links=okay, daily=okay,
        result_sha256=sha(folder/'result.json')))
    save(folder/'completion.json', dict(state='completed', audit=okay, month=month_row(result),
        input_hashes=manifest['contract']['settings']['expected_input'],
        result_sha256=sha(folder/'result.json'), reused=result['arm']=='NO_REALLOC'))


@pytest.fixture
def runs(tmp_path):
    primary, block = tmp_path/'primary', tmp_path/'block'
    for seed in SEEDS:
        for arm in ALL_ARMS:
            _make_run((primary if arm in ARMS else block)/'months'/str(seed)/arm, seed, arm)
    return primary, block, tmp_path/'merged'


def test_merge_keeps_original_statistics_losses_backlog_and_whole_month_units(runs):
    primary, block, output = runs
    original_rows = [read(p)['month'] for p in (primary/'months').glob('*/*/completion.json')]
    original = paired_summary(original_rows)
    save(primary/'summary.json', original)
    raw_hashes = {p:sha(p) for source in (primary, block) for p in source.rglob('*') if p.is_file()}
    summary = merge_runs(*runs)
    assert summary['policy_runs'] == 80 and summary['months'] == 20
    assert summary['original_analysis'] == original
    assert not summary['claim_eligible'] and not summary['all_requested_work_completed']
    extra = summary['additional_analysis']
    assert extra['bootstrap_samples'] == 20_000 and extra['bootstrap_seed'] == 9_900_723
    first = extra['comparisons']['28d:NO_REALLOC-RL_SPACE']
    assert first['mean_saving_krw'] == 119 and first['losses'] == 1 and first['wins'] == 19
    assert first['interval_level'] == .975
    assert extra['comparisons']['30d:RL_SPACE-RL']['interval_level'] == .95
    assert len(summary['load_description']['per_month']) == 160
    assert len((output/'results.md').read_text().splitlines()) <= 200
    assert all(sha(p) == digest for p, digest in raw_hashes.items())


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'hash', 'raw_hash', 'audit',
    'input', 'plan', 'network', 'runtime', 'checkpoint', 'temporal', 'hidden_delay', 'identity'])
def test_merge_rejects_incomplete_changed_or_noncomparable_evidence(runs, problem):
    primary, block, output = runs
    folder = block/'months'/str(SEEDS[0])/'RL_SPACE'
    manifest, result = read(folder/'manifest.json'), read(folder/'result.json')
    if problem == 'missing':
        (folder/'completion.json').unlink()
    elif problem == 'duplicate':
        duplicate = block/'months'/('0'+str(SEEDS[0]))/'RL_SPACE'
        shutil.copytree(folder, duplicate)
    elif problem == 'hash':
        result['space'] += 1
        save(folder/'result.json', result)
    elif problem == 'raw_hash':
        (folder/'operating-state.jsonl.gz').write_bytes(b'changed')
    elif problem == 'audit':
        audit = read(folder/'independent-audit.json'); audit['daily']['checks']['bad']=False
        save(folder/'independent-audit.json', audit)
    else:
        settings = manifest['contract']['settings']
        if problem == 'input': settings['expected_input']['schedule_sha256']='changed'
        elif problem == 'plan':
            settings['days'][0]['load'] += 1; result['plan']=deepcopy(settings['days'])
            result['days'][0]['load'] += 1
        elif problem == 'network': settings['seller_net']['state_sha256']='changed'
        elif problem == 'runtime': manifest['contract']['runtime']['engine']='changed'
        elif problem == 'checkpoint':
            manifest['repro']['checkpoint_sha256']='changed'; result['repro']=deepcopy(manifest['repro'])
        elif problem == 'temporal': result['time']=1
        elif problem == 'identity': result['requested_identity_sha256']='changed'
        elif problem == 'hidden_delay':
            with gzip.open(folder/'requests.jsonl.gz', 'wt') as stream:
                stream.write(json.dumps(dict(requested_arrival_s=0., final_reserved_arrival_s=60.))+'\n')
            result['request_ledger_sha256']=sha(folder/'requests.jsonl.gz')
        save(folder/'manifest.json', manifest); save(folder/'result.json', result); _refresh(folder)
    with pytest.raises(ValueError): merge_runs(*runs)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'nonfinite', 'identity'])
def test_additional_statistics_refuse_partial_unpaired_or_nonfinite_rows(runs, problem):
    rows = [read(p)['month'] for source in runs[:2] for p in (source/'months').glob('*/*/completion.json')]
    if problem == 'missing': rows.pop()
    elif problem == 'duplicate': rows.append(rows[0])
    elif problem == 'nonfinite': rows[0]['cost_30d_krw']=float('nan')
    else: rows[0]['requested_identity_sha256']='different'
    with pytest.raises(ValueError): additional_summary(rows, bootstrap_samples=10)


def test_real_recovered_baseline_passes_read_only_compatibility_check():
    folder = Path(__file__).resolve().parents[2]/'outputs/reports/yr317_v3_independent_eval/run-7e2fb14/months/20300000/NO_REALLOC'
    if not folder.exists(): pytest.skip('Original completed month is not in this checkout')
    row, result, manifest, evidence = _validated_run(folder, 20_300_000, 'NO_REALLOC')
    assert evidence['reused'] and row['requested'] == 166500 and row['completed'] == 166498
