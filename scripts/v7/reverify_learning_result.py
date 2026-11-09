"""Independently recalculate published ledgers and replay twelve fixed evaluations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

SOURCE = Path('outputs/reports/yr332_reachable_teacher_corrected')
OUT = Path('outputs/reports/yr332_learning_reverification')
PLAN = Path('docs/research/v7-discharge/learning-result-reverification.md')
LOADS = (3500, 7500, 15000)
SEEDS = (9914701, 9914702, 9914703)
ARMS = ('initial', 'legacy_force-5', 'reachable-5', 'sf_spt')
REPLAYS = [(arm, 0 if arm == 'sf_spt' else SEEDS[0], load, 0, 0)
           for load in LOADS for arm in ARMS]
FIELDS = ('gate_in_s', 'block_in_s', 'service_start_s', 'job_done_s', 'gate_out_s')
HASHES = ('candidates_sha256', 'choices_sha256', 'events_sha256',
          'jobs_sha256', 'records_sha256', 'result_sha256')


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-7), (actual, expected)


def filename(case):
    return '-'.join(map(str, case))+'.json.gz'


def ledger_metrics(rows):
    """No imports of the production metric or the original report's calculations."""
    n = len(rows)
    assert n and len({r['job_id'] for r in rows}) == n
    for row in rows:
        assert row['admitted'] is True and row['review_count'] == 1 and row['reason'] is None
        times = [row['times'][k] for k in FIELDS]
        assert times[0] is not None and 0 <= times[0] <= 86400
        assert times[0] == row['scheduled_gate_s']
        close(times[1], times[0]+row['travel_s'])
        present = [v for v in times if v is not None]
        assert all(math.isfinite(v) for v in present)
        assert all(a <= b+1e-6 for a, b in zip(present, present[1:]))
        # Missing later milestones are allowed; a later event cannot lack its predecessor.
        for earlier, later in zip(times, times[1:]):
            assert later is None or earlier is not None
        for horizon, suffix in ((86400., '24'), (93600., '26')):
            observed = {k: v if v is not None and v <= horizon+1e-9 else None
                        for k, v in zip(FIELDS, times)}
            assert observed == row['records'+suffix]
    result = dict(missing=0)
    for horizon, suffix in ((86400., '24'), (93600., '26')):
        # Calculate from the observed record at this horizon, not the published mean.
        observed = [r['records'+suffix] for r in rows]
        durations = [(horizon if r['gate_out_s'] is None else r['gate_out_s'])-r['gate_in_s']
                     for r in observed]
        assert min(durations) >= 0
        result['turn_s'+suffix] = math.fsum(durations)/n
        result['unexited'+suffix] = sum(r['gate_out_s'] is None for r in observed)
        result['yard_unfinished'+suffix] = sum(r['job_done_s'] is None for r in observed)
    return result


def interval(values):
    from scipy.stats import t
    n = len(values)
    mean, sd = st.fmean(values), st.stdev(values)
    margin = float(t.ppf(.975, n-1))*sd/math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, ci_lo=mean-margin, ci_hi=mean+margin)


def audit():
    manifest, original, verified = [read(SOURCE/p) for p in
                                  ('campaign.json', 'summary.json', 'verification.json')]
    expected_hashes = {**manifest['sources'], **verified['artifact_sha256'],
                      str(SOURCE/'summary.json'): verified['summary_sha256'],
                      str(SOURCE/'README.md'): verified['report_sha256']}
    for path, expected in expected_hashes.items():
        assert sha(path) == expected, ('changed artifact', path)
    cases = [tuple(c) for c in manifest['configuration']['evaluation']]
    assert len(cases) == len(set(cases)) == 208
    assert {p.name for p in (SOURCE/'evaluation').glob('*.json.gz')} == {filename(c) for c in cases}
    values, identities, sources, totals = {}, {}, {}, 0
    for case in cases:
        path = SOURCE/'evaluation'/filename(case)
        data, rows = read(path), read(SOURCE/'orders'/path.name)
        arm, seed, load, day, repeat = case
        assert data['case'] == list(case)
        assert data['input_seed'] == 9915900+100*day+10*LOADS.index(load)
        assert data['runtime'] == manifest['configuration']['runtime']
        assert len(rows) == load == data['result']['admitted'] == data['result']['n_trucks']
        assert data['result']['policy_exceptions'] == data['result']['rollout_worlds'] == 0
        assert data['result']['traded_edges'] == 0
        if data['checkpoint']:
            expected_path = SOURCE/f'model-{seed}'/(arm+'.pt')
            assert Path(data['checkpoint']) == expected_path
            assert sha(expected_path) == data['checkpoint_sha256']
        else:
            assert seed == 0
        fixed = [(r['job_id'], r['flow'], r['block'], r['scheduled_gate_s'], r['notice_s'], r['travel_s'])
                 for r in sorted(rows, key=lambda r: r['job_id'])]
        identity = (data['input_seed'], data['probe']['metadata']['input_sha256'],
                    hashlib.sha256(json.dumps(fixed).encode()).hexdigest())
        assert identities.setdefault((load, day), identity) == identity
        metrics = ledger_metrics(rows)
        close(metrics['turn_s24'], data['result']['mean_turn_time_s'])
        assert metrics['unexited24'] == data['result']['n_censored']
        metrics['cost'] = data['result']['phi_krw']
        for key in ('vessel_unfinished', 'vessel_idle_s', 'vessel_delay_s'):
            metrics[key] = data['probe']['operational'][key]
        values[case] = metrics
        sources[path.as_posix()] = sha(path)
        sources[(SOURCE/'orders'/path.name).as_posix()] = sha(SOURCE/'orders'/path.name)
        totals += load
    assert totals == original['checked_order_rows'] == 1809000
    aggregate = {}
    for published in original['aggregate']:
        selected = [v for (a, s, l, d, r), v in values.items()
                    if a == published['arm'] and l == published['load'] and r == 0]
        assert len(selected) == published['runs']
        calculated = {k: st.fmean(v[k] for v in selected) for k in selected[0]}
        for k, v in calculated.items():
            close(v, published[k])
        aggregate[published['arm'], published['load']] = calculated
    daily = {}
    for arm in sorted({case[0] for case in cases if not case[4]}):
        for load in LOADS:
            for day in range(3):
                selected = [v for (a, s, l, d, r), v in values.items() if (a, l, d, r) == (arm, load, day, 0)]
                assert len(selected) == (3 if arm not in manifest['configuration']['rules'] else 1)
                daily[arm, load, day] = {k: st.fmean(v[k] for v in selected) for k in selected[0]}
    comparisons = []
    for published in original['comparisons']:
        control, metric = published['control'], published['metric']
        pairs = [(daily['reachable-5', load, day][metric], daily[control, load, day][metric])
                 for load in LOADS for day in range(3)]
        raw = interval([x-y for x, y in pairs])
        percentages = interval([100*(x/y-1) for x, y in pairs]) if all(y for x, y in pairs) else None
        for calc, old in ((raw, published['paired_raw']), (percentages, published['paired_percent'])):
            if calc is not None:
                for k in ('n', 'mean', 'sd', 'ci_lo', 'ci_hi'):
                    close(calc[k], old[k])
            else:
                assert old is None
        comparisons.append(dict(control=control, metric=metric, paired_raw=raw, paired_percent=percentages))
    model_changes = []
    for published in original['model_turn_change_percent']:
        seed, control = published['model_seed'], published['control']
        changes = []
        for load in LOADS:
            for day in range(3):
                x = values['reachable-5', seed, load, day, 0]['turn_s24']
                y = values[control, 0 if control == 'sf_spt' else seed, load, day, 0]['turn_s24']
                changes.append(100*(x/y-1))
        close(st.fmean(changes), published['mean_percent'])
        model_changes.append(dict(model_seed=seed, control=control, mean_percent=st.fmean(changes)))
    sensitivity = []
    for control in ('initial', 'legacy_force-5', 'sf_spt'):
        for metric in ('turn_s24', 'turn_s26'):
            pairs = [(load, daily['reachable-5', load, day][metric], daily[control, load, day][metric])
                     for load in LOADS for day in range(3)]
            sensitivity.append(dict(control=control, metric=metric,
                mean_input_percent=st.fmean(100*(x/y-1) for n, x, y in pairs),
                ratio_equal_input_means_percent=100*(math.fsum(x for n, x, y in pairs)/math.fsum(y for n, x, y in pairs)-1),
                pooled_truck_time_percent=100*(math.fsum(n*x for n, x, y in pairs)/math.fsum(n*y for n, x, y in pairs)-1)))
    train_seeds = set(manifest['repro']['seeds']['training'])
    eval_seeds = {v[0] for v in identities.values()}
    assert not train_seeds.intersection(eval_seeds)
    by_load = [dict(load=load, **{arm: aggregate[arm, load]['turn_s24']/60 for arm in ARMS},
        saved_minutes=(aggregate['initial', load]['turn_s24']-aggregate['reachable-5', load]['turn_s24'])/60)
        for load in LOADS]
    result = dict(status='PASS', checked_runs=len(cases), checked_order_rows=totals,
        original_hashes_verified=len(expected_hashes), distinct_inputs=len(identities),
        training_evaluation_seeds_disjoint=True, sources=sources, by_load=by_load,
        comparisons=comparisons, model_changes=model_changes, sensitivity=sensitivity,
        calculation='Independent ledger arithmetic; model means first, nine input-level percentages second',
        unit='All admitted trucks; durations censored at each observation horizon')
    save(OUT/'independent-ledger-audit.json', result)
    print(json.dumps(dict(ledger_audit='PASS', order_rows=totals, by_load=by_load,
                         sensitivity=sensitivity), ensure_ascii=False), flush=True)


def replay(index):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from six_step_learning import load_net
    torch.set_num_threads(1)
    case = REPLAYS[index]
    previous = read(SOURCE/'evaluation'/filename(case))
    net = load_net(previous['checkpoint']) if previous['checkpoint'] else None
    original, announcer = ledger.install_announcer()
    probe = ledger.LedgerProbe(announcer)
    started = time.perf_counter()
    try:
        ep = episode.run_episode(load=case[2], seed=previous['input_seed'],
            dispatcher='SF_SPT' if case[0] == 'sf_spt' else 'RL_CRANE', crane_net=net,
            evaluation_probe=probe, **previous['runtime'])
    finally:
        episode.V3Announcer = original
    actual = dict(case=case, input_seed=previous['input_seed'], runtime=previous['runtime'],
        result=ep.as_dict(), probe=probe.result, seconds=time.perf_counter()-started,
        checkpoint=previous['checkpoint'], checkpoint_sha256=previous['checkpoint_sha256'])
    save(OUT/'evaluation'/filename(case), actual)
    save(OUT/'orders'/filename(case), probe.rows)
    checks = dict(result=actual['result'] == previous['result'],
        metadata=probe.result['metadata'] == previous['probe']['metadata'],
        operational=probe.result['operational'] == previous['probe']['operational'],
        ledger_audit=probe.result['ledger_audit'] == previous['probe']['ledger_audit'],
        orders=probe.rows == read(SOURCE/'orders'/filename(case)),
        **{key: probe.result[key] == previous['probe'][key] for key in HASHES})
    save(OUT/'checks'/(filename(case).removesuffix('.gz')), dict(case=case, checks=checks,
         exact=all(checks.values()), seconds=actual['seconds']))
    assert all(checks.values()), (case, checks)
    print(f'{case}: exact; {actual["seconds"]:.1f}s', flush=True)


def run():
    from yard_rl.integrated.repro import repro_stamp
    git = lambda *args: subprocess.check_output(['git', *args], encoding='utf-8').strip()
    assert not git('status', '--porcelain', '--untracked-files=no'), 'commit audit implementation first'
    config = dict(cases=REPLAYS, workers=4, torch_threads=1, training_changed=False,
        independent_extra_inputs=0, raw_ledger_runs=208, original_source=SOURCE.as_posix())
    manifest = dict(configuration=config, original_campaign_sha256=sha(SOURCE/'campaign.json'),
        sources={p.as_posix(): sha(p) for p in (Path(__file__).resolve().relative_to(Path.cwd()), PLAN)},
        repro=repro_stamp(experiment='YR-332-c-reverification',
            seeds=dict(model=list(SEEDS), evaluation=[9915900, 9915910, 9915920]),
            params=config, prereg=PLAN.as_posix()))
    assert not (OUT/'manifest.json').exists(), 'preserve earlier audit artifacts'
    save(OUT/'manifest.json', manifest)
    audit()
    # Reuse the old refit verifier as an additional check, not as the independent metric calculation.
    from report_reachable_teacher import verify_training
    training, artifacts = verify_training()
    save(OUT/'training-refit.json', dict(status='PASS', training=training, artifact_sha256=artifacts))
    print('36 checkpoint refits: exact', flush=True)
    def one(i):
        log = OUT/'logs'/f'replay-{i}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', encoding='utf-8') as stream:
            p = subprocess.run([sys.executable, __file__, '--operation', 'replay', '--index', str(i)],
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert p.returncode == 0, log.read_text(encoding='utf-8')[-3000:]
        print(log.read_text(encoding='utf-8').strip(), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, i) for i in range(len(REPLAYS))]):
            future.result()
    checks = [read(OUT/'checks'/(filename(case).removesuffix('.gz'))) for case in REPLAYS]
    assert all(row['exact'] for row in checks)
    save(OUT/'verification.json', dict(status='PASS', ledger_runs=208, order_rows=1809000,
        replay_runs=12, replay_exact=True, checkpoint_refits=36,
        artifact_sha256={p.as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()
                         and p.suffix != '.log'},
        statement='Published arithmetic and selected deterministic executions reproduced; no new performance claim'))
    print('REVERIFICATION PASS: 208 ledgers, 12 fresh replays, 36 checkpoint refits', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('run', 'audit', 'replay'), default='run')
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    if args.operation == 'run':
        run()
    elif args.operation == 'audit':
        audit()
    else:
        replay(args.index)
