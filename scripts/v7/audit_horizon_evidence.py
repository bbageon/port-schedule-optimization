"""Read existing horizon artifacts; never simulate, fit, or change a policy.

This is an exploratory audit, not a preregistered performance experiment.
Branch endpoints are compared without pretending their difference is a
time-interval cost: the saved branches contain no component or event traces.
Only aggregate field availability is exported from the private DGT sample.
"""
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/reports/yr323_horizon_sweep'
OLD = ROOT / 'outputs/reports/yr332_reachable_teacher_corrected'
OUT = ROOT / 'outputs/reports/yr323_horizon_cause_audit'
KS = (1, 2, 3, 6)
SEEDS = (9914701, 9914702, 9914703)
LOADS = (3500, 7500, 15000)
COSTS = ('c_wait', 'c_vessel', 'c_move', 'c_rehandle')
INPUT_HASHES = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    INPUT_HASHES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6), (a, b)


def sign(x):
    return (x > 0) - (x < 0)


def group_stats(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: {'n': len(rs), 'reversals': sum(r['flip'] for r in rs)}
            for key, rs in sorted(groups.items())}


def pair_audit():
    pairs, common_samples, fits = [], [], defaultdict(list)
    missing = Counter()
    equal_endpoints = 0
    for seed in SEEDS:
        training = {}
        for k in KS:
            folder = OLD / f'model-{seed}' if k == 3 else SOURCE / f'k-{k}/model-{seed}'
            training[k] = read(folder / 'training.json')
        for iteration in range(6):
            datasets, comparisons, samples = {}, {}, {}
            for k in KS:
                path = (OLD / f'model-{seed}/labels-reachable-{iteration}.json.gz'
                        if k == 3 else SOURCE / f'k-{k}/model-{seed}/labels-{iteration}.json.gz')
                data = datasets[k] = read(path)
                tr = next(r for r in training[k]['rows'] if r['iteration'] == iteration
                          and (k != 3 or r['contract'] == 'reachable'))
                assert tr['corpus_sha256'] == INPUT_HASHES[path.relative_to(ROOT).as_posix()]
                fit = tr['fit']
                assert fit['steps'] == 50 and fit['loss_kind'] == 'pairwise'
                fits[k].append({key: fit[key] for key in
                                ('n', 'n_val', 'loss', 'val_loss', 'val_order_accuracy',
                                 'val_nonzero_pairs', 'scale_krw', 'floor_bound', 'steps')})
                cs = data['result']['crane']
                comparisons[k] = {tuple(c['key']): c for c in cs['comparisons']}
                assert len(comparisons[k]) == cs['labeled']
                samples[k] = defaultdict(list)
                for sample in data['samples']:
                    samples[k][tuple(sample['decision_key'])].append(sample)
                assert set(samples[k]) == set(comparisons[k])
                assert all(len(ss) == 2 for ss in samples[k].values())
                scale = cs['scale']
                assert scale > 0 and not cs['floor_bound']
                close(scale, st.median(abs(c['phi_alt'] - c['phi_factual'])
                                       for c in comparisons[k].values()))
                for key, c in comparisons[k].items():
                    ss = samples[k][key]
                    assert [s['action'] for s in ss] == ['PICKED', 'ALT']
                    assert c['factual_ok'] and c['forced_ok']
                    close(c['cost_end_s'], min(c['t'] + 3600 * k, 93600))
                    close(c['effective_horizon_s'], c['cost_end_s'] - c['t'])
                    for s, phi in zip(ss, (c['phi_factual'], c['phi_alt'])):
                        close(s['phi'], phi)
                        close(s['target'], (phi - s['base']) / scale)
                        close(s['base'], (c['phi_alt'] + c['phi_factual']) / 2)
                    if abs(c['phi_alt'] - c['phi_factual']) > 1e-6:
                        assert sign(ss[1]['target'] - ss[0]['target']) == sign(c['phi_alt'] - c['phi_factual'])
                    missing['component_cost_pairs_absent'] += not any(
                        f in c for f in ('breakdown_factual', 'components_factual', 'c_wait_factual'))
                    missing['branch_trace_pairs_absent'] += 'trace_factual' not in c
            base = datasets[1]
            for k in KS:
                assert set(comparisons[k]) == set(comparisons[1])
                assert datasets[k]['result']['crane']['main_choices_sha256'] == base['result']['crane']['main_choices_sha256']
                assert datasets[k]['input_seed'] == base['input_seed']
                assert datasets[k]['runtime'] == base['runtime']
            for key, c in comparisons[1].items():
                ss = samples[1][key]
                for k in KS:
                    ck = comparisons[k][key]
                    for name in ('picked_job', 'alt_job', 'assignment_factual', 'assignment_alt', 'witness'):
                        assert ck[name] == c[name], (k, key, name)
                    fields = ('row', 'action', 'job', 'flow', 'is_store')
                    assert [[s[f] for f in fields] for s in samples[k][key]] == [[s[f] for f in fields] for s in ss]
                gaps = {str(k): comparisons[k][key]['phi_alt'] - comparisons[k][key]['phi_factual'] for k in KS}
                flow_pair = [s['flow'] or 'NON_JOB' for s in ss]
                row = dict(model_seed=seed, iteration=iteration, input_seed=base['input_seed'],
                           load=base['load'], decision_key=list(key), picked_job=c['picked_job'],
                           alternative_job=c['alt_job'], flows=flow_pair,
                           same_flow=flow_pair[0] == flow_pair[1], gap_alt_minus_factual_krw=gaps,
                           flip=gaps['1'] * gaps['6'] < 0,
                           k6_clipped=comparisons[6][key]['effective_horizon_s'] < 21600 - 1e-6,
                           time_band=f'{int(c["t"] // 21600) * 6:02d}-{int(c["t"] // 21600) * 6 + 6:02d}',
                           effective_hours={str(k): comparisons[k][key]['effective_horizon_s'] / 3600 for k in KS})
                row['endpoint_gap_changes_krw'] = {
                    f'{a}_to_{b}': gaps[str(b)] - gaps[str(a)] for a, b in zip(KS, KS[1:])}
                for a, b in combinations(KS, 2):
                    ca, cb = comparisons[a][key], comparisons[b][key]
                    if ca['cost_end_s'] == cb['cost_end_s']:
                        close(ca['phi_factual'], cb['phi_factual'])
                        close(ca['phi_alt'], cb['phi_alt'])
                        equal_endpoints += 1
                pairs.append(row)
                common_samples.extend(dict(load=base['load'], flow=s['flow'], row=s['row']) for s in ss)
    assert len(pairs) == 295 and len(common_samples) == 590
    features = {}
    for load in LOADS:
        ss = [s for s in common_samples if s['load'] == load]
        vessels = [s for s in ss if s['row'][0] == 1]
        trucks = [s for s in ss if s['flow'].startswith('GATE_')]
        features[str(load)] = dict(samples=len(ss), vessel_samples=len(vessels),
            vessel_slack_at_cap=sum(abs(s['row'][3]) == 2 for s in vessels),
            truck_samples=len(trucks), truck_wait_at_cap=sum(s['row'][2] == 2 for s in trucks),
            candidate_count_at_cap=sum(s['row'][7] == 1 for s in ss))
    summary = dict(pairs_per_k=len(pairs), label_files=72, branch_pairs_checked=4*len(pairs),
        k1_k6_reversals=sum(r['flip'] for r in pairs),
        robust_reversals_above_1krw=sum(r['flip'] and min(abs(r['gap_alt_minus_factual_krw'][str(k)]) for k in (1,6)) > 1 for r in pairs),
        tie_status_changed=sum((r['gap_alt_minus_factual_krw']['1'] == 0) != (r['gap_alt_minus_factual_krw']['6'] == 0) for r in pairs),
        by_load=group_stats(pairs, 'load'), by_model=group_stats(pairs, 'model_seed'),
        by_clipping=group_stats(pairs, 'k6_clipped'), by_same_flow=group_stats(pairs, 'same_flow'),
        by_flow_pair=group_stats(pairs, 'flows'), by_time_band=group_stats(pairs, 'time_band'),
        equal_endpoint_pairs_verified=equal_endpoints, missing_measurements=dict(missing),
        feature_caps=features, training_fit={})
    for k in KS:
        ff = fits[k]
        n = sum(r['val_nonzero_pairs'] for r in ff)
        summary['training_fit'][str(k)] = dict(updates=len(ff), optimization_steps=sum(r['steps'] for r in ff),
            heldout_nonzero_pairs=n, weighted_order_accuracy=sum(r['val_order_accuracy']*r['val_nonzero_pairs'] for r in ff)/n,
            median_absolute_gap_krw=st.median(abs(r['gap_alt_minus_factual_krw'][str(k)]) for r in pairs),
            mean_effective_hours=st.mean(r['effective_hours'][str(k)] for r in pairs),
            clipped_pairs=sum(r['effective_hours'][str(k)] < k-1e-9 for r in pairs))
    return summary, pairs


def evaluation_audit():
    buckets = defaultdict(list)
    count = 0
    for path in sorted((SOURCE / 'evaluation').glob('*-0.json.gz')):
        data = read(path)
        arm, _, load, _, repeat = data['case']
        assert repeat == 0
        result = data['result']
        close(sum(result[k] for k in COSTS), result['phi_krw'])
        metrics = data['probe']['flow_timing']['at24']
        buckets[(arm, load)].append({**{k: result[k] for k in (*COSTS, 'phi_krw')},
            **{f'{flow}_mean_min': metrics[flow]['mean_s']/60 for flow in
               ('TRUCK', 'GATE_IN', 'GATE_OUT', 'VESSEL_DISCHARGE', 'VESSEL_LOAD')},
            'truck_unfinished': metrics['TRUCK']['unfinished']})
        count += 1
    assert count == 144
    original = read(SOURCE / 'summary.json')
    groups, deltas = [], []
    for (arm, load), values in sorted(buckets.items()):
        assert len(values) == (3 if arm == 'sf_spt' else 9)
        means = {key: st.mean(r[key] for r in values) for key in values[0]}
        previous = next(r for r in original['by_load'] if r['arm'] == arm and r['load'] == load)
        close(means['phi_krw'], previous['cost'])
        close(means['TRUCK_mean_min']*60, previous['TRUCK_mean_s_24'])
        groups.append(dict(arm=arm, load=load, runs=len(values), **means))
    for load in LOADS:
        a = next(r for r in groups if r['arm'] == 'k1' and r['load'] == load)
        b = next(r for r in groups if r['arm'] == 'k6' and r['load'] == load)
        delta = {key: b[key]-a[key] for key in (*COSTS, 'phi_krw')}
        deltas.append(dict(load=load, k6_minus_k1=delta,
                           truck_wait_fraction_of_cost_increase=delta['c_wait']/delta['phi_krw']))
    total_delta = {key: st.mean(r['k6_minus_k1'][key] for r in deltas) for key in (*COSTS,'phi_krw')}
    return dict(runs=count, independent_inputs=9, per_load=groups, k6_minus_k1_by_load=deltas,
                k6_minus_k1_mean_cost=total_delta,
                truck_wait_fraction_of_cost_increase=total_delta['c_wait']/total_delta['phi_krw'],
                interpretation='Whole-policy 24h outcomes; not a component decomposition of individual branch labels.')


def raw_availability():
    path = ROOT / 'data/raw/dgt_visits_sample_20260818_09h.jsonl'
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    INPUT_HASHES[path.relative_to(ROOT).as_posix()] = digest
    assert digest == '00816921df3393395b10491eaa09fc4f4c51bb30ba7c886a1c06c6e606ada431'
    rows = [json.loads(s) for s in data.decode('utf-8-sig').splitlines() if s.strip()]
    filled = {key: sum(r.get(key) not in (None,'') for r in rows) for key in
              ('reserved_s','gate_in_s','block_in_s','job_done_s','gate_out_s')}
    notice = [r for r in rows if r['extra'].get('copino_notice_s')]
    leads = [(datetime.fromisoformat(r['gate_in_s']) - datetime.fromisoformat(r['extra']['copino_notice_s'])).total_seconds() for r in notice]
    return dict(raw_file=path.relative_to(ROOT).as_posix(), sha256=digest, rows=len(rows),
                filled=filled, notice_rows=len(notice), arrival_before_notice=sum(v<0 for v in leads),
                reservation_error_distribution_supported=False,
                transport_fleet_calibration_fields_present=False,
                newer_reservation_file_located=False,
                search_scope='Workspace data directory and repository file/path references; no whole-disk search.',
                data_files=[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'data').rglob('*')) if p.is_file()],
                caution='Notice-to-arrival lead time is not reservation error; no private row identifiers are exported.')


def arrival_contract_audit():
    """Check saved actual arrivals against the schedule used as reservations.

The archive does not separately save the public reservation field. Its source
mapping is therefore an explicitly separate code finding, not a measured field.
"""
    rows_checked, max_error = 0, 0.0
    for load in LOADS:
        for day in range(3):
            rows = read(SOURCE / f'orders/initial-{SEEDS[0]}-{load}-{day}-0.json.gz')
            assert len(rows) == load
            for row in rows:
                actual = row['times']['gate_in_s']
                assert actual is not None
                error = abs(actual-row['scheduled_gate_s'])
                close(actual, row['scheduled_gate_s'])
                max_error = max(max_error, error)
                rows_checked += 1
    return dict(independent_inputs=9, rows_checked=rows_checked,
        max_scheduled_actual_error_s=max_error, public_reservation_field_saved_separately=False,
        code_findings=[
            'stage/vessels.py:build_diurnal_v3 stores arrival_s directly from the arrival stream.',
            'stage/orders.py:orders_from_schedule maps in_out_reserve_s to round(arrival_s, 3).',
            'stage/orders.py:V3Announcer.review passes that arrival_s to _job_from_entry and admission.',
            'world/integrated/terminal_stream.py:_job_from_entry defaults realized_gate_in_s to the supplied gate-in.',
            'Its provided_eta uses the same gate-in plus expected travel; actual travel can still differ.',
            'The 600-second adherence setting exists in another builder, not this daily evaluation path.'])


def main():
    campaign = read(SOURCE / 'campaign.json')
    for filename, expected in {**campaign['sources'], **campaign['reused_artifacts']}.items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected, filename
    summary, pairs = pair_audit()
    summary = dict(scope='Exploratory read-only audit of existing artifacts; no simulations or model fits.',
                   pairs=summary, evaluation=evaluation_audit(), raw_availability=raw_availability(),
                   arrival_contract=arrival_contract_audit())
    source_paths = [Path(__file__).resolve(), ROOT/'src/yard_rl/v7/crane/rollout.py',
                   ROOT/'src/yard_rl/v7/stage/orders.py', ROOT/'src/yard_rl/v7/stage/vessels.py',
                   ROOT/'src/yard_rl/v7/world/integrated/terminal_stream.py']
    for path in source_paths:
        INPUT_HASHES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    git = lambda *args: subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()
    script_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    script_in_commit = subprocess.run(['git','cat-file','-e',f'HEAD:{script_rel}'],cwd=ROOT,capture_output=True).returncode == 0
    manifest = dict(git_commit=git('rev-parse','HEAD'), tracked_dirty=bool(git('status','--short','--untracked-files=no')),
                    audit_script_in_commit=script_in_commit,
                    audit_script_sha256=INPUT_HASHES[Path(__file__).resolve().relative_to(ROOT).as_posix()],
                    source_hashes=INPUT_HASHES)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.json',summary),('input-manifest.json',manifest)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (OUT/'pairs.json.gz').write_bytes(gzip.compress(json.dumps(pairs,ensure_ascii=False,sort_keys=True).encode(),mtime=0))
    print(json.dumps({'pairs':len(pairs),'reversals':summary['pairs']['k1_k6_reversals'],
                      'clipping':summary['pairs']['by_clipping'],
                      'evaluation_runs':summary['evaluation']['runs'],
                      'raw_reservations':summary['raw_availability']['filled']['reserved_s']},ensure_ascii=False))


if __name__ == '__main__':
    main()
