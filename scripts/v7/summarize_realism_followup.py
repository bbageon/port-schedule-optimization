"""Read saved bc evaluations by observed-volume support; no new experiment.

This is post-hoc scope reporting under YR-332-ap. Neither the hourly support
classification nor the reductions certify overall physical realism.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/reports/yr332_global_policy'
READINESS = ROOT / 'outputs/reports/yr332_realism_readiness/2026-11-20'
OUT = ROOT / 'outputs/reports/yr332_realism_followup/2026-11-21'
FLOWS = ('TRUCK', 'VESSEL', 'GATE_IN', 'GATE_OUT', 'VESSEL_DISCHARGE', 'VESSEL_LOAD')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def combine(rows):
    means = {f: st.mean(r['completion']['at24'][f]['total_s'] /
                       r['completion']['at24'][f]['n'] / 60 for r in rows) for f in FLOWS}
    return dict(cases=len(rows), cost=st.mean(r['result']['phi_krw'] for r in rows),
                vessel_cost=st.mean(r['result']['c_vessel'] for r in rows),
                mean_min=means,
                unfinished26=st.mean(r['completion']['at26']['ALL_ORDERS']['unfinished'] for r in rows))


def change(base, candidate):
    pct = lambda a, b: 100 * (b / a - 1)
    return dict(cost_pct=pct(base['cost'], candidate['cost']),
                vessel_cost_pct=pct(base['vessel_cost'], candidate['vessel_cost']),
                time_pct={f: pct(base['mean_min'][f], candidate['mean_min'][f]) for f in FLOWS},
                unfinished26_delta=candidate['unfinished26'] - base['unfinished26'])


def run():
    cfg = read(SOURCE / 'campaign.json')['configuration']
    old = read(SOURCE / 'summary.json')
    support = read(READINESS / 'summary.json')
    protected = [SOURCE / n for n in ('summary.json', 'campaign.json', 'bundle.json', 'current-gates.json')]
    protected += [READINESS / n for n in ('summary.json', 'verification.json')]
    protected_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    rows, source_hashes = [], {}
    for i, case in enumerate(cfg['evaluation_cases']):
        if case['split'] != 'holdout' or case['mode'] != 'argmax':
            continue
        if case['stage'] not in ('initial', 'final', 'rule'):
            continue
        p = SOURCE / 'evaluation' / f'{i:03d}.json.gz'
        row = read(p)
        assert row['case'] == case
        row['label'] = 'initial' if case['stage'] == 'initial' else case['arm']
        source_hashes[p.relative_to(ROOT).as_posix()] = sha(p)
        for f in FLOWS:
            c = row['completion']['at24'][f]
            assert math.isclose(c['mean_s'], c['total_s'] / c['n'], rel_tol=1e-12)
        rows.append(row)
    assert len(rows) == 90
    by_load = {}
    for load in (3500, 7500, 15000):
        groups = defaultdict(list)
        for row in rows:
            if row['case']['load'] == load:
                groups[row['label']].append(row)
        assert {k: len(v) for k, v in groups.items()} == {'initial': 9, 'zero': 9, 'global': 9, 'SF_SPT': 3}
        def key(row):
            c = row['case']
            return c['index'], c['input_seed'], c['seed']
        paired = [{key(r): r for r in groups[label]} for label in ('zero', 'global')]
        assert paired[0].keys() == paired[1].keys()
        for k in paired[0]:
            assert paired[0][k]['schedule_sha256'] == paired[1][k]['schedule_sha256']
        results = {label: combine(part) for label, part in groups.items()}
        for label, values in results.items():
            expected = old['by_load'][str(load)]['groups'][label]
            for metric in ('cost', 'vessel_cost'):
                assert math.isclose(values[metric], expected[metric], rel_tol=1e-12)
            for flow in FLOWS:
                assert math.isclose(values['mean_min'][flow], expected['flows'][flow]['mean_min'], rel_tol=1e-12)
            assert values['unfinished26'] == expected['unfinished26_mean']
        seeds = {}
        for seed in cfg['model_seeds']:
            g = {label: combine([r for r in part if r['case']['seed'] == seed])
                 for label, part in groups.items() if label != 'SF_SPT'}
            seeds[str(seed)] = change(g['zero'], g['global'])
        volume = support['synthetic_input_support']['by_daily_order_load'][str(load)]
        by_load[str(load)] = dict(groups=results, hourly_support=volume,
            observed_max_hourly_visits=support['confirmed_sample']['max_physical_visits_per_hour'],
            volume_class='within_observed_hourly_max' if volume['hours_above_observed_max'] == 0 else 'extrapolated_stress',
            global_vs_zero=change(results['zero'], results['global']),
            global_vs_rule=change(results['SF_SPT'], results['global']),
            zero_vs_rule=change(results['SF_SPT'], results['zero']),
            global_vs_initial=change(results['initial'], results['global']),
            zero_vs_initial=change(results['initial'], results['zero']),
            by_model_seed=seeds)
    terminal_counts = Counter()
    for p in (ROOT / '오더데이터/선박/양적하목록').glob('*.json'):
        terminal_counts[read(p)['response']['data']['terminalCode']] += 1
    assert sum(terminal_counts.values()) == 263
    raw_manifest = read(READINESS / 'raw-replay/source-manifest.json')
    for entry in raw_manifest:
        assert sha(ROOT / '오더데이터' / entry['path']) == entry['sha256']
    actual_paths = {p.relative_to(ROOT / '오더데이터').as_posix()
                    for p in (ROOT / '오더데이터').rglob('*') if p.is_file()}
    assert actual_paths == {e['path'] for e in raw_manifest}
    assert protected_hashes == {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    save('volume-results.json', dict(schema='yard_rl.realism_followup.v1', task='YR-332-ap',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        by_daily_order_load=by_load, source_evaluations_sha256=source_hashes,
        protected_files_sha256=protected_hashes, script_sha256=sha(Path(__file__)),
        raw_files_unchanged=len(raw_manifest), vessel_terminal_code_counts=dict(terminal_counts),
        definitions=dict(truck_time='gate entry to job completion, censored at 24h',
                         vessel_time='stream plan start to yard job completion, censored at 24h',
                         aggregation='equal day and model weights within each fixed load',
                         sign='positive percentage means worse cost or longer time'),
        new_training_runs=0, new_simulator_runs=0, realism_pass=False,
        interpretation='Post-hoc scope analysis; hourly maximum support does not establish terminal coverage, fleet realism or queue semantics.'))
    save('verification.json', dict(passed=True, saved_evaluations=90,
        paired_final_cases=27, groups_matching_previous_summary=12,
        original_files_unchanged=600, protected_files_unchanged=True,
        volume_results_sha256=sha(OUT / 'volume-results.json')))
    for load, values in by_load.items():
        print(load, json.dumps(values['global_vs_zero']), flush=True)


if __name__ == '__main__':
    run()
