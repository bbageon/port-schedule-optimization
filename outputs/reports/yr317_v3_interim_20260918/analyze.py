"""Descriptive snapshot of completed runs; never changes evaluation or its final statistics."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CUTOFF = datetime.fromisoformat('2026-09-18T03:58:35+00:00')
ARMS = ('NO_REALLOC', 'RL', 'RL_TIME', 'RL_SPACE')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    runs, evidence = {}, []
    for base in ('yr317_v3_independent_eval/run-7e2fb14', 'yr317_v3_block_only/run-ea9164b'):
        root = ROOT / 'outputs/reports' / base
        for path in sorted((root / 'months').glob('*/*/completion.json')):
            raw = path.read_bytes()
            completion = json.loads(raw)
            if datetime.fromisoformat(completion['at']) > CUTOFF:
                continue
            assert completion['state'] == 'completed' and completion['audit']['passed']
            result_raw = (path.parent / 'result.json').read_bytes()
            assert sha(result_raw) == completion['result_sha256']
            result = json.loads(result_raw)
            days = {d['index']: d for d in result['days']}
            assert set(days) == set(range(30))
            assert all(not d.get('provisional', False) for d in days.values())
            m = completion['month']
            assert abs(sum(days[i]['phi_krw'] for i in range(1, 29)) - m['cost_28d_krw']) < .02
            key = (m['seed'], m['arm'])
            assert key not in runs
            runs[key] = dict(month=m, days=days, inputs=completion['input_hashes'])
            evidence.append(dict(path=path.relative_to(ROOT).as_posix(),
                                 completion_sha256=sha(raw), result_sha256=sha(result_raw)))
    sets = {a: {s for s, b in runs if b == a} for a in ARMS}

    def cohort(seeds, arms):
        assert seeds
        for seed in seeds:
            reference = runs[seed, 'NO_REALLOC']
            for arm in arms:
                r = runs[seed, arm]
                assert r['inputs'] == reference['inputs']
                assert r['month']['requested_identity_sha256'] == reference['month']['requested_identity_sha256']
                assert [r['days'][i]['load'] for i in range(30)] == [reference['days'][i]['load'] for i in range(30)]
        costs = {a: sum(runs[s, a]['month']['cost_28d_krw'] for s in seeds) for a in arms}
        summary, loads, by_seed = {}, defaultdict(dict), []
        for seed in seeds:
            base = runs[seed, 'NO_REALLOC']['month']['cost_28d_krw']
            by_seed.append(dict(seed=seed, savings_percent={
                a: 100 * (base - runs[seed, a]['month']['cost_28d_krw']) / base for a in arms}))
        for arm in arms:
            changes = [row['savings_percent'][arm] for row in by_seed]
            records = [(s, i, runs[s, 'NO_REALLOC']['days'][i], runs[s, arm]['days'][i])
                       for s in seeds for i in range(1, 29)]
            deltas = [b['phi_krw'] - p['phi_krw'] for _, _, b, p in records]
            positive = sum(max(x, 0) for x in deltas)
            summary[arm] = dict(cost_28d_krw=costs[arm],
                aggregate_saving_percent=100 * (costs['NO_REALLOC'] - costs[arm]) / costs['NO_REALLOC'],
                seed_mean_percent=statistics.mean(changes), seed_median_percent=statistics.median(changes),
                seed_min_percent=min(changes), seed_max_percent=max(changes), winning_seeds=sum(x > 0 for x in changes),
                truck_unfinished_30d=sum(runs[s, arm]['month']['censored'] for s in seeds),
                vessel_yard_unfinished_30d=sum(runs[s, arm]['month']['unfinished_vessel_jobs'] for s in seeds),
                positive_daily_savings_krw=positive, daily_losses_krw=-sum(min(x, 0) for x in deltas),
                daily_wins=sum(x > 0 for x in deltas), days=len(deltas),
                top4_share_positive_savings_percent=100 * sum(sorted((max(x, 0) for x in deltas), reverse=True)[:4]) / positive if positive else None,
                components_28d_krw={c: sum(p[c] for _, _, _, p in records) for c in ('c_wait', 'c_move', 'c_rehandle', 'c_vessel')})
            for load in sorted({b['load'] for _, _, b, _ in records}):
                sub = [(b, p) for _, _, b, p in records if b['load'] == load]
                baseline = sum(b['phi_krw'] for b, _ in sub)
                saving = sum(b['phi_krw'] - p['phi_krw'] for b, p in sub)
                space, time = (sum(p[k] for _, p in sub) for k in ('n_space', 'n_time'))
                loads[load][arm] = dict(days=len(sub), baseline_cost_krw=baseline, saving_krw=saving,
                    saving_percent=100 * saving / baseline, mean_daily_saving_krw=saving / len(sub),
                    spatial=space, temporal=time, spatial_share_changes_percent=100 * space / (space + time) if space + time else None)
        return dict(seeds=seeds, summary=summary, by_seed=by_seed, by_load=dict(loads))

    common3 = sorted(sets['NO_REALLOC'] & sets['RL'] & sets['RL_TIME'])
    common4 = sorted(set.intersection(*sets.values()))
    block = sorted(sets['NO_REALLOC'] & sets['RL_SPACE'])
    result = dict(schema='yr317.interim-descriptive-snapshot.v1', cutoff_utc=CUTOFF.isoformat(),
        scope='Completed-run descriptive snapshot only; no confidence intervals, p-values, final aggregation or manuscript changes.',
        limitation='Completion-time selection bias; days within a run are dependent; block cohort differs; costs omit external delay and rescheduling.',
        counts=dict(Counter(a for _, a in runs)), source_files=evidence,
        common_three=cohort(common3, ARMS[:3]), block_pair=cohort(block, ('NO_REALLOC', 'RL_SPACE')),
        common_four=cohort(common4, ARMS) if common4 else None)
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(counts=result['counts'], common3=len(common3), block_pair=len(block), common4=len(common4))))


if __name__ == '__main__':
    main()
