"""Read-only four-policy merge; whole months remain the independent samples."""
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from independent_eval_checks import ARMS, SEEDS, month_row, paired_summary, read, save, sha

ALL_ARMS = (*ARMS, 'RL_SPACE')
BOOTSTRAP_SEED = 9_900_723
BOOTSTRAP_SAMPLES = 20_000
INPUT_KEYS = {'schedule_sha256', 'initial_scenarios_sha256', 'vessels_sha256'}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite value in saved evidence')
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _passed(audit):
    return (audit.get('passed') is True and bool(audit.get('checks'))
            and all(v is True for v in audit['checks'].values()))


def _normalized_contract(contract, *, across_seeds=False):
    normalized = deepcopy(contract)
    normalized.pop('label')
    settings = normalized['settings']
    settings.pop('arm')
    if across_seeds:
        for field in ('seed', 'days', 'expected_input'):
            settings.pop(field)
    return normalized


def _validate_days(result):
    days, plan = result['days'], result['plan']
    if (len(days) != 30 or len(plan) != 30
            or [d['index'] for d in days] != list(range(30))
            or [p['index'] for p in plan] != list(range(30))
            or [d['index'] for d in days if d['train']] != list(range(1, 29))):
        raise ValueError('Expected 30 ordered days and the original 28-day window')
    for day, planned in zip(days, plan):
        if (day['load'] != planned['load'] or day['phi_krw'] < 0
                or type(day['train']) is not bool):
            raise ValueError('Daily input or measurement window changed')


def _validate_no_time_changes(folder, result):
    if result['time'] != 0 or any(d['n_time'] != 0 for d in result['days']):
        raise ValueError('RL_SPACE has temporal actions')
    count = 0
    with gzip.open(folder / 'requests.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            original, final = row['requested_arrival_s'], row['final_reserved_arrival_s']
            if (not isinstance(original, (int, float)) or not isinstance(final, (int, float))
                    or not math.isfinite(original) or not math.isfinite(final)
                    or not math.isclose(original, final, rel_tol=0, abs_tol=1e-6)):
                raise ValueError('RL_SPACE request has changed or missing reserved arrival')
            count += 1
    if count != result['request_summary']['requested']:
        raise ValueError('RL_SPACE request count differs from its saved result')


def _validated_run(folder, seed, arm):
    """Check stored passed audits and current hashes without rewriting raw evidence."""
    if (folder / 'failure.json').exists():
        raise ValueError('Completed run still carries a failure marker')
    result, manifest = read(folder / 'result.json'), read(folder / 'manifest.json')
    completion = read(folder / 'completion.json')
    audit = read(folder / 'independent-audit.json')
    for value in (result, manifest, completion, audit):
        _finite(value)
    result_hash = sha(folder / 'result.json')
    contract, repro = manifest['contract'], manifest['repro']
    settings = contract['settings']
    if (completion['state'] != 'completed'
            or completion['result_sha256'] != result_hash
            or audit['result_sha256'] != result_hash):
        raise ValueError('Completion/audit does not match the final result hash')
    if ((result['seed'], result['arm'], result['label']) != (seed, arm, arm)
            or (settings['seed'], settings['arm'], contract['label']) != (seed, arm, arm)):
        raise ValueError('Seed/policy identity disagrees with its directory or contract')
    if (result['repro'] != repro or repro['code']['git_dirty'] is not False
            or not repro['checkpoint_sha256'] or not repro['runner_sha256']):
        raise ValueError('Run provenance/model is missing, dirty, or inconsistent')
    if (settings['days'] != result['plan'] or settings['n_days'] != 30
            or set(settings['expected_input']) != INPUT_KEYS
            or any(not value for value in settings['expected_input'].values())
            or completion['input_hashes'] != settings['expected_input']):
        raise ValueError('Frozen input hashes or monthly plan changed')
    if (not _passed(completion['audit']) or not _passed(audit)
            or not _passed(audit['links']) or not _passed(audit['daily'])
            or not result['recording_checks']
            or any(v is not True for v in result['recording_checks'].values())
            or result['policy_exceptions'] != 0 or result['rollout_calls'] != 0):
        raise ValueError('Recording/model/physical audits did not pass')
    if (result['request_summary']['all_requests_admitted'] is not True
            or result['vessel_work_summary']['unadmitted_moves'] != 0):
        raise ValueError('Requested work was dropped')
    artifacts = {'requests.jsonl.gz': result['request_ledger_sha256'],
                 'container-links.jsonl.gz': result['container_links_sha256']}
    if set(result['daily_artifacts']) != {'operating-state.jsonl.gz', 'daily-final.jsonl'}:
        raise ValueError('Required daily evidence is missing')
    artifacts.update(result['daily_artifacts'])
    checks = [*artifacts.items(), *completion.get('raw_artifact_hashes', {}).items()]
    for name, expected in checks:
        path = folder / name
        if path.parent != folder or sha(path) != expected:
            raise ValueError('Saved raw artifact hash changed: ' + name)
    _validate_days(result)
    row = month_row(result)
    if (completion['month'] != row or row['completed'] > row['requested']
            or any(row[key] < 0 for key in ('completed', 'censored', 'unbound_at_cutoff',
                                           'unfinished_vessel_jobs', 'spatial', 'temporal'))):
        raise ValueError('Completion monthly values differ from final daily data')
    if arm == 'RL_SPACE':
        _validate_no_time_changes(folder, result)
    return row, result, manifest, dict(folder=str(folder), result_sha256=result_hash,
        completion_sha256=sha(folder / 'completion.json'),
        manifest_sha256=sha(folder / 'manifest.json'),
        audit_sha256=sha(folder / 'independent-audit.json'), reused=completion.get('reused', False))


def _index_rows(rows):
    by_seed = {}
    for row in rows:
        _finite(row)
        pair = by_seed.setdefault(row['seed'], {})
        if row['arm'] in pair:
            raise ValueError('Duplicate seed/policy result')
        pair[row['arm']] = row
        if row['cost_28d_krw'] <= 0 or row['cost_30d_krw'] <= 0:
            raise ValueError('Invalid monthly cost')
    if set(by_seed) != set(SEEDS) or any(set(p) != set(ALL_ARMS) for p in by_seed.values()):
        raise ValueError('All 20 months x 4 policies are required; no partial inference')
    if any(len({r['requested_identity_sha256'] for r in pair.values()}) != 1
           for pair in by_seed.values()):
        raise ValueError('Unpaired request inputs across four policies')
    return by_seed


def additional_summary(rows, *, bootstrap_samples=BOOTSTRAP_SAMPLES):
    """New two-comparison family; original three-policy analysis is untouched."""
    import numpy as np
    by_seed = _index_rows(rows)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(SEEDS), size=(bootstrap_samples, len(SEEDS)))
    comparisons = {}
    for scope in ('28d', '30d'):
        field = f'cost_{scope}_krw'
        for base, candidate in (('NO_REALLOC', 'RL_SPACE'), ('RL_SPACE', 'RL')):
            a = np.array([by_seed[s][base][field] for s in SEEDS], dtype=float)
            b = np.array([by_seed[s][candidate][field] for s in SEEDS], dtype=float)
            difference = a - b
            alpha = .025 if scope == '28d' else .05
            low, high = np.quantile(difference[draws].mean(axis=1), [alpha/2, 1-alpha/2])
            comparisons[f'{scope}:{base}-{candidate}'] = dict(
                primary=scope == '28d', family='additional_block_only',
                mean_saving_krw=float(difference.mean()), interval_level=1-alpha,
                mean_saving_interval_krw=[float(low), float(high)],
                ratio_of_sums_saving_pct=float(100*difference.sum()/a.sum()),
                mean_month_saving_pct=float(np.mean(100*difference/a)),
                wins=int(np.sum(difference > 0)), losses=int(np.sum(difference < 0)),
                ties=int(np.sum(difference == 0)), monthly_savings_krw=difference.tolist())
    return dict(comparisons=comparisons, bootstrap_samples=bootstrap_samples,
        bootstrap_seed=BOOTSTRAP_SEED, seed_order=list(SEEDS),
        correction_scope='97.5% intervals protect the two added 28-day comparisons only; '
                         'they do not redefine the original comparison family.',
        independent_unit='one continuous month', descriptive_30d=True)


def _load_summary(results):
    per_month, groups = [], {}
    for result in results:
        local = {}
        for day in result['days']:
            if day['train']:
                local.setdefault(day['load'], []).append(day['phi_krw'])
        for load, costs in sorted(local.items()):
            row = dict(seed=result['seed'], arm=result['arm'], daily_requested_trucks=load,
                       days=len(costs), cost_krw=math.fsum(costs),
                       mean_day_cost_krw=math.fsum(costs)/len(costs))
            per_month.append(row)
            groups.setdefault((load, result['arm']), []).append(row)
    pooled = []
    for (load, arm), rows in sorted(groups.items()):
        cost, days = math.fsum(r['cost_krw'] for r in rows), sum(r['days'] for r in rows)
        pooled.append(dict(daily_requested_trucks=load, arm=arm, months=len(rows),
            days=days, total_cost_krw=cost, pooled_mean_day_cost_krw=cost/days,
            equal_month_mean_day_cost_krw=math.fsum(r['mean_day_cost_krw'] for r in rows)/len(rows)))
    return dict(scope='Descriptive grouping by scheduled daily truck volume within days 1..28. '
        'Daily states carry over; these are not independent day samples or causal load effects.',
        per_month=per_month, pooled=pooled, confidence_intervals=None)


def _report(path, rows, summary):
    lines = ['# Four-policy fixed-model comparison', '',
        '20 independent continuous months, four policies each, no additional training.', '',
        '| Policy | Mean 28-day cost (KRW) | Mean 30-day cost (KRW) | Unfinished trucks | Unfinished vessel jobs |',
        '|---|---:|---:|---:|---:|']
    for arm in ALL_ARMS:
        group = [r for r in rows if r['arm'] == arm]
        lines.append(f"| {arm} | {math.fsum(r['cost_28d_krw'] for r in group)/20:,.0f} | "
            f"{math.fsum(r['cost_30d_krw'] for r in group)/20:,.0f} | "
            f"{sum(r['requested']-r['completed'] for r in group):,} | "
            f"{sum(r['unfinished_vessel_jobs'] for r in group):,} |")
    lines += ['', 'Lower recorded cost alone does not establish improvement when work is unfinished.', '',
        '| Added comparison (positive = saving) | Mean saving (KRW) | Interval | Saving ratio | Wins / losses / ties |',
        '|---|---:|---|---:|---|']
    for name, item in summary['additional_analysis']['comparisons'].items():
        low, high = item['mean_saving_interval_krw']
        lines.append(f"| {name} | {item['mean_saving_krw']:,.0f} | "
            f"{100*item['interval_level']:g}% [{low:,.0f}, {high:,.0f}] | "
            f"{item['ratio_of_sums_saving_pct']:.2f}% | {item['wins']} / {item['losses']} / {item['ties']} |")
    lines += ['', 'Positive baseline-minus-block-only savings show the effect of allowing spatial actions alone.',
        'Positive block-only-minus-full savings show the further effect of allowing temporal actions.',
        'The original three-policy analysis is preserved unchanged in summary.json: original_analysis.',
        'All intervals resample whole months, using 20,000 draws. The two added 28-day comparisons use',
        '97.5% intervals as a separate comparison family; 30-day intervals are descriptive.', '',
        '| Scheduled trucks/day | Baseline mean day cost | Full mean day cost | Time-only mean day cost | Block-only mean day cost |',
        '|---:|---:|---:|---:|---:|']
    loads = summary['load_description']['pooled']
    for load in sorted({r['daily_requested_trucks'] for r in loads}):
        lookup = {r['arm']: r['pooled_mean_day_cost_krw'] for r in loads if r['daily_requested_trucks'] == load}
        lines.append('| ' + ' | '.join([str(load)] + [f'{lookup[a]:,.0f}' for a in ALL_ARMS]) + ' |')
    lines += ['', 'Load groups describe dependent days 1..28, with carryover state; no day-level significance test is made.',
        'The full per-month load-group costs are saved in summary.json. Lower volume does not guarantee low backlog.',
        'All losses and unfinished work remain in the tables. Claim eligibility remains false pending scientific review.',
        'This synthetic experiment adds neither appointment quotas nor external waiting/rescheduling charges.', '']
    if len(lines) > 200:
        raise ValueError('Report exceeds the repository Markdown limit')
    path.write_text('\n'.join(lines), encoding='utf-8')


def merge_runs(primary_dir, block_dir, out_dir):
    """Validate all 80 completed runs, then write a new analysis directory."""
    primary_dir, block_dir, out_dir = map(lambda p: Path(p).resolve(),
                                        (primary_dir, block_dir, out_dir))
    if primary_dir == block_dir or any(out_dir == p or out_dir.is_relative_to(p)
                                      or p.is_relative_to(out_dir) for p in (primary_dir, block_dir)):
        raise ValueError('Analysis output must be separate from both source run directories')
    if out_dir.exists():
        raise ValueError('Use a new output directory; preserve previous analysis')
    found = []
    for source, arms in ((primary_dir, ARMS), (block_dir, ('RL_SPACE',))):
        expected = {(s, a) for s in SEEDS for a in arms}
        actual = set()
        for path in (source / 'months').glob('*/*/completion.json'):
            try:
                key = (int(path.parent.parent.name), path.parent.name)
            except ValueError as exc:
                raise ValueError('Unexpected completed-run directory') from exc
            if key in actual:
                raise ValueError('Duplicate completed-run directory')
            actual.add(key)
        if actual != expected:
            raise ValueError('Missing or unexpected completed runs; require all 20 x 4')
        found.extend((source / 'months' / str(seed) / arm, seed, arm) for seed, arm in sorted(expected))
    rows, results, evidence, per_seed, global_contract, global_repro = [], [], [], {}, None, None
    for folder, seed, arm in found:
        row, result, manifest, provenance = _validated_run(folder, seed, arm)
        contract = _normalized_contract(manifest['contract'])
        common = _normalized_contract(manifest['contract'], across_seeds=True)
        repro = {k: manifest['repro'][k] for k in ('checkpoint_sha256', 'runner_sha256', 'profile_id')}
        if seed in per_seed and per_seed[seed] != contract:
            raise ValueError('Four-policy input/plan/runtime/model contracts differ')
        if global_contract is not None and (global_contract != common or global_repro != repro):
            raise ValueError('Runtime, engine, checkpoint, networks, or execution settings differ across runs')
        per_seed[seed], global_contract, global_repro = contract, common, repro
        rows.append(row)
        # Do not retain all container/event ledgers from 80 month results in RAM.
        results.append(dict(seed=seed, arm=arm, days=[
            {k: day[k] for k in ('train', 'load', 'phi_krw')} for day in result['days']]))
        evidence.append(dict(seed=seed, arm=arm, **provenance))
    rows.sort(key=lambda r: (r['seed'], r['arm']))
    _index_rows(rows)
    original = paired_summary([r for r in rows if r['arm'] in ARMS])
    if (primary_dir / 'summary.json').exists() and read(primary_dir / 'summary.json') != original:
        raise ValueError('Original three-policy analysis changed')
    summary = dict(months=20, policy_runs=80, independent_unit='one continuous month',
        original_analysis=original, additional_analysis=additional_summary(rows),
        load_description=_load_summary(results), evidence=evidence,
        all_requested_work_completed=all(r['completed'] == r['requested']
                                        and r['vessel_all_completed'] for r in rows),
        claim_eligible=False, new_training_runs=0)
    out_dir.mkdir(parents=True, exist_ok=False)
    save(out_dir / 'month-results.json', rows)
    save(out_dir / 'summary.json', summary)
    _report(out_dir / 'results.md', rows, summary)
    return summary
