"""Paired statistics and report for a declared campaign (YR-318).

Why a second module: `independent_eval_checks.paired_summary` is the frozen
YR-317-d contract -- 20 seeds, three arms, three comparisons, bootstrap seed
9,900,721. The confirmatory campaign fixes a different design (a fresh seed
band, rule baselines, a declared primary comparison), so it gets its own
function rather than a widened version of the frozen one. Every design element
here arrives as an argument and must come from the campaign's preregistration:
nothing is chosen from the results.
"""
from pathlib import Path


def campaign_summary(rows, *, seeds, arms, comparisons, bootstrap_seed,
                     primary=(), primary_alpha=None, derived=None,
                     bootstrap_samples=20_000, windows=('28d', '30d')):
    """Month-level paired bootstrap over a declared design.

    `comparisons` is a list of (base, candidate) pairs; positive saving means the
    candidate is cheaper. `primary` is the subset of "<window>:<base>-<candidate>"
    names that the preregistration declared confirmatory. The confirmatory family
    splits one 5% error budget across its members (Bonferroni), so k primary
    comparisons each carry a 1 - 0.05/k interval unless `primary_alpha` overrides
    it; every other interval is descriptive at 95%. The repository convention is
    one primary comparison, so k is normally 1.

    `derived` maps a new column name to the columns it averages, per month. It
    exists so that several training runs of one procedure become a single
    estimand -- "what this training procedure produces" -- instead of several
    confirmatory tests of the same claim, which the convention forbids and which
    would invite picking the run that worked.
    """
    import numpy as np

    seeds, arms = tuple(seeds), tuple(arms)
    by_seed = {}
    for row in rows:
        pair = by_seed.setdefault(row['seed'], {})
        if row['arm'] in pair:
            raise ValueError('Duplicate seed/policy result')
        pair[row['arm']] = row
    if set(by_seed) != set(seeds) or any(set(p) != set(arms) for p in by_seed.values()):
        raise ValueError(f'All {len(seeds)} months x {len(arms)} policies are required; '
                         'no partial inference')
    if any(len({r['requested_identity_sha256'] for r in pair.values()}) != 1
           for pair in by_seed.values()):
        raise ValueError('Unpaired request inputs')
    for name, sources in (derived or {}).items():
        if name in arms:
            raise ValueError(f'Derived column {name} collides with a measured arm')
        missing = [item for item in sources if item not in arms]
        if missing:
            raise ValueError(f'Derived column {name} averages unknown arms: {missing}')
        for seed in seeds:
            by_seed[seed][name] = {field: sum(by_seed[seed][item][field] for item in sources)
                                   / len(sources)
                                   for field in (f'cost_{w}_krw' for w in windows)}
    known = set(arms) | set(derived or ())
    for base, candidate in comparisons:
        if base not in known or candidate not in known:
            raise ValueError(f'Comparison {base}-{candidate} names an arm outside the design')

    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    primary = set(primary)
    if primary_alpha is None:
        primary_alpha = .05 / len(primary) if primary else .05
    out = {}
    for window in windows:
        field = f'cost_{window}_krw'
        for base, candidate in comparisons:
            a = np.array([by_seed[s][base][field] for s in seeds], dtype=float)
            b = np.array([by_seed[s][candidate][field] for s in seeds], dtype=float)
            difference = a - b
            if np.any(a <= 0) or not np.all(np.isfinite(difference)):
                raise ValueError('Invalid monthly cost')
            name = f'{window}:{base}-{candidate}'
            alpha = primary_alpha if name in primary else .05
            lo, hi = np.quantile(difference[draws].mean(axis=1), [alpha / 2, 1 - alpha / 2])
            out[name] = dict(primary=name in primary,
                mean_saving_krw=float(difference.mean()), interval_level=1 - alpha,
                mean_saving_interval_krw=[float(lo), float(hi)],
                ratio_of_sums_saving_pct=float(100 * difference.sum() / a.sum()),
                mean_month_saving_pct=float(np.mean(100 * difference / a)),
                median_month_saving_pct=float(np.median(100 * difference / a)),
                wins=int(np.sum(difference > 0)), losses=int(np.sum(difference < 0)),
                ties=int(np.sum(difference == 0)), monthly_savings_krw=difference.tolist())
    if not primary & set(out):
        raise ValueError('The preregistration must name at least one primary comparison')
    completed = all(r['completed'] == r['requested'] and r['vessel_all_completed'] for r in rows)
    return dict(months=len(seeds), policy_runs=len(rows), arms=list(arms),
        independent_unit='one continuous month', bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed, primary_comparisons=sorted(primary),
        primary_alpha=primary_alpha, derived_columns={k: list(v) for k, v in (derived or {}).items()},
        primary_window='day indices 1..28; original paper window',
        secondary_window='all 30 days; same continuous runs', comparisons=out,
        all_requested_work_completed=completed, claim_eligible=False,
        scope='Frozen-model synthetic evaluation on a fresh seed band; external '
              'validity and operational guards remain separate judgments.',
        new_training_runs=0)


def write_campaign_report(path, rows, summary):
    months = summary['months']
    lines = ['# Confirmatory campaign -- fresh seed band, repaired engine', '',
        f"{months} independent continuous months; {len(summary['arms'])} policies per month. "
        'No new training during evaluation.', '',
        '| Policy | Mean 28-day cost (KRW) | Mean 30-day cost (KRW) | '
        'Unfinished trucks (sum) | Unfinished vessel jobs (sum) |',
        '|---|---:|---:|---:|---:|']
    for arm in summary['arms']:
        group = [r for r in rows if r['arm'] == arm]
        lines.append(f"| {arm} | {sum(r['cost_28d_krw'] for r in group)/months:,.0f} | "
            f"{sum(r['cost_30d_krw'] for r in group)/months:,.0f} | "
            f"{sum(r['requested']-r['completed'] for r in group):,} | "
            f"{sum(r['unfinished_vessel_jobs'] for r in group):,} |")
    lines += ['', 'Lower recorded cost alone does not establish improvement when work is '
        'unfinished; read cost per completed truck alongside this table.', '',
        '| Comparison (positive = saving) | Mean saving (KRW) | Interval | '
        'Ratio-of-sums | Median month | Wins |', '|---|---:|---|---:|---:|---:|']
    for name, row in summary['comparisons'].items():
        low, high = row['mean_saving_interval_krw']
        mark = ' **(primary)**' if row['primary'] else ''
        lines.append(f"| {name}{mark} | {row['mean_saving_krw']:,.0f} | "
            f"{row['interval_level']*100:g}% [{low:,.0f}, {high:,.0f}] | "
            f"{row['ratio_of_sums_saving_pct']:.2f}% | {row['median_month_saving_pct']:.2f}% | "
            f"{row['wins']}/{months} |")
    lines += ['', f"Intervals resample whole months ({summary['bootstrap_samples']:,} draws), "
        f"not days, with declared seed {summary['bootstrap_seed']:,}.",
        f"The {len(summary['primary_comparisons'])} primary comparisons split one 5% error "
        f"budget, so each carries a {100*(1-summary['primary_alpha']):g}% interval; "
        'the rest are descriptive.',
        'The rule arms are the non-learning baselines the frozen convention requires; '
        'no-reallocation alone has no discriminating power.',
        'The synthetic environment has no appointment quota and no external waiting or '
        'rescheduling charge.',
        'Per-day observations, all seeds, unsuccessful outcomes, and exact input/model hashes '
        'are retained.', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path
