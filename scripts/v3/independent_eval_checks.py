"""Read-only checks and month-level summaries for the fixed 20 x 3 evaluation."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ARMS = ('NO_REALLOC', 'RL', 'RL_TIME')
SEEDS = tuple(range(20_000_000, 22_000_000, 100_000))
FROZEN_SCHEMA = 'yr317.independent-evaluation.v1'
DIAGNOSTIC_BAND = 9_900_000


def campaign_specs(cfg):
    """설정이 선언한 (시드, 팔 사양).

    사양 하나가 결과표의 한 열이다: `label`(고유 이름) · `arm`(엔진 팔) ·
    `checkpoint`(가중치). 팔 하나를 **서로 다른 학습 판**으로 여러 열 두려면
    label 만 다르게 준다 — 학습 시드 안정성을 한 캠페인 안에서 재는 방법이다.
    기존 20x3 설계는 글자 하나 못 바꾼다.
    """
    seeds = tuple(cfg['seeds'])
    if cfg.get('schema') == FROZEN_SCHEMA:
        if seeds != SEEDS or tuple(cfg['arms']) != ARMS:
            raise ValueError('Expected the preregistered 20 x 3 design')
        return seeds, tuple(dict(label=a, arm=a, checkpoint=cfg['checkpoint'],
                                 checkpoint_sha256=cfg['checkpoint_sha256']) for a in ARMS)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be a non-empty set of distinct values')
    if any(s // 100_000 * 100_000 == DIAGNOSTIC_BAND for s in seeds):
        raise ValueError('Diagnostic band seeds cannot carry a confirmatory campaign')
    if set(seeds) & set(SEEDS):
        raise ValueError('Seeds already used for judgement cannot be reused')
    raw = cfg.get('arm_specs')
    if raw is None:
        raw = [dict(label=a, arm=a) for a in cfg.get('arms', ())]
    specs = tuple(dict(label=item['label'], arm=item['arm'],
                       checkpoint=item.get('checkpoint', cfg.get('checkpoint')),
                       checkpoint_sha256=item.get('checkpoint_sha256',
                                                  cfg.get('checkpoint_sha256')))
                  for item in raw)
    labels = [item['label'] for item in specs]
    if not specs or len(set(labels)) != len(labels):
        raise ValueError('arms must be a non-empty set of distinct values')
    # A label becomes a directory name under outputs/, on a Windows filesystem
    # reached through WSL. Catch an unusable one now, not six hours into a run.
    import re as _re
    illegal = [item for item in labels if not _re.fullmatch(r'[A-Za-z0-9_.-]{1,48}', item)]
    if illegal:
        raise ValueError(f'Arm labels must be filename-safe: {illegal}')
    import sys
    src = str(Path(__file__).resolve().parents[2] / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    from yard_rl.v3.stage.episode import ARMS as ENGINE_ARMS
    unknown = sorted({item['arm'] for item in specs} - set(ENGINE_ARMS))
    if unknown:
        raise ValueError(f'Unknown arms: {unknown}')
    missing = [item['label'] for item in specs
               if not item['checkpoint'] or not item['checkpoint_sha256']]
    if missing:
        raise ValueError(f'These arms declare no weights: {missing}')
    return seeds, specs


def campaign_contract(cfg):
    """설정이 선언한 (시드, 열 이름). 열 이름은 결과표와 비교쌍이 쓰는 이름이다."""
    seeds, specs = campaign_specs(cfg)
    return seeds, tuple(item['label'] for item in specs)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def verify_links(folder, result):
    path = folder / 'container-links.jsonl.gz'
    seen, issues = set(), []
    count = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['block'], row['target'])
            source = row['source']
            if (key in seen or source is None or row['service_start_s'] is None
                    or source['available_s'] > row['service_start_s'] + 1e-6
                    or row['service_end_s'] < row['service_start_s']):
                issues.append(row['job_id'])
            seen.add(key)
            count += 1
    chain = result['container_flow_summary']
    checks = dict(file_hash=sha(path) == result['container_links_sha256'],
                  removal_count=count == chain['completed_removals'],
                  physical_chain=chain['passed'] is True and not chain['issues'],
                  no_reuse_or_future_supply=not issues,
                  stock_identity=chain['initial_containers'] + chain['produced_containers']
                      - chain['completed_removals'] == chain['remaining_containers'])
    return dict(passed=all(checks.values()), checks=checks, bad_jobs=issues[:10])


def audit_run(folder, *, daily=True):
    from audit_demand_outcomes import audit_folder
    from audit_daily_observation import audit
    result = read(folder / 'result.json')
    outcome = audit_folder(folder)
    # Keep per-request unfinished evidence in its own file, not the month table.
    save(folder / 'outcome-audit.json', outcome)
    links = verify_links(folder, result)
    daily_result = audit(folder) if daily else {'passed': True, 'scope': 'old diagnostic has no daily observer'}
    checks = dict(recording=outcome['passed'], links=links['passed'],
                  daily=daily_result['passed'], runner=all(result['recording_checks'].values()),
                  no_truck_loss=result['request_summary']['all_requests_admitted'] is True,
                  no_vessel_loss=result['vessel_work_summary']['unadmitted_moves'] == 0)
    save(folder / 'independent-audit.json', dict(passed=all(checks.values()), checks=checks,
         links=links, daily=daily_result, result_sha256=sha(folder / 'result.json')))
    return dict(passed=all(checks.values()), checks=checks,
                truck_unfinished=len(outcome['unfinished_trucks']),
                vessel_unfinished=result['vessel_work_summary']['outstanding_yard_jobs'],
                unbound_at_cutoff=outcome['unbound_jobs_at_end'],
                all_trucks_completed=outcome['all_trucks_completed'],
                all_vessels_completed=outcome['all_vessel_work_completed'])


def supply_preflight(run, checkpoint_hash):
    """Require finished, consistent supply evidence; never select by policy cost."""
    if read(run / 'progress.json')['state'] != 'completed':
        raise ValueError('Supply diagnosis is not complete')
    if (run / 'failure.json').exists() or not read(run / 'full/summary.json')['passed']:
        raise ValueError('Supply diagnosis failed')
    results, audits = {}, {}
    for name in ('original_baseline', 'balanced_baseline', 'balanced_full'):
        folder = run / 'full' / name
        result = read(folder / 'result.json')
        manifest = read(folder / 'manifest.json')
        if manifest['repro']['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('Supply diagnosis used another checkpoint')
        audits[name] = audit_run(folder, daily=False)
        results[name] = result
    checks = dict(all_evidence_passed=all(a['passed'] for a in audits.values()),
        matched_requests=len({r['requested_identity_sha256'] for r in results.values()}) == 1,
        balanced_quantity=results['balanced_baseline']['supply_plan_audit']['quantity_feasible'],
        balanced_plan_same=results['balanced_baseline']['supply_plan_audit'] == results['balanced_full']['supply_plan_audit'])
    return dict(passed=all(checks.values()), checks=checks, remaining_work=audits,
        artifacts={name: sha(run / 'full' / name / 'result.json') for name in results},
        limits='Physical input/record checks only. Unfinished work remains an outcome; '
               'this does not certify sustainable operation, performance or real-terminal validity.')


def month_row(result):
    days = result['days']
    measurement = [d for d in days if d['train']]
    return dict(seed=result['seed'], arm=result['arm'],
        cost_28d_krw=math.fsum(d['phi_krw'] for d in measurement),
        cost_30d_krw=math.fsum(d['phi_krw'] for d in days),
        requested=result['request_summary']['requested'],
        completed=result['request_summary']['states'].get('COMPLETED', 0),
        censored=result['request_summary']['states'].get('CENSORED', 0),
        unbound_at_cutoff=result['request_summary']['unbound_jobs_at_end'],
        unfinished_vessel_jobs=result['vessel_work_summary']['outstanding_yard_jobs'],
        vessel_all_completed=result['vessel_work_summary']['all_requested_work_completed'],
        requested_identity_sha256=result['requested_identity_sha256'],
        elapsed_s=result['elapsed_s'], spatial=result['space'], temporal=result['time'])


def smoke_summary(completed):
    """Wiring validity and work completion are separate, including in probes."""
    checks = dict(three_policies=len(completed) == 3
        and {c['month']['arm'] for c in completed} == set(ARMS),
        diagnostic_seed=all(c['month']['seed'] == 9_900_722 for c in completed),
        all_records_valid=bool(completed) and all(c['audit']['passed'] for c in completed))
    return dict(passed=all(checks.values()), checks=checks, runs=completed, independent_runs=0,
        all_work_completed=all(c['audit']['all_trucks_completed'] and
                               c['audit']['all_vessels_completed'] for c in completed),
        scope='Execution, fixed inputs/weights and saved-record checks; remaining work is retained, not discarded.')


def paired_summary(rows, *, bootstrap_samples=20_000):
    """Resample whole months, never the dependent daily observations."""
    import numpy as np
    by_seed = {}
    for row in rows:
        pair = by_seed.setdefault(row['seed'], {})
        if row['arm'] in pair:
            raise ValueError('Duplicate seed/policy result')
        pair[row['arm']] = row
    if set(by_seed) != set(SEEDS) or any(set(p) != set(ARMS) for p in by_seed.values()):
        raise ValueError('All 20 months x 3 policies are required; no partial inference')
    if any(len({r['requested_identity_sha256'] for r in pair.values()}) != 1 for pair in by_seed.values()):
        raise ValueError('Unpaired request inputs')
    rng = np.random.default_rng(9_900_721)
    draws = rng.integers(0, len(SEEDS), size=(bootstrap_samples, len(SEEDS)))
    comparisons = {}
    for scope in ('28d', '30d'):
        field = f'cost_{scope}_krw'
        for base, candidate in (('NO_REALLOC', 'RL'), ('RL_TIME', 'RL'), ('NO_REALLOC', 'RL_TIME')):
            a = np.array([by_seed[s][base][field] for s in SEEDS], dtype=float)
            b = np.array([by_seed[s][candidate][field] for s in SEEDS], dtype=float)
            difference = a - b
            if np.any(a <= 0) or not np.all(np.isfinite(difference)):
                raise ValueError('Invalid monthly cost')
            # 97.5% intervals for the two primary 28-day comparisons form a
            # conservative Bonferroni family; all other intervals are descriptive.
            primary = scope == '28d' and candidate == 'RL'
            alpha = .025 if primary else .05
            lo, hi = np.quantile(difference[draws].mean(axis=1), [alpha/2, 1-alpha/2])
            comparisons[f'{scope}:{base}-{candidate}'] = dict(primary=primary,
                mean_saving_krw=float(difference.mean()), interval_level=1-alpha,
                mean_saving_interval_krw=[float(lo), float(hi)],
                ratio_of_sums_saving_pct=float(100 * difference.sum() / a.sum()),
                mean_month_saving_pct=float(np.mean(100 * difference/a)),
                wins=int(np.sum(difference > 0)), losses=int(np.sum(difference < 0)),
                ties=int(np.sum(difference == 0)), monthly_savings_krw=difference.tolist())
    all_completed = all(r['completed'] == r['requested'] and r['vessel_all_completed'] for r in rows)
    return dict(months=20, policy_runs=60, independent_unit='one continuous month',
        bootstrap_samples=bootstrap_samples, bootstrap_seed=9_900_721,
        primary_window='day indices 1..28; original paper window',
        secondary_window='all 30 days; same continuous runs', comparisons=comparisons,
        all_requested_work_completed=all_completed, claim_eligible=False,
        scope='Frozen-model synthetic evaluation; operational guards and external validity require separate judgment.',
        new_training_runs=0)


def write_report(path, rows, summary):
    lines = ['# Frozen-model independent evaluation', '',
        '20 independent continuous months; three policies per month. No new training.', '',
        '| Policy | Mean 28-day cost (KRW) | Mean 30-day cost (KRW) | Unfinished trucks (sum) | Unfinished vessel jobs (sum) |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        lines.append(f"| {arm} | {sum(r['cost_28d_krw'] for r in group)/20:,.0f} | "
            f"{sum(r['cost_30d_krw'] for r in group)/20:,.0f} | "
            f"{sum(r['requested']-r['completed'] for r in group):,} | "
            f"{sum(r['unfinished_vessel_jobs'] for r in group):,} |")
    lines += ['', 'Lower recorded cost alone does not establish improvement when work is unfinished.', '',
        '| Comparison (positive = saving) | Mean saving (KRW) | Interval | Ratio-of-sums saving | Wins / 20 |',
        '|---|---:|---|---:|---:|']
    for name, r in summary['comparisons'].items():
        low, high = r['mean_saving_interval_krw']
        lines.append(f"| {name} | {r['mean_saving_krw']:,.0f} | "
            f"{r['interval_level']*100:g}% [{low:,.0f}, {high:,.0f}] | "
            f"{r['ratio_of_sums_saving_pct']:.2f}% | {r['wins']} |")
    lines += ['', 'Intervals resample whole months (20,000 draws), not days. Two primary 28-day comparisons use 97.5% intervals.',
        'Full vs time-only measures the contribution of permitting spatial actions with the same fixed networks.',
        'The synthetic environment has no appointment quota or new external waiting/rescheduling charge.',
        'Per-day observations, all seeds, unsuccessful outcomes, and exact input/model hashes are retained.',
        'Final operational-validity and paper-claim judgment remains separate from this computed table.', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
