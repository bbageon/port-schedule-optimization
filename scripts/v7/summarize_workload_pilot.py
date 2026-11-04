"""Read all preregistered runs; preserve failures and compare fixed time windows."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement(report):
    days = {row['time_s']: row for row in report['telemetry']['days']}
    before, after = days[86400], days[172800]
    cost = after['cost_krw'] - before['cost_krw']
    sensitivity = cost + sum(
        sign * (row['request']['original_request_wait_krw'] - row['request']['actual_gate_wait_krw'])
        for sign, row in ((-1, before), (1, after)))
    jobs = after['cargo']['active_unfinished_jobs']
    trade = report['telemetry']['measured_trades'][0]
    return dict(cost_krw=cost, original_request_cost_sensitivity_krw=sensitivity,
        full_run_cost_krw=report['cost_krw'],
        unfinished_trucks=after['request']['unfinished'],
        unfinished_vessel_jobs=sum(n for f,n in jobs.items() if f.startswith('VESSEL')),
        unfinished_vessels=after['cargo']['active_unfinished_vessels'],
        vessel_idle_cost_krw=after['c_vessel']-before['c_vessel'],
        **trade, cohort=report['telemetry']['measured_cohorts'][0])


def summarize(root, prereg):
    registration = read(prereg)
    pilot = registration['pilot']
    pairs, failures, artifacts = [], [], {}
    for train_seed, eval_seed in zip(pilot['training_seeds'], pilot['evaluation_seeds'], strict=True):
        arms = {}
        for arm in ('cost', 'workload', 'rule'):
            path = root / f'eval-{eval_seed}-{arm}' / 'report.json'
            if not path.exists():
                failure = path.with_name('failure.json')
                failures.append(dict(seed=eval_seed, arm=arm,
                    reason=read(failure) if failure.exists() else 'missing report'))
                continue
            report = read(path)
            artifacts[str(path.relative_to(root))] = digest(path)
            if (report['status'] != 'complete' or not report['invariants'] or report['cf_calls']
                    or report['admission']['skipped'] or report['admission']['vessel_failed']):
                failures.append(dict(seed=eval_seed, arm=arm, reason='execution guard failed'))
            arms[arm] = measurement(report)
            if arm != 'rule':
                train_path = root / f'train-{train_seed}-{arm}' / 'report.json'
                tr = read(train_path)
                artifacts[str(train_path.relative_to(root))] = digest(train_path)
                if tr['optimizer_updates'] != 72 or tr['parameter_l2_change'] <= 0 or tr['status'] != 'complete':
                    failures.append(dict(seed=train_seed, arm=arm, reason='incomplete learning'))
                arms[arm]['training_updates'] = tr['optimizer_updates']
                arms[arm]['training_parameter_l2_change'] = tr['parameter_l2_change']
        if len(arms) != 3:
            continue
        a, b, rule = arms['cost'], arms['workload'], arms['rule']
        improvement = 100*(a['cost_krw']-b['cost_krw'])/a['cost_krw']
        guards = dict(
            original_request_cost=b['original_request_cost_sensitivity_krw'] <= a['original_request_cost_sensitivity_krw'],
            unfinished_trucks=b['unfinished_trucks'] <= a['unfinished_trucks'],
            unfinished_vessel_jobs=b['unfinished_vessel_jobs'] <= a['unfinished_vessel_jobs'],
            vessel_idle=b['vessel_idle_cost_krw'] <= a['vessel_idle_cost_krw'])
        pairs.append(dict(training_seed=train_seed, evaluation_seed=eval_seed, arms=arms,
            cost_reduction_percent=improvement,
            versus_rule_cost_reduction_percent=100*(rule['cost_krw']-b['cost_krw'])/rule['cost_krw'],
            guards=guards, pilot_pair_pass=improvement > 0 and all(guards.values())))
    values = [p['cost_reduction_percent'] for p in pairs]
    mean = statistics.mean(values) if values else None
    interval = None
    if len(values) == 3:
        margin = 4.302652729911275 * statistics.stdev(values) / 3**.5
        interval = [mean-margin, mean+margin]
    passed = len(pairs) == 3 and not failures and all(p['pilot_pair_pass'] for p in pairs)
    return dict(schema='yr331.pilot.result.v1', prereg_sha256=digest(prereg),
        paired_results=pairs, failed_or_missing_runs=failures, artifact_sha256=artifacts,
        mean_cost_reduction_percent=mean, descriptive_t95_interval_percent=interval,
        pilot_success=passed, adoption_authorized=False,
        interpretation='Three-pair diagnostic only; provisional queue cap and synthetic cargo classes. No operational claim.',
        gates=dict(performance='INCONCLUSIVE' if passed else 'FAIL' if len(pairs) == 3 else 'INCONCLUSIVE',
                   reliability='PASS' if len(pairs) == 3 and not failures else 'FAIL',
                   operational_realism='INCONCLUSIVE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prereg', type=Path, default=Path('docs/research/v6-workload-reward/training-prereg.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.prereg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('paired_results', 'artifact_sha256')}, ensure_ascii=False))
