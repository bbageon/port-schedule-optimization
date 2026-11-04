"""Independently recompute truck money/time from frozen evaluation event records."""
import argparse
import gzip
import json
from pathlib import Path


def cost(seconds):
    seconds = max(0.0, seconds)
    return 40000/3600 * (seconds + max(0.0, seconds-3600))


def recompute(rows, time_s, origin):
    total, requested, done = 0.0, 0, 0
    for row in rows:
        start = row[origin]
        if start is None or start > time_s:
            continue
        requested += 1
        end = min(time_s, row['gate_out_s']) if row['gate_out_s'] is not None else time_s
        total += cost(end-start)
        done += row['gate_out_s'] is not None and row['gate_out_s'] <= time_s
    return dict(cost_krw=total, requested=requested, completed=done)


def audit(root):
    results = []
    for report_path in sorted(root.glob('eval-*/report.json')):
        record_path = report_path.with_name('execution-records.json.gz')
        if not record_path.exists():
            results.append(dict(run=report_path.parent.name, status='not_collected_in_earlier_reference_build'))
            continue
        report = json.loads(report_path.read_text(encoding='utf-8'))
        with gzip.open(record_path, 'rt', encoding='utf-8') as source:
            rows = json.load(source)
        errors = []
        times = [x for x in report['telemetry']['days'] if x['time_s'] in (86400, 172800, 259200)]
        for row in times:
            current = recompute(rows, row['time_s'], 'gate_in_s')
            original = recompute(rows, row['time_s'], 'original_requested_gate_s')
            errors.extend([abs(current['cost_krw']-row['request']['actual_gate_wait_krw']),
                           abs(original['cost_krw']-row['request']['original_request_wait_krw'])])
            if original['requested'] != row['request']['requested'] or original['completed'] != row['request']['completed']:
                raise AssertionError(f'{report_path}: requested/completed counts differ')
        lifecycle_errors = 0
        for row in rows:
            observed = [row[k] for k in ('gate_in_s', 'block_in_s', 'service_start_s', 'job_done_s', 'gate_out_s') if row[k] is not None]
            lifecycle_errors += any(a > b + 1e-7 for a,b in zip(observed, observed[1:]))
        max_error = max(errors, default=0)
        if max_error > .01 or lifecycle_errors:
            raise AssertionError(f'{report_path}: money/lifecycle audit failed')
        results.append(dict(run=report_path.parent.name, status='passed', rows=len(rows),
                            max_money_error_krw=max_error, lifecycle_errors=lifecycle_errors))
    return dict(results=results, all_available_records_passed=True,
                limitation='The two earlier rule-reference builds have aggregate reports and completed-job logs but no truck event export.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
