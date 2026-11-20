"""Read-only i/f/ap evidence review; no policy execution or parameter fitting.

Reuses an independently replayed raw audit, checks the current confirmed arrival
sample, and counts physical visits in bc's nine already evaluated input cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs/reports/yr332_transport_queue_recheck'
BB = ROOT / 'outputs/reports/yr332_confirmed_demand'
BC = ROOT / 'outputs/reports/yr332_global_policy'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + '\n', encoding='utf-8', newline='\n')


def schema_inventory():
    out = {}
    for name, subdir in [('driver_events', '기사/이벤트'),
                         ('driver_orders', '기사/방문'),
                         ('vessel_orders', '선박/양적하목록')]:
        files = sorted((ROOT / '오더데이터' / subdir).glob('*.csv'))
        assert files, subdir
        fields = set()
        rows = 0
        for p in files:
            with p.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                fields.update(reader.fieldnames or [])
                rows += sum(1 for _ in reader)
        out[name] = dict(files=len(files), rows=rows, fields=sorted(fields))
    return out


def template_stats(path):
    data = read(path)
    dates = data['dates']
    counts = data['minute_counts']
    assert set(counts) == {str(h) for h in range(24)}
    hours = []
    per_date = Counter()
    for hour in range(24):
        values = counts[str(hour)]
        assert len(values) == len(dates)
        for date, minutes in zip(dates, values):
            assert len(minutes) == 60 and all(isinstance(n, int) and n >= 0 for n in minutes)
            n = sum(minutes)
            hours.append(n)
            per_date[date] += n
    assert sum(hours) == sum(per_date.values())
    return dict(dates=dates, date_count=len(dates), physical_visits=sum(hours),
                max_physical_visits_per_hour=max(hours), hourly_bins=len(hours),
                visits_by_date=dict(sorted(per_date.items())))


def synthetic_stats(observed_max):
    from audit_day_shapes import check_stage
    from global_policy_inputs import bundle, holdout_cases
    from yard_rl.v7.evaluation_probe import digest
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile

    preflight = read(BC / 'input-preflight.json')['cases']
    references = {(r['case']['split'], r['case']['index'], r['case']['load']): r
                  for r in preflight}
    rows = []
    for case in holdout_cases():
        built = build_stage(profile=build_h21_profile(),
                            **bundle().case(case['split'], case['index'], case['load']))
        ref = references[(case['split'], case['index'], case['load'])]
        checks = dict(
            full_input_audit_matches=check_stage(built, case['load']) == ref['audit'],
            scenario_digest_matches=digest(built['scenarios']) == ref['scenarios_sha256'],
            vessel_digest_matches=digest(built['vessel_schedule']) == ref['vessels_sha256'])
        assert all(checks.values()), (case, checks)
        visits = {}
        for entry in built['schedule']:
            key = entry.get('visit', {}).get('id', entry['job_id'])
            arrival = entry['arrival_s']
            assert visits.setdefault(key, arrival) == arrival
        assert len(visits) == built['visit_links']['visits']
        hours = Counter(int(t // 3600) for t in visits.values())
        assert set(hours).issubset(range(24))
        over = {str(h): hours[h] for h in range(24) if hours[h] > observed_max}
        row = dict(case=case, checks=checks, orders=len(built['schedule']),
                   physical_visits=len(visits), visits_by_hour=[hours[h] for h in range(24)],
                   max_physical_visits_per_hour=max(hours.values()),
                   hours_above_observed_max=len(over), over_limit_hours=over,
                   current_within_hour_method=built['demand_profile']['within_hour']['method'])
        rows.append(row)
        print(json.dumps(dict(case=case, visits=len(visits),
                              max_per_hour=row['max_physical_visits_per_hour'],
                              hours_above_observed_max=len(over))), flush=True)
    by_load = {}
    for load in sorted({r['orders'] for r in rows}):
        part = [r for r in rows if r['orders'] == load]
        peaks = [r['max_physical_visits_per_hour'] for r in part]
        by_load[str(load)] = dict(cases=len(part), max_hour_range=[min(peaks), max(peaks)],
            hours_above_observed_max=sum(r['hours_above_observed_max'] for r in part),
            total_hours=24 * len(part))
    return dict(cases=rows, by_daily_order_load=by_load,
                meaning='Input support check only; exceeding this sample is not exceeding terminal capacity.',
                new_policy_evaluations=0, new_training_runs=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    assert out.resolve().is_relative_to(ROOT / 'outputs/reports/yr332_realism_readiness')
    replay = out / 'raw-replay'
    raw = read(replay / 'audit.json')
    assert all(raw['checks'].values())
    stable_names = ['source-manifest.json', 'source-comparison.json',
                    'queue-reconciliation.json', 'vessel-reconciliation.json',
                    'calibration-overlap.json']
    replay_checks = {n: sha(replay / n) == sha(OLD / n) for n in stable_names}
    assert all(replay_checks.values()), replay_checks
    protected = [BB / 'bundle.json', BB / 'demand-base.json', BC / 'bundle.json',
                 BC / 'campaign.json', BC / 'input-preflight.json', BC / 'current-gates.json']
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    schemas = schema_inventory()
    assert schemas['vessel_orders']['rows'] == read(replay / 'vessel-reconciliation.json')['rows']
    confirmed = template_stats(BB / 'confirmed/minute-templates.json')
    legacy = template_stats(BB / 'legacy/minute-templates.json')
    daily = read(BB / 'confirmed/daily.json')
    assert confirmed['physical_visits'] == daily['visits']
    confirmed['container_orders'] = daily['orders']
    assert sum(r['orders'] for r in daily['per_date']) == daily['orders']
    synthetic = synthetic_stats(confirmed['max_physical_visits_per_hour'])
    assert before == {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    summary = dict(schema='yard_rl.realism_readiness.v1', tasks=['YR-332-i', 'YR-332-f', 'YR-332-ap'],
        review_date='2026-11-20', kind='read_only_evidence_review',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_comparison=read(replay / 'source-comparison.json'),
        stable_raw_replay_sha256_equal=replay_checks, raw_audit_checks=raw['checks'],
        schemas=schemas, confirmed_sample=confirmed, legacy_sample=legacy,
        synthetic_input_support=synthetic, protected_files_sha256=before,
        script_sha256=sha(__file__),
        measurement_gaps=dict(
            transport='No internal-vehicle assignment, loaded handoff, empty return and availability time series in supplied schemas; vessel timestamp stages unconfirmed.',
            queue='Explicit service-start timestamp and counting unit/inclusion/measurement-time contract unconfirmed. External truck identifiers do exist.',
            high_volume='No added source files; sample terminal coverage unknown. Observed maximum is not a terminal capacity estimate.'),
        result='NO_NEW_EXTERNAL_EVIDENCE; i/f/ap remain waiting for definitions or operating traces',
        limitations=['The old raw replay retains historical unconfirmed timestamps. ba/bb exclude these from the current calibration.',
                     'Rebuilding frozen synthetic inputs checks support, not real-world performance.',
                     'No new realism PASS, parameter change, training or evaluation claim.'])
    save(out / 'summary.json', summary)
    save(out / 'verification.json', dict(passed=True,
        stable_raw_reports=len(replay_checks), raw_checks=raw['checks'],
        frozen_synthetic_cases=len(synthetic['cases']),
        all_input_digests_match=all(all(r['checks'].values()) for r in synthetic['cases']),
        source_files=summary['source_comparison']['files'],
        confirmed_physical_visits=confirmed['physical_visits'],
        protected_files_unchanged=True, summary_sha256=sha(out / 'summary.json')))
    print(json.dumps(synthetic['by_daily_order_load'], indent=2), flush=True)


if __name__ == '__main__':
    main()
