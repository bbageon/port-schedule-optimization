"""Audit private order CSVs; publish only aggregate timing calibration and hashes."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

OUT = Path('outputs/reports/yr332_real_order_information')


def stamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Source also contains YYYYMMDDH:MM, with an unpadded hour.
        match = re.fullmatch(r'(\d{4})(\d{2})(\d{2})(\d{1,2}):(\d{2})(?::(\d{2}))?', value)
        if match:
            try:
                return datetime(*[int(x or 0) for x in match.groups()])
            except ValueError:
                pass
    return None


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return {'n': 0}
    return dict(n=len(xs), mean=statistics.mean(xs), negative=sum(x < 0 for x in xs),
                **{k: xs[round((len(xs)-1)*q)] for k, q in
                   [('min', 0), ('p10', .1), ('median', .5), ('p90', .9), ('p99', .99), ('max', 1)]})


def read(files):
    return [r for f in files for r in csv.DictReader(f.open(encoding='utf-8-sig', newline=''))]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def audit(root):
    files = sorted(root.rglob('*.csv'))
    manifest = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    visits = read(sorted(root.glob('기사/방문/*.csv')))
    orders = read(sorted(root.glob('기사/오더/*.csv')))
    vessels = read(sorted(root.glob('선박/양적하목록/*.csv')))
    events = read(sorted(root.glob('기사/이벤트/*.csv')))
    assert len(visits) == len({r['visit_id'] for r in visits}), 'duplicate visits'
    physical_keys = [(r['truck_hash'],r['con_hash'],r['gate_in_time'],r['in_out'])
                     for r in visits if r['gate_in_time'] and r['truck_hash'] and r['con_hash']]
    assert len(physical_keys) == len(set(physical_keys)), 'duplicate physical truck/container visits'
    by_doc = defaultdict(list)
    for r in orders:
        t, reserved = stamp(r['log_time']), stamp(r['in_out_reserve_time'])
        # doc_status is retained as a source code, not guessed to mean accepted.
        # Both 1 and 9 occur in records; selection is based on the NM error flag.
        if r['copino_err_type'] == 'NM' and t and reserved and reserved.year >= 2000:
            by_doc[r['doc_key']].append((t, reserved, r['in_out']))
    for values in by_doc.values():
        values.sort()
    cohorts = defaultdict(list)
    rejected = Counter()
    by_day = defaultdict(Counter)
    raw_times = defaultdict(list)
    field_bad = Counter()
    for r in visits:
        for k in ('in_out_reserve_time', 'copino_first_time', 'gate_in_time',
                  'enter_block_time', 'job_done_time', 'gate_out_time'):
            if r[k] and stamp(r[k]) is None:
                field_bad[k] += 1
        a = stamp(r['gate_in_time'])
        if a is None:
            rejected['no_gate_entry'] += 1
            continue
        by_day[a.date().isoformat()][r['in_out']] += 1
        for start, finish in [('gate_in_time', 'enter_block_time'), ('gate_in_time', 'job_done_time'),
                              ('job_done_time', 'gate_out_time'), ('in_out_reserve_time', 'gate_in_time')]:
            s, e = stamp(r[start]), stamp(r[finish])
            if s and e:
                raw_times[start+'->'+finish].append((e-s).total_seconds())
        first = stamp(r['copino_first_time'])
        candidates = sorted({x for key in re.split(r'[|;,]', r['doc_keys'])
                             for x in by_doc.get(key.strip(), [])
                             if x[0] <= a and x[2] == r['in_out']})
        if not candidates:
            rejected['no_matching_pre_entry_valid_reservation'] += 1
            continue
        n, reserved, flow = candidates[0]
        # A prior notice missing from the collected logs is left censored.
        # Do not replace its unknown original booking with a later revision.
        if first is None or n != first:
            rejected['first_notice_not_observed_with_valid_reservation'] += 1
            continue
        split = 'calibration' if a.date().isoformat() <= '2026-11-05' else 'holdout'
        cohorts[(split, flow)].append((int((a-reserved).total_seconds()), int((a-n).total_seconds())))
    profiles, summaries = {}, {}
    for (split, flow), pairs in sorted(cohorts.items()):
        summaries[split+'_'+flow] = dict(error_s=stats([x[0] for x in pairs]),
                                        lead_s=stats([x[1] for x in pairs]))
        if split == 'calibration':
            counts = Counter(pairs)
            profiles['GATE_IN' if flow == 'IN' else 'GATE_OUT'] = dict(
                n=len(pairs), eta_offset_s=statistics.median(x[0] for x in pairs),
                samples=[[e, lead, count] for (e, lead), count in sorted(counts.items())])
    vessel_keys = sorted(set().union(*(r.keys() for r in vessels)))
    vessel_summary = dict(rows=len(vessels), flow=dict(Counter(r['loadDiscGbn'] for r in vessels)),
        status=dict(Counter(r['oprStatus'] for r in vessels)), fields=vessel_keys,
        time_fields={k: sum(bool(r.get(k)) for r in vessels) for k in vessel_keys
                     if any(x in k.lower() for x in ('time', 'date'))},
        limitation='Snapshot rows are not uniquely deduplicated physical moves or timestamped crane cycles; no tractor fleet, handover or return events.')
    profile = dict(schema='empirical-reservation-notice-pairs-v1', all_orders_reserved=True,
        calibration_dates=['2026-10-28', '2026-11-05'], holdout_dates=['2026-11-06', '2026-11-09'],
        selection='first observed NM-flag doc-key-matched reservation (year>=2000), notice equals visit first notice and precedes entry; doc_status not interpreted',
        timezone='source local wall-clock; same-row differences, no timezone conversion',
        error_definition='actual_entry_minus_public_reservation_seconds',
        lead_definition='actual_entry_minus_first_notice_seconds', flows=profiles)
    save(OUT/'source-manifest.json', manifest)
    save(OUT/'profile.json', profile)
    summary = dict(visits=len(visits), distinct_visits=len({r['visit_id'] for r in visits}),
        distinct_physical_keys=len(set(physical_keys)),
        orders=len(orders), events=len(events), visit_status=dict(Counter(r['status'] for r in visits)),
        source_status_codes=dict(Counter(r['doc_status'] for r in orders)),
        reservation_years=dict(Counter(str(stamp(r['in_out_reserve_time']).year)
            for r in visits if stamp(r['in_out_reserve_time']) is not None)),
        coverage={k: sum(bool(r[k]) for r in visits) for k in visits[0] if 'time' in k},
        malformed_times=dict(field_bad), excluded=dict(rejected), by_entry_day=dict(by_day),
        raw_intervals_s={k: stats(v) for k, v in raw_times.items()}, cohorts=summaries, vessels=vessel_summary,
        profile_sha256=hashlib.sha256((OUT/'profile.json').read_bytes()).hexdigest())
    save(OUT/'audit.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('vessels','coverage','by_entry_day')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
