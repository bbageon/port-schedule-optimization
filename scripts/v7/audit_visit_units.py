"""YR-332-ak read-only audit: order/visit units, repeated gate records, and arrival clustering.

Definitions fixed here (no simulator change):
- Order: one container transaction row. The source column `visit_id` is an order-level key.
- Gate record: rows sharing (truck_hash, gate_in_time).
- Superseded gate record: a gate record with no block entry, job completion or gate exit in any
  row, followed by the same truck's next gate record within MERGE_S seconds. It is a repeated gate
  transaction, not another truck arrival. Raw rows are kept and counted, never deleted.
- Physical visit: the proceeding gate record plus the superseded records chained into it. Its
  arrival is the proceeding record's gate-in time, when the truck actually entered.
The within-hour minute counts of physical visits are written as arrival templates. Statistics
describe clustering; they are not causal attributions and the 13 days are development data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import statistics as st

from benchmark_default_eval import save
from six_step_learning import sha

OUT = Path('outputs/reports/yr332_visit_arrivals')
ROOT = Path('오더데이터')
AJ = Path('outputs/reports/yr332_order_representativeness')
MERGE_S = 1800.0
SENSITIVITY_S = (600.0, 3600.0)
BINS = (10, 30, 60, 120, 300, 600, 900, 1800)
EVENTS = ('enter_block_time', 'job_done_time', 'gate_out_time')
BLOCK = re.compile(r'^[A-Z][0-9]{2}')


def stamp(text):
    return datetime.fromisoformat(text) if text else None


def load_rows():
    files = sorted((ROOT / '기사' / '방문').glob('*.csv'))
    rows = []
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows.extend(csv.DictReader(stream))
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in [ROOT / '기사' / 'index.csv', *files]}
    return rows, manifest


def has_events(rows):
    return any(r[k] for r in rows for k in EVENTS)


def gate_records(entered):
    records = defaultdict(list)
    for r in entered:
        records[(r['truck_hash'], r['gate_in_time'])].append(r)
    return records


def physical_visits(records, merge_s):
    by_truck = defaultdict(list)
    for (truck, gate_in), rows in records.items():
        by_truck[truck].append((stamp(gate_in), rows))
    visits = []
    for truck, items in by_truck.items():
        items.sort(key=lambda x: x[0])
        chain = []
        for i, (t, rows) in enumerate(items):
            chain.append((t, rows))
            nxt = items[i + 1][0] if i + 1 < len(items) else None
            if not has_events(rows) and nxt is not None and (nxt - t).total_seconds() <= merge_s:
                continue
            visits.append(dict(truck=truck, arrival=t, first_scan=chain[0][0], rows=rows,
                               superseded=[(a, b) for a, b in chain[:-1]]))
            chain = []
    return visits


def block_of(row):
    for key in ('enter_block_loc', 'gate_in_loc', 'last_con_loc'):
        m = BLOCK.match(row[key] or '')
        if m:
            return m.group(0)
    return None


def visit_id_structure(rows):
    entered = [r for r in rows if r['gate_in_time']]
    exact = sum(r['visit_id'] == '_'.join([stamp(r['gate_in_time']).strftime('%Y%m%d%H%M%S'),
                r['truck_hash'][:8], r['con_no'], r['in_out']]) for r in entered)
    other = Counter()
    for r in rows:
        if r['gate_in_time']:
            continue
        times = [stamp(r[k]) for k in (*EVENTS, 'cancel_time') if r[k]]
        prefix = r['visit_id'].split('_')[0]
        other['earliest_recorded_event' if times and min(times).strftime('%Y%m%d%H%M%S') == prefix else 'other'] += 1
    return dict(rows=len(rows), unique_visit_id=len({r['visit_id'] for r in rows}), entered=len(entered),
                entered_id_is_gate_time_truck_container_direction=exact, non_entered_id_time=dict(other),
                meaning='visit_id identifies one container transaction (order), not one truck visit')


def superseded_summary(records, visits):
    gaps, same, diff = [], 0, 0
    for v in visits:
        chain = v['superseded'] + [(v['arrival'], v['rows'])]
        for (t, rows), (t2, rows2) in zip(chain, chain[1:]):
            gaps.append((t2 - t).total_seconds())
            if {r['con_hash'] for r in rows} & {r['con_hash'] for r in rows2}:
                same += 1
            else:
                diff += 1
    no_event = sum(not has_events(rows) for rows in records.values())
    return dict(no_event_gate_records=no_event, superseded_records=len(gaps), same_container_as_next=same,
                different_container_from_next=diff,
                gap_le_s={str(x): sum(g <= x for g in gaps) for x in (60, 300, 600, 1800)},
                gap_median_s=st.median(gaps) if gaps else None,
                no_event_records_not_superseded=no_event - len(gaps))


def visit_relations(visits):
    sizes, composition, gate_out, blocks, sequence = Counter(), Counter(), Counter(), Counter(), Counter()
    out_spread = []
    for v in visits:
        rows = v['rows'];sizes[len(rows)] += 1
        if len(rows) < 2:
            continue
        flows = '+'.join(sorted({r['in_out'] for r in rows}))
        composition[f'{len(rows)}:{flows}'] += 1
        outs = {r['gate_out_time'] for r in rows}
        if len(outs) == 1:
            gate_out['same_nonempty' if '' not in outs else 'all_missing'] += 1
        elif outs - {''} and len(outs - {''}) == 1:
            gate_out['same_where_recorded_some_missing'] += 1
        else:
            gate_out['different'] += 1
            t = sorted(stamp(x) for x in outs if x)
            out_spread.append((t[-1] - t[0]).total_seconds())
        bs = [block_of(r) for r in rows]
        blocks[(flows, 'unknown' if None in bs else 'same' if len(set(bs)) == 1 else 'different')] += 1
        if len(rows) != 2 or not all(r['enter_block_time'] and r['job_done_time'] for r in rows):
            continue
        a, b = sorted(rows, key=lambda r: r['in_out'])
        if bs[0] is not None and bs[0] == bs[1]:
            da, db = stamp(a['job_done_time']), stamp(b['job_done_time'])
            sequence[(flows, 'same_block', 'tie' if da == db else f'{a["in_out"]}_done_first' if da < db else f'{b["in_out"]}_done_first')] += 1
        elif None not in bs:
            if stamp(a['job_done_time']) <= stamp(b['enter_block_time']):
                kind = f'{a["in_out"]}_then_{b["in_out"]}'
            elif stamp(b['job_done_time']) <= stamp(a['enter_block_time']):
                kind = f'{b["in_out"]}_then_{a["in_out"]}'
            else:
                kind = 'overlapping'
            sequence[(flows, 'different_block', kind)] += 1
    key = lambda c: {'|'.join(k) if isinstance(k, tuple) else str(k): v for k, v in sorted(c.items())}
    return dict(rows_per_visit=key(sizes), composition=key(composition), gate_out_within_visit=key(gate_out),
                gate_out_spread_max_s=max(out_spread) if out_spread else None, blocks=key(blocks),
                observed_sequence=key(sequence),
                rule='Observed order of recorded times only. Not converted into a precedence constraint.')


def presence_overlap(visits):
    by_truck = defaultdict(list)
    for v in visits:
        ends = [stamp(r[k]) for r in v['rows'] for k in EVENTS if r[k]]
        by_truck[v['truck']].append((v['arrival'], max(ends) if ends else v['arrival']))
    overlaps = 0
    for items in by_truck.values():
        items.sort()
        overlaps += sum(b[0] < a[1] for a, b in zip(items, items[1:]))
    return dict(trucks=len(by_truck), consecutive_visit_overlaps=overlaps)


def cancellations(rows, visits):
    superseded = {id(r) for v in visits for _, rs in v['superseded'] for r in rs}
    multi = {id(r) for v in visits if len(v['rows']) > 1 for r in v['rows']}
    entered = [r for r in rows if r['gate_in_time'] and r['cancel_time']]
    minutes = sorted((stamp(r['cancel_time']) - stamp(r['gate_in_time'])).total_seconds() / 60 for r in entered)
    kind = Counter()
    for r in entered:
        if id(r) in superseded:
            kind['repeated_gate_record'] += 1
        elif r['job_done_time']:
            kind['job_done_then_cancelled_meaning_unknown'] += 1
        elif r['enter_block_time']:
            kind['cancelled_after_block_entry_no_job'] += 1
        elif r['gate_out_time']:
            kind['cancelled_before_block_entry_truck_exited'] += 1
        else:
            kind['cancelled_no_later_event'] += 1
    return dict(entered_cancelled=len(entered), reasons=dict(Counter(r['cancel_reason'] or '(blank)' for r in entered)),
                classes=dict(kind), in_multi_row_visit=sum(id(r) in multi for r in entered),
                minutes_from_gate_in=dict(min=minutes[0], median=st.median(minutes), p90=minutes[int(.9 * len(minutes))], max=minutes[-1]),
                non_entered_cancelled=sum(bool(r['cancel_time']) for r in rows if not r['gate_in_time']),
                handling='Raw rows retained. No physical cancellation is implemented in the synthetic stage; 0.6% of entered rows.')


def marginal_effect(entered, visits):
    aj = json.loads((AJ / 'demand-profiles.json').read_text(encoding='utf-8'))['profiles']['observed_mix_hours']
    pure, superseded = set(), set()
    for v in visits:
        kept = {(r['con_hash'], r['in_out']) for r in v['rows']}
        for _, rs in v['superseded']:
            for r in rs:
                superseded.add(id(r))
                if (r['con_hash'], r['in_out']) in kept:
                    pure.add(id(r))

    def marg(selected):
        hours = [0] * 24
        for r in selected:
            hours[stamp(r['gate_in_time']).hour] += 1
        return dict(n=len(selected), gate_out_share=sum(r['in_out'] == 'OUT' for r in selected) / len(selected), hour_counts=hours)
    base = marg(entered)
    assert base['hour_counts'] == aj['hour_counts'] and math.isclose(base['gate_out_share'], aj['gate_out_share'])
    out = dict(aj_profile_reproduced=True)
    for name, drop in (('without_pure_duplicates', pure), ('without_all_superseded_rows', superseded)):
        m = marg([r for r in entered if id(r) not in drop])
        out[name] = dict(n=m['n'], gate_out_share=m['gate_out_share'],
            gate_out_share_change_pp=100 * (m['gate_out_share'] - base['gate_out_share']),
            max_hour_share_change_pp=max(abs(100 * (a / m['n'] - b / base['n'])) for a, b in zip(m['hour_counts'], base['hour_counts'])))
    out['decision'] = 'aj marginals unchanged in this task; effect size recorded for a separate decision'
    return out


def dispersion(times, bin_s, control=None):
    """Mean over observed (date, hour) strata of var/mean of counts in bin_s bins inside the hour."""
    by = defaultdict(list)
    for t in times:
        by[(t.date().isoformat(), t.hour)].append(t.minute * 60 + t.second)
    nb = 3600 // bin_s
    values = []
    for (day, hour), xs in sorted(by.items()):
        n = len(xs)
        if control == 'poisson':
            rng = random.Random(f'ak-poisson-control:{day}:{hour}')
            xs = [3600 * rng.random() for _ in range(n)]
        elif control == 'stratified':
            rng = random.Random(f'observed-hour-control:{day}:{hour}')
            xs = [3600 * (i + rng.random()) / n for i in range(n)]
        c = Counter(min(nb - 1, int(x // bin_s)) for x in xs)
        values.append(st.pvariance([c[i] for i in range(nb)]) / (n / nb))
    return st.mean(values)


def gap_stats(times):
    rate = Counter((t.date(), t.hour) for t in times)
    by_day = defaultdict(list)
    for t in times:
        by_day[t.date()].append(t)
    norm = []
    for day, xs in by_day.items():
        xs.sort()
        for a, b in zip(xs, xs[1:]):
            norm.append((b - a).total_seconds() * rate[(a.date(), a.hour)] / 3600)
    m, sd = st.mean(norm), st.pstdev(norm)
    return dict(n=len(norm), normalized_mean=m, cv=sd / m, p_lt_0_1=sum(x < .1 for x in norm) / len(norm),
                p_lt_0_5=sum(x < .5 for x in norm) / len(norm), p_gt_3=sum(x > 3 for x in norm) / len(norm),
                exponential_reference=dict(cv=1.0, p_lt_0_1=1 - math.exp(-.1), p_lt_0_5=1 - math.exp(-.5), p_gt_3=math.exp(-3)))


def co_arrival(visits, bin_s=60):
    """Pearson correlation of IN and OUT order counts across bins inside each hour, averaged."""
    by = defaultdict(lambda: ([0] * (3600 // bin_s), [0] * (3600 // bin_s)))
    for v in visits:
        k = (v['arrival'].date(), v['arrival'].hour);i = (v['arrival'].minute * 60 + v['arrival'].second) // bin_s
        for r in v['rows']:
            by[k][0 if r['in_out'] == 'IN' else 1][i] += 1
    return correlation(by.values())


def correlation(series):
    values = []
    for a, b in series:
        if st.pvariance(a) > 0 and st.pvariance(b) > 0:
            ma, mb = st.mean(a), st.mean(b)
            cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
            values.append(cov / math.sqrt(st.pvariance(a) * st.pvariance(b)))
    return dict(hours=len(values), mean_correlation=st.mean(values))


def templates(visits, manifest):
    dates = sorted({v['arrival'].date().isoformat() for v in visits})
    counts = {h: [[0] * 60 for _ in dates] for h in range(24)}
    for v in visits:
        counts[v['arrival'].hour][dates.index(v['arrival'].date().isoformat())][v['arrival'].minute] += 1
    return dict(schema='observed-visit-minute-templates-v1', dates=dates,
        minute_counts={str(h): counts[h] for h in range(24)},
        pooled_visits_by_hour=[sum(map(sum, counts[h])) for h in range(24)],
        definition='Physical truck visits (repeated gate records merged, merge window 1,800 s); arrival = proceeding gate-in time; counts per minute inside each date-hour.',
        scope='Development data 2026-10-28..11-09 already viewed; aggregate counts only, no identifiers.',
        source_sha256=manifest)


def main():
    rows, manifest = load_rows()
    entered = [r for r in rows if r['gate_in_time']]
    records = gate_records(entered)
    visits = physical_visits(records, MERGE_S)
    sensitivity = {str(int(s)): len(physical_visits(records, s)) for s in SENSITIVITY_S}
    unit = dict(visit_id=visit_id_structure(rows), gate_records=len(records), merge_window_s=MERGE_S,
        superseded=superseded_summary(records, visits), physical_visits=len(visits),
        physical_visits_by_merge_window=sensitivity, presence=presence_overlap(visits),
        relations=visit_relations(visits), cancellations=cancellations(rows, visits),
        aj_marginal_effect=marginal_effect(entered, visits),
        metric='Primary truck metric stays per order: gate entry to that job completion. The stage keeps one order per synthetic truck; multi-order visits are not linked.',
        source_sha256=manifest, script_sha256=sha(Path(__file__)))
    arrival = {v['arrival'] for v in visits}
    times = dict(orders=[stamp(r['gate_in_time']) for r in entered], gate_records=[stamp(k[1]) for k in records],
                 visits=[v['arrival'] for v in visits], visits_first_scan=[v['first_scan'] for v in visits])
    process = dict(dispersion={name: {str(b): dispersion(ts, b) for b in BINS} for name, ts in times.items()},
        controls_on_visits={c: {str(b): dispersion(times['visits'], b, c) for b in BINS} for c in ('poisson', 'stratified')},
        visits_by_date_60s={d: dispersion([t for t in times['visits'] if t.date().isoformat() == d], 60)
                            for d in sorted({t.date().isoformat() for t in times['visits']})},
        gaps=gap_stats(times['visits']), co_arrival_60s=co_arrival(visits, 60), co_arrival_600s=co_arrival(visits, 600),
        same_second_visit_pairs=dict(observed=sum(c * (c - 1) // 2 for c in Counter(times['visits']).values()),
            expected_uniform=sum(n * (n - 1) / 2 / 3600 for n in Counter((t.date(), t.hour) for t in times['visits']).values())),
        distinct_visit_arrival_seconds=len(arrival),
        definition='Index = variance/mean of counts in equal bins inside each observed date-hour, averaged over date-hours (aj definition). Poisson control: uniform times with the same counts; stratified control: aj even spacing.',
        script_sha256=sha(Path(__file__)))
    save(OUT / 'visit-unit-audit.json', unit)
    save(OUT / 'arrival-process-audit.json', process)
    save(OUT / 'visit-arrival-templates.json', templates(visits, manifest))
    print(json.dumps(dict(unit={k: unit[k] for k in ('gate_records', 'physical_visits', 'physical_visits_by_merge_window', 'superseded', 'presence')},
        d60={k: round(v['60'], 3) for k, v in process['dispersion'].items()}, gaps=process['gaps']), ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
