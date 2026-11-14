"""YR-332-am read-only measurement: day-to-day variation, flow-time coupling, volume vs bunching.

Uses the ak physical-visit definition (repeated gate records merged, 1,800 s). No input or
simulator change. Development data 2026-10-28..11-09, already viewed; aggregate values only.

A. Day variation: per-date totals, same-clock-hour CV across dates versus a Poisson reference,
   and 24-hour shape distance versus a multinomial control with each date's own total.
B. Flow-time coupling inside each date-hour: order level (ak 0.336), single-flow visits only,
   and a label-permutation control that keeps the real visit times (bunching) fixed.
C. Volume vs bunching: per date-hour dispersion by visits-per-hour bin and slope; synthetic
   ak inputs and policy sensitivity at 7,500/15,000 are compiled, not judged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics as st

from benchmark_default_eval import save
from six_step_learning import sha
from audit_visit_units import (MERGE_S, correlation, dispersion, gate_records, load_rows,
                               physical_visits, stamp)

OUT = Path('outputs/reports/yr332_day_flow_volume')
AK = Path('outputs/reports/yr332_visit_arrivals')
WEEKEND = {5, 6}
PERMUTATIONS = 50
HOUR_BINS = ((1, 30), (30, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 10 ** 6))


def visits_and_orders():
    rows, manifest = load_rows()
    entered = [r for r in rows if r['gate_in_time']]
    visits = physical_visits(gate_records(entered), MERGE_S)
    return entered, visits, manifest


def tvd(a, b):
    return .5 * sum(abs(x - y) for x, y in zip(a, b))


def day_variation(entered, visits):
    dates = sorted({v['arrival'].date() for v in visits})
    hours = {d: [0] * 24 for d in dates}
    for v in visits:
        hours[v['arrival'].date()][v['arrival'].hour] += 1
    orders = Counter(stamp(r['gate_in_time']).date() for r in entered)
    outs = Counter(stamp(r['gate_in_time']).date() for r in entered if r['in_out'] == 'OUT')
    pooled = [sum(hours[d][h] for d in dates) for h in range(24)]
    total = sum(pooled);share = [x / total for x in pooled]
    per_date = []
    for d in dates:
        n = sum(hours[d]);obs = tvd([x / n for x in hours[d]], share)
        rng = random.Random(f'am:shape:{d.isoformat()}')
        ctrl = []
        for _ in range(200):
            c = Counter(rng.choices(range(24), weights=share, k=n))
            ctrl.append(tvd([c[h] / n for h in range(24)], share))
        per_date.append(dict(date=d.isoformat(), weekday=d.weekday(), weekend=d.weekday() in WEEKEND,
            orders=orders[d], visits=n, gate_out_share=outs[d] / orders[d],
            shape_tvd=obs, shape_tvd_multinomial_mean=st.mean(ctrl),
            shape_tvd_ratio=obs / st.mean(ctrl)))
    cv, excess = [], []
    for h in range(24):
        xs = [hours[d][h] for d in dates];m = st.mean(xs)
        if m > 0:
            cv.append(st.pstdev(xs) / m);excess.append((st.pstdev(xs) / m) / (1 / math.sqrt(m)))
    gos = [p['gate_out_share'] for p in per_date]
    pbar = sum(outs.values()) / sum(orders.values())
    binom = st.mean(math.sqrt(pbar * (1 - pbar) / p['orders']) for p in per_date)
    week = {k: [p['visits'] for p in per_date if p['weekend'] == k] for k in (False, True)}
    return dict(per_date=per_date,
        daily_orders=dict(min=min(orders.values()), max=max(orders.values()), cv=st.pstdev(orders.values()) / st.mean(orders.values())),
        daily_visits=dict(min=min(p['visits'] for p in per_date), max=max(p['visits'] for p in per_date),
                          weekday_mean=st.mean(week[False]), weekend_mean=st.mean(week[True])),
        same_hour_cv=dict(median=st.median(cv), min=min(cv), max=max(cv)),
        same_hour_cv_over_poisson=dict(median=st.median(excess), min=min(excess), max=max(excess)),
        shape_tvd_ratio=dict(median=st.median(p['shape_tvd_ratio'] for p in per_date),
                             min=min(p['shape_tvd_ratio'] for p in per_date), max=max(p['shape_tvd_ratio'] for p in per_date)),
        gate_out_share=dict(min=min(gos), max=max(gos), sd=st.pstdev(gos), binomial_sd=binom, pooled=pbar),
        coverage='Source is a driver/booking event feed; whole-terminal coverage is unconfirmed, so daily totals are not terminal volumes.')


def bins_by_hour(items, bin_s):
    """items: (time, value-tuple index) -> per date-hour pair of count series."""
    nb = 3600 // bin_s
    by = defaultdict(lambda: ([0] * nb, [0] * nb))
    for t, side, weight in items:
        by[(t.date(), t.hour)][side][(t.minute * 60 + t.second) // bin_s] += weight
    return by


def coupling(visits):
    out = {}
    for bin_s in (60, 600):
        orders = [(v['arrival'], 0 if r['in_out'] == 'IN' else 1, 1) for v in visits for r in v['rows']]
        single = []
        labels = defaultdict(list)
        for v in visits:
            flows = {r['in_out'] for r in v['rows']}
            if len(flows) == 1:
                side = 0 if flows == {'IN'} else 1
                single.append((v['arrival'], side, 1))
                labels[(v['arrival'].date(), v['arrival'].hour)].append(side)
        observed = correlation(bins_by_hour(single, bin_s).values())
        perm = []
        for k in range(PERMUTATIONS):
            rng, shuffled = random.Random(f'am:perm:{bin_s}:{k}'), []
            by_hour = defaultdict(list)
            for t, side, w in single:
                by_hour[(t.date(), t.hour)].append(t)
            for key, times in by_hour.items():
                sides = labels[key][:];rng.shuffle(sides)
                shuffled += [(t, s, 1) for t, s in zip(times, sides)]
            perm.append(correlation(bins_by_hour(shuffled, bin_s).values())['mean_correlation'])
        out[str(bin_s)] = dict(orders=correlation(bins_by_hour(orders, bin_s).values()),
            single_flow_visits=observed,
            single_flow_permutation=dict(mean=st.mean(perm), min=min(perm), max=max(perm), n=PERMUTATIONS),
            excess_over_permutation=observed['mean_correlation'] - st.mean(perm))
    mixed = sum(len(v['rows']) for v in visits if len({r['in_out'] for r in v['rows']}) > 1)
    out['orders_in_mixed_visits_share'] = mixed / sum(len(v['rows']) for v in visits)
    out['definition'] = ('Pearson correlation of IN and OUT counts across equal bins inside each date-hour, '
                         'averaged over date-hours (ak co_arrival). Permutation keeps visit times and per-hour '
                         'IN/OUT totals, shuffling only which visit is IN or OUT.')
    return out


def per_hour_dispersion(visits):
    by = defaultdict(list)
    for v in visits:
        by[(v['arrival'].date(), v['arrival'].hour)].append(v['arrival'])
    rows = []
    for key, times in by.items():
        rows.append(dict(n=len(times), d60=dispersion(times, 60), d600=dispersion(times, 600)))
    out = []
    for lo, hi in HOUR_BINS:
        sel = [r for r in rows if lo <= r['n'] < hi]
        if sel:
            out.append(dict(visits_per_hour=f'{lo}-{hi if hi < 10 ** 6 else "max"}', date_hours=len(sel),
                            d60=st.mean(r['d60'] for r in sel), d600=st.mean(r['d600'] for r in sel)))
    big = [r for r in rows if r['n'] >= 60]
    mx, my = st.mean(r['n'] for r in big), st.mean(r['d600'] for r in big)
    slope = (sum((r['n'] - mx) * (r['d600'] - my) for r in big) / sum((r['n'] - mx) ** 2 for r in big))
    my60 = st.mean(r['d60'] for r in big)
    slope60 = (sum((r['n'] - mx) * (r['d60'] - my60) for r in big) / sum((r['n'] - mx) ** 2 for r in big))
    return dict(by_bin=out, max_visits_per_hour=max(r['n'] for r in rows),
                slope_per_100_visits=dict(d600=100 * slope, d60=100 * slope60, date_hours=len(big)),
                definition='Index per date-hour = variance/mean of counts in equal bins (ak definition).')


def synthetic_reference():
    pre = json.loads((AK / 'input-preflight.json').read_text(encoding='utf-8'))
    syn = [dict(variant=r['variant'], load=r['load'], d60=r['stats']['dispersion']['60'],
                d600=r['stats']['dispersion']['600'], gap_cv=r['stats']['gaps']['cv'],
                coupling_60s=r['stats']['co_arrival_60s']['mean_correlation'],
                coupling_600s=r['stats']['co_arrival_600s']['mean_correlation'])
           for r in pre['rows'] if r['variant'] != 'observed_mix_hours']
    s = json.loads((AK / 'summary.json').read_text(encoding='utf-8'))
    return dict(inputs=syn, policy_sensitivity_source=(AK / 'summary.json').as_posix(),
                summary_keys=sorted(s.keys()))


def main():
    entered, visits, manifest = visits_and_orders()
    assert len(visits) == 27415, len(visits)
    data = dict(day_variation=day_variation(entered, visits), flow_time_coupling=coupling(visits),
        volume_bunching=per_hour_dispersion(visits), synthetic=synthetic_reference(),
        visits=len(visits), orders=len(entered), merge_window_s=MERGE_S,
        scope='Development 13 days already used for calibration; descriptive measurement, no input change.',
        source_sha256=manifest, script_sha256=sha(Path(__file__)))
    save(OUT / 'measurement.json', data)
    dv, fc, vb = data['day_variation'], data['flow_time_coupling'], data['volume_bunching']
    print(json.dumps(dict(daily=dv['daily_visits'], same_hour_cv=dv['same_hour_cv'],
        cv_over_poisson=dv['same_hour_cv_over_poisson'], shape=dv['shape_tvd_ratio'], gate_out=dv['gate_out_share'],
        coupling={k: (v['orders']['mean_correlation'], v['single_flow_visits']['mean_correlation'],
                      v['single_flow_permutation']['mean']) for k, v in fc.items() if k in ('60', '600')},
        mixed=fc['orders_in_mixed_visits_share'], bunching=vb['by_bin'], slope=vb['slope_per_100_visits'],
        max_hour=vb['max_visits_per_hour']), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
