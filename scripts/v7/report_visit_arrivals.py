"""YR-332-ak verification and summary from the saved per-order ledgers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from six_step_learning import gz_read, sha
from evaluate_visit_arrivals import OUT, AJ, CASES, VARIANTS, LOADS, RULES, SEEDS, HASHES, name, read
from report_observed_demand import independent_flows, summarize, FLOWS
from verify_job_completion import raw_metrics, assert_metrics
from yard_rl.v7.evaluation_probe import digest

ARMS = (*RULES, 'initial', 'trained')


def frozen_sources(manifest):
    head = manifest['repro']['code']['git_head']
    assert manifest['repro']['code']['git_dirty'] is False
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip()).resolve()
    for path, h in manifest['sources'].items():
        assert sha(path) == h, ('current source changed', path)
        blob = subprocess.check_output(['git', 'show', f'{head}:{Path(path).resolve().relative_to(repo).as_posix()}'])
        assert hashlib.sha256(blob).hexdigest() == h, ('frozen committed source differs', path)
    return head


def check_case(case, d, rows, flows):
    assert d['case'] == list(case) and len(rows) == case[3]
    raw = raw_metrics(rows);assert_metrics(raw, d['result'])
    assert raw == d['metrics_by_flow'] and digest(rows) == d['orders_sha256']
    assert len({r['job_id'] for r in flows}) == len(flows)
    for row in rows:
        assert row['admitted'] and row['review_count'] == 1
        assert abs(row['attempt']['review_s'] - row['notice_s']) <= 1e-6
        stamps = [row['times'][k] for k in ('gate_in_s', 'block_in_s', 'service_start_s', 'job_done_s', 'gate_out_s')
                  if row['times'][k] is not None]
        assert all(a <= b + 1e-6 for a, b in zip(stamps, stamps[1:]))
    by_job = {r['job_id']: r for r in flows}
    for row in rows:
        f = by_job[row['job_id']]
        assert f['origin_s'] == row['times']['gate_in_s'] and f['finish_s'] == row['times']['job_done_s']
    for h in (24, 26):
        for flow, values in independent_flows(flows, h * 3600.).items():
            reported = d['completion_by_flow'][f'at{h}'][flow]
            for key, value in values.items():
                assert math.isclose(value, reported[key], rel_tol=1e-12, abs_tol=1e-6), (case, flow, key)
    return digest([{k: r[k] for k in ('job_id', 'block', 'flow', 'origin_s', 'planned_release_s', 'stream')}
                   for r in flows if r['flow'].startswith('VESSEL_')])


def template_structure():
    """Between-date correlation of minute profiles in the same clock hour, with a
    different-hour placebo. Explains why stacking dates adds a common pattern."""
    tpl = read(OUT / 'visit-arrival-templates.json')

    def corr(a, b):
        if st.pvariance(a) == 0 or st.pvariance(b) == 0:
            return None
        ma, mb = st.mean(a), st.mean(b)
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a) / math.sqrt(st.pvariance(a) * st.pvariance(b))
    out = {}
    for width in (1, 10):
        coarse = {h: [[sum(d[i:i + width]) for i in range(0, 60, width)] for d in tpl['minute_counts'][str(h)]] for h in range(24)}
        same = [corr(days[i], days[j]) for days in coarse.values() for i in range(len(days)) for j in range(i + 1, len(days))]
        placebo = [corr(coarse[h][i], coarse[h + 1][(i + 3) % len(coarse[h])]) for h in range(23) for i in range(len(coarse[h]))]
        same, placebo = [x for x in same if x is not None], [x for x in placebo if x is not None]
        out[f'{width}min'] = dict(same_clock_hour_pairs=len(same), same_clock_hour_mean=st.mean(same),
                                  next_hour_other_date_pairs=len(placebo), next_hour_other_date_mean=st.mean(placebo))
    totals = {h: [sum(d) for d in tpl['minute_counts'][str(h)]] for h in range(24)}
    cv = [st.pstdev(v) / st.mean(v) for v in totals.values()]
    out['day_to_day'] = dict(visits_per_date=[sum(totals[h][i] for h in range(24)) for i in range(len(tpl['dates']))],
                             dates=tpl['dates'], clock_hour_cv_median=st.median(cv), clock_hour_cv_range=[min(cv), max(cv)])
    return out


def block_level_check():
    """What each crane sees: arrivals split by block. Real rows use the physical-visit
    arrival time (audit definitions); synthetic rows use the saved SF_SPT inputs (the
    input is identical for every policy). Expected index under random placement with
    fixed counts is 1 - 1/bins."""
    from collections import Counter, defaultdict
    import audit_visit_units as audit

    def index(strata, width):
        bins = 3600 // width;values = []
        for xs in strata.values():
            if len(xs) < 2:
                continue
            c = Counter(min(bins - 1, int(x // width)) for x in xs)
            values.append(st.pvariance([c[i] for i in range(bins)]) / (len(xs) / bins))
        return dict(index=st.mean(values), strata=len(values))
    rows, _ = audit.load_rows()
    visits = audit.physical_visits(audit.gate_records([r for r in rows if r['gate_in_time']]), audit.MERGE_S)
    orders, trucks = defaultdict(list), defaultdict(set)
    for n, v in enumerate(visits):
        t = v['arrival'];x = t.minute * 60 + t.second
        for r in v['rows']:
            b = audit.block_of(r)
            if b:
                orders[(t.date(), t.hour, b)].append(x);trucks[(t.date(), t.hour, b)].add((n, x))
    trucks = {k: [x for _, x in v] for k, v in trucks.items()}
    out = dict(expected_random_placement={'60': 1 - 1 / 60, '600': 1 - 1 / 6},
               real={'orders': {str(w): index(orders, w) for w in (60, 600)},
                     'truck_visits': {str(w): index(trucks, w) for w in (60, 600)}}, synthetic={})
    for variant in VARIANTS:
        for load in LOADS:
            gate, block = defaultdict(list), defaultdict(list)
            for r in gz_read(OUT / 'orders' / f'{variant}-SF_SPT-0-{load}-0.json.gz'):
                t = r['times']['gate_in_s'];gate[(int(t // 3600), r['block'])].append(t % 3600)
            out['synthetic'][f'{variant}:{load}'] = {str(w): index(gate, w) for w in (60, 600)}
    return out


def verify():
    manifest = read(OUT / 'campaign.json');head = frozen_sources(manifest)
    inputs = {(r['variant'], r['load']): r for r in read(OUT / 'input-preflight.json')['rows']}
    data, cohorts, vessels, trucks, jobs, aj_equal = {}, {}, {}, 0, 0, 0
    for case in CASES:
        d = gz_read(OUT / 'evaluation' / (name(case) + '.json.gz'))
        rows = gz_read(OUT / 'orders' / (name(case) + '.json.gz'))
        flows = gz_read(OUT / 'flow-orders' / (name(case) + '.json.gz'))
        vessel = check_case(case, d, rows, flows)
        key = (case[0], case[3])
        assert d['schedule_sha256'] == inputs[key]['schedule_sha256']
        sig = (d['schedule_sha256'], d['probe']['metadata']['input_sha256'])
        assert cohorts.setdefault(key, sig) == sig
        assert vessels.setdefault(case[3], vessel) == vessel, 'vessel plan must not depend on truck gaps'
        if case[0] == 'observed_mix_hours':
            old = gz_read(AJ / 'evaluation' / (name(case) + '.json.gz'))
            assert d['exact_aj_reproduction'] and d['result'] == old['result'] and d['orders_sha256'] == old['orders_sha256']
            assert all(d['probe'][k] == old['probe'][k] for k in HASHES) and d['completion_by_flow'] == old['completion_by_flow']
            aj_equal += 1
        trucks += len(rows);jobs += len(flows);data[case] = d
    repeats = 0
    for case, d in data.items():
        if case[-1]:
            first = data[(*case[:-1], 0)]
            assert first['result'] == d['result'] and first['orders_sha256'] == d['orders_sha256']
            assert all(first['probe'][k] == d['probe'][k] for k in HASHES)
            repeats += 1
    xml = ET.parse(OUT / 'contract-tests.xml').getroot()
    assert not xml.findall('.//failure') and not xml.findall('.//error')
    unique = [d for case, d in data.items() if not case[-1]]
    pick = lambda **kw: [d for d in unique if all(d['case'][i] == v for i, v in
                         ((0, kw.get('variant')), (1, kw.get('arm')), (2, kw.get('seed')), (3, kw.get('load'))) if v is not None)]
    by_variant = {v: {a: summarize(pick(variant=v, arm=a)) for a in ARMS} for v in VARIANTS}
    by_load = {v: {str(l): {a: summarize(pick(variant=v, arm=a, load=l)) for a in ARMS} for l in LOADS} for v in VARIANTS}
    by_model = {v: {str(s): {a: summarize(pick(variant=v, arm=a, seed=s)) for a in ('initial', 'trained')} for s in SEEDS}
                for v in VARIANTS}
    by_model_load = {v: {str(s): {str(l): {a: summarize(pick(variant=v, arm=a, seed=s, load=l)) for a in ('initial', 'trained')}
                     for l in LOADS} for s in SEEDS} for v in VARIANTS}
    pct = lambda new, old: 100 * (new - old) / old
    mean = lambda block, flow='TRUCK': block[flow]['mean_min']
    gaps = {v: {scope: {f'trained_vs_{b}': {flow: pct(mean(grp['trained'], flow), mean(grp[b], flow)) for flow in FLOWS}
                for b in ('SF_SPT', 'initial')}
                for scope, grp in [('ALL', by_variant[v])] + [(str(l), by_load[v][str(l)]) for l in LOADS]} for v in VARIANTS}
    environment = {f'{a}_to_{b}': {scope: {arm: {flow: pct(mean(gb[arm], flow), mean(ga[arm], flow)) for flow in FLOWS} for arm in ARMS}
                   for scope, ga, gb in [('ALL', by_variant[a], by_variant[b])] +
                   [(str(l), by_load[a][str(l)], by_load[b][str(l)]) for l in LOADS]}
                   for a, b in ((VARIANTS[0], VARIANTS[1]), (VARIANTS[1], VARIANTS[2]), (VARIANTS[0], VARIANTS[2]))}
    ranking = {v: {scope: sorted(ARMS, key=lambda a: mean(grp[a])) for scope, grp in
                   [('ALL', by_variant[v])] + [(str(l), by_load[v][str(l)]) for l in LOADS]} for v in VARIANTS}
    models = {v: {str(s): {str(l): pct(mean(by_model_load[v][str(s)][str(l)]['trained']), mean(by_model_load[v][str(s)][str(l)]['initial']))
              for l in LOADS} for s in SEEDS} for v in VARIANTS}
    verification = dict(status='PASS', evaluations=len(CASES), unique_evaluations=len(unique), exact_repeats=repeats,
        aj_exact_reproductions=aj_equal, truck_rows=trucks, all_order_rows=jobs, common_inputs=len(cohorts),
        vessel_plans_identical_across_variants=True, source_commit=head, committed_source_hashes=len(manifest['sources']),
        tests_passed=len(xml.findall('.//testcase')), verification_script_sha256=sha(Path(__file__)))
    summary = dict(by_variant=by_variant, by_load=by_load, by_model=by_model, by_model_load=by_model_load,
        trained_gap_percent=gaps, environment_change_percent=environment, ranking_by_mean_truck_time=ranking,
        trained_vs_initial_by_model_load_percent=models, template_structure=template_structure(),
        block_level=block_level_check(),
        definition='Truck gate entry to job completion; vessel stream planned start to yard completion; unfinished censored at 24h. Percent = (new - old)/old, positive = longer.',
        aggregation='Job-count-weighted means. Three frozen models share each input; they are not independent days.',
        load_weight={str(l): l / sum(LOADS) for l in LOADS},
        scope='3,500 observed-range input (one disclosed gap-variability miss); 7,500 and 15,000 stack observed days: sensitivity with strong bunching, not calibration.',
        uncertainty='One input per load, no significance test; diagnostic only. The 13 observed days are development data.')
    save(OUT / 'verification.json', verification);save(OUT / 'summary.json', summary)
    print(json.dumps(dict(verification=verification, trained_vs_SF_SPT={v: round(gaps[v]['ALL']['trained_vs_SF_SPT']['TRUCK'], 2) for v in VARIANTS},
        trained_vs_initial={v: round(gaps[v]['ALL']['trained_vs_initial']['TRUCK'], 2) for v in VARIANTS}), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    verify()
