"""YR-332-al verification and summary from the saved per-order ledgers and visit plans."""
from __future__ import annotations

import json
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from six_step_learning import gz_read, sha
from evaluate_visit_links import (OUT, AK, CASES, VARIANTS, UNLINKED, LINKED, LOADS, RULES, SEEDS, HASHES,
                                  name, read)
from report_observed_demand import summarize, FLOWS
from report_visit_arrivals import frozen_sources, check_case

ARMS = (*RULES, 'initial', 'trained')


def ak_case(case):
    return (UNLINKED, *case[1:])


def load_case(case):
    """Linked runs and the three reruns live here; the other unlinked runs are the saved ak files."""
    root = OUT if case[0] == LINKED or (OUT / 'evaluation' / (name(case) + '.json.gz')).exists() else AK
    return [gz_read(root / sub / (name(case) + '.json.gz')) for sub in ('evaluation', 'orders', 'flow-orders')], root


def engine_check(d, rows):
    """Every visit: stop 0 at gate + travel, stop k at previous stop's last completion + travel, one exit."""
    times = {r['job_id']: r['times'] for r in rows}
    plans = d['visits']['plans']
    stops = arrived = exits = 0
    visit_rows = []
    for vid, p in plans.items():
        done_all = True
        for k, (blk, flow, ids) in enumerate(p['stops']):
            stops += 1
            for i in ids:
                t = times[i]
                assert t['gate_in_s'] == p['gate_in'], vid
                if t['block_in_s'] is None:
                    assert k > 0
                    done_all = False
                    continue
                prev = [times[j]['job_done_s'] for j in p['stops'][k - 1][2]] if k else None
                expect = p['gate_in'] + p['travel'][0] if k == 0 else max(prev) + p['travel'][k]
                assert abs(t['block_in_s'] - expect) <= 1e-6, (vid, k)
                arrived += 1
                done_all &= t['job_done_s'] is not None
        outs = {times[i]['gate_out_s'] for _, _, ids in p['stops'] for i in ids}
        if done_all:
            last = max(times[i]['job_done_s'] for i in p['stops'][-1][2])
            assert outs == {last + p['exit']} and p['gate_out'] == last + p['exit'], vid
            exits += 1
        else:
            assert outs <= {None, p['gate_out']}, vid
        ids = [i for _, _, x in p['stops'] for i in x]
        finish = [times[i]['job_done_s'] for i in ids]
        visit_rows.append(dict(size=len(ids), gate_in=p['gate_in'],
                               last_done=max(finish) if None not in finish else None,
                               gate_out=p['gate_out'] if done_all else None))
    return dict(visits=len(plans), stops=stops, orders_arrived=arrived, single_exits=exits), visit_rows


def visit_metrics(rows, plans, end=86400.):
    """Per visit: gate entry to last job completion and to gate exit, censored at 24h."""
    linked = {i for p in plans.values() for _, _, ids in p['stops'] for i in ids}
    groups = [[i for _, _, ids in p['stops'] for i in ids] for p in plans.values()]
    groups += [[r['job_id']] for r in rows if r['flow'].startswith('GATE_') and r['job_id'] not in linked]
    times = {r['job_id']: r['times'] for r in rows}
    work, turn, unfinished = [], [], 0
    for g in groups:
        a = times[g[0]]['gate_in_s']
        if a is None or a > end:
            continue
        done = [times[i]['job_done_s'] for i in g]
        last = max(done) if None not in done else None
        unfinished += last is None or last > end
        work.append(min(last if last is not None else end, end) - a)
        outs = {times[i]['gate_out_s'] for i in g}
        o = outs.pop() if len(outs) == 1 else None
        turn.append(min(o if o is not None else end, end) - a)
    return dict(visits=len(work), mean_gate_to_last_job_min=st.mean(work) / 60,
                mean_gate_to_exit_min=st.mean(turn) / 60, unfinished=unfinished,
                multi_order_visits=len(plans))


def verify():
    manifest = read(OUT / 'campaign.json');head = frozen_sources(manifest)
    inputs = {(r['variant'], r['load']): r for r in read(OUT / 'input-preflight.json')['rows']}
    data, cohorts, vessels, trucks, jobs, reruns, engine, visits = {}, {}, {}, 0, 0, 0, {}, {}
    all_cases = list(CASES) + [(UNLINKED, arm, seed, load, 0) for load in LOADS
                               for arm, seed in [(r, 0) for r in RULES] + [(a, s) for a in ('initial', 'trained') for s in SEEDS]
                               if (UNLINKED, arm, seed, load, 0) not in CASES]
    for case in all_cases:
        (d, rows, flows), root = load_case(case)
        vessel = check_case(case, d, rows, flows)
        key = (case[0], case[3])
        if case[0] == LINKED:
            assert d['schedule_sha256'] == inputs[key]['schedule_sha256']
            stats = d['visits']['stats']
            assert stats['late_releases'] == stats['late_exits'] == 0
            engine[name(case)], vrows = engine_check(d, rows)
            visits[case] = visit_metrics(rows, d['visits']['plans'])
        else:
            visits[case] = visit_metrics(rows, {})
            if root == OUT:
                assert d['exact_ak_reproduction'];reruns += 1
        sig = d['probe']['metadata']['input_sha256'] if case[0] == LINKED else d['schedule_sha256']
        assert cohorts.setdefault((root.name, *key), sig) == sig
        assert vessels.setdefault(case[3], vessel) == vessel, 'vessel plan must not depend on truck links'
        trucks += len(rows);jobs += len(flows);data[case] = d
    repeats = 0
    for case in CASES:
        if case[-1]:
            first, d = data[(*case[:-1], 0)], data[case]
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
    pct = lambda new, old: 100 * (new - old) / old
    mean = lambda block, flow='TRUCK': block[flow]['mean_min']
    scopes = lambda v: [('ALL', by_variant[v])] + [(str(l), by_load[v][str(l)]) for l in LOADS]
    gaps = {v: {scope: {f'trained_vs_{b}': {flow: pct(mean(grp['trained'], flow), mean(grp[b], flow)) for flow in FLOWS}
                for b in ('SF_SPT', 'initial')} for scope, grp in scopes(v)} for v in VARIANTS}
    change = {scope: {arm: {flow: pct(mean(gb[arm], flow), mean(ga[arm], flow)) for flow in FLOWS} for arm in ARMS}
              for (scope, ga), (_, gb) in zip(scopes(UNLINKED), scopes(LINKED))}
    ranking = {v: {scope: sorted(ARMS, key=lambda a: mean(grp[a])) for scope, grp in scopes(v)} for v in VARIANTS}

    def vsum(variant, arm, load=None):
        sel = [visits[c] for c in visits if c[0] == variant and c[1] == arm and not c[-1] and (load is None or c[3] == load)]
        n = sum(x['visits'] for x in sel)
        return dict(visits=n, mean_gate_to_last_job_min=sum(x['mean_gate_to_last_job_min'] * x['visits'] for x in sel) / n,
                    mean_gate_to_exit_min=sum(x['mean_gate_to_exit_min'] * x['visits'] for x in sel) / n,
                    unfinished_per_run=st.mean(x['unfinished'] for x in sel))
    visit_summary = {v: {a: vsum(v, a) for a in ARMS} for v in VARIANTS}
    visit_by_load = {v: {str(l): {a: vsum(v, a, l) for a in ARMS} for l in LOADS} for v in VARIANTS}
    verification = dict(status='PASS', evaluations=len(CASES), linked_runs=sum(c[0] == LINKED for c in CASES),
        exact_repeats=repeats, ak_exact_reruns=reruns, ak_saved_runs_used=len(all_cases) - len(CASES),
        truck_rows=trucks, all_order_rows=jobs, engine_checks=engine,
        visits_checked=sum(e['visits'] for e in engine.values()),
        stop_arrivals_checked=sum(e['orders_arrived'] for e in engine.values()),
        single_exits_checked=sum(e['single_exits'] for e in engine.values()),
        late_releases=0, late_exits=0, vessel_plans_identical_across_variants=True, source_commit=head,
        committed_source_hashes=len(manifest['sources']), tests_passed=len(xml.findall('.//testcase')),
        verification_script_sha256=sha(Path(__file__)))
    summary = dict(by_variant=by_variant, by_load=by_load, trained_gap_percent=gaps, link_change_percent=change,
        ranking_by_mean_truck_time=ranking, visit_metrics=visit_summary, visit_metrics_by_load=visit_by_load,
        inputs={f'{r["variant"]}:{r["load"]}': r['stats'] for r in read(OUT / 'input-preflight.json')['rows']},
        definition='Primary: per-order gate entry to job completion, censored at 24h. Visit: gate entry to last job and to the single exit. Percent = (new - old)/old, positive = longer.',
        aggregation='Job-count-weighted means. Three frozen models share each input; they are not independent days.',
        scope='3,500 inside the observed range; 7,500 and 15,000 exceed the observed hourly visits (sensitivity).',
        uncertainty='One input per load; no significance test; saved models trained without links.')
    save(OUT / 'verification.json', verification);save(OUT / 'summary.json', summary)
    print(json.dumps(dict(verification={k: v for k, v in verification.items() if k != 'engine_checks'},
        trained_vs_SF_SPT={v: round(gaps[v]['ALL']['trained_vs_SF_SPT']['TRUCK'], 2) for v in VARIANTS},
        trained_vs_initial={v: round(gaps[v]['ALL']['trained_vs_initial']['TRUCK'], 2) for v in VARIANTS},
        change={a: round(change['ALL'][a]['TRUCK'], 2) for a in ARMS}), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    verify()
