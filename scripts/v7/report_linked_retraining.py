"""YR-332-aq verification and summary: exact refits, per-order recomputation, reductions, paired cost."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import statistics as st

from benchmark_default_eval import save
from six_step_learning import gz_read, sha, load_net, labelset
from retrain_linked_environment import (OUT, CASES, SEEDS, LOADS, RULES, ARMS, name, read, model_path,
                                        previous_folder, eval_seed)
from report_observed_demand import summarize, FLOWS
from report_visit_arrivals import frozen_sources
from report_visit_links import engine_check, visit_metrics
from verify_job_completion import raw_metrics, assert_metrics
from horizon_metrics import aggregate
from yard_rl.v7.evaluation_probe import digest

HASHES = ('candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256', 'records_sha256', 'result_sha256')
POLICIES = (*RULES, *ARMS)


def bootstrap(xs, seed=9923899):
    rng = random.Random(seed);n = len(xs)
    b = sorted(st.mean(rng.choices(xs, k=n)) for _ in range(10000))
    return [b[249], b[9749]]


def refits():
    import torch
    from yard_rl.v7.crane.fit import CraneTrainer
    torch.set_num_threads(1)
    pairs, rows_out = 0, {}
    for seed in SEEDS:
        root = OUT / f'model-{seed}'
        net = load_net(root / 'initial.pt')
        old = load_net(previous_folder(seed) / 'initial.pt').state_dict()
        assert all(torch.equal(v, old[k]) for k, v in net.state_dict().items())
        trainer = CraneTrainer(net, loss_kind='pairwise')
        rows = read(root / 'training.json')['rows'];assert len(rows) == 6
        for r in rows:
            assert sha(r['corpus']) == r['corpus_sha256'] and sha(r['checkpoint']) == r['checkpoint_sha256']
            data = gz_read(r['corpus']);c = data['result']['crane'];pairs += c['labeled']
            assert data['result']['truck_time_contract'] == 'gate-in-to-job-done-v1'
            assert c['factual_mismatch'] == c['force_failed'] == c['hook_errors'] == 0
            fit = trainer.fit(labelset(data), seed=r['input_seed'])
            assert asdict(fit) == r['fit']
            saved = load_net(r['checkpoint']).state_dict()
            assert all(torch.equal(v, saved[k]) for k, v in net.state_dict().items())
        rows_out[str(seed)] = [dict(iteration=r['iteration'], load=r['load'], labeled=r['stats']['labeled'],
            sampled=r['stats']['sampled'], val_order_accuracy=r['fit']['val_order_accuracy'],
            val_pair_loss=r['fit']['val_pair_loss'], median_gap_krw=r['fit']['median_gap_krw'],
            minutes=r['seconds'] / 60) for r in rows]
    return pairs, rows_out


def check_case(case, d, rows, flows):
    arm, seed, load, day, _ = case
    assert d['case'] == list(case) and len(rows) == load and d['input_seed'] == eval_seed(load, day)
    raw = raw_metrics(rows);assert_metrics(raw, d['result'])
    assert raw == d['metrics_by_flow'] and digest(rows) == d['orders_sha256']
    for h in (24, 26):
        assert aggregate(flows, h * 3600.) == d['completion_by_flow'][f'at{h}']
    for row in rows:
        assert row['admitted'] and row['review_count'] == 1 and abs(row['attempt']['review_s'] - row['notice_s']) <= 1e-6
        stamps = [row['times'][k] for k in ('gate_in_s', 'block_in_s', 'service_start_s', 'job_done_s', 'gate_out_s')
                  if row['times'][k] is not None]
        assert all(a <= b + 1e-6 for a, b in zip(stamps, stamps[1:]))
    cp = model_path(arm, seed)
    assert d['checkpoint_sha256'] == (sha(cp) if cp else None)


def verify():
    manifest = read(OUT / 'campaign.json');head = frozen_sources(manifest)
    pairs, training = refits()
    data, cohorts, trucks, jobs, engine, visits = {}, {}, 0, 0, {}, {}
    for case in CASES:
        d, rows, flows = [gz_read(OUT / sub / (name(case) + '.json.gz')) for sub in ('evaluation', 'orders', 'flow-orders')]
        check_case(case, d, rows, flows)
        assert d['visits']['stats']['late_releases'] == d['visits']['stats']['late_exits'] == 0
        engine[name(case)], _ = engine_check(d, rows)
        visits[case] = visit_metrics(rows, d['visits']['plans'])
        key = (case[2], case[3])
        sig = (d['schedule_sha256'], d['probe']['metadata']['input_sha256'])
        assert cohorts.setdefault(key, sig) == sig
        trucks += len(rows);jobs += len(flows);data[case] = d
    repeats = 0
    for case in CASES:
        if case[-1]:
            first, d = data[(*case[:-1], 0)], data[case]
            assert first['result'] == d['result'] and first['orders_sha256'] == d['orders_sha256']
            assert all(first['probe'][k] == d['probe'][k] for k in HASHES)
            repeats += 1
    unique = [d for case, d in data.items() if not case[-1]]
    pick = lambda **kw: [d for d in unique if all(d['case'][i] == v for i, v in
                         ((0, kw.get('arm')), (1, kw.get('seed')), (2, kw.get('load'))) if v is not None)]
    by_arm = {a: summarize(pick(arm=a)) for a in POLICIES}
    by_load = {str(l): {a: summarize(pick(arm=a, load=l)) for a in POLICIES} for l in LOADS}
    by_model = {str(s): {a: summarize(pick(arm=a, seed=s)) for a in ARMS} for s in SEEDS}
    by_model_load = {str(s): {str(l): {a: summarize(pick(arm=a, seed=s, load=l)) for a in ARMS} for l in LOADS} for s in SEEDS}
    red = lambda base, new: 100 * (base - new) / base          # positive = shorter than the baseline
    mean = lambda block, flow='TRUCK': block[flow]['mean_min']
    scopes = [('ALL', by_arm)] + [(str(l), by_load[str(l)]) for l in LOADS]
    reduction = {scope: {f'trained_vs_{b}': {f: red(mean(g[b], f), mean(g['trained'], f)) for f in FLOWS}
                         for b in (*RULES, 'initial', 'previous')} for scope, g in scopes}
    reduction_previous = {scope: {f'previous_vs_{b}': {f: red(mean(g[b], f), mean(g['previous'], f)) for f in FLOWS}
                                  for b in ('SF_SPT', 'initial')} for scope, g in scopes}
    model_reduction = {str(s): {str(l): red(mean(by_model_load[str(s)][str(l)]['initial']),
                                            mean(by_model_load[str(s)][str(l)]['trained'])) for l in LOADS}
                       for s in SEEDS}
    for s in SEEDS:
        model_reduction[str(s)]['ALL'] = red(mean(by_model[str(s)]['initial']), mean(by_model[str(s)]['trained']))
    ranking = {scope: sorted(POLICIES, key=lambda a: mean(g[a])) for scope, g in scopes}
    # Paired total cost over the nine inputs: mean of three retrained models minus SF_SPT.
    from yard_rl.integrated.evalkit import paired
    from yard_rl.experiments.gate_harness import judge_performance
    deltas, paired_rows = [], []
    for load in LOADS:
        for day in range(3):
            base = data[('SF_SPT', 0, load, day, 0)]['result']['phi_krw']
            delta = st.mean(data[('trained', s, load, day, 0)]['result']['phi_krw'] for s in SEEDS) - base
            deltas.append(delta);paired_rows.append(dict(load=load, day=day, delta_phi_krw=delta, baseline_phi_krw=base))
    minimum = by_arm['SF_SPT']['mean_phi_krw'] * .01
    lo, hi = bootstrap(deltas)
    direction = '재학습 정책 비용 증가' if lo > 0 else ('재학습 정책 비용 감소' if hi < 0 else '차이 방향 미확정')
    paired_result = replace(paired(deltas, delta_interest=minimum), ci_lo=lo, ci_hi=hi,
        label=f'짝 재표집 95% 구간: {direction}; 검정력·최소 검출 효과는 기존 짝 비교 근사')
    trained = [d for d in unique if d['case'][0] == 'trained']
    guards = dict(completion=all(d['result']['n_job_censored'] == 0 for d in trained),
        backlog_zero=all(d['completion_by_flow']['at24']['ALL_ORDERS']['unfinished'] == 0 for d in trained),
        physical_valid=all(d['result']['policy_exceptions'] == d['probe']['ledger_audit']['skipped'] == 0 for d in unique),
        vessel_protection=all(d['result']['c_vessel'] <= data[('SF_SPT', 0, d['case'][2], d['case'][3], 0)]['result']['c_vessel'] + 1e-6
                              for d in trained))
    performance = judge_performance(paired_result, minimum_improvement=minimum, interest_effect=minimum, hard_guards=guards)

    def vsum(arm, load=None):
        sel = [visits[c] for c in visits if c[0] == arm and not c[-1] and (load is None or c[2] == load)]
        n = sum(x['visits'] for x in sel)
        return dict(visits=n, mean_gate_to_last_job_min=sum(x['mean_gate_to_last_job_min'] * x['visits'] for x in sel) / n,
                    mean_gate_to_exit_min=sum(x['mean_gate_to_exit_min'] * x['visits'] for x in sel) / n,
                    unfinished_per_run=st.mean(x['unfinished'] for x in sel))
    import xml.etree.ElementTree as ET
    xml = ET.parse(Path('outputs/reports/yr332_visit_links/contract-tests.xml')).getroot()
    assert not xml.findall('.//failure') and not xml.findall('.//error')
    verification = dict(status='PASS', evaluations=len(CASES), unique_evaluations=len(unique), exact_repeats=repeats,
        training_updates=18, training_pairs=pairs, checkpoints_exact_refit=18, initial_weights_equal_ag=True,
        truck_rows=trucks, all_order_rows=jobs, common_inputs=len(cohorts),
        visits_checked=sum(e['visits'] for e in engine.values()),
        stop_arrivals_checked=sum(e['orders_arrived'] for e in engine.values()),
        single_exits_checked=sum(e['single_exits'] for e in engine.values()), late_releases=0, late_exits=0,
        source_commit=head, committed_source_hashes=len(manifest['sources']),
        tests_passed=len(xml.findall('.//testcase')), verification_script_sha256=sha(Path(__file__)))
    summary = dict(by_arm=by_arm, by_load=by_load, by_model=by_model, by_model_load=by_model_load,
        reduction_percent=reduction, previous_model_reduction_percent=reduction_previous,
        trained_vs_initial_by_model_percent=model_reduction, ranking_by_mean_truck_time=ranking,
        visit_metrics={a: vsum(a) for a in POLICIES},
        visit_metrics_by_load={str(l): {a: vsum(a, l) for a in POLICIES} for l in LOADS},
        training=training, paired_cost=asdict(paired_result), paired_rows=paired_rows, performance=performance.as_dict(),
        definition='Per-order gate entry to job completion, censored at 24h; reduction = (baseline - policy)/baseline, positive = shorter.',
        aggregation='Order-weighted means; three models pooled equally on the same nine inputs; repeats excluded.',
        limitations=['synthetic physics; transport 3 units and 470 s remain assumptions (i)',
                     'queue feature counts waiting orders, not trucks (f)', 'day-to-day shape variation not modelled (ao)',
                     '7,500 and 15,000 bunching exceeds the observed range (ap)',
                     'calibration days already viewed; new synthetic inputs are not an independent real period'])
    save(OUT / 'verification.json', verification);save(OUT / 'summary.json', summary)
    print(json.dumps(dict(verification=verification, reduction_ALL={k: round(v['TRUCK'], 2) for k, v in reduction['ALL'].items()},
        by_model=model_reduction, performance=performance.as_dict()['status']), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    verify()
