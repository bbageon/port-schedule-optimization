"""YR-332-ak frozen-policy sensitivity to the arrival process inside each clock hour.

Conditions differ in one input axis only: within-hour gaps.
  observed_mix_hours           aj final input (evenly spaced inside each hour)
  observed_hours_poisson       independent uniform times inside each hour
  observed_hours_visit_minutes observed truck-visit minute counts of the same clock hour
Hourly counts, entry order, flows, blocks, targets, physics, information and policies are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

from benchmark_default_eval import save
from six_step_learning import gz_read, gz_save, load_net, sha
from completion_learning import LOADS, RULES, SEEDS, RUNTIME, PROFILE
from evaluate_observed_demand import POLICIES, checkpoint, read

OUT = Path('outputs/reports/yr332_visit_arrivals')
AJ = Path('outputs/reports/yr332_order_representativeness')
PREREG = Path('docs/research/v7-discharge/visit-arrival-prereg.md')
VARIANTS = ('observed_mix_hours', 'observed_hours_poisson', 'observed_hours_visit_minutes')
INPUT_SEEDS = {load: 9921100 + 10 * i for i, load in enumerate(LOADS)}
CASES = [(variant, arm, seed, load, 0) for variant in VARIANTS for load in LOADS for arm, seed in POLICIES]
CASES += [('observed_hours_poisson', 'SF_SPT', 0, 7500, 1), ('observed_hours_visit_minutes', 'SF_SPT', 0, 7500, 1),
          ('observed_hours_visit_minutes', 'trained', SEEDS[0], 7500, 1)]
BINS = (10, 30, 60, 120, 300, 600, 900, 1800)
# Arrival-derived schedule fields; everything else must equal the aj input.
TIME_FIELDS = {'arrival_s', 'lead_s', 'reserved_s', 'eta_offset_s'}
# Preregistered input criteria at the observed-range load (docs/research/v7-discharge/visit-arrival-prereg.md).
CRITERIA = dict(load=3500, variant='observed_hours_visit_minutes',
                tolerance={'dispersion_60s': .10, 'dispersion_600s': .15, 'gap_cv': .10})
HASHES = ('candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256', 'records_sha256', 'result_sha256')


def name(case):
    return '-'.join(map(str, case))


def profiles():
    aj = read(AJ / 'demand-profiles.json')['profiles']['observed_mix_hours']
    tpl_path = OUT / 'visit-arrival-templates.json'
    tpl = read(tpl_path)
    source = dict(template_path=tpl_path.as_posix(), template_sha256=sha(tpl_path),
                  unit_audit_sha256=sha(OUT / 'visit-unit-audit.json'),
                  definition=tpl['definition'], scope=tpl['scope'])
    out = dict(observed_mix_hours=aj,
        observed_hours_poisson={**aj, 'within_hour': dict(method='poisson',
            source=dict(definition='Independent uniform times inside each clock hour; random gaps without clustering.'))},
        observed_hours_visit_minutes={**aj, 'within_hour': dict(method='observed_visit_minutes',
            dates=tpl['dates'], minute_counts=tpl['minute_counts'], source=source)})
    save(OUT / 'demand-profiles.json', dict(profiles=out, aj_profile_sha256=sha(AJ / 'demand-profiles.json')))


def demand_profile(variant):
    return read(OUT / 'demand-profiles.json')['profiles'][variant]


def profile_sha(variant):
    return hashlib.sha256(json.dumps(demand_profile(variant), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def strata_from_seconds(times):
    by = defaultdict(list)
    for t in times:
        by[int(t // 3600)].append(t - 3600 * int(t // 3600))
    return by


def dispersion(strata, bin_s):
    nb = 3600 // bin_s
    values = []
    for xs in strata.values():
        c = Counter(min(nb - 1, int(x // bin_s)) for x in xs)
        values.append(st.pvariance([c[i] for i in range(nb)]) / (len(xs) / nb))
    return st.mean(values)


def gap_stats(times):
    ts = sorted(times)
    rate = Counter(int(t // 3600) for t in ts)
    norm = [(b - a) * rate[int(a // 3600)] / 3600 for a, b in zip(ts, ts[1:])]
    m, sd = st.mean(norm), st.pstdev(norm)
    return dict(n=len(norm), normalized_mean=m, cv=sd / m, p_lt_0_1=sum(x < .1 for x in norm) / len(norm),
                p_lt_0_5=sum(x < .5 for x in norm) / len(norm), p_gt_3=sum(x > 3 for x in norm) / len(norm))


def co_arrival(schedule, bin_s):
    nb = 3600 // bin_s
    by = defaultdict(lambda: ([0] * nb, [0] * nb))
    for e in schedule:
        h = int(e['arrival_s'] // 3600)
        by[h][0 if e['flow'] == 'GATE_IN' else 1][min(nb - 1, int((e['arrival_s'] - 3600 * h) // bin_s))] += 1
    values = []
    for a, b in by.values():
        if st.pvariance(a) > 0 and st.pvariance(b) > 0:
            ma, mb = st.mean(a), st.mean(b)
            values.append(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / nb / math.sqrt(st.pvariance(a) * st.pvariance(b)))
    return dict(hours=len(values), mean_correlation=st.mean(values))


def input_stats(schedule):
    times = [e['arrival_s'] for e in schedule]
    strata = strata_from_seconds(times)
    return dict(dispersion={str(b): dispersion(strata, b) for b in BINS}, gaps=gap_stats(times),
                co_arrival_60s=co_arrival(schedule, 60), co_arrival_600s=co_arrival(schedule, 600),
                hour_counts=[len(strata.get(h, [])) for h in range(24)])


def build(variant, load):
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    return build_stage(load=load, seed=INPUT_SEEDS[load], profile=build_h21_profile(),
                       information_profile=read(PROFILE), demand_profile=demand_profile(variant))


def preflight():
    from yard_rl.v7.evaluation_probe import digest
    real = read(OUT / 'arrival-process-audit.json')
    target = {'dispersion_60s': real['dispersion']['visits']['60'], 'dispersion_600s': real['dispersion']['visits']['600'],
              'gap_cv': real['gaps']['cv']}
    rows = []
    for load in LOADS:
        base = build('observed_mix_hours', load)
        old = gz_read(AJ / 'evaluation' / f'observed_mix_hours-SF_SPT-0-{load}-0.json.gz')
        assert digest(base['schedule']) == old['schedule_sha256'], 'aj input changed'
        for variant in VARIANTS:
            built = base if variant == 'observed_mix_hours' else build(variant, load)
            schedule = built['schedule']
            assert len(schedule) == load and built['flow_fallbacks_total'] == 0
            assert built['scenarios'] == base['scenarios'] and built['vessel_schedule'] == base['vessel_schedule']
            for a, b in zip(base['schedule'], schedule):
                assert {k: v for k, v in a.items() if k not in TIME_FIELDS} == {k: v for k, v in b.items() if k not in TIME_FIELDS}
                assert int(a['arrival_s'] // 3600) == int(b['arrival_s'] // 3600)
                assert 0 <= b['lead_s'] <= b['arrival_s'] < 86400
            times = [e['arrival_s'] for e in schedule]
            assert times == sorted(times)
            meta = built.get('within_hour_arrivals')
            assert meta is None or meta['cycled_hours'] == 0
            stats = input_stats(schedule)
            row = dict(variant=variant, load=load, seed=INPUT_SEEDS[load], schedule_sha256=digest(schedule),
                       flows=dict(Counter(e['flow'] for e in schedule)), stats=stats,
                       within_hour=meta, profile_sha256=profile_sha(variant))
            if variant == CRITERIA['variant'] and load == CRITERIA['load']:
                observed = {'dispersion_60s': stats['dispersion']['60'], 'dispersion_600s': stats['dispersion']['600'],
                            'gap_cv': stats['gaps']['cv']}
                row['criteria'] = {k: dict(target=target[k], synthetic=observed[k], tolerance=t,
                    passed=abs(observed[k] - target[k]) <= t) for k, t in CRITERIA['tolerance'].items()}
            rows.append(row)
    passed = all(c['passed'] for r in rows for c in r.get('criteria', {}).values())
    save(OUT / 'input-preflight.json', dict(rows=rows, real_visit_targets=target, criteria=CRITERIA,
        criteria_passed=passed, aj_inputs_identical=True, script_sha256=sha(Path(__file__))))
    for r in rows:
        print(r['variant'], r['load'], {b: round(r['stats']['dispersion'][b], 3) for b in ('10', '60', '600', '900')},
              'gapCV', round(r['stats']['gaps']['cv'], 3), 'co60', round(r['stats']['co_arrival_60s']['mean_correlation'], 3))
    print('criteria passed:', passed)


def sources():
    paths = [Path(__file__), PREREG, PROFILE, OUT / 'demand-profiles.json', OUT / 'visit-arrival-templates.json',
             OUT / 'visit-unit-audit.json', OUT / 'arrival-process-audit.json', OUT / 'input-preflight.json',
             AJ / 'demand-profiles.json', *Path('src/yard_rl/v7').rglob('*.py')]
    paths += [Path('scripts/v7') / p for p in ('audit_visit_units.py', 'evaluate_observed_demand.py', 'completion_learning.py',
        'six_step_learning.py', 'horizon_metrics.py', 'audit_admission_ledger.py', 'verify_job_completion.py',
        'benchmark_default_eval.py', 'evaluate_classical_rules.py', 'evaluate_arrival_order.py', 'verify_reservation_contract.py')]
    paths += [checkpoint(arm, seed) for arm, seed in POLICIES if seed]
    return {p.as_posix(): sha(p) for p in paths}


def check():
    for p, h in read(OUT / 'campaign.json')['sources'].items():
        assert sha(p) == h, ('Frozen experiment input changed', p)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no']).strip()
    assert not (OUT / 'campaign.json').exists()
    assert read(OUT / 'input-preflight.json')['criteria_passed'] is True
    config = dict(variants=VARIANTS, policies=POLICIES, cases=CASES, loads=LOADS, evaluation_seeds=INPUT_SEEDS,
        runtime=RUNTIME, primary_seconds=86400, engine_seconds=93600, training=False, teacher_worlds=0,
        time_contract='actual gate entry to job completion, censored at 24h; retain unfinished work',
        selected_policy='all three saved initial and final models; no result-based selection',
        comparisons=['observed_hours_poisson minus observed_mix_hours: random instead of even gaps',
                     'observed_hours_visit_minutes minus observed_hours_poisson: observed clustering instead of random gaps'],
        identity='All observed_mix_hours runs must equal the saved aj evaluations exactly.',
        scope='One input per load; 3,500 is inside the observed hourly range, larger loads stack observed days (assumption).',
        execution_workers=12, information_profile_sha256=sha(PROFILE),
        profile_sha256={v: profile_sha(v) for v in VARIANTS})
    save(OUT / 'campaign.json', dict(configuration=config, sources=sources(),
        repro=repro_stamp(experiment='YR-332-ak-within-hour-arrivals',
            seeds=dict(input=list(INPUT_SEEDS.values()), models=list(SEEDS)), params=config, prereg=PREREG.as_posix())))


def evaluate(index):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    torch.set_num_threads(1)
    check()
    variant, arm, seed, load, repeat = CASES[index]
    cp = checkpoint(arm, seed)
    original, ann = ledger.install_announcer();probe = FlowProbe(ann)
    started = time.perf_counter()
    try:
        ep = episode.run_episode(load=load, seed=INPUT_SEEDS[load], dispatcher='RL_CRANE' if seed else arm,
            crane_net=load_net(cp) if cp else None, information_profile=read(PROFILE),
            demand_profile=demand_profile(variant), evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = ep.as_dict();metrics = raw_metrics(probe.rows);assert_metrics(metrics, result)
    flows = copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'):
            r['finish_s'] = r['yard_done_s']
    completion = {f'at{h}': aggregate(flows, h * 3600.) for h in (24, 26)}
    data = dict(case=CASES[index], input_seed=INPUT_SEEDS[load], profile_sha256=profile_sha(variant),
        result=result, probe=probe.result, metrics_by_flow=metrics, completion_by_flow=completion,
        orders_sha256=digest(probe.rows), schedule_sha256=digest(probe.schedule),
        checkpoint_sha256=sha(cp) if cp else None, seconds=time.perf_counter() - started)
    assert ep.admitted == load and ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert probe.result['ledger_audit']['skipped'] == 0
    assert all(r['review_count'] == 1 for r in probe.rows)
    assert all(abs(r['attempt']['review_s'] - r['notice_s']) <= 1e-6 for r in probe.rows)
    if variant == 'observed_mix_hours':
        old = gz_read(AJ / 'evaluation' / (name(CASES[index]) + '.json.gz'))
        assert result == old['result'] and data['orders_sha256'] == old['orders_sha256']
        assert all(probe.result[k] == old['probe'][k] for k in HASHES) and completion == old['completion_by_flow']
        data['exact_aj_reproduction'] = True
    for sub, value in [('evaluation', data), ('orders', probe.rows), ('flow-orders', flows)]:
        path = OUT / sub / (name(CASES[index]) + '.json.gz');assert not path.exists();gz_save(path, value)
    print(f'{index}: {name(CASES[index])}, truck={metrics["ALL"]["mean_gate_to_job_done_s"] / 60:.3f}min, {data["seconds"]:.1f}s', flush=True)


def run_all():
    check()

    def job(index):
        existing = [(OUT / sub / (name(CASES[index]) + '.json.gz')).exists() for sub in ('evaluation', 'orders', 'flow-orders')]
        if all(existing):
            return dict(index=index, state='already complete')
        if any(existing):
            raise RuntimeError(f'Partial artifacts for {index}; inspect, do not overwrite')
        log = OUT / 'logs' / f'evaluate-{index}.log';log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with log.open('w', encoding='utf-8') as handle:
            p = subprocess.run([sys.executable, __file__, 'evaluate', str(index)], stdout=handle, stderr=subprocess.STDOUT, env=env)
        if p.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8'))
        return dict(index=index, state='complete')
    indices = sorted(range(len(CASES)), key=lambda i: (-CASES[i][3], i))
    done = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(job, i) for i in indices]):
            row = future.result();done.append(row)
            save(OUT / 'progress.json', dict(completed=len(done), total=len(CASES), rows=done))
            print(f'{len(done)}/{len(CASES)} {row}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('profiles', 'preflight', 'prepare', 'evaluate', 'run-all'))
    p.add_argument('index', type=int, nargs='?')
    args = p.parse_args()
    if args.command == 'profiles':profiles()
    elif args.command == 'preflight':preflight()
    elif args.command == 'prepare':prepare()
    elif args.command == 'run-all':run_all()
    else:evaluate(args.index)
