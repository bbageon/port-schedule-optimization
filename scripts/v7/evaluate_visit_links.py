"""YR-332-al frozen-policy sensitivity to linking multi-container truck visits.

One input axis changes: orders of one truck visit share a gate time, stops run in order
(inbound first), and the truck exits once. Hourly counts, order attributes, the ak within-hour
visit template, physics, information and policies are fixed. Prereg:
docs/research/v7-discharge/visit-link-prereg.md
  observed_hours_visit_minutes  ak condition 3 (unlinked); saved ak results, 3 exact reruns
  observed_visit_links          condition 3 + observed visit links
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
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

OUT = Path('outputs/reports/yr332_visit_links')
AK = Path('outputs/reports/yr332_visit_arrivals')
PREREG = Path('docs/research/v7-discharge/visit-link-prereg.md')
UNLINKED, LINKED = 'observed_hours_visit_minutes', 'observed_visit_links'
VARIANTS = (UNLINKED, LINKED)
INPUT_SEEDS = {load: 9921100 + 10 * i for i, load in enumerate(LOADS)}
DIST_SEEDS = tuple(range(9921800, 9921820))
CASES = [(LINKED, arm, seed, load, 0) for load in LOADS for arm, seed in POLICIES]
CASES += [(UNLINKED, 'SF_SPT', 0, 3500, 0), (UNLINKED, 'trained', SEEDS[0], 3500, 0), (UNLINKED, 'SF_SPT', 0, 7500, 0),
          (LINKED, 'SF_SPT', 0, 7500, 1), (LINKED, 'trained', SEEDS[0], 7500, 1)]
HASHES = ('candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256', 'records_sha256', 'result_sha256')
TIME_FIELDS = {'arrival_s', 'lead_s', 'reserved_s', 'eta_offset_s', 'gate_in_s'}
CRITERIA = dict(load=3500, tolerance=dict(multi_order_share=.05, block_orders_60s=.10, gate_visits_60s=.10,
                                          gate_visits_600s=.15, multi_same_block_share=.10, multi_mixed_flow_share=.10))


def name(case):
    return '-'.join(map(str, case))


def profiles():
    ak = read(AK / 'demand-profiles.json')['profiles'][UNLINKED]
    spec = read(OUT / 'link-definition.json')['spec']
    spec = {**spec, 'source': {**spec['source'], 'audit_sha256': sha(OUT / 'link-definition.json')}}
    save(OUT / 'demand-profiles.json', dict(profiles={UNLINKED: ak, LINKED: {**ak, 'visit_link': spec}},
        ak_profile_sha256=sha(AK / 'demand-profiles.json'),
        identity='The unlinked profile is the ak condition-3 profile byte for byte.'))


def demand_profile(variant):
    return read(OUT / 'demand-profiles.json')['profiles'][variant]


def profile_sha(variant):
    return hashlib.sha256(json.dumps(demand_profile(variant), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build(variant, load, seed=None):
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    return build_stage(load=load, seed=INPUT_SEEDS[load] if seed is None else seed, profile=build_h21_profile(),
                       information_profile=read(PROFILE), demand_profile=demand_profile(variant))


def index(strata, width):
    bins = 3600 // width
    values = []
    for xs in strata.values():
        if len(xs) < 2:
            continue
        c = Counter(min(bins - 1, int(x // width)) for x in xs)
        values.append(st.pvariance([c[i] for i in range(bins)]) / (len(xs) / bins))
    return st.mean(values)


def visit_of(e):
    return e['visit']['id'] if 'visit' in e else e['job_id']


def input_stats(built):
    """ak index definitions: gate level (orders / visits) and per block (orders / visits)."""
    go, gv, bo, bv = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(set)
    first = {}
    for e in built['schedule']:
        t = e['arrival_s'];h = int(t // 3600);x = t - 3600 * h
        go[h].append(x);bo[(h, e['block'])].append(x);bv[(h, e['block'])].add((visit_of(e), x))
        first.setdefault(visit_of(e), t)
    for t in first.values():
        gv[int(t // 3600)].append(t - 3600 * int(t // 3600))
    bv = {k: [x for _, x in v] for k, v in bv.items()}
    multi = defaultdict(list)
    for e in built['schedule']:
        if 'visit' in e:
            multi[e['visit']['id']].append(e)
    n_multi = max(1, len(multi))
    return dict(visits=len(first), orders=len(built['schedule']),
        multi_order_share=sum(map(len, multi.values())) / len(built['schedule']),
        multi_visits=len(multi), multi_same_block_share=sum(len({e['block'] for e in v}) == 1 for v in multi.values()) / n_multi,
        multi_mixed_flow_share=sum(len({e['flow'] for e in v}) == 2 for v in multi.values()) / n_multi,
        max_visits_per_hour=max(Counter(int(t // 3600) for t in first.values()).values()),
        **{f'{k}_{w}s': index(s, w) for k, s in (('gate_orders', go), ('gate_visits', gv),
                                                  ('block_orders', bo), ('block_visits', bv)) for w in (60, 600)})


def rule_checks(built):
    from yard_rl.v7.world.domain.enums import ContainerSize
    boxes = {}
    for scn in built['scenarios'].values():
        boxes.update({k: c.size for k, c in scn.containers.items()})
    groups = defaultdict(list)
    for e in built['schedule']:
        if 'visit' in e:
            groups[e['visit']['id']].append(e)
    capacity = order = time_split = stop_mismatch = 0
    for rows in groups.values():
        v = rows[0]['visit']
        time_split += len({e['arrival_s'] for e in rows}) != 1
        flows = [s[1] for s in v['stops']]
        order += flows != sorted(flows, key=lambda f: f != 'GATE_IN')
        for flow in ('GATE_IN', 'GATE_OUT'):
            sel = [e for e in rows if e['flow'] == flow]
            if len(sel) > 2:
                capacity += 1
            elif len(sel) == 2:
                big = [e['size_ft40'] if flow == 'GATE_IN' else boxes[e['target']] != ContainerSize.FT20 for e in sel]
                capacity += any(big)
        by_id = {e['job_id']: e for e in rows}
        stop_mismatch += any(by_id[i]['block'] != b or by_id[i]['flow'] != f or by_id[i]['visit']['stop'] != k
                             for k, (b, f, ids) in enumerate(v['stops']) for i in ids)
    return dict(capacity_violations=capacity, order_violations=order, split_visit_times=time_split,
                stop_mismatches=stop_mismatch)


def preflight():
    from yard_rl.v7.evaluation_probe import digest
    real = read(OUT / 'link-definition.json')
    target = dict(multi_order_share=real['orders_in_multi_share'], block_orders_60s=real['clustering_targets']['block_orders_60s'],
                  gate_visits_60s=real['clustering_targets']['gate_visits_60s'],
                  gate_visits_600s=real['clustering_targets']['gate_visits_600s'],
                  multi_same_block_share=real['multi_same_block_share'], multi_mixed_flow_share=real['multi_mixed_flow_share'])
    saved = {(r['variant'], r['load']): r['schedule_sha256'] for r in read(AK / 'input-preflight.json')['rows']}
    rows = []
    for load in LOADS:
        base = build(UNLINKED, load)
        assert base['bnct_gate_in_resynced'] == load
        new = build(LINKED, load)
        assert len(new['schedule']) == load and new['flow_fallbacks_total'] == 0
        assert new['scenarios'] == base['scenarios'] and new['vessel_schedule'] == base['vessel_schedule']
        for a, b in zip(base['schedule'], new['schedule']):
            assert {k: v for k, v in a.items() if k not in TIME_FIELDS} == \
                   {k: v for k, v in b.items() if k not in TIME_FIELDS | {'visit'}}
            assert int(a['arrival_s'] // 3600) == int(b['arrival_s'] // 3600)
            assert 0 <= b['lead_s'] <= b['arrival_s'] < 86400 and b['gate_in_s'] == b['arrival_s']
        meta = new['visit_links']
        assert meta['within_hour']['cycled_hours'] == 0
        checks = rule_checks(new)
        assert not any(checks.values()), checks
        for variant, built in ((UNLINKED, base), (LINKED, new)):
            row = dict(variant=variant, load=load, seed=INPUT_SEEDS[load], schedule_sha256=digest(built['schedule']),
                       stats=input_stats(built), profile_sha256=profile_sha(variant))
            if variant == LINKED:
                row.update(rule_checks=checks, link_meta={k: v for k, v in meta.items() if k != 'within_hour'},
                           within_hour=meta['within_hour'])
                if load == CRITERIA['load']:
                    row['criteria'] = {k: dict(target=target[k], synthetic=row['stats'][k], tolerance=tol,
                        passed=abs(row['stats'][k] - target[k]) <= tol) for k, tol in CRITERIA['tolerance'].items()}
            else:
                row['ak_schedule_sha256_legacy_alias'] = saved[(UNLINKED, load)]
            rows.append(row)
    dist = {}
    for variant in VARIANTS:
        vals = [input_stats(build(variant, 3500, seed)) for seed in DIST_SEEDS]
        dist[variant] = {k: dict(mean=st.mean(v[k] for v in vals), sd=st.pstdev(v[k] for v in vals),
                                 min=min(v[k] for v in vals), max=max(v[k] for v in vals))
                         for k in ('gate_orders_60s', 'gate_orders_600s', 'gate_visits_60s', 'gate_visits_600s',
                                   'block_orders_60s', 'block_visits_60s', 'multi_order_share')}
    misses = [k for r in rows for k, c in r.get('criteria', {}).items() if not c['passed']]
    save(OUT / 'input-preflight.json', dict(rows=rows, real_targets=target, criteria=CRITERIA,
        criteria_passed=not misses, criteria_missed=misses, seed_distribution_3500=dict(seeds=list(DIST_SEEDS), stats=dist),
        observed_max_visits_per_hour=245, script_sha256=sha(Path(__file__))))
    for r in rows:
        s = r['stats']
        print(r['variant'], r['load'], 'multi', round(s['multi_order_share'], 3), 'gateO', round(s['gate_orders_60s'], 3),
              'gateV', round(s['gate_visits_60s'], 3), round(s['gate_visits_600s'], 3), 'blockO', round(s['block_orders_60s'], 3),
              'blockV', round(s['block_visits_60s'], 3), 'maxV/h', s['max_visits_per_hour'])
    print('dist', json.dumps({v: {k: round(x['mean'], 3) for k, x in d.items()} for v, d in dist.items()}))
    print('criteria missed:', misses)


def sources():
    paths = [Path(__file__), PREREG, PROFILE, OUT / 'demand-profiles.json', OUT / 'link-definition.json',
             OUT / 'input-preflight.json', AK / 'demand-profiles.json', AK / 'visit-arrival-templates.json',
             *Path('src/yard_rl/v7').rglob('*.py')]
    paths += [Path('scripts/v7') / p for p in ('audit_visit_units.py', 'audit_visit_links.py', 'evaluate_observed_demand.py',
        'evaluate_visit_arrivals.py', 'completion_learning.py', 'six_step_learning.py', 'horizon_metrics.py',
        'audit_admission_ledger.py', 'verify_job_completion.py', 'benchmark_default_eval.py', 'evaluate_classical_rules.py',
        'evaluate_arrival_order.py', 'verify_reservation_contract.py')]
    paths += [checkpoint(arm, seed) for arm, seed in POLICIES if seed]
    return {p.as_posix(): sha(p) for p in paths}


def check():
    for p, h in read(OUT / 'campaign.json')['sources'].items():
        assert sha(p) == h, ('Frozen experiment input changed', p)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no']).strip()
    assert not (OUT / 'campaign.json').exists()
    assert read(OUT / 'input-preflight.json')['criteria_missed'] == []
    config = dict(variants=VARIANTS, policies=POLICIES, cases=CASES, loads=LOADS, evaluation_seeds=INPUT_SEEDS,
        runtime=RUNTIME, primary_seconds=86400, engine_seconds=93600, training=False, teacher_worlds=0,
        time_contract='actual gate entry to job completion per order, censored at 24h; visit metrics reported separately',
        selected_policy='all three saved initial and ag final models; no result-based selection',
        execution_workers=16, information_profile_sha256=sha(PROFILE),
        profile_sha256={v: profile_sha(v) for v in VARIANTS})
    save(OUT / 'campaign.json', dict(configuration=config, sources=sources(),
        repro=repro_stamp(experiment='YR-332-al-visit-links', seeds=dict(input=list(INPUT_SEEDS.values()),
            models=list(SEEDS)), params=config, prereg=PREREG.as_posix())))


def evaluate(index_):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    torch.set_num_threads(1)
    check()
    variant, arm, seed, load, repeat = CASES[index_]
    cp = checkpoint(arm, seed)
    original, ann = ledger.install_announcer()

    class Probe(FlowProbe):
        def finish(self, mbt, records, episode_):
            self.visit_state = dict(stats=dict(mbt.visit_stats), plans={vid: dict(stops=s['stops'], travel=s['travel'],
                exit=s['exit'], gate_in=s['gate_in'], gate_out=s['gate_out'], release={str(k): v for k, v in s['release'].items()})
                for vid, s in mbt.visits.items()})
            return super().finish(mbt, records, episode_)
    probe = Probe(ann)
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
    data = dict(case=CASES[index_], input_seed=INPUT_SEEDS[load], profile_sha256=profile_sha(variant),
        result=result, probe=probe.result, metrics_by_flow=metrics, completion_by_flow=completion,
        orders_sha256=digest(probe.rows), schedule_sha256=digest(probe.schedule), visits=probe.visit_state,
        checkpoint_sha256=sha(cp) if cp else None, seconds=time.perf_counter() - started)
    assert ep.admitted == load and ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert probe.result['ledger_audit']['skipped'] == 0
    assert all(r['review_count'] == 1 for r in probe.rows)
    assert all(abs(r['attempt']['review_s'] - r['notice_s']) <= 1e-6 for r in probe.rows)
    stats = probe.visit_state['stats']
    assert stats['late_releases'] == stats['late_exits'] == 0, stats
    if variant == UNLINKED:
        old = gz_read(AK / 'evaluation' / (name(CASES[index_]) + '.json.gz'))
        assert result == old['result'] and data['orders_sha256'] == old['orders_sha256']
        assert all(probe.result[k] == old['probe'][k] for k in HASHES) and completion == old['completion_by_flow']
        assert probe.visit_state['plans'] == {}
        data['exact_ak_reproduction'] = True
    for sub, value in [('evaluation', data), ('orders', probe.rows), ('flow-orders', flows)]:
        path = OUT / sub / (name(CASES[index_]) + '.json.gz');assert not path.exists();gz_save(path, value)
    print(f'{index_}: {name(CASES[index_])}, truck={metrics["ALL"]["mean_gate_to_job_done_s"] / 60:.3f}min, '
          f'{data["seconds"]:.1f}s', flush=True)


def run_all():
    check()

    def job(i):
        existing = [(OUT / sub / (name(CASES[i]) + '.json.gz')).exists() for sub in ('evaluation', 'orders', 'flow-orders')]
        if all(existing):
            return dict(index=i, state='already complete')
        if any(existing):
            raise RuntimeError(f'Partial artifacts for {i}; inspect, do not overwrite')
        log = OUT / 'logs' / f'evaluate-{i}.log';log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with log.open('w', encoding='utf-8') as handle:
            p = subprocess.run([sys.executable, __file__, 'evaluate', str(i)], stdout=handle, stderr=subprocess.STDOUT, env=env)
        if p.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8'))
        return dict(index=i, state='complete')
    order = sorted(range(len(CASES)), key=lambda i: (-CASES[i][3], i))
    done = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in as_completed([pool.submit(job, i) for i in order]):
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
