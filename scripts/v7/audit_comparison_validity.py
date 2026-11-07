"""YR-332-t: exact re-runs of the frozen seven-rule comparison with read-only cost evidence.

No training, reward, physics, candidate or policy change. Every case must reproduce the
YR-332-v artifact exactly (six hashes, result, input metadata and operational counts).
The extra probe only reads: truck lifecycle times, the 24-hour cost counters used by the
official cost, raw values behind clipped policy inputs, and agreement with the shortest job.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import evaluate_arrival_order as prior
import evaluate_classical_rules as classical
from six_step_learning import gz_read, gz_save, load_net, sha

OUT = Path('outputs/reports/yr332_comparison_audit')
PREREG = Path('docs/research/v7-discharge/comparison-audit-prereg.md')
LOADS, SEEDS, ARMS, RULES = prior.LOADS, prior.SEEDS, prior.ARMS, classical.RULES
CASES = [tuple(c) for c in classical.CASES if not c[3]]
WORKERS = 8
END_S = 86400.0
#: Clip points of the learned policy inputs (crane/features.py), raw units.
CLIPS = dict(duration_s=3600.0, empty_gantry_m=200.0, rehandles=30.0,
             cum_wait_s=7200.0, slack_abs_s=7200.0)
COUNT_CLIP = 18.0
COST_KEYS = ('phi_krw', 'c_wait', 'c_move', 'c_rehandle', 'c_vessel', 'n_trucks',
             'n_censored', 'mean_turn_time_s')
name, save, git = prior.name, prior.save, prior.git


def path_for(case):
    return OUT/'evaluation'/(name(case)+'.json.gz')


def trucks_path(case):
    return OUT/'trucks'/(name(case)+'.json.gz')


def source_path(case):
    return classical.path_for(case)


def source_hashes():
    files = [Path(__file__), Path(classical.__file__), Path(prior.__file__),
             Path('scripts/v7/six_step_learning.py'), PREREG, *Path('src/yard_rl/v7').rglob('*.py')]
    return {p.resolve().relative_to(Path.cwd()).as_posix(): sha(p) for p in sorted(files)}


def stage_split(rec, end_s=END_S):
    """Censored turn time split into A->B, B->S, S->C, C->O; the parts sum to min(O,T)-A."""
    a = rec.gate_in_s
    if a is None:
        return None
    o = rec.gate_out_s if rec.gate_out_s is not None and rec.gate_out_s <= end_s else end_s
    o = max(o, a)
    stamps = [rec.block_in_s, rec.service_start_s, rec.job_done_s]
    # A missing middle stamp followed by a later one is a lifecycle gap; it is counted.
    gap = any(stamps[i] is None and any(t is not None for t in stamps[i+1:])
              for i in range(3)) or (rec.gate_out_s is not None and None in stamps)
    marks = [a]
    for t in stamps:
        prev = marks[-1]
        marks.append(min(max(t, prev), o) if t is not None and t <= o else o)
    marks.append(o)
    parts = [marks[i+1]-marks[i] for i in range(4)]
    return dict(travel_in=parts[0], queue=parts[1], service=parts[2], exit=parts[3],
                total=o-a, censored=rec.gate_out_s is None or rec.gate_out_s > end_s, gap=gap)


def truck_sums(records, flow_of, end_s=END_S):
    """Exact sums needed to rescore the truck term with any linear coefficients."""
    from yard_rl.v7.schema.lifecycle import censored_turn_time_s
    out = dict(n=0, censored=0, lifecycle_gaps=0, sum_tt=0.0, over_3600=0.0, over_1800=0.0,
               stages={}, by_flow={})
    for key, rec in sorted(records.items()):
        tt = censored_turn_time_s(rec, end_s)
        if tt is None:
            continue
        split = stage_split(rec, end_s)
        assert math.isclose(split['total'], tt, rel_tol=0, abs_tol=1e-6), key
        flow = flow_of.get(key, 'UNKNOWN')
        out['n'] += 1
        out['censored'] += split['censored']
        out['lifecycle_gaps'] += split['gap']
        out['sum_tt'] += tt
        out['over_3600'] += max(0.0, tt-3600.0)
        out['over_1800'] += max(0.0, tt-1800.0)
        for part in ('travel_in', 'queue', 'service', 'exit'):
            out['stages'][part] = out['stages'].get(part, 0.0)+split[part]
            row = out['by_flow'].setdefault(flow, dict(n=0, censored=0, sum_tt=0.0))
            row[part] = row.get(part, 0.0)+split[part]
        row = out['by_flow'][flow]
        row['n'] += 1
        row['censored'] += split['censored']
        row['sum_tt'] += tt
    return out


def install_tape_recorder():
    """Subclass the episode counter tape so its 24-hour reading can be inspected."""
    import yard_rl.v7.stage.episode as episode
    base = episode._CounterTape
    if getattr(base, 'audit_recorder', False):
        return base

    class RecordingTape(base):
        audit_recorder = True
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_t, self.streams = None, None
            RecordingTape.instances.append(self)

        def snap(self, mbt, t):
            super().snap(mbt, t)
            self.last_t = float(t)
            self.streams = [dict(block=bid, stream=vid, flow=v.work_type.value,
                                 sts_wait_s=float(getattr(v, 'sts_wait_accum_s', 0.0)))
                            for bid, sim in sorted(mbt.blocks.items())
                            for vid, v in sorted(getattr(sim, 'vessels', {}).items())]

    episode._CounterTape = RecordingTape
    return RecordingTape


class AuditProbe(classical.StudyProbe):
    """Read-only: no value is supplied to any preference, resolver or engine."""

    def __init__(self, tape_class):
        super().__init__()
        self.tape_class = tape_class
        self.inputs = Counter()
        self.maxima = {}
        self.choice = Counter()
        self.excess_s = 0.0

    def start(self, built, mbt):
        super().start(built, mbt)
        self.flow_of = {e['job_id']: e['flow'] for e in built['schedule']}
        self.fleet = [dict(vessel_id=v.vessel_id, vessel_class=v.cls.name, gt=v.gt,
                           moves=v.moves, port_time_s=v.port_time_s,
                           berth_start_s=v.berth_start_s, work_s=v.work_s())
                      for v in built.get('fleet', [])]

    @staticmethod
    def _serve(items):
        from yard_rl.v7.world.contract.schema import CandidateKind
        return [g for g in items if g.feasible and g.job_ref is not None
                and g.plan is not None and g.kind == CandidateKind.SERVE]

    def _observe(self, key, value):
        self.inputs[key+'_n'] += 1
        self.inputs[key+'_over_clip'] += value > CLIPS[key]
        self.maxima[key] = max(self.maxima.get(key, value), value)

    def before_select(self, sim, dp, gb, pref):
        from yard_rl.v7.crane.features import _live_cands
        waiting = _live_cands(sim, '')
        for cid in dp.crane_ids:
            serve = self._serve(gb[cid].items)
            if not serve:
                continue
            self.inputs['crane_decisions'] += 1
            self.inputs['waiting_count_over_18'] += waiting > COUNT_CLIP
            self.inputs['serve_candidates'] += len(serve)
            for g in serve:
                plan, ref = g.plan, g.job_ref
                self._observe('duration_s', float(plan.duration_s))
                self._observe('empty_gantry_m', float(plan.empty_gantry_m or 0.0))
                self._observe('rehandles', float(plan.rehandles or 0))
                if ref.is_external:
                    self._observe('cum_wait_s', float(sim.cum_wait(ref.job_id)))
                if ref.is_vessel:
                    job = sim.jobs.get(ref.job_id)
                    vid = getattr(job, 'vessel_id', None) if job is not None else None
                    vessel = sim.vessels.get(vid) if vid else None
                    slack = vessel.slack_s(sim.clock) if vessel is not None else None
                    if slack is not None:
                        self._observe('slack_abs_s', abs(float(slack)))
                        self.inputs['slack_above_plus_clip'] += slack > CLIPS['slack_abs_s']
                        self.inputs['slack_below_minus_clip'] += slack < -CLIPS['slack_abs_s']

    def decision(self, sim, dp, generated, assignment, gen_ns, select_ns):
        super().decision(sim, dp, generated, assignment, gen_ns, select_ns)
        from yard_rl.v7.world.contract.schema import CandidateKind
        for cid in dp.crane_ids:
            serve = self._serve(generated[cid].items)
            chosen = assignment.get(cid)
            if (len(serve) < 2 or chosen is None or chosen.kind != CandidateKind.SERVE
                    or chosen.plan is None):
                continue
            shortest = min(g.plan.duration_s for g in serve)
            self.choice['serve_choices_with_alternatives'] += 1
            self.choice['chose_shortest'] += chosen.plan.duration_s <= shortest
            self.excess_s += chosen.plan.duration_s-shortest

    def finish(self, mbt, records, episode):
        super().finish(mbt, records, episode)
        from yard_rl.v7.reward import krw
        from yard_rl.v7.reward.phi import terminal_cost_krw
        tape = self.tape_class.instances[-1]
        assert tape.last_t == END_S, tape.last_t
        idle, yc_s, rehandles = tape.read(END_S)
        recomputed = terminal_cost_krw(records, end_s=END_S, vessel_idle=idle,
                                       yc_extra_move_s=yc_s, rehandles=rehandles).as_dict()
        stored = episode.as_dict()
        mismatch = [k for k in COST_KEYS if recomputed[k] != stored[k]]
        sums = truck_sums(records, self.flow_of)
        wait = krw.KRW_TRUCK_HOUR*sums['sum_tt']/3600+krw.KRW_TRUCK_OVER_HOUR*sums['over_3600']/3600
        ships = {k: dict(gt=gt, idle_s=s) for k, (gt, s) in sorted(idle.items())}
        gt_hours = sum(v['gt']*v['idle_s']/3600 for v in ships.values())
        sts_of = {r['vessel_id']: int(r['sts']) for r in self.block_vessel.values()}
        ship_of = {b: r['vessel_id'] for b, r in self.block_vessel.items()}
        streams = [dict(s, ship=ship_of.get(s['block']), sts=sts_of.get(ship_of.get(s['block'])))
                   for s in tape.streams]
        self.result['audit'] = dict(
            end_s=END_S, tape_last_snap_s=tape.last_t, phi_recomputed_equal=not mismatch,
            phi_mismatch=mismatch,
            truck_wait_from_sums_rel_error=abs(wait-stored['c_wait'])/max(1.0, stored['c_wait']),
            vessel_from_gt_hours_rel_error=abs(krw.KRW_VESSEL_GT_HOUR*gt_hours-stored['c_vessel'])
            / max(1.0, stored['c_vessel']),
            trucks=sums, orders=len(self.flow_of), records=len(records),
            yc_empty_travel_s=yc_s, rehandles=rehandles, ships_24h=ships,
            vessel_gt_idle_hours=gt_hours, streams_24h=streams, fleet=self.fleet,
            policy_inputs=dict(self.inputs), input_maxima=self.maxima,
            shortest_job_agreement=dict(self.choice, excess_duration_s=self.excess_s))
        self.truck_rows = dict(
            key=[], flow=[], block=[], gate_in_s=[], block_in_s=[], service_start_s=[],
            job_done_s=[], gate_out_s=[])
        for key, rec in sorted(records.items()):
            self.truck_rows['key'].append(key)
            self.truck_rows['flow'].append(self.flow_of.get(key))
            self.truck_rows['block'].append(key.split(':', 1)[0])
            for field in ('gate_in_s', 'block_in_s', 'service_start_s', 'job_done_s', 'gate_out_s'):
                value = getattr(rec, field)
                self.truck_rows[field].append(None if value is None else round(value, 3))


def evaluate_case(case):
    import torch
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
    torch.set_num_threads(1)
    tape_class = install_tape_recorder()
    arm, model_seed, load, repeat = case
    is_rule = arm in RULES or arm == 'rule_all'
    model = None if is_rule else prior.PRIOR/f'model-{model_seed}'/(
        'initial.pt' if arm == 'initial' else f'{arm}-2.pt')
    net = None if model is None else load_net(model)
    dispatcher = RULES.get(arm, 'SF_SPT' if arm == 'rule_all' else 'RL_CRANE')
    seed = 9913900+10*LOADS.index(load)
    probe = AuditProbe(tape_class)
    start = time.perf_counter()
    r = run_episode(load=load, seed=seed, dispatcher=dispatcher, crane_net=net,
                    candidate_cap=None, evaluation_probe=probe, reuse_rehandle_slots=False,
                    decision_probe_shortcircuit=True)
    assert r.policy_exceptions == 0 and r.rollout_worlds == 0
    assert r.traded_edges == 0 and r.n_space == 0 and r.n_time == 0
    d = dict(arm=arm, model_seed=None if is_rule else model_seed, load=load, seed=seed,
             repeat=repeat, dispatcher=dispatcher, python=sys.version, torch=torch.__version__,
             device='cpu', wall_seconds=time.perf_counter()-start,
             parameters=dict(candidate_cap=None, shared_slots=False, existence_scan=True,
                             teacher=False),
             exec_config=asdict(DEFAULT_EXEC_CONFIG), result=r.as_dict(), probe=probe.result,
             checkpoint=None if model is None else model.as_posix(),
             checkpoint_sha256=None if model is None else sha(model))
    gz_save(path_for(case), d)
    gz_save(trucks_path(case), probe.truck_rows)
    a = probe.result['audit']
    print(f'{name(case)}: cost={r.phi_krw:.3f}; recomputed_equal={a["phi_recomputed_equal"]}',
          flush=True)


def replay_check(case):
    d = gz_read(path_for(case))
    old = gz_read(source_path(case))
    prior.replay_equal(d, old)
    assert d['probe']['operational'] == old['probe']['operational'], 'operational differs'
    a = d['probe']['audit']
    assert a['phi_recomputed_equal'], a['phi_mismatch']
    assert a['truck_wait_from_sums_rel_error'] < 1e-9, a['truck_wait_from_sums_rel_error']
    assert a['vessel_from_gt_hours_rel_error'] < 1e-9, a['vessel_from_gt_hours_rel_error']
    assert a['tape_last_snap_s'] == END_S
    assert a['trucks']['n'] == d['result']['n_trucks']
    assert a['trucks']['censored'] == d['result']['n_censored']
    return d


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not git('status', '--short', '--untracked-files=no'), 'commit before running'
    source = json.loads((classical.OUT/'campaign.json').read_text(encoding='utf-8'))
    source_hash = {p: h for p, h in source['result_sha256'].items()}
    used = {}
    for case in CASES:
        p = source_path(case).as_posix()
        assert sha(p) == source_hash[p], p
        used[p] = source_hash[p]
    assert len(used) == 69
    for p, h in source['model_sha256'].items():
        assert sha(p) == h, p
    config = dict(blocks=21, cranes=42, observation_s=END_S, drain_s=7200, initial_fill=.65,
                  loads=LOADS, input_seeds=[9913900, 9913910, 9913920], candidate_cap=None,
                  decision_probe_shortcircuit=True, reuse_rehandle_slots=False,
                  reallocation=False, training=False, threads_per_case=1, workers=WORKERS,
                  cases=len(CASES), cost_contract='unchanged v7 KRW four terms',
                  probe='read-only lifecycle, 24h counters, raw policy inputs, shortest-job agreement')
    return dict(task='YR-332-t', commit=git('rev-parse', 'HEAD'), dirty=False,
                sources=source_hashes(), source_campaign_commit=source['commit'],
                source_result_sha256=used, model_sha256=source['model_sha256'],
                cases=CASES, completed=[], configuration=config,
                scope='Exact re-runs of YR-332-v without repeats; frozen policies; audit only.',
                repro=repro_stamp(experiment='YR-332-t', seeds=dict(
                    input=[9913900, 9913910, 9913920], model=list(SEEDS)),
                    params=dict(configuration=config), prereg=PREREG.as_posix()))


def campaign():
    marker = OUT/'campaign.json'
    stamp = prepare()
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        for k in ('sources', 'source_result_sha256', 'model_sha256', 'commit'):
            assert old[k] == stamp[k], f'{k} changed'
        stamp = old
    save(marker, stamp)

    def run(i):
        case = CASES[i]
        log = OUT/'logs'/(name(case)+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', encoding='utf-8') as f:
            subprocess.run([sys.executable, __file__, '--operation', 'one', '--index', str(i)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        return i, replay_check(case)

    order = sorted(range(len(CASES)), key=lambda i: (-CASES[i][2], i))
    pending = iter(i for i in order if name(CASES[i]) not in stamp['completed'])
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = set()
        for _ in range(WORKERS):
            i = next(pending, None)
            if i is not None:
                futures.add(pool.submit(run, i))
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                i, d = f.result()
                case = CASES[i]
                stamp['completed'].append(name(case))
                stamp.setdefault('result_sha256', {})[path_for(case).as_posix()] = sha(path_for(case))
                stamp.setdefault('truck_sha256', {})[trucks_path(case).as_posix()] = sha(trucks_path(case))
                save(marker, stamp)
                print(f'DONE {len(stamp["completed"])}/{len(CASES)} {name(case)} '
                      f'{d["wall_seconds"]:.1f}s exact', flush=True)
            for _ in done:
                i = next(pending, None)
                if i is not None:
                    futures.add(pool.submit(run, i))
    verify()


def verify():
    from yard_rl.experiments.gate_harness import judge_runtime_evidence
    stamp = json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    assert set(stamp['completed']) == {name(c) for c in CASES} and len(CASES) == 69
    assert source_hashes() == stamp['sources'], 'source changed after the campaign'
    for section in ('source_result_sha256', 'model_sha256', 'result_sha256', 'truck_sha256'):
        for p, h in stamp[section].items():
            assert sha(p) == h, p
    for case in CASES:
        replay_check(case)
    hashes = {**stamp['result_sha256'], **stamp['truck_sha256'], PREREG.as_posix(): sha(PREREG)}
    runtime = asdict(judge_runtime_evidence(stamp['repro'], artifact_hashes=hashes, root='.'))
    assert runtime['status'] == 'PASS', runtime
    save(OUT/'replay-verification.json', dict(
        runtime=runtime, cases=len(CASES), exact_replays=len(CASES),
        checked=['candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256',
                 'records_sha256', 'result_sha256', 'result', 'metadata', 'flow_status_counts',
                 'operational', 'phi recomputed from records and 24h counters'],
        source_campaign_commit=stamp['source_campaign_commit'], audit_commit=stamp['commit']))
    print(f'VERIFIED: {len(CASES)} exact replays with read-only audit evidence', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('campaign', 'one', 'verify'), default='campaign')
    p.add_argument('--index', type=int)
    args = p.parse_args()
    if args.operation == 'one':
        evaluate_case(CASES[args.index])
    elif args.operation == 'verify':
        verify()
    else:
        campaign()
