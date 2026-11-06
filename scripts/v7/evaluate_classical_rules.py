"""Preregistered seven-rule evaluation of all frozen policies (no training)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import evaluate_arrival_order as prior
from six_step_learning import gz_read, gz_save, load_net, sha
from yard_rl.v7.evaluation_probe import EvaluationProbe

OUT = Path('outputs/reports/yr332_classical_eval')
PREREG = Path('docs/research/v7-discharge/classical-rules-prereg.md')
LOADS, SEEDS, ARMS = prior.LOADS, prior.SEEDS, prior.ARMS
RULES = dict(fcfs_arrival='FCFS_ARRIVAL', spt='SPT', nearest='NEAREST',
             vessel_fcfs='VESSEL_FCFS', vessel_nearest='VESSEL_NEAREST',
             unidirectional='UNIDIRECTIONAL', distance_wait='DISTANCE_WAIT')
CASES = [(a,0,n,0) for n in LOADS for a in RULES] + [
    (a,s,n,0) for n in LOADS for a in ARMS for s in SEEDS] + [
    ('rule_all',0,n,0) for n in LOADS] + [
    ('nearest',0,3500,1), ('unidirectional',0,3500,1)]
WORKERS = 6
save, git, name = prior.save, prior.git, prior.name


def path_for(case):
    return OUT/'evaluation'/(name(case)+'.json.gz')


def source_hashes():
    files = [Path(__file__), Path(prior.__file__), Path('scripts/v7/six_step_learning.py'),
             PREREG, *Path('src/yard_rl/v7').rglob('*.py')]
    return {p.resolve().relative_to(Path.cwd()).as_posix(): sha(p) for p in files}


class StudyProbe(EvaluationProbe):
    """Read final truth for evaluation only; never supplied to any preference."""

    def start(self, built, mbt):
        super().start(built,mbt)
        self.block_vessel = built['block_vessel']
        self.end = float(built['sim_end_s'])

    def finish(self, mbt, records, episode):
        from yard_rl.v7.stage.episode import vessel_idle_of
        super().finish(mbt,records,episode)
        streams, ships = [], {}
        for bid,sim in sorted(mbt.blocks.items()):
            for vid,v in sorted(sim.vessels.items()):
                ship = self.block_vessel[bid]['vessel_id']
                row = dict(block=bid,stream=vid,ship=ship,flow=v.work_type.value,
                    planned_s=v.plan.planned_completion_s,
                    actual_s=v.truth.actual_completion_s,done=v.done,
                    total_moves=v.plan.total_moves,remaining_moves=v.remaining_moves,
                    sts_wait_s=v.sts_wait_accum_s)
                streams.append(row)
                ships.setdefault(ship,[]).append(row)
        ship_rows = []
        for ship,rows in sorted(ships.items()):
            assert all(r['planned_s'] is not None for r in rows)
            done = all(r['done'] and r['actual_s'] is not None for r in rows)
            actual = max(r['actual_s'] for r in rows) if done else None
            plan = max(r['planned_s'] for r in rows)
            observed = min(actual,self.end) if done else self.end
            ship_rows.append(dict(ship=ship,planned_s=plan,actual_s=actual,
                                  done=done,delay_s=max(0.,observed-plan)))
        idle = vessel_idle_of(mbt,self.end,self.block_vessel)
        self.result['operational'] = dict(horizon_s=self.end,
            vessel_delay_s=sum(r['delay_s'] for r in ship_rows),
            vessel_unfinished=sum(not r['done'] for r in ship_rows),
            vessel_idle_s=sum(x[1] for x in idle.values()),
            trucks_unfinished=sum(r.gate_in_s is not None and r.gate_in_s<=self.end
                and (r.gate_out_s is None or r.gate_out_s>self.end) for r in records.values()),
            ships=ship_rows,streams=streams,
            ship_idle={k:dict(gt=v[0],idle_s=v[1]) for k,v in idle.items()})


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    stamp = prior.prepare()  # Verifies all published 51 old artifacts and 15 models.
    previous = json.loads((prior.OUT/'campaign.json').read_text(encoding='utf-8'))
    changes = [p for p,h in previous['sources'].items() if sha(p)!=h]
    assert changes==['src/yard_rl/v7/dispatch.py'], changes
    for p,h in previous['result_sha256'].items():
        assert sha(p)==h,p
    stamp.update(task='YR-332-v',sources=source_hashes(),cases=CASES,completed=[],workers=WORKERS,
                 prior_arrival_sha256=previous['result_sha256'],retraining=False)
    stamp['configuration'].update(rule_dispatchers=RULES,reference_dispatcher='SF_SPT',
        new_evaluations=len(CASES),frozen_model_evaluations=45,worker_limit=WORKERS,
        final_operational_horizon_s=93600,smoke_seed=9913999)
    stamp['scope']='All seven rules and all 15 stored models freshly evaluated; same three prior days.'
    stamp['repro'] = repro_stamp(experiment='YR-332-v',
        seeds=dict(input=[9913900,9913910,9913920],model=list(SEEDS)),
        params=stamp['configuration'],prereg=PREREG.as_posix())
    return stamp


def evaluate_case(case, *, smoke=False):
    import torch
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
    torch.set_num_threads(1)
    arm,model_seed,load,repeat = case
    is_rule = arm in RULES or arm=='rule_all'
    model = None if is_rule else prior.PRIOR/f'model-{model_seed}'/(
        'initial.pt' if arm=='initial' else f'{arm}-2.pt')
    net = None if model is None else load_net(model)
    dispatcher = RULES.get(arm,'SF_SPT' if arm=='rule_all' else 'RL_CRANE')
    seed = 9913999 if smoke else 9913900+10*LOADS.index(load)
    probe = StudyProbe()
    start = time.perf_counter()
    r = run_episode(load=load,seed=seed,dispatcher=dispatcher,crane_net=net,
        candidate_cap=None,evaluation_probe=probe,reuse_rehandle_slots=False,
        decision_probe_shortcircuit=True)
    assert r.policy_exceptions==0 and r.rollout_worlds==0
    assert r.traded_edges==0 and r.n_space==0 and r.n_time==0
    d=dict(arm=arm,model_seed=None if is_rule else model_seed,load=load,seed=seed,
        repeat=repeat,dispatcher=dispatcher,python=sys.version,torch=torch.__version__,device='cpu',
        wall_seconds=time.perf_counter()-start,
        parameters=dict(candidate_cap=None,shared_slots=False,existence_scan=True,teacher=False),
        exec_config=asdict(DEFAULT_EXEC_CONFIG),result=r.as_dict(),probe=probe.result,
        checkpoint=None if model is None else model.as_posix(),
        checkpoint_sha256=None if model is None else sha(model))
    dest = OUT/'smoke'/(name(case)+'.json.gz') if smoke else path_for(case)
    gz_save(dest,d)
    print(f'{name(case)}: cost={r.phi_krw:.3f}; turn_s={r.breakdown["mean_turn_time_s"]:.3f}',flush=True)
    return d


def replay_path(case):
    arm,_,_,repeat = case
    if repeat:
        return path_for((*case[:3],0))
    if arm=='fcfs_arrival':
        return prior.OUT/'evaluation'/(name(case)+'.json.gz')
    if arm in ARMS or arm=='rule_all':
        return prior.PRIOR/'evaluation'/(name(case)+'.json.gz')
    return None


def campaign():
    marker=OUT/'campaign.json'
    stamp=prepare()
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        for k in ('sources','prior_evaluation_sha256','prior_arrival_sha256','model_sha256','commit'):
            assert old[k]==stamp[k],f'{k} changed'
        stamp=old
    save(marker,stamp)
    def run(i):
        case=CASES[i]
        print('START '+name(case),flush=True)
        log=OUT/'logs'/(name(case)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,__file__,'--operation','one','--index',str(i)],
                           stdout=f,stderr=subprocess.STDOUT,check=True)
        d=gz_read(path_for(case))
        p=replay_path(case)
        if p is not None:
            prior.replay_equal(d,gz_read(p))
        return i,d
    order=sorted(range(len(CASES)),key=lambda i:(CASES[i][3],CASES[i][2],i))
    pending=iter(i for i in order if name(CASES[i]) not in stamp['completed'])
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures=set()
        for _ in range(WORKERS):
            i=next(pending,None)
            if i is not None:futures.add(pool.submit(run,i))
        while futures:
            done,futures=wait(futures,return_when=FIRST_COMPLETED)
            for f in done:
                i,d=f.result()
                stamp['completed'].append(name(CASES[i]))
                stamp.setdefault('result_sha256',{})[path_for(CASES[i]).as_posix()]=sha(path_for(CASES[i]))
                save(marker,stamp)
                print(f'DONE {len(stamp["completed"])}/{len(CASES)} {name(CASES[i])} '
                      f'{d["wall_seconds"]:.1f}s',flush=True)
            for _ in done:
                i=next(pending,None)
                if i is not None:futures.add(pool.submit(run,i))
    verify()


def comparison(a,b):
    d=prior.compare(a,b)
    ao,bo=a['probe']['operational'],b['probe']['operational']
    d.update(vessel_delay_26h_s_change=bo['vessel_delay_s']-ao['vessel_delay_s'],
        vessel_unfinished_26h_change=bo['vessel_unfinished']-ao['vessel_unfinished'],
        vessel_idle_26h_s_change=bo['vessel_idle_s']-ao['vessel_idle_s'],
        truck_unfinished_26h_change=bo['trucks_unfinished']-ao['trucks_unfinished'],
        p90_turn_s_change=b['result']['p90_turn_time_s']-a['result']['p90_turn_time_s'],
        p99_turn_s_change=b['result']['p99_turn_time_s']-a['result']['p99_turn_time_s'],
        vessel_cost_change=b['result']['c_vessel']-a['result']['c_vessel'])
    return d


def aggregate(records):
    averages=[]
    for arm,baseline in sorted({(r['arm'],r['baseline']) for r in records}):
        all_rows=[r for r in records if r['arm']==arm and r['baseline']==baseline]
        for load in (None,*LOADS):
            rows=[r for r in all_rows if load is None or r['load']==load]
            if not rows:continue
            row=dict(arm=arm,baseline=baseline,load=load,n=len(rows))
            row.update({k:statistics.mean(r[k] for r in rows) for k in rows[0]
                        if k not in ('arm','model_seed','load','baseline')})
            row.update(cost_wins=sum(r['cost_change_pct']<0 for r in rows),
                turn_wins=sum(r['turn_change_pct']<0 for r in rows),
                admission_drop=sum(r['admitted_change']<0 for r in rows),
                unfinished_increase=sum(r['censored_24h_change']>0 for r in rows),
                vessel_drop=sum(r['vessel_done_26h_change']<0 for r in rows),
                vessel_delay_increase=sum(r['vessel_delay_26h_s_change']>1e-6 for r in rows))
            averages.append(row)
    return averages


def verify():
    from yard_rl.experiments.gate_harness import judge_runtime_evidence
    stamp=json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    assert len(CASES)==71 and set(stamp['completed'])=={name(c) for c in CASES}
    assert source_hashes()==stamp['sources']
    for section in ('prior_evaluation_sha256','prior_arrival_sha256','model_sha256','result_sha256'):
        for p,h in stamp[section].items():assert sha(p)==h,p
    current={tuple(c):gz_read(path_for(c)) for c in CASES}
    exact=0
    for c,d in current.items():
        assert d['result']['policy_exceptions']==0
        p=replay_path(c)
        if p is not None:
            prior.replay_equal(d,gz_read(p));exact+=1
            if c[3]:assert d['probe']['operational']==gz_read(p)['probe']['operational']
        op=d['probe']['operational']
        assert op['horizon_s']==93600
        assert len(op['ships'])==d['result']['n_vessels']
        assert len(op['streams'])==d['result']['n_vessel_streams']
        assert all(r['actual_s'] is None or r['actual_s']<=93600 for r in op['ships'])
    records,learning=[],[]
    for load in LOADS:
        fcfs=current[('fcfs_arrival',0,load,0)]
        for rule in (*RULES,'rule_all'):
            base=current[(rule,0,load,0)]
            assert base['probe']['metadata']==fcfs['probe']['metadata']
            if rule!='fcfs_arrival':
                records.append(dict(arm=rule,model_seed=None,load=load,baseline='fcfs_arrival',
                                    **comparison(fcfs,base)))
            for arm in ARMS:
                for seed in SEEDS:
                    d=current[(arm,seed,load,0)]
                    assert d['probe']['metadata']==base['probe']['metadata']
                    records.append(dict(arm=arm,model_seed=seed,load=load,baseline=rule,
                                        **comparison(base,d)))
        for arm in ARMS[1:]:
            for seed in SEEDS:
                learning.append(dict(arm=arm,model_seed=seed,load=load,baseline='own_initial',
                    **comparison(current[('initial',seed,load,0)],current[(arm,seed,load,0)])))
    hashes={**stamp['result_sha256'],PREREG.as_posix():sha(PREREG)}
    runtime=asdict(judge_runtime_evidence(stamp['repro'],artifact_hashes=hashes,root='.'))
    assert runtime['status']=='PASS',runtime
    save(OUT/'summary.json',dict(comparisons=records,averages=aggregate(records),
        learning_comparisons=learning,learning_averages=aggregate(learning),
        fresh_evaluations=71,exact_prior_replays=exact-2,exact_repeats=2,
        independent_input_days=3,learning_methods=4,trained_models=12,retraining=False))
    save(OUT/'verification.json',dict(runtime=runtime,source_files=len(stamp['sources']),
        source_unchanged=True,prior_artifacts=len(stamp['prior_evaluation_sha256']),
        prior_arrival_artifacts=len(stamp['prior_arrival_sha256']),models=15,
        new_artifacts=71,exact_prior_replays=exact-2,exact_repeats=2,
        all_input_metadata_equal=True,all_operational_counts_valid=True))
    print(f'VERIFIED: {exact-2} exact prior replays, 2 exact repeats, all 71 fresh evaluations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=('campaign','one','verify','smoke'),default='campaign')
    p.add_argument('--index',type=int)
    p.add_argument('--rule',choices=tuple(RULES),default='nearest')
    args=p.parse_args()
    if args.operation=='one':evaluate_case(CASES[args.index])
    elif args.operation=='verify':verify()
    elif args.operation=='smoke':evaluate_case((args.rule,0,80,0),smoke=True)
    else:campaign()
