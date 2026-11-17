"""Fixed-budget on-policy scheduling diagnostic; no teacher worlds or best-run selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import statistics
import subprocess
import time

from completion_learning import LOADS, RUNTIME, profile as information
from six_step_learning import gz_read, gz_save, sha, load_net
from synthetic_arrivals import profile as arrival_profile
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_resched_ppo')
PREREG = Path('docs/research/v7-discharge/resched-ppo-prereg.md')
SEEDS = (9951101, 9951102, 9951103)
KINDS = ('mlp', 'transformer')
UPDATES = 6


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def demand(split, i):
    return arrival_profile(dict(axis='minutes', arm='parametric', split=split, index=i))


def cases():
    rows = []
    for kind in KINDS:
        for seed in SEEDS:
            for stage in ('initial', 'final'):
                for i, load in enumerate(LOADS):
                    rows.append(dict(arm=f'{kind}_{stage}', kind=kind, model_seed=seed,
                                     load=load, index=i+3, input_seed=9951300+i, stage=stage))
    for arm, seeds in [('old200', (9917101, 9917102, 9917103)), ('SF_SPT', (0,)),
                        ('FIFO', (0,)), ('LWKR', (0,))]:
        for seed in seeds:
            for i, load in enumerate(LOADS):
                rows.append(dict(arm=arm, model_seed=seed, load=load, index=i+3, input_seed=9951300+i))
    return rows


def sources():
    paths = [Path('scripts/v7/resched_ppo.py'), PREREG, *Path('src/yard_rl/v7').rglob('*.py'),
             *[Path('scripts/v7')/name for name in ('synthetic_arrivals.py', 'synthetic_inputs.py',
                 'six_step_learning.py','horizon_metrics.py','audit_admission_ledger.py',
                 'verify_job_completion.py','completion_learning.py')],
             Path('outputs/reports/yr332_real_order_information/profile.json'),
             Path('outputs/reports/yr332_synthetic_inputs/bank.json'),
             Path('outputs/reports/yr332_synthetic_inputs/multiscale-minute-calibration.json')]
    return {p.as_posix():sha(p) for p in paths}


def prepare():
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
    assert not (OUT/'campaign.json').exists()
    scale_rows = []
    for path in sorted(Path('outputs/reports/yr332_parametric_arrivals/evaluation').glob('*.json.gz')):
        row = gz_read(path)
        if 'repeat' not in path.name and row['case']['arm']=='parametric' and row['case']['rule']=='SF_SPT':
            assert row['case']['split']=='train'
            scale_rows.append(dict(path=path.as_posix(), sha256=sha(path), cost=row['result']['phi_krw']))
    assert len(scale_rows)==9
    scale = statistics.median(r['cost'] for r in scale_rows)
    config = dict(kinds=KINDS, model_seeds=SEEDS, updates=UPDATES, days_per_update=2,
        trajectories=[dict(load=LOADS[i%3], split='train', index=i, input_seed=9951200+i) for i in range(12)],
        evaluation_cases=cases(), max_samples_per_update=2048, epochs=4, batch_size=64,
        lr=.0003, clip=.2, value_coefficient=.5, entropy_coefficient=.01, gradient_clip=1.,
        discount=1., return_kind='terminal Monte Carlo minus preceding synchronized epoch cost',
        width=32, heads=4, layers=2, return_scale=scale, scale_evidence=scale_rows,
        runtime=RUNTIME, observation_seconds=86400, engine_seconds=93600,
        decoder='sequential legal joint construction; mandatory/service-first tiers retained',
        evaluation='single deterministic greedy final checkpoint; no best-of samples',
        scope='user-requested diagnostic; real-world and independent-period validity not established')
    save(OUT/'campaign.json', dict(configuration=config,sources=sources(),
         repro=repro_stamp(experiment='YR-332-av', seeds=dict(models=list(SEEDS),
            training=list(range(9951200,9951212)), evaluation=list(range(9951300,9951303))),
            params=config, prereg=PREREG.as_posix())))


def check():
    obj = json.loads((OUT/'campaign.json').read_text('utf-8'))
    for path, h in obj['sources'].items():
        assert sha(path)==h, ('Frozen source changed',path)
    for row in obj['configuration']['scale_evidence']:
        assert sha(row['path'])==row['sha256']
    return obj['configuration']


def folder(kind, seed):
    return OUT/f'{kind}-{seed}'


def encode_sample(s):
    return dict(features=s.state.features.tolist(), operations=s.state.operations.tolist(),
                cranes=s.state.cranes.tolist(), time_fraction=s.state.time_fraction,
                masks=s.masks, choices=s.choices, old_log_probability=s.old_log_probability,
                old_value=s.old_value, past_cost=s.past_cost, target=s.target)


def decode_sample(obj):
    import torch
    from yard_rl.v7.crane.actor_critic import GraphState
    from yard_rl.v7.crane.on_policy import DecisionSample
    d = dict(obj)
    st = GraphState(torch.tensor(d.pop('features')), torch.tensor(d.pop('operations')),
                    torch.tensor(d.pop('cranes')), d.pop('time_fraction'))
    return DecisionSample(state=st, **d)


def train(task, replay=False):
    import torch
    from yard_rl.v7.crane.actor_critic import SchedulingActorCritic
    from yard_rl.v7.crane.on_policy import OnPolicyResolver, ppo_update
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    cfg=check();kind,seed=task
    torch.manual_seed(seed)
    net=SchedulingActorCritic(kind).eval()
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['lr'])
    root=folder(kind,seed)
    if not replay:
        root.mkdir(parents=True,exist_ok=True)
        assert not (root/'initial.pt').exists()
        net.save(root/'initial.pt',seed=seed)
    history=[];started=time.perf_counter()
    for it in range(UPDATES):
        begin=time.perf_counter();rows=[];samples=[]
        if replay:
            corpus=gz_read(root/f'samples-{it}.json.gz')
            selected=[decode_sample(s) for s in corpus['samples']]
        else:
            for case in cfg['trajectories'][2*it:2*it+2]:
                resolver=OnPolicyResolver(net,stochastic=True,collect=True,
                    seed=seed+1000+case['index'])
                ep=run_episode(load=case['load'],seed=case['input_seed'],dispatcher='RL_CRANE',
                    crane_net=net,joint_resolver=resolver,information_profile=information(),
                    demand_profile=demand('train',case['index']),**RUNTIME)
                assert ep.admitted==case['load'] and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
                stats=resolver.finish(ep.phi_krw,cfg['return_scale'])
                samples.extend(resolver.samples)
                rows.append(dict(case=case,result=ep.as_dict(),accounting=stats,cost_epochs=resolver.costs))
                print(f'train {kind} {seed} update={it} day={case["index"]} decisions={stats["samples"]}',flush=True)
            rng=torch.Generator().manual_seed(seed+2000+it)
            indices=torch.randperm(len(samples),generator=rng)[:cfg['max_samples_per_update']].tolist()
            selected=[samples[i] for i in indices]
            gz_save(root/f'samples-{it}.json.gz',dict(source_decisions=len(samples),indices=indices,
                                                     samples=[encode_sample(s) for s in selected]))
        optimization=ppo_update(net,optimizer,selected,seed=seed+3000+it,
                                max_samples=cfg['max_samples_per_update'],epochs=cfg['epochs'],batch_size=cfg['batch_size'])
        path=root/f'update-{it}.pt'
        if replay:
            expected=SchedulingActorCritic.load(path)
            assert all(torch.equal(v,expected.state_dict()[k]) for k,v in net.state_dict().items())
        else:
            net.save(path,seed=seed,iteration=it,return_scale=cfg['return_scale'])
        history.append(dict(iteration=it,days=rows,optimization=optimization,
                             selected_sha256=sha(root/f'samples-{it}.json.gz'),checkpoint_sha256=sha(path),
                             seconds=time.perf_counter()-begin))
        if not replay:
            save(root/'training.json',dict(seed=seed,kind=kind,history=history,seconds=time.perf_counter()-started))
        print(f'fit {kind} {seed} update={it} time={history[-1]["seconds"]:.1f}s {optimization}',flush=True)
    if replay:
        save(root/'replay.json',dict(passed=True,updates=len(history),seconds=time.perf_counter()-started,
             meaning='exact optimization replay from recorded on-policy actions; not independent simulation replay'))
    else:
        net.save(root/'final.pt',seed=seed,return_scale=cfg['return_scale'])


def run_eval(case, *, override_cp=None, collect=False):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from yard_rl.v7.crane.actor_critic import SchedulingActorCritic
    from yard_rl.v7.crane.on_policy import OnPolicyResolver
    from horizon_metrics import FlowProbe,aggregate
    from verify_job_completion import raw_metrics,assert_metrics
    torch.set_num_threads(1)
    cp=None;net=None;resolver=None
    if 'kind' in case:
        cp=override_cp or folder(case['kind'],case['model_seed'])/f'{case["stage"]}.pt'
        net=SchedulingActorCritic.load(cp).eval()
        resolver=OnPolicyResolver(net,stochastic=collect,seed=case['input_seed'],collect=collect)
    elif case['arm']=='old200':
        cp=Path(f'outputs/reports/yr332_context_learning/model-{case["model_seed"]}/point_steps.pt')
        net=load_net(cp)
    original,announcer=ledger.install_announcer()
    class Probe(FlowProbe):
        def finish(self,mbt,records,ep):
            for sim in mbt.blocks.values():sim.check_invariants()
            self.visits=dict(mbt.visit_stats)
            return super().finish(mbt,records,ep)
    probe=Probe(announcer);started=time.perf_counter()
    try:
        ep=episode.run_episode(load=case['load'],seed=case['input_seed'],dispatcher='RL_CRANE' if net else case['arm'],
            crane_net=net,joint_resolver=resolver,information_profile=information(),
            demand_profile=demand(case.get('split','holdout'),case['index']),evaluation_probe=probe,**RUNTIME)
    finally:episode.V3Announcer=original
    result=ep.as_dict();metrics=raw_metrics(probe.rows);assert_metrics(metrics,result)
    flows=copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'):r['finish_s']=r['yard_done_s']
    data=dict(case=case,result=result,metrics=metrics,probe=probe.result,visits=probe.visits,
        completion={f'at{h}':aggregate(flows,h*3600.) for h in (24,26)},orders_sha256=digest(probe.rows),
        schedule_sha256=digest(probe.schedule),checkpoint_sha256=sha(cp) if cp else None,seconds=time.perf_counter()-started)
    assert ep.admitted==case['load'] and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
    assert probe.result['ledger_audit']['skipped']==0 and all(r['review_count']==1 for r in probe.rows)
    assert probe.visits['late_releases']==probe.visits['late_exits']==0
    if resolver:
        data['on_policy']=dict(decisions=resolver.decisions,nontrivial=resolver.nontrivial,
                              max_candidates=resolver.max_candidates,oracle_calls=resolver.oracle_calls)
    return data,probe.rows,flows,resolver


def evaluate(i,repeat=False):
    check();case=cases()[i]
    data,orders,flows,_=run_eval(case)
    name=f'{i:03d}'+('-repeat' if repeat else '')
    for sub,value in [('evaluation',data),('orders',orders),('flow-orders',flows)]:
        path=OUT/sub/f'{name}.json.gz';assert not path.exists();gz_save(path,value)
    if repeat:
        prior=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz')
        for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
            assert prior['probe'][key]==data['probe'][key],key
        assert prior['orders_sha256']==data['orders_sha256']
    print(f'eval {name} {case} {data["result"]["mean_gate_to_job_done_s"]/60:.3f}min {data["seconds"]:.1f}s',flush=True)


def smoke():
    import torch
    from yard_rl.v7.crane.actor_critic import SchedulingActorCritic
    from yard_rl.v7.crane.on_policy import ppo_update
    torch.set_num_threads(1);torch.manual_seed(9950001)
    root=OUT/'smoke-current';root.mkdir(parents=True,exist_ok=True)
    for kind in KINDS:
        cp=root/f'{kind}.pt'
        net=SchedulingActorCritic(kind).eval();net.save(cp)
        case=dict(arm=f'{kind}_smoke',kind=kind,stage='initial',model_seed=9950001,
                  load=3500,index=23,input_seed=9950002,split='train')
        data,orders,flows,resolver=run_eval(case,override_cp=cp,collect=True)
        accounting=resolver.finish(data['result']['phi_krw'],data['result']['phi_krw'])
        start=time.perf_counter()
        optim=ppo_update(resolver.net,torch.optim.Adam(resolver.net.parameters(),lr=.0003),
                         resolver.samples,seed=9950003,max_samples=256,epochs=1,batch_size=64)
        save(root/f'{kind}.json',dict(result=data,accounting=accounting,optimization=optim,
                                    optimization_seconds=time.perf_counter()-start))
        print(f'smoke {kind} rollout={data["seconds"]:.1f}s update={time.perf_counter()-start:.1f}s samples={len(resolver.samples)}',flush=True)


def regression():
    prior_path=next(p for p in sorted(Path('outputs/reports/yr332_parametric_arrivals/evaluation').glob('*.json.gz'))
        if all(gz_read(p)['case'].get(k)==v for k,v in dict(arm='parametric',rule='SF_SPT',index=0,load=3500).items()))
    prior=gz_read(prior_path)
    case=dict(arm='SF_SPT',model_seed=0,load=3500,index=0,input_seed=prior['case']['seed'],split='train')
    data,orders,flows,_=run_eval(case)
    keys=('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256')
    checks={k:prior['probe'][k]==data['probe'][k] for k in keys}
    checks['orders']=prior['orders_sha256']==data['orders_sha256']
    assert all(checks.values()),checks
    save(OUT/'default-regression.json',dict(passed=True,checks=checks,source=prior_path.as_posix(),
                                          source_sha256=sha(prior_path),result=data))


def run(command, workers):
    tasks=[(kind,seed) for kind in KINDS for seed in SEEDS] if command=='train' else list(range(len(cases())))
    fn=train if command=='train' else evaluate
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(fn,t):t for t in tasks}
        for n,f in enumerate(as_completed(jobs),1):
            f.result();print(f'{command}: {n}/{len(tasks)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['smoke','prepare','train','evaluate','repeat','replay','regression'])
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='smoke':smoke()
    elif args.command=='regression':regression()
    elif args.command=='repeat':evaluate(3,True);evaluate(21,True)
    elif args.command=='replay':train(('mlp',SEEDS[0]),True);train(('transformer',SEEDS[0]),True)
    else:run(args.command,args.workers)
