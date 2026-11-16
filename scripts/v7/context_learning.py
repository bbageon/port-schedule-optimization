"""Registered synthetic-day label collection, budget diagnostics, and context networks."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import statistics as st
import subprocess
import time

from audit_day_shapes import INFO, read
from completion_learning import LOADS, RUNTIME, SEEDS
from six_step_learning import checkpoint, gz_read, gz_save, load_net, sha
from yard_rl.integrated.repro import write_result, repro_stamp
from yard_rl.v7.evaluation_probe import digest

OUT=Path('outputs/reports/yr332_context_learning')
PREREG=Path('docs/research/v7-discharge/context-learning-prereg.md')
ARMS=('point_current','point_replay','point_half','point_steps','mean_context','star_attention')
RULES=('SF_SPT','FIFO','LWKR')


def save(name,value):write_result(OUT/name,value)


def demand(split,index):
    from synthetic_arrivals import profile
    return profile(dict(split=split,index=index,arm='parametric',axis='minutes'))


def collection_cases():
    return [dict(split=split,index=i,load=LOADS[i%3],seed=start+i) for split,size,start in
            [('train',6,9940100),('selection',3,9940200)] for i in range(size)]


def evaluation_cases():
    cases=[dict(arm=arm,model_seed=seed,index=i,load=load,input_seed=9940300+i)
           for i,load in enumerate(LOADS) for arm in ('initial',*ARMS) for seed in SEEDS]
    cases += [dict(arm=rule,model_seed=0,index=i,load=load,input_seed=9940300+i)
              for i,load in enumerate(LOADS) for rule in RULES]
    return cases


def prepare():
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    assert not (OUT/'campaign.json').exists()
    from synthetic_arrivals import input_case
    input_cases=[dict(c,axis='minutes',arm='parametric') for c in collection_cases()]
    input_cases += [dict(axis='minutes',arm='parametric',split='holdout',index=i,load=load,seed=9940300+i)
                    for i,load in enumerate(LOADS)]
    preflight=[input_case(c) for c in input_cases]
    save('input-preflight.json',dict(records=preflight,cases=len(preflight),all_passed=True))
    cfg=dict(collection=collection_cases(),evaluation=evaluation_cases(),max_questions=48,horizon_s=10800,
             behavior_seed=SEEDS[0],model_seeds=SEEDS,arms=ARMS,steps_per_update=[50,200],runtime=RUNTIME,
             collection_parent_workers=3,branch_workers_per_parent=3,
             normalization='median absolute gap from training half only; floor 1000',claim_scope='limited synthetic diagnostic')
    paths=[Path(__file__),PREREG,INFO,Path('scripts/v7/synthetic_arrivals.py'),
        Path('outputs/reports/yr332_synthetic_inputs/bank.json'),
        Path('outputs/reports/yr332_synthetic_inputs/multiscale-minute-calibration.json'),
        *Path('src/yard_rl/v7').rglob('*.py')]
    save('campaign.json',dict(configuration=cfg,sources={p.as_posix():sha(p) for p in paths},
        repro=repro_stamp(experiment='YR-323-d/332-as',seeds=dict(models=list(SEEDS),
        collection=[c['seed'] for c in collection_cases()],evaluation=[9940300,9940301,9940302]),
        params=cfg,prereg=PREREG.as_posix())))


def check():
    d=read(OUT/'campaign.json')
    for p,h in d['sources'].items():assert sha(Path(p))==h,('changed source',p)
    return d


def collect(i):
    import torch
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    from yard_rl.v7.crane.policy import CraneNet,add_store_input
    torch.set_num_threads(1);check();case=collection_cases()[i];torch.manual_seed(SEEDS[0])
    net=add_store_input(CraneNet(feature_mode='feasible'));net.capture_context=True
    started=time.perf_counter()
    ep=run_episode(load=case['load'],seed=case['seed'],dispatcher='RL_CRANE',crane_net=net,
        information_profile=read(INFO),demand_profile=demand(case['split'],case['index']),
        crane_budget=RolloutBudget(max_labels=48,identity_checks=0),horizon_s=10800.,workers=3,
        crane_sampling='coverage',crane_alternative_contract='reachable',**RUNTIME)
    data=dict(case=case,result=ep.as_dict(),samples=[asdict(s) for s in ep.crane_labels],seconds=time.perf_counter()-started)
    path=OUT/'corpora'/f'{i:02d}.json.gz';assert not path.exists();gz_save(path,data)
    stats=ep.crane_stats
    assert ep.admitted==case['load'] and ep.policy_exceptions==stats['hook_errors']==stats['factual_mismatch']==stats['force_failed']==0
    contexts=stats['candidate_contexts'];assert len(contexts)*2==len(data['samples']) and len(contexts)>=4
    for j,context in enumerate(contexts):
        assert len(context['rows'])==len(context['jobs'])
        assert context['rows'][context['pair'][0]]==data['samples'][2*j]['row']
        assert context['rows'][context['pair'][1]]==data['samples'][2*j+1]['row']
    print(f'collect {i} {case} pairs={len(contexts)} seconds={data["seconds"]:.1f}',flush=True)


def examples(data):
    result=[]
    for j,c in enumerate(data['result']['crane']['candidate_contexts']):
        a,b=data['samples'][2*j:2*j+2]
        assert list(a['decision_key'])==c['key']
        result.append(dict(rows=c['rows'],pair=c['pair'],gap=a['phi']-b['phi'],key=c['key'],flow=[a['flow'],b['flow']]))
    return result


def pack(examples_,scale):
    import torch
    rows=[];groups=[];pairs=[];target=[]
    for i,e in enumerate(examples_):
        offset=len(rows);rows.extend(e['rows']);groups.extend([i]*len(e['rows']))
        pairs.append([offset+j for j in e['pair']]);target.append(e['gap']/scale)
    return (torch.tensor(rows,dtype=torch.float32),torch.tensor(groups,dtype=torch.long),
            torch.tensor(pairs,dtype=torch.long),torch.tensor(target,dtype=torch.float32))


def predict(net,batch):
    x,g,p,y=batch
    scores=net(x,g) if getattr(net,'requires_candidate_context',False) else net(x)
    return scores[p[:,0]]-scores[p[:,1]],y


def measure(net,items,scale):
    import torch
    from torch.nn.functional import smooth_l1_loss
    net.eval()
    with torch.no_grad():
        pred,y=predict(net,pack(items,scale));nz=y.abs()>1e-9
        return dict(pairs=len(y),nonzero=int(nz.sum()),correct=int((pred[nz].sign()==y[nz].sign()).sum()),
                    accuracy=float((pred[nz].sign()==y[nz].sign()).float().mean()) if nz.any() else None,
                    loss=float(smooth_l1_loss(pred,y)))


def train(seed,repeat=False):
    import torch
    from torch import nn
    from yard_rl.v7.crane.policy import CraneNet,add_store_input
    from yard_rl.v7.crane.context import ContextCraneNet
    torch.set_num_threads(1);check();started=time.perf_counter()
    corpora=[gz_read(OUT/'corpora'/f'{i:02d}.json.gz') for i in range(9)]
    days=[examples(c) for c in corpora[:6]];validation=[e for c in corpora[6:] for e in examples(c)]
    halves=[]
    for i,day in enumerate(days):
        rng=torch.Generator().manual_seed(collection_cases()[i]['seed'])
        indices=torch.randperm(len(day),generator=rng).tolist()[:max(1,len(day)//2)]
        halves.append([day[j] for j in indices])
    scale=max(1000.,st.median(abs(e['gap']) for day in halves for e in day))
    torch.manual_seed(seed);initial=add_store_input(CraneNet(feature_mode='feasible'))
    folder=OUT/(f'model-{seed}'+('-repeat' if repeat else ''))
    checkpoint(folder/'initial.pt',initial,model_seed=seed)
    rows=[]
    for arm in ARMS:
        torch.manual_seed(seed)
        net=(ContextCraneNet(arm) if arm in ('mean_context','star_attention')
             else copy.deepcopy(initial))
        opt=torch.optim.Adam(net.parameters(),lr=.0003);seen=[];history=[]
        initial_validation=measure(net,validation,scale)
        for it,full in enumerate(days):
            current=halves[it] if arm=='point_half' else full;seen.extend(current)
            pool=current if arm=='point_current' else seen
            rng=torch.Generator().manual_seed(9941000+100*SEEDS.index(seed)+it)
            steps=200 if arm=='point_steps' else 50;net.train()
            for _ in range(steps):
                indices=torch.randint(len(pool),(128,),generator=rng).tolist()
                batch=pack([pool[j] for j in indices],scale);pred,y=predict(net,batch)
                loss=nn.functional.smooth_l1_loss(pred,y)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            history.append(dict(iteration=it,train=measure(net,pool,scale),validation=measure(net,validation,scale),steps=steps))
        path=folder/f'{arm}.pt';assert not path.exists()
        checkpoint(path,net,model_seed=seed,diagnostic=arm,scale=scale)
        if repeat:
            old=load_net(OUT/f'model-{seed}'/f'{arm}.pt').state_dict()
            assert all(torch.equal(v,old[k]) for k,v in net.state_dict().items())
        rows.append(dict(arm=arm,parameters=sum(p.numel() for p in net.parameters()),scale=scale,
                         initial_validation=initial_validation,
                         train=measure(net,[e for d in days for e in d],scale),validation=measure(net,validation,scale),
                         history=history,checkpoint_sha256=sha(path)))
        write_result(folder/'training.json',dict(rows=rows,seed=seed,repeat_exact=repeat,seconds=time.perf_counter()-started,
             corpus_sha256={str(i):sha(OUT/'corpora'/f'{i:02d}.json.gz') for i in range(9)}))
        print(f'fit {seed} {arm} val={rows[-1]["validation"]}',flush=True)


def evaluate(i,repeat=False):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe,aggregate
    from verify_job_completion import raw_metrics,assert_metrics
    torch.set_num_threads(1);check();case=evaluation_cases()[i]
    cp=OUT/f'model-{case["model_seed"]}'/f'{case["arm"]}.pt' if case['model_seed'] else None
    original,announcer=ledger.install_announcer()
    class Probe(FlowProbe):
        def finish(self,mbt,records,ep):
            for sim in mbt.blocks.values():sim.check_invariants()
            self.visits=dict(mbt.visit_stats)
            return super().finish(mbt,records,ep)
    probe=Probe(announcer);started=time.perf_counter()
    try:
        ep=episode.run_episode(load=case['load'],seed=case['input_seed'],dispatcher='RL_CRANE' if cp else case['arm'],
            crane_net=load_net(cp) if cp else None,information_profile=read(INFO),demand_profile=demand('holdout',case['index']),
            evaluation_probe=probe,**RUNTIME)
    finally:episode.V3Announcer=original
    result=ep.as_dict();metrics=raw_metrics(probe.rows);assert_metrics(metrics,result)
    flows=copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'):r['finish_s']=r['yard_done_s']
    data=dict(case=case,result=result,metrics=metrics,probe=probe.result,visits=probe.visits,
        completion={f'at{h}':aggregate(flows,h*3600.) for h in (24,26)},orders_sha256=digest(probe.rows),
        schedule_sha256=digest(probe.schedule),checkpoint_sha256=sha(cp) if cp else None,seconds=time.perf_counter()-started)
    name=f'{i:03d}'+('-repeat' if repeat else '')
    for sub,value in [('evaluation',data),('orders',probe.rows),('flow-orders',flows)]:
        path=OUT/sub/f'{name}.json.gz';assert not path.exists();gz_save(path,value)
    assert ep.admitted==case['load'] and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
    assert probe.result['ledger_audit']['skipped']==0 and all(r['review_count']==1 for r in probe.rows)
    assert probe.visits['late_releases']==probe.visits['late_exits']==0
    if repeat:
        old=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz')
        assert all(old['probe'][k]==data['probe'][k] for k in
                   ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'))
        assert old['orders_sha256']==data['orders_sha256']
    print(f'eval {name} {case} {result["mean_gate_to_job_done_s"]/60:.3f}min {data["seconds"]:.1f}s',flush=True)


def run(kind,workers):
    check();values=list(range(9)) if kind=='collect' else list(SEEDS) if kind=='train' else list(range(len(evaluation_cases())))
    fn={'collect':collect,'train':train,'evaluate':evaluate}[kind]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,i):i for i in values}
        for n,f in enumerate(as_completed(futures),1):f.result();print(f'{kind}: {n}/{len(values)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','collect','train','evaluate','repeat','repeat-training'])
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='repeat':evaluate(6,True);evaluate(18,True)
    elif a.command=='repeat-training':train(SEEDS[0],True)
    else:run(a.command,a.workers)
