"""YR-332-w: replay frozen training states and audit selectable comparison actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import count_transform_study as frozen
from six_step_learning import gz_read,gz_save,sha,load_net
from benchmark_default_eval import save
from action_reachability import reachable,replay_certificate
from yard_rl.v7.crane.features import crane_features
from yard_rl.v7.crane.policy import CranePolicy
from yard_rl.v7.world.integrated.baselines import ResolverPolicy

OUT=Path('outputs/reports/yr332_action_contract')
PRIOR=frozen.PRIOR
PREREG=Path('docs/research/v7-discharge/action-contract-prereg.md')
CASES=[(s,i,0) for s in frozen.SEEDS for i in range(3)]+[(frozen.SEEDS[0],0,1)]
WORKERS=6


def name(case):
    s,i,r=case
    return f'{s}-{i}'+('-repeat' if r else '')


def source_hashes():
    files=[Path(__file__),Path('scripts/v7/action_reachability.py'),
        Path('scripts/v7/report_action_contract.py'),Path('scripts/v7/count_transform_study.py'),
        Path('scripts/v7/six_step_learning.py'),Path('scripts/v7/evaluate_classical_rules.py'),
        Path('scripts/v7/evaluate_arrival_order.py'),Path('scripts/v7/benchmark_default_eval.py'),
        PREREG,*Path('src/yard_rl/v7').rglob('*.py')]
    return {p.resolve().relative_to(Path.cwd()).as_posix():sha(p) for p in files}


def index_of(pairs,cid,gc):
    return next(i for i,(c,g) in enumerate(pairs) if c==cid and g.candidate_id==gc.candidate_id)


class WitnessPreference:
    """A supplied score table, with the unchanged mandatory and SERVE tiers."""
    def __init__(self,pairs,items,scores,shared=None):
        self.index={(c,g.candidate_id):i for i,(c,g) in enumerate(pairs)}
        self.items=items;self.scores=scores;self.shared=shared
        self.by_features={}
        if shared:
            for i,r in enumerate(items):
                if r['row10'] is None:continue
                key=self.feature_key(r)
                self.by_features[key]=min(self.by_features.get(key,0.),scores.get(i,0.))
    def feature_key(self,row):
        values=list(row['row10'])
        if self.shared=='linear':values[7]=row['raw_count']/18.
        return tuple(values)
    def rank(self,sim,crane,gc):
        i=self.index[(crane,gc.candidate_id)];r=self.items[i]
        if gc.job_ref is None:return (2,0.,'')
        score=self.by_features[self.feature_key(r)] if self.shared else self.scores.get(i,0.)
        return (r['tier'][1],score,gc.job_ref.job_id)


def score_witness(sim,dp,gb,pairs,items,target,proof,shared=None):
    scores={i:-3. for i in proof.get('rejected_prefix',[])}
    scores[proof['witness'][0]]=-2.;scores[target]=-1.
    policy=ResolverPolicy(WitnessPreference(pairs,items,scores,shared))
    assignment=policy.decide(sim,dp,gb)
    chosen={c:index_of(pairs,c,g) for c,g in assignment.items()}
    return dict(target_selected=chosen.get(items[target]['crane'])==target,
                chosen=chosen,scores={str(k):v for k,v in scores.items()},shared_features=shared)


class ContractProbe(frozen.CountProbe):
    def __init__(self,data4,data5):
        super().__init__(data5['samples'])
        self.sources={4:data4,5:data5}
        self.targets_by_decision={}
        self.rows=[];self.graphs=[];self.visited=set()
        for stage,d in self.sources.items():
            for i in range(0,len(d['samples']),2):
                a,b=d['samples'][i:i+2]
                assert a['action']=='PICKED' and b['action']=='ALT'
                assert a['decision_key']==b['decision_key']
                self.targets_by_decision.setdefault(tuple(a['decision_key']),[]).append((stage,i//2,a,b))

    def before_select(self,sim,dp,gb,pref):
        super().before_select(sim,dp,gb,pref)
        block=self.block_names[id(sim)]
        wanted=[(key,r) for c in dp.crane_ids for key in [(round(sim.clock,6),block,c)]
                for r in self.targets_by_decision.get(key,[])]
        if not wanted:return
        assert len(dp.crane_ids)<=2
        pairs=[(c,g) for c in dp.crane_ids for g in gb[c].items if g.feasible]
        cache={}
        def compatible(*indices):
            key=tuple(sorted(indices))
            if key not in cache:
                choice={pairs[i][0]:pairs[i][1] for i in indices}
                if len(choice)!=len(indices):cache[key]=False
                else:
                    projected=sim.dry_run_commit({c:g.job_ref for c,g in choice.items()})
                    work={c for c,g in choice.items() if g.job_ref is not None}
                    cache[key]=set(projected.plans)==work
            return cache[key]
        ranks=pref.rank_many(sim,pairs)
        items=[]
        for i,((c,g),rank) in enumerate(zip(pairs,ranks)):
            items.append(dict(crane=c,candidate_id=g.candidate_id,kind=g.kind.value,
                job=g.job_ref.job_id if g.job_ref else None,token=g.job_ref.token if g.job_ref else None,
                mandatory=g.mandatory,tier=[0 if g.mandatory else 1,rank[0]],
                single_ok=compatible(i),raw_count=g.feasible_serve_count,
                row10=crane_features(sim,c,g,feature_mode='feasible',include_flow=True) if g.job_ref else None))
        normal=ResolverPolicy(pref).decide(sim,dp,gb)
        graph_id=len(self.graphs)
        graph=dict(time=sim.clock,block=block,cranes=list(dp.crane_ids),items=items,oracle={})
        for key,(stage,pair_index,picked,alt) in wanted:
            assert (stage,pair_index) not in self.visited
            self.visited.add((stage,pair_index))
            cid=key[2]
            def locate(sample):
                matches=[i for i,((c,g),r) in enumerate(zip(pairs,items)) if c==cid and r['job']==sample['job']
                    and r['row10'][:9]==sample['row'] and r['row10'][9]==float(sample['is_store'])]
                assert len(matches)==1,(key,sample['job'],len(matches))
                return matches[0]
            pi,ai=locate(picked),locate(alt)
            assert index_of(pairs,cid,normal[cid])==pi
            assert items[pi]['mandatory']==items[ai]['mandatory']
            proof=reachable(items,ai,compatible)
            replay_certificate(items,ai,proof,compatible)
            witnesses={}
            if proof['reachable']:
                for shared in (None,'clipped','linear'):
                    witnesses[shared or 'independent']=score_witness(sim,dp,gb,pairs,items,ai,proof,shared)
                assert witnesses['independent']['target_selected'],(key,proof)
            forced=CranePolicy(pref.net,serve_first=pref.serve_first,batched=pref.batched)
            forced.force_once[cid]=alt['job']
            assignment=ResolverPolicy(forced).decide(sim,dp,gb)
            force_success=index_of(pairs,cid,assignment[cid])==ai
            assert force_success,(key,'frozen successful force did not replay')
            joint=[index_of(pairs,c,g) for c,g in assignment.items()]
            assert compatible(*joint)
            self.rows.append(dict(stage=stage,pair_index=pair_index,key=key,graph=graph_id,picked=pi,alternative=ai,
                phi_picked=picked['phi'],phi_alternative=alt['phi'],
                target_picked=picked['target'],target_alternative=alt['target'],
                mixed_service_tier=items[pi]['tier'][1]!=items[ai]['tier'][1],
                current_pick_tier=items[pi]['tier'],alternative_tier=items[ai]['tier'],
                proof=proof,score_witnesses=witnesses,forced_selected=force_success,
                forced_joint_safe=True,forced_assignment=joint,forced_rank=list(forced.rank(sim,cid,pairs[ai][1]))))
        graph['oracle']={'/'.join(map(str,k)):v for k,v in sorted(cache.items())}
        self.graphs.append(graph)


def evaluate(index):
    import torch
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    seed,it,repeat=CASES[index];root=PRIOR/f'model-{seed}'
    sources={stage:root/f'labels-{stage}-{it}.json.gz' for stage in (4,5)}
    ds={k:gz_read(p) for k,p in sources.items()}
    behavior=root/f'behavior-{it}.pt'
    assert all(sha(behavior)==d['behavior_sha256'] for d in ds.values())
    probe=ContractProbe(ds[4],ds[5]);net=load_net(behavior)
    start=time.perf_counter()
    result=run_episode(load=ds[5]['load'],seed=ds[5]['seed'],dispatcher='RL_CRANE',crane_net=net,
        candidate_cap=None,evaluation_probe=probe,reuse_rehandle_slots=False,decision_probe_shortcircuit=True).as_dict()
    assert len(probe.visited)==sum(len(d['samples'])//2 for d in ds.values())
    assert len(probe.recovered)==len(ds[5]['samples'])
    ignored={'crane','n_crane_labels'}
    for d in ds.values():
        assert probe.result['main_choices_sha256']==d['result']['crane']['main_choices_sha256']
        assert {k:v for k,v in result.items() if k not in ignored}=={k:v for k,v in d['result'].items() if k not in ignored}
    recovered=gz_read(frozen.recovery_path(seed,it))
    for key in frozen.HASHES:assert probe.result[key]==recovered['probe'][key],key
    data=dict(case=CASES[index],input_seed=ds[5]['seed'],load=ds[5]['load'],wall_seconds=time.perf_counter()-start,
        source_sha256={p.as_posix():sha(p) for p in sources.values()},behavior_sha256=sha(behavior),
        result=result,probe=probe.result,pairs=sorted(probe.rows,key=lambda r:(r['stage'],r['pair_index'])),
        graphs=probe.graphs,exact_features=True,exact_original_choices=True,exact_original_result=True,
        original_recovery_sha256=sha(frozen.recovery_path(seed,it)))
    gz_save(OUT/'evaluation'/f'{name(CASES[index])}.json.gz',data)
    print(f'{name(CASES[index])}: pairs={len(probe.rows)}; unreachable={sum(not r["proof"]["reachable"] for r in probe.rows)}',flush=True)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not frozen.git('status','--short','--untracked-files=no'),'commit before campaign'
    inputs=set()
    for seed,it,_ in CASES:
        root=PRIOR/f'model-{seed}'
        inputs.update([root/f'labels-{s}-{it}.json.gz' for s in (4,5)])
        inputs.update([root/f'behavior-{it}.pt',frozen.recovery_path(seed,it)])
        inputs.update([root/f'{a}-{it}.pt' for a in ('pairwise','coverage','flow')])
        inputs.update([root/f'stage-{s}.json' for s in (4,5,6)])
        inputs.add(frozen.model_path('linear',seed,f'iteration-{it}'))
    config=dict(cases=CASES,workers=WORKERS,torch_threads=1,new_training=False,new_teacher_branches=False,
        unchanged_priority=['mandatory','serve_first','joint_safety','unique_job'],
        reachability='Exact greedy upper bound with at most two deciding cranes and arbitrary per-candidate scores',
        shared_score_witness='same 10-input vector has same score; no finite-network realizability claim')
    save(OUT/'campaign.json',dict(task='YR-332-w',configuration=config,sources=source_hashes(),
        input_sha256={p.as_posix():sha(p) for p in sorted(inputs)},
        repro=repro_stamp(experiment='YR-332-w',seeds=dict(model=list(frozen.SEEDS),
            input=[9913600+100*j+10*i for j in range(3) for i in range(3)]),params=config,prereg=PREREG.as_posix())))


def campaign():
    prepare()
    def one(i):
        log=OUT/'logs'/f'{i}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,__file__,'--operation','one','--index',str(i)],stdout=f,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if p.returncode:raise RuntimeError(log.read_text(encoding='utf-8')[-3500:])
        return i
    order=list(range(len(CASES)));random.Random(9914501).shuffle(order);done=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for future in as_completed([pool.submit(one,i) for i in order]):
            done.append(future.result());save(OUT/'progress.json',dict(completed=done,total=len(CASES)))
            print(f'action contract: {len(done)}/{len(CASES)}',flush=True)
    from report_action_contract import report
    report()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=('campaign','one'),default='campaign')
    p.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.operation=='campaign':campaign()
    else:evaluate(a.index)
