"""Trace the first 9-vs-zero-extended-10 input initialization disagreement."""
import copy
import json
from pathlib import Path

import torch

from count_transform_study import OUT,PRIOR,REFERENCE,load_net,gz_read,save,sha
from yard_rl.v7.evaluation_probe import EvaluationProbe
from yard_rl.v7.crane.policy import CranePolicy
from yard_rl.v7.crane.features import crane_features
from yard_rl.v7.world.integrated.baselines import ResolverPolicy
from yard_rl.v7.stage.episode import run_episode


class Found(Exception):
    pass


def main():
    torch.set_num_threads(1)
    source=PRIOR/'model-9913501/initial.pt'
    extended=OUT/'models/clipped-9913501-initial.pt'
    a,b=load_net(source),load_net(extended)
    assert b.net[0].weight[:,9].count_nonzero()==0
    assert torch.equal(a.net[0].weight,b.net[0].weight[:,:9])
    for k,v in a.state_dict().items():
        if k!='net.0.weight':assert torch.equal(v,b.state_dict()[k])
    nine=gz_read(REFERENCE/'evaluation/initial-9913501-15000.json.gz')
    ten=gz_read(OUT/'evaluation/clipped-9913501-initial-15000.json.gz')
    first=next(i for i,(x,y) in enumerate(zip(nine['probe']['decision_context'],ten['probe']['decision_context'])) if x!=y)
    until=max(nine['probe']['decision_context'][first][0],ten['probe']['decision_context'][first][0])
    start=max(0.,until-3600.)
    d64a,d64b=copy.deepcopy(a).double(),copy.deepcopy(b).double()
    class Probe(EvaluationProbe):
        def before_select(self,sim,dp,gb,pref):
            if sim.clock<start:return
            assert sim.clock<=until+1.,'no choice disagreement found before observed trajectory divergence'
            left=ResolverPolicy(pref).decide(sim,dp,gb)
            right_pref=CranePolicy(b)
            right=ResolverPolicy(right_pref).decide(sim,dp,gb)
            ids=lambda chosen:{c:g.job_ref.job_id if g.job_ref else None for c,g in chosen.items()}
            if ids(left)==ids(right):return
            pairs=[(c,g) for c in dp.crane_ids for g in gb[c].items if g.feasible]
            lr,rr=pref.rank_many(sim,pairs),right_pref.rank_many(sim,pairs)
            affected={(c,g.job_ref.job_id) for assignment in (left,right) for c,g in assignment.items() if g.job_ref}
            rows=[]
            for (c,g),old_score,new_score in zip(pairs,lr,rr):
                if not g.job_ref or (c,g.job_ref.job_id) not in affected:continue
                x=crane_features(sim,c,g,feature_mode='feasible')
                y=crane_features(sim,c,g,feature_mode='feasible',include_flow=True)
                assert x==y[:9]
                with torch.no_grad():
                    s64a=float(d64a(torch.tensor([x],dtype=torch.float64))[0])
                    s64b=float(d64b(torch.tensor([y],dtype=torch.float64))[0])
                rows.append(dict(crane=c,job=g.job_ref.job_id,row9=x,row10=y,
                    rank9=old_score,rank10=new_score,double9=s64a,double10=s64b))
            dest=OUT/'initial-replay-diagnosis.json'
            save(dest,dict(status='DIAGNOSED',source=str(source),source_sha256=sha(source),
                extended=str(extended),extended_sha256=sha(extended),shared_weights_exact=True,
                tenth_weight_zero=True,shared_features_exact=True,time=sim.clock,
                block=self.block_names[id(sim)],assignment9=ids(left),assignment10=ids(right),
                rows=rows,first_context_difference_index=first,observed_context_time=until,
                maximum_double_output_difference=max(abs(r['double9']-r['double10']) for r in rows),
                source_code='scripts/v7/diagnose_initial_count_replay.py',source_code_sha256=sha(__file__)))
            print(json.dumps(readable(rows,sim.clock,ids(left),ids(right))),flush=True)
            raise Found()
    try:
        run_episode(load=15000,seed=9913920,dispatcher='RL_CRANE',crane_net=a,candidate_cap=None,
            evaluation_probe=Probe(),reuse_rehandle_slots=False,decision_probe_shortcircuit=True)
    except Found:
        return
    raise AssertionError('expected disagreement was not observed')


def readable(rows,t,a,b):
    return dict(time=t,assignment9=a,assignment10=b,rows=[{k:r[k] for k in (
        'crane','job','rank9','rank10','double9','double10')} for r in rows])


if __name__=='__main__':main()
