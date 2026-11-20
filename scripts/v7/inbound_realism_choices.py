"""Read-only same-state preference audit of all saved bc validation experiences."""
from collections import defaultdict
import statistics as st
import torch

from inbound_realism_common import SOURCE, cases, read, save, sha
from global_value_samples import decode_sample, condition_sample
from value_stability import restore
from ppo_credit_metrics import first_choice, flow_id
from yard_rl.v7.crane.actor_critic import pack_states


def summarize(rows):
    n=len(rows)
    if not n: return dict(n=0)
    return dict(n=n,**{label:dict(inbound_conditional=st.mean(r[label]['inbound'] for r in rows),
        inbound_argmax=st.mean(r[label]['top']==0 for r in rows),outbound_argmax=st.mean(r[label]['top']==1 for r in rows))
        for label in ('initial','zero','global')},
        inbound_shorter_fraction=st.mean(r['shorter_inbound'] for r in rows),
        candidate_service_min={flow:st.mean(r['service'][i] for r in rows) for i,flow in enumerate(('inbound','outbound'))},
        candidate_rehandles={flow:st.mean(r['rehandles'][i] for r in rows) for i,flow in enumerate(('inbound','outbound'))})


def main():
    torch.set_num_threads(1);details=[];sources={};used=0;rawcounts=defaultdict(int)
    for seed in (9951101,9951102,9951103):
        nets={}
        for a,stage in [('initial','initial'),('zero','final'),('global','final')]:
            p=SOURCE/'policy'/f'{"zero" if a=="initial" else a}-{seed}'/f'{stage}.pt'
            nets[a]=restore(p)[0].eval();sources[str(p.relative_to(SOURCE))]=sha(p)
        for i,c in enumerate(cases()):
            if c['mode']!='stochastic' or c['seed']!=seed: continue
            p=SOURCE/'validation-samples'/f'{i:03d}.json.gz';sources[str(p.relative_to(SOURCE))]=sha(p)
            samples=[decode_sample(s) for s in read(p)['samples']];used+=len(samples)
            for begin in range(0,len(samples),32):
                batch=samples[begin:begin+32]
                with torch.no_grad():
                    predictions={a:net(pack_states([condition_sample(s,'global' if a=='global' else 'zero').state
                                                     for s in batch]))[0] for a,net in nets.items()}
                for j,s in enumerate(batch):
                    stage,mask,_=first_choice(s)
                    single=len({int(s.state.cranes[k]) for k in mask})==1
                    groups=torch.tensor([flow_id(s.state.features[k]) for k in mask])
                    mixed=bool((groups==0).any() and (groups==1).any())
                    rawcounts['samples']+=1;rawcounts['single_crane']+=single;rawcounts['mixed_single']+=single and mixed
                    if not single or not mixed: continue
                    fs=s.state.features[mask]
                    service=[float(fs[groups==g,5].mean())*60 for g in (0,1)]
                    rehandles=[float(fs[groups==g,6].mean())*30 for g in (0,1)]
                    row=dict(seed=seed,source_arm=c['arm'],source_stage=c['stage'],load=c['load'],
                        sample_index=begin+j,evaluation_index=i,decision_stage=stage,
                        shorter_inbound=service[0]<service[1],service=service,rehandles=rehandles)
                    for a,pred in predictions.items():
                        scores=pred[j,mask].double();p=torch.softmax(scores,0)
                        # Conditional probability among truck candidates; softmax again avoids underflow.
                        truck=groups<2;pt=torch.softmax(scores[truck],0)
                        row[a]=dict(top=int(groups[int(scores.argmax())]),inbound=float(pt[groups[truck]==0].sum()))
                    details.append(row)
            print('choice file',i,'states',len(samples),'mixed total',len(details),flush=True)
    summaries=[]
    for seed in (0,9951101,9951102,9951103):
        for load in (0,3500,7500,15000):
            rows=[r for r in details if (not seed or r['seed']==seed) and (not load or r['load']==load)]
            summaries.append(dict(seed=seed,load=load,**summarize(rows)))
    assert used==27*2048
    save('choices.json',dict(task='YR-332-bd',states=used,counts=dict(rawcounts),groups=summaries,sources=sources,
        source_groups=[dict(source_arm=a,source_stage=t,**summarize([r for r in details if r['source_arm']==a and r['source_stage']==t]))
                       for a,t in [('zero','initial'),('zero','final'),('global','final')]],
        scope='Same stored masks, single-crane mixed-truck subset; conditional preference, not causal contribution or measured real-terminal behavior'))
    print('choices',summaries[0],flush=True)


if __name__=='__main__': main()
