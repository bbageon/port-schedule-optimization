"""Read-only ranking audit on saved comparison inputs; never refit a model."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from benchmark_default_eval import save
from six_step_learning import gz_read, load_net, sha
from completion_learning import SEEDS, folder

OUT=Path('outputs/reports/yr332_order_representativeness')


def main():
    torch.set_num_threads(1)
    rows=[]; sources={}; appearances=Counter(); unique=defaultdict(set)
    for seed in SEEDS:
        nets={name:load_net(folder(seed)/filename) for name,filename in
              [('initial','initial.pt'),('final','final-5.pt')]}
        for name,filename in [('initial','initial.pt'),('final','final-5.pt')]:
            path=folder(seed)/filename;sources[path.as_posix()]=sha(path)
        for it in range(6):
            path=folder(seed)/f'labels-{it}.json.gz';data=gz_read(path);samples=data['samples']
            sources[path.as_posix()]=sha(path)
            for name,index in [('after_update',it),('before_update',it-1)]:
                p=folder(seed)/(f'final-{index}.pt' if index>=0 else 'initial.pt')
                nets[name]=load_net(p);sources[p.as_posix()]=sha(p)
            xs=torch.tensor([s['row'] for s in samples],dtype=torch.float32)
            with torch.no_grad():
                scores={name:net(xs).reshape(-1).tolist() for name,net in nets.items()}
            n=len(samples)//2
            order=torch.randperm(n,generator=torch.Generator().manual_seed(data['input_seed'])).tolist()
            validation=set(order[:max(1,int(n*.2))])
            for i in range(n):
                a,b=samples[2*i:2*i+2]
                assert a['decision_key']==b['decision_key']
                for s in (a,b):
                    appearances[s['flow']]+=1
                    unique[s['flow']].add((data['input_seed'],s['job']))
                gap=a['phi']-b['phi'];teacher='tie' if abs(gap)<=1e-7 else 'a' if gap<0 else 'b'
                picks={name:('tie' if values[2*i]==values[2*i+1] else 'a' if values[2*i]<values[2*i+1] else 'b') for name,values in scores.items()}
                flow=lambda pick: a['flow'] if pick=='a' else b['flow'] if pick=='b' else 'tie'
                rows.append(dict(model_seed=seed,iteration=it,input_seed=data['input_seed'],load=data['load'],
                    pair_index=i,fit_partition='same_update_validation' if i in validation else 'same_update_training',
                    flow_a=a['flow'],flow_b=b['flow'],teacher=teacher,teacher_flow=flow(teacher),gap=gap,
                    picks=picks,pick_flows={name:flow(pick) for name,pick in picks.items()},
                    correct={name:pick==teacher for name,pick in picks.items()}))
    mixed=[r for r in rows if {r['flow_a'],r['flow_b']}=={'GATE_IN','GATE_OUT'}]
    summary=dict(pairs=len(rows),candidate_appearances=dict(appearances),
        unique_order_candidates={k:len(v) for k,v in unique.items()},
        mixed_truck_pairs=len(mixed),teacher_flows=dict(Counter(r['teacher_flow'] for r in mixed)),
        network_flows={k:dict(Counter(r['pick_flows'][k] for r in mixed)) for k in ('initial','before_update','after_update','final')},
        final_IN_when_teacher_OUT=sum(r['teacher_flow']=='GATE_OUT' and r['pick_flows']['final']=='GATE_IN' for r in mixed))
    summary['by_model_update']=[]
    for seed in SEEDS:
        for it in range(6):
            group=[r for r in rows if r['model_seed']==seed and r['iteration']==it and r['teacher']!='tie']
            summary['by_model_update'].append(dict(model_seed=seed,iteration=it,nonzero_pairs=len(group),
                accuracy={k:sum(r['correct'][k] for r in group)/len(group) if group else None for k in ('initial','before_update','after_update','final')}))
    save(OUT/'saved-learning-diagnostic.json',dict(summary=summary,rows=rows,source_sha256=sources,
        scope='Saved comparison inputs only; final-model scores are not realized dispatch counts or independent generalization. No fitting or new trajectory collection.',
        script_sha256=sha(Path(__file__))))
    print(json.dumps({k:v for k,v in summary.items() if k!='by_model_update'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
