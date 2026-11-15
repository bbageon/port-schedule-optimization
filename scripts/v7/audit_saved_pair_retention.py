"""Read-only checkpoint scoring, with original decision-level validation splits."""
from collections import Counter, defaultdict
from pathlib import Path

from six_step_learning import gz_read, load_net, sha
from yard_rl.integrated.repro import repro_stamp, write_result

ROOT=Path('outputs/reports/yr332_linked_retraining')
OUT=Path('outputs/reports/yr332_network_review/saved-pair-retention.json')
PLAN=Path('docs/research/v7-discharge/saved-pair-audit-plan.md')
SEEDS=(9917101,9917102,9917103)


def main():
    import torch
    torch.set_num_threads(1)
    paths=[Path(__file__),PLAN,Path('src/yard_rl/v7/crane/fit.py'),
           Path('src/yard_rl/v7/crane/policy.py'),Path('scripts/v7/six_step_learning.py')]
    paths += [p for s in SEEDS for p in (ROOT/f'model-{s}').glob('*') if p.suffix in ('.gz','.pt','.json')]
    hashes={p.as_posix():sha(p) for p in paths}
    rows=[]; splits=[]
    for seed in SEEDS:
        folder=ROOT/f'model-{seed}'
        import json
        history=json.loads((folder/'training.json').read_text(encoding='utf-8'))['rows']
        initial,final=load_net(folder/'initial.pt'),load_net(folder/'final-5.pt')
        initial.eval();final.eval()
        for it in range(6):
            data=gz_read(folder/f'labels-{it}.json.gz');samples=data['samples'];n=len(samples)//2
            assert len(samples)==2*n
            x=torch.tensor([s['row'] for s in samples],dtype=torch.float32)
            y=torch.tensor([s['target'] for s in samples],dtype=torch.float32)
            gaps=y[::2]-y[1::2]
            rng=torch.Generator().manual_seed(data['input_seed'])
            val=set(torch.randperm(n,generator=rng).tolist()[:max(1,int(n*.2))])
            report=history[it]['fit']
            assert report['n']==2*(n-len(val)) and report['n_val']==2*len(val)
            after=load_net(folder/f'final-{it}.pt');after.eval()
            predicted={}
            with torch.no_grad():
                for name,net in [('initial',initial),('after',after),('final',final)]:
                    values=net(x);predicted[name]=values[::2]-values[1::2]
            actual_valid=[];correct_valid=[]
            for d in range(n):
                a,b=samples[2*d:2*d+2]
                assert a['decision_key']==b['decision_key']
                gap=float(gaps[d]);active=abs(gap)>1e-9
                rawgap=a['phi']-b['phi']
                assert not active or (rawgap>0)==(gap>0)
                ps={k:float(v[d]) for k,v in predicted.items()}
                correct={k: bool(torch.sign(v[d])==torch.sign(gaps[d])) for k,v in predicted.items()}
                if d in val and active:
                    actual_valid.append(d);correct_valid.append(correct['after'])
                rows.append(dict(model_seed=seed,iteration=it,pair=d,split='validation' if d in val else 'train',
                    flows=[a['flow'],b['flow']],flow_pair=' / '.join(sorted((a['flow'],b['flow']))),
                    target_gap=gap,raw_cost_gap=rawgap,active=active,predicted=ps,correct=correct,
                    identical_inputs=bool(torch.equal(x[2*d],x[2*d+1])),
                    input_seed=data['input_seed'],decision_key=a['decision_key']))
            accuracy=sum(correct_valid)/len(correct_valid) if correct_valid else 0.
            assert abs(accuracy-report['val_order_accuracy'])<1e-6
            assert len(actual_valid)==report['val_nonzero_pairs']
            splits.append(dict(model_seed=seed,iteration=it,train=n-len(val),validation=len(val),
                               recreated_validation_accuracy=accuracy))

    def summarize(group):
        active=[r for r in group if r['active']]
        return dict(pairs=len(group),nonzero_pairs=len(active),
            correct={k:sum(r['correct'][k] for r in active) for k in ('initial','after','final')},
            indistinguishable_nonzero=sum(r['identical_inputs'] for r in active))

    earlier=[r for r in rows if r['split']=='train' and r['iteration']<5 and r['active']]
    retention=dict(nonzero_earlier_train_pairs=len(earlier),
        retained=sum(r['correct']['after'] and r['correct']['final'] for r in earlier),
        lost=sum(r['correct']['after'] and not r['correct']['final'] for r in earlier),
        gained=sum(not r['correct']['after'] and r['correct']['final'] for r in earlier),
        neither=sum(not r['correct']['after'] and not r['correct']['final'] for r in earlier))
    summary=dict(all=summarize(rows),train=summarize([r for r in rows if r['split']=='train']),
        validation=summarize([r for r in rows if r['split']=='validation']),retention=retention,
        by_model={str(s):summarize([r for r in rows if r['model_seed']==s]) for s in SEEDS},
        by_flow={f:summarize([r for r in rows if r['flow_pair']==f]) for f in sorted({r['flow_pair'] for r in rows})})
    assert hashes=={p.as_posix():sha(p) for p in paths},'Saved model or label files changed during read-only scoring'
    write_result(OUT,dict(summary=summary,splits=splits,rows=rows,sources=hashes,
        scope='read-only saved-label diagnosis; no optimization or new environment run; not operational performance',
        repro=repro_stamp(experiment='YR-323-d-read-only',seeds=dict(models=list(SEEDS),
            inputs=sorted({r['input_seed'] for r in rows})),params={'configuration':dict(
                dtype='float32',target_tolerance=1e-9,validation_fraction=.2,
                checkpoints=['initial','after-own-update','final-5'],training_performed=False)},prereg=PLAN.as_posix())))
    print(summary)


if __name__=='__main__': main()
