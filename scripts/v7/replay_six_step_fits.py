"""Replay retained learning corpora and require exact checkpoint reproduction."""
import argparse
import copy
import hashlib
from pathlib import Path

from benchmark_default_eval import save
from six_step_learning import MODEL_SEEDS, gz_read, labelset, load_net


def main(folder):
    import torch
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.crane.policy import add_store_input

    torch.set_num_threads(1)
    root=Path(folder)
    checks=[]
    for model_seed in MODEL_SEEDS:
        base=root/f'model-{model_seed}'
        initial=load_net(base/'initial.pt')
        for arm,stage in (('pointwise',4),('pairwise',4),('coverage',5),('flow',6)):
            if not (base/f'{arm}-2.pt').exists():
                continue
            net=add_store_input(initial) if arm=='flow' else copy.deepcopy(initial)
            trainer=CraneTrainer(net,loss_kind='pointwise' if arm=='pointwise' else 'pairwise')
            for it in range(3):
                corpus=base/f'labels-{4 if stage==4 else 5}-{it}.json.gz'
                data=gz_read(corpus)
                ls=labelset(data)
                if arm=='flow':
                    for s in ls.samples:
                        s.row.append(float(s.is_store))
                rep=trainer.fit(ls,seed=data['seed'])
                path=base/f'{arm}-{it}.pt'
                checkpoint=torch.load(path,weights_only=True)
                assert checkpoint['corpus_sha256']==hashlib.sha256(corpus.read_bytes()).hexdigest()
                equal=all(torch.equal(t,checkpoint['crane'][k]) for k,t in net.state_dict().items())
                checks.append(dict(model_seed=model_seed,arm=arm,iteration=it,
                    steps=rep.steps,exact_weights=equal,checkpoint=path.as_posix(),
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    corpus_sha256=checkpoint['corpus_sha256']))
                assert equal,checks[-1]
    result=dict(checks=checks,all_exact=bool(checks) and all(c['exact_weights'] for c in checks),
                scope='fit reproducibility from retained corpora; does not repeat environment collection')
    save(root/'fit-replay-verification.json',result)
    print(f'{len(checks)} checkpoint updates reproduced with exactly identical weights')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    main(p.parse_args().folder)
