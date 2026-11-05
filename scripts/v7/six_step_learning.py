"""Controlled learning experiments: common behaviour, fixed labels and held-out days."""
import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from benchmark_default_eval import save, memory

LOADS = (3500, 7500, 15000)
MODEL_SEEDS = (9913501, 9913502, 9913503)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gz_save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False)


def gz_read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def checkpoint(path, net, trainer=None, **metadata):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(crane=net.state_dict(), feature_mode=net.feature_mode,
                    dim=net.net[0].in_features, hidden=net.net[0].out_features,
                    include_flow=getattr(net,'include_flow',False),
                    opt=None if trainer is None else trainer.opt.state_dict(),
                    loss_kind=None if trainer is None else trainer.loss_kind, **metadata), path)


def load_net(path):
    import torch
    from yard_rl.v7.crane.policy import CraneNet
    ck = torch.load(path, weights_only=True)
    kwargs = dict(dim=ck['dim'], hidden=ck['hidden'], feature_mode=ck['feature_mode'])
    if ck.get('include_flow',False):
        kwargs['include_flow'] = True
    net = CraneNet(**kwargs)
    net.load_state_dict(ck['crane'])
    return net.eval()


def labelset(data):
    from yard_rl.v7.crane.teacher import CraneLabelSet, CraneSample
    cs = data['result']['crane']
    return CraneLabelSet(samples=[CraneSample(**s) for s in data['samples']],
        worlds=cs['worlds'], zero=cs['zero'], scale=cs['scale'],
        median_gap_krw=cs['median_gap_krw'], floor_bound=cs['floor_bound'])


def train_seed(args):
    import torch
    from yard_rl.v7.crane.policy import CraneNet
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1)
    root = Path(args.out)
    folder = root/f'model-{args.model_seed}'
    folder.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    idx = MODEL_SEEDS.index(args.model_seed)
    if stage == 4:
        torch.manual_seed(args.model_seed)
        point = CraneNet(feature_mode='feasible')
        checkpoint(folder/'initial.pt', point, model_seed=args.model_seed)
        nets = {'pointwise': point, 'pairwise': copy.deepcopy(point)}
    elif stage == 5:
        nets = {'coverage': load_net(folder/'initial.pt')}
    else:
        from yard_rl.v7.crane.policy import add_store_input
        nets = {'flow': add_store_input(load_net(folder/'initial.pt'))}
    trainers = {k: CraneTrainer(v, loss_kind='pointwise' if k == 'pointwise' else 'pairwise')
                for k,v in nets.items()}
    report = dict(stage=stage, model_seed=args.model_seed, rows=[],
                  collection='pointwise behaviour fixed per iteration for all comparisons')
    for it, load in enumerate(LOADS):
        seed = 9913600 + 100*idx + 10*it
        t0 = time.perf_counter()
        if stage == 4:
            behavior = nets['pointwise']
            checkpoint(folder/f'behavior-{it}.pt', behavior, model_seed=args.model_seed, it=it)
        elif stage == 5:
            behavior = load_net(folder/f'behavior-{it}.pt')
        if stage in (4,5):
            kwargs = dict(load=load, seed=seed, dispatcher='RL_CRANE', crane_net=behavior,
                          crane_budget=RolloutBudget(max_labels=24,identity_checks=0),
                          candidate_cap=None, horizon_s=10800., workers=args.workers,
                          reuse_rehandle_slots=args.shared_slots,
                          decision_probe_shortcircuit=args.existence_scan)
            if stage == 5:
                kwargs['crane_sampling'] = 'coverage'
            ep = run_episode(**kwargs)
            data = dict(stage=stage, load=load, seed=seed, model_seed=args.model_seed,
                behavior_sha256=sha(folder/f'behavior-{it}.pt'),
                seconds=time.perf_counter()-t0, memory=memory(), result=ep.as_dict(),
                samples=[asdict(s) for s in ep.crane_labels])
            corpus = folder/f'labels-{stage}-{it}.json.gz'
            gz_save(corpus,data)
            cs = ep.crane_stats
            assert ep.policy_exceptions == 0 and cs['hook_errors'] == 0
            assert cs['factual_mismatch'] == 0, cs
            assert len(ep.crane_labels) >= 4, 'fewer than two valid comparison decisions'
            if stage == 5:
                standard = gz_read(folder/f'labels-4-{it}.json.gz')['result']
                assert cs['main_choices_sha256'] == standard['crane']['main_choices_sha256']
                for key in ('phi_krw','admitted','c_wait','c_vessel','c_move','c_rehandle','n_censored'):
                    assert data['result'][key] == standard[key], (key,'behaviour trajectory changed')
        else:
            corpus = folder/f'labels-5-{it}.json.gz'
            data = gz_read(corpus)
        labels = labelset(data)
        if stage == 6:
            for sample in labels.samples:
                sample.row.append(float(sample.is_store))
        fits = {}
        for name, trainer in trainers.items():
            fit_start = time.perf_counter()
            fit = trainer.fit(labels,seed=seed)
            assert fit.steps > 0
            fits[name] = dict(asdict(fit), seconds=time.perf_counter()-fit_start)
            checkpoint(folder/f'{name}-{it}.pt',nets[name],trainer,
                       model_seed=args.model_seed,it=it,corpus_sha256=sha(corpus))
        report['rows'].append(dict(it=it,load=load,seed=seed,corpus=corpus.as_posix(),
            corpus_sha256=sha(corpus),collection_seconds=data['seconds'] if stage in (4,5) else 0.,
            total_seconds=time.perf_counter()-t0, fits=fits, stats=data['result']['crane']))
        save(folder/f'stage-{stage}.json',report)
        print(f'stage {stage} model {args.model_seed} update {it+1}/3: '
              f'{len(labels.samples)//2} valid pairs; {time.perf_counter()-t0:.1f}s',flush=True)


def evaluate(args):
    import torch
    from yard_rl.v7.evaluation_probe import EvaluationProbe
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    root = Path(args.out)
    rule = args.arm.startswith('rule')
    model_path = None if rule else root/f'model-{args.model_seed}'/(
        'initial.pt' if args.arm == 'initial' else f'{args.arm}-2.pt')
    net = None if rule else load_net(model_path)
    probe = EvaluationProbe()
    seed = 9913900 + 10*LOADS.index(args.load)
    start = time.perf_counter()
    r = run_episode(load=args.load,seed=seed,dispatcher='SF_SPT' if rule else 'RL_CRANE',
        crane_net=net,candidate_cap=12 if args.arm == 'rule_cap12' else None,
        evaluation_probe=probe,reuse_rehandle_slots=args.shared_slots,
        decision_probe_shortcircuit=args.existence_scan)
    assert r.policy_exceptions == 0 and r.rollout_worlds == 0
    d = dict(arm=args.arm, model_seed=None if rule else args.model_seed,load=args.load,seed=seed,
        wall_seconds=time.perf_counter()-start,result=r.as_dict(),probe=probe.result,
        checkpoint=None if rule else str(model_path),checkpoint_sha256=None if rule else sha(model_path))
    path = root/'evaluation'/f'{args.arm}-{0 if rule else args.model_seed}-{args.load}.json.gz'
    gz_save(path,d)
    print(f'eval {args.arm}/{args.model_seed}/{args.load}: '
          f'cost={r.phi_krw:.2f}; turn={r.breakdown["mean_turn_time_s"]:.2f}s',flush=True)


def campaign(args):
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    git=lambda *a: subprocess.check_output(['git',*a],encoding='utf-8').strip()
    assert not git('status','--short','--untracked-files=no'),'commit code before campaign'
    stamp=dict(stage=args.stage,commit=git('rev-parse','HEAD'),workers=args.workers,
               loads=LOADS,model_seeds=MODEL_SEEDS,shared_slots=args.shared_slots,
               existence_scan=args.existence_scan,completed=[])
    stamp['sources']={p.as_posix():sha(p) for p in [Path(__file__),
        Path('docs/research/v7-discharge/six-step-prereg.md'),*Path('src/yard_rl/v7').rglob('*.py')]}
    marker=root/f'campaign-stage-{args.stage}.json'
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        assert old['sources']==stamp['sources'],'code changed during campaign'
        stamp=old
    common=[sys.executable,__file__,'--out',str(root),'--stage',str(args.stage),
            '--workers',str(args.workers)]+(['--shared-slots'] if args.shared_slots else [])
    common += ['--existence-scan'] if args.existence_scan else []
    jobs=[('train',seed,None,None) for seed in MODEL_SEEDS]
    names={4:['initial','pointwise','pairwise'],5:['coverage'],6:['flow']}[args.stage]
    eval_jobs=[('evaluate',seed,arm,load) for seed in MODEL_SEEDS for arm in names for load in LOADS]
    if args.stage==4:
        eval_jobs += [('evaluate',MODEL_SEEDS[0],arm,load) for arm in ('rule_all','rule_cap12') for load in LOADS]
    random.Random(9913800+args.stage).shuffle(eval_jobs)
    jobs+=eval_jobs
    for operation,seed,arm,load in jobs:
        key=f'{operation}/{seed}/{arm}/{load}'
        if key in stamp['completed']:
            continue
        cmd=common+['--operation',operation,'--model-seed',str(seed)]
        if operation=='evaluate':
            cmd+=['--arm',arm,'--load',str(load)]
        subprocess.run(cmd,check=True)
        stamp['completed'].append(key)
        save(marker,stamp)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=('campaign','train','evaluate'),default='campaign')
    p.add_argument('--stage',type=int,choices=(4,5,6),required=True)
    p.add_argument('--model-seed',type=int,choices=MODEL_SEEDS,default=MODEL_SEEDS[0])
    p.add_argument('--load',type=int,choices=LOADS,default=LOADS[0])
    p.add_argument('--arm',default='initial')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shared-slots',action='store_true')
    p.add_argument('--existence-scan',action='store_true')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    {'campaign':campaign,'train':train_seed,'evaluate':evaluate}[args.operation](args)
