"""Frozen-label, single-feature count-transform study (YR-332-s)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from benchmark_default_eval import memory
from evaluate_arrival_order import git, save, HASHES
from evaluate_classical_rules import StudyProbe
from six_step_learning import gz_read, gz_save, sha, labelset
from yard_rl.v7.crane.features import crane_features, encode_candidate_count

OUT = Path('outputs/reports/yr332_count_transform')
PRIOR = Path('outputs/reports/yr332_six_steps/learning')
REFERENCE = Path('outputs/reports/yr332_classical_eval')
PREREG = Path('docs/research/v7-discharge/count-transform-prereg.md')
SEEDS = (9913501, 9913502, 9913503)
LOADS = (3500, 7500, 15000)
TRANSFORMS = ('clipped', 'linear', 'log1p')
RECOVERIES = [(s, i) for s in SEEDS for i in range(3)]
CASES = [(a, s, phase, n, 0) for a in TRANSFORMS for s in SEEDS
         for phase in ('initial', 'final') for n in LOADS] + [
         (a, SEEDS[0], 'final', 3500, 1) for a in ('linear', 'log1p')] + [
         ('spt', 0, 'rule', n, 0) for n in LOADS]
WORKERS = 6


def source_hashes():
    helpers = ('benchmark_default_eval.py', 'six_step_learning.py',
               'evaluate_arrival_order.py', 'evaluate_classical_rules.py',
               'report_classical_rules.py', 'report_count_transform.py')
    paths = [Path(__file__), PREREG, *Path('src/yard_rl/v7').rglob('*.py'),
             *(Path('scripts/v7')/p for p in helpers)]
    return {p.resolve().relative_to(Path.cwd()).as_posix(): sha(p) for p in paths}


def load_net(path):
    import torch
    from yard_rl.v7.crane.policy import CraneNet
    return CraneNet.from_checkpoint(torch.load(path, weights_only=True))


def weights_hash(net):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                   for v in net.state_dict().values())).hexdigest()


def case_name(c):
    a, s, phase, n, repeat = c
    return f'{a}-{s}-{phase}-{n}' + ('-repeat' if repeat else '')


def case_path(c):
    return OUT/'evaluation'/(case_name(c)+'.json.gz')


def recovery_path(seed, iteration):
    return OUT/'recovered'/f'{seed}-{iteration}.json.gz'


def model_path(arm, seed, phase):
    return OUT/'models'/f'{arm}-{seed}-{phase}.pt'


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    from yard_rl.v7.crane.fit import CraneTrainer, GRAD_CLIP
    from yard_rl.v7.crane.policy import add_store_input
    from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
    assert not git('status', '--short', '--untracked-files=no'), 'commit before campaign'
    old = json.loads((REFERENCE/'campaign.json').read_text(encoding='utf-8'))
    old_artifacts = {**old['result_sha256'], **old['model_sha256']}
    for p, h in old_artifacts.items():
        assert sha(p) == h, p
    changes = {p for p,h in old['sources'].items() if sha(p) != h}
    assert changes == {f'src/yard_rl/v7/crane/{p}.py' for p in ('features','policy','collect','train')} | {
        'scripts/v7/six_step_learning.py'}, changes
    inputs = [REFERENCE/'campaign.json']
    for s in SEEDS:
        root = PRIOR/f'model-{s}'
        inputs.extend([root/'initial.pt', root/'stage-6.json'])
        for i in range(3):
            inputs.extend([root/f'behavior-{i}.pt', root/f'labels-5-{i}.json.gz', root/f'flow-{i}.pt'])
    template=CraneTrainer(add_store_input(load_net(PRIOR/f'model-{SEEDS[0]}'/'initial.pt')),
                          loss_kind='pairwise')
    config = dict(loads=LOADS,model_seeds=SEEDS,evaluation_seeds=[9913900,9913910,9913920],
        transforms=TRANSFORMS,reference_count=18,candidate_cap=None,include_flow=True,
        feature_mode='feasible',loss_kind='pairwise',iterations=3,teacher_horizon_s=10800,
        teacher_recomputed=False,workers=WORKERS,torch_threads=1,eval_cases=CASES,
        recovery_cases=RECOVERIES,observation_s=86400,engine_end_s=93600,
        default_transform_unchanged=True,exec_config=asdict(DEFAULT_EXEC_CONFIG),
        model_config=template.net.model_config(),optimizer=template.opt.defaults,
        minibatch=template.minibatch,steps_coef=template.steps_coef,steps_per_iteration=50,
        validation_fraction=.2,huber_beta=template.loss_fn.beta,gradient_clip=GRAD_CLIP)
    d = dict(task='YR-332-s',sources=source_hashes(),configuration=config,
        source_changes=sorted(changes),input_sha256={p.as_posix():sha(p) for p in inputs},
        prior_artifact_sha256=old_artifacts,
        repro=repro_stamp(experiment='YR-332-s',seeds=dict(model=list(SEEDS),
            evaluation=[9913900,9913910,9913920],training=[9913600+100*j+10*i
                for j in range(3) for i in range(3)]),params=config,prereg=PREREG.as_posix()))
    save(OUT/'campaign.json', d)
    return d


class CountProbe(StudyProbe):
    def __init__(self, samples=()):
        super().__init__()
        self.targets = {}
        for i, s in enumerate(samples):
            self.targets.setdefault(tuple(s['decision_key']), []).append((i,s))
        self.recovered = {}
        self.action_digest = hashlib.sha256()
        self.counts = Counter()

    def decision(self, sim, dp, generated, assignment, gen_ns, select_ns):
        super().decision(sim,dp,generated,assignment,gen_ns,select_ns)
        block = self.block_names[id(sim)]
        self.action_digest.update(repr((sim.clock, block, [
            (cid,g.kind.value,g.job_ref.job_id if g.job_ref else None)
            for cid,g in sorted(assignment.items())])).encode())
        for cid in dp.crane_ids:
            candidates = generated[cid].items
            for gc in candidates:
                if gc.job_ref is not None:
                    self.counts[gc.feasible_serve_count] += 1
            key = (round(sim.clock,6),block,cid)
            for i, sample in self.targets.get(key, []):
                assert i not in self.recovered, ('duplicate', key)
                matches = [g for g in candidates if g.job_ref and g.job_ref.job_id == sample['job']
                    and crane_features(sim,cid,g,feature_mode='feasible') == sample['row']]
                assert len(matches) == 1, (key,sample['job'],len(matches))
                gc = matches[0]
                n = gc.feasible_serve_count
                assert isinstance(n,int) and n >= 0
                assert encode_candidate_count(n) == sample['row'][7]
                if sample['action'] == 'PICKED':
                    assert assignment[cid].job_ref.job_id == sample['job']
                flow = crane_features(sim,cid,gc,feature_mode='feasible',include_flow=True)
                assert flow[-1] == float(sample['is_store'])
                self.recovered[i] = n

    def finish(self, mbt, records, episode):
        super().finish(mbt,records,episode)
        operational = self.result['operational']
        operational['gate_exit_censored_24h'] = operational.pop('trucks_unfinished')
        self.result['main_choices_sha256'] = self.action_digest.hexdigest()
        self.result['count_histogram'] = {str(k):v for k,v in sorted(self.counts.items())}


def recover(index):
    import torch
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    seed, iteration = RECOVERIES[index]
    root = PRIOR/f'model-{seed}'
    source = root/f'labels-5-{iteration}.json.gz'
    data = gz_read(source)
    net = load_net(root/f'behavior-{iteration}.pt')
    assert sha(root/f'behavior-{iteration}.pt') == data['behavior_sha256']
    probe = CountProbe(data['samples'])
    start = time.perf_counter()
    result = run_episode(load=data['load'],seed=data['seed'],dispatcher='RL_CRANE',
        crane_net=net,candidate_cap=None,evaluation_probe=probe,
        reuse_rehandle_slots=False,decision_probe_shortcircuit=True).as_dict()
    assert len(probe.recovered) == len(data['samples'])
    assert probe.result['main_choices_sha256'] == data['result']['crane']['main_choices_sha256']
    ignored = {'crane','n_crane_labels'}
    assert {k:v for k,v in result.items() if k not in ignored} == {
        k:v for k,v in data['result'].items() if k not in ignored}, 'trajectory/accounting changed'
    raw = [probe.recovered[i] for i in range(len(data['samples']))]
    assert raw[::2] == raw[1::2]
    gz_save(recovery_path(seed,iteration),dict(model_seed=seed,iteration=iteration,
        source=source.as_posix(),source_sha256=sha(source),raw_counts=raw,
        samples=len(raw),wall_seconds=time.perf_counter()-start,memory=memory(),
        exact_original_features=True,exact_original_choices=True,exact_original_result=True,
        result=result,probe=probe.result))
    print(f'recovered {seed}/{iteration}: {len(raw)} rows',flush=True)


def train(seed):
    import torch
    from yard_rl.v7.crane.policy import add_store_input
    from yard_rl.v7.crane.fit import CraneTrainer
    torch.set_num_threads(1)
    initial = add_store_input(load_net(PRIOR/f'model-{seed}'/'initial.pt'))
    original = json.loads((PRIOR/f'model-{seed}'/'stage-6.json').read_text(encoding='utf-8'))
    rows = []
    for arm in TRANSFORMS:
        net = copy.deepcopy(initial)
        net.count_transform = arm
        trainer = CraneTrainer(net,loss_kind='pairwise')
        def checkpoint(phase):
            p = model_path(arm,seed,phase)
            p.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(crane=net.state_dict(),model_config=net.model_config(),
                opt=trainer.opt.state_dict(),loss_kind='pairwise',model_seed=seed),p)
        checkpoint('initial')
        assert weights_hash(net) == weights_hash(initial)
        for i in range(3):
            recovered = gz_read(recovery_path(seed,i))
            assert sha(recovered['source']) == recovered['source_sha256']
            data = gz_read(recovered['source'])
            labels = labelset(data)
            for sample,n in zip(labels.samples,recovered['raw_counts'],strict=True):
                sample.row[7] = encode_candidate_count(n,arm)
                sample.row.append(float(sample.is_store))
            start = time.perf_counter()
            fit = trainer.fit(labels,seed=data['seed'])
            assert fit.steps == 50
            old_fit = original['rows'][i]['fits']['flow']
            assert all(getattr(fit,k) == old_fit[k] for k in ('n','n_val','steps','scale_krw'))
            if arm == 'clipped':
                old_net = load_net(PRIOR/f'model-{seed}'/f'flow-{i}.pt')
                assert weights_hash(net) == weights_hash(old_net), ('baseline training changed',seed,i)
            checkpoint(f'iteration-{i}')
            rows.append(dict(arm=arm,iteration=i,seed=data['seed'],fit=asdict(fit),
                weights_sha256=weights_hash(net),source_sha256=recovered['source_sha256'],
                recovery_sha256=sha(recovery_path(seed,i)),seconds=time.perf_counter()-start))
        checkpoint('final')
    save(OUT/'training'/f'{seed}.json',dict(rows=rows,initial_weights_sha256=weights_hash(initial),
        baseline_all_iterations_exact=True,model_seed=seed))
    print(f'trained {seed}: 9 iterations; baseline exactly reproduced',flush=True)


def evaluate(index, *, smoke=False):
    import torch
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    case = CASES[index]
    arm,seed,phase,load,repeat = case
    rule = arm == 'spt'
    path = None if rule else model_path(arm,seed,phase)
    if smoke and not rule:
        from yard_rl.v7.crane.policy import add_store_input
        net = add_store_input(load_net(PRIOR/f'model-{seed}'/'initial.pt'))
        net.count_transform = arm
    else:
        net = None if rule else load_net(path)
    before = None if rule else weights_hash(net)
    probe = CountProbe()
    start = time.perf_counter()
    input_seed = 9914009 if smoke else 9913900 + 10*LOADS.index(load)
    r = run_episode(load=80 if smoke else load,seed=input_seed,
        dispatcher='SPT' if rule else 'RL_CRANE',crane_net=net,candidate_cap=None,
        evaluation_probe=probe,reuse_rehandle_slots=False,decision_probe_shortcircuit=True)
    assert r.policy_exceptions == r.rollout_worlds == r.traded_edges == r.n_space == r.n_time == 0
    assert rule or weights_hash(net) == before
    d = dict(arm=arm,model_seed=seed,phase=phase,load=80 if smoke else load,repeat=repeat,seed=input_seed,
        result=r.as_dict(),probe=probe.result,wall_seconds=time.perf_counter()-start,memory=memory(),
        checkpoint=None if rule or smoke else path.as_posix(),
        checkpoint_sha256=None if rule or smoke else sha(path),
        weights_sha256=before,python=sys.version,torch=torch.__version__,threads=1)
    dest = OUT/'smoke'/f'{arm}.json.gz' if smoke else case_path(case)
    gz_save(dest,d)
    print(f'{case_name(case)}: turn={r.breakdown["mean_turn_time_s"]/60:.4f} min',flush=True)


def run_jobs(operation, indices):
    def one(i):
        log = OUT/'logs'/f'{operation}-{i}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as f:
            done = subprocess.run([sys.executable,__file__,'--operation',operation,'--index',str(i)],
                stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if done.returncode:
            raise RuntimeError(f'{operation}/{i} failed: {log.read_text(encoding="utf-8")[-4000:]}')
        return i
    indices = list(indices)
    random.Random(9914000).shuffle(indices)
    completed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for future in as_completed([pool.submit(one,i) for i in indices]):
            completed.append(future.result())
            save(OUT/f'{operation}-progress.json',dict(completed=completed,total=len(indices)))
            print(f'{operation}: {len(completed)}/{len(indices)} complete',flush=True)


def campaign():
    prepare()
    run_jobs('recover',range(len(RECOVERIES)))
    for seed in SEEDS:
        train(seed)
    run_jobs('evaluate',range(len(CASES)))
    from report_count_transform import report
    report()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=('campaign','recover','train','evaluate','smoke','report'),default='campaign')
    p.add_argument('--index',type=int,default=0)
    args = p.parse_args()
    if args.operation == 'campaign':
        campaign()
    elif args.operation == 'recover':
        recover(args.index)
    elif args.operation == 'train':
        train(SEEDS[args.index])
    elif args.operation == 'evaluate':
        evaluate(args.index)
    elif args.operation == 'smoke':
        for arm in TRANSFORMS:
            evaluate(next(i for i,c in enumerate(CASES) if c[0] == arm),smoke=True)
    else:
        from report_count_transform import report
        report()
