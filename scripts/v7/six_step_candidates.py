"""Sequential, fresh-process comparisons for the six-step candidate study."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from benchmark_default_eval import memory, save

LOADS = (3500, 7500, 15000)
FIELDS = ('candidates_sha256', 'choices_sha256', 'events_sha256',
          'jobs_sha256', 'records_sha256', 'result_sha256')


def worker(args):
    import torch
    from yard_rl.v7.crane.policy import CraneNet
    from yard_rl.v7.evaluation_probe import EvaluationProbe
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    torch.manual_seed(args.seed + 2)
    net = CraneNet().eval()
    with torch.no_grad():
        for _ in range(3):
            net(torch.zeros(32, 9))
    model_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes()
                                        for v in net.state_dict().values())).hexdigest()
    probe = EvaluationProbe()
    before = memory()
    start = time.perf_counter()
    result = run_episode(load=args.load, seed=args.seed, dispatcher='RL_CRANE',
                         crane_net=net, evaluation_probe=probe,
                         reuse_rehandle_slots=args.arm == 'shared')
    wall = time.perf_counter() - start
    assert result.policy_exceptions == 0 and result.rollout_worlds == 0
    assert not result.crane_labels and result.traded_edges == 0
    data = dict(load=args.load, seed=args.seed, arm=args.arm, wall_seconds=wall,
                model_sha256=model_hash, result=result.as_dict(), probe=probe.result,
                memory_before=before, memory_after=memory(), threads=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, 'wt', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f'{args.load}/{args.seed}/{args.arm}: {wall:.2f}s', flush=True)


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def campaign(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    assert not git('status', '--short', '--untracked-files=no'), 'commit before comparison'
    paths = [Path(__file__), Path('docs/research/v7-discharge/six-step-prereg.md'),
             *Path('src/yard_rl/v7').rglob('*.py')]
    sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    stamp_path = out/'campaign.json'
    stamp = dict(schema='yard.v7.six-step.candidates.v1', commit=git('rev-parse','HEAD'),
                 stage=1, sources=sources, rows=[])
    if stamp_path.exists():
        stamp = json.loads(stamp_path.read_text(encoding='utf-8'))
        assert stamp['sources'] == sources, 'source changed during campaign'
    cases = [(load, seed, 0) for load in LOADS for seed in (9913320, 9913330, 9913340)]
    cases += [(load, 9913320, 1) for load in LOADS]
    for load, seed, repeat in cases:
        if any((r['load'], r['seed'], r['repeat']) == (load, seed, repeat) for r in stamp['rows']):
            continue
        arms = ['legacy', 'shared']
        random.Random(load + seed + repeat).shuffle(arms)
        row = dict(load=load, seed=seed, repeat=repeat, order=arms, arms={})
        raw = {}
        for arm in arms:
            path = out/f'{load}-{seed}-{repeat}-{arm}.json.gz'
            subprocess.run([sys.executable, __file__, '--worker', '--load', str(load),
                            '--seed', str(seed), '--arm', arm, '--out', str(path)], check=True)
            d = raw[arm] = read(path)
            row['arms'][arm] = dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                wall_seconds=d['wall_seconds'], run_seconds=d['probe']['run_seconds'],
                generation=d['probe']['generation'], selection=d['probe']['selection'],
                decision=d['probe']['decision'], memory=d['memory_after'], cache=d['probe']['cache'])
        a, b = raw['legacy'], raw['shared']
        row['equal'] = {f: a['probe'][f] == b['probe'][f] for f in FIELDS}
        row['equal']['input'] = a['probe']['metadata']['input_sha256'] == b['probe']['metadata']['input_sha256']
        row['equal']['model'] = a['model_sha256'] == b['model_sha256']
        stamp['rows'].append(row)
        save(stamp_path, stamp)
        assert all(row['equal'].values()), row['equal']
        print(f'pair {len(stamp["rows"])}/{len(cases)}: exact outputs match', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--load', type=int, default=300)
    p.add_argument('--seed', type=int, default=9913320)
    p.add_argument('--arm', choices=('legacy', 'shared'), default='shared')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    worker(args) if args.worker else campaign(args)
