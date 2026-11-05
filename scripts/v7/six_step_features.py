"""Audit corrected feature meaning on three fixed daily input trajectories."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark_default_eval import save


def worker(args):
    import torch
    from yard_rl.v7.crane.policy import CraneNet
    from yard_rl.v7.feature_probe import FeatureProbe
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    torch.manual_seed(9913422)
    net = CraneNet(feature_mode='legacy')
    probe = FeatureProbe()
    start = time.perf_counter()
    r = run_episode(load=args.load, seed=9913420, dispatcher='RL_CRANE', crane_net=net,
                    candidate_cap=None, evaluation_probe=probe,
                    reuse_rehandle_slots=args.shared_slots, decision_probe_shortcircuit=args.existence_scan)
    assert r.policy_exceptions == 0
    with gzip.open(args.out, 'wt', encoding='utf-8', newline='\n') as f:
        json.dump(dict(load=args.load, seed=9913420, wall_seconds=time.perf_counter()-start,
                       result=r.as_dict(), probe=probe.result), f)
    print(f'feature audit {args.load}: {probe.result["feature_audit"]}', flush=True)


def campaign(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    assert not git('status', '--short', '--untracked-files=no')
    stamp = dict(commit=git('rev-parse', 'HEAD'), stage=3, seed=9913420, rows=[],
                 parameters=dict(candidate_cap=None, policy='legacy; feasible shadow only',
                                 shared_slots=args.shared_slots, existence_scan=args.existence_scan,
                                 loads=[3500,7500,15000]))
    stamp['sources'] = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__),*Path('src/yard_rl/v7').rglob('*.py')]}
    for load in (3500,7500,15000):
        path = out/f'{load}.json.gz'
        subprocess.run([sys.executable, __file__, '--worker', '--load',str(load),'--out',str(path)]
                       + (['--shared-slots'] if args.shared_slots else [])
                       + (['--existence-scan'] if args.existence_scan else []),check=True)
        with gzip.open(path,'rt',encoding='utf-8') as f:
            d=json.load(f)
        reference_equal = None
        if args.reference_campaign:
            ref_path=Path(args.reference_campaign)/f'{load}-9913420-0-all.json.gz'
            with gzip.open(ref_path,'rt',encoding='utf-8') as f:
                reference=json.load(f)
            reference_equal={k:d['probe'][k]==v for k,v in reference['probe'].items()
                             if k.endswith('_sha256')}
            assert all(reference_equal.values()),reference_equal
        stamp['rows'].append(dict(load=load,path=path.as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), audit=d['probe']['feature_audit'],
            stage2_reference_equal=reference_equal))
        save(out/'campaign.json',stamp)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--load',type=int,default=3500)
    p.add_argument('--shared-slots',action='store_true')
    p.add_argument('--existence-scan',action='store_true')
    p.add_argument('--reference-campaign')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    worker(args) if args.worker else campaign(args)
