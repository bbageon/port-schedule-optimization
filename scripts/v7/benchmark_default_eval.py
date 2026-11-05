"""Fresh-process paired daily evaluation, with fixed models and no learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

LOADS = (3500, 7500, 15000)
SEEDS = (9913320, 9913330, 9913340)
MODES = ('reference', 'optimized')


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',
                          encoding='utf-8', newline='\n')


def memory():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return dict(rss_bytes=counters.WorkingSetSize, peak_rss_bytes=counters.PeakWorkingSetSize,
                    private_bytes=counters.PrivateUsage, peak_commit_bytes=counters.PeakPagefileUsage)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(peak_rss_bytes=peak*(1 if sys.platform == 'darwin' else 1024))


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
    model_hash = lambda: hashlib.sha256(b''.join(v.detach().numpy().tobytes()
                                               for v in net.state_dict().values())).hexdigest()
    before_hash = model_hash()
    probe = EvaluationProbe()
    before_mem = memory()
    started = time.perf_counter()
    result = run_episode(load=args.load, seed=args.seed, dispatcher='RL_CRANE',
                         crane_net=net, evaluation_mode=args.mode, evaluation_probe=probe)
    wall = time.perf_counter()-started
    assert before_hash == model_hash(), 'model changed during evaluation'
    assert result.policy_exceptions == 0 and result.rollout_worlds == 0
    assert not result.crane_labels and result.traded_edges == 0
    save(args.out, dict(load=args.load, seed=args.seed, mode=args.mode,
        model_sha256=before_hash, model_type='untrained fixed current 9-input network',
        wall_seconds=wall, memory_before=before_mem, memory_after=memory(),
        python=platform.python_version(), torch=torch.__version__, threads=1,
        result=result.as_dict(), probe=probe.result))
    print(f'{args.load}/{args.seed}/{args.mode}: {wall:.2f}s; '
          f"decisions={probe.result['decision']['count']}", flush=True)


def campaign(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    stamp = dict(schema='yard.v7.daily-evaluation-latency.v1', commit=git('rev-parse', 'HEAD'),
        tracked_dirty=bool(git('status', '--short', '--untracked-files=no')),
        parameters=dict(loads=LOADS, seeds=SEEDS, modes=MODES, observe_hours=24, drain_hours=2,
            threads=1, teacher=False, training=False, candidate_cap='default 12; mandatory overflow retained',
            processes='one at a time; fresh child for each arm',
            timing='decision excludes probe/apply; episode includes setup/probe/final hashes; run excludes setup/final hashes',
            memory='whole-process OS peak; includes PyTorch, simulator and evidence buffers'), rows=[])
    assert not stamp['tracked_dirty'], 'commit code/prereg before campaign'
    paths = [Path('docs/research/v7-discharge/default-eval-prereg.md'), Path(__file__).resolve().relative_to(Path.cwd()),
             *Path('src/yard_rl/v7').rglob('*.py')]
    stamp['source_sha256'] = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(out/'campaign.json', stamp)
    for load in LOADS:
        for seed in SEEDS:
            modes = list(MODES)
            random.Random(seed+load).shuffle(modes)
            row = dict(load=load, seed=seed, order=modes, arms={})
            for mode in modes:
                path = out/f'{load}-{seed}-{mode}.json'
                subprocess.run([sys.executable, __file__, '--worker', '--load', str(load),
                                '--seed', str(seed), '--mode', mode, '--out', str(path)], check=True)
                raw = json.loads(path.read_text(encoding='utf-8'))
                row['arms'][mode] = dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    wall_seconds=raw['wall_seconds'], run_seconds=raw['probe']['run_seconds'],
                    decision=raw['probe']['decision'], memory=raw['memory_after'])
            a, b = [json.loads(Path(row['arms'][mode]['path']).read_text(encoding='utf-8')) for mode in MODES]
            fields = ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256')
            row['equal'] = {f:a['probe'][f] == b['probe'][f] for f in fields}
            row['equal']['input'] = a['probe']['metadata']['input_sha256'] == b['probe']['metadata']['input_sha256']
            row['equal']['model'] = a['model_sha256'] == b['model_sha256']
            stamp['rows'].append(row)
            save(out/'campaign.json', stamp)
            assert all(row['equal'].values()), row['equal']
            print(f'paired {load}/{seed}: exact outputs match', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--load', type=int, default=300)
    p.add_argument('--seed', type=int, default=9913300)
    p.add_argument('--mode', choices=('reference','batch','optimized'), default='optimized')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    worker(args) if args.worker else campaign(args)
