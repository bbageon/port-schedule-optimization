"""YR-332-h diagnostic only: scalar vs batched shared-net decision latency.

No production policy replacement. All candidates are preserved in the stress
probe; the live engine's cap/priority/reward contracts are unchanged.
Run: PYTHONPATH=src python scripts/v7/benchmark_crane_latency.py --out PATH
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from yard_rl.v7.crane import features as F
from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.domain.enums import ContainerSize, JobFlow, LoadStatus
from yard_rl.v7.world.domain.models import Job
from yard_rl.v7.world.integrated.baselines import ResolverPolicy
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile
from yard_rl.v7.world.integrated.scenario import TerminalScenario

SEED = 9_903_320


def stamp():
    def git(*args):
        return subprocess.check_output(['git', *args], encoding='utf-8').strip()
    paths = [Path(__file__), Path('src/yard_rl/v7/crane/features.py'),
             Path('src/yard_rl/v7/crane/policy.py'),
             Path('src/yard_rl/v7/world/integrated/engine.py'),
             Path('docs/research/v7-discharge/prereg.md')]
    return dict(commit=git('rev-parse', 'HEAD'),
        tracked_dirty=bool(git('status', '--short', '--untracked-files=no')),
        python=platform.python_version(), torch=torch.__version__, cpu=platform.processor(),
        threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available(), seed=SEED,
        sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def measure(fn, repetitions, warmup=3):
    start = time.perf_counter_ns()
    fn()
    first = (time.perf_counter_ns() - start) / 1e6
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return dict(first_call_in_phase_ms=first, samples_ms=times,
        median_ms=statistics.median(times), p95_ms=sorted(times)[math.ceil(.95*len(times))-1],
        repetitions=repetitions, warmup=warmup)


def identity(assign):
    return {c: (g.kind.value, g.job_ref.job_id if g.job_ref else None)
            for c, g in assign.items()}


class BatchPrototype(CranePolicy):
    """Same rank keys; cache state-only feature work once per decision."""
    def prepare(self, sim, gb):
        pairs = [(cid, g) for cid, gen in gb.items() for g in gen.items
                 if g.feasible and g.job_ref is not None]
        old = F._live_cands
        waiting = old(sim, '')
        try:
            F._live_cands = lambda _sim, _cid: waiting
            rows = [F.crane_features(sim, cid, g) for cid, g in pairs]
        finally:
            F._live_cands = old
        with torch.no_grad():
            values = self.net(torch.tensor(rows, dtype=torch.float32)).tolist() if rows else []
        self.keys = {(cid, g.candidate_id):
                     (0 if g.kind == CandidateKind.SERVE else 1, val, g.job_ref.job_id)
                     for (cid, g), val in zip(pairs, values)}
        self.rows, self.values = rows, values

    def rank(self, sim, crane_id, gc):
        if gc.job_ref is None:
            return (2, 0.0, '')
        return self.keys[(crane_id, gc.candidate_id)]


def snapshot(n):
    p = build_integrated_profile()
    p = replace(p, cranes=(p.cranes[0],))
    jobs = [Job(job_id=f'T{i:06}', flow=JobFlow.GATE_IN, release_time=0.,
        actual_gate_in=0., actual_block_arrival=0.,
        inbound_size=ContainerSize.FT40 if i % 2 else ContainerSize.FT20,
        inbound_load=LoadStatus.FULL) for i in range(n)]
    s = TerminalScenario('latency-snapshot', SEED, 3600., 0., {}, jobs, [])
    sim = TerminalSimulator(p, s, yard_handover_cap=p.transfer.n_units)
    return sim, sim.run_until_decision()


def run(out, repeats=30, pipeline_repeats=10):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    net = CraneNet().eval()
    result = dict(schema='yard.v7.discharge-latency.v1', environment=stamp(),
        scope='CPU 9-input current network; batch path is a diagnostic prototype',
        excludes=['future set encoder', 'network/TOS IO', 'live dispatch deadlines',
                  '42-crane simultaneous dispatch'], network=[], pipeline=[])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    for n in (12, 100, 1000, 10000):
        x = torch.randn(n, F.CRANE_DIM)
        def scalar():
            with torch.no_grad():
                return torch.cat([net(x[i:i+1]) for i in range(n)])
        def batch():
            with torch.no_grad():
                return net(x)
        def chunks():
            with torch.no_grad():
                return torch.cat([net(part) for part in x.split(1024)])
        a, b, c = scalar(), batch(), chunks()
        error = float(torch.max(torch.abs(a-b)))
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
        assert torch.allclose(a, c, atol=1e-6, rtol=1e-5)
        row = dict(candidates=n, scalar=measure(scalar, repeats), batch=measure(batch, repeats),
                   chunks1024=measure(chunks, repeats), max_abs_error=error,
                   same_best=bool(torch.argmin(a) == torch.argmin(b)))
        result['network'].append(row)
        save()
        print(f"network n={n}: scalar={row['scalar']['median_ms']:.3f}ms "
              f"batch={row['batch']['median_ms']:.3f}ms", flush=True)

    for n in (12, 100, 1000):
        sim, dp = snapshot(n)
        gen = CandidateGenerator(k_max=n+1, block_pre_rehandle=True, config=DEFAULT_EXEC_CONFIG)
        scalar_pref, batch_pref = CranePolicy(net), BatchPrototype(net)
        scalar_pol, batch_pol = ResolverPolicy(scalar_pref), ResolverPolicy(batch_pref)
        def generate():
            return {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        def scalar_pipeline():
            gb = generate()
            return scalar_pol.decide(sim, dp, gb)
        def batch_pipeline():
            gb = generate()
            batch_pref.prepare(sim, gb)
            return batch_pol.decide(sim, dp, gb)
        gb = generate()
        pairs = [(cid, g) for cid, gg in gb.items() for g in gg.items
                 if g.feasible and g.job_ref is not None]
        raw_rows = [F.crane_features(sim, cid, g) for cid, g in pairs]
        raw_scores = [scalar_pref.rank(sim, cid, g)[1] for cid, g in pairs]
        batch_pref.prepare(sim, gb)
        assert raw_rows == batch_pref.rows
        assert torch.allclose(torch.tensor(raw_scores), torch.tensor(batch_pref.values),
                              atol=1e-6, rtol=1e-5)
        a, b = identity(scalar_pipeline()), identity(batch_pipeline())
        assert a == b, (a, b)
        def feature_only():
            return [F.crane_features(sim, cid, g) for cid, g in pairs]
        row = dict(orders=n, cranes=len(dp.crane_ids), feasible_pairs=len(pairs),
            generation=measure(generate, pipeline_repeats),
            scalar_features=measure(feature_only, pipeline_repeats),
            scalar_total=measure(scalar_pipeline, pipeline_repeats),
            batch_cached_total=measure(batch_pipeline, pipeline_repeats),
            same_features=True, same_assignment=a == b, assignment=a)
        result['pipeline'].append(row)
        save()
        print(f"pipeline orders={n}: scalar={row['scalar_total']['median_ms']:.3f}ms "
              f"batch+cache={row['batch_cached_total']['median_ms']:.3f}ms", flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.out)
