"""YR-332-j paired state transitions: scalar, batch, index, plan reuse."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import runpy
import statistics
import subprocess
import time

import torch

from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.world.domain.enums import ContainerSize, LoadStatus, JobStatus
from yard_rl.v7.world.domain.models import Container
from yard_rl.v7.world.integrated.baselines import ResolverPolicy
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.reservation import Reservation, Corridor
from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG

SEED = 9903321
ARMS = ('scalar_scan', 'batch_common', 'batch_index', 'batch_index_plans')
CASES = ('cold', 'unchanged', 'arrival', 'cancel', 'crane_move', 'yard_change', 'reservation')


def quantiles(samples):
    return dict(samples_ms=samples, median_ms=statistics.median(samples),
                p95_ms=sorted(samples)[math.ceil(.95*len(samples))-1])


def identity(a):
    return {cid: (g.kind.value, g.job_ref.job_id if g.job_ref else None) for cid, g in a.items()}


def mutate(sim, case, inbound):
    # Includes status index maintenance in the timed transition.
    if case not in ('cold', 'unchanged'):
        sim._advance(1.0)
    if case == 'arrival':
        j = inbound('NEW', at=sim.clock)
        j.status = JobStatus.WAITING
        sim.jobs[j.job_id] = j
        sim.kpis.truck_arrived(j.job_id, sim.clock)
    elif case == 'cancel':
        sim.jobs['T000000'].status = JobStatus.CANCELLED
        sim.kpis._waiting.pop('T000000', None)
    elif case == 'crane_move':
        sim.fleet.get('YC-A').state.position_bay += 1
        sim.reservations.set_idle_position('YC-A', sim.fleet.get('YC-A').state.position_bay)
    elif case == 'yard_change':
        sim.stacks.place(Container('NEW_BOX', ContainerSize.FT20, LoadStatus.FULL,
                                  'B1', 1, 1, 1), 1, 1)
    elif case == 'reservation':
        sim.reservations.reserve(Reservation('OTHER', 'OTHER', Corridor(38, 40),
                                             frozenset({(1, 1)}), None, 100))


def run(output):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    fixture = runpy.run_path('tests/v7/test_dispatch_reuse.py')
    net = CraneNet().eval()
    with torch.no_grad():
        for _ in range(3):
            net(torch.zeros(1024, 9))
    paths = [Path(__file__), Path('tests/v7/test_dispatch_reuse.py'),
             Path('docs/research/v7-discharge/dispatch-cache-prereg.md'),
             *[Path('src/yard_rl/v7')/p for p in (
                 'job_index.py', 'dispatch_cache.py', 'crane/features.py', 'crane/policy.py',
                 'world/domain/models.py', 'world/integrated/engine.py',
                 'world/integrated/candidates.py', 'world/integrated/resolver.py')]]
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    result = dict(schema='yard.v7.dispatch-reuse.v1', seed=SEED, repetitions=20,
        environment=dict(commit=git('rev-parse', 'HEAD'),
            tracked_dirty=bool(git('status', '--short', '--untracked-files=no')),
            python=platform.python_version(), torch=torch.__version__, cpu=platform.processor(),
            threads=1, cuda_used=False),
        parameters=dict(orders=[12, 100, 1000], history_case=10000, cranes=1,
            model=[9, 64, 64, 1], cases=CASES, arms=ARMS, cap='orders+2 (no candidate dropped)',
            warmup='3 network calls; non-cold cases prime one complete decision',
            timing='state transition plus generation/features/ranking/feasibility; clone and checks excluded'),
        sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, rows=[])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    conditions = [(n, 0, case) for n in (12, 100, 1000) for case in CASES]
    conditions.append((1000, 10000, 'unchanged'))
    for n, history, case in conditions:
        base, dp = fixture['snapshot'](n=n, history=history)
        samples = {a: [] for a in ARMS}
        transitions = {a: [] for a in ARMS}
        details = {}
        for repeat in range(20):
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                sim = copy.deepcopy(base)
                if arm in ('scalar_scan', 'batch_common'):
                    sim.jobs = dict(sim.jobs)
                gen = CandidateGenerator(k_max=n+2, block_pre_rehandle=True,
                    config=DEFAULT_EXEC_CONFIG, reuse_plans=arm == 'batch_index_plans')
                pol = ResolverPolicy(CranePolicy(net, batched=arm != 'scalar_scan'))
                def select():
                    gb = {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
                    return gb, pol.decide(sim, dp, gb)
                if case != 'cold':
                    select()
                before = sim.dispatch_cache.stats.copy()
                start = time.perf_counter_ns()
                mutate(sim, case, fixture['inbound'])
                middle = time.perf_counter_ns()
                gb, chosen = select()
                stop = time.perf_counter_ns()
                samples[arm].append((stop-start)/1e6)
                transitions[arm].append((middle-start)/1e6)
                if repeat == 0:
                    oracle = copy.deepcopy(sim)
                    oracle.jobs = dict(oracle.jobs)
                    rg = CandidateGenerator(k_max=n+2, block_pre_rehandle=True,
                                            config=DEFAULT_EXEC_CONFIG, reuse_plans=False)
                    reference = {c: rg.generate(oracle, c, oracle.info_level) for c in dp.crane_ids}
                    expected = ResolverPolicy(CranePolicy(net, batched=False)).decide(oracle, dp, reference)
                    details[arm] = dict(same_candidates=gb == reference,
                        same_assignment=identity(chosen) == identity(expected), assignment=identity(chosen),
                        candidate_count=sum(g.job_ref is not None for cs in gb.values() for g in cs.items),
                        feasible_count=sum(g.feasible and g.job_ref is not None for cs in gb.values() for g in cs.items),
                        cache_delta={k: sim.dispatch_cache.stats[k]-v for k,v in before.items()})
                    assert details[arm]['same_candidates'] and details[arm]['same_assignment'], details[arm]
        row = dict(orders=n, history=history, case=case, arms={arm: dict(
            **quantiles(samples[arm]), transition=quantiles(transitions[arm]), **details[arm]) for arm in ARMS})
        result['rows'].append(row)
        save()
        print(f"n={n} history={history} {case}: " + ' / '.join(
            f"{a}={row['arms'][a]['median_ms']:.3f}ms" for a in ARMS), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    run(parser.parse_args().out)
