"""Continuous mixed-work diagnostic: reuse with real state invalidations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import runpy
import statistics
import subprocess
import time

import torch

from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.world.domain.enums import ContainerSize, InformationLevel, JobStatus
from yard_rl.v7.world.integrated.baselines import ResolverPolicy, run_joint_episode
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.cost_config import RewardCalculator
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG

ARMS = ('scalar_scan', 'batch_common', 'batch_index', 'batch_index_plans')


def run(output):
    torch.set_num_threads(1)
    fixture = runpy.run_path('tests/v7/test_dispatch_reuse.py')
    scenario = build_minimal_terminal_scenario()
    for i in range(60):
        scenario.jobs.append(fixture['inbound'](f'EXTRA{i:04}', at=60+40*i,
            size=ContainerSize.FT20 if i % 2 else ContainerSize.FT40))
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    result = dict(schema='yard.v7.dispatch-stream.v1', commit=git('rev-parse', 'HEAD'),
        tracked_dirty=bool(git('status', '--short', '--untracked-files=no')),
        parameters=dict(seeds=[9903321, 9903322, 9903323], repeats=5, extra_trucks=60,
            interval_s=40, cranes=2, candidate_cap=12, threads=1, horizon_s=scenario.horizon_s,
            drain_s=scenario.drain_window_s, includes='event processing and dispatch',
            excludes='initialization, IO, real terminal integration'), rows=[])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in result['parameters']['seeds']:
        torch.manual_seed(seed)
        net = CraneNet().eval()
        with torch.no_grad():
            for _ in range(3):
                net(torch.zeros(32, 9))
        rng = random.Random(seed)
        samples = {a: [] for a in ARMS}
        outcomes = {}
        for repeat in range(5):
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                sim = TerminalSimulator(build_integrated_profile(), copy.deepcopy(scenario),
                    info_level=InformationLevel.PRE_ADVICE, yard_handover_cap=2)
                if arm in ('scalar_scan', 'batch_common'):
                    sim.jobs = dict(sim.jobs)
                gen = CandidateGenerator(k_max=12, config=DEFAULT_EXEC_CONFIG,
                                         reuse_plans=arm == 'batch_index_plans')
                policy = ResolverPolicy(CranePolicy(net, batched=arm != 'scalar_scan'))
                start = time.perf_counter_ns()
                raw = run_joint_episode(sim, policy, RewardCalculator.numeraire_v1(), generator=gen)
                samples[arm].append((time.perf_counter_ns()-start)/1e6)
                raw = {key: value.as_dict() if hasattr(value, 'as_dict') else value
                       for key, value in raw.items()}
                if repeat == 0:
                    payload = dict(events=sim.event_log, result=raw,
                        jobs=[(j.job_id, j.status.value, j.service_start, j.service_end)
                              for j in sim.jobs.values()])
                    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                    outcomes[arm] = dict(trajectory_sha256=digest, decisions=raw.get('n_decisions'),
                        remaining=sum(j.status != JobStatus.DONE for j in sim.jobs.values()),
                        cache=sim.dispatch_cache.stats.copy(), result=raw)
        same = len({o['trajectory_sha256'] for o in outcomes.values()}) == 1
        row = dict(seed=seed, same_trajectory_and_cost=same, arms={a: dict(
            samples_ms=samples[a], median_ms=statistics.median(samples[a]), **outcomes[a]) for a in ARMS})
        result['rows'].append(row)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'seed={seed} same={same}: ' + ' / '.join(
            f"{a}={row['arms'][a]['median_ms']:.2f}ms" for a in ARMS), flush=True)
        assert same, row
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    run(p.parse_args().out)
