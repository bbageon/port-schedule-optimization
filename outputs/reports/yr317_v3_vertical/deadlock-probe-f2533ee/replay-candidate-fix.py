"""Replay a saved blocked block, changing only the candidate generator.

Diagnostic local tails; no new terminal arrivals, learning, or performance claim.
"""
from pathlib import Path
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
report = Path(__file__).resolve().parent
workspace = report.parents[3]
frozen = Path('/home/geonu/yr317-v3-vertical-f2533ee')
sys.path.insert(0, str(frozen / 'src'))

from yard_rl.v3.world.domain.enums import JobStatus
from yard_rl.v3.world.integrated.candidates import CandidateGenerator
from yard_rl.v3.world.integrated.engine import ReviewEpoch
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stats(sim):
    return dict(clock_s=sim.clock, job_status=dict(Counter(j.status.value for j in sim.jobs.values())),
        yard_done=sum(j.status == JobStatus.DONE for j in sim.jobs.values()),
        vessel_yard_done=sum(j.status == JobStatus.DONE and j.is_vessel_linked for j in sim.jobs.values()),
        containers=len(sim.stacks.containers), active_plans=len(sim._active_plans),
        reservations=len(sim.reservations.active()), discharge_pipeline=dict(sim._discharge_pipeline),
        positions={c.crane_id: dict(bay=c.state.position_bay, row=c.state.trolley_row,
            yielded=c.yielded, assigned=c.state.assigned_job) for c in sim.fleet.all()},
        event_stream_hash=sim.event_stream_hash())


def replay(snapshot, generator, rule_policy):
    sim = copy.deepcopy(snapshot)
    # Identical, explicit initial intervention; no later forced waking/assignment.
    sim._clear_yields()
    sim.end = min(sim.end, sim.clock + 3600.0)
    sim._check = True
    policy, exceptions = rule_policy('SF_SPT', seed=99194001,
                                    candidate_generator_cls=generator)
    initial = stats(sim)
    initial_candidates = {cid: [dict(kind=g.kind.value, feasible=g.feasible,
        mask_reason=g.mask_reason, job=None if g.job_ref is None else g.job_ref.job_id)
        for g in generator().generate(sim, cid, sim.info_level).items] for cid in sim.fleet.ids()}
    decisions, reviews, checks = 0, 0, 0
    chosen = Counter()
    sim.check_invariants()
    while True:
        if decisions + reviews > 10000:
            raise RuntimeError('Diagnostic tail exceeded 10000 events')
        dp = sim.run_until_decision()
        sim.check_invariants()
        checks += 1
        if dp is None:
            break
        if isinstance(dp, ReviewEpoch):
            reviews += 1
            continue
        policy(sim, dp)
        decisions += 1
        if exceptions['n']:
            raise RuntimeError(f"Policy fallback occurred: {exceptions}")
        for assignment in sim.last_assignments().values():
            chosen[assignment.action.value] += 1
        sim.check_invariants()
        checks += 1
    final = stats(sim)
    return dict(initial=initial, initial_candidates=initial_candidates, final=final,
        newly_completed=final['yard_done']-initial['yard_done'], decisions=decisions,
        reviews=reviews, chosen=dict(chosen), invariant_checks=checks,
        policy_exceptions=exceptions['n'], original_snapshot_not_mutated=True)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if b'python' in (p/'cmdline').read_bytes() and os.sched_getaffinity(int(p.name)) == {19}:
                raise RuntimeError(f'CPU19 occupied: {p.name}')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    os.sched_setaffinity(0, {19})
    os.nice(10)
    candidate_path = workspace/'src/yard_rl/v3/layouts/candidates.py'
    episode_path = workspace/'src/yard_rl/v3/stage/episode.py'
    fixed = load_module('yard_rl.v3.layouts._probe_candidate_fix', candidate_path)
    episode = load_module('yard_rl.v3.stage._probe_episode', episode_path)
    identity = dict(frozen_source_commit=subprocess.check_output(
        ['git', '-C', str(frozen), 'rev-parse', 'HEAD'], text=True).strip(),
        candidate_file=str(candidate_path), candidate_sha256=sha(candidate_path),
        rule_policy_file=str(episode_path), rule_policy_file_sha256=sha(episode_path),
        frozen_runtime_sha256=sha(frozen/'src/yard_rl/v3/layouts/runtime.py'),
        frozen_base_candidates_sha256=sha(inspect.getfile(CandidateGenerator)),
        diagnostic_script_sha256=sha(__file__), cpu=19, pid=os.getpid())
    result = dict(purpose='Saved local-state regression diagnostic; not performance or whole-horizon qualification',
        conditions=dict(horizon_s=3600, policy='SF_SPT', initial_clear_yields_both=True,
            initial_check_invariants_both=True, no_new_external_admissions=True,
            no_learning=True, no_repeated_manual_wake=True), identity=identity, blocks={})
    started = time.monotonic()
    try:
        for bid in ('Y05', 'Y14'):
            path = report/(bid+'-snapshot.pkl.gz')
            with gzip.open(path, 'rb') as stream:
                snapshot = pickle.load(stream)
            original_bytes = pickle.dumps(snapshot, protocol=4)
            rows = dict(snapshot_sha256=sha(path))
            for label, cls in (('original', CandidateGenerator), ('fixed', fixed.FeasibleCandidateGenerator)):
                rows[label] = replay(snapshot, cls, episode._rule_policy)
                assert pickle.dumps(snapshot, protocol=4) == original_bytes
                print(json.dumps(dict(block=bid, arm=label,
                    newly_completed=rows[label]['newly_completed'],
                    policy_exceptions=rows[label]['policy_exceptions'],
                    invariant_checks=rows[label]['invariant_checks'])), flush=True)
            rows['regression_passed'] = rows['original']['newly_completed'] == 0 and rows['fixed']['newly_completed'] > 0
            result['blocks'][bid] = rows
        result['passed'] = all(r['regression_passed'] for r in result['blocks'].values())
        if not result['passed']:
            raise RuntimeError('Blocked-state progress regression failed')
    except BaseException:
        result['failure'] = traceback.format_exc()
        raise
    finally:
        result['elapsed_s'] = time.monotonic()-started
        (report/'candidate-fix-replay.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
