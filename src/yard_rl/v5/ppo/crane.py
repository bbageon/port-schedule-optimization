"""Sequential conditional choices, with joint feasibility checked BEFORE sampling.

No alternative future world is rolled. dry_run_commit only checks today's physical
reservation feasibility. The sampled action is the exact action sent to the engine.
"""
from __future__ import annotations

import torch

from ..world.contract.schema import CandidateKind
from ..world.integrated.baselines import _apply
from ..world.integrated.candidates import CandidateGenerator
from ..world.integrated.policy_config import LEGACY_DEFAULT
from ..stage.episode import INFO_LEVEL

KINDS = (CandidateKind.SERVE, CandidateKind.PRE_REHANDLE,
         CandidateKind.REPOSITION, CandidateKind.WAIT)


def joint_mask(sim, items, selected):
    committed = {cid: c.job_ref for cid, c in selected.items()
                 if c.kind != CandidateKind.WAIT}
    tokens = {ref.token for ref in committed.values() if ref.token is not None}
    masks = []
    for cid, gc in items:
        ok = gc.feasible
        if ok and gc.kind != CandidateKind.WAIT:
            ok = gc.job_ref is not None and (gc.job_ref.token is None or gc.job_ref.token not in tokens)
            if ok:
                trial = {**committed, cid: gc.job_ref}
                result = sim.dry_run_commit(trial)
                ok = set(result.plans) == set(trial)
        masks.append(bool(ok))
    if not any(masks):
        raise RuntimeError("No feasible action including WAIT")
    return torch.tensor(masks, dtype=torch.bool)


def candidate_row(sim, gc, block_row, selected):
    ref, plan = gc.job_ref, gc.plan
    j = None if ref is None else sim.jobs.get(ref.job_id)
    wait = sim.cum_wait(ref.job_id) if ref is not None and ref.is_external and j is not None else 0.0
    # cum_wait is realized waiting so far, never an unobserved future arrival.
    row = list(block_row) + [float(gc.kind == k) for k in KINDS]
    row += [float(ref is not None and ref.is_vessel),
            float(ref is not None and ref.is_external),
            0.0 if plan is None else plan.duration_s / 3600,
            float(wait or 0.0) / 3600,
            0.0 if plan is None else plan.empty_gantry_m / 100,
            0.0 if plan is None else plan.rehandles / 10,
            0.0 if plan is None else plan.end_bay / 100]
    # The second crane observes the first crane's conditional commitment.
    prior = next(reversed(selected.values()), None) if selected else None
    row += [float(prior is not None and prior.kind == k) for k in KINDS]
    row += [0.0 if prior is None or prior.plan is None else prior.plan.end_bay / 100]
    return row


class CraneActor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.generators = {}

    def __call__(self, sim, dp):
        rt = self.runtime
        bid = rt.block_of[id(sim)]
        # Synchronize only this timestamp's realized events; future stamps are gated.
        # Each block can be inside the same 60-second barrier at a different clock.
        rt.bridge._sync(rt.mbt, sim.now, block_ids=(bid,))
        bf = rt.block_state(bid, sim.now)
        generator = self.generators.setdefault(bid, CandidateGenerator(config=LEGACY_DEFAULT))
        selected = {}
        for cid in sorted(dp.crane_ids):
            candidates = generator.generate(sim, cid, INFO_LEVEL).items
            mask = joint_mask(sim, [(cid, gc) for gc in candidates], selected)
            rows = [candidate_row(sim, gc, bf, selected) for gc in candidates]
            idx = rt.select("crane", bid, sim.now, rows, mask=mask)
            selected[cid] = candidates[idx]
            name = candidates[idx].kind.name
            rt.crane_actions[name] = rt.crane_actions.get(name, 0) + 1
        _apply(sim, selected)  # Exceptions intentionally propagate; no hidden WAIT fallback.
