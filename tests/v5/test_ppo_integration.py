"""Short real 21-block runs: every role, continuous state, no CF teacher, reload."""
import copy

import pytest
import torch

from yard_rl.v5.ppo.checkpoint import load_policy, save_checkpoint
from yard_rl.v5.ppo.run import run_debug
from yard_rl.v5.ppo.model import BlockPolicy
from yard_rl.v5.ppo.runtime import DebugStop, PPOConfig, PPORuntime
from yard_rl.v5.stage.branchpool import BranchPool
from yard_rl.v5.stage.month import DAY_S, DayPlan
from yard_rl.v5.stage.month_run import run_month
from yard_rl.v5.world.integrated.engine import TerminalSimulator


def test_real_engine_all_roles_no_world_clone_continuous_updates(monkeypatch, tmp_path, fixed_container_input):
    def forbidden(*args, **kwargs):
        raise AssertionError("PPO must not clone a terminal or start a CF worker pool")
    monkeypatch.setattr(TerminalSimulator, "__deepcopy__", forbidden, raising=False)
    monkeypatch.setattr(BranchPool, "__enter__", forbidden)
    seen = []
    original = PPORuntime._update
    def checked_update(self, bootstrap):
        world = (id(self.mbt), tuple(id(self.mbt.blocks[b]) for b in self.bids))
        before = self.mbt.now
        jobs = tuple((b, tuple(s.jobs)) for b, s in self.mbt.blocks.items())
        original(self, bootstrap)
        assert self.mbt.now == before
        assert jobs == tuple((b, tuple(s.jobs)) for b, s in self.mbt.blocks.items())
        seen.append(world)
    monkeypatch.setattr(PPORuntime, "_update", checked_update)
    rt, report = run_debug(duration_s=43200, config=PPOConfig(rollout_intervals=120, epochs=1))
    assert len(set(seen)) == 1 and len(seen) == 6
    assert report["blocks"] == 21 and report["counterfactual_worlds"] == 0
    assert all(report["roles"].get(role, 0) > 0 for role in ("seller", "buyer", "crane"))
    assert all(report["crane_actions"].get(kind, 0) > 0
               for kind in ("SERVE", "WAIT", "REPOSITION", "PRE_REHANDLE"))
    assert report["txn_failed"] == 0 and report["traded_edges"] > 0
    assert report["engine_end_s"] > report["time_s"] == 43200
    assert report["team_reward"] == pytest.approx(-report["cost_krw"] / 1e6)
    # Vessel supply jobs really progressed; a quiet truck-only smoke is insufficient.
    assert any(v.started and v.remaining_moves < v.plan.total_moves
               for s in rt.mbt.blocks.values() for v in s.vessels.values())
    assert any(s.jobs for s in rt.mbt.blocks.values())  # unfinished world retained
    for s in rt.mbt.blocks.values():
        s.check_invariants()
    save_checkpoint(tmp_path / "p.pt", rt)
    loaded = load_policy(tmp_path / "p.pt")
    states = rt.states_at(rt.time_s)
    assert torch.equal(rt.policy.value(states), loaded.value(states))
    snapshot = copy.deepcopy(loaded.state_dict())
    _, a = run_debug(policy=loaded, training=False, duration_s=3600)
    _, b = run_debug(policy=loaded, training=False, duration_s=3600)
    for key in ("roles", "cost_krw", "crane_actions", "team_reward", "traded_edges"):
        assert a[key] == b[key]
    assert a["updates"] == [] and a["counterfactual_worlds"] == 0
    assert all(torch.equal(snapshot[k], v) for k, v in loaded.state_dict().items())


@pytest.mark.parametrize("kwargs", [{"labels_per_day": 1}, {"explore": .1},
                                  {"arm": "NO_REALLOC"}])
def test_no_mixing_legacy_teacher_or_policy(kwargs):
    with pytest.raises(ValueError, match="PPO"):
        run_month(seed=9900302, ppo=object(), **kwargs)


def test_only_diagnostic_seeds_allowed():
    with pytest.raises(ValueError, match="diagnostic"):
        run_debug(seed=9400000, duration_s=60)


def test_midnight_cleanup_does_not_reset_cost_or_world(fixed_container_input):
    torch.manual_seed(9900302)
    torch.set_num_threads(1)
    rt = PPORuntime(BlockPolicy(), training=False, stop_s=DAY_S + 60)
    days = [DayPlan(i, 60, "midnight-debug", 9901302 + 1000 * i, i * DAY_S, 2)
            for i in range(2)]
    with pytest.raises(DebugStop):
        run_month(seed=9900302, days=days, ppo=rt)
    assert rt.time_s == DAY_S + 60
    assert rt.intervals == 1441
    assert rt.total_reward == pytest.approx(-rt.cost_krw / rt.config.reward_scale_krw)
    assert len(rt.mbt.blocks) == 21 and rt.cost_krw > 0
