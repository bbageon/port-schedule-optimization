"""Calendar-window collection and durable continuous-run regressions."""
import json
import os

import numpy as np
import pytest
import torch

from yard_rl.v5.ppo import continuous, provenance
from yard_rl.v5.ppo.checkpoint import load_policy
from yard_rl.v5.ppo.model import BlockPolicy, encode
from yard_rl.v5.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v5.stage.branchpool import BranchPool
from yard_rl.v5.stage.month import DAY_S
from yard_rl.v5.world.integrated.engine import TerminalSimulator


def test_learning_window_excludes_edges_but_keeps_sampling_and_bootstrap():
    rt = PPORuntime(BlockPolicy(), learning_window_s=(60, 180))
    rt.bids, rt.index = ["b"], {"b": 0}
    rt.states_at = lambda t: encode([[t / 3600]], "state")
    rt.read_cost = lambda t: 100 + 2 * t
    batches, bootstraps, hooks = [], [], []
    def collect(bootstrap):
        if rt.buffer:
            batches.append(list(rt.buffer))
            bootstraps.append(bootstrap.copy())
            rt.buffer.clear()
    rt._update = collect
    rt.on_boundary = lambda r: hooks.append((r.time_s, len(batches)))
    for t in (0, 60, 120, 180, 240):
        rt.boundary(t)
        before = rt.action_rng.get_state().clone()
        rt.select("seller", "b", t, [[0], [1]])
        assert not torch.equal(before, rt.action_rng.get_state())
        assert len(rt.pending[0]) == int(60 <= t < 180)
    rt.finish(300)
    assert [[r.start_s for r in batch] for batch in batches] == [[60, 120]]
    assert rt.intervals == 5 and rt.learning_intervals == 2
    assert rt.total_reward == pytest.approx(-600 / 1e6)
    assert rt.learning_reward == pytest.approx(-240 / 1e6)
    expected = rt.policy.value(rt.states_at(180)).detach().numpy()
    np.testing.assert_array_equal(bootstraps[0], expected)
    assert not any(r.terminated for r in batches[0])
    assert (180, 1) in hooks  # End-of-learning flush precedes the day checkpoint.


@pytest.mark.parametrize("window", [(-1, 60), (60, 60), (120, 60), (0, float("inf"))])
def test_invalid_learning_window(window):
    with pytest.raises(ValueError, match="window"):
        PPORuntime(BlockPolicy(), learning_window_s=window)


def test_crossed_window_boundary_is_not_silently_misattributed():
    rt = PPORuntime(BlockPolicy(), learning_window_s=(60, 180))
    rt.bids = ["b"]
    rt.states_at = lambda t: encode([[0]], "state")
    rt.read_cost = lambda t: t
    rt.boundary(0)
    with pytest.raises(RuntimeError, match="exactly"):
        rt.boundary(120)


@pytest.mark.parametrize("kwargs", [dict(seed=9400000), dict(seed=True), dict(n_days=2),
                                    dict(n_days=31), dict(n_days=3.5), dict(load=0),
                                    dict(load=True), dict(seed=9999000)])
def test_bad_plan_rejected_before_world(kwargs):
    with pytest.raises(ValueError):
        continuous.make_plan(**(dict(seed=9900306, n_days=30) | kwargs))


def test_registered_plan_is_30_continuous_days_with_28_learning_days():
    days = continuous.make_plan(9900306, 30)
    assert days == continuous.make_plan(9900306, 30)
    assert sum(d.is_train for d in days) == 28
    assert not days[0].is_train and not days[-1].is_train
    assert [d.t0 for d in days] == [i * DAY_S for i in range(30)]
    assert all(d.load in (3500, 5000, 7500, 12500, 15000) for d in days)


@pytest.mark.parametrize("dirty,wrong_root", [(True, False), (False, True)])
def test_provenance_rejects_dirty_or_wrong_source(monkeypatch, tmp_path, dirty, wrong_root):
    root = tmp_path / "source"
    monkeypatch.setattr(provenance, "__file__", str(root / "src/yard_rl/v5/ppo/provenance.py"))
    def git(args, **kwargs):
        if args[-1] == "--show-toplevel":
            return str(tmp_path if wrong_root else root)
        return " M dirty.py" if dirty else ""
    monkeypatch.setattr(provenance.subprocess, "check_output", git)
    with pytest.raises(RuntimeError, match="worktree"):
        provenance.code_stamp()


def test_failed_world_is_reported_without_claiming_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(continuous, "code_stamp", lambda: {"pid": os.getpid(), "test_only": True})
    def failed(**kwargs):
        raise RuntimeError("injected world failure")
    monkeypatch.setattr(continuous, "run_month", failed)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        continuous.run_continuous(output=out, n_days=3, load=60)
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "failed" and "NEW directory" in status["restart"]
    assert (out / "initial.pt").exists() and not (out / "final.pt").exists()
    with pytest.raises(FileExistsError):
        continuous.run_continuous(output=out, n_days=3, load=60)


def test_real_three_day_run_window_checkpoints_no_clones(monkeypatch, tmp_path, fixed_container_input):
    monkeypatch.setattr(continuous, "code_stamp", lambda: {"pid": os.getpid(), "test_only": True})
    def forbidden(*args, **kwargs):
        raise AssertionError("No counterfactual world is allowed")
    monkeypatch.setattr(TerminalSimulator, "__deepcopy__", forbidden, raising=False)
    monkeypatch.setattr(BranchPool, "__enter__", forbidden)
    identities = set()
    original = PPORuntime._update
    def checked(self, bootstrap):
        identities.add((id(self.mbt), tuple(id(s) for s in self.mbt.blocks.values())))
        original(self, bootstrap)
    monkeypatch.setattr(PPORuntime, "_update", checked)
    out = tmp_path / "three-day"
    rt, report = continuous.run_continuous(output=out, n_days=3, load=60)
    assert len(identities) == 1
    assert report["counterfactual_worlds"] == 0 and report["state"] == "completed"
    assert rt.learning_intervals == 1440 and len(rt.updates) == 24
    days = json.loads((out / "days.json").read_text())
    assert [d["updates"] for d in days] == [0, 24, 0]
    assert [d["train"] for d in days] == [False, True, False]
    assert days[1]["interval_cost_krw"] == pytest.approx(-rt.learning_reward * 1e6)
    policies = [load_policy(out / name).state_dict()
                for name in ("initial.pt", "day_01.pt", "day_02.pt", "day_03.pt", "final.pt")]
    equal = lambda a, b: all(torch.equal(a[k], b[k]) for k in a)
    assert equal(policies[0], policies[1]) and not equal(policies[1], policies[2])
    assert equal(policies[2], policies[3]) and equal(policies[3], policies[4])
    assert all(d["checkpoint"]["pending_intervals"] == 0 for d in days)
    assert not list(out.glob("*.partial"))
    assert json.loads((out / "status.json").read_text())["completed_days"] == 3
    admissions = json.loads((out / "admissions.json").read_text())
    assert json.loads((out / 'container_contract.json').read_text())['passed']
    assert admissions["admitted"] == 180 and admissions["skipped"] == 0
    assert admissions["vessel_failed"] == 0 and not admissions["truck_failures"]
    assert len(json.loads((out / "cohort_live.json").read_text())) == 3
    result = json.loads((out / "month_result.json").read_text())
    assert result["admitted"] == admissions["admitted"] and result["truck_skips"] == []
    assert not (out / "failed-policy.pt").exists()
