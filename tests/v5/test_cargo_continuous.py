"""The real new generator -> bundle -> 3-day PPO execution path, without CF."""
import os

import pytest

from yard_rl.v5.ppo import continuous
from yard_rl.v5.stage.fixed_seed import build_fixed_seed
from yard_rl.v5.stage.month import build_month, plan_month_vessels, truck_net_by_block
from yard_rl.v5.stage.seed_bundle import save_seed_bundle
from yard_rl.v5.world.integrated.yard_layout import terminal_layout


def generate(tmp_path):
    days = continuous.make_plan(9900306,3,60)
    built = build_month(9900306,days=days)
    vessels = plan_month_vessels(days,terminal_layout(),truck_net=truck_net_by_block(built['schedule']))
    document,_ = build_fixed_seed(built,vessels,seed=9900306)
    path = tmp_path/'seed.json.gz'
    info = save_seed_bundle(document,path)
    return path,info


def test_actual_new_bundle_three_days_calendar_cost_and_full_workload(monkeypatch,tmp_path):
    from yard_rl.v5.stage.branchpool import BranchPool
    from yard_rl.v5.world.integrated.engine import TerminalSimulator
    def forbidden(*args,**kwargs):
        raise AssertionError('Fixed cargo PPO may not clone a world or use CF labels')
    monkeypatch.setattr(TerminalSimulator,'__deepcopy__',forbidden,raising=False)
    monkeypatch.setattr(BranchPool,'__enter__',forbidden)
    path,info = generate(tmp_path)
    monkeypatch.setattr(continuous,'code_stamp',lambda:dict(test_only=True,pid=os.getpid()))
    runtime,report = continuous.run_continuous(output=tmp_path/'run',seed=9900306,n_days=3,load=60,
                                              seed_bundle=path,seed_sha256=info['sha256'])
    assert report['state'] == 'completed' and report['admitted'] == 180 and report['skipped'] == 0
    assert report['counterfactual_worlds'] == 0 and len(runtime.updates) == 24
    assert runtime.learning_intervals == 1440
    assert report['cargo']['physical_inventory'] == report['cargo']['expected_inventory']
    assert report['cargo']['dependent_reroutes'] > 0
    assert report['cargo']['remote_vessel_handoffs'] > 0
    assert all(runtime.mbt.ready_times[c] <= t for c,t in runtime.mbt.exit_times.items())
    assert (tmp_path/'run/final.pt').exists() and (tmp_path/'run/cargo-status.json').exists()


def test_wrong_bundle_hash_rejected_before_output_or_world(monkeypatch,tmp_path):
    monkeypatch.setattr(continuous,'code_stamp',lambda:dict(test_only=True,pid=os.getpid()))
    path,info = generate(tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        continuous.run_continuous(output=tmp_path/'run',seed=9900306,n_days=3,load=60,
                                  seed_bundle=path,seed_sha256='0'*64)
    assert not (tmp_path/'run').exists()
