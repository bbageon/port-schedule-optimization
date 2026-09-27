"""Preregistered six-hour execution check with the empirical reward unit."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from yard_rl.v6.ppo.checkpoint import save_checkpoint, load_policy
from yard_rl.v6.ppo.journal import RunJournal, write_json
from yard_rl.v6.ppo.model import BlockPolicy
from yard_rl.v6.ppo.provenance import code_stamp, file_sha256
from yard_rl.v6.ppo.runtime import DebugStop, PPOConfig, PPORuntime
from yard_rl.v6.stage.month import build_month, plan_days, plan_month_vessels, truck_net_by_block
from yard_rl.v6.stage.fixed_seed import build_fixed_seed
from yard_rl.v6.stage.month_run import run_month
from yard_rl.v6.stage.seed_bundle import save_seed_bundle
from yard_rl.v6.world.integrated.terminal_stream import OBS_24H
from yard_rl.v6.world.integrated.yard_layout import terminal_layout


def run(output):
    torch.set_num_threads(1)
    seed, stop = 9900628, 21600.
    torch.manual_seed(seed)
    config = PPOConfig()
    stamp = code_stamp()
    out = Path(output)
    days = plan_days(seed, loads=[7500, 7500])
    prereg = Path('.claude/docs/dashboard-task-specs/YR-331-d-reward-normalization.md')
    manifest = dict(code=stamp, seed=seed, ppo=asdict(config), stop_s=stop,
                    learning_window_s=[0., stop], training_during_first_day=True,
                    prereg_sha256=file_sha256(prereg), scope='execution check, no performance claim')
    journal = RunJournal(out, days, manifest)
    rt = None
    try:
        built = build_month(seed, days=days)
        ships = plan_month_vessels(days, terminal_layout(), obs=OBS_24H,
                                  truck_net=truck_net_by_block(built['schedule']))
        document, audit = build_fixed_seed(built, ships, seed=seed)
        save_seed_bundle(document, out/'seed-data.json.gz')
        write_json(out/'input-audit.json', audit)
        policy = BlockPolicy()
        before = {k: v.clone() for k,v in policy.state_dict().items()}
        rt = PPORuntime(policy, config=config, seed=seed, training=True, stop_s=stop,
                         on_update=journal.update, on_boundary=journal.boundary)
        try:
            run_month(seed=seed, days=days, ppo=rt, seed_data=document,
                      on_admission=journal.admission, on_container_contract=journal.container_contract)
        except DebugStop:
            pass
        for sim in rt.mbt.blocks.values():
            sim.check_invariants()
        report = rt.report()
        change = float(torch.sqrt(sum((v-before[k]).square().sum() for k,v in policy.state_dict().items())))
        assert rt.time_s == stop and len(rt.updates) == 6 and change > 0
        assert all(x['minibatches'] > 0 for x in rt.updates)
        assert all(math.isfinite(v) for row in rt.updates for v in row.values()
                   if isinstance(v, float))
        assert math.isclose(-rt.total_reward*config.reward_scale_krw,
                            rt.cost_krw-rt.initial_cost, rel_tol=1e-10)
        save_checkpoint(out/'policy.pt', rt)
        restored = load_policy(out/'policy.pt')
        assert all(torch.equal(v, restored.state_dict()[k]) for k,v in policy.state_dict().items())
        PPORuntime(restored, config=config)
        report.update(status='complete', source=stamp, parameter_l2_change=change,
                      physical_invariants=True, admission=journal.admissions,
                      checkpoint_reload_exact=True, performance_claim=False)
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(status='complete', time_s=rt.time_s, updates=len(rt.updates)))
        print(json.dumps(dict(status='complete', updates=len(rt.updates),
                               parameter_l2_change=change, scale_krw=config.reward_scale_krw)), flush=True)
    except BaseException as exc:
        if rt is None:
            write_json(out/'status.json', dict(status='failed', error=str(exc), phase='input'))
        else:
            journal.fail(exc, rt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
