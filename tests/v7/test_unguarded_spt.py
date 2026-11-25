from dataclasses import replace

import pytest

from yard_rl.v7.stage.episode import _rule_policy
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
from yard_rl.v7.world.integrated.scenario import InjectedEvent


@pytest.mark.parametrize(("rule", "expected"), (("SPT_UNGUARDED", "short"), ("SF_SPT", "long")))
def test_dispatch_old_blocked_container_and_new_short_job(rule, expected):
    profile = build_integrated_profile()
    profile = replace(profile, cranes=(profile.cranes[0],), long_wait_sla_s=1800.)
    scenario = build_minimal_terminal_scenario()
    long_job = replace(scenario.jobs[0], job_id="long", actual_gate_in=0., actual_block_arrival=0.)
    short_job = replace(long_job, job_id="short", target_container="C-A2",
                        actual_gate_in=1390., actual_block_arrival=1400.)
    crane = profile.cranes[0].crane_id
    events = [InjectedEvent(0., "EQUIPMENT_DOWN", crane), InjectedEvent(1440., "EQUIPMENT_UP", crane)]
    sim = TerminalSimulator(profile, replace(scenario, jobs=[long_job, short_job], vessels=[],
                                             injected_events=events))
    selected = []
    def capture(world, decision, candidates, assignment):
        choice = assignment[crane]
        if choice.job_ref is not None:
            selected.append((world.clock, choice.job_ref.job_id))
    policy, errors = _rule_policy(rule, candidate_cap=None, on_crane=capture)
    while not selected and (decision := sim.run_until_decision()) is not None:
        policy(sim, decision)
    assert errors == {"n": 0, "hook": 0}
    assert selected == [(1440., expected)]
