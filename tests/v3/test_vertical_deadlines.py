"""Layout diagnostics hold external deadlines fixed when transport changes.

The intentionally long road makes the physical lower bound exceed the external
deadline, exposing a silent deadline-relaxation regression without a month run.
"""
from dataclasses import asdict

import pytest

from yard_rl.v3.layouts import VerticalEnvironment, synthetic_vertical_spec
from yard_rl.v3.stage.month_engine import MonthTerminal, inject_vessel
from yard_rl.v3.world.integrated.engine import TerminalSimulator
from yard_rl.v3.world.integrated.profiles import build_h21_profile
from yard_rl.v3.world.integrated.scenario import TerminalScenario
from yard_rl.v3.world.integrated.yard_layout import terminal_layout


def terminal(*, vertical=False):
    profile = build_h21_profile()
    scenario = TerminalScenario('deadline-contract', 99194003, 100_000, 0, {}, [], [])
    if vertical:
        spec = synthetic_vertical_spec(profile)
        spec['geometry']['quay_approach_m'] = 20_000
        env = VerticalEnvironment.from_dict(spec, profile, terminal_layout())
        sim = env.make_sim(scenario, profile, 'Y01')
    else:
        sim = TerminalSimulator(profile, scenario)
    return MonthTerminal({'Y01': sim}), sim, profile


def admit(world, work, **kwargs):
    return inject_vessel(world, 'Y01',
        dict(start_s=600., work=work, cadence_s=144., moves=3),
        key='same-request', size_seed='same-size-draws',
        defer_load_targets=True, **kwargs)


@pytest.mark.parametrize('work', ['LOAD', 'DISCHARGE'])
def test_vertical_transport_does_not_relax_canonical_vessel_or_job_deadlines(work):
    source, baseline, canonical = terminal()
    changed, vertical, _ = terminal(vertical=True)
    original = admit(source, work)
    derived = admit(changed, work, planning_profile=canonical)
    before = baseline.vessels['same-request'].plan
    after = vertical.vessels['same-request'].plan

    assert vertical.profile.transfer.move_time_s > baseline.profile.transfer.move_time_s
    assert asdict(original) == asdict(derived)
    assert after.planned_start_s == before.planned_start_s == 600.
    assert after.planned_completion_s == before.planned_completion_s == 1464.
    assert after.etd_s == before.etd_s == 1896.
    assert after.phys_min_completion_s > after.planned_completion_s
    assert after.phys_min_completion_s > before.phys_min_completion_s
    assert vertical.vessels['same-request'].structural_min_overrun_s() > 0
    # Preserve job identities, sizes, release times and externally promised
    # deadlines too; only the runtime world and its physical bound differ.
    assert {k: asdict(v) for k, v in vertical.jobs.items()} == {
        k: asdict(v) for k, v in baseline.jobs.items()}
    assert len(vertical.jobs) == 3


@pytest.mark.parametrize('work', ['LOAD', 'DISCHARGE'])
def test_default_injection_keeps_existing_physical_deadline_adjustment(work):
    implicit, first, _ = terminal(vertical=True)
    explicit_none, second, _ = terminal(vertical=True)
    a = admit(implicit, work)
    b = admit(explicit_none, work, planning_profile=None)
    plan = first.vessels['same-request'].plan

    assert asdict(a) == asdict(b)
    assert plan == second.vessels['same-request'].plan
    assert plan.planned_completion_s == plan.phys_min_completion_s
    assert plan.planned_completion_s > 1464.
    assert plan.etd_s == plan.planned_completion_s + 432.
    assert first.vessels['same-request'].structural_min_overrun_s() == 0
    assert {k: asdict(v) for k, v in first.jobs.items()} == {
        k: asdict(v) for k, v in second.jobs.items()}
