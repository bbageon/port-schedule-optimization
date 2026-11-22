from types import SimpleNamespace as NS

import pytest

from yard_rl.v7.stage.vessel_turnaround import VesselTurnaround
from yard_rl.v7.stage.vessels import DailyVessel, VESSEL_CLASSES, plan_streams


def setup_calls():
    ship = DailyVessel("A", VESSEL_CLASSES[0], 101, 20000., 1800.)
    schedule = plan_streams([ship], NS(ids=("b1", "b2")), 7, full_calls=True)
    blocks = {}
    for row, end in zip(schedule, (23*3600., 30*3600.)):
        process = NS(plan=NS(total_moves=row["moves"]), truth=NS(actual_completion_s=end),
                     started=True, remaining_moves=0)
        blocks[row["block"]] = NS(vessels={"stream":process})
    return {"fleet":[ship], "vessel_schedule":schedule}, NS(blocks=blocks)


def test_one_clock_for_two_cranes_and_overnight_service():
    built, terminal = setup_calls()
    ledger = VesselTurnaround(built, terminal)
    assert sorted(r["moves"] for r in built["vessel_schedule"]) == [50, 51]
    before, partial, final = (ledger.rows(t)[0] for t in (60., 24*3600., 32*3600.))
    assert before["hours"] == 0.
    assert partial["hours"] == 23.5 and partial["finish_s"] is None
    assert final["hours"] == 29.5 and final["finish_s"] == 30*3600.
    assert final["cost"] == pytest.approx(35.88*built["fleet"][0].gt*29.5)


def test_unfinished_and_missing_streams_cannot_be_counted_as_completed():
    built, terminal = setup_calls()
    stream = next(iter(next(iter(terminal.blocks.values())).vessels.values()))
    stream.truth.actual_completion_s = None
    stream.remaining_moves = 7
    row = VesselTurnaround(built, terminal).rows(32*3600.)[0]
    assert row["finish_s"] is None and row["remaining"] == 7
    built["vessel_schedule"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        VesselTurnaround(built, terminal)


def test_last_quay_move_after_intake_cutoff_enters_policy_return(monkeypatch):
    import torch
    from yard_rl.v7.crane.actor_critic import SchedulingActorCritic
    from yard_rl.v7.crane.on_policy import OnPolicyResolver
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage import vessels
    from yard_rl.v7.world.integrated.terminal_stream import ObservationContract

    ship = DailyVessel("fixture", VESSEL_CLASSES[0], 16, 20000., 3300.)
    monkeypatch.setattr(vessels, "sample_day_vessels", lambda *a, **kw: [ship])
    torch.manual_seed(190022)
    net = SchedulingActorCritic(kind="mlp").eval()
    obs = ObservationContract(warmup_s=0., measure_s=3600.)
    resolver = OnPolicyResolver(net, collect=True, stochastic=True, seed=190022,
                                horizon=obs.observe_s, collect_until=39600.)
    ep = run_episode(load=80, seed=190023, dispatcher="RL_CRANE", crane_net=net,
                     joint_resolver=resolver, obs=obs, vessel_drain_s=36000.)
    calls = ep.breakdown["vessel_calls"]
    assert ep.breakdown["vessel_complete"] and calls[0]["finish_s"] > obs.observe_s
    before = dict(resolver.costs)[obs.observe_s]
    tail = (calls[0]["finish_s"] - obs.observe_s)*35.88*ship.gt/3600.
    assert ep.phi_krw-before == pytest.approx(tail)
    assert resolver.finish(ep.phi_krw, 100_000.)["samples"] > 0
    assert any(s.state.time_fraction >= 1 for s in resolver.samples)
