"""YR-027 Direct-Job environment contract tests."""
from dataclasses import FrozenInstanceError, replace

import pytest

from yard_rl.domain.enums import ContainerSize, JobFlow, LoadStatus
from yard_rl.domain.models import Container, Job
from yard_rl.domain.scenario import Scenario
from yard_rl.envs.direct_job_env import (DirectJobBucketConfig, DirectJobEnv,
                                         DirectJobEpisodeError, SLAMode, _bucket)
from yard_rl.io.profile_loader import load_profile
from yard_rl.sim.travel_time import estimate_reach_s, move_container

P = load_profile("configs/terminals/poc_single_crane.yaml")


def _container(cid, bay, row, tier=1):
    return Container(cid, ContainerSize.FT40, LoadStatus.FULL, "B1", bay, row, tier)


def _out(jid, target, arrival, *, gate=0.0, eta=None, deadline=None):
    return Job(jid, JobFlow.GATE_OUT, 0.0, gate, arrival, provided_eta=eta,
               deadline=deadline, target_container=target)


def _scenario(jobs, containers, *, horizon=5000.0, drain=0.0):
    return Scenario("direct", 0, horizon, drain, jobs, containers)


def test_rejects_any_actual_non_external_job_and_n_config_mismatch():
    c = _container("C1", 2, 1)
    vessel = Job("JV", JobFlow.VESSEL_LOAD, 0.0, None, None,
                 deadline=100.0, target_container="C1")
    with pytest.raises(ValueError, match="n_vessel=0"):
        DirectJobEnv(P).reset(_scenario([vessel], {"C1": c}))
    with pytest.raises(ValueError, match="N_config 불일치"):
        DirectJobEnv(P, expected_n_config=2).reset(
            _scenario([_out("J1", "C1", 1.0)], {"C1": c}))
    configured = _scenario([_out("J1", "C1", 1.0)], {"C1": c})
    configured.scenario_id = "syn_n1_v2_f10_r0_norm_s0"
    with pytest.raises(ValueError, match="scenario_id"):
        DirectJobEnv(P).reset(configured)


def test_block_entry_release_dynamic_actions_and_policy_safe_features():
    containers = {"C1": _container("C1", 2, 1), "C2": _container("C2", 8, 1)}
    jobs = [_out("JZ", "C2", 100.0, gate=90.0, eta=1.0, deadline=2.0),
            _out("JA", "C1", 100.0, gate=0.0, eta=9999.0, deadline=9999.0),
            Job("JI", JobFlow.GATE_IN, 0.0, 0.0, 200.0,
                inbound_size=ContainerSize.FT20, inbound_load=LoadStatus.FULL)]
    state, info = DirectJobEnv(P).reset(_scenario(jobs, containers))
    assert state is not None
    assert info.allowed_job_ids == ("JA", "JZ")  # 동시 BLOCK_ENTRY 전부 + job_id 결정론
    assert {c.transfer_direction for c in info.candidates} == {"YARD_TO_TRUCK"}
    assert all(c.feature[0] == "YARD_TO_TRUCK" for c in info.candidates)
    assert not ({"actual_gate_in", "provided_eta", "deadline"}
                & set(info.candidates[0].__dataclass_fields__))


def test_candidates_are_existing_constraint_engine_feasible_only():
    narrow = replace(P, crane=replace(P.crane, service_bay_max=10))
    containers = {"CIN": _container("CIN", 3, 1), "COUT": _container("COUT", 20, 1)}
    jobs = [_out("J_IN_RANGE", "CIN", 10.0), _out("J_OUT_RANGE", "COUT", 10.0)]
    env = DirectJobEnv(narrow, strict_clear_out=False)
    _, info = env.reset(_scenario(jobs, containers, horizon=1000.0))
    assert info.allowed_job_ids == ("J_IN_RANGE",)
    with pytest.raises(ValueError, match="허용 후보"):
        env.step("J_OUT_RANGE")


def test_step_executes_selected_job_and_cost_sums_to_mean_wait_minutes():
    containers = {"C1": _container("C1", 2, 1), "C2": _container("C2", 8, 1)}
    env = DirectJobEnv(P, expected_n_config=2)
    _, info0 = env.reset(_scenario([_out("J1", "C1", 100.0),
                                    _out("J2", "C2", 100.0)], containers))
    _, cost1, done1, info1 = env.step(info0.candidates[0])
    assert not done1 and info1.selected_job == "J1"
    assert info1.queue_area_delta_s == pytest.approx(info1.elapsed_s)
    _, cost2, done2, info2 = env.step("J2")
    assert done2 and info2.episode_success
    assert cost1 + cost2 == pytest.approx(env.sim.kpis.queue_area_s / (60.0 * 2))
    assert cost1 + cost2 == pytest.approx(sum(env.sim.kpis.wait_samples_s) / (60.0 * 2))
    assert info2.cumulative_cost == pytest.approx(cost1 + cost2)


def test_sla_on_masks_only_after_physical_feasibility_and_is_inclusive():
    containers = {"C0": _container("C0", 2, 1), "C1": _container("C1", 5, 1),
                  "C2": _container("C2", 8, 1)}
    probe = DirectJobEnv(P)
    _, first = probe.reset(_scenario([_out("J0", "C0", 0.0)], {"C0": containers["C0"]}))
    duration = first.candidates[0].service_s
    sla = duration / 2.0
    profile = replace(P, long_wait_sla_s=sla)
    jobs = [_out("J0", "C0", 0.0), _out("J_OLD", "C1", duration - sla),
            _out("J_NEW", "C2", duration)]
    env = DirectJobEnv(profile, sla_mode=SLAMode.ON)
    _, start = env.reset(_scenario(jobs, containers))
    _, _, _, after = env.step("J0")
    assert after.sla_restricted
    assert after.allowed_job_ids == ("J_OLD",)  # wait == SLA 포함
    assert after.masked_job_ids == ("J_NEW",)


def test_bucket_fit_json_roundtrip_preserves_sla_edge_and_is_immutable(tmp_path):
    cfg = DirectJobBucketConfig.fit(
        queue_lengths=[1, 2, 3, 4], oldest_waits_s=[10, 20, 30, 40],
        own_waits_s=[11, 21, 31, 41], reaches_s=[1, 2, 3, 4],
        service_times_s=[100, 200, 300, 400], sla_s=1800.0)
    assert cfg.fitted and 1800.0 in cfg.oldest_wait_s and 1800.0 in cfg.own_wait_s
    assert _bucket(1799.999, (1800.0,)) == 0
    assert _bucket(1800.0, (1800.0,)) == 1  # inclusive SLA-side bucket
    path = tmp_path / "direct-job-buckets.json"
    cfg.save(path)
    assert DirectJobBucketConfig.load(path) == cfg
    with pytest.raises(FrozenInstanceError):
        cfg.fitted = False


def test_inbound_reach_and_service_use_the_same_selected_slot():
    inbound = Job("JI", JobFlow.GATE_IN, 0.0, 0.0, 0.0,
                  inbound_size=ContainerSize.FT40, inbound_load=LoadStatus.FULL)
    env = DirectJobEnv(P)
    _, info = env.reset(_scenario([inbound], {"C1": _container("C1", 1, 1)}))
    candidate = info.candidates[0]
    sim, geom, spec = env.sim, P.block, P.crane
    dest = sim.stacks.find_slot(inbound.inbound_size, spec,
                                sim.crane.position_bay, sim.crane.trolley_row)
    expected_reach = estimate_reach_s(spec, geom, sim.crane.position_bay,
                                      sim.crane.trolley_row, float(dest[0]),
                                      float(geom.transfer_row))
    tier = sim.stacks.top_tier(*dest) + 1
    expected_service = move_container(
        spec, geom, sim.crane.position_bay, sim.crane.trolley_row,
        (dest[0], geom.transfer_row, 1), (dest[0], dest[1], tier)).duration_s
    expected_service += spec.truck_positioning_time_s
    assert candidate.reach_s == pytest.approx(expected_reach)
    assert candidate.service_s == pytest.approx(expected_service)


def test_outbound_service_estimator_matches_actual_engine_duration_with_blocker():
    containers = {"TARGET": _container("TARGET", 4, 2, 1),
                  "BLOCKER": _container("BLOCKER", 4, 2, 2),
                  "OTHER": _container("OTHER", 12, 1, 1)}
    env = DirectJobEnv(P)
    _, info = env.reset(_scenario([_out("JO", "TARGET", 0.0)], containers))
    candidate = info.candidates[0]
    record = env.sim.execute_job("JO")
    assert candidate.blocker_count == 1
    assert candidate.service_s == pytest.approx(record.duration_s)


def test_clear_out_backlog_is_explicit_episode_failure():
    narrow = replace(P, crane=replace(P.crane, service_bay_max=10))
    scenario = _scenario([_out("J", "C", 10.0)], {"C": _container("C", 20, 1)},
                         horizon=100.0)
    with pytest.raises(DirectJobEpisodeError, match="clear-out 실패"):
        DirectJobEnv(narrow).reset(scenario)
