"""이동시간 모델 단위 테스트 (05 §1.1)."""
from yard_rl.io.profile_loader import load_profile
from yard_rl.sim.travel_time import gantry_m, hoist_leg_s, move_container

P = load_profile("configs/terminals/poc_single_crane.yaml")
SPEC, GEOM = P.crane, P.block


def test_zero_travel_cycle_is_handling_only():
    """같은 bay·row 간 이동: gantry/trolley 0, 취급시간만."""
    mv = move_container(SPEC, GEOM, 5.0, 2.0, (5, 2, 3), (5, 2, 1))
    expected = (SPEC.lock_time_s + SPEC.unlock_time_s
                + hoist_leg_s(GEOM, SPEC, 3, loaded=False) + hoist_leg_s(GEOM, SPEC, 3, loaded=True)
                + hoist_leg_s(GEOM, SPEC, 1, loaded=True) + hoist_leg_s(GEOM, SPEC, 1, loaded=False))
    assert abs(mv.duration_s - expected) < 1e-9
    assert mv.loaded_gantry_m == 0.0 and mv.empty_gantry_m == 0.0


def test_gantry_distance_split_loaded_vs_empty():
    """빈 주행 = 크레인→src, 적재 주행 = src→dst."""
    mv = move_container(SPEC, GEOM, 1.0, 0.0, (11, 3, 1), (5, 3, 1))
    assert abs(mv.empty_gantry_m - gantry_m(GEOM, 1, 11)) < 1e-9
    assert abs(mv.loaded_gantry_m - gantry_m(GEOM, 11, 5)) < 1e-9
    assert mv.end_bay == 5.0 and mv.end_row == 3.0


def test_farther_is_slower():
    near = move_container(SPEC, GEOM, 1.0, 0.0, (2, 1, 1), (2, 0, 1))
    far = move_container(SPEC, GEOM, 1.0, 0.0, (20, 1, 1), (20, 0, 1))
    assert far.duration_s > near.duration_s


def test_deeper_pick_takes_longer():
    """낮은 tier(깊이 묻힘) 픽업이 더 오래 걸림 — hoist 거리 증가."""
    deep = move_container(SPEC, GEOM, 3.0, 1.0, (3, 1, 1), (3, 0, 1))
    shallow = move_container(SPEC, GEOM, 3.0, 1.0, (3, 1, 4), (3, 0, 1))
    assert deep.duration_s > shallow.duration_s
