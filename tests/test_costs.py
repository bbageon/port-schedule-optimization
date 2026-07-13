"""YR-025 원화비용 목적함수 — CostConfig 변환·일관성 검증."""
from yard_rl.envs.rewards import CostConfig, RewardCalculator
from yard_rl.sim.kpis import KpiSnapshot

COST = CostConfig(cost_id="TEST", truck_wait_krw_per_hour=30000,
                  tail_extra_krw_per_hour=40000, gantry_move_krw_per_km=1000,
                  rehandle_krw_per_move=5000, vessel_delay_krw_per_hour=1000000)


def test_to_reward_config_units():
    rc = COST.to_reward_config()
    assert rc.fitted and rc.s_wait == rc.s_move == 1.0
    assert abs(rc.w_wait - 30000 / 3600 / 10000) < 1e-12       # 만원/(대·s)
    assert abs(rc.w_vessel - 1000000 / 3600 / 10000) < 1e-12   # 만원/s
    assert abs(rc.w_rehandle - 0.5) < 1e-12                    # 만원/회


def test_interval_reward_equals_negative_cost():
    """구간 보상 합 == -cost_of_metrics — reward 와 리포트 비용의 산식 일치."""
    calc = RewardCalculator(COST.to_reward_config())
    s0 = KpiSnapshot(0, 0, 0, 0, 0, 0, 0, 0.0)
    s1 = KpiSnapshot(queue_area_s=7200.0, tail_area_s=1800.0,
                     loaded_gantry_m=500.0, empty_gantry_m=500.0,
                     rehandle_count=3, completed_external=5, completed_vessel=1,
                     vessel_delay_s=600.0)
    calc.reset(s0)
    r = calc.interval_reward(s1)
    metrics = {"queue_area_h": 2.0, "tail_area_h": 0.5, "travel_km": 1.0,
               "rehandles": 3.0, "vessel_delay_min": 10.0}
    assert abs(-r - COST.cost_of_metrics(metrics)) < 1e-9


def test_cost_reward_nonpositive():
    """비용 ≥0 → 보상 ≤0 — QTable 의 '보상 항상 ≤0' 가정 유지 확인."""
    calc = RewardCalculator(COST.to_reward_config())
    s0 = KpiSnapshot(0, 0, 0, 0, 0, 0, 0, 0.0)
    s1 = KpiSnapshot(100.0, 0.0, 10.0, 5.0, 1, 0, 0, 0.0)
    calc.reset(s0)
    assert calc.interval_reward(s1) <= 0.0


def test_load_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("cost_id: X\ntruck_wait_krw_per_hour: 1\ntail_extra_krw_per_hour: 2\n"
                 "gantry_move_krw_per_km: 3\nrehandle_krw_per_move: 4\n"
                 "vessel_delay_krw_per_hour: 5\n", encoding="utf-8")
    c = CostConfig.load(p)
    assert (c.cost_id, c.rehandle_krw_per_move, c.assumed) == ("X", 4.0, True)
