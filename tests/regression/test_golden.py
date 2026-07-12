"""회귀 golden 테스트 (05 §1.4) — 고정 seed·시나리오의 결과 고정.

의존성·모델·설정 변경 시 이 값이 달라지면 원인을 검토한 뒤에만 갱신한다.
(부동소수점 tolerance 1e-3, 작업 선택순서 변화는 별도 검토 원칙)
golden 채집: 2026-07-12, 리뷰 확정 9건 수정 직후.
갱신 사유: 생성기 ETA 필드 추가(RNG 스트림 변화)·동시각 이벤트 소진 후
의사결정·적분 end 절단·검열 표본 — 전부 리뷰 확정건 반영으로 검토됨.
"""
from yard_rl.domain.enums import InformationLevel, PriorityRule
from yard_rl.envs.yard_env import YardEnv
from yard_rl.experiments.runner import run_episode
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import generate
from yard_rl.policies.baselines import FixedRulePolicy

GOLDEN = {
    "queue_area_s": 26060.112,
    "tail_area_s": 0.0,
    "loaded_m": 65.0,
    "empty_m": 3464.5,
    "rehandles": 37,
    "completed_external": 100,
    "completed_vessel": 8,
    "n_events": 325,
    "n_decisions": 108,
    "mean_wait_min": 4.3434,
}


def test_fifo_seed42_golden():
    p = load_profile("configs/terminals/poc_single_crane.yaml")
    sc = generate(p, seed=42)
    env = YardEnv(p, info_level=InformationLevel.BLOCK_ARRIVAL, check_invariants=True)
    r = run_episode(FixedRulePolicy(PriorityRule.FIFO), env, sc)
    k = env.sim.kpis
    assert abs(k.queue_area_s - GOLDEN["queue_area_s"]) < 1e-3
    assert abs(k.tail_area_s - GOLDEN["tail_area_s"]) < 1e-3
    assert abs(k.loaded_gantry_m - GOLDEN["loaded_m"]) < 1e-3
    assert abs(k.empty_gantry_m - GOLDEN["empty_m"]) < 1e-3
    assert k.rehandle_count == GOLDEN["rehandles"]
    assert k.completed_external == GOLDEN["completed_external"]
    assert k.completed_vessel == GOLDEN["completed_vessel"]
    assert len(env.sim.event_log) == GOLDEN["n_events"]
    assert env.n_steps == GOLDEN["n_decisions"]
    assert abs(r.metrics["mean_wait_min"] - GOLDEN["mean_wait_min"]) < 1e-3
    # 정합성 항등식: queue-area == Σ 대기표본 (검열 포함) — 적분 절단과 함께 성립
    assert abs(k.queue_area_s - sum(k.wait_samples_s)) < 1e-6
