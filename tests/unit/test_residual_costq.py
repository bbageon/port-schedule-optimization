"""YR-030-c — 잔차 Cost-Q 계약 테스트 (사전등록 §1·§5~§8 대응)."""
import math

import pytest

from yard_rl.envs.direct_job_env import DirectJobBucketConfig, DirectJobEnv, SLAMode
from yard_rl.experiments.residual_costq import (ResidualConfig,
                                                quick_residual_config,
                                                run_residual_experiment)
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams, generate
from yard_rl.policies.cost_q import CostQConfig
from yard_rl.policies.direct_baselines import DirectJobRulePolicy, DirectRule
from yard_rl.policies.residual_cost_q import ResidualCostQAgent

PROFILE = "configs/terminals/hjnc_armg.yaml"


def _scenario(profile, seed=110_001, n=10):
    return generate(profile, seed, GenParams(n_external=n, n_vessel=0,
                                             drain_window_s=86_400.0))


class _C:
    """CandidateProtocol 스텁 — 잔차 수식 검증용."""

    def __init__(self, jid, prior, future=(0, 1, 1, 2, 0), wait=0.0, service=100.0):
        self.job_id = jid
        self.feature = ("YARD_TO_TRUCK", 1, 1, 1, 0)
        self.future_feature = future
        self.wait_s = wait
        self.estimated_service_s = service
        self.block_entry_s = 0.0
        self.prior_cost = prior


def test_untrained_policy_equals_greedy_exactly():
    """§5: ΔQ=0 전부 → 정책 ≡ IMMEDIATE_COST_GREEDY (에피소드 단위 확인)."""
    profile = load_profile(PROFILE)
    greedy = DirectJobRulePolicy(DirectRule.IMMEDIATE_COST_GREEDY)
    for key_mode in ("state_job", "future"):
        agent = ResidualCostQAgent(CostQConfig(learning_rate_power=1.0, gamma=0.95),
                                   seed=0, key_mode=key_mode)
        env_a = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                             expected_n_config=10)
        env_b = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                             expected_n_config=10)
        sa, ia = env_a.reset(_scenario(profile))
        sb, ib = env_b.reset(_scenario(profile))
        while sa is not None:
            pick_a = agent.act(sa, ia.candidates)
            pick_b = greedy.act(sb, ib.candidates)
            assert pick_a.job_id == pick_b.job_id
            sa, _c, _d, ia = env_a.step(pick_a)
            sb, _c2, _d2, ib = env_b.step(pick_b)
        assert sb is None  # 두 에피소드가 같은 길이로 종료


def test_update_stores_residual_not_total():
    """§6: 테이블에는 Y−G 만 저장, Q_total = G + ΔQ. 음수 보정 허용."""
    agent = ResidualCostQAgent(CostQConfig(learning_rate_power=1.0, gamma=1.0),
                               seed=0, key_mode="future")
    a = _C("A", prior=3.0)
    nxt = [_C("N1", prior=2.0, future=(1, 1, 1, 2, 1))]
    # Y = c + γ·min(G'+Δ') = 1.0 + 2.0 = 3.0 → Y_Δ = 3.0 − 3.0 = 0.0
    agent.update("s", a, 1.0, "s2", nxt, done=False)
    assert agent.table.value(agent.key("s", a)) == pytest.approx(0.0)
    # 종료 스텝: Y = c = 0.5 → Y_Δ = 0.5 − 3.0 = −2.5 (α₂=1/2 평균화)
    agent.update("s", a, 0.5, None, [], done=True)
    assert agent.table.value(agent.key("s", a)) == pytest.approx(-1.25)
    assert agent.q_total("s", a) == pytest.approx(3.0 - 1.25)


def test_learned_negative_residual_flips_greedy_order():
    """§8: A(181s)+30 vs B(227s)−40 → B 선택. G 차이는 유지된 채 반전."""
    agent = ResidualCostQAgent(CostQConfig(learning_rate_power=1.0, gamma=1.0),
                               seed=0, key_mode="future")
    a = _C("A", prior=181 / 60, future=(0, 2, 2, 2, 2), service=181.0)
    b = _C("B", prior=227 / 60, future=(1, 2, 2, 1, 0), service=227.0)
    assert agent.act("s", [a, b]).job_id == "A"  # 학습 전 = greedy
    agent.table.update(agent.key("s", a), 30 / 60, 1.0)
    agent.table.update(agent.key("s", b), -40 / 60, 1.0)
    assert agent.act("s", [a, b]).job_id == "B"
    assert agent.q_total("s", a) == pytest.approx((181 + 30) / 60)
    assert agent.q_total("s", b) == pytest.approx((227 - 40) / 60)


def test_future_feature_attached_and_bounded():
    """§2: v1_final env 가 future_situation 을 부착 — 단계 범위·'없음' 코드."""
    profile = load_profile(PROFILE)
    env = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                       expected_n_config=10)
    state, info = env.reset(_scenario(profile))
    seen_solo = False
    while state is not None:
        raw = info.raw_global
        for c in info.candidates:
            f = c.future_feature
            assert len(f) == 5
            zone, jobs_lv, work_lv, mix_lv, near_lv = f
            assert 0 <= zone <= 3 and 0 <= jobs_lv <= 3 and 0 <= work_lv <= 3
            assert 0 <= mix_lv <= 3 and 0 <= near_lv <= 3
            if raw.waiting_truck_count == 1:   # 혼자 대기 → 잔여 없음 코드
                assert jobs_lv == 0
            if len(info.candidates) == 1:
                assert (work_lv, mix_lv, near_lv) == (0, 0, 3)
                seen_solo = True
        state, _c, _d, info = env.step(info.candidates[0])
    assert seen_solo  # drain 종반 단독 후보 상황이 실제로 관측됨


def test_bucket_fit_future_edges_and_backcompat(tmp_path):
    cfg = DirectJobBucketConfig.fit(
        queue_lengths=[1, 3, 6], service_times_s=[100, 200, 400, 900],
        jobs_left_counts=[0, 1, 2, 3, 5, 8], work_left_totals_s=[0, 300, 900, 2400],
        sla_s=1800.0)
    assert len(cfg.jobs_left) == 2 and len(cfg.work_left_s) == 2
    assert cfg.short_service_s[0] == 400  # train service 중앙값
    path = tmp_path / "b.json"
    cfg.save(path)
    assert DirectJobBucketConfig.load(path).jobs_left == cfg.jobs_left
    # 구버전(신규 필드 없는) 저장본도 로드 가능 — 기본값 유지
    legacy = DirectJobBucketConfig()
    legacy.save(path)
    assert DirectJobBucketConfig.load(path).short_service_s == (300.0,)


def test_prior_replacement_mode_rejected():
    with pytest.raises(ValueError):
        ResidualCostQAgent(CostQConfig(use_greedy_prior=True), seed=0)


def test_seed_band_guard():
    with pytest.raises(ValueError):
        ResidualConfig(train_seed0=70_000)  # YR-030-b band 재사용 금지


def test_agent_save_load_roundtrip(tmp_path):
    agent = ResidualCostQAgent(CostQConfig(learning_rate_power=1.0, gamma=0.95),
                               seed=7, key_mode="future")
    a = _C("A", prior=1.0)
    agent.update("s", a, 0.2, None, [], done=True)
    path = tmp_path / "agent.json"
    agent.save(path)
    loaded = ResidualCostQAgent.load(path)
    assert loaded.key_mode == "future"
    assert loaded.q_total("s", a) == pytest.approx(agent.q_total("s", a))


def test_quick_run_end_to_end(tmp_path):
    """quick 설정 전체 파이프라인 — 산출물·판정 필드 존재."""
    report = run_residual_experiment(
        out_dir=str(tmp_path / "out"), cfg=quick_residual_config(),
        progress=lambda _m: None)
    text = report.read_text(encoding="utf-8")
    assert "ResidualCostQ[state_job]" in text
    assert "ResidualCostQ[future]" in text
    assert "guardrail" in text  # 사전등록 §4 — 동시 보고
    import json
    results = json.loads(
        (tmp_path / "out" / "residual_results.json").read_text(encoding="utf-8"))
    for entry in results["paired"].values():
        g = entry["guardrails"]
        assert {"p95_within_5pct", "completion_all_100pct", "max_backlog",
                "invariants_all_ok"} <= set(g)
    assert (tmp_path / "out" / "direct_buckets.json").exists()
