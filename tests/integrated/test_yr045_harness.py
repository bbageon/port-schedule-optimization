"""YR-045 하네스 계약 — arm 기제(ETA_NO_PRE 차단·NO_ETA 제거)·전략적 WAIT 금지·러너 E2E.

사전등록(2026-07-16) 집행 코드의 기제를 고정한다. 실험 결과가 아니라 **기제**의 테스트다.
seed 310000 대역은 실험에서 폐기됐지만 기제 검증용 테스트에는 사용 가능하다 (소각 대상은
실험 판정이지 코드 검증이 아님).
"""
from yard_rl.contract.schema import CandidateKind
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (CandidateGenerator, JointRolloutGreedy, ResolverPolicy,
                                ServiceFirstSPTPreference, TerminalSimulator,
                                build_integrated_profile, run_joint_episode)
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.dqn_learner import run_episode
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.scenario_gen import generate_terminal_scenario
from yard_rl.experiments.yr045_locked_rerun import (Yr045Config, generator_for_arm,
                                                    quick_yr045_config, run_yr045,
                                                    scenario_for_arm)

PROF = build_integrated_profile()
PA = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
SEED = 310000


def _kind_counts(sim, gen, max_decisions=200):
    from yard_rl.integrated import BaselinePreference, CentralResolver
    r = CentralResolver(BaselinePreference())
    counts: dict[str, int] = {}
    for _ in range(max_decisions):
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: gen.generate(sim, c, PA) for c in dp.crane_ids}
        for g in gen_by.values():
            for gc in g.items:
                counts[gc.kind.value] = counts.get(gc.kind.value, 0) + 1
        r.apply(sim, r.resolve(sim, dp, gen_by), gen_by)
    return counts


def test_eta_no_pre_arm_blocks_pre_but_keeps_reposition():
    """ETA_NO_PRE: ETA 는 보이되 선제 재조작만 차단 — 위치선점 경로 순효과 분리 계약."""
    sc = generate_terminal_scenario(PROF, SEED)
    blocked = _kind_counts(TerminalSimulator(PROF, sc, info_level=PA),
                           generator_for_arm("ETA_NO_PRE"))
    full = _kind_counts(TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED),
                                          info_level=PA), generator_for_arm("FULL"))
    assert full.get("PRE_REHANDLE", 0) > 0
    assert blocked.get("PRE_REHANDLE", 0) == 0, "차단 arm 에서 PRE 후보 발행 — arm 오염"
    assert blocked.get("REPOSITION", 0) > 0


def test_no_eta_arm_strips_eta_and_both_paths():
    sc = scenario_for_arm(PROF, SEED, None, "NO_ETA")
    assert all(j.provided_eta is None for j in sc.jobs)
    assert sc.meta["eta_error_s"] is None and sc.meta["arm"] == "NO_ETA"
    counts = _kind_counts(TerminalSimulator(PROF, sc, info_level=PA),
                          generator_for_arm("NO_ETA"))
    assert counts.get("PRE_REHANDLE", 0) == 0


def test_forbid_strategic_wait_rollout_reduces_wait_and_completes():
    """금지 모드: 실작업 조합이 공동 실행가능하면 WAIT 조합 배제 (구조적 WAIT 은 보존).

    YR-052 실측에서 효과가 컸던 seed 310003 — 완주·건전성 유지 + WAIT 선택 감소.
    """
    rows = {}
    for mode in ("allow", "forbid"):
        sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, 310003),
                                info_level=PA)
        pol = JointRolloutGreedy(RC, horizon_s=600.0,
                                 forbid_strategic_wait=(mode == "forbid"))
        rows[mode] = run_joint_episode(sim, pol, RC, level=PA)
    assert rows["forbid"]["completion_rate"] == 1.0
    a = rows["allow"]["action_mix"]["counts"].get("WAIT", 0)
    f = rows["forbid"]["action_mix"]["counts"].get("WAIT", 0)
    assert f <= a, f"금지 모드가 WAIT 를 늘림 ({a}->{f})"


def test_forbid_strategic_wait_rl_path_keeps_structural_wait():
    """QPreference 경로: WAIT 점수 +∞ 강제 — 완주 유지, WAIT 는 구조적 경로만."""
    res = {}
    for mode in (False, True):
        sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED),
                                info_level=PA)
        r = run_episode(sim, level=PA, preference=QPreference(),
                        forbid_strategic_wait=mode)
        res[mode] = r
        assert r.completion_rate == 1.0 and r.backlog == 0
        for key in ("action_counts", "cand_listed", "term_contrib", "rehandles",
                    "sts_wait_s", "empty_travel_m"):
            assert key in r.extras
    assert (res[True].extras["action_counts"].get("WAIT", 0)
            <= res[False].extras["action_counts"].get("WAIT", 0))


def test_run_episode_honors_arm_generator():
    """run_episode 가 generator 를 capture 에 전달하는지 — 미전달이면 기본 생성기가
    쓰여 ETA_NO_PRE arm 이 조용히 FULL 이 된다 (YR-045 locked run 에서 실측 발견·정정)."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED), info_level=PA)
    r = run_episode(sim, level=PA, preference=QPreference(),
                    generator=generator_for_arm("ETA_NO_PRE"))
    assert r.extras["cand_listed"].get("PRE_REHANDLE", 0) == 0, \
        "차단 generator 가 무시됨 — arm 오염 재발"
    sim2 = TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED), info_level=PA)
    r2 = run_episode(sim2, level=PA, preference=QPreference(),
                     generator=generator_for_arm("FULL"))
    assert r2.extras["cand_listed"].get("PRE_REHANDLE", 0) > 0


def test_config_guards_seed_hygiene():
    import pytest
    with pytest.raises(ValueError):
        Yr045Config(train_seed0=300_000)          # 소각 대역 재사용 금지
    with pytest.raises(ValueError):
        Yr045Config(validation_seed0=400_010)     # 대역 중첩 금지
    cfg = Yr045Config()
    assert cfg.train_seeds[0] == 400_000 and cfg.test_seeds[-1] == 420_059


def test_quick_e2e_produces_gates_and_report(tmp_path):
    """quick 모드 전 구간 — precheck→fit→학습→locked→게이트·리포트 (사전등록 §7 산출물)."""
    report = run_yr045(out_dir=str(tmp_path), cfg=quick_yr045_config(),
                       progress=lambda s: None)
    assert report.exists()
    import json
    payload = json.loads((tmp_path / "yr045_results.json").read_text(encoding="utf-8"))
    assert payload["analysis"]["gates"], "게이트 판정 없음"
    assert payload["analysis"]["arm_contributions"]
    assert (tmp_path / "phase_a_precheck.json").exists()
    assert (tmp_path / "phase_b_scale.json").exists()
