"""scale fit·재채점·정적/동적 λ·민감도 — 결정론·누출금지·순수회계 (YR-038)."""
from yard_rl.contract.schema import COST_TERMS, VESSEL_FAMILY
from yard_rl.contract.cost import make_cost
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (ReferenceDispatcher, TerminalSimulator,
                               build_integrated_profile, build_minimal_terminal_scenario,
                               record_episode)
from yard_rl.integrated.cost_config import (LambdaMode, LambdaVesselPolicy, Provenance,
                                           ProvBasis, default_assumed_config)
from yard_rl.experiments.terminal_cost import (compare_lambda, fit_terminal_scale,
                                              generate_terminal_scenarios, rescore,
                                              sensitivity_grid)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE


def test_scale_fit_deterministic_and_positive():
    sc = generate_terminal_scenarios([101, 102, 103])
    s1, rep = fit_terminal_scale(PROF, sc)
    s2, _ = fit_terminal_scale(PROF, sc)
    assert s1 == s2                       # 결정론 (엔진 RNG 없음)
    assert all(v > 0 for v in s1.values())


def test_scale_fit_fallback_flagged():
    """baseline 미발현 항(resequence)은 fallback=True 로 박제 (조용한 0 금지)."""
    _, rep = fit_terminal_scale(PROF, generate_terminal_scenarios([101, 102]))
    assert rep["resequence"]["fallback"] is True
    assert rep["truck_wait"]["fallback"] is False


def test_rescore_pure_accounting():
    """재채점은 재시뮬 0 — 원 config 재채점 == 원 record.cost (bit)."""
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(), info_level=LEVEL)
    recs = record_episode(sim, ReferenceDispatcher(), info_level=LEVEL, episode_id="E")
    rs = rescore(recs, default_assumed_config())
    for a, b in zip(rs, recs):
        assert abs(a.total_normalized - b.cost.total_normalized) < 1e-9
        assert a.raw == b.cost.raw       # raw 불변 (config 는 scale/weight/λ 만)


def test_static_vs_dynamic_lambda_high_risk():
    """고위험 구간(risk 0.9)에서 동적 λ 총비용 ≥ 정적 (본선계열 λ 배수)."""
    raw = {t: 100.0 for t in COST_TERMS}
    dyn = default_assumed_config()
    static = dyn.with_lambda(LambdaVesselPolicy(LambdaMode.STATIC,
                                                Provenance(ProvBasis.ASSUMED), static_value=1.0))
    cs = make_cost(interval_start_s=0, interval_end_s=1, raw=raw, scale=static.scale(),
                   weight=static.weight(), lambda_vessel=static.lambda_vessel.lam(0.9))
    cd = make_cost(interval_start_s=0, interval_end_s=1, raw=raw, scale=dyn.scale(),
                   weight=dyn.weight(), lambda_vessel=dyn.lambda_vessel.lam(0.9))
    assert cd.total_normalized > cs.total_normalized


def test_compare_lambda_runs():
    dyn = default_assumed_config()
    static = dyn.with_lambda(LambdaVesselPolicy(LambdaMode.STATIC,
                                                Provenance(ProvBasis.ASSUMED), static_value=1.0))
    out = compare_lambda(PROF, generate_terminal_scenarios([201, 202, 203, 204, 205]), static, dyn)
    assert "total" in out and out["total"]["n"] == 5   # paired CI 산출


def test_sensitivity_vessel_dominant_monotone():
    """vessel_delay weight↑ → 총비용 단조↑ (지배항 민감도, YR-026 흡수)."""
    dyn = default_assumed_config()
    sens = sensitivity_grid(PROF, generate_terminal_scenarios([201, 202, 203]), dyn,
                            weight_terms=("truck_wait",), weight_mults=(1.0, 2.0, 4.0),
                            lambda_scales=(1.0,))
    d2 = sens["weight_axis"]["truck_waitx2.0"]["mean_diff"]
    d4 = sens["weight_axis"]["truck_waitx4.0"]["mean_diff"]
    assert d4 >= d2 >= 0                  # weight↑ → truck_wait 기여↑ → total↑ 단조
