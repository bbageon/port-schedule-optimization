"""터미널 비용 config·RewardCalculator — 계약 동치·provenance·정적/동적 λ (YR-038)."""
import pytest
from dataclasses import replace

from yard_rl.contract.schema import COST_TERMS, VESSEL_FAMILY
from yard_rl.contract.validate import validate_cost
from yard_rl.integrated.cost import ASSUMED_SCALE, ASSUMED_WEIGHT, assumed_lambda_vessel
from yard_rl.integrated.cost_config import (CostConfigError, LambdaMode, LambdaVesselPolicy,
                                           Provenance, ProvBasis, RewardCalculator,
                                           RiskBand, TerminalCostConfig, default_assumed_config)

CFG = default_assumed_config()


def test_config_matches_legacy_assumed():
    assert CFG.scale() == ASSUMED_SCALE
    assert CFG.weight() == ASSUMED_WEIGHT
    for r in (0.0, 0.19, 0.2, 0.4, 0.6, 0.79, 0.8, 1.0):
        assert CFG.lambda_vessel.lam(r) == assumed_lambda_vessel(r)


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    CFG.save(p)
    loaded = TerminalCostConfig.load(p)
    assert loaded.scale() == CFG.scale()
    assert loaded.weight() == CFG.weight()
    assert all(loaded.lambda_vessel.lam(r) == CFG.lambda_vessel.lam(r)
               for r in (0.0, 0.3, 0.5, 0.7, 0.9))


def test_reward_calculator_identity():
    """cost_for 출력이 계약 validate_cost(항등식·no double-count) 통과."""
    rc = RewardCalculator(CFG)
    raw = {t: 100.0 + i for i, t in enumerate(COST_TERMS)}
    c = rc.cost_for(interval_start_s=0.0, interval_end_s=300.0, raw=raw, risk_max=0.5)
    validate_cost(c)
    assert abs(c.reward + c.total_normalized) < 1e-6


def test_lambda_only_vessel_family():
    """정적 vs 동적 λ 는 VESSEL_FAMILY 4항 기여만 다르고 나머지 9항 동일."""
    raw = {t: 100.0 for t in COST_TERMS}
    static = CFG.with_lambda(LambdaVesselPolicy(LambdaMode.STATIC,
                                                Provenance(ProvBasis.ASSUMED), static_value=1.0))
    cs = RewardCalculator(static).cost_for(interval_start_s=0, interval_end_s=1, raw=raw, risk_max=0.9)
    cd = RewardCalculator(CFG).cost_for(interval_start_s=0, interval_end_s=1, raw=raw, risk_max=0.9)
    cons_s, cons_d = cs.contributions(), cd.contributions()
    for t in COST_TERMS:
        if t in VESSEL_FAMILY:
            assert cons_d[t] == pytest.approx(6.0 * cons_s[t])   # risk 0.9 → λ=6.0
        else:
            assert cons_d[t] == pytest.approx(cons_s[t])


def test_bad_config_rejected():
    a = Provenance(ProvBasis.ASSUMED)
    from yard_rl.integrated.cost_config import TermCost
    # scale 0
    bad = replace(CFG, terms={**CFG.terms,
                              "truck_wait": TermCost("truck_wait", 0.0, 1.0, a, a)})
    with pytest.raises(CostConfigError, match="scale"):
        bad.validate()
    # 항 누락
    miss = replace(CFG, terms={k: v for k, v in CFG.terms.items() if k != "imbalance"})
    with pytest.raises(CostConfigError, match="13항"):
        miss.validate()
    # 밴드 역순 (오름차순 → 오류)
    pol = LambdaVesselPolicy(LambdaMode.DYNAMIC, a,
                             bands=(RiskBand(0.2, 1.5), RiskBand(0.8, 6.0)))
    with pytest.raises(CostConfigError, match="내림차순"):
        replace(CFG, lambda_vessel=pol).validate()
    # schema 불일치
    with pytest.raises(CostConfigError, match="schema_version"):
        replace(CFG, schema_version="itc-v999").validate()


def test_provenance_fit_freeze():
    """with_scale 는 scale_fitted=True + FITTED provenance 박제."""
    scale = {t: 42.0 for t in COST_TERMS}
    prov = Provenance(ProvBasis.FITTED_BASELINE, source="baseline", fit_stat="mean/interval")
    fitted = CFG.with_scale(scale, prov=prov)
    assert fitted.scale_fitted is True
    assert fitted.terms["truck_wait"].scale_prov.basis == ProvBasis.FITTED_BASELINE
    assert fitted.scale() == scale
