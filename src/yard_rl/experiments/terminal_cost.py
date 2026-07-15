"""터미널 비용 실험 — scale fit·재채점·정적/동적 λ 비교·민감도·report (YR-038).

재시뮬 0: raw 는 record 에 박제되므로 config 변주는 make_cost 재채점(rescore)으로 후처리한다.
fit 은 TRAIN 밴드(seed 101+) baseline 만 입력(test 누출 금지). 전 항목 assumed — 탐색 전용,
가중치 확정 금지(YR-002 후 실측). 엔진 RNG 없음 → 동일 시나리오 동일 raw → fit 결정론.
"""
from __future__ import annotations

import copy
import random

from ..contract.cost import make_cost
from ..contract.schema import COST_TERMS, VESSEL_FAMILY
from ..domain.enums import InformationLevel
from ..integrated import (ReferenceDispatcher, TerminalSimulator,
                          build_integrated_profile, build_minimal_terminal_scenario,
                          record_episode)
from ..integrated.adapter import _max_vessel_risk_state
from ..integrated.cost import ASSUMED_SCALE
from ..integrated.cost_config import (LambdaMode, LambdaVesselPolicy, Provenance,
                                      ProvBasis, RewardCalculator, default_assumed_config)
from ..integrated.ledger import assert_ledger_identity, build_ledger_report
from .statistics import paired_diff

_LEVEL = InformationLevel.PRE_ADVICE
_FALLBACK_SCALE = dict(ASSUMED_SCALE)   # baseline 미발현 항의 문서화된 fallback (assumed 유지)


# ------------------------------------------------------------ 시나리오 (fit 전용)
def generate_terminal_scenarios(seeds: list[int]) -> list:
    """minimal fixture 를 seed 별 결정론 지터로 변주 (fit 전용, 범용 generator 아님).

    RNG 는 생성에만 — 트럭 도착시각·본선 완료시각·injected 시각을 소폭 이동. 엔진은 결정론 소비.
    """
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        sc = copy.deepcopy(build_minimal_terminal_scenario())
        sc.scenario_id = f"tc-{seed}"
        sc.seed = seed
        for j in sc.jobs:
            if j.is_external_truck and j.actual_block_arrival is not None:
                j.actual_block_arrival = max(30.0, j.actual_block_arrival + rng.uniform(-100.0, 250.0))
                j.actual_gate_in = max(0.0, j.actual_block_arrival - 600.0)
        for v in sc.vessels:
            if v.plan.planned_completion_s is not None:
                from dataclasses import replace
                v.plan = replace(v.plan,
                                 planned_completion_s=v.plan.planned_completion_s + rng.uniform(-400.0, 400.0))
        for i, ie in enumerate(sc.injected_events):
            from dataclasses import replace
            sc.injected_events[i] = replace(ie, time=max(1.0, ie.time + rng.uniform(-60.0, 60.0)))
        out.append(sc)
    return out


def _baseline_records(profile, sc, *, generator=None):
    sim = TerminalSimulator(profile, sc, info_level=_LEVEL)
    recs = record_episode(sim, ReferenceDispatcher(), info_level=_LEVEL,
                          episode_id=sc.scenario_id, generator=generator)
    return sim, recs


# ------------------------------------------------------------ scale fit
def fit_terminal_scale(profile, scenarios, *, generator=None) -> tuple[dict, dict]:
    """baseline per-interval raw 평균으로 scale fit. (scale, per-term report) 반환."""
    tot = {k: 0.0 for k in COST_TERMS}
    n_int = 0
    for sc in scenarios:
        sim, recs = _baseline_records(profile, sc, generator=generator)
        er = sim.cost.episode_raw()
        for k in COST_TERMS:
            tot[k] += er[k]
        n_int += len(recs)
    n = max(1, n_int)
    scale, rep = {}, {}
    for k in COST_TERMS:
        per = tot[k] / n
        fb = per <= 1e-9
        scale[k] = _FALLBACK_SCALE.get(k, 1.0) if fb else max(1e-6, per)
        rep[k] = {"episode_raw_sum": tot[k], "n_intervals": n_int,
                  "per_interval": per, "fallback": fb}
    return scale, rep


def freeze_fitted_config(profile, seeds, *, out_path=None):
    """TRAIN baseline 에서 scale 을 fit·동결한 config 반환 (+ 선택적 저장)."""
    scenarios = generate_terminal_scenarios(seeds)
    scale, rep = fit_terminal_scale(profile, scenarios)
    prov = Provenance(ProvBasis.FITTED_BASELINE, source="ReferenceDispatcher TRAIN baseline",
                      note="합성 proxy·잠정 — 실측 scale 은 YR-002",
                      fit_stat=f"mean episode_raw/interval, n_int={rep[COST_TERMS[0]]['n_intervals']}")
    cfg = default_assumed_config().with_scale(scale, prov=prov)
    from dataclasses import replace
    cfg = replace(cfg, cost_id="TERMINAL-COST-V2")
    if out_path:
        cfg.save(out_path)
    return cfg, rep


# ------------------------------------------------------------ 재채점
def rescore(records, cfg) -> list:
    """record 의 raw 를 config 로 재채점 (재시뮬 0). make_cost 재사용."""
    out = []
    for r in records:
        lam = cfg.lambda_vessel.lam(_max_vessel_risk_state(r.state))
        out.append(make_cost(interval_start_s=r.cost.interval_start_s,
                             interval_end_s=r.cost.interval_end_s, raw=r.cost.raw,
                             scale=cfg.scale(), weight=cfg.weight(),
                             lambda_vessel=lam, assumed=cfg.assumed))
    return out


# ------------------------------------------------------------ 정적/동적 λ 비교
def compare_lambda(profile, scenarios, static_cfg, dynamic_cfg) -> dict:
    """시나리오당 baseline 1회 → 두 config 재채점. paired diff(alt=dynamic)."""
    tot_s, tot_d, vd_s, vd_d = [], [], [], []
    for sc in scenarios:
        _, recs = _baseline_records(profile, sc)
        cs, cd = rescore(recs, static_cfg), rescore(recs, dynamic_cfg)
        tot_s.append(sum(c.total_normalized for c in cs))
        tot_d.append(sum(c.total_normalized for c in cd))
        vd_s.append(sum(c.contributions()["vessel_delay"] for c in cs))
        vd_d.append(sum(c.contributions()["vessel_delay"] for c in cd))
    return {"total": paired_diff(tot_s, tot_d),
            "vessel_delay_contrib": paired_diff(vd_s, vd_d)}


# ------------------------------------------------------------ 민감도 (YR-026 흡수)
def sensitivity_grid(profile, scenarios, base_cfg, *,
                     weight_terms=("vessel_delay", "truck_wait"),
                     weight_mults=(0.5, 1.0, 2.0, 4.0),
                     lambda_scales=(0.5, 1.0, 2.0)) -> dict:
    """weight 축·λ 축을 독립 변주해 총비용 반응 측정 (시나리오당 baseline 1회)."""
    traces = [self_recs for _, self_recs in (_baseline_records(profile, sc) for sc in scenarios)]

    def total_for(cfg):
        return [sum(c.total_normalized for c in rescore(recs, cfg)) for recs in traces]

    base_total = total_for(base_cfg)
    weight_axis = {}
    for term in weight_terms:
        for m in weight_mults:
            w = base_cfg.weight()
            w[term] = w[term] * m
            cfg = base_cfg.with_weight(w)
            weight_axis[f"{term}x{m}"] = paired_diff(base_total, total_for(cfg))
    lambda_axis = {}
    for m in lambda_scales:
        pol = base_cfg.lambda_vessel
        if pol.mode == LambdaMode.DYNAMIC:
            from dataclasses import replace
            bands = tuple(replace(b, lam=b.lam * m) for b in pol.bands)
            npol = replace(pol, bands=bands)
        else:
            from dataclasses import replace
            npol = replace(pol, static_value=pol.static_value * m)
        cfg = base_cfg.with_lambda(npol)
        lambda_axis[f"lam_x{m}"] = paired_diff(base_total, total_for(cfg))
    return {"weight_axis": weight_axis, "lambda_axis": lambda_axis,
            "base_total_mean": sum(base_total) / len(base_total)}


def episode_ledger_check(profile, sc) -> dict:
    """ledger 활성 완주 → 인과 항등식 검증 + report."""
    sim = TerminalSimulator(profile, sc, info_level=_LEVEL, enable_cost_ledger=True)
    record_episode(sim, ReferenceDispatcher(), info_level=_LEVEL, episode_id=sc.scenario_id)
    assert_ledger_identity(sim.cost)
    return build_ledger_report(sim.cost)


# ------------------------------------------------------------ report
def build_cost_report(*, train_seeds=(101, 102, 103, 104, 105),
                      val_seeds=(201, 202, 203, 204, 205)) -> dict:
    """cost identity·scale fit·정적/동적 λ·민감도 종합 (결정론 dict)."""
    profile = build_integrated_profile()
    ledger_rep = episode_ledger_check(profile, build_minimal_terminal_scenario())
    _, fit_rep = freeze_fitted_config(profile, list(train_seeds))
    val = generate_terminal_scenarios(list(val_seeds))
    dyn = default_assumed_config()
    static = dyn.with_lambda(LambdaVesselPolicy(
        LambdaMode.STATIC, Provenance(ProvBasis.ASSUMED, "static A/B"), static_value=1.0))
    lam_cmp = compare_lambda(profile, val, static, dyn)
    sens = sensitivity_grid(profile, val, dyn)
    return {"report_id": "terminal-cost-report-v1", "assumed": True,
            "ledger_identity": ledger_rep, "scale_fit": fit_rep,
            "lambda_static_vs_dynamic": lam_cmp, "sensitivity": sens,
            "guardrail_note": "전 항목 assumed·synthetic — 탐색 전용, 가중치 확정 금지(YR-002 후 실측)"}
