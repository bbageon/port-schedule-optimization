"""정규화 터미널 비용 config·RewardCalculator — 하드코딩 scale/weight/λ 형식화 (YR-038).

계약(contract/cost.py make_cost·CostBreakdown·COST_TERMS·VESSEL_FAMILY) **무변경**. config 는
scale/weight/λ 실수치만 주입한다. 전 항목 assumed(PoC) — 실측 확정은 YR-002, baseline-fit 은 YR-038.
default_assumed_config() 는 현 ASSUMED_SCALE/WEIGHT·assumed_lambda_vessel 을 바이트 재현(golden 불변).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from ..contract.cost import CostBreakdown, make_cost
from ..contract.schema import COST_TERMS, SCHEMA_VERSION
from .cost import ASSUMED_SCALE, ASSUMED_WEIGHT


class ProvBasis(str, Enum):
    ASSUMED = "assumed"                # PoC 임시값
    REGULATION = "regulation"          # 제도 anchor (안전운임 등)
    FITTED_BASELINE = "fitted_baseline"  # baseline 통계 fit
    MEASURED = "measured"              # 실측


@dataclass(frozen=True)
class Provenance:
    basis: ProvBasis
    source: str = ""
    note: str = ""
    to_be_validated: bool = True
    fit_stat: str = ""                 # FITTED 전용 통계량 설명


@dataclass(frozen=True)
class TermCost:
    name: str
    scale: float
    weight: float
    scale_prov: Provenance
    weight_prov: Provenance
    unit: str = ""


class LambdaMode(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class RiskBand:
    risk_ge: float
    lam: float
    label: str = ""


@dataclass(frozen=True)
class LambdaVesselPolicy:
    """§10.6 동적 본선계수 — 정적(상수) 또는 위험도 밴드(risk_ge 내림차순)."""

    mode: LambdaMode
    prov: Provenance
    static_value: float = 1.0
    bands: tuple[RiskBand, ...] = ()

    def lam(self, risk_max: float) -> float:
        if self.mode == LambdaMode.STATIC:
            return self.static_value
        for b in self.bands:            # 내림차순 — 첫 충족 밴드 (경계 >= 포함)
            if risk_max >= b.risk_ge:
                return b.lam
        return 1.0


class CostConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TerminalCostConfig:
    cost_id: str
    schema_version: str
    assumed: bool
    terms: dict[str, TermCost]         # keys == set(COST_TERMS)
    lambda_vessel: LambdaVesselPolicy
    scale_fitted: bool = False
    provenance_note: str = ""

    def scale(self) -> dict[str, float]:
        return {t: self.terms[t].scale for t in COST_TERMS}

    def weight(self) -> dict[str, float]:
        return {t: self.terms[t].weight for t in COST_TERMS}

    def validate(self) -> "TerminalCostConfig":
        if self.schema_version != SCHEMA_VERSION:
            raise CostConfigError(f"schema_version {self.schema_version} != {SCHEMA_VERSION}")
        if set(self.terms) != set(COST_TERMS):
            raise CostConfigError("terms 키가 13항(COST_TERMS)과 불일치")
        for t in COST_TERMS:
            tc = self.terms[t]
            if tc.scale <= 0:
                raise CostConfigError(f"{t}: scale 은 양수여야 함 (make_cost ZERO_SCALE 선제)")
            if tc.weight < 0:
                raise CostConfigError(f"{t}: weight 음수 불가")
        pol = self.lambda_vessel
        if pol.mode == LambdaMode.DYNAMIC:
            ge = [b.risk_ge for b in pol.bands]
            if ge != sorted(ge, reverse=True):
                raise CostConfigError("DYNAMIC λ 밴드는 risk_ge 내림차순이어야 함")
        elif pol.static_value < 0:
            raise CostConfigError("static λ 음수 불가")
        return self

    def with_scale(self, scale: dict[str, float], *, prov: Provenance) -> "TerminalCostConfig":
        terms = {t: replace(self.terms[t], scale=float(scale[t]), scale_prov=prov) for t in COST_TERMS}
        return replace(self, terms=terms, scale_fitted=True).validate()

    def with_weight(self, weight: dict[str, float]) -> "TerminalCostConfig":
        terms = {t: replace(self.terms[t], weight=float(weight[t])) for t in COST_TERMS}
        return replace(self, terms=terms).validate()

    def with_lambda(self, pol: LambdaVesselPolicy) -> "TerminalCostConfig":
        return replace(self, lambda_vessel=pol).validate()

    @classmethod
    def load(cls, path) -> "TerminalCostConfig":
        import yaml
        from pathlib import Path
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        terms = {}
        for t in COST_TERMS:
            d = raw["terms"][t]
            sp = Provenance(ProvBasis(d.get("scale_basis", "assumed")),
                            d.get("scale_source", ""), d.get("scale_note", ""),
                            bool(d.get("scale_tbv", True)), d.get("scale_fit_stat", ""))
            wp = Provenance(ProvBasis(d.get("weight_basis", "assumed")),
                            d.get("weight_source", ""), d.get("weight_note", ""))
            terms[t] = TermCost(t, float(d["scale"]), float(d["weight"]), sp, wp, d.get("unit", ""))
        lv = raw["lambda_vessel"]
        pol = LambdaVesselPolicy(
            LambdaMode(lv["mode"]),
            Provenance(ProvBasis(lv.get("basis", "assumed")), lv.get("source", ""), lv.get("note", "")),
            static_value=float(lv.get("static_value", 1.0)),
            bands=tuple(RiskBand(float(b["risk_ge"]), float(b["lam"]), b.get("label", ""))
                        for b in lv.get("bands", [])))
        return cls(cost_id=raw["cost_id"], schema_version=str(raw["schema_version"]),
                   assumed=bool(raw.get("assumed", True)), terms=terms, lambda_vessel=pol,
                   scale_fitted=bool(raw.get("scale_fitted", False)),
                   provenance_note=raw.get("provenance_note", "")).validate()

    def to_yaml_dict(self) -> dict:
        return {
            "cost_id": self.cost_id, "schema_version": self.schema_version,
            "assumed": self.assumed, "scale_fitted": self.scale_fitted,
            "provenance_note": self.provenance_note,
            "terms": {t: {
                "scale": self.terms[t].scale, "weight": self.terms[t].weight,
                "unit": self.terms[t].unit, "scale_basis": self.terms[t].scale_prov.basis.value,
                "scale_source": self.terms[t].scale_prov.source,
                "scale_note": self.terms[t].scale_prov.note,
                "scale_tbv": self.terms[t].scale_prov.to_be_validated,
                "scale_fit_stat": self.terms[t].scale_prov.fit_stat,
                "weight_basis": self.terms[t].weight_prov.basis.value} for t in COST_TERMS},
            "lambda_vessel": {
                "mode": self.lambda_vessel.mode.value, "static_value": self.lambda_vessel.static_value,
                "basis": self.lambda_vessel.prov.basis.value, "source": self.lambda_vessel.prov.source,
                "note": self.lambda_vessel.prov.note,
                "bands": [{"risk_ge": b.risk_ge, "lam": b.lam, "label": b.label}
                          for b in self.lambda_vessel.bands]},
        }

    def save(self, path) -> None:
        import yaml
        from pathlib import Path
        Path(path).write_text(
            yaml.safe_dump(self.to_yaml_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8")


@dataclass(frozen=True)
class RewardCalculator:
    """config → CostBreakdown (make_cost 재사용, total/reward 항등식 자동)."""

    config: TerminalCostConfig

    def cost_for(self, *, interval_start_s: float, interval_end_s: float,
                 raw: dict[str, float], risk_max: float) -> CostBreakdown:
        return make_cost(
            interval_start_s=interval_start_s, interval_end_s=interval_end_s, raw=raw,
            scale=self.config.scale(), weight=self.config.weight(),
            lambda_vessel=self.config.lambda_vessel.lam(risk_max), assumed=self.config.assumed)

    @classmethod
    def assumed_default(cls) -> "RewardCalculator":
        return cls(default_assumed_config())

    @classmethod
    def numeraire_v1(cls) -> "RewardCalculator":
        return cls(numeraire_v1_config())


def default_lambda_bands() -> tuple[RiskBand, ...]:
    """assumed_lambda_vessel 과 값 동치 (§10.6 초기후보)."""
    return (RiskBand(0.8, 6.0, "실제지연"), RiskBand(0.6, 4.0, "위험"),
            RiskBand(0.4, 2.5, "경계"), RiskBand(0.2, 1.5, "주의"), RiskBand(0.0, 1.0, "정상"))


def neutral_lambda_config() -> TerminalCostConfig:
    """YR-043 정정 트랙 중립 config — λ_vessel=1.0 정적 (동적 위험 비활성).

    본선 악화 축은 **별도 실험으로 분리**(사용자 결정 2026-07-15): 스트레스 시나리오 제외·본선 KPI 는
    기록만(고위험 성능 주장 금지). 동적 밴드 설계는 YR-041. λ=2.5 잠정 결정은 본 트랙에서 대체.
    """
    return default_assumed_config().with_lambda(LambdaVesselPolicy(
        LambdaMode.STATIC,
        Provenance(ProvBasis.ASSUMED, "YR-043 정정 트랙",
                   "본선 악화 축 분리 — 동적 위험 비활성·본선 KPI 기록만. 밴드 설계는 YR-041"),
        static_value=1.0))


# ---------------------------------------------------------------- 기준재 (YR-080 단계4)
# scale = **순수 단위환산**(스케일 함정 제거): 시간항 3600s=1h · travel 7200m=크레인 1h
# (POC gantry 2.0 m/s 기준) · count 1. weight = **경제 비율 ρ** (기준재: 트럭대기 1h = 1.0).
# 문헌 근거: 상대계수(numeraire)가 지배 관행 — 2026-07-21 본선전략 v5 §2 (Imai 2003·
# Meisel&Bierwirth 2009·Ng 1999 양의 선형변환 불변). 결정 비율 vs 학습 스케일 분리 —
# 학습 표적 축소는 make_cost 밖 learner cost_scale 로만 (여기 금지).
NUMERAIRE_SCALE: dict[str, float] = {
    "truck_wait": 3600.0, "long_wait": 3600.0, "crane_travel": 7200.0, "empty_travel": 7200.0,
    "rehandle": 1.0, "sts_wait": 3600.0, "transfer_wait": 3600.0, "vessel_delay": 3600.0,
    "depart_delay": 3600.0, "lane_cong": 100.0, "interference": 100.0,
    "resequence": 10.0, "imbalance": 1.0}
NUMERAIRE_WEIGHT: dict[str, float] = {
    "truck_wait": 1.0,       # 기준재 (고정)
    "long_wait": 1.0,        # SLA 초과 대기 = 가산 페널티 (tail 이 queue 에 더해져 2배 아픔)
    "vessel_delay": 33.0,    # ρ_vessel (assumed) — 접안료요율÷트럭시간비용 근사, 민감도 ×⅓~3 필수
    "rehandle": 0.05,        # ρ_rehandle = 재조작 1회 ≈ 크레인 0.05h(180s) × ρ_crane
    "crane_travel": 1.0, "empty_travel": 1.0,   # ρ_crane = 1.0 (assumed — 장비 1h ≈ 트럭대기 1h)
    "depart_delay": 0.0,     # 선석초과 단일 정의 — etd 축 이중계상 방지 (단계0 일치권고)
    "sts_wait": 0.0, "transfer_wait": 0.0,      # 내부 대기 = 선석초과에 흡수되는 proxy
    "lane_cong": 0.0, "interference": 0.0,      # 혼잡 proxy 제거 — 마스크/resolver 소관 (문헌)
    "resequence": 0.0, "imbalance": 0.0}


def numeraire_v1_config() -> TerminalCostConfig:
    """기준재 상대비용 v1 — 트럭대기 1h = 1.0, 나머지는 경제 비율 ρ (YR-080 단계4).

    λ_vessel 정적 1.0 — **ρ_vessel 이 유일한 본선 배율** (이중곱 금지 계약,
    test_yr080_numeraire). 13항 이름·순서 불변(계약 유지) — 미사용 항은 weight 0.
    원화 복원: 기여분 × (트럭 시간비용 원/h) 로 언제든 복원 가능 (v5 §2 유효조건).
    """
    sp = Provenance(ProvBasis.ASSUMED, "YR-080 단계4",
                    "순수 단위환산 (3600s=1h·7200m=크레인1h@2.0m/s·count=1)")
    wp = Provenance(ProvBasis.ASSUMED, "본선전략 v5 §2·문헌조사 2026-07-21",
                    "ρ_vessel=33 assumed(접안료÷트럭시간비용 근사) — 민감도 ×1/3~3 보고 의무")
    terms = {t: TermCost(t, NUMERAIRE_SCALE[t], NUMERAIRE_WEIGHT[t], sp, wp)
             for t in COST_TERMS}
    lp = LambdaVesselPolicy(
        LambdaMode.STATIC,
        Provenance(ProvBasis.ASSUMED, "YR-080 단계4", "ρ_vessel 단일 배율 — λ 이중곱 금지"),
        static_value=1.0)
    return TerminalCostConfig(
        "TERMINAL-COST-NUMERAIRE-V1", SCHEMA_VERSION, True, terms, lp,
        provenance_note="기준재 상대비용 (트럭대기 1h=1.0). 전 ρ assumed — 실측/계약자료 시 교체"
    ).validate()


def default_assumed_config() -> TerminalCostConfig:
    """현 ASSUMED_SCALE/WEIGHT·assumed_lambda_vessel 바이트 재현 (파일 IO 없음 → golden·격리 안전)."""
    a = Provenance(ProvBasis.ASSUMED, note="PoC placeholder, YR-002 확정 대상")
    terms = {t: TermCost(t, ASSUMED_SCALE[t], ASSUMED_WEIGHT[t], a, a) for t in COST_TERMS}
    lp = LambdaVesselPolicy(
        LambdaMode.DYNAMIC,
        Provenance(ProvBasis.ASSUMED, "최종전략 §10.6 초기후보", "확정 아님·민감도 대상"),
        bands=default_lambda_bands())
    return TerminalCostConfig(
        "TERMINAL-COST-V1", SCHEMA_VERSION, True, terms, lp,
        provenance_note="전 항목 assumed. 실측=YR-002, baseline-fit=YR-038").validate()
