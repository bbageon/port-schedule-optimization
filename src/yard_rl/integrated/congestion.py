"""혼잡도 구성 컴포넌트 — 환경의 '얼마나 붐비나'를 설정 가능한 다이얼로 모듈화.

터미널 선택기(terminal_registry)와 짝을 이룬다: **터미널(어디)** × **혼잡도(얼마나 붐비나)**
→ 시나리오. 계층형:
  ① 직교 다이얼 4축 — 서로 독립적으로 조절:
     - trucks_per_hour_per_crane : 외부트럭 도착 강도 (문헌 4~10대/h/크레인)
     - arrival_peak (amp·width)  : 러시아워 시간 집중도 (0=균등)
     - yard_fill_ratio           : 초기 장치율 (높을수록 재조작↑)
     - vessel_pressure           : 본선 압력 off|normal|tight (마감·STS 수요)
  ② 이름표 레벨 — 위 다이얼의 문헌 근거 조합 (idle→saturation). 아무 다이얼이나 override 가능.

`to_gen_params(profile)` 가 시나리오 생성기(`TerminalGenParams`)로 컴파일한다. 크레인 수·도착창
길이는 프로파일에서 읽어 n_external 을 유도(전체 크레인수÷도착창 아님 — trucks/h/crane × 크레인 ×
시간). 강도·창폭은 assumed(형태만 문헌) — provenance 로 등급 박제, 성능 주장은 claim gate 준수.

호환: congestion("normal").to_gen_params(2크레인 profile) == calibrated_load_params("mid")
(n_external 56·피크 1.0·gate 210·기본 본선). 즉 기존 부하 프리셋의 상위 일반화다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .profile import IntegratedProfile
from .scenario_gen import TerminalGenParams

# 본선 압력 → (본선 수, move, STS 간격s, 마감배율). tight=YR-041 vessel_rush 근사.
_VESSEL = {
    "off":    dict(n_vessels=0, vessel_moves=15, sts_move_interval_s=144.0,
                   vessel_deadline_mult=2.0),   # 본선 없음 — 순수 트럭 혼잡
    "normal": dict(n_vessels=2, vessel_moves=15, sts_move_interval_s=144.0,
                   vessel_deadline_mult=2.0),   # 느슨 마감 (기존 기본)
    "tight":  dict(n_vessels=2, vessel_moves=24, sts_move_interval_s=110.0,
                   vessel_deadline_mult=1.15),  # 빡빡 ETD·높은 STS 수요
}

_PROVENANCE = {
    "trucks_per_hour_per_crane": "문헌 4~10대/h/크레인(결정자료 §5-6) — 형태 문헌·강도 assumed",
    "arrival_peak": "부산신항 반출입 주간 피크 형태 문헌 — 강도·창폭 assumed",
    "yard_fill_ratio": "assumed (실측 장치율 미확보 — D5)",
    "vessel_pressure": "assumed (부산 특정 선석 스케줄 비공개 — D5). tight=YR-041 근사",
}


@dataclass(frozen=True)
class CongestionSpec:
    """혼잡도 4축 + 레이블. to_gen_params 로 시나리오 파라미터 컴파일."""

    trucks_per_hour_per_crane: float
    arrival_peak_amp: float
    arrival_peak_width_frac: float
    yard_fill_ratio: float
    vessel_pressure: str                    # off | normal | tight
    label: str = "custom"
    provenance: dict = field(default_factory=lambda: dict(_PROVENANCE))

    def __post_init__(self) -> None:
        if self.trucks_per_hour_per_crane <= 0:
            raise ValueError("trucks_per_hour_per_crane 은 양수")
        if self.vessel_pressure not in _VESSEL:
            raise ValueError(f"vessel_pressure 는 {sorted(_VESSEL)} 중: {self.vessel_pressure!r}")
        if not (0.0 < self.yard_fill_ratio < 0.9):
            raise ValueError("yard_fill_ratio 는 (0,0.9)")
        if self.arrival_peak_amp < 0:
            raise ValueError("arrival_peak_amp 는 0 이상")

    def n_external_for(self, profile: IntegratedProfile,
                       horizon_s: float = 14_400.0) -> int:
        """trucks/h/crane × 크레인수 × 도착창(h) → 외부트럭 수 (전체수÷블록 아님)."""
        n_cranes = len(profile.cranes)
        return max(1, round(self.trucks_per_hour_per_crane * n_cranes * (horizon_s / 3600.0)))

    def to_gen_params(self, profile: IntegratedProfile, *,
                      horizon_s: float = 14_400.0, gate_travel_mu_s: float = 210.0,
                      gaussian: bool = True, **overrides) -> TerminalGenParams:
        base = dict(
            n_external=self.n_external_for(profile, horizon_s),
            arrival_peak_amp=self.arrival_peak_amp,
            arrival_peak_center_frac=0.5,
            arrival_peak_width_frac=self.arrival_peak_width_frac,
            fill_ratio=self.yard_fill_ratio,
            gate_travel_mu_s=gate_travel_mu_s,
            horizon_s=horizon_s, gaussian=gaussian,
            **_VESSEL[self.vessel_pressure],
        )
        base.update(overrides)
        return TerminalGenParams(**base)


# ── 이름표 레벨 (문헌 근거 조합). 아무 다이얼이나 override 가능.
_LEVELS = {
    # level      : (tph, peak_amp, peak_width, fill, vessel)
    "idle":       (4.0, 0.0, 0.25, 0.25, "off"),     # 한산 — 균등·본선없음
    "light":      (5.0, 0.5, 0.25, 0.30, "normal"),  # 저부하 (기존 current 40 근처)
    "normal":     (7.0, 1.0, 0.25, 0.30, "normal"),  # 중부하 (= calibrated mid 56)
    "busy":       (10.0, 1.0, 0.25, 0.35, "normal"), # 고부하 (= calibrated high 80)
    "rush":       (10.0, 2.0, 0.20, 0.40, "tight"),  # 러시+본선 겹침 (coincident 근사)
    "saturation": (14.0, 1.0, 0.25, 0.65, "tight"),  # 초과수요·고장치율·빡빡 본선
}


def congestion(level: str = "normal", **overrides) -> CongestionSpec:
    """이름표 레벨 → CongestionSpec (다이얼 override 가능).

    예) congestion("busy")                      # 고부하 프리셋
        congestion("normal", vessel_pressure="tight")   # 중부하지만 본선 빡빡
        congestion("rush", trucks_per_hour_per_crane=12) # 러시 + 트럭 강도 상향
    """
    if level not in _LEVELS:
        raise ValueError(f"level 은 {sorted(_LEVELS)} 중: {level!r}")
    tph, amp, width, fill, vessel = _LEVELS[level]
    spec = dict(trucks_per_hour_per_crane=tph, arrival_peak_amp=amp,
                arrival_peak_width_frac=width, yard_fill_ratio=fill,
                vessel_pressure=vessel, label=level)
    spec.update(overrides)
    return CongestionSpec(**spec)


def list_levels() -> list[dict]:
    """선택 표면 — 레벨별 다이얼 요약."""
    out = []
    for lv, (tph, amp, width, fill, vessel) in _LEVELS.items():
        out.append({"level": lv, "trucks_per_hour_per_crane": tph,
                    "arrival_peak_amp": amp, "yard_fill_ratio": fill,
                    "vessel_pressure": vessel})
    return out
