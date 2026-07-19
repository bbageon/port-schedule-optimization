"""통합 터미널 시나리오 생성기 — seed 결정론 (YR-039 §4).

RNG 는 여기에만 존재하고 엔진은 결정론 입력만 소비한다 (YR-036 계약).
fixture(build_minimal_terminal_scenario)의 형태 계약을 따르되 규모·구성을
매개변수화한다. 전 항목 assumed — 실측 캘리브레이션은 YR-002/009 후.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..contract.vessel import CompletionBasis
from ..domain.enums import ContainerSize, JobFlow, LoadStatus
from ..domain.models import Container, Job
from .profile import IntegratedProfile
from .scenario import TerminalScenario
from .vessel import VesselPlan, VesselProcess, VesselWorkType


@dataclass(frozen=True)
class TerminalGenParams:
    """기본값은 **평균조건 μ** — gaussian=True 면 seed 별로 TruncatedNormal(μ,σ,±2σ) 추출 (YR-043).

    본선 악화 축은 본 트랙에서 분리한다 (사용자 결정): 스트레스 시나리오 제외·λ_vessel=1.0 중립·
    본선 KPI 기록만. 고부하/타이트 deadline 축은 YR-041.
    """

    n_external: int = 40             # 외부트럭 (반입/반출 혼합) — μ
    gate_out_share: float = 0.6
    n_vessels: int = 2               # DISCHARGE(RISK)·LOAD(SYMPTOM) 교대
    vessel_moves: int = 15           # 본선당 계획 move 수 — μ
    fill_ratio: float = 0.30         # 초기 장치율 (전 슬롯 대비) — μ
    horizon_s: float = 14_400.0      # 4h 도착 구간
    drain_window_s: float = 7_200.0
    size_mix_ft40: float = 0.7
    sts_move_interval_s: float = 144.0   # STS 간격 — μ
    gaussian: bool = True            # 평균조건 가우시안 변주 (YR-043)
    sigma_frac: float = 0.12         # σ ≈ 10~15% of μ (assumed)
    # YR-048: 제공 ETA 오차 — 단일야드 관행(io/scenario_gen §18.2 EMPIRICAL)과 동일하게
    # eta = 실제도착 ± uniform(eta_error_s). 0 이면 PERFECT. 품질 매트릭스는 YR-019.
    eta_error_s: float = 300.0
    # YR-002 재기준화 (D5): 도착 피크 — 0.0 이면 기존 stratified-uniform 과 바이트 동일.
    # amp>0 이면 [center−width/2, center+width/2]·horizon 창의 도착률이 (1+amp)배.
    # 형태 근거는 문헌(부산신항 반출입 주간 피크·8h 주기 — 결정자료 §5-6), 강도는 assumed.
    arrival_peak_amp: float = 0.0
    arrival_peak_center_frac: float = 0.5
    arrival_peak_width_frac: float = 0.25
    # 게이트→블록 순주행 μ (기존 하드코딩 600s 를 파라미터화 — 기본값 동일, 문헌은 210s 대)
    gate_travel_mu_s: float = 600.0

    def __post_init__(self) -> None:
        if self.n_external < 1 or self.n_vessels < 0 or self.vessel_moves < 1:
            raise ValueError("n_external>=1, n_vessels>=0, vessel_moves>=1")
        if not (0.0 <= self.gate_out_share <= 1.0 and 0.0 < self.fill_ratio < 0.9):
            raise ValueError("share/fill_ratio 범위 위반")
        if self.horizon_s <= 0 or self.drain_window_s <= 0:
            raise ValueError("horizon/drain 은 양수")
        if not (0.0 <= self.sigma_frac < 0.5):
            raise ValueError("sigma_frac 은 [0,0.5)")
        if self.eta_error_s < 0:
            raise ValueError("eta_error_s 는 0 이상")
        if self.arrival_peak_amp < 0:
            raise ValueError("arrival_peak_amp 는 0 이상")
        if not (0.0 <= self.arrival_peak_center_frac <= 1.0
                and 0.0 < self.arrival_peak_width_frac <= 1.0):
            raise ValueError("peak center∈[0,1]·width∈(0,1] 위반")
        if self.gate_travel_mu_s <= 0:
            raise ValueError("gate_travel_mu_s 는 양수")


def trunc_normal(rng: random.Random, mu: float, sigma_frac: float, *,
                 lo: float | None = None, hi: float | None = None) -> float:
    """TruncatedNormal(μ, σ=sigma_frac·μ) 를 ±2σ 에서 절단 (YR-043 평균조건).

    RNG 는 시나리오 생성에만 — 엔진은 결정론 입력만 소비 (YR-036 계약).
    """
    if sigma_frac <= 0 or mu == 0:
        return mu
    sigma = abs(mu) * sigma_frac
    lo = mu - 2.0 * sigma if lo is None else max(lo, mu - 2.0 * sigma)
    hi = mu + 2.0 * sigma if hi is None else min(hi, mu + 2.0 * sigma)
    for _ in range(16):                      # 절단 재추출 (유한 시도 — 결정론)
        x = rng.gauss(mu, sigma)
        if lo <= x <= hi:
            return x
    return min(hi, max(lo, mu))              # fallback: μ clamp


def _peak_warp(u: float, amp: float, center: float, width: float) -> float:
    """stratified-uniform 분위 u∈[0,1] 를 피크 밀도의 역CDF 로 사상 (YR-002 D5).

    도착률 = 창 밖 1, 창 안 (1+amp) 인 구간상수 밀도. amp=0 이면 항등(u 그대로)
    — 추가 난수 소비 없음 → 기존 seed 시나리오 바이트 동일 보존이 계약이다.
    """
    if amp <= 0.0:
        return u
    a = max(0.0, center - width / 2.0)
    b = min(1.0, center + width / 2.0)
    w = b - a
    total = 1.0 + amp * w                      # 정규화 전 총질량
    m = u * total
    if m < a:
        return m
    if m < a + (1.0 + amp) * w:
        return a + (m - a) / (1.0 + amp)
    return b + (m - a - (1.0 + amp) * w)


def calibrated_load_params(level: str = "mid", **overrides) -> TerminalGenParams:
    """문헌 보정 부하 프리셋 (YR-002 재기준화 — 결정자료 §5-6·§5-8).

    문헌: 외부트럭 4~10대/h/크레인 × 크레인 2기 × 도착창 4h →
    현행 40대(=5/h/크레인)는 하한권. mid=7/h/크레인(56대)·high=상한 10(80대)·
    current=기존 40 대조용. 피크 창(중앙 1h, 2배)·게이트 순주행 210s(문헌 §5-7,
    프로파일 yaml 과 정합) 동반. 강도·창폭은 assumed (형태만 문헌).
    """
    n_ext = {"current": 40, "mid": 56, "high": 80}
    if level not in n_ext:
        raise ValueError(f"level 은 {sorted(n_ext)} 중 하나: {level!r}")
    base = dict(n_external=n_ext[level], arrival_peak_amp=1.0,
                arrival_peak_center_frac=0.5, arrival_peak_width_frac=0.25,
                gate_travel_mu_s=210.0)
    base.update(overrides)
    return TerminalGenParams(**base)


def _place_containers(rng: random.Random, profile: IntegratedProfile,
                      params: TerminalGenParams) -> dict[str, Container]:
    """초기 스택 — 슬롯 충돌·공중적재 없음 (아래부터 채움)."""
    g = profile.block
    containers: dict[str, Container] = {}
    heights: dict[tuple[int, int], int] = {}
    n_slots = g.bay_count * g.row_count * g.tier_max
    target = max(params.n_external + params.n_vessels * params.vessel_moves,
                 int(n_slots * params.fill_ratio))
    cells = [(b, r) for b in range(1, g.bay_count + 1)
             for r in range(1, g.row_count + 1)]
    idx = 0
    while len(containers) < min(target, n_slots - g.bay_count):  # 여유 슬롯 보존
        b, r = cells[rng.randrange(len(cells))]
        h = heights.get((b, r), 0)
        if h >= g.tier_max:
            continue
        idx += 1
        cid = f"C{idx:04d}"
        size = (ContainerSize.FT40 if rng.random() < params.size_mix_ft40
                else ContainerSize.FT20)
        containers[cid] = Container(container_id=cid, size=size,
                                    load_status=LoadStatus.FULL, block=g.block_id,
                                    bay=b, row=r, tier=h + 1)
        heights[(b, r)] = h + 1
    return containers


def generate_terminal_scenario(profile: IntegratedProfile, seed: int,
                               params: TerminalGenParams | None = None
                               ) -> TerminalScenario:
    params = params or TerminalGenParams()
    rng = random.Random(seed)
    if params.gaussian:
        # 평균조건 가우시안 (YR-043): 기본값을 μ 로 seed 별 TruncatedNormal 추출.
        # 축 — 트럭 물량·장치율·본선 물량·STS 간격 (ETA 오차·작업시간은 아래 job 생성에서).
        from dataclasses import replace as _rep
        sf = params.sigma_frac
        params = _rep(
            params,
            n_external=max(1, round(trunc_normal(rng, params.n_external, sf))),
            fill_ratio=min(0.85, max(0.05, trunc_normal(rng, params.fill_ratio, sf))),
            vessel_moves=max(1, round(trunc_normal(rng, params.vessel_moves, sf))),
            sts_move_interval_s=max(10.0, trunc_normal(rng, params.sts_move_interval_s, sf)),
            gaussian=False)     # 이하 결정론 소비 (재추출 금지)
    containers = _place_containers(rng, profile, params)
    # 대상 예약: top-of-stack 부터 소진 (재조작은 자연 발생 — blocker 위 배치 반출도 섞임)
    free_targets = sorted(containers)
    rng.shuffle(free_targets)
    jobs: list[Job] = []

    # ---- 외부트럭 (도착 horizon 내 균등 + jitter)
    # YR-048: 제공 ETA 주입 — 없으면 PRE_ADVICE 레벨에서도 PRE_REHANDLE(선제 재조작) 후보가
    # 전혀 생성되지 않아 H2(ETA 선제정리) 축이 통째로 비활성이 된다 (YR-047 리뷰 파생 발견).
    # **전용 RNG 스트림**을 쓰는 이유: 기존 draw 열(도착·대상·본선)을 밀지 않아 같은 seed 의
    # 시나리오 구조가 이전과 바이트 동일하게 유지되고, 변화가 정확히 "ETA 추가"로 한정된다.
    eta_rng = random.Random(f"eta:{seed}")
    for i in range(params.n_external):
        # 기존 수식 보존 (부동소수점 결합 순서까지 — 골든 계약). 피크는 opt-in 후처리.
        arrival = params.horizon_s * (i + rng.random()) / params.n_external
        if params.arrival_peak_amp > 0.0:
            arrival = params.horizon_s * _peak_warp(
                arrival / params.horizon_s, params.arrival_peak_amp,
                params.arrival_peak_center_frac, params.arrival_peak_width_frac)
        # 게이트→블록 소요 μ (기본 600s — YR-043 평균조건 가우시안, YR-002 D5 파라미터화)
        gate_travel = trunc_normal(rng, params.gate_travel_mu_s,
                                   params.sigma_frac or 0.12, lo=60.0)
        gate_in = max(0.0, arrival - gate_travel)
        eta = max(0.0, arrival + eta_rng.uniform(-params.eta_error_s, params.eta_error_s))
        if rng.random() < params.gate_out_share and free_targets:
            target = free_targets.pop()
            jobs.append(Job(job_id=f"J-OUT-{i:03d}", flow=JobFlow.GATE_OUT,
                            release_time=0.0, actual_gate_in=gate_in,
                            actual_block_arrival=arrival, provided_eta=eta,
                            target_container=target))
        else:
            jobs.append(Job(job_id=f"J-IN-{i:03d}", flow=JobFlow.GATE_IN,
                            release_time=0.0, actual_gate_in=gate_in,
                            actual_block_arrival=arrival, provided_eta=eta,
                            inbound_size=(ContainerSize.FT40
                                          if rng.random() < params.size_mix_ft40
                                          else ContainerSize.FT20),
                            inbound_load=LoadStatus.FULL))

    # ---- 본선 (DISCHARGE=RISK 완결근거 / LOAD=SYMPTOM 결측 — fixture 계약)
    vessels: list[VesselProcess] = []
    for v in range(params.n_vessels):
        start = params.horizon_s * (0.1 + 0.7 * v / max(1, params.n_vessels))
        cadence = params.sts_move_interval_s     # 평균조건 가우시안 추출값 (YR-043)
        n_moves = params.vessel_moves
        work = (VesselWorkType.DISCHARGE if v % 2 == 0 else VesselWorkType.LOAD)
        vid = f"V-{work.value[:4]}-{v}"
        if work == VesselWorkType.DISCHARGE:
            plan = VesselPlan(planned_start_s=start,
                              planned_completion_s=start + n_moves * cadence * 2.0,
                              completion_basis=CompletionBasis.PLAN_COMPUTED,
                              etd_s=start + n_moves * cadence * 3.0,
                              total_moves=n_moves, sts_move_interval_s=cadence)
        else:
            plan = VesselPlan(planned_start_s=start, planned_completion_s=None,
                              completion_basis=None, etd_s=None,
                              total_moves=n_moves, sts_move_interval_s=cadence)
        vessels.append(VesselProcess(vid, work, plan))
        n_linked = min(max(2, n_moves // 3), len(free_targets))
        flow = (JobFlow.VESSEL_DISCHARGE if work == VesselWorkType.DISCHARGE
                else JobFlow.VESSEL_LOAD)
        for m in range(n_linked):
            target = free_targets.pop()
            jobs.append(Job(
                job_id=f"J-{vid}-{m:02d}", flow=flow,
                release_time=start + m * cadence,
                actual_gate_in=None, actual_block_arrival=None,
                target_container=target,
                deadline=start + n_moves * cadence * 2.0 + 1800.0,
                priority_class=1))

    jobs.sort(key=lambda j: j.job_id)
    # eta_error_s 박제 — YR-019 품질축 arm 정체성 (리뷰 반영). 부하 현실화(YR-002 D5)
    # 필드는 비기본일 때만 추가 — 기본 시나리오 meta 는 기존과 완전 동일 유지.
    meta: dict = {"generator": "terminal-gen-v1", "assumed": True,
                  "eta_error_s": params.eta_error_s}
    if params.arrival_peak_amp > 0.0:
        meta["arrival_peak"] = (params.arrival_peak_amp,
                                params.arrival_peak_center_frac,
                                params.arrival_peak_width_frac)
    if params.gate_travel_mu_s != 600.0:
        meta["gate_travel_mu_s"] = params.gate_travel_mu_s
    return TerminalScenario(
        scenario_id=f"gen-t{params.n_external}v{params.n_vessels}-s{seed}",
        seed=seed, horizon_s=params.horizon_s,
        drain_window_s=params.drain_window_s,
        containers=containers, jobs=jobs, vessels=vessels,
        injected_events=[], meta=meta)
