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

    def __post_init__(self) -> None:
        if self.n_external < 1 or self.n_vessels < 0 or self.vessel_moves < 1:
            raise ValueError("n_external>=1, n_vessels>=0, vessel_moves>=1")
        if not (0.0 <= self.gate_out_share <= 1.0 and 0.0 < self.fill_ratio < 0.9):
            raise ValueError("share/fill_ratio 범위 위반")
        if self.horizon_s <= 0 or self.drain_window_s <= 0:
            raise ValueError("horizon/drain 은 양수")
        if not (0.0 <= self.sigma_frac < 0.5):
            raise ValueError("sigma_frac 은 [0,0.5)")


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
    for i in range(params.n_external):
        arrival = params.horizon_s * (i + rng.random()) / params.n_external
        # ETA 오차 축 (YR-043): 게이트→블록 소요를 μ=600s 평균조건 가우시안으로
        gate_travel = trunc_normal(rng, 600.0, params.sigma_frac or 0.12, lo=60.0)
        gate_in = max(0.0, arrival - gate_travel)
        if rng.random() < params.gate_out_share and free_targets:
            target = free_targets.pop()
            jobs.append(Job(job_id=f"J-OUT-{i:03d}", flow=JobFlow.GATE_OUT,
                            release_time=0.0, actual_gate_in=gate_in,
                            actual_block_arrival=arrival, target_container=target))
        else:
            jobs.append(Job(job_id=f"J-IN-{i:03d}", flow=JobFlow.GATE_IN,
                            release_time=0.0, actual_gate_in=gate_in,
                            actual_block_arrival=arrival,
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
    return TerminalScenario(
        scenario_id=f"gen-t{params.n_external}v{params.n_vessels}-s{seed}",
        seed=seed, horizon_s=params.horizon_s,
        drain_window_s=params.drain_window_s,
        containers=containers, jobs=jobs, vessels=vessels,
        injected_events=[],
        meta={"generator": "terminal-gen-v1", "assumed": True})
