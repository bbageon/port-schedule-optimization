"""YR-150 1단계 — H-21 터미널 전체 유입 스트림과 21블록 최초배정 (지속 유입 계약).

■ 무엇이 달라지는가 (기존 생성기 대비)
1. **터미널 master stream 하나**를 만들고 배분벡터 `p` 로 21블록에 나눈다. 블록을 따로
   생성해 터미널 합계가 21배로 부풀어나는 방식은 금지다(spec 엔진 선결 2). 트럭의
   도착시각·규격·flow·예약오차는 **블록과 무관하게** 먼저 정해지고, `p` 는 오직
   "어느 블록으로 갈지"만 정한다 — 그래서 `p` 를 바꿔도 같은 트럭이 재배치될 뿐이다.
2. **관측창 전체에 걸쳐 계속 도착**한다. 기존 생성기는 `horizon` 안에서만 도착을 만들고
   `drain` 동안 비웠다("쌓인 물량을 얼마나 빨리 비우는가" 구조). 여기서는 warm-up +
   측정구간 끝까지 유입이 이어지고 **관측시간에서 그대로 종료**한다(미완도 장부에 남긴다).
3. **게이트→블록 주행이 목적지에 따라 다르다**(YR-150 0단계 `yard_layout`).

■ 부하 L 의 뜻 — ★4차 재정의(사용자 결정 2026-08-08): **고정 재공량(WIP)**
`L ∈ {50,75,100,125,150}` 은 **터미널(21블록 합계) 안에 유지하는 외부트럭 대수**다.
초기 채움 뒤 트럭이 나갈 때마다 대기 pool 에서 새 트럭을 투입해 내부 대수를 L 로 유지한다
(`build_fixed_wip` + `WipAdmissionController`). 유지는 투입주기(기본 60초) 단위 근사다.

  · 이전 계약(고정 유입량 — 4시간당 총 L 대 도착)은 `build_terminal` 로 **보조 보존**된다.
    1~3차 재정의 시절의 주 계약이었고, 도착률 고정 진단·회귀 테스트가 이를 쓴다.
  · **알려진 한계(계약에 박제)**: 고정 WIP 에서는 빨리 처리하는 정책일수록 트럭을 더 받아
    **정책별 처리 물량이 달라진다**. 따라서 성능 비교는 "같은 재공량에서 처리량과 시간당
    비용"의 공동 판정이어야 하며, 에피소드 총비용 단독 비교는 금지다.
  · 고정 WIP 모드는 **walk-in** 이다 — 예약·준수오차 개념이 없고, 투입 시각이 곧 gate-in
    이다. 예측 블록도착 = gate-in + 기대 주행(실현 잔여편차 미참조 — 누출 0).
  · lead(사전 통지) 0 이면 투입 즉시 gate-in 이므로 **PRE_GATE 창이 없다** — 블록 간
    재배정(SELL) 연구 단계(0B)에서는 lead>0 설계가 선행돼야 한다(정직 고지).

■ 축소 가정 (정직 기록)
같은 블록으로 가는 트럭들 사이의 잔여 주행편차 σ 를 5초로 둔다. 기존 계약은 게이트→블록
시간을 중심 300초·σ60초로 뭉뚱그렸는데, 그 편차의 대부분은 이제 **어느 블록으로 가는가**로
설명된다(블록별 190~410초). 잔여 σ 는 전 블록이 계약 지원범위 180~420초 안에 남도록
**결정된 값**이며 자유변수가 아니다. 주행잡음을 더 크게 두는 것은 성능 단계 전 별도 등록.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

from ..domain.enums import ContainerSize, JobFlow, LoadStatus
from ..domain.models import Job
from .profile import IntegratedProfile
from .scenario import TerminalScenario
from .scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MIN_S, TerminalGenParams,
                           generate_terminal_scenario, trunc_normal)
from .time_grid import on_grid
from .yard_layout import YardLayout, terminal_layout

LOAD_WINDOW_S = 14_400.0        # L 의 기준 창 = 4시간 (부하 정의의 분모)
RESID_TRAVEL_SIGMA_S = 5.0      # 같은 블록 안 잔여 주행편차 — 지원범위에서 결정된 값


@dataclass(frozen=True)
class ObservationContract:
    """warm-up → 측정구간 → **관측시간 종료**. 비우기 구간은 두지 않는다."""

    warmup_s: float = 3_600.0
    measure_s: float = 14_400.0
    snapshot_s: float = 300.0

    def __post_init__(self) -> None:
        if self.warmup_s < 0 or self.measure_s <= 0:
            raise ValueError("warmup>=0·measure>0")
        if not (300.0 <= self.snapshot_s <= 600.0):
            raise ValueError("스냅샷 간격은 5~10분(spec 계약)")
        if self.measure_s % self.snapshot_s != 0:
            raise ValueError("측정구간은 스냅샷 간격의 배수여야 함")

    @property
    def observe_s(self) -> float:
        return self.warmup_s + self.measure_s

    def snapshot_times(self) -> list[float]:
        n = int(self.measure_s // self.snapshot_s)
        return [self.warmup_s + i * self.snapshot_s for i in range(n + 1)]

    def as_dict(self) -> dict:
        return {"warmup_s": self.warmup_s, "measure_s": self.measure_s,
                "snapshot_s": self.snapshot_s, "observe_s": self.observe_s,
                "end_rule": "관측시간에서 종료 — 미완 작업도 장부에 남긴다"}


@dataclass(frozen=True)
class TerminalStreamParams:
    """터미널 전체 계약. 블록별이 아니라 **터미널 합계**로 준다."""

    load_4h: int                                # L — 21블록 합계 4시간 도착량
    hotspot_blocks: tuple[str, ...] = ()        # 비면 균등 배분
    hotspot_weight: float = 3.0
    gate_out_share: float = 0.6
    size_mix_ft40: float = 0.7
    # ★장치율 재보정(사용자 결정 2026-08-10): 0.30 → 0.65 — 부산항 평시 관측·문헌
    # 대역 [0.60, 0.75] 의 중심값(앵커 등록부 initial_yard_occupancy). 구 0.30 은
    # 문헌 최저 실측(52%)보다 낮은 assumed 값이었다. 주의: 구세대 경로
    # (scenario_gen.TerminalGenParams 기본 0.30)는 골든 보존을 위해 불변 — 이 값은
    # H-21 지속유입 계약 전용이며 변경 시 전면 재자격 필요.
    fill_ratio: float = 0.65
    # ★본선 물량 재보정 (2026-08-08 — 척당 15건은 2블록 시절 값의 무비판 승계였다).
    #
    # 유도 사슬 (근거: configs/anchors/external_anchors_v1.json vessel_workload):
    #   HJNC 연간 처리량 2,310,000 TEU(manifest confirmed) ÷ 8,760h = 264 TEU/h
    #   ÷ TEU/van 1.6~1.8(문헌 가정) = 터미널 전체 본선 작업률 **146~165 moves/h**.
    #   교차검증: 선석생산성 99.2 moves/h(KSG) × 동시 접안 1.5~1.7척(척당 5,000TEU·
    #   1.3일, 부산일보) = 149~169 moves/h ✓ 수렴.
    # 엔진 대응: VesselProcess 1개 = STS 크레인 1기의 작업 스트림(cadence 144s = 25/h).
    #   실제 선박 1척(≈97 moves/h) ≈ 3~4 스트림 ✓. 목표 작업률 ÷ 25 ≈ 6 개의 상시
    #   스트림이 필요하고, 시작을 관측창에 분산하면(창 절단) **12 process × 120 moves**
    #   가 계획 작업률 ≈150 moves/h 를 만든다(관측범위 145~170 안 — 시작 슬롯을
    #   5%~95% 로 정정한 2026-08-09 이후 값. 구판 5%~87.5% 는 ≈161).
    # 척당 실물량(≈3,000 moves·1.3일)은 4~5시간 관측창에 들어갈 수 없으므로, 창중
    # 모형은 "그 시간 동안 진행 중인 스트림"만 표현한다 — process 물량 120 =
    # 스트림 1개가 관측창을 꽉 채울 때의 절단 상한((18,000−900)s ÷ 144s).
    vessels_total: int = 12                     # 터미널 전체 본선 process(STS 스트림) 수
    vessel_moves: int = 120
    eta_error_s: float = 300.0
    appt_window_s: float = 3_600.0
    appt_adherence_sigma_s: float = 600.0
    exit_travel_mu_s: float = 300.0
    sts_move_interval_s: float = 144.0
    resid_travel_sigma_s: float = RESID_TRAVEL_SIGMA_S


# ------------------------------------------------------------------ 배분벡터
def distribution_vector(layout: YardLayout, params: TerminalStreamParams
                        ) -> dict[str, float]:
    """`p` — 균등 또는 hotspot 가중. 합은 정확히 1."""
    w = {b: (params.hotspot_weight if b in params.hotspot_blocks else 1.0)
         for b in layout.ids}
    tot = sum(w.values())
    return {b: v / tot for b, v in w.items()}


def allocate(p: dict[str, float], n: int) -> dict[str, int]:
    """최대잉여법 — 합이 정확히 n 이고 결정론적이다(반올림 누락·초과 금지)."""
    base = {b: int(n * v) for b, v in p.items()}
    rest = n - sum(base.values())
    order = sorted(p, key=lambda b: (-(n * p[b] - base[b]), b))
    for b in order[:rest]:
        base[b] += 1
    if sum(base.values()) != n:
        raise AssertionError("배분 합 불일치")
    return base


# ------------------------------------------------------------------ 블록 배경
def _background(profile: IntegratedProfile, seed: int, bid: str,
                obs: ObservationContract, params: TerminalStreamParams,
                n_vessels: int, *, vessel_type_offset: int = 0) -> TerminalScenario:
    """블록의 **배경**(초기 적재·본선)만 만든다 — 외부트럭은 master stream 이 준다.

    vessel_type_offset: 블록당 1 process 구성에서 전 블록이 양하만 갖는 결함을 막는다
    (2026-08-08 정정) — 블록 index 를 넣어 터미널 전체로 양하/적하를 교대시킨다.
    """
    gp = TerminalGenParams(
        n_external=1,                    # 최소 1 — 아래에서 제거한다(배경만 쓴다)
        gate_out_share=0.0,              # 배경 트럭은 반출 대상을 소비하지 않게
        n_vessels=n_vessels, vessel_moves=params.vessel_moves,
        fill_ratio=params.fill_ratio, horizon_s=obs.observe_s, drain_window_s=1.0,
        size_mix_ft40=params.size_mix_ft40,
        sts_move_interval_s=params.sts_move_interval_s,
        gaussian=False,                  # 블록별 변주는 seed 로만 — 물량은 계약값 고정
        time_contract_v2=True, gate_block_contract=True,
        vessel_deadline_achievable=True, vessel_type_offset=vessel_type_offset)
    scn = generate_terminal_scenario(profile, seed, gp)
    keep = [j for j in scn.jobs if not j.is_external_truck]
    return dataclasses.replace(scn, jobs=keep, drain_window_s=0.0,
                               scenario_id=f"H21-{bid}-s{seed}")


def _shift_vessels(scn: TerminalScenario, delta_s: float) -> TerminalScenario:
    """본선 시작을 delta 만큼 통째로 민다 — 계획·ETD·물리하한이 함께 이동해 정합 유지."""
    if delta_s == 0.0 or not scn.vessels:
        return scn
    out = []
    for v in scn.vessels:
        p = v.plan
        shifted = dataclasses.replace(
            p, planned_start_s=p.planned_start_s + delta_s,
            planned_completion_s=(None if p.planned_completion_s is None
                                  else p.planned_completion_s + delta_s),
            etd_s=None if p.etd_s is None else p.etd_s + delta_s,
            phys_min_completion_s=(None if p.phys_min_completion_s is None
                                   else p.phys_min_completion_s + delta_s))
        out.append(dataclasses.replace(v, plan=shifted))
    return dataclasses.replace(scn, vessels=out)


# ------------------------------------------------------------------ master stream
def _master_arrivals(seed: int, n: int, obs: ObservationContract) -> list[float]:
    """관측창 **전체**에 걸친 예약 게이트 시각 — 층화균등(기존 도착식과 같은 형태)."""
    rng = random.Random(f"h21:stream:{seed}")
    return [obs.observe_s * (i + rng.random()) / n for i in range(n)]


# ================================================== 5차 계약: 24시간 이중 피크 (2026-08-11)
# 사전등록 동결본: .claude/docs/strategy-history/
#   2026-08-11-24시간-이중피크-상수-유도-사전등록.md  (§A — 결과 열람 전 확정)
DIURNAL_PEAKS = ((11.0, 1.0, 1.0), (15.0, 2.0, 2.0))   # (중심 h, σ h, 질량 가중)
DIURNAL_NIGHT_FRAC = 0.15        # 야간 저점 ÷ 일평균 (설계 파라미터 — 한계 박제)
DIURNAL_DAY_TOTAL = 3_600        # 하루 총 도착 대수 (평균 150대/h)
DIURNAL_DAY_S = 86_400.0


def diurnal_rate(t_s: float, *, day_s: float = DIURNAL_DAY_S,
                 total: int = DIURNAL_DAY_TOTAL,
                 night_frac: float = DIURNAL_NIGHT_FRAC,
                 peaks=DIURNAL_PEAKS) -> float:
    """시각 t 의 도착률 λ(t) [대/초] — 야간 상수 + 이중 정규 봉우리 (사전등록 §A3).

    형상만 정의하고 총량은 아래에서 정규화한다. 두 봉우리 높이는 (질량 ∝ 창 폭)
    규칙 때문에 자동으로 같아진다(A₁φ(0)/σ₁ = A₂φ(0)/σ₂ when A₂/A₁ = σ₂/σ₁).
    """
    import math
    mean_h = total / (day_s / 3600.0)          # 대/시간 평균
    b = night_frac * mean_h                    # 야간 저점 [대/h]
    w_sum = sum(w for _, _, w in peaks)
    peak_mass = total - b * (day_s / 3600.0)   # 봉우리에 남는 총 대수
    h = t_s / 3600.0
    lam_h = b
    for mu, sig, w in peaks:
        a = peak_mass * (w / w_sum)            # 이 봉우리의 총 대수
        lam_h += a * math.exp(-0.5 * ((h - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    return lam_h / 3600.0                      # 대/초


def diurnal_arrivals(seed: int, *, day_s: float = DIURNAL_DAY_S,
                     total: int = DIURNAL_DAY_TOTAL,
                     night_frac: float = DIURNAL_NIGHT_FRAC,
                     peaks=DIURNAL_PEAKS, grid_s: float = 60.0) -> list[float]:
    """λ(t) 의 역CDF 로 도착 시각 n=total 개 생성 — 층화균등·결정론(사전등록 §A6).

    격자(60초) 위에서 누적분포를 만들고 선형보간으로 역사상한다. 시드는 층화 칸 안의
    위치만 흔든다(형상은 시드 무관 — 같은 곡선, 다른 표본).
    """
    n_grid = int(day_s // grid_s)
    edges = [i * grid_s for i in range(n_grid + 1)]
    cum = [0.0]
    for i in range(n_grid):                     # 사다리꼴 적분
        lo, hi = edges[i], edges[i + 1]
        cum.append(cum[-1] + 0.5 * (diurnal_rate(lo, day_s=day_s, total=total,
                                                 night_frac=night_frac, peaks=peaks)
                                    + diurnal_rate(hi, day_s=day_s, total=total,
                                                   night_frac=night_frac, peaks=peaks))
                   * grid_s)
    tot = cum[-1]
    rng = random.Random(f"h21d:arr:{seed}")
    out, j = [], 0
    for i in range(total):
        q = tot * (i + rng.random()) / total    # 층화균등 분위
        while j + 1 < len(cum) and cum[j + 1] < q:
            j += 1
        span = cum[j + 1] - cum[j] if j + 1 < len(cum) else 1.0
        frac = 0.0 if span <= 0 else (q - cum[j]) / span
        out.append(min(day_s - 1e-6, edges[j] + frac * grid_s))
    out.sort()
    return out


def build_terminal(profile: IntegratedProfile, seed: int, *,
                   params: TerminalStreamParams,
                   obs: ObservationContract | None = None,
                   layout: YardLayout | None = None) -> dict:
    """21블록 시나리오 묶음 + 배정 원장을 만든다."""
    obs = obs or ObservationContract()
    layout = layout or terminal_layout()
    n_total = round(params.load_4h * obs.observe_s / LOAD_WINDOW_S)
    if n_total < len(layout.ids):
        raise ValueError(f"유입 {n_total}대가 블록 수 {len(layout.ids)} 보다 적다")

    # ① 본선을 블록·관측창 전체에 분산 (후반부가 트럭 전용 실험이 되지 않게 — spec 7)
    per_block_vessels = allocate({b: 1.0 / len(layout.ids) for b in layout.ids},
                                 params.vessels_total)
    scns: dict[str, TerminalScenario] = {}
    k = 0
    for b in layout.ids:
        bg = _background(profile, seed + 1000 * (layout.ids.index(b) + 1), b, obs,
                         params, per_block_vessels[b],
                         vessel_type_offset=layout.ids.index(b) % 2)
        if bg.vessels:
            # 이 블록 본선들의 시작을 터미널 순번 k 기준 위치로 옮긴다.
            target = obs.observe_s * (0.05 + 0.9 * k / max(1, params.vessels_total))
            bg = _shift_vessels(bg, target - bg.vessels[0].plan.planned_start_s)
            k += len(bg.vessels)
        scns[b] = bg

    # ② 블록별 반출 대상 후보 (배경 본선이 이미 쓴 컨테이너는 제외)
    free: dict[str, list[str]] = {}
    for b, scn in scns.items():
        used = {j.target_container for j in scn.jobs if j.target_container}
        cand = sorted(set(scn.containers) - used)
        random.Random(f"h21:tgt:{seed}:{b}").shuffle(cand)
        free[b] = cand

    # ③ master stream — 블록과 무관한 속성만 먼저 정한다
    appts = _master_arrivals(seed, n_total, obs)
    adh = random.Random(f"h21:adh:{seed}")
    exit_rng = random.Random(f"h21:exit:{seed}")
    eta_rng = random.Random(f"h21:eta:{seed}")
    mix_rng = random.Random(f"h21:mix:{seed}")
    resid = random.Random(f"h21:resid:{seed}")

    # ④ `p` 로 배정 — 배정 순서를 섞어 특정 블록이 앞부분만 받는 편향을 없앤다
    p = distribution_vector(layout, params)
    counts = allocate(p, n_total)
    slots = [b for b in layout.ids for _ in range(counts[b])]
    random.Random(f"h21:assign:{seed}").shuffle(slots)

    jobs: dict[str, list[Job]] = {b: [] for b in layout.ids}
    ledger: list[dict] = []
    for i, (appt, bid) in enumerate(zip(appts, slots)):
        sig = params.appt_adherence_sigma_s
        d = max(-2 * sig, min(2 * sig, adh.gauss(0.0, sig))) if sig > 0 else 0.0
        a_in = max(0.0, appt + d)                       # 실제 게이트 진입
        base = layout.gate_to_block_s(bid)              # ★목적지별 주행
        sr = params.resid_travel_sigma_s
        # ±2σ 절단 (기존 계약과 같은 관행). 조용히 자르지 않으므로 결과가 계약 지원범위를
        # 벗어나면 아래에서 예외 — 배치·σ 조합이 계약을 깨면 즉시 드러난다.
        travel = base + (max(-2 * sr, min(2 * sr, resid.gauss(0.0, sr))) if sr > 0 else 0.0)
        if not (GATE_BLOCK_MIN_S - 1e-9 <= travel <= GATE_BLOCK_MAX_S + 1e-9):
            raise ValueError(f"게이트→블록 주행 {travel:.1f}s 가 계약 지원범위 밖 "
                             f"[{GATE_BLOCK_MIN_S}, {GATE_BLOCK_MAX_S}] — 배치·잔여 σ 조합 위반")
        b_arr = a_in + travel
        est = appt + base                               # 예측 = 예약 + 기대 주행(실현 미참조)
        eta_rng.random()          # 스트림 정렬만 유지 — 실현 기반 ETA 는 **만들지 않는다**
        exit_t = trunc_normal(exit_rng, params.exit_travel_mu_s, 0.12, lo=60.0)
        out = mix_rng.random() < params.gate_out_share and bool(free[bid])
        if out:
            j = Job(job_id=f"T-OUT-{i:05d}", flow=JobFlow.GATE_OUT, release_time=0.0,
                    actual_gate_in=a_in, actual_block_arrival=b_arr, provided_eta=est,
                    target_container=free[bid].pop())
        else:
            j = Job(job_id=f"T-IN-{i:05d}", flow=JobFlow.GATE_IN, release_time=0.0,
                    actual_gate_in=a_in, actual_block_arrival=b_arr, provided_eta=est,
                    inbound_size=(ContainerSize.FT40
                                  if mix_rng.random() < params.size_mix_ft40
                                  else ContainerSize.FT20),
                    inbound_load=LoadStatus.FULL)
        for key, val in (("actual_gate_in", a_in), ("actual_block_arrival", b_arr),
                         ("provided_eta", est), ("estimated_block_arrival", est),
                         ("appointment_gate_time", appt),
                         ("appointment_window_start", appt - params.appt_window_s / 2),
                         ("appointment_window_end", appt + params.appt_window_s / 2),
                         ("exit_travel_s", exit_t)):
            setattr(j, key, val)
        jobs[bid].append(j)
        ledger.append({"job_id": j.job_id, "block": bid, "appt_s": round(appt, 3),
                       "gate_in_s": round(a_in, 3), "travel_s": round(travel, 3),
                       "block_arrival_s": round(b_arr, 3), "flow": j.flow.value})

    for b in layout.ids:
        scns[b] = dataclasses.replace(
            scns[b], jobs=scns[b].jobs + jobs[b],
            meta={**scns[b].meta, "h21_block": b, "h21_load_4h": params.load_4h,
                  "h21_assigned": len(jobs[b]), "observation": obs.as_dict()})
    return {"scenarios": scns, "assignment": ledger, "p": p, "counts": counts,
            "n_total": n_total, "observation": obs.as_dict(),
            "layout": layout.as_dict(),
            "vessels_per_block": per_block_vessels}


# ================================================================== 고정 WIP (4차 재정의)
WIP_ADMISSION_PERIOD_S = 60.0   # 투입 검토 주기 — 유지 근사 단위 (동결)
WIP_FILL_SPAN_S = 600.0         # 초기 채움을 펼치는 구간 (동시 진입 폭주 방지)
WIP_POOL_FACTOR = 30            # 대기 pool 크기 = L × factor (소진 시 결과에 명시 실패)


DIURNAL_VESSEL_STREAMS = 30      # 하루 총 본선 스트림 (동시 활성 6 — 사전등록 §C1)


def vessel_schedule_24h(layout: YardLayout, seed: int,
                        params: TerminalStreamParams,
                        obs: ObservationContract,
                        n_streams: int = DIURNAL_VESSEL_STREAMS) -> list[dict]:
    """24시간 본선 스트림 교대 배치 (사전등록 §C1) — 30개를 창 전체에 균등 분산.

    시작 시각을 [0, 창−스트림길이] 에 등간격으로 놓고 **시작 순서대로** 블록을
    돌려 배정한다 → 같은 블록의 두 스트림은 21칸(≈13.9h) 떨어져 **겹치지 않는다**
    (스트림 길이 4.8h). 블록 순서만 시드 셔플(특정 블록 운 제거), 양하/적하는 교대.
    """
    dur = params.vessel_moves * params.sts_move_interval_s
    span = max(0.0, obs.observe_s - dur)
    starts = sorted(span * i / max(1, n_streams - 1) for i in range(n_streams))
    blocks = list(layout.ids)
    random.Random(f"h21d:vblk:{seed}").shuffle(blocks)
    return [{"block": blocks[i % len(blocks)], "start_s": s,
             "work": "DISCHARGE" if i % 2 == 0 else "LOAD",
             "type_offset": i % 2}
            for i, s in enumerate(starts)]


def _retime_vessels(scn: TerminalScenario, starts: list[float]) -> TerminalScenario:
    """블록 시나리오의 본선 시작 시각을 지정값으로 옮긴다 — 계획·ETD·물리하한과
    **본선 연계 야드 작업의 release_time 까지** 함께 이동(desync 방지)."""
    if not scn.vessels:
        return scn
    deltas: dict[str, float] = {}
    out_v = []
    for v, s in zip(scn.vessels, sorted(starts)):
        p = v.plan
        d = s - p.planned_start_s
        deltas[v.vessel_id] = d
        out_v.append(dataclasses.replace(v, plan=dataclasses.replace(
            p, planned_start_s=s,
            planned_completion_s=(None if p.planned_completion_s is None
                                  else p.planned_completion_s + d),
            etd_s=None if p.etd_s is None else p.etd_s + d,
            phys_min_completion_s=(None if p.phys_min_completion_s is None
                                   else p.phys_min_completion_s + d))))
    out_j = []
    for j in scn.jobs:
        d = deltas.get(getattr(j, "vessel_id", None))
        if d:
            j = dataclasses.replace(j, release_time=j.release_time + d)
        out_j.append(j)
    return dataclasses.replace(scn, vessels=out_v, jobs=out_j)


def vessel_placement(layout: YardLayout, seed: int, params: TerminalStreamParams,
                     obs: ObservationContract) -> dict[str, dict]:
    """본선 process 의 (블록, 시작슬롯, 작업종류) **독립 추첨** — ★감사 정정 2026-08-09.

    구판(균등 allocate + 순번 슬롯)은 본선을 이름순 앞 12블록(Y01~Y12)에만 두고 시작
    시각도 블록 순서와 단조 상관이라, "뒷번호 블록 = 본선 없음·게이트 근접 = 이른 본선"
    이라는 인공 상관이 시드 구조에 새겨졌다 — 판매 정책이 배치 지름길("본선 없는 먼
    블록으로 밀기")을 학습해도 부하 평탄화와 구분이 안 된다. 세 축을 **서로 다른 전용
    난수열**로 독립 배정해 상관을 제거한다:
      · 블록 선택  `h21w:vsel`  — len(layout) 곳 중 vessels_total 곳 완전 추첨(중복 없음)
      · 시작 슬롯  `h21w:vslot` — 관측창 5%~95% 를 (n−1) 등분한 n 개 슬롯의 시드 순열.
        구판 분모 n 은 실제 5%~87.5% 로 문서(5%~95%)와 어긋났다 — 분모 n−1 로 계약 정정
        (계획 본선 작업률 ≈161→≈150 moves/h, 유도 앵커 145~170 안 유지).
      · 작업 종류  `h21w:vtype` — 양하 ⌈n/2⌉ / 적하 ⌊n/2⌋ 셔플(offset 0=양하·1=적하).
    반환: {블록: {slot_k, start_s, type_offset, work}} — 원장·자격 보고에 그대로 저장.
    """
    n = params.vessels_total
    if n == 0:
        return {}
    if n > len(layout.ids):
        raise ValueError(f"본선 process {n}개가 블록 수 {len(layout.ids)}를 넘는다 — "
                         "블록당 1 스트림 계약(초과 배치는 별도 등록)")
    blocks = sorted(random.Random(f"h21w:vsel:{seed}").sample(layout.ids, n))
    slots = list(range(n))
    random.Random(f"h21w:vslot:{seed}").shuffle(slots)
    offsets = [0] * ((n + 1) // 2) + [1] * (n // 2)
    random.Random(f"h21w:vtype:{seed}").shuffle(offsets)
    placement: dict[str, dict] = {}
    for i, b in enumerate(blocks):
        k = slots[i]
        placement[b] = {
            "slot_k": k,
            "start_s": obs.observe_s * (0.05 + 0.90 * (k / max(1, n - 1))),
            "type_offset": offsets[i],
            "work": "DISCHARGE" if offsets[i] == 0 else "LOAD"}
    return placement


def _clamp_travel(base: float, resid: float) -> float:
    return min(GATE_BLOCK_MAX_S, max(GATE_BLOCK_MIN_S, base + resid))


def _job_from_entry(e: dict, gate_in_s: float, *,
                    realized_gate_in_s: float | None = None) -> Job:
    """pool/채움 항목 → Job. 예측 도착 = **통지** gate-in + 기대 주행(누출 0).

    realized_gate_in_s (YR-162A): 준수오차 주입 시 실현 게이트인이 통지와 달라진다 —
    공개값(통지·예측·예약)은 **통지 기반 그대로**, 실현(actual_*)만 어긋난다.
    None(기본) = 오차 0 — 기존 완전 예측 조건과 바이트 동일(골든 불변).
    """
    realized = gate_in_s if realized_gate_in_s is None else realized_gate_in_s
    arr = realized + e["travel_s"]
    est = gate_in_s + e["travel_base_s"]          # 공개 예측 — 통지 기반(실현 미참조)
    if e["flow"] == "GATE_OUT":
        j = Job(job_id=e["job_id"], flow=JobFlow.GATE_OUT, release_time=0.0,
                actual_gate_in=realized, actual_block_arrival=arr, provided_eta=est,
                target_container=e["target"])
    else:
        j = Job(job_id=e["job_id"], flow=JobFlow.GATE_IN, release_time=0.0,
                actual_gate_in=realized, actual_block_arrival=arr, provided_eta=est,
                inbound_size=(ContainerSize.FT40 if e["size_ft40"]
                              else ContainerSize.FT20),
                inbound_load=LoadStatus.FULL)
    for key, val in (("estimated_block_arrival", est),
                     ("appointment_gate_time", gate_in_s),
                     ("notified_gate_in_s", gate_in_s),   # 공개 통지 시각 — 정책이 읽는 값
                     ("exit_travel_s", e["exit_travel_s"])):
        setattr(j, key, val)
    return j


def build_fixed_wip(profile: IntegratedProfile, seed: int, *,
                    wip_target: int,
                    obs: ObservationContract | None = None,
                    layout: YardLayout | None = None,
                    params: TerminalStreamParams | None = None,
                    pool_factor: int = WIP_POOL_FACTOR,
                    background_seed: int | None = None,
                    master_load: int | None = None) -> dict:
    """고정 재공량 계약의 시나리오 묶음 — 초기 채움 L 대 + 교체 투입용 대기 pool.

    · 초기 채움: `p` 배분으로 L 대를 [0, WIP_FILL_SPAN_S] 에 펼쳐 사전 배치(장부 활성화 겸).
    · 대기 pool: 속성(블록·flow·규격·반출대상·주행·출문)을 **전부 사전 추첨**한 항목열.
      런타임에는 투입 시각만 정해지고 무작위를 소비하지 않는다(결정론).
    · pool 의 반출 대상은 블록 초기 적재의 미사용 컨테이너에서 **중복 없이** 예약한다.
    · 본선 배치는 `vessel_placement`(블록·슬롯·종류 독립 추첨 — 감사 정정 2026-08-09).
    · background_seed(★32차 감사): 배경(초기 적재·본선 배치·반출대상 순서)의 시드를
      트럭 시드와 분리한다 — 부하 L 셀들이 **같은 배경**을 공유해야 L 효과가 배경
      변주와 교락하지 않는다. None 이면 기존과 동일하게 seed 를 그대로 쓴다.
    · master_load(★34차 감사 — 완전한 짝 비교): 트럭 명단(채움+pool 속성열)을
      master_load 기준으로 한 번 추첨하고 **앞부분만 잘라** 쓴다 — 같은 (seed,
      master) 에서 L100 은 L150 세계의 접두(같은 트럭·같은 순서)가 되어 "트럭 수만
      바꾼 비교"가 성립한다. 구판(L 별 재추첨)은 트럭 구성까지 달라져 인과 주장
      불가(감사 지적). None = 기존 동작(골든 불변). 채움 투입 간격은 L 계약값
      (WIP_FILL_SPAN/L)을 따른다 — 명단이 정본, 간격은 계약.
    """
    obs = obs or ObservationContract()
    layout = layout or terminal_layout()
    params = params or TerminalStreamParams(load_4h=wip_target)
    if wip_target < len(layout.ids):
        raise ValueError(f"WIP 목표 {wip_target}가 블록 수 {len(layout.ids)}보다 작다")
    bseed = seed if background_seed is None else background_seed

    # ① 배경(초기 적재·본선) — 배치 세 축은 vessel_placement 가 독립 추첨한다.
    placement = vessel_placement(layout, bseed, params, obs)
    per_block_vessels = {b: (1 if b in placement else 0) for b in layout.ids}
    scns: dict[str, TerminalScenario] = {}
    for b in layout.ids:
        pl = placement.get(b)
        bg = _background(profile, bseed + 1000 * (layout.ids.index(b) + 1), b, obs,
                         params, per_block_vessels[b],
                         vessel_type_offset=(pl["type_offset"] if pl else 0))
        if bg.vessels:
            bg = _shift_vessels(bg, pl["start_s"] - bg.vessels[0].plan.planned_start_s)
        scns[b] = bg

    # ② 반출 대상 후보 (배경 본선 사용분 제외) — 초기 채움과 pool 이 순서대로 소비
    free: dict[str, list[str]] = {}
    for b, scn in scns.items():
        used = {j.target_container for j in scn.jobs if j.target_container}
        cand = sorted(set(scn.containers) - used)
        random.Random(f"h21w:tgt:{bseed}:{b}").shuffle(cand)
        free[b] = cand

    p = distribution_vector(layout, params)
    # ★난수열 분리(감사 2026-08-09): flow 와 규격이 한 스트림(mix)을 나눠 쓰면 한 축의
    # 소비가 다른 축을 밀어낸다 — 축별 전용 스트림으로 분리(구 h21w:mix 폐기).
    flow_rng = random.Random(f"h21w:flow:{seed}")
    size_rng = random.Random(f"h21w:size:{seed}")
    exit_rng = random.Random(f"h21w:exit:{seed}")
    resid_rng = random.Random(f"h21w:resid:{seed}")
    flow_fallbacks: dict[str, int] = {b: 0 for b in layout.ids}

    def draw(bid: str, jid: str) -> dict:
        """트럭 1대 속성 사전 추첨 — 투입 시각과 무관하게 고정된다.

        ★fallback 원장(감사 2026-08-09): 반출을 원했는데(추첨) 블록 반출 재고가 없어
        반입으로 바뀐 건은 requested/realized 를 나눠 기록한다 — 조용한 혼합비 드리프트
        금지. 공식 자격시험(yr150_h21_wip_pilot W9)은 fallback 0 을 요구한다.
        """
        sr = params.resid_travel_sigma_s
        resid = (max(-2 * sr, min(2 * sr, resid_rng.gauss(0.0, sr))) if sr > 0 else 0.0)
        want_out = flow_rng.random() < params.gate_out_share
        out = want_out and bool(free[bid])
        if want_out and not out:
            flow_fallbacks[bid] += 1
        return {"job_id": jid, "block": bid,
                "flow": "GATE_OUT" if out else "GATE_IN",
                "requested_flow": "GATE_OUT" if want_out else "GATE_IN",
                "fallback_reason": ("no_free_target" if (want_out and not out)
                                    else None),
                "target": free[bid].pop() if out else None,
                "size_ft40": size_rng.random() < params.size_mix_ft40,
                "travel_s": _clamp_travel(layout.gate_to_block_s(bid), resid),
                "travel_base_s": layout.gate_to_block_s(bid),
                "exit_travel_s": trunc_normal(exit_rng, params.exit_travel_mu_s,
                                              0.12, lo=60.0)}

    # ③ 초기 채움 — ★34차: master 명단(속성열)을 뽑고 앞 wip_target 대만 쓴다.
    m = wip_target if master_load is None else master_load
    if m < wip_target:
        raise ValueError(f"master_load {m} < wip_target {wip_target}")
    master_counts = allocate(p, m)
    slots = [b for b in layout.ids for _ in range(master_counts[b])]
    random.Random(f"h21w:fill:{seed}").shuffle(slots)
    master_fill = [draw(bid, f"{bid}:F-{i:05d}") for i, bid in enumerate(slots)]
    fill_ledger = []
    fill_jobs: dict[str, list[Job]] = {b: [] for b in layout.ids}
    used_fill = master_fill[:wip_target]
    counts = {b: 0 for b in layout.ids}
    for i, e in enumerate(used_fill):
        bid = e["block"]
        # The common prefix must keep the same event time across load cells.
        # Dividing by wip_target made 99/100 shared jobs occur at different
        # times in L100 and L150, invalidating the paired comparison.
        gate_in = WIP_FILL_SPAN_S * i / max(1, m)
        counts[bid] += 1
        fill_jobs[bid].append(_job_from_entry(e, gate_in))
        # 32차 감사(부채 2): fill 도 requested/realized 를 원장에 남긴다 — 공식 자격
        # 결과가 "실제 투입된 작업"의 fallback 을 작업 단위로 복기할 수 있어야 한다.
        fill_ledger.append({"job_id": e["job_id"], "block": bid, "flow": e["flow"],
                            "requested_flow": e["requested_flow"],
                            "fallback_reason": e["fallback_reason"],
                            "gate_in_s": round(gate_in, 3),
                            "travel_s": round(e["travel_s"], 3)})

    # ④ 교체 pool — master 순서열의 접두 (블록 배분 비율은 master 기준)
    pool_counts = allocate(p, m * pool_factor)
    pool_slots = [b for b in layout.ids for _ in range(pool_counts[b])]
    random.Random(f"h21w:pool:{seed}").shuffle(pool_slots)
    master_pool = [draw(bid, f"{bid}:W-{i:05d}") for i, bid in enumerate(pool_slots)]
    pool = master_pool[:wip_target * pool_factor]

    for b in layout.ids:
        scns[b] = dataclasses.replace(
            scns[b], jobs=scns[b].jobs + fill_jobs[b],
            meta={**scns[b].meta, "h21_block": b, "h21_wip_target": wip_target,
                  "h21_mode": "fixed_wip", "observation": obs.as_dict()})
    return {"scenarios": scns, "pool": pool, "fill": fill_ledger, "p": p,
            "counts": counts, "master_counts": master_counts,
            "wip_target": wip_target,
            "observation": obs.as_dict(), "layout": layout.as_dict(),
            "vessels_per_block": per_block_vessels,
            "vessel_placement": {b: {**pl, "start_s": round(pl["start_s"], 3)}
                                 for b, pl in placement.items()},
            "flow_fallbacks": flow_fallbacks,
            "flow_fallbacks_total": sum(flow_fallbacks.values()),
            "master_load": m,
            "mode": "fixed_wip",
            "fairness_note": "고정 WIP: 정책별 유입량이 달라진다 — 성능은 처리량과 "
                             "시간당 비용의 공동 판정만 허용(에피소드 총비용 단독 금지)"}


def hotspot_rotation(layout: YardLayout, seed: int, n: int = 4) -> tuple[str, ...]:
    """YR-157 — hotspot 블록 시드 추첨 (사용자 확정 2026-08-09: 몰아주기 주축 6셀).

    특정 블록이 늘 hotspot 이 되는 운을 제거한다(본선 배치 독립화와 같은 원리).
    전용 난수열(`h21w:hspot`)이라 본선(vsel)·트럭 열과 독립 — hotspot 이 본선 블록과
    겹칠 수 있고 그것이 자연스러운 표본이다(겹침 배제는 인공 규칙).
    """
    if not (1 <= n <= len(layout.ids)):
        raise ValueError(f"hotspot 수 {n}는 1~{len(layout.ids)} 여야 한다")
    return tuple(sorted(random.Random(f"h21w:hspot:{seed}").sample(layout.ids, n)))


# ================================================== 5차 계약 빌더 (도착률·24시간)
OBS_24H = ObservationContract(warmup_s=7_200.0, measure_s=79_200.0,
                              snapshot_s=300.0)     # 사전등록 §E (워밍업 2h + 측정 22h)


def build_diurnal(profile: IntegratedProfile, seed: int, *,
                  obs: ObservationContract | None = None,
                  layout: YardLayout | None = None,
                  params: TerminalStreamParams | None = None,
                  day_total: int = DIURNAL_DAY_TOTAL,
                  n_streams: int = DIURNAL_VESSEL_STREAMS,
                  background_seed: int | None = None) -> dict:
    """5차 계약 — **도착 명단 사전 확정** + 24시간 배경(초기 적재·본선 30스트림 교대).

    · 트럭: `diurnal_arrivals` 로 하루 3,600대의 **도착 시각**을 확정하고 블록·flow·
      규격·주행을 사전 추첨한다(런타임 무작위 소비 0). 통지 시각 = 도착 − lead 이며
      스케줄러가 그 시각에 엔진으로 투입한다(피드백 없음 = 개방 루프).
    · 초기 채움(fill) 개념이 없다 — 야간 저부하에서 시작해 명단대로 채워진다.
    · 배경 시드 분리(background_seed): 여러 셀이 같은 배경을 공유하게 한다.
    """
    obs = obs or OBS_24H
    layout = layout or terminal_layout()
    params = params or TerminalStreamParams(load_4h=day_total)
    bseed = seed if background_seed is None else background_seed

    # ① 배경 — 본선 교대 배치(블록별 개수만큼 생성 후 지정 시각으로 이동)
    sched = vessel_schedule_24h(layout, bseed, params, obs, n_streams)
    per_block: dict[str, list[dict]] = {}
    for s in sched:
        per_block.setdefault(s["block"], []).append(s)
    scns: dict[str, TerminalScenario] = {}
    for b in layout.ids:
        rows = sorted(per_block.get(b, []), key=lambda r: r["start_s"])
        bg = _background(profile, bseed + 1000 * (layout.ids.index(b) + 1), b, obs,
                         params, len(rows),
                         vessel_type_offset=(rows[0]["type_offset"] if rows else 0))
        if rows:
            bg = _retime_vessels(bg, [r["start_s"] for r in rows])
        scns[b] = bg

    # ② 반출 대상 후보 (배경 본선 사용분 제외)
    free: dict[str, list[str]] = {}
    for b, scn in scns.items():
        used = {j.target_container for j in scn.jobs if j.target_container}
        cand = sorted(set(scn.containers) - used)
        random.Random(f"h21d:tgt:{bseed}:{b}").shuffle(cand)
        free[b] = cand

    # ③ 트럭 명단 — 도착 시각(이중 피크) × 블록 배분 × 속성 사전 추첨
    p = distribution_vector(layout, params)
    times = diurnal_arrivals(seed, day_s=obs.observe_s, total=day_total)
    counts = allocate(p, day_total)
    slots = [b for b in layout.ids for _ in range(counts[b])]
    random.Random(f"h21d:assign:{seed}").shuffle(slots)
    flow_rng = random.Random(f"h21d:flow:{seed}")
    size_rng = random.Random(f"h21d:size:{seed}")
    exit_rng = random.Random(f"h21d:exit:{seed}")
    resid_rng = random.Random(f"h21d:resid:{seed}")
    fallbacks = {b: 0 for b in layout.ids}
    schedule = []
    for i, (t, bid) in enumerate(zip(times, slots)):
        sr = params.resid_travel_sigma_s
        resid = (max(-2 * sr, min(2 * sr, resid_rng.gauss(0.0, sr))) if sr > 0 else 0.0)
        want_out = flow_rng.random() < params.gate_out_share
        out = want_out and bool(free[bid])
        if want_out and not out:
            fallbacks[bid] += 1
        schedule.append({
            "job_id": f"{bid}:D-{i:05d}", "block": bid,
            "arrival_s": round(t, 3),
            "flow": "GATE_OUT" if out else "GATE_IN",
            "requested_flow": "GATE_OUT" if want_out else "GATE_IN",
            "fallback_reason": ("no_free_target" if (want_out and not out) else None),
            "target": free[bid].pop() if out else None,
            "size_ft40": size_rng.random() < params.size_mix_ft40,
            "travel_s": _clamp_travel(layout.gate_to_block_s(bid), resid),
            "travel_base_s": layout.gate_to_block_s(bid),
            "exit_travel_s": trunc_normal(exit_rng, params.exit_travel_mu_s,
                                          0.12, lo=60.0)})

    for b in layout.ids:
        scns[b] = dataclasses.replace(scns[b], meta={
            **scns[b].meta, "h21_block": b, "h21_mode": "diurnal_24h",
            "h21_day_total": day_total, "observation": obs.as_dict()})
    return {"scenarios": scns, "schedule": schedule, "p": p, "counts": counts,
            "vessel_schedule": sched, "day_total": day_total,
            "flow_fallbacks": fallbacks,
            "flow_fallbacks_total": sum(fallbacks.values()),
            "observation": obs.as_dict(), "layout": layout.as_dict(),
            "mode": "diurnal_24h",
            "fairness_note": "도착률 계약: 전 정책이 동일 명단·동일 시각을 받는다 — "
                             "이득은 재공량·대기 감소로 나타난다(처리량은 동일)"}


class ScheduledAnnouncer:
    """명단대로 투입하는 개방 루프 스케줄러 (5차 계약) — 전역 계수를 보지 않는다.

    통지 시각 = 도착 − lead 에 엔진으로 투입하고, 엔진 안에서 도착 시각에 gate-in 이
    실현된다. 피드백이 없으므로 **정책이 무엇을 하든 같은 트럭이 같은 시각에** 온다.
    """

    def __init__(self, schedule: list[dict], *, lead_s: float,
                 end_s: float | None = None,
                 period_s: float = WIP_ADMISSION_PERIOD_S):
        self.lead_s = lead_s
        self.period_s = period_s
        self.end_s = end_s
        self.by_epoch: dict[float, list[dict]] = {}
        for e in schedule:
            notify = max(0.0, e["arrival_s"] - lead_s)
            slot = round((notify // period_s) * period_s, 6)
            self.by_epoch.setdefault(slot, []).append(e)
        self.n_admitted = 0
        self.cursor = 0
        self.ledger: list[dict] = []

    def review(self, mbt, t: float) -> None:
        from .multiblock import TransferError
        if not on_grid(t, self.period_s):
            return
        for e in self.by_epoch.get(round(t, 6), []):
            arr = e["arrival_s"]
            if self.end_s is not None and arr + e["travel_s"] > self.end_s:
                self.ledger.append({"t": t, "event": "SKIP_TAIL",
                                    "job_id": e["job_id"]})
                continue
            job = _job_from_entry(e, arr)
            try:
                mbt.admit_external_job(e["block"], job, gate_in_s=arr,
                                       travel_s=e["travel_s"])
                self.n_admitted += 1
                self.cursor += 1
                self.ledger.append({"t": t, "event": "ADMIT",
                                    "job_id": e["job_id"], "block": e["block"],
                                    "flow": e["flow"], "arrival_s": arr})
            except TransferError as ex:
                self.ledger.append({"t": t, "event": "SKIP",
                                    "job_id": e["job_id"], "reason": str(ex)})
        self.ledger.append({"t": t, "event": "EPOCH"})


def admission_epochs(obs: ObservationContract,
                     period_s: float = WIP_ADMISSION_PERIOD_S) -> tuple[float, ...]:
    """투입 검토 시각열 — 0 부터 관측 종료까지 등간격."""
    n = int(obs.observe_s // period_s)
    return tuple(i * period_s for i in range(n + 1))


class WipAdmissionController:
    """**60초 격자** review epoch 마다 내부 대수를 세고 부족분만큼 pool 에서 투입한다.

    격자 밖 호출(gate-in 유발 epoch)은 무시한다 — on_grid 가드 (감사 2026-08-09).

    · 내부 = 시간 장부의 A ≤ t < O (외부트럭만). lead>0 이면 투입 확정·미진입(pipeline)
      도 목표에 포함해 과잉 투입을 막는다.
    · 투입 불가 항목(용량·대상 부재·창 밖)은 **재시도 없이 건너뛰고 전량 기록**한다
      (조용한 누락 금지 — 결정론 유지).
    · pool 소진은 실패로 숨기지 않고 원장에 남긴다.
    """

    def __init__(self, pool: list[dict], *, wip_target: int,
                 lead_s: float = 0.0, max_tries_per_epoch: int = 200,
                 end_s: float | None = None,
                 period_s: float = WIP_ADMISSION_PERIOD_S,
                 adherence_sigma_s: float = 0.0):
        self.pool = pool
        self.wip_target = wip_target
        self.lead_s = lead_s
        self.max_tries = max_tries_per_epoch
        # YR-162A — 예약 준수오차 σ: 실현 게이트인 = 통지 + ε(±2σ 절단·작업별 전용
        # 난수열 h21w:adh:{job_id} — 실행 순서 무관 결정론). 공개값(통지·예측)은 통지
        # 기반 그대로(정보경계 불변). 기본 0 = 완전 예측(골든 불변).
        self.adherence_sigma_s = adherence_sigma_s
        # 32차 감사(부채 4): 격자 가드 주기를 인스턴스가 보유 — admission_epochs 와
        # 컨트롤러가 **같은 period 를 받아야** 한다(기본값이 같은 단일 상수라 기본
        # 구성은 자동 일치. 한쪽만 바꾸는 것은 계약 위반).
        self.period_s = period_s
        # 관측 종료 — 도착이 창을 벗어날 투입은 처음부터 하지 않는다(꼬리 pool 낭비 방지).
        # 따라서 **마지막 GATE_BLOCK_MAX_S(7분) 동안은 WIP 가 자연 감소**한다 — 유지 판정은
        # 이 꼬리를 제외한 구간에서 한다(하네스 계약).
        self.end_s = end_s
        self.cursor = 0
        self.n_admitted = 0
        self.ledger: list[dict] = []
        self.exhausted_at: float | None = None
        self._tail_stopped = False

    @staticmethod
    def wip_now(mbt, t: float) -> tuple[int, int]:
        """(내부, pipeline) — 시간 장부 기준 A≤t<O 와 투입확정·미진입."""
        inside = pipeline = 0
        for sim in mbt.blocks.values():
            tl = getattr(sim, "time_ledger", None)
            if tl is None:
                continue
            for r in tl.records.values():
                if r.gate_in > t + 1e-9:
                    pipeline += 1
                elif r.gate_out is None or r.gate_out > t:
                    inside += 1
        return inside, pipeline

    def review(self, mbt, t: float) -> None:
        from .multiblock import TransferError
        # ★격자 가드(감사 2026-08-09): 엔진 review epoch 은 격자+gate-in 합집합이라
        # 가드 없이는 투입 판단이 주기 계약을 벗어난다(판매 검토와 같은 결함 —
        # 치명 5 의 투입 쪽 잔여). 격자 밖 호출은 전부 무시한다.
        if not on_grid(t, self.period_s):
            return
        if self.end_s is not None and t + self.lead_s + GATE_BLOCK_MAX_S > self.end_s:
            if not self._tail_stopped:
                self._tail_stopped = True
                self.ledger.append({"t": t, "event": "TAIL_STOP",
                                    "note": "도착이 관측창을 벗어날 투입 중단"})
            return
        inside, pipeline = self.wip_now(mbt, t)
        deficit = self.wip_target - inside - pipeline
        admitted = tries = 0
        while admitted < deficit and tries < self.max_tries:
            if self.cursor >= len(self.pool):
                if self.exhausted_at is None:
                    self.exhausted_at = t
                    self.ledger.append({"t": t, "event": "POOL_EXHAUSTED"})
                break
            e = self.pool[self.cursor]
            self.cursor += 1
            tries += 1
            notified = t + self.lead_s
            realized = notified
            if self.adherence_sigma_s > 0:
                sr = self.adherence_sigma_s
                eps = max(-2 * sr, min(2 * sr, random.Random(
                    f"h21w:adh:{e['job_id']}").gauss(0.0, sr)))
                realized = max(t, notified + eps)   # 실현이 투입 결정을 앞설 수 없다
            job = _job_from_entry(e, notified, realized_gate_in_s=realized)
            try:
                mbt.admit_external_job(e["block"], job, gate_in_s=realized,
                                       travel_s=e["travel_s"])
                admitted += 1
                self.n_admitted += 1
                self.ledger.append({"t": t, "event": "ADMIT", "job_id": e["job_id"],
                                    "block": e["block"], "flow": e["flow"],
                                    "eps_s": round(realized - notified, 3)})
            except TransferError as ex:
                self.ledger.append({"t": t, "event": "SKIP", "job_id": e["job_id"],
                                    "block": e["block"], "reason": str(ex)})
        # 32차 감사: EPOCH 은 **모든 격자 시각**에 기록한다 — 유지 판정(W1)의 관측
        # 대상이 "내부+확약(pipeline)=L" 계약량이므로 조용한 epoch 도 증거여야 한다.
        self.ledger.append({"t": t, "event": "EPOCH", "inside": inside,
                            "pipeline": pipeline, "deficit": deficit,
                            "admitted": admitted})
