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

■ 부하 L 의 뜻 — **고정 유입량**이지 고정 재공량이 아니다
`L ∈ {50,75,100,125,150}` 은 **21블록 전체 합계의 4시간당 명목 예약 도착량**이다. 블록별
물량이 아니고 혼잡등급 이름도 아니다. 관측창이 4시간보다 길면 같은 **도착률**로 늘린다.

  · 채택 = **고정 유입량**: 4시간 동안 총 L 대가 도착한다. 터미널 안 대수는 **결과로 측정**.
  · 폐기 = 고정 재공량(WIP): 터미널 안에 항상 L 대가 있도록 보충하는 방식. 빨리 처리하는
    정책일수록 트럭을 더 받아 **정책별 입력량이 달라지므로** 공정 비교가 깨진다.

여기서 만드는 것은 **예약(appointment) 시각**이고, 실제 gate-in 은 예약 준수오차만큼
어긋난다 — 그래서 측정창 안 실제 진입 수는 L 과 정확히 같지 않다.

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
    fill_ratio: float = 0.30
    vessels_total: int = 6                      # 터미널 전체 본선 process 수
    vessel_moves: int = 15
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
                n_vessels: int) -> TerminalScenario:
    """블록의 **배경**(초기 적재·본선)만 만든다 — 외부트럭은 master stream 이 준다."""
    gp = TerminalGenParams(
        n_external=1,                    # 최소 1 — 아래에서 제거한다(배경만 쓴다)
        gate_out_share=0.0,              # 배경 트럭은 반출 대상을 소비하지 않게
        n_vessels=n_vessels, vessel_moves=params.vessel_moves,
        fill_ratio=params.fill_ratio, horizon_s=obs.observe_s, drain_window_s=1.0,
        size_mix_ft40=params.size_mix_ft40,
        sts_move_interval_s=params.sts_move_interval_s,
        gaussian=False,                  # 블록별 변주는 seed 로만 — 물량은 계약값 고정
        time_contract_v2=True, gate_block_contract=True,
        vessel_deadline_achievable=True)
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
                         params, per_block_vessels[b])
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
