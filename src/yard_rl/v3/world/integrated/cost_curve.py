"""YR-123 — 공통 지연 한계비용 곡선 API (본선·트럭 단일 numeraire).

목적: "이 작업의 처리를 Δt 늦추면 추가로 얼마가 드는가"를 작업 유형과 무관하게 같은
기준재 단위(트럭 대기 1h = 1.0)로 반환한다. BlockQ-v3 의 Q 학습표적·YR-133 판매/수용
견적(OutRelief/InBurden)의 공용 원료 — 단 YR-132 식 "미완 개수" 보정 용도는 금지(7차).

계약 (spec YR-123):
 ① 같은 기준시각(sim.now)·단위(numeraire)·평가창(sim.end 검열 캡) — 호출자 간 공유.
 ② 공개정보만: 도착한 트럭 = 관측 대기, 미도착 = provided_eta 만. actual_* 미열람.
 ③ 유형별 내부 계산은 달라도 출력 단위는 하나. 원시 종류/마감의 상태 노출 여부는
    YR-124 의 몫 (이 모듈은 계산기일 뿐).
 ④ 값 + 불확실성 대역(lo·hi) + 급증 시작 지연(escalation_start_s) 반환.
 ⑤ 단조성·기울기·단위변환·결측·결정론은 tests/integrated/test_yr123_cost_curve.py 가 고정.

유형별 정의 (전부 현행 동결 비용계약에서 유도 — 새 가격 발명 없음):
 · 외부트럭: 대기 w_truck/h + SLA 초과분 w_long/h 가산 (기준재 1.0 + 1.0). 급증 시작 =
   남은 SLA 여유. 미도착·ETA 有 = 같은 곡선(하한 0 — 도착 전이라 실현 불확실),
   미도착·ETA 無 = 값 0·상한만(최악 = 즉시 SLA 초과 대기).
 · 본선 LOAD: 서빙을 Δt 늦춘 공급 반사실 — vessel_cost 의 paired anchor 규약 그대로
   (후보 독립 anchor·pace 상계·검열 캡). 불확실성 대역 = [hinge(κ=0), softplus(κ)].
   ρ_vessel=33 assumed — ×⅓~3 민감도는 호출자 보고 의무(기존 계약 승계).
 · 본선 DISCHARGE: 0 — 현 비용계약에서 양하는 크레인 레버 0 (yard_handover_cap=None
   감사 실측) + sts/transfer_wait 가중 0 (선석초과 proxy 흡수).
 · 기타(내부 등): 0 — 현 계약에 과금 항 없음 (basis 에 명시).
"""
from __future__ import annotations

from dataclasses import dataclass

from .block_congestion import SVC_REF_S
from .cost_config import RewardCalculator
from .vessel import VesselWorkType
from .vessel_cost import KAPPA_S, _gap_s, completion_cost, load_supply_state

DTS_DEFAULT = (60.0, 300.0, 600.0)      # 피드백 지정 1/5/10분
ESCALATION_EPS = 0.05                   # 급증 시작 판정 문턱 [numeraire] (assumed)
_SCAN_STEP_S = 60.0


@dataclass(frozen=True)
class DelayCostPoint:
    dt_s: float
    cost: float          # 지금 처리 대비 Δt 지연의 추가 비용 [numeraire]
    lo: float            # 불확실성 하한
    hi: float            # 상한


@dataclass(frozen=True)
class DelayCostCurve:
    job_id: str
    kind: str            # truck | truck_eta | truck_no_eta | vessel_load | vessel_discharge | other
    points: tuple[DelayCostPoint, ...]
    escalation_start_s: float | None     # 한계비용이 유의미해지는 최초 지연 (없으면 None)
    basis: str


def _truck_point(waited_s: float, sla_s: float, dt_s: float, cap_s: float,
                 w_wait: float, w_long: float) -> float:
    """외부트럭 순수 곡선 — 대기 w_wait/h + SLA 초과분 w_long/h. 창 검열 캡."""
    d = max(0.0, min(dt_s, cap_s))
    slack = max(0.0, sla_s - waited_s)
    over = max(0.0, d - slack)
    return (w_wait * d + w_long * over) / 3600.0


def _vessel_point(sim, v, dt_s: float, *, dur_s: float, kappa_s: float) -> float:
    """본선 LOAD 순수 곡선 — 서빙 공급을 Δt 늦춘 paired 반사실 (vessel_cost 규약)."""
    gap = _gap_s(sim)
    cad = v.plan.sts_move_interval_s
    t_ref = sim.now + cad + gap
    pace = max(cad, gap)
    w0 = load_supply_state(sim, v, supply_etas=(sim.now + dur_s + gap,),
                           steady_onset_s=t_ref, steady_pace_s=pace)
    w1 = load_supply_state(sim, v, supply_etas=(sim.now + dt_s + dur_s + gap,),
                           steady_onset_s=t_ref, steady_pace_s=pace)
    if w0 is None or w1 is None:
        return 0.0
    return max(0.0, completion_cost(w1, kappa_s=kappa_s)
               - completion_cost(w0, kappa_s=kappa_s))


def _escalation(fn, max_dt_s: float) -> float | None:
    t = _SCAN_STEP_S
    while t <= max_dt_s + 1e-9:
        if fn(t) >= ESCALATION_EPS:
            return t
        t += _SCAN_STEP_S
    return None


def delay_cost_curve(sim, job_id: str, dts: tuple[float, ...] = DTS_DEFAULT,
                     rc: RewardCalculator | None = None) -> DelayCostCurve:
    """작업 하나의 지연 한계비용 곡선 — 공개정보만·numeraire·결정론."""
    rc = rc or RewardCalculator.numeraire_v1()
    w = rc.config.weight()
    j = sim.jobs[job_id]
    cap = max(0.0, float(sim.end) - sim.now)
    max_dt = max(dts)

    if getattr(j, "is_external_truck", False):
        sla = float(sim.profile.long_wait_sla_s)
        wt, wl = w["truck_wait"], w["long_wait"]
        arrived = j.status.name != "PLANNED"
        if arrived:
            waited = max(0.0, sim.cum_wait(job_id))
            f = lambda d: _truck_point(waited, sla, d, cap, wt, wl)
            pts = tuple(DelayCostPoint(d, c := f(d), c, c) for d in dts)
            esc = max(0.0, sla - waited)
            return DelayCostCurve(job_id, "truck", pts,
                                  esc if esc <= max_dt and cap > 0 else None,
                                  f"관측 대기 {waited:.0f}s·SLA {sla:.0f}s·"
                                  f"w={wt}/{wl} — 창 검열 {cap:.0f}s")
        eta = getattr(j, "provided_eta", None)
        if eta is not None:
            f = lambda d: _truck_point(0.0, sla, d, cap, wt, wl)
            pts = tuple(DelayCostPoint(d, c := f(d), 0.0, c) for d in dts)
            return DelayCostCurve(job_id, "truck_eta", pts,
                                  sla if sla <= max_dt else None,
                                  "미도착(공개 ETA 기반) — 도착 전이라 하한 0")
        f_hi = lambda d: _truck_point(sla, sla, d, cap, wt, wl)   # 최악: 즉시 SLA 초과
        pts = tuple(DelayCostPoint(d, 0.0, 0.0, f_hi(d)) for d in dts)
        return DelayCostCurve(job_id, "truck_no_eta", pts, None,
                              "ETA 결측 — 청구 추정 0·상한만 (fail-closed)")

    vid = getattr(j, "vessel_id", None)
    if vid is not None and getattr(j, "flow", None) is not None:
        v = sim.vessels.get(vid)
        if v is not None and j.flow.name == "VESSEL_LOAD" \
                and v.work_type == VesselWorkType.LOAD and not v.done:
            dur = float(SVC_REF_S)          # 표준 서비스 가정 (assumed — basis 명시)
            f = lambda d: _vessel_point(sim, v, d, dur_s=dur, kappa_s=KAPPA_S)
            f0 = lambda d: _vessel_point(sim, v, d, dur_s=dur, kappa_s=0.0)
            pts = []
            for d in dts:
                c, c0 = f(d), f0(d)
                pts.append(DelayCostPoint(d, c, min(c, c0), max(c, c0)))
            return DelayCostCurve(job_id, "vessel_load", tuple(pts),
                                  _escalation(f, max_dt),
                                  f"공급 반사실(paired anchor)·dur={dur:.0f}s assumed·"
                                  "ρ_vessel=33 assumed(민감도 ×⅓~3 보고 의무)")
        if j.flow.name == "VESSEL_DISCHARGE":
            pts = tuple(DelayCostPoint(d, 0.0, 0.0, 0.0) for d in dts)
            return DelayCostCurve(job_id, "vessel_discharge", pts, None,
                                  "현 비용계약에서 양하 지연의 과금 레버 0 (감사 실측)")

    pts = tuple(DelayCostPoint(d, 0.0, 0.0, 0.0) for d in dts)
    return DelayCostCurve(job_id, "other", pts, None, "현 비용계약에 해당 과금 항 없음")


def marginal_delay_cost(sim, job_id: str, dt_s: float,
                        rc: RewardCalculator | None = None) -> DelayCostPoint:
    """단일 Δt 조회 — delay_cost_curve 와 같은 정의의 1점."""
    return delay_cost_curve(sim, job_id, (float(dt_s),), rc).points[0]
