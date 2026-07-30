"""YR-136 v2.1 — softplus 확률 지연비용 계약 (opt-in v2 — v1 cost_curve 는 바이트 불변).

계약 (spec YR-136 v2.1 — 결정 근거·수식·앵커는 spec 이 정본):
  트럭:  J_T(Ô) = (Ô−A)/3600 + (κ_T/3600)·sp((Ô−D_T)/κ_T),  r_T = 1 + σ(·) ∈ (1,2)
  본선:  J_V(F̂) = (10·κ_V/3600)·sp((F̂−P)/κ_V),             r_V = 10·σ(·) ∈ (0,10)
  후보 비교 = ΔC = J(예측+Δt) − J(예측).  해석: 시간당 비용률 = 기본 + 초과확률×벌점.
  κ·b(중심 보정)는 0단계 적합 후 kappa_fit.json 으로 동결 — 판정런 내 조정 금지.

정보 경계: Ô 예측기는 공개정보만 (크레인 잔여·대기 큐·표준서비스 180s·출문주행 평균
300s — 작업별 exit_travel_s 실현값 미열람). F̂ = vessel_cost 공개 공급 fold 재사용.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .block_congestion import SVC_REF_S
from .cost_curve import DelayCostCurve, DelayCostPoint, DTS_DEFAULT
from .scenario_gen import GATE_BLOCK_MEAN_S
from .vessel import VesselWorkType
from .vessel_cost import load_supply_state, projected_completion_s

RHO_VESSEL_V2 = 10.0
# L_T 유도 앵커 (새 상수 발명 없음): 입문 300 + 대기 SLA(프로파일) + 서비스 180 + 출문 300
KAPPA_PATH = Path("outputs/reports/yr136_softplus_contract/kappa_fit.json")


def sp(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return 0.0
    return math.log1p(math.exp(x))


def sigma(x: float) -> float:
    if x > 30.0:
        return 1.0
    if x < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class KappaFit:
    """0단계 적합 산출물 — 위치(b)·척도(κ) [초]. 동결 후 판정런 내 조정 금지."""
    kappa_t_s: float
    bias_t_s: float
    kappa_v_s: float
    bias_v_s: float
    n_truck: int = 0
    n_vessel: int = 0

    @classmethod
    def load(cls, path: Path = KAPPA_PATH) -> "KappaFit":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(kappa_t_s=d["kappa_t_s"], bias_t_s=d["bias_t_s"],
                   kappa_v_s=d["kappa_v_s"], bias_v_s=d["bias_v_s"],
                   n_truck=d.get("n_truck", 0), n_vessel=d.get("n_vessel", 0))


# ---------------------------------------------------------------- 순수 비용 함수
def j_truck(o_hat_s: float, a_s: float, d_target_s: float, kappa_s: float) -> float:
    return (o_hat_s - a_s) / 3600.0 + (kappa_s / 3600.0) * sp((o_hat_s - d_target_s) / kappa_s)


def r_truck(o_hat_s: float, d_target_s: float, kappa_s: float) -> float:
    return 1.0 + sigma((o_hat_s - d_target_s) / kappa_s)


def delta_cost_truck(o_hat_s: float, a_s: float, d_target_s: float,
                     dt_s: float, kappa_s: float) -> float:
    return j_truck(o_hat_s + dt_s, a_s, d_target_s, kappa_s) \
        - j_truck(o_hat_s, a_s, d_target_s, kappa_s)


def j_vessel(f_hat_s: float, p_s: float, kappa_s: float,
             rho: float = RHO_VESSEL_V2) -> float:
    return (rho * kappa_s / 3600.0) * sp((f_hat_s - p_s) / kappa_s)


def r_vessel(f_hat_s: float, p_s: float, kappa_s: float,
             rho: float = RHO_VESSEL_V2) -> float:
    return rho * sigma((f_hat_s - p_s) / kappa_s)


def delta_cost_vessel(f_hat_s: float, p_s: float, dt_s: float, kappa_s: float,
                      rho: float = RHO_VESSEL_V2) -> float:
    return j_vessel(f_hat_s + dt_s, p_s, kappa_s, rho) - j_vessel(f_hat_s, p_s, kappa_s, rho)


# ---------------------------------------------------------------- 예측기 (공개정보만)
def truck_target_s(sim, a_s: float) -> float:
    """D_T = A + L_T. L_T = 입문 300 + 대기 SLA + 서비스 180 + 출문 300 (유도 앵커)."""
    return a_s + GATE_BLOCK_MEAN_S + float(sim.profile.long_wait_sla_s) \
        + SVC_REF_S + GATE_BLOCK_MEAN_S


def predict_gate_out(sim, job_id: str) -> float | None:
    """Ô — 도착 트럭: now + (크레인 잔여 + 180×선행 대기)/크레인수 + 180 + 300.
    미도착: 도착시각 = provided_eta·현재 큐 proxy (고지). ETA 결측 = None (fail-closed)."""
    j = sim.jobs[job_id]
    if not getattr(j, "is_external_truck", False):
        return None
    crane_ids = [c.crane_id for c in sim.profile.cranes]
    now = sim.now
    run_rem = sum(max(0.0, sim.fleet.get(c).state.available_at - now) for c in crane_ids)
    arrived = j.status.name != "PLANNED"
    if arrived:
        my_arr = j.actual_block_arrival if j.actual_block_arrival is not None else now
        ahead = sum(1 for o in sim.jobs.values()
                    if o.is_external_truck and o.status.name == "WAITING"
                    and o.job_id != job_id
                    and (o.actual_block_arrival if o.actual_block_arrival is not None
                         else now) < my_arr)
        t0 = now
    else:
        eta = getattr(j, "provided_eta", None)
        if eta is None:
            return None
        ahead = sum(1 for o in sim.jobs.values()
                    if o.is_external_truck and o.status.name == "WAITING")
        t0 = max(now, eta)
    wait_pred = (run_rem + SVC_REF_S * ahead) / max(1, len(crane_ids))
    return t0 + wait_pred + SVC_REF_S + GATE_BLOCK_MEAN_S


def predict_vessel_completion(sim, v) -> float | None:
    if v.work_type != VesselWorkType.LOAD or v.done:
        return None
    st = load_supply_state(sim, v)
    return None if st is None else projected_completion_s(st)


# ---------------------------------------------------------------- 곡선 API (v1 과 동형)
def delay_cost_curve_v2(sim, job_id: str, kappa: KappaFit,
                        dts: tuple[float, ...] = DTS_DEFAULT) -> DelayCostCurve:
    """v2.1 곡선 — κ 는 0단계 동결값. 불확실성은 κ 에 내재 (lo=cost=hi)."""
    j = sim.jobs[job_id]
    if getattr(j, "is_external_truck", False):
        a = getattr(j, "actual_gate_in", None)
        o_hat = predict_gate_out(sim, job_id)
        if a is None or o_hat is None:
            pts = tuple(DelayCostPoint(d, 0.0, 0.0,
                                       2.0 * min(d, max(0.0, sim.end - sim.now)) / 3600.0)
                        for d in dts)
            return DelayCostCurve(job_id, "truck_no_eta", pts, None,
                                  "v2.1: 게이트인/ETA 결측 — fail-closed (청구 0·상한만)")
        o_c = o_hat + kappa.bias_t_s                 # 중심 보정
        d_t = truck_target_s(sim, a)
        arrived = j.status.name != "PLANNED"
        pts = []
        for d in dts:
            c = delta_cost_truck(o_c, a, d_t, d, kappa.kappa_t_s)
            pts.append(DelayCostPoint(d, c, c if arrived else 0.0, c))
        return DelayCostCurve(job_id, "truck" if arrived else "truck_eta", tuple(pts),
                              max(0.0, d_t - o_c),
                              f"v2.1 softplus·κ_T {kappa.kappa_t_s:.0f}s·b "
                              f"{kappa.bias_t_s:+.0f}s·D_T=A+{d_t - a:.0f}s")
    vid = getattr(j, "vessel_id", None)
    if vid is not None and getattr(j, "flow", None) is not None \
            and j.flow.name == "VESSEL_LOAD":
        v = sim.vessels.get(vid)
        f_hat = None if v is None else predict_vessel_completion(sim, v)
        if f_hat is not None and v.plan.planned_completion_s is not None:
            f_c = f_hat + kappa.bias_v_s
            p = v.plan.planned_completion_s
            pts = tuple(DelayCostPoint(d, c := delta_cost_vessel(f_c, p, d, kappa.kappa_v_s),
                                       c, c) for d in dts)
            return DelayCostCurve(job_id, "vessel_load", pts, max(0.0, p - f_c),
                                  f"v2.1 softplus·κ_V {kappa.kappa_v_s:.0f}s·"
                                  f"b {kappa.bias_v_s:+.0f}s·ρ=10")
    pts = tuple(DelayCostPoint(d, 0.0, 0.0, 0.0) for d in dts)
    return DelayCostCurve(job_id, "other", pts, None, "v2.1: 과금 레버 없음 (양하 포함)")
