"""YR-103 — 동적 블록 혼잡도 관측 (state·진단 전용).

**금지**: 이 점수를 terminal total cost / reward 에 가산하는 것 (기존 대기·부하·장치율
비용과 중복 — 과거 interference proxy 지배 재발 경로. spec §정책 연결 원칙).

원료 5종 — raw 를 그대로 더하지 않고 각각 [0,1] 로 정규화 (spec §블록 혼잡도 원료):
  Q = 전체 WAITING 외부트럭 수 / QUEUE_REF
  W = 전체 WAITING 외부트럭 평균(now − 실제 블록도착) / long_wait_sla_s   ← pruning 무관 전체
  L = (YC 실행 잔여초 + 대기작업 표준 서비스초) / (YC수 × HORIZON)        ← 절대부하 (불균형 아님)
  F = 현재 적재 컨테이너 수 / (bay×row×tier 물리용량)                     ← runtime (초기 fill 아님)
  D = 미도착(PLANNED) 공개작업의 clip((예상 g2b − 180)/240) 평균, 없으면 0

합성 (사용자 가중치 보존 — 재척도 금지, spec):
  C_internal = 0.30Q + 0.25W + 0.20L + 0.15F
  C_minus = clip(C_int − 0.10D)  ·  C_zero = clip(C_int)  ·  C_plus = clip(C_int + 0.10D)

정규화 기준 (결과 열람 전 고정 — 전부 assumed·D5, 민감도 전 확정값 주장 금지):
  QUEUE_REF=16 (2크레인 × 8대 대기 = 만원), WORKLOAD_HORIZON_S=900, SVC_REF_S=180
  (대기작업 1건 표준 서비스 — 프로파일 cadence 급), wait_ref = long_wait_sla_s.

정보경계: 현재 관측(도착한 트럭의 실측 대기 포함)과 공개 계획(estimated/appointment)만.
미도착 작업의 **미래 actual** 은 절대 미열람 — D 는 estimated−appointment 만 사용.
현 단계 한계(정직): 갱신 ETA 시스템 전이라 예상 g2b 가 계약 중심값 상수 → D 는 사실상
{0, 상수} 근처 — 부호 검증의 변별력은 "미도착 유입 존재량"에 한정 (spec 재검토 조항).
"""
from __future__ import annotations

from statistics import fmean

QUEUE_REF = 16.0
WORKLOAD_HORIZON_S = 900.0
SVC_REF_S = 180.0
W_Q, W_W, W_L, W_F, W_D = 0.30, 0.25, 0.20, 0.15, 0.10


def _c01(x: float) -> float:
    return max(0.0, min(1.0, x))


def block_congestion(sim, *, queue_ref: float = QUEUE_REF, wait_ref_s: float | None = None,
                     workload_horizon_s: float = WORKLOAD_HORIZON_S,
                     svc_ref_s: float = SVC_REF_S) -> dict:
    """현재 sim 의 블록 혼잡 스냅샷 — {Q,W,L,F,D,C_minus,C_zero,C_plus} 전부 [0,1]."""
    now = sim.now
    waiting_ext = [j for j in sim.jobs.values()
                   if j.is_external_truck and j.status.name == "WAITING"]
    q = _c01(len(waiting_ext) / queue_ref)
    ref_w = wait_ref_s if wait_ref_s is not None else sim.profile.long_wait_sla_s
    if waiting_ext:
        w = _c01(fmean(now - (j.actual_block_arrival if j.actual_block_arrival is not None
                              else now) for j in waiting_ext) / ref_w)
    else:
        w = 0.0
    crane_ids = [c.crane_id for c in sim.profile.cranes]
    run_rem = sum(max(0.0, sim.fleet.get(cid).state.available_at - now) for cid in crane_ids)
    n_pend = sum(1 for j in sim.jobs.values() if j.status.name in ("WAITING", "RELEASED"))
    l = _c01((run_rem + svc_ref_s * n_pend) / (len(crane_ids) * workload_horizon_s))
    g = sim.profile.block
    f = _c01(len(sim.stacks.containers) / float(g.bay_count * g.row_count * g.tier_max))
    ds = []
    for j in sim.jobs.values():
        if not j.is_external_truck or j.status.name != "PLANNED":
            continue
        est = getattr(j, "estimated_block_arrival", None)
        appt = getattr(j, "appointment_gate_time", None)
        if est is None or appt is None:
            continue
        ds.append(_c01(((est - appt) - 180.0) / 240.0))
    d = fmean(ds) if ds else 0.0
    c_int = W_Q * q + W_W * w + W_L * l + W_F * f
    return {"Q": q, "W": w, "L": l, "F": f, "D": d,
            "C_minus": _c01(c_int - W_D * d), "C_zero": _c01(c_int),
            "C_plus": _c01(c_int + W_D * d)}
