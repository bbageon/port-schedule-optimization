"""YR-100 — 본선 비용 계산식 (ExecutionQ·TransferResolver 공유 원료).

본선 긴급도는 학습(4축 실패: v6·flow_margin·n-step·YR-090)이 아니라 **계산**한다.
4계약 게이트 (spec: YR-100-vessel-cost-formula.md) + 적대검증 정정 (2026-07-26, 4렌즈):

① **기준시간 정합** — Risk 기준 = `plan.planned_completion_s` (구조 강제: 상태에 ETD
  필드 없음). surrogate = softplus, κ→0·margin=0 극한서 평가비용과 일치. **검열 캡**:
  horizon(sim.end) 을 넘는 F 는 min(F, horizon) — 엔진 CLEAROUT(end−pc)와 정합
  (검증 정정: 검열영역 유령 이득 제거). ρ·scale 은 민감도 스윕용 인자 주입 가능.
② **공급 인과** — F = 엔진 점화식 `t_k = max(t_{k-1}+cadence, a_k)` (박스는 move
  **완료**에 필요 — 굶은 STS 는 도착 즉시 완료. 검증 정정: 이전 +1 cadence 비관 제거).
  공급열 = 버퍼 + **이송중 확정도착(TRANSFER_ARRIVE 스캔)** + 미확정 steady 병합.
③ **재배정 반사실** — 같은 now·pc·**공급외 상태 전부 동일**을 assert (검증 정정:
  remaining/buffer/cadence/start 불일치 침묵통과 차단). 상태差 비교 금지.
④ **정보 공개** — 비용이 청구되는 계획시각(pc)·자기 야드가 보낸 이송 도착예정은 터미널
  공개정보 (flow_margin 과 같은 관측등급). truth(actual_*) 미열람.

pairing (검증 정정 2건 — v6형 게임 벡터 제거):
- 기준세계 anchor 는 **후보 duration 과 독립** (t_ref = now + gap + cadence) — 느린
  후보가 더 큰 credit 을 받던 knee 역전 제거. 후보 확정공급만 now+dur+gap.
- 동일 선박 다중 서빙은 **선박별 1회 joint 계산** (합산 2배 과대 제거).
- gap = move_time×(1+YT대기/n) — flow_margin 관례 통일 (혼잡 반영).

LOAD 전용 — 양하(DISCHARGE)는 yard_handover_cap=None 에서 크레인 레버 0 (감사 실측).
margin_s>0 은 delta(결정 shaping) 경로 전용 — C_known 절대값 주입 경로는 margin=0 유지.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .cost_config import NUMERAIRE_SCALE, NUMERAIRE_WEIGHT
from .vessel import VesselWorkType

RHO_VESSEL = NUMERAIRE_WEIGHT["vessel_delay"]      # 33.0 — 단일소스 (cost_config)
SCALE_VESSEL = NUMERAIRE_SCALE["vessel_delay"]     # 3600.0
KAPPA_S = 300.0        # surrogate 부드러움 (초). κ→0 = hinge. assumed — 민감도 대상
MARGIN_S = 0.0         # 안전여유 (초). 0 = 평가정합(기본·C_known 경로 강제)


def softplus_s(x_s: float, kappa_s: float = KAPPA_S) -> float:
    """κ·log(1+exp(x/κ)) [초] — max(0,x) 의 C¹ surrogate. κ≤0 이면 hinge 그 자체."""
    if kappa_s <= 0.0:
        return max(0.0, x_s)
    z = x_s / kappa_s
    if z > 30.0:
        return x_s
    if z < -30.0:
        return 0.0
    return kappa_s * math.log1p(math.exp(z))


@dataclass(frozen=True)
class VesselSupplyState:
    """공개 계획정보만의 본선(LOAD) 공급 스냅샷 — 계약 ①②③ 공용 순수 입력.

    ETD 필드 없음(계약①). actual_* 없음(계약④). 반사실은 이 스냅샷 두 벌 비교(계약③)
    — 차이는 공급측(supply_etas·steady_*)에만 둔다 (compare 가 assert).
    """
    now: float
    planned_completion_s: float          # 계약① 기준시각 (비용 청구 기준과 동일)
    planned_start_s: float
    cadence_s: float                     # STS move 간격 (계획값)
    remaining_moves: int                 # 미개시면 total_moves
    buffer_level: int                    # 안벽 staged 박스 (즉시 가용)
    supply_etas: tuple[float, ...] = ()  # 확정 공급 도착시각 (이송중·후보 기여)
    steady_onset_s: float = 0.0          # 미확정 공급 최초 가용시각 (절대시각)
    steady_pace_s: float = 0.0           # 이후 공급 간격 (0 = cadence 페이스 유지 가정)
    horizon_s: float | None = None       # 평가창 끝 — 검열 캡 (None = 캡 없음)


def projected_completion_s(st: VesselSupplyState) -> float:
    """계약② — 엔진 점화식 공급 fold: t_k = max(t_{k-1}+cadence, a_k).

    박스는 move **완료** 시점에 필요 (굶은 STS 는 도착 즉시 완료 — engine._transfer_arrive
    → 즉시 STS_MOVE). a_k = k번째로 작은 (확정: [now]×buffer+etas) ∪ (가정: onset+i·pace).
    """
    m = st.remaining_moves
    if m <= 0:
        return st.now
    pace = st.steady_pace_s if st.steady_pace_s > 0 else st.cadence_s
    known = sorted([st.now] * max(0, st.buffer_level) + list(st.supply_etas))
    t = max(st.now, st.planned_start_s)
    ki = si = 0
    for _ in range(m):
        s_val = st.steady_onset_s + si * pace
        if ki < len(known) and known[ki] <= s_val:
            avail, ki = known[ki], ki + 1
        else:
            avail, si = s_val, si + 1
        t = max(t + st.cadence_s, avail)
    return t


def overrun_cost(f_s: float, pc_s: float, *, kappa_s: float = KAPPA_S,
                 margin_s: float = MARGIN_S, horizon_s: float | None = None,
                 rho: float = RHO_VESSEL, scale: float = SCALE_VESSEL) -> float:
    """계약① — surrogate 선석초과 비용 [numeraire].

    κ→0·margin=0 극한 = 평가 vessel_delay. horizon 캡 = 엔진 CLEAROUT(end−pc) 정합 —
    검열영역(F>end)의 실현 불가능한 이득/손실을 차단. rho/scale 인자 = ρ 민감도 스윕 정합.
    """
    if horizon_s is not None:
        f_s = min(f_s, horizon_s)
    return rho * softplus_s(f_s - pc_s + margin_s, kappa_s) / scale


def completion_cost(st: VesselSupplyState, **kw) -> float:
    kw.setdefault("horizon_s", st.horizon_s)
    return overrun_cost(projected_completion_s(st), st.planned_completion_s, **kw)


def compare_completion_cost(keep: VesselSupplyState, transfer: VesselSupplyState,
                            **kw) -> float:
    """계약③ — 같은 now·pc·공급외 상태의 KEEP vs TRANSFER 완료비용差 (음수 = TRANSFER 이득).

    공급측(supply_etas·steady_*)만 다를 수 있다 — 나머지 필드 불일치는 이중계상이므로 거부.
    """
    for f in ("now", "planned_completion_s", "planned_start_s", "cadence_s",
              "remaining_moves", "buffer_level", "horizon_s"):
        if getattr(keep, f) != getattr(transfer, f):
            raise ValueError(f"반사실 계약 위반: {f} 는 KEEP/TRANSFER 동일이어야 함 (공급측만 자유)")
    return completion_cost(transfer, **kw) - completion_cost(keep, **kw)


# ---------------------------------------------------------------- live-sim 빌더 (공개정보만)
def _gap_s(sim) -> float:
    """YT 한 트립 예상 (혼잡 반영) — flow_margin_s 와 같은 싼 근사 관례."""
    n = max(1, sim.transfer.n_units)
    return sim.transfer.move_time_s * (1.0 + sim.transfer.waiting_count() / n)


def inflight_supply_etas(sim, vid: str) -> tuple[float, ...]:
    """이송중 확정공급 도착시각 — 자기 야드가 보낸 이송의 도착예정 (공개 관측등급).

    TRANSFER_ARRIVE(vid) 예약 이벤트 스캔 (predictive_rollout 의 _heap 접근 선례).
    """
    try:
        return tuple(sorted(e.time for e in sim.queue._heap
                            if e.kind_name == "TRANSFER_ARRIVE" and e.payload == vid
                            and e.time >= sim.now))
    except AttributeError:
        return ()


def load_supply_state(sim, v, *, supply_etas: tuple[float, ...] = (),
                      steady_onset_s: float | None = None,
                      steady_pace_s: float = 0.0,
                      include_inflight: bool = True) -> VesselSupplyState | None:
    """살아있는 sim → LOAD 공급 스냅샷. PLANNED/현재 관측만 — truth 미열람(계약④)."""
    if v.work_type != VesselWorkType.LOAD or v.done:
        return None
    pc = v.plan.planned_completion_s
    if pc is None:
        return None
    now = sim.now
    rem = max(0, v.remaining_moves) if v.started else v.plan.total_moves
    etas = list(supply_etas)
    if include_inflight:
        etas.extend(inflight_supply_etas(sim, v.vessel_id))
    if steady_onset_s is None:
        steady_onset_s = now + v.plan.sts_move_interval_s + _gap_s(sim)
    return VesselSupplyState(
        now=now, planned_completion_s=pc, planned_start_s=v.plan.planned_start_s,
        cadence_s=v.plan.sts_move_interval_s, remaining_moves=rem,
        buffer_level=max(0, v.buffer_level), supply_etas=tuple(sorted(etas)),
        steady_onset_s=steady_onset_s, steady_pace_s=steady_pace_s,
        horizon_s=float(sim.end) if getattr(sim, "end", None) is not None else None)


def vessel_serves_delta(sim, v, serve_durations: tuple[float, ...], **kw) -> float:
    """ExecutionQ 후보용 ΔC_vessel — 같은 now 의 paired 비교, 선박별 **1회 joint** 계산.

    두 세계가 **같은** 미확정 공급열을 공유: anchor = now+cadence+gap (후보 duration 과
    독립 — knee 역전 제거), pace = max(cadence, gap) (현 혼잡 하 공급망 처리량 상계 —
    이보다 빠른 미래 공급을 가정하면 후보 기여가 0 으로 붕괴). 후보세계만 후보 박스들
    (now+dur_i+gap 확정공급)이 **추가**된다 → delta ≤ 0 보장(공급 추가는 F 를 늦출 수 없음)·
    duration 단조(느릴수록 credit↓)·동일선박 다중서빙 joint 1회(2배 금지).
    음수 = 본선 이득. 여유 크면 softplus 꼬리 ≈0 · 버퍼 과잉이면 fold 자동 0 (포화).
    """
    if v.work_type != VesselWorkType.LOAD or v.done or not serve_durations:
        return 0.0
    if v.plan.planned_completion_s is None:
        return 0.0
    gap = _gap_s(sim)
    cad = v.plan.sts_move_interval_s
    t_ref = sim.now + cad + gap                     # 후보-독립 기준 anchor
    pace = max(cad, gap)                            # 공급망 처리량 상계
    base = load_supply_state(sim, v, steady_onset_s=t_ref, steady_pace_s=pace)
    cand_etas = tuple(sim.now + d + gap for d in serve_durations)
    cand = load_supply_state(sim, v, supply_etas=cand_etas,
                             steady_onset_s=t_ref, steady_pace_s=pace)
    return completion_cost(cand, **kw) - completion_cost(base, **kw)


def candidate_vessel_delta(sim, assign: dict, **kw) -> float:
    """공동조합(assign: crane→GenCandidate) 의 ΔC_vessel — 본선/트럭 무구분 통합비용의
    본선 항. SERVE(VESSEL_LOAD) 만 공급 기여. 동일 선박 다중 서빙 = joint 1회 (2배 금지)."""
    by_vessel: dict[str, list[float]] = {}
    for gc in assign.values():
        ref = getattr(gc, "job_ref", None)
        plan = getattr(gc, "plan", None)
        if ref is None or plan is None or not getattr(ref, "is_vessel", False):
            continue
        j = sim.jobs.get(ref.job_id)
        if j is None or j.vessel_id is None or j.flow.name != "VESSEL_LOAD":
            continue
        by_vessel.setdefault(j.vessel_id, []).append(plan.duration_s)
    tot = 0.0
    for vid, durs in by_vessel.items():
        v = sim.vessels.get(vid)
        if v is not None:
            tot += vessel_serves_delta(sim, v, tuple(durs), **kw)
    return tot
