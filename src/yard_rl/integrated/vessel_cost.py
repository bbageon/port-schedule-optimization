"""YR-100 — 본선 비용 계산식 (ExecutionQ·TransferResolver 공유 원료).

본선 긴급도는 학습(4축 실패: v6·flow_margin·n-step·YR-090)이 아니라 **계산**한다.
4계약 게이트 (2026-07-26 사용자 확정, spec: YR-100-vessel-cost-formula.md):

① **기준시간 정합** — Risk 기준 = `plan.planned_completion_s` (ETD 아님 — 평가비용
  vessel_delay·berth KPI 와 같은 시각, engine._vessel_finish/_finalize 와 동일).
  구조적 강제: `VesselSupplyState` 에 ETD 필드 자체가 없다. surrogate 는 softplus 로,
  κ→0·margin=0 극한에서 평가비용 33·max(0,F−pc)/3600 과 일치(테스트 고정).
② **공급경로 인과** — 예상완료 F 는 naive `m·cadence` 금지. STS 는 박스가 안벽에
  있어야 move 가능 — F = (버퍼·이송중 확정공급 + 미확정 공급의 최소 리드타임) fold.
  미확정 공급 가정: `steady_onset_s`(첫 미확정 박스는 아무리 빨라도 YC서빙+YT이송 뒤 =
  flow_margin 과 같은 싼 근사 move_time×(1+대기/n)) 이후 `steady_pace_s` 간격 —
  "지금 확정된 정보로 피할 수 없는 정지"만 F 에 반영 (미래 결정의 몫은 잔여망/rollout).
③ **재배정 반사실** — KEEP/TRANSFER 는 상태差 Risk(s′)−Risk(s) 가 아니라 **같은 now·
  같은 pc·같은 지평**의 완료비용 비교 (`compare_completion_cost` 가 assert 로 강제).
④ **정보 공개** — LOAD pc 는 YR-080 결정3(사용자 승인 2026-07-22)로 존재하며 엔진이
  그 시각 기준으로 비용을 청구한다(engine:891). **비용이 청구되는 계획시각은 정책
  가시**(비용-관측 정합 — 청구 기준을 숨기고 최적화를 요구할 수 없다). 후보 feature
  deadline_slack_s 는 이미 이 값을 소비 중(adapter:211) — 본 모듈이 그 관행을 명시
  provenance 로 승격한다. PLANNED/현재 관측만 사용, `VesselTruth`(actual_*) 미열람.

LOAD 전용 — 양하(DISCHARGE)는 yard_handover_cap=None 에서 크레인 레버 0 (감사 실측).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .cost_config import NUMERAIRE_SCALE, NUMERAIRE_WEIGHT
from .vessel import VesselWorkType

RHO_VESSEL = NUMERAIRE_WEIGHT["vessel_delay"]      # 33.0 — 단일소스 (cost_config)
SCALE_VESSEL = NUMERAIRE_SCALE["vessel_delay"]     # 3600.0
KAPPA_S = 300.0        # surrogate 부드러움 (초). κ→0 = hinge. assumed — 민감도 대상
MARGIN_S = 0.0         # 안전여유 (초). 0 = 평가비용과 정합(기본). >0 = 선행 보수화


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

    ETD 필드 없음(계약①). actual_* 없음(계약④). KEEP/TRANSFER 반사실은 이 스냅샷을
    두 벌 만들어 비교한다(계약③) — 차이는 공급측(supply_etas·steady_*)에만 둔다.
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


def projected_completion_s(st: VesselSupplyState) -> float:
    """계약② — 공급 fold 로 결정론 예상완료시각 F.

    k번째 move 는 k번째 박스 가용시각과 STS 가용시각 중 늦은 때 시작, cadence 소요.
    박스 가용열 = (확정: [now]×buffer + supply_etas) ∪ (가정: onset+k·pace) 의
    **병합 정렬 스트림** — 확정 공급이 가정 공급을 밀어내지 않는다(k번째로 작은 값).
    """
    m = st.remaining_moves
    if m <= 0:
        return st.now
    pace = st.steady_pace_s if st.steady_pace_s > 0 else st.cadence_s
    known = sorted([st.now] * max(0, st.buffer_level) + list(st.supply_etas))
    t = max(st.now, st.planned_start_s)              # STS 가용 시작
    ki = si = 0
    for _ in range(m):
        s_val = st.steady_onset_s + si * pace
        if ki < len(known) and known[ki] <= s_val:
            avail, ki = known[ki], ki + 1
        else:
            avail, si = s_val, si + 1
        t = max(t, avail) + st.cadence_s
    return t


def overrun_cost(f_s: float, pc_s: float, *, kappa_s: float = KAPPA_S,
                 margin_s: float = MARGIN_S) -> float:
    """계약① — surrogate 선석초과 비용 [numeraire]. κ→0·margin=0 극한 = 평가 vessel_delay."""
    return RHO_VESSEL * softplus_s(f_s - pc_s + margin_s, kappa_s) / SCALE_VESSEL


def completion_cost(st: VesselSupplyState, **kw) -> float:
    return overrun_cost(projected_completion_s(st), st.planned_completion_s, **kw)


def compare_completion_cost(keep: VesselSupplyState, transfer: VesselSupplyState,
                            **kw) -> float:
    """계약③ — 같은 now·pc·지평의 KEEP vs TRANSFER 완료비용差 (음수 = TRANSFER 이득).

    상태差(Risk(s′)−Risk(s)) 금지 — 시간이 흐른 두 상태 비교는 이득을 못 잡는다.
    """
    if keep.now != transfer.now or keep.planned_completion_s != transfer.planned_completion_s:
        raise ValueError("반사실 계약 위반: KEEP/TRANSFER 는 같은 now·planned_completion 이어야 함")
    return completion_cost(transfer, **kw) - completion_cost(keep, **kw)


# ---------------------------------------------------------------- live-sim 빌더 (공개정보만)
def _first_unassured_gap_s(sim) -> float:
    """미확정 공급 최소 리드타임 — flow_margin_s 와 같은 싼 근사 (YC서빙+YT 1트립)."""
    n = max(1, sim.transfer.n_units)
    return sim.transfer.move_time_s * (1.0 + sim.transfer.waiting_count() / n)


def load_supply_state(sim, v, *, supply_etas: tuple[float, ...] = (),
                      steady_onset_s: float | None = None) -> VesselSupplyState | None:
    """살아있는 sim → LOAD 공급 스냅샷. PLANNED/현재 관측만 — truth 미열람(계약④)."""
    if v.work_type != VesselWorkType.LOAD or v.done:
        return None
    pc = v.plan.planned_completion_s
    if pc is None:
        return None
    now = sim.now
    rem = max(0, v.remaining_moves) if v.started else v.plan.total_moves
    if steady_onset_s is None:
        steady_onset_s = now + _first_unassured_gap_s(sim)
    return VesselSupplyState(
        now=now, planned_completion_s=pc, planned_start_s=v.plan.planned_start_s,
        cadence_s=v.plan.sts_move_interval_s, remaining_moves=rem,
        buffer_level=max(0, v.buffer_level), supply_etas=tuple(sorted(supply_etas)),
        steady_onset_s=steady_onset_s, steady_pace_s=0.0)


def serve_supply_delta(sim, v, serve_duration_s: float, **kw) -> float:
    """ExecutionQ 후보용 ΔC_vessel — 같은 now 의 paired 비교 (계약③ 의 블록 내 형태).

    직렬 크레인 체인 모델: 미확정 본선 공급 슬롯열은 두 세계가 공유하고,
    - 기준행동(지금 서빙 안 함): 첫 슬롯이 한 박자(DEFER=cadence) 뒤 = t0+DEFER 시작.
    - 후보(지금 서빙): t0 = now+dur+YT 에 확정 공급 + 이후 슬롯열이 pace 간격으로 앞당겨짐.
    → 이득 = 공급 사슬을 한 박자 당긴 값 × softplus 긴급도 게이트. 여유 크면 ≈0
    ("여유 있으면 트럭처럼"). 음수 = 본선 이득. 공급 과잉(버퍼 여유)이면 자동 0 (포화)."""
    if v.work_type != VesselWorkType.LOAD or v.done:
        return 0.0
    pc = v.plan.planned_completion_s
    if pc is None:
        return 0.0
    t0 = sim.now + serve_duration_s + sim.transfer.move_time_s
    defer = pace = v.plan.sts_move_interval_s
    base = load_supply_state(sim, v, steady_onset_s=t0 + defer)
    cand = load_supply_state(sim, v, supply_etas=(t0,), steady_onset_s=t0 + pace)
    return completion_cost(cand, **kw) - completion_cost(base, **kw)


def candidate_vessel_delta(sim, assign: dict, **kw) -> float:
    """공동조합(assign: crane→GenCandidate) 의 ΔC_vessel 합 — 본선/트럭 무구분 통합비용의
    본선 항. SERVE(VESSEL_LOAD) 만 공급 기여, 그 외 0 (type-agnostic: 숫자만 argmin 에).
    동일 선박 동시 2서빙은 서빙별 delta 합 근사 (드묾 — 문서화된 근사)."""
    tot = 0.0
    for gc in assign.values():
        ref = getattr(gc, "job_ref", None)
        plan = getattr(gc, "plan", None)
        if ref is None or plan is None or not getattr(ref, "is_vessel", False):
            continue
        j = sim.jobs.get(ref.job_id)
        if j is None or j.vessel_id is None or j.flow.name != "VESSEL_LOAD":
            continue
        v = sim.vessels.get(j.vessel_id)
        if v is None:
            continue
        tot += serve_supply_delta(sim, v, plan.duration_s, **kw)
    return tot
