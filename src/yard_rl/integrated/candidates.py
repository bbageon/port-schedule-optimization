"""동적 후보 생성기 — 4종·mandatory·padding·feasible 노출 (YR-037, 최종전략 §8).

engine.candidates_for(최소 SERVE·feasible-only, golden 하네스용 compat shim)와 별개로, 정책·
resolver 가 소비할 **풍부한 CandidateSet** 을 생성한다.
- 4종: SERVE / PRE_REHANDLE(§8.4 도착 전 재조작) / REPOSITION(§8.2 위치조정) / WAIT.
- feasible_mask = **committed(직전 결정까지) 예약 대비 marginal 실행가능성**. 형제 크레인 충돌은
  resolver 몫(D-FEASIBLE) → mask 가 resolve 순서 무관·결정적. 오늘 조용히 드롭하던 충돌 SERVE 를
  feasible=False·mask_reason 으로 노출한다.
- mandatory(SLA 임박, YR-029 흡수)는 pruning 절대 금지. 정보시점 게이팅으로 누출 0.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..domain.enums import InformationLevel, JobFlow, JobStatus
from ..sim.constraints import ConstraintViolation
from ..contract.schema import CandidateKind
from .jobplan import JobPlan, JobRef

_KIND_RANK = {CandidateKind.SERVE: 0, CandidateKind.PRE_REHANDLE: 1,
              CandidateKind.REPOSITION: 2, CandidateKind.WAIT: 3}


@dataclass(frozen=True)
class GenCandidate:
    candidate_id: int                 # 튜플 위치 == CandidateSet items 인덱스
    kind: CandidateKind
    job_ref: JobRef | None            # SERVE/PRE/REPO 는 JobRef; WAIT=None
    plan: JobPlan | None              # marginal feasibility plan (어댑터 feature 원천); WAIT=None
    mandatory: bool
    feasible: bool                    # committed 대비 marginal
    mask_reason: str | None           # feasible=True ⟺ None
    score: float                      # §8.3 pruning 전용 — features·net 진입 금지


@dataclass(frozen=True)
class GeneratedCandidates:
    crane_id: str
    items: tuple[GenCandidate, ...]   # 실후보(WAIT 포함, padding 제외). 위치==candidate_id


class CandidateGenerator:
    def __init__(self, *, k_max: int = 12, mandatory_wait_frac: float = 0.8,
                 pre_rehandle_min_window_s: float = 600.0):
        self.k_max = k_max
        self.mandatory_wait_frac = mandatory_wait_frac
        # YR-043: mask 아님 — "도착 전 완료 가능" 은 §8.4 운영 트레이드오프라 State/Cost 로 이관.
        # 참고값으로만 보존 (후보 생성 게이트로 사용 금지).
        self.pre_window = pre_rehandle_min_window_s

    # -------------------------------------------------------- entry points
    def serve_refs(self, sim, crane_id: str) -> list[JobRef]:
        """engine.candidates_for 호환 — feasible SERVE JobRef 만 (오늘과 동치)."""
        out = []
        for gc in self._serve(sim, crane_id, sim.now):
            if gc.feasible:
                out.append(gc.job_ref)
        return out

    def generate(self, sim, crane_id: str, level: InformationLevel) -> GeneratedCandidates:
        yc = sim.fleet.get(crane_id)
        if not yc.idle or yc.yielded:
            return GeneratedCandidates(crane_id, (replace(self._wait(), candidate_id=0),))
        now = sim.now
        raw = (self._serve(sim, crane_id, now)
               + self._pre_rehandle(sim, crane_id, now, level)
               + self._reposition(sim, crane_id, now, level))
        keep = sorted(self._prune(raw), key=self._order_key)
        items = list(keep) + [self._wait()]
        return GeneratedCandidates(
            crane_id, tuple(replace(gc, candidate_id=i) for i, gc in enumerate(items)))

    # -------------------------------------------------------- 4종 생성
    def _serve(self, sim, cid, now) -> list[GenCandidate]:
        yc = sim.fleet.get(cid)
        spec = sim.fleet.spec(cid)
        out = []
        for jid in sorted(sim.jobs):
            j = sim.jobs[jid]
            if not sim._dispatchable(j, cid):
                continue
            ref = sim._jobref(j, spec, yc)
            if ref is None:
                continue
            cum = sim.cum_wait(jid) if j.is_external_truck else None
            mand = self._is_mandatory(sim, j, cum)
            plan = sim._plan(cid, ref)
            if plan is None:
                if not mand:
                    continue
                out.append(GenCandidate(0, CandidateKind.SERVE, ref, None, True, False,
                                        "PLAN_FAILED", self._score(sim, ref, None, now, cum)))
                continue
            reason = self._committed_reason(sim, plan)
            out.append(GenCandidate(0, CandidateKind.SERVE, ref, plan, mand, reason is None,
                                    reason, self._score(sim, ref, plan, now, cum)))
        return out

    def _pre_rehandle(self, sim, cid, now, level) -> list[GenCandidate]:
        if level != InformationLevel.PRE_ADVICE:
            return []          # ETA 불가시 레벨은 미발행 (누출 0)
        yc = sim.fleet.get(cid)
        spec = sim.fleet.spec(cid)
        horizon = sim.profile.decision_horizon_s
        out = []
        for jid in sorted(sim.jobs):
            j = sim.jobs[jid]
            if j.flow != JobFlow.GATE_OUT or j.status != JobStatus.PLANNED:
                continue
            if j.target_container is None:
                continue
            c = sim.stacks.containers.get(j.target_container)
            if c is None or not c.work_available:
                continue
            if not (spec.service_bay_min <= c.bay <= spec.service_bay_max):
                continue
            if not sim.stacks.blockers_above(j.target_container):
                continue
            if not sim.stacks.rehandle_capacity_ok(j.target_container, spec):
                continue
            eta = self._visible_eta(j, level)
            # 정보 제약(ETA 미가시) + §8.3 탐색축소(결정 지평)만 게이트.
            # YR-043: "도착 전 완료 가능"(pre_window) 게이트 제거 — §8.4 운영 트레이드오프는
            # mask 가 아니라 State/Cost 로 제공(predicted_arrival_gap_s feature)해 RL 이 학습한다.
            if eta is None or eta - now > horizon:
                continue
            ref = JobRef(job_id=jid, token=jid, kind=CandidateKind.PRE_REHANDLE,
                         target_container=j.target_container, lane_id=sim._lane_for(c.bay),
                         eligible_crane_ids=sim.eligible_cranes(c.bay),
                         is_vessel=False, is_external=True)
            plan = sim._plan(cid, ref)
            if plan is None:
                continue
            reason = self._committed_reason(sim, plan)
            out.append(GenCandidate(0, CandidateKind.PRE_REHANDLE, ref, plan, False,
                                    reason is None, reason,
                                    self._score(sim, ref, plan, now, None)))
        return out

    def _reposition(self, sim, cid, now, level) -> list[GenCandidate]:
        yc = sim.fleet.get(cid)
        out = []
        for tb in sorted(self._future_target_bays(sim, cid, now, level)):
            if abs(tb - yc.state.position_bay) <= 1.0:   # 이동가치·0-루프 방지
                continue
            ref = JobRef(job_id=f"REPO:{cid}:{int(tb)}", token=None,
                         kind=CandidateKind.REPOSITION, target_container=None, lane_id=None,
                         eligible_crane_ids=(cid,), is_vessel=False, is_external=False,
                         reposition_target_bay=tb)
            plan = sim._plan(cid, ref)
            if plan is None:
                continue
            reason = self._committed_reason(sim, plan)
            out.append(GenCandidate(0, CandidateKind.REPOSITION, ref, plan, False,
                                    reason is None, reason, -1000.0 + tb))  # 낮은 우선
        return out

    def _wait(self) -> GenCandidate:
        return GenCandidate(0, CandidateKind.WAIT, None, None, False, True, None, float("-inf"))

    # -------------------------------------------------------- 공통
    def _committed_reason(self, sim, plan) -> str | None:
        """1차 mask = ReservationTable.reject_reason (2·3차와 동일 소스)."""
        return sim.reservations.reject_reason(sim._reservation(plan))

    def _is_mandatory(self, sim, j, cum) -> bool:
        return bool(j.is_external_truck and cum is not None
                    and cum >= self.mandatory_wait_frac * sim.profile.long_wait_sla_s)

    def _visible_eta(self, j, level) -> float | None:
        """정책 가시 도착예상 — PRE_ADVICE 의 provided_eta 만. actual_* 절대 미열람(누출 0)."""
        if level == InformationLevel.PRE_ADVICE and j.provided_eta is not None:
            return j.provided_eta
        return None

    def _future_job_bay(self, sim, j, spec, yc) -> float | None:
        if j.target_container is not None and j.target_container in sim.stacks.containers:
            return float(sim.stacks.containers[j.target_container].bay)
        if j.flow == JobFlow.GATE_IN and j.inbound_size is not None:
            slot = sim.stacks.find_slot(j.inbound_size, spec, yc.state.position_bay,
                                        yc.state.trolley_row)
            return float(slot[0]) if slot else None
        return None

    def _future_target_bays(self, sim, cid, now, level) -> set:
        spec = sim.fleet.spec(cid)
        yc = sim.fleet.get(cid)
        horizon = sim.profile.decision_horizon_s
        bays = set()
        for j in sim.jobs.values():
            if j.status == JobStatus.DONE:
                continue
            if j.is_external_truck:
                if j.status == JobStatus.WAITING:
                    continue                      # 이미 도착 → SERVE 몫
                eta = self._visible_eta(j, level)
            else:
                eta = j.release_time if j.status == JobStatus.PLANNED else None
            if eta is None or eta <= now or eta - now > horizon:
                continue
            bay = self._future_job_bay(sim, j, spec, yc)
            if bay is None:
                continue
            bays.add(float(min(max(bay, spec.service_bay_min), spec.service_bay_max)))
        return bays

    def _score(self, sim, ref, plan, now, cum) -> float:
        """§8.3 탐색축소 필터 (계수 assumed) — pruning 전용, features 진입 금지."""
        s = 0.0
        if cum is not None:
            s += cum
        j = sim.jobs.get(ref.job_id)
        if j is not None and j.is_vessel_linked and j.deadline is not None:
            s += max(0.0, sim.profile.long_wait_sla_s - (j.deadline - now))
        if plan is not None:
            s -= 0.1 * plan.duration_s + 50.0 * plan.rehandles
        return s

    # -------------------------------------------------------- prune·order
    def _prune(self, raw) -> list:
        budget = self.k_max - 1                       # WAIT 1칸 예약
        mand = [c for c in raw if c.mandatory]
        rest = [c for c in raw if not c.mandatory]
        if len(mand) > budget:
            raise ConstraintViolation("K_TOO_SMALL",
                                      f"mandatory {len(mand)} > budget {budget} — 조용한 유실 금지")
        rest.sort(key=lambda c: (-c.score,) + self._order_key(c))
        return mand + rest[:budget - len(mand)]

    def _order_key(self, gc) -> tuple:
        """canonical id 순 — score 미세동률이 텐서 레이아웃을 흔들지 않게 membership/id 분리."""
        ref = gc.job_ref
        return (_KIND_RANK[gc.kind], ref.job_id if ref else "",
                ref.reposition_target_bay if (ref and ref.reposition_target_bay is not None) else -1.0)
