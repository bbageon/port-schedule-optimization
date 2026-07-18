"""계약 어댑터 — sim 상태 → GlobalState/LocalObservation/CostBreakdown/TransitionRecord (YR-036).

정답 조건: 매 결정마다 validate_all(rec) 통과 + canonical 왕복. feature raw 는 observable_stacks
+ TOK 게이팅만 사용하고, 진실값(actual arrival·VesselTruth)은 realized_at 게이팅·비용에만.
scale/weight/λ 는 assumed placeholder(YR-038 위임) — raw 물리 delta 만 sim 이 산출.
"""
from __future__ import annotations

from ..contract import (SCHEMA_VERSION, Assignment, Candidate, CandidateSet,
                        CandidateKind, CostBreakdown, GlobalState, JointAction,
                        LocalObservation, TransitionAudit, TransitionRecord,
                        VesselUrgency, build_feature_vector, dumps, loads,
                        make_cost, padding_candidate, resolve_mode, validate_all)
from ..domain.enums import ControlScope, InformationLevel, JobFlow, JobStatus
from ..sim.travel_time import estimate_reach_s
from .candidates import CandidateGenerator
from .cost_config import RewardCalculator
from .engine import CraneAssignment, _pstdev
from .resolver import BaselinePreference, CentralResolver, DispatcherPreference

_KINDS = list(CandidateKind)
_WAITING = (JobStatus.WAITING, JobStatus.RELEASED)
_GEN = CandidateGenerator()          # 기본 생성기 (k_max=12·mandatory_frac=0.8, YR-037)
_DEFAULT_RC = RewardCalculator.assumed_default()   # 기본 비용 config (현 assumed, YR-038)


def _c01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _rehandle_time(plan) -> float:
    return sum(m.duration_s for m in plan.moves if not m.depart and m.inbound is None)


def _lane_local(sim, lane_id: str | None) -> float:
    if lane_id is None:
        return 0.0
    occ = frozenset(r.lane_id for r in sim.reservations.active() if r.lane_id)
    deg = len(sim.lanes.neighbors(lane_id))
    load = (1.0 if lane_id in occ else 0.0) + sum(1.0 for n in sim.lanes.neighbors(lane_id) if n in occ)
    return _c01(load / (1.0 + deg))


def _max_vessel_risk(sim, now: float) -> float:
    best = 0.0
    for v in sim.vessels.values():
        if v.done or v.is_symptom():
            continue
        best = max(best, _c01((v.expected_delay_s(now) or 0.0) / 1800.0))
    return best


# ------------------------------------------------------------ feature raw
def _global_raw(sim, now: float) -> dict:
    occ = frozenset(r.lane_id for r in sim.reservations.active() if r.lane_id)
    mean_c, max_c = sim.lanes.occupancy(occ)
    ext = sum(1 for j in sim.jobs.values() if j.is_external_truck and j.status in _WAITING)
    ves = sum(1 for j in sim.jobs.values() if j.is_vessel_linked and j.status in _WAITING)
    return {
        "time_frac": _c01(now / sim.end),
        "shift_idx": float(int(now // sim.profile.shift_len_s)),
        "vessel_count": float(sum(1 for v in sim.vessels.values() if not v.done)),
        "lane_congestion_mean": _c01(mean_c), "lane_congestion_max": _c01(max_c),
        "sts_wait_accum_s": sum(v.sts_wait_accum_s for v in sim.vessels.values()),
        "transfer_wait_accum_s": sim.transfer.transfer_wait_accum_s,
        "backlog_external": float(ext), "backlog_vessel": float(ves),
        "crane_count": float(len(sim.fleet)),
        # §10.2 작업부하 불균형 I(t)∈[0,1] — cost imbalance 와 동일 정의 (YR-043 재정의).
        # 누적 완료건수 pstdev 폐기: 처리건수 균등화는 목적이 아니었고 총비용을 지배했다.
        "load_imbalance": sim.load_imbalance(),
    }


def _nearest_other(sim, yc):
    """최근접 상대 크레인 (bay 거리, 동률 시 crane_id) — COORD 관측 기준점 (YR-056)."""
    others = [c for c in sim.fleet.all() if c.crane_id != yc.crane_id]
    if not others:
        return None
    return min(others, key=lambda o: (abs(yc.state.position_bay - o.state.position_bay),
                                      o.crane_id))


def _corridor_overlap(a: tuple, b: tuple) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _contention_risk(sim, cid: str, gc) -> float:
    """후보의 경합 위험 (COORD, YR-056) — 관측 사실 기반 결정론 산식.

    max( 같은 작업을 idle 상대도 수행가능 1.0 / busy 상대도 eligible 0.5 /
         상대 실행 corridor 와 본 후보 corridor 겹침 0.7 ), 신호 없음 0.
    """
    if gc.kind == CandidateKind.WAIT or gc.plan is None:
        return 0.0
    risk = 0.0
    for o in sim.fleet.all():
        if o.crane_id == cid:
            continue
        o_plan = sim.active_plan(o.crane_id)
        ref = gc.job_ref
        if ref is not None and o.crane_id in ref.eligible_crane_ids:
            risk = max(risk, 0.5 if o_plan is not None else 1.0)
        if o_plan is not None and _corridor_overlap(gc.plan.corridor, o_plan.corridor):
            risk = max(risk, 0.7)
    return risk


def _yc_raw(sim, cid: str, now: float):
    yc = sim.fleet.get(cid)
    spec = sim.fleet.spec(cid)
    others = [c for c in sim.fleet.all() if c.crane_id != cid]
    oldest = None
    own_q = 0
    for j in sim.jobs.values():
        if j.is_external_truck and j.status == JobStatus.WAITING and j.target_container:
            c = sim.stacks.containers.get(j.target_container)
            if c and spec.service_bay_min <= c.bay <= spec.service_bay_max:
                own_q += 1
                if oldest is None or j.actual_block_arrival < oldest:
                    oldest = j.actual_block_arrival
    raw = {
        "crane_bay": yc.state.position_bay, "trolley_row": yc.state.trolley_row,
        "available_in_s": max(0.0, yc.state.available_at - now),
        "is_loaded": 1.0 if yc.is_loaded else 0.0, "last_move_dir": yc.last_move_dir,
        "recent_throughput": float(yc.recent_completions),
        "recent_empty_travel_s": yc.recent_empty_travel_s,
        "assigned_load": 1.0 if yc.state.assigned_job else 0.0,
        "own_queue_len": float(own_q),
        "own_oldest_wait_s": (now - oldest) if oldest is not None else 0.0,
        "neighbor_load_gap": (yc.served_count - sum(o.served_count for o in others) / len(others))
        if others else None,
        "neighbor_min_gap_bay": min((abs(yc.state.position_bay - o.state.position_bay)
                                     for o in others), default=None),
    }
    # COORD (YR-056): 최근접 상대의 현재 의도 — busy 면 실행 계획의 종류/종료 bay,
    # idle 이면 결측(None→known=0). 전부 commit 된 관측 사실 (진실·예측 아님).
    near = _nearest_other(sim, yc)
    n_plan = sim.active_plan(near.crane_id) if near is not None else None
    raw.update({
        "neighbor_busy_kind": (_KINDS.index(n_plan.kind) / (len(_KINDS) - 1)
                               if n_plan is not None else None),
        "neighbor_busy_target_bay": n_plan.end_bay if n_plan is not None else None,
        "neighbor_available_in_s": (max(0.0, near.state.available_at - now)
                                    if near is not None else None),
        "recent_yield_count": float(yc.recent_yield_count),
    })
    realized = {"own_oldest_wait_s": oldest} if oldest is not None else {}
    return raw, realized


def _vessel_urgency(sim, v, now: float, level: InformationLevel, ablation_off=()) -> VesselUrgency:
    mode, assumed = resolve_mode(v.plan.planned_completion_s, v.plan.completion_basis)
    ed = v.expected_delay_s(now)
    rem = float(max(0, v.remaining_moves) if v.started else v.plan.total_moves)
    common = {"remaining_moves": rem, "sts_wait_s": v.sts_wait_accum_s,
              "transfer_wait_s": sim.transfer.transfer_wait_accum_s}
    if mode.value == "RISK":
        raw = {"slack_s": v.slack_s(now), "risk": _c01((ed or 0.0) / 1800.0),
               "delay_symptom_score": None,
               "remaining_service_time_s": v.remaining_service_time_s(),
               "expected_delay_s": ed, **common}
    else:   # SYMPTOM
        sym = _c01((v.sts_wait_accum_s + sim.transfer.transfer_wait_accum_s) / 3600.0)
        raw = {"slack_s": None, "risk": None, "delay_symptom_score": sym,
               "remaining_service_time_s": None, "expected_delay_s": None, **common}
    fv = build_feature_vector("vessel", raw, now=now, info_level=level, ablation_off=ablation_off)
    # SYMPTOM 은 완료근거 없음이 계약 불변식 — 계획변경으로 basis 만 생겨도 mode 에 맞춘다.
    basis = v.plan.completion_basis if mode.value == "RISK" else None
    return VesselUrgency(v.vessel_id, mode, basis, assumed, fv)


def _cand_raw(sim, cid, gc, now):
    """GenCandidate → (feature raw dict, realized_at) — kind 분기 (YR-037)."""
    yc = sim.fleet.get(cid)
    spec = sim.fleet.spec(cid)
    geom = sim.profile.block
    kind = gc.kind
    ref, plan = gc.job_ref, gc.plan
    # WAIT 또는 계획실패(mandatory SERVE 가 혼잡으로 unplannable — PLAN_FAILED, feasible=False,
    # YR-029 무손실 보존). 물리 feature 없음 — 중립 raw (infeasible 이라 정책 선택 대상 아님).
    if kind == CandidateKind.WAIT or plan is None:
        is_ext = bool(ref and kind != CandidateKind.WAIT and ref.is_external)
        is_ves = bool(ref and kind != CandidateKind.WAIT and ref.is_vessel)
        raw = {"action_kind_idx": _KINDS.index(kind) / (len(_KINDS) - 1),
               "is_external": 1.0 if is_ext else 0.0, "is_vessel": 1.0 if is_ves else 0.0,
               "cum_wait_s": None, "long_wait_excess_s": None, "predicted_arrival_gap_s": None,
               "eta_confidence": None, "deadline_slack_s": None, "reach_s": 0.0,
               "expected_service_time_s": 0.0, "expected_handling_count": 0.0, "blocker_count": 0.0,
               "expected_rehandle_time_s": 0.0, "end_bay": yc.state.position_bay,
               "lane_congestion_local": 0.0, "interference_penalty_s": 0.0,
               "resequence_count": 0.0, "vessel_risk_delta": None,
               "contention_risk": 0.0}
        return raw, {}
    reach = estimate_reach_s(spec, geom, yc.state.position_bay, yc.state.trolley_row,
                             plan.end_bay, geom.transfer_row)
    j = sim.jobs.get(ref.job_id)
    # PRE_REHANDLE 은 미도착 미래 트럭 대상 → 누적대기 없음(None); SERVE 외부만 cum.
    cum = sim.cum_wait(ref.job_id) if (kind == CandidateKind.SERVE and ref.is_external) else None
    # YR-043: PRE_REHANDLE 의 "도착까지 남은 시간" 을 mask 가 아니라 feature 로 제공 →
    # RL 이 선처리 가치를 학습. 계약이 tok=PRE_ADVICE 로 게이팅하므로 누출 없음.
    eta_gap = (j.provided_eta - now if (kind == CandidateKind.PRE_REHANDLE and j is not None
                                        and j.provided_eta is not None) else None)
    raw = {
        "action_kind_idx": _KINDS.index(kind) / (len(_KINDS) - 1),
        "is_external": 1.0 if ref.is_external else 0.0,
        "is_vessel": 1.0 if ref.is_vessel else 0.0,
        "cum_wait_s": cum,
        "long_wait_excess_s": max(0.0, cum - sim.profile.long_wait_sla_s) if cum is not None else None,
        "predicted_arrival_gap_s": eta_gap, "eta_confidence": None,
        "deadline_slack_s": (j.deadline - now) if (ref.is_vessel and j and j.deadline is not None) else None,
        "reach_s": reach, "expected_service_time_s": plan.duration_s,
        "expected_handling_count": float(len(plan.moves)), "blocker_count": float(plan.rehandles),
        "expected_rehandle_time_s": _rehandle_time(plan), "end_bay": plan.end_bay,
        "lane_congestion_local": _lane_local(sim, ref.lane_id),
        "interference_penalty_s": 0.0, "resequence_count": 0.0, "vessel_risk_delta": None,
        "contention_risk": _contention_risk(sim, cid, gc),
    }
    realized = {}
    if cum is not None and j is not None and j.actual_block_arrival is not None:
        realized = {"cum_wait_s": j.actual_block_arrival, "long_wait_excess_s": j.actual_block_arrival}
    return raw, realized


def _build_candidate_set(sim, cid, gen, now, level, ablation_off, k_max):
    items = []
    svc, reach_l, waits, outbound, work = [], [], [], 0, 0
    for gc in gen.items:
        raw, realized = _cand_raw(sim, cid, gc, now)
        fv = build_feature_vector("candidate", raw, now=now, info_level=level,
                                  realized_at=realized, ablation_off=ablation_off)
        ref = gc.job_ref
        items.append(Candidate(
            candidate_id=gc.candidate_id, kind=gc.kind, features=fv, mandatory=gc.mandatory,
            ref_job_id=(ref.job_id if (ref and gc.kind in (CandidateKind.SERVE, CandidateKind.PRE_REHANDLE))
                        else None),
            resolver_token=(ref.token if ref else None),
            eligible_crane_ids=(ref.eligible_crane_ids if ref else ()),
            lane_id=(ref.lane_id if ref else None)))
        if gc.kind != CandidateKind.WAIT and gc.plan is not None:
            work += 1
            svc.append(gc.plan.duration_s)
            reach_l.append(raw["reach_s"])
            if gc.kind == CandidateKind.SERVE and ref.is_external:
                c = sim.cum_wait(ref.job_id)
                waits.append(c)
                if (sim.jobs.get(ref.job_id) or gc.job_ref) and gc.job_ref.target_container is not None:
                    outbound += 1
    n_real = len(items)
    med = sorted(svc)[len(svc) // 2] if svc else 0.0
    qraw = {
        "cand_count": float(work),
        "service_min_s": min(svc) if svc else 0.0, "service_mean_s": (sum(svc) / len(svc)) if svc else 0.0,
        "service_max_s": max(svc) if svc else 0.0,
        "reach_min_s": min(reach_l) if reach_l else 0.0,
        "reach_mean_s": (sum(reach_l) / len(reach_l)) if reach_l else 0.0,
        "wait_max_s": max(waits) if waits else 0.0,
        "wait_mean_s": (sum(waits) / len(waits)) if waits else 0.0,
        "outbound_share": (outbound / work) if work else 0.0,
        "short_service_share": (sum(1 for s in svc if s <= med) / len(svc)) if svc else 0.0,
        "vessel_urgency_max": _max_vessel_risk(sim, now),
        "lane_cong_mean": _c01(sim.lanes.occupancy(
            frozenset(r.lane_id for r in sim.reservations.active() if r.lane_id))[0]),
        "over_sla_count": float(sum(1 for w in waits if w >= sim.profile.long_wait_sla_s)),
    }
    qfv = build_feature_vector("queue", qraw, now=now, info_level=level, ablation_off=ablation_off)
    # mandatory 전량 보존으로 실후보가 k_max 를 넘을 수 있다 (YR-044) → K 는 max(k_max, n_real).
    pad = tuple(padding_candidate(n_real + i) for i in range(max(0, k_max - n_real)))
    pad_mask = (True,) * n_real + (False,) * len(pad)
    feasible = tuple(gc.feasible for gc in gen.items) + (False,) * len(pad)
    reason = tuple(gc.mask_reason for gc in gen.items) + ("PADDING",) * len(pad)
    return CandidateSet(cid, tuple(items) + pad, pad_mask, feasible, reason, qfv)


# ------------------------------------------------------------ capture + record
def _scan_audit(state: GlobalState, obs) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing, assumed = [], []

    def scan(path, fv):
        for nm, kn, asm in zip(fv.names, fv.known, fv.assumed):
            if not kn:
                missing.append(f"{path}.{nm}")
            if asm:
                assumed.append(f"{path}.{nm}")

    scan("state", state.features)
    for i, v in enumerate(state.vessels):
        scan(f"state.vessels[{i}]", v.features)
    for o in obs:
        scan(f"obs[{o.crane_id}]", o.features)
        scan(f"obs[{o.crane_id}].queue", o.candidates.queue_summary)
        for c in o.candidates.items:
            scan(f"obs[{o.crane_id}].cand[{c.candidate_id}]", c.features)
    return tuple(sorted(missing)), tuple(sorted(assumed))


def capture(sim, crane_ids, level, episode_id, k, ablation_off=(), generator=None):
    """결정 시점 상태 s_k 를 계약 객체로 (state, observations, gen_by_crane).

    후보는 CandidateGenerator 가 4종(SERVE/PRE_REHANDLE/REPOSITION/WAIT)·mandatory·padding 으로
    생성. gen_by_crane 을 resolver 에 전달(candidate_id 직수, YR-037).
    """
    gen = generator or _GEN
    now = sim.now
    gfv = build_feature_vector("global", _global_raw(sim, now), now=now,
                               info_level=level, ablation_off=ablation_off)
    vessels = tuple(_vessel_urgency(sim, sim.vessels[vid], now, level, ablation_off)
                    for vid in sorted(sim.vessels))
    state = GlobalState(SCHEMA_VERSION, episode_id, k, now, level.value,
                        ControlScope.PLUS_PRE_REHANDLE.value, sim.profile.assumed,
                        gfv, vessels, sim.profile.lane_graph)
    obs, gen_by = [], {}
    for cid in crane_ids:
        yc_raw, yc_real = _yc_raw(sim, cid, now)
        yfv = build_feature_vector("yc", yc_raw, now=now, info_level=level,
                                   realized_at=yc_real, ablation_off=ablation_off)
        gc = gen.generate(sim, cid, level)
        cs = _build_candidate_set(sim, cid, gc, now, level, ablation_off, gen.k_max)
        obs.append(LocalObservation(SCHEMA_VERSION, cid, now, yfv, cs))
        gen_by[cid] = gc
    return state, tuple(obs), gen_by


def _assemble(state, obs, cranes_k, assigns, raw, dt, next_state, next_obs, terminal,
              episode_id, k, level, ablation_off, ehash, calc=None) -> TransitionRecord:
    joint = JointAction(SCHEMA_VERSION, state.now_s, tuple(
        Assignment(cid, assigns[cid][0], assigns[cid][1],
                   "yield" if assigns[cid][1] == CandidateKind.WAIT else "central_resolver")
        for cid in sorted(cranes_k)))
    rc = calc or _DEFAULT_RC
    cost = rc.cost_for(interval_start_s=state.now_s, interval_end_s=state.now_s + dt,
                       raw=raw, risk_max=_max_vessel_risk_state(state))
    miss, asm = _scan_audit(state, obs)
    audit = TransitionAudit(built_at_now_s=state.now_s, info_level=level.value,
                            ablation_off=tuple(sorted((a.value if hasattr(a, "value") else a)
                                                      for a in ablation_off)),
                            missing_fields=miss, assumed_fields=asm, forbidden_touched=(),
                            event_stream_hash=ehash)
    rec = TransitionRecord(SCHEMA_VERSION, episode_id, k, dt_s=dt, state=state,
                           observations=obs, joint_action=joint, cost=cost,
                           next_state=next_state, next_observations=next_obs,
                           terminal=terminal, audit=audit)
    validate_all(rec)
    canon = loads(dumps(rec))
    validate_all(canon)
    return canon


def _max_vessel_risk_state(state) -> float:
    best = 0.0
    for v in state.vessels:
        val, kn, _ = v.features.channel("risk")
        if kn:
            best = max(best, val)
    return best


def record_episode(sim, dispatcher=None, *, info_level: InformationLevel, episode_id: str,
                   ablation_off=(), generator=None, reward_calc=None) -> list[TransitionRecord]:
    """중앙 resolver 로 완주하며 결정마다 validate_all 통과 TransitionRecord 산출 (YR-037).

    dispatcher 를 주면 그 tie-break 규칙(DispatcherPreference)을, 없으면 BaselinePreference.
    reward_calc 를 주면 그 비용 config 로, 없으면 기본 assumed config (YR-038).
    """
    gen = generator or _GEN
    resolver = CentralResolver(DispatcherPreference(dispatcher) if dispatcher else BaselinePreference())
    sim.info_level = info_level
    records: list[TransitionRecord] = []
    dp = sim.run_until_decision()
    sim.cost.cut()   # [0, t0) 대기비용은 선행 결정 없음 — 폐기
    cur = capture(sim, dp.crane_ids, info_level, episode_id, 0, ablation_off, gen) if dp else None
    k = 0
    while dp is not None:
        state, obs, gen_by = cur
        cranes_k, t_k = dp.crane_ids, dp.time
        res = resolver.resolve(sim, dp, gen_by)
        resolver.apply(sim, res, gen_by)
        assigns = {r.crane_id: ((None, CandidateKind.WAIT) if r.action == CandidateKind.WAIT
                                else (r.chosen_candidate_id, r.action)) for r in res.resolutions}
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        nxt = capture(sim, dp.crane_ids, info_level, episode_id, k + 1, ablation_off, gen) if dp else None
        rec = _assemble(state, obs, cranes_k, assigns, raw, sim.now - t_k,
                        nxt[0] if nxt else None, nxt[1] if nxt else (), dp is None,
                        episode_id, k, info_level, ablation_off, sim.event_stream_hash(), reward_calc)
        records.append(rec)
        cur = nxt
        k += 1
    return records

