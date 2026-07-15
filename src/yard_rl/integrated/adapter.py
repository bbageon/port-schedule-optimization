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
                        make_cost, resolve_mode, validate_all)
from ..domain.enums import ControlScope, InformationLevel, JobFlow, JobStatus
from ..sim.travel_time import estimate_reach_s
from .cost import ASSUMED_SCALE, ASSUMED_WEIGHT, assumed_lambda_vessel
from .engine import _pstdev

_KINDS = list(CandidateKind)
_WAITING = (JobStatus.WAITING, JobStatus.RELEASED)


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
        "load_imbalance": _pstdev([c.served_count for c in sim.fleet.all()]),
    }


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
    return VesselUrgency(v.vessel_id, mode, v.plan.completion_basis, assumed, fv)


def _candidates(sim, cid: str, refs, now: float, level: InformationLevel, ablation_off=()):
    spec = sim.fleet.spec(cid)
    geom = sim.profile.block
    yc = sim.fleet.get(cid)
    plans = [sim._plan(cid, r) for r in refs]
    svc_all = sorted(p.duration_s for p in plans)
    median = svc_all[len(svc_all) // 2] if svc_all else 0.0
    items, svc, reach_l, waits, outbound, short = [], [], [], [], 0, 0
    for i, (ref, plan) in enumerate(zip(refs, plans)):
        j = sim.jobs[ref.job_id]
        reach = estimate_reach_s(spec, geom, yc.state.position_bay, yc.state.trolley_row,
                                 plan.end_bay, geom.transfer_row)
        cum = sim.cum_wait(ref.job_id) if ref.is_external else None
        raw = {
            "action_kind_idx": _KINDS.index(ref.kind) / (len(_KINDS) - 1),
            "is_external": 1.0 if ref.is_external else 0.0,
            "is_vessel": 1.0 if ref.is_vessel else 0.0,
            "cum_wait_s": cum,
            "long_wait_excess_s": max(0.0, cum - sim.profile.long_wait_sla_s) if cum is not None else None,
            "predicted_arrival_gap_s": None, "eta_confidence": None,
            "deadline_slack_s": (j.deadline - now) if (ref.is_vessel and j.deadline is not None) else None,
            "reach_s": reach, "expected_service_time_s": plan.duration_s,
            "expected_handling_count": float(len(plan.moves)),
            "blocker_count": float(plan.rehandles),
            "expected_rehandle_time_s": _rehandle_time(plan), "end_bay": plan.end_bay,
            "lane_congestion_local": _lane_local(sim, ref.lane_id),
            "interference_penalty_s": 0.0, "resequence_count": 0.0, "vessel_risk_delta": None,
        }
        realized = {}
        if ref.is_external and j.actual_block_arrival is not None:
            realized = {"cum_wait_s": j.actual_block_arrival,
                        "long_wait_excess_s": j.actual_block_arrival}
        fv = build_feature_vector("candidate", raw, now=now, info_level=level,
                                  realized_at=realized, ablation_off=ablation_off)
        items.append(Candidate(candidate_id=i, kind=ref.kind, features=fv,
                               mandatory=bool(cum is not None and cum >= sim.profile.long_wait_sla_s),
                               ref_job_id=ref.job_id, resolver_token=ref.token,
                               eligible_crane_ids=ref.eligible_crane_ids, lane_id=ref.lane_id))
        svc.append(plan.duration_s)
        reach_l.append(reach)
        if cum is not None:
            waits.append(cum)
        if ref.is_external and j.flow == JobFlow.GATE_OUT:
            outbound += 1
        if plan.duration_s <= median:
            short += 1
    n = len(items)
    qraw = {
        "cand_count": float(n),
        "service_min_s": min(svc) if svc else 0.0, "service_mean_s": (sum(svc) / n) if n else 0.0,
        "service_max_s": max(svc) if svc else 0.0,
        "reach_min_s": min(reach_l) if reach_l else 0.0, "reach_mean_s": (sum(reach_l) / n) if n else 0.0,
        "wait_max_s": max(waits) if waits else 0.0, "wait_mean_s": (sum(waits) / len(waits)) if waits else 0.0,
        "outbound_share": (outbound / n) if n else 0.0, "short_service_share": (short / n) if n else 0.0,
        "vessel_urgency_max": _max_vessel_risk(sim, now), "lane_cong_mean": _c01(
            sim.lanes.occupancy(frozenset(r.lane_id for r in sim.reservations.active() if r.lane_id))[0]),
        "over_sla_count": float(sum(1 for w in waits if w >= sim.profile.long_wait_sla_s)),
    }
    qfv = build_feature_vector("queue", qraw, now=now, info_level=level, ablation_off=ablation_off)
    cs = CandidateSet(cid, tuple(items), (True,) * n, (True,) * n, (None,) * n, qfv)
    tok_idx = {c.resolver_token: c.candidate_id for c in items}
    return cs, tok_idx


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


def capture(sim, crane_ids, level, episode_id, k, ablation_off=()):
    """결정 시점 상태 s_k 를 계약 객체로 (state, observations, token→index map)."""
    now = sim.now
    gfv = build_feature_vector("global", _global_raw(sim, now), now=now,
                               info_level=level, ablation_off=ablation_off)
    vessels = tuple(_vessel_urgency(sim, sim.vessels[vid], now, level, ablation_off)
                    for vid in sorted(sim.vessels))
    state = GlobalState(SCHEMA_VERSION, episode_id, k, now, level.value,
                        ControlScope.PLUS_PRE_REHANDLE.value, sim.profile.assumed,
                        gfv, vessels, sim.profile.lane_graph)
    obs, tok_map = [], {}
    for cid in crane_ids:
        yc_raw, yc_real = _yc_raw(sim, cid, now)
        yfv = build_feature_vector("yc", yc_raw, now=now, info_level=level,
                                   realized_at=yc_real, ablation_off=ablation_off)
        cs, tok_idx = _candidates(sim, cid, sim.candidates_for(cid), now, level, ablation_off)
        obs.append(LocalObservation(SCHEMA_VERSION, cid, now, yfv, cs))
        tok_map[cid] = tok_idx
    return state, tuple(obs), tok_map


def _assemble(state, obs, cranes_k, assigns, raw, dt, next_state, next_obs, terminal,
              episode_id, k, level, ablation_off, ehash) -> TransitionRecord:
    joint = JointAction(SCHEMA_VERSION, state.now_s, tuple(
        Assignment(cid, assigns[cid][0], assigns[cid][1],
                   "yield" if assigns[cid][1] == CandidateKind.WAIT else "central_resolver")
        for cid in sorted(cranes_k)))
    lam = assumed_lambda_vessel(_max_vessel_risk_state(state))
    cost = make_cost(interval_start_s=state.now_s, interval_end_s=state.now_s + dt,
                     raw=raw, scale=ASSUMED_SCALE, weight=ASSUMED_WEIGHT,
                     lambda_vessel=lam, assumed=True)
    miss, asm = _scan_audit(state, obs)
    audit = TransitionAudit(built_at_now_s=state.now_s, info_level=level.value,
                            ablation_off=tuple(sorted(str(a) for a in ablation_off)),
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


def record_episode(sim, dispatcher, *, info_level: InformationLevel, episode_id: str,
                   ablation_off=()) -> list[TransitionRecord]:
    """참조 디스패처로 완주하며 결정마다 validate_all 통과 TransitionRecord 산출."""
    records: list[TransitionRecord] = []
    dp = sim.run_until_decision()
    sim.cost.cut()   # [0, t0) 대기비용은 선행 결정 없음 — 폐기
    cur = capture(sim, dp.crane_ids, info_level, episode_id, 0, ablation_off) if dp else None
    k = 0
    while dp is not None:
        state, obs, tok_map = cur
        cranes_k, t_k = dp.crane_ids, dp.time
        assigns: dict[str, tuple] = {}
        from .engine import CraneAssignment
        for cid in cranes_k:
            live = sim.candidates_for(cid)
            if not live:
                sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))
                assigns[cid] = (None, CandidateKind.WAIT)
            else:
                ref = dispatcher.select(sim, cid, live)
                sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=ref))
                assigns[cid] = (tok_map[cid][ref.token], CandidateKind.SERVE)
        sim.close_decision()
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        nxt = capture(sim, dp.crane_ids, info_level, episode_id, k + 1, ablation_off) if dp else None
        rec = _assemble(state, obs, cranes_k, assigns, raw, sim.now - t_k,
                        nxt[0] if nxt else None, nxt[1] if nxt else (), dp is None,
                        episode_id, k, info_level, ablation_off, sim.event_stream_hash())
        records.append(rec)
        cur = nxt
        k += 1
    return records

