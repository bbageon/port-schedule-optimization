"""YR-047 등가성 가드 — `_plan` 이 전체 스택 deepcopy 없이 구버전과 같은 계획을 내는가.

구버전(deepcopy 기반)을 대조군으로 본 파일에 박제하고, 에피소드의 **모든 결정·모든 후보**에서
신·구 JobPlan 을 비교한다. 등가 논거: 가상 진행의 격리는 exclude 집합이 운반한다 —
blocker 목적지는 배치 즉시 exclude 에 들어가고 원천 스택은 호출마다 제외되므로, 가상으로
변형된 자리를 이후 읽기(find_slot/top_tier)가 다시 보지 않는다.

**예외 — PRE_REHANDLE 은 의도된 차이 (적대 리뷰 발견)**: 구버전은 place() 가 b.bay/b.row 를
목적지로 덮어쓴 '뒤' 장부를 기록하는 별칭 버그로 원천 pile 을 slots·corridor 에서 누락했다
(과소예약 — SERVE 는 target 반출이 같은 pile 을 재추가해 우연히 은폐, PRE_REHANDLE 만 노출).
신버전은 원천 pile 을 포함한다. 그 외 필드(moves·시간·거리·rehandles)는 PRE_REHANDLE 도 동일.
"""
import copy
from dataclasses import replace

from yard_rl.contract import CandidateKind
from yard_rl.domain.enums import InformationLevel, JobFlow
from yard_rl.domain.models import Container
from yard_rl.integrated import (BaselinePreference, CandidateGenerator, CentralResolver,
                                TerminalSimulator, build_integrated_profile,
                                build_minimal_terminal_scenario)
from yard_rl.integrated.jobplan import JobPlan, Move
from yard_rl.integrated.scenario_gen import generate_terminal_scenario
from yard_rl.sim.travel_time import gantry_m, move_container, trolley_m

PROF = build_integrated_profile()
GEN = CandidateGenerator()
LEVEL = InformationLevel.PRE_ADVICE


def _plan_reference(sim, crane_id, ref, *, extra_exclude=frozenset()):
    """YR-047 이전 `_plan` 원문 (deepcopy 기반) — 등가성 대조군. 수정 금지."""
    yc = sim.fleet.get(crane_id)
    spec = sim.fleet.spec(crane_id)
    geom = sim.profile.block
    cur_bay, cur_row = yc.state.position_bay, yc.state.trolley_row

    if ref.kind == CandidateKind.REPOSITION:
        tb = ref.reposition_target_bay
        if tb is None:
            return None
        tb = float(min(max(tb, spec.service_bay_min), spec.service_bay_max))
        dist = gantry_m(geom, cur_bay, tb)
        t_dist = trolley_m(geom, cur_row, geom.transfer_row)
        dur = dist / spec.gantry_speed_mps + t_dist / spec.trolley_speed_mps
        return JobPlan(crane_id=crane_id, job_id=ref.job_id, token=None,
                       kind=CandidateKind.REPOSITION, moves=(),
                       corridor=(min(cur_bay, tb), max(cur_bay, tb)), slots=frozenset(),
                       lane_id=None, start_s=sim.clock, duration_s=dur, end_bay=tb,
                       end_row=float(geom.transfer_row), rehandles=0,
                       loaded_gantry_m=0.0, empty_gantry_m=dist)

    work = copy.deepcopy(sim.stacks)
    exclude = set(sim.reservations.reserved_slots()) | set(extra_exclude)
    moves = []
    touched_bays = {cur_bay}
    slots = set()
    total_s = loaded_m = empty_m = 0.0
    rehandles = 0
    j = sim.jobs[ref.job_id]

    if j.flow == JobFlow.GATE_IN and ref.kind == CandidateKind.SERVE:
        dest = work.find_slot(j.inbound_size, spec, cur_bay, cur_row, exclude=frozenset(exclude))
        if dest is None:
            return None
        db, dr = dest
        dtier = work.top_tier(db, dr) + 1
        src = (db, geom.transfer_row, 1)
        mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
        inbound = Container(container_id=f"IN_{j.job_id}", size=j.inbound_size,
                            load_status=j.inbound_load, block=geom.block_id,
                            bay=db, row=dr, tier=dtier)
        moves.append(Move(inbound.container_id, src, (db, dr, dtier),
                          mv.loaded_gantry_m, mv.empty_gantry_m, mv.duration_s, inbound=inbound))
        total_s += mv.duration_s + spec.truck_positioning_time_s
        loaded_m += mv.loaded_gantry_m
        empty_m += mv.empty_gantry_m
        cur_bay, cur_row = mv.end_bay, mv.end_row
        touched_bays |= {db}
        slots.add((db, dr))
    else:
        target_id = j.target_container
        for blocker_id in work.blockers_above(target_id):
            b = work.containers[blocker_id]
            src = (b.bay, b.row, b.tier)
            dest = work.find_slot(b.size, spec, float(b.bay), float(b.row),
                                  exclude=frozenset(exclude | {(b.bay, b.row)}))
            if dest is None:
                return None
            db, dr = dest
            dtier = work.top_tier(db, dr) + 1
            work.remove(blocker_id)
            mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
            work.place(b, db, dr)
            moves.append(Move(blocker_id, src, (db, dr, dtier),
                              mv.loaded_gantry_m, mv.empty_gantry_m, mv.duration_s))
            total_s += mv.duration_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row
            rehandles += 1
            exclude.add((db, dr))
            touched_bays |= {b.bay, db}
            slots |= {(b.bay, b.row), (db, dr)}
        if ref.kind == CandidateKind.SERVE:
            target = work.containers[target_id]
            src = (target.bay, target.row, target.tier)
            dst = (target.bay, geom.transfer_row, 1)
            mv = move_container(spec, geom, cur_bay, cur_row, src, dst)
            moves.append(Move(target_id, src, dst, mv.loaded_gantry_m, mv.empty_gantry_m,
                              mv.duration_s, depart=True))
            total_s += mv.duration_s
            if j.is_external_truck:
                total_s += spec.truck_positioning_time_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row
            touched_bays |= {target.bay}
            slots.add((target.bay, target.row))

    lo, hi = min(touched_bays), max(touched_bays)
    return JobPlan(crane_id=crane_id, job_id=j.job_id, token=ref.token, kind=ref.kind,
                   moves=tuple(moves), corridor=(lo, hi), slots=frozenset(slots),
                   lane_id=ref.lane_id, start_s=sim.clock, duration_s=total_s,
                   end_bay=cur_bay, end_row=cur_row, rehandles=rehandles,
                   loaded_gantry_m=loaded_m, empty_gantry_m=empty_m)


def _stack_fingerprint(sim):
    """pile 구성 + 컨테이너 좌표 필드까지 — 필드 변형(bay/row/tier 오염)도 잡는다 (리뷰 반영)."""
    return ({k: tuple(v) for k, v in sim.stacks._stacks.items()},
            {cid: (c.bay, c.row, c.tier) for cid, c in sim.stacks.containers.items()})


def _assert_plans_match(cid, ref, new, old):
    """신==구. 단 PRE_REHANDLE 은 slots·corridor 만 의도된 차이(원천 pile 포함) 허용."""
    if ref.kind != CandidateKind.PRE_REHANDLE or new is None or old is None:
        assert new == old, f"{cid}:{ref.job_id} 신·구 계획 불일치"
        return
    assert replace(new, slots=frozenset(), corridor=(0.0, 0.0)) == \
           replace(old, slots=frozenset(), corridor=(0.0, 0.0)), \
        f"{cid}:{ref.job_id} PRE_REHANDLE slots/corridor 외 필드 불일치"
    assert new.slots >= old.slots                     # 신버전 = 구버전 + 원천 pile (과소예약 수정)
    assert new.corridor[0] <= old.corridor[0] and new.corridor[1] >= old.corridor[1]


def _compare_episode(sim, *, with_extra_exclude=False):
    """에피소드 완주하며 매 결정·매 후보에서 신·구 비교 + 스택 미변형 검증. (비교수, 재조작수) 반환."""
    r = CentralResolver(BaselinePreference())
    sim.info_level = LEVEL
    n_cmp = n_rehandle = 0
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            return n_cmp, n_rehandle
        gen_by = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}
        before = _stack_fingerprint(sim)
        for cid in dp.crane_ids:
            extra = frozenset()
            if with_extra_exclude:                    # dry_run 순차예약 경로 재현
                extra = frozenset(list(sim.stacks._stacks)[:3])
            for gc in gen_by[cid].items:
                if gc.job_ref is None:                # WAIT 는 _plan 미경유
                    continue
                new = sim._plan(cid, gc.job_ref, extra_exclude=extra)
                old = _plan_reference(sim, cid, gc.job_ref, extra_exclude=extra)
                _assert_plans_match(cid, gc.job_ref, new, old)
                n_cmp += 1
                if new is not None and new.rehandles > 0:
                    n_rehandle += 1
        assert _stack_fingerprint(sim) == before      # _plan 은 스택을 읽기만 한다
        resn = r.resolve(sim, dp, gen_by)
        r.apply(sim, resn, gen_by)


def test_plan_equivalent_minimal_scenario():
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario())
    n_cmp, _ = _compare_episode(sim)
    assert n_cmp > 0


def test_plan_equivalent_generated_congested():
    """생성 시나리오 (혼잡 — 재조작 경로 포함 강제)."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, 310000))
    n_cmp, n_rehandle = _compare_episode(sim)
    assert n_cmp > 50
    assert n_rehandle > 0, "재조작(blocker) 경로가 한 번도 비교되지 않음 — 커버리지 무효"


def test_plan_equivalent_with_extra_exclude():
    """dry_run_commit 의 순차예약(extra_exclude) 경로도 등가."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, 310007))
    n_cmp, _ = _compare_episode(sim, with_extra_exclude=True)
    assert n_cmp > 0


def test_pre_rehandle_divergence_is_intended_and_bounded():
    """PRE_REHANDLE 은 등가가 아니라 **의도된 수정** — 구버전은 place() 별칭 버그로 원천 pile 을
    예약(slots·corridor)에서 누락했다(과소예약). 신버전은 포함한다. 그 외 필드는 전부 동일.

    현행 통합 시나리오는 provided_eta 미설정이라 PRE_REHANDLE 후보가 생성되지 않으므로(커버리지
    공백 — 리뷰 발견), blocker 있는 SERVE ref 에서 PRE_REHANDLE ref 를 합성해 직접 비교한다.
    """
    compared = 0
    for build in (lambda: TerminalSimulator(PROF, build_minimal_terminal_scenario()),
                  lambda: TerminalSimulator(PROF, generate_terminal_scenario(PROF, 310000))):
        sim = build()
        sim.info_level = LEVEL
        r = CentralResolver(BaselinePreference())
        while compared < 5:
            dp = sim.run_until_decision()
            if dp is None:
                break
            gen_by = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}
            for cid in dp.crane_ids:
                for gc in gen_by[cid].items:
                    ref = gc.job_ref
                    if (ref is None or ref.kind != CandidateKind.SERVE
                            or ref.target_container is None
                            or ref.target_container not in sim.stacks.containers
                            or not sim.stacks.blockers_above(ref.target_container)):
                        continue
                    pr = replace(ref, kind=CandidateKind.PRE_REHANDLE)
                    new, old = sim._plan(cid, pr), _plan_reference(sim, cid, pr)
                    if new is None or old is None:
                        assert new == old
                        continue
                    _assert_plans_match(cid, pr, new, old)
                    t = sim.stacks.containers[ref.target_container]
                    assert (t.bay, t.row) in new.slots        # 신: 원천 pile 예약 (수정의 실체)
                    assert (t.bay, t.row) not in old.slots    # 구: 별칭 버그로 누락 (박제)
                    compared += 1
            resn = r.resolve(sim, dp, gen_by)
            r.apply(sim, resn, gen_by)
        if compared >= 5:
            break
    assert compared > 0, "blocker 있는 PRE_REHANDLE 이 한 번도 비교되지 않음 — 커버리지 무효"


def test_plan_determinism_across_repeats():
    """같은 상태에서 반복 호출해도 같은 계획 (읽기 전용이므로 자기영향 0)."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, 310000))
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    for cid in dp.crane_ids:
        for gc in GEN.generate(sim, cid, LEVEL).items:
            if gc.job_ref is None:
                continue
            assert sim._plan(cid, gc.job_ref) == sim._plan(cid, gc.job_ref)
