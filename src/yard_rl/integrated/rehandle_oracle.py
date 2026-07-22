"""YR-075-a — 재조작 목적지 오라클 (미래 반출시각 인지 배치).

CTDE·평가 전용 지위: 시나리오 jobs 에서 전지적으로 각 컨테이너의 반출시각을 뽑아,
방해 컨테이너를 "자기보다 먼저 반출될 컨테이너 위에 얹지 않도록" 배치한다 — 미래
재조작(새 blocker) 생성을 최소화. greedy(find_slot)와 같은 후보 슬롯 집합을 보되
정렬 기준만 (미래 blocked 수, greedy 비용) 사전식으로 교체. 결정론(bay,row tie).
"""
from __future__ import annotations

from ..domain.enums import JobStatus
from ..sim.travel_time import gantry_m, trolley_m

_INF = float("inf")


def future_retrieval_times(sim) -> dict:
    """container_id → 가장 이른 반출(제거) 시각. 무예약 컨테이너는 없음(=∞ 취급)."""
    out: dict[str, float] = {}
    for j in sim.jobs.values():
        c = j.target_container
        if c is None:
            continue
        if j.is_external_truck:                 # GATE_OUT: 블록 도착시각이 반출 시점
            t = (j.actual_block_arrival if j.actual_block_arrival is not None
                 else j.release_time)
        else:                                   # 본선 반출: release_time
            t = j.release_time
        if c not in out or t < out[c]:
            out[c] = t
    return out


def _ft(sim) -> dict:
    ft = getattr(sim, "_yr075_ft", None)
    if ft is None:
        ft = future_retrieval_times(sim)
        sim._yr075_ft = ft                       # 정적 — rollout deepcopy 도 승계
    return ft


def observable_retrieval_times(sim) -> dict:
    """배포형 — **관측 가능** 정보만: 외부트럭 provided_eta·본선 release_time.

    전지적 actual_block_arrival 을 안 쓴다 (미도착 트럭의 진짜 도착시각 미열람 =
    누출 0). ETA 오차(±)가 그대로 실려 오라클보다 신호가 흐리다 — 그 손실이
    "배포 가능성의 값" 이다. PRE_ADVICE 정보수준 가정.
    """
    out: dict[str, float] = {}
    for j in sim.jobs.values():
        c = j.target_container
        if c is None:
            continue
        if j.is_external_truck:
            t = j.provided_eta if j.provided_eta is not None else _INF
        else:
            t = j.release_time
        if c not in out or t < out[c]:
            out[c] = t
    return out


def _oft(sim) -> dict:
    oft = getattr(sim, "_yr075_oft", None)
    if oft is None:
        oft = observable_retrieval_times(sim)
        sim._yr075_oft = oft
    return oft


def _select(stk, blocker, spec, exclude, times: dict):
    geom = stk.geom
    b_time = times.get(blocker.container_id, _INF)
    best = None
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        for row in range(1, geom.row_count + 1):
            if (bay, row) in exclude:
                continue
            top = stk.top_tier(bay, row)
            if top >= geom.tier_max:
                continue
            if not stk.stack_size_ok(bay, row, blocker.size):
                continue
            future_blocked = sum(1 for cid in stk.stack(bay, row)
                                 if times.get(cid, _INF) < b_time)
            greedy_cost = (gantry_m(geom, float(blocker.bay), bay)
                           + trolley_m(geom, float(blocker.row), row)
                           + top * geom.tier_height_m)
            key = (future_blocked, greedy_cost, bay, row)
            if best is None or key < best:
                best = key
    return None if best is None else (best[2], best[3])


def deployable_future_selector(sim, stk, blocker, spec, exclude):
    """H1 — 배포형(관측 ETA·마감 인지) 목적지. 오라클과 동형, 시각원만 관측값."""
    return _select(stk, blocker, spec, exclude, _oft(sim))


def mask_only_selector(sim, stk, blocker, spec, exclude):
    """H1-mask — 사용자 제안: 방해물 생성 **딱딱한 금지**(future_blocked==0 만 허용),
    그 안에서 이동은 비용(=지금 거리)이 고르게. 비방해물 슬롯이 없으면 pure greedy
    폴백. H1(부드러운 사전식)과 고포화 불가능 케이스에서만 갈린다 (mask 는 폴백에서
    future_blocked 를 아예 무시).
    """
    times = _oft(sim)
    geom = stk.geom
    b_time = times.get(blocker.container_id, _INF)
    clean, dirty = None, None
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        for row in range(1, geom.row_count + 1):
            if (bay, row) in exclude:
                continue
            top = stk.top_tier(bay, row)
            if top >= geom.tier_max:
                continue
            if not stk.stack_size_ok(bay, row, blocker.size):
                continue
            cost = (gantry_m(geom, float(blocker.bay), bay)
                    + trolley_m(geom, float(blocker.row), row)
                    + top * geom.tier_height_m)
            key = (cost, bay, row)
            fb = sum(1 for cid in stk.stack(bay, row)
                     if times.get(cid, _INF) < b_time)
            if fb == 0 and (clean is None or key < clean):
                clean = key
            if dirty is None or key < dirty:
                dirty = key
    pick = clean if clean is not None else dirty
    return None if pick is None else (pick[1], pick[2])


def _next_demand_bay(stk, ft, now):
    """지금 이후 가장 이른 반출 컨테이너의 현재 bay = 크레인의 유력한 다음 목적지."""
    best = None
    for cid, t in ft.items():
        if t <= now:
            continue
        c = stk.containers.get(cid)
        if c is None:
            continue
        if best is None or t < best[0]:
            best = (t, c.bay)
    return best[1] if best else None


def strong_oracle_slot_selector(sim, stk, blocker, spec, exclude):
    """YR-075-a 0b — 강한 오라클: 비방해물(1차) + **미래 이동**(2차: 지금 이동 +
    다음 수요 방향 거리 = 선제 위치선점) 사전식. 사용자 3층 설계(비방해물→다음작업
    방향→최소이동)를 전지적 미래로 상한 측정. 기존 오라클은 '지금 이동'만 봤음.
    """
    ft = _ft(sim)
    geom = stk.geom
    b_time = ft.get(blocker.container_id, _INF)
    nd_bay = _next_demand_bay(stk, ft, sim.now)
    best = None
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        for row in range(1, geom.row_count + 1):
            if (bay, row) in exclude:
                continue
            top = stk.top_tier(bay, row)
            if top >= geom.tier_max:
                continue
            if not stk.stack_size_ok(bay, row, blocker.size):
                continue
            future_blocked = sum(1 for cid in stk.stack(bay, row)
                                 if ft.get(cid, _INF) < b_time)
            immediate = (gantry_m(geom, float(blocker.bay), bay)
                         + trolley_m(geom, float(blocker.row), row)
                         + top * geom.tier_height_m)
            fwd = gantry_m(geom, float(bay), nd_bay) if nd_bay is not None else 0.0
            key = (future_blocked, immediate + fwd, bay, row)
            if best is None or key < best:
                best = key
    return None if best is None else (best[2], best[3])


def oracle_slot_selector(sim, stk, blocker, spec, exclude):
    """미래 재조작 회피 배치. find_slot 과 동일 후보 집합·동일 유효성 규칙."""
    ft = _ft(sim)
    geom = stk.geom
    b_time = ft.get(blocker.container_id, _INF)
    best = None
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        for row in range(1, geom.row_count + 1):
            if (bay, row) in exclude:
                continue
            top = stk.top_tier(bay, row)
            if top >= geom.tier_max:
                continue
            if not stk.stack_size_ok(bay, row, blocker.size):
                continue
            # 미래 blocked = 이 더미에서 blocker 보다 먼저 반출될 컨테이너 수
            future_blocked = sum(1 for cid in stk.stack(bay, row)
                                 if ft.get(cid, _INF) < b_time)
            greedy_cost = (gantry_m(geom, float(blocker.bay), bay)
                           + trolley_m(geom, float(blocker.row), row)
                           + top * geom.tier_height_m)
            key = (future_blocked, greedy_cost, bay, row)
            if best is None or key < best:
                best = key
    return None if best is None else (best[2], best[3])


def deployable_store_selector(sim, stk, job, spec, near_bay, near_row, exclude):
    """신규 적재(STORE — 트럭 반입·본선 양하) 비방해 규칙 (YR-080 단계6, 기본 off).

    find_slot 과 동일 제약·거리비용 위에 **매장(burial) 페널티를 최우선 축**으로 추가:
    이 슬롯에 쌓으면 그 아래(현재 top)가 '아직 안 나간 반출 대상'인 슬롯을 피한다 —
    미래 재조작 예방 (재조작 H1 규칙의 적재 대응물). 관측만 사용(진실 미열람):
    jobs 의 target_container·야드 배치만 읽고 미래 도착·순서는 안 본다. 결정론:
    (매장, 비용, bay, row) 사전식 tie-break.
    """
    pending = {j.target_container for j in sim.jobs.values()
               if j.target_container is not None
               and j.status not in (JobStatus.DONE, JobStatus.CANCELLED)}
    best = None
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        for row in range(1, stk.geom.row_count + 1):
            if (bay, row) in exclude:
                continue
            top = stk.top_tier(bay, row)
            if top >= stk.geom.tier_max:
                continue
            if not stk.stack_size_ok(bay, row, job.inbound_size):
                continue
            pile = stk.stack(bay, row)
            buries = 1 if (pile and pile[-1] in pending) else 0
            cost = (gantry_m(stk.geom, near_bay, bay)
                    + trolley_m(stk.geom, near_row, row)
                    + top * stk.geom.tier_height_m)
            key = (buries, cost, bay, row)
            if best is None or key < best:
                best = key
    return None if best is None else (best[2], best[3])
