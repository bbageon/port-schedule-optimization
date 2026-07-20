"""YR-075-a — 재조작 목적지 오라클 (미래 반출시각 인지 배치).

CTDE·평가 전용 지위: 시나리오 jobs 에서 전지적으로 각 컨테이너의 반출시각을 뽑아,
방해 컨테이너를 "자기보다 먼저 반출될 컨테이너 위에 얹지 않도록" 배치한다 — 미래
재조작(새 blocker) 생성을 최소화. greedy(find_slot)와 같은 후보 슬롯 집합을 보되
정렬 기준만 (미래 blocked 수, greedy 비용) 사전식으로 교체. 결정론(bay,row tie).
"""
from __future__ import annotations

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
