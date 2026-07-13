"""Replay recorder — 구현계획 04 §3 데이터 계약 (UI-1, YR-015-a).

의사결정 루프를 직접 돌며 manifest·decisions(+snapshot)·events 를 기록한다.
- 읽기 전용: 기록이 시뮬레이션 결과를 바꾸지 않음은 테스트로 보장 (04 §8.2).
- 정보 필터: 대기열·후보는 env 의 visible pool 만 저장 — 미공개 작업은
  hidden_job_count 로 개수만 남긴다 (04 §3.2, Exp-1/2 화면 누출 방지).
- 저장 형식: 단일 replay.json (04 는 Parquet 권고 — MVP 는 의존성 최소화를
  위해 JSON, 스키마는 §3 준수. 크기가 문제되면 Parquet 전환).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..domain.enums import JobStatus, PriorityRule
from ..envs.yard_env import YardEnv
from ..domain.scenario import Scenario
from .runner import collect_metrics


def _stack_heights(sim) -> list[list[int]]:
    """bay(1..B) × row(1..R) top tier 행렬 — 야드 평면도 채움용."""
    g = sim.profile.block
    return [[sim.stacks.top_tier(b, r) for r in range(1, g.row_count + 1)]
            for b in range(1, g.bay_count + 1)]


def _event_stream_hash(scenario: Scenario) -> str:
    """같은 시나리오(진실 이벤트) 여부 검증용 — Baseline/RL 비교 가드 (04 §3.1)."""
    payload = json.dumps(
        [(j.job_id, j.flow.value, j.release_time, j.actual_block_arrival, j.deadline)
         for j in sorted(scenario.jobs, key=lambda x: x.job_id)], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _job_meta(sim) -> dict:
    """작업 정적 메타 (tooltip·위치 표시용). 원본 식별자 없음 — 합성 ID 그대로."""
    out = {}
    for j in sim.jobs.values():
        c = sim.stacks.containers.get(j.target_container) if j.target_container else None
        out[j.job_id] = {
            "flow": j.flow.value,
            "external": j.is_external_truck,
            "vessel": j.is_vessel_linked,
            # decisions 의 t 와 동일한 0.1s 반올림 — round 는 단조라 순서 보존
            "arrival": (round(j.actual_block_arrival, 1)
                        if j.actual_block_arrival is not None else None),
            "deadline": round(j.deadline, 1) if j.deadline is not None else None,
            "target_bay": c.bay if c else None,
            "target_row": c.row if c else None,
        }
    return out


def _q_row(policy, state) -> dict[str, float] | None:
    """QL 정책이면 시도된 action 의 Q-value (해석 패널용). 아니면 None."""
    table = getattr(policy, "table", None)
    if table is None or state not in table.q:
        return None
    return {PriorityRule(a).name: round(q, 4)
            for a, (q, n) in enumerate(zip(table.q[state], table.n[state])) if n > 0}


def record_episode(policy, env: YardEnv, scenario: Scenario, *, run_id: str,
                   policy_name: str | None = None, out_dir: str | Path) -> Path:
    """에피소드 1개를 실행하며 replay 를 기록한다. 반환: replay.json 경로."""
    pname = policy_name or policy.name
    state, info = env.reset(scenario)
    sim = env.sim
    decisions: list[dict] = []
    i = 0
    while state is not None:
        t0 = sim.now
        bay0 = sim.crane.position_bay
        cands, _future = env._pools()
        queue = [{"job": j.job_id,
                  "wait_s": round(t0 - j.actual_block_arrival, 1),
                  "flow": j.flow.value}
                 for j in cands if j.is_external_truck
                 and j.status == JobStatus.WAITING]
        vessels = [j.job_id for j in cands if j.is_vessel_linked]
        hidden = sum(1 for j in sim.jobs.values()
                     if j.status == JobStatus.PLANNED) - len(_future)
        mask = info.action_mask
        s_before = state
        q_vals = _q_row(policy, s_before)
        a = policy.act(s_before, mask)
        state, _r, _done, info = env.step(a)
        k = sim.kpis
        decisions.append({
            "i": i, "t": round(t0, 1),
            "crane_bay_before": round(bay0, 2),
            "crane_bay_after": round(sim.crane.position_bay, 2),
            "task_end_t": round(sim.crane.available_at, 1),
            "rule": PriorityRule(a).name,
            "selected": info.selected_job,
            "mask": [PriorityRule(r).name for r, ok in enumerate(mask) if ok],
            "q_values": q_vals,
            "state_key": list(s_before),
            "queue": queue, "vessel_candidates": vessels,
            "hidden_job_count": hidden,
            "stack_heights": _stack_heights(sim),
            "kpis": {"queue_area_h": round(k.queue_area_s / 3600.0, 3),
                     "tail_area_h": round(k.tail_area_s / 3600.0, 3),
                     "travel_km": round((k.loaded_gantry_m + k.empty_gantry_m) / 1000.0, 3),
                     "rehandles": k.rehandle_count,
                     "vessel_delay_min": round(k.vessel_delay_s / 60.0, 2),
                     "completed": k.completed_external + k.completed_vessel,
                     "waiting_now": k.waiting_count()},
        })
        i += 1
    g = env.profile.block
    replay = {
        "manifest": {
            "run_id": run_id,
            "terminal_id": env.profile.terminal_id,
            "profile_assumed": env.profile.assumed,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "policy_id": pname,
            "info_level": env.level.value,
            "control_scope": env.scope.value,
            "event_stream_hash": _event_stream_hash(scenario),
            "horizon_s": scenario.horizon_s,
            "end_time_s": scenario.end_time,
            "sla_s": env.profile.long_wait_sla_s,
            "n_decisions": len(decisions),
            "final_metrics": collect_metrics(env),
            "block": {"bay_count": g.bay_count, "row_count": g.row_count,
                      "tier_max": g.tier_max, "transfer_row": g.transfer_row},
        },
        "jobs": _job_meta(sim),
        "decisions": decisions,
        "events": [(round(t, 1), kind, payload) for t, kind, payload in sim.event_log],
    }
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "replay.json"
    path.write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
    return path
