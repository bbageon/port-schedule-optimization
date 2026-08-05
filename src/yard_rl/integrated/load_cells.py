"""YR-149 5부하 셀 전용 생성기 — A/B 동결 template + A150 master 중첩추출 (동결 2026-08-05).

■ 왜 중첩추출인가 (가짜 pairing 금지)
`generate_terminal_scenario` 의 도착식은 `horizon × (i + u_i) / n_external` 이고 컨테이너·
대상·규격·본선이 **한 난수열을 공유**한다. 그래서 트럭 수만 바꿔 재생성하면 같은 정수 seed
라도 초기 적재부터 전 궤적이 달라진다 — 부하만 다른 짝비교가 성립하지 않는다(spec 금지).
따라서 **A150 master 를 한 번만 생성**하고 외부트럭만 150→125→100→75→50 으로 균등 솎기해
`A50 ⊂ A75 ⊂ A100 ⊂ A125 ⊂ A150` 을 만든다. 컨테이너·초기 적재·본선·B 블록은 master 그대로다.

■ 동결 template (두 블록 동일 — 트럭 유입 수만 다르다)
본선 2 process × 15 move · 장치율 0.30 · 도착 피크(문헌 보정) · 게이트 주행 210s ·
`gaussian=False`(트럭 수 정확 고정) · 시간계약 v2 · 게이트→블록 계약 · 달성 가능 본선 마감.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

from .profiles import build_calibrated_profile
from .scenario import TerminalScenario
from .scenario_gen import calibrated_load_params, generate_terminal_scenario

CELL_SIZES = (50, 75, 100, 125, 150)      # A 블록 4시간 유입 트럭 수 (동결)
B_SIZE = 50                               # B 블록 고정 (전 셀 동일)
MASTER_SIZE = max(CELL_SIZES)


def cell_params(n_external: int):
    """동결 template — n_external 만 다르다."""
    return calibrated_load_params(
        "mid", n_external=n_external, n_vessels=2, vessel_moves=15, fill_ratio=0.30,
        gaussian=False, time_contract_v2=True, gate_block_contract=True,
        vessel_deadline_achievable=True)


def generate_block(seed: int, n_external: int) -> TerminalScenario:
    return generate_terminal_scenario(build_calibrated_profile(), seed,
                                      cell_params(n_external))


def _thin(seq: list, m: int) -> list:
    """균등 솎기 — 도착 순서 상 등간격 m 개 (인덱스 floor(j·n/m) 는 j 에 대해 강증가)."""
    n = len(seq)
    if m >= n:
        return list(seq)
    return [seq[(j * n) // m] for j in range(m)]


def nested_external(ext: list, m: int) -> list:
    """150→125→100→75→50 사슬로만 솎아 **부분집합 관계**를 구조적으로 보장한다."""
    cur = list(ext)
    for s in sorted(CELL_SIZES, reverse=True):
        if s > len(cur):
            continue
        if s < m:
            break
        cur = _thin(cur, s)
    if len(cur) != m:
        raise ValueError(f"중첩추출 실패: {len(cur)} != {m} (허용 크기 {CELL_SIZES})")
    return cur


def make_a_cell(master_seed: int, m: int) -> TerminalScenario:
    """A 블록 셀 — master 를 매번 새로 생성해(같은 seed = 동일 master) 상태 공유를 피한다."""
    master = generate_block(master_seed, MASTER_SIZE)
    ext = [j for j in master.jobs if j.is_external_truck]
    keep = {j.job_id for j in nested_external(ext, m)}
    jobs = [j for j in master.jobs if (not j.is_external_truck) or j.job_id in keep]
    return dataclasses.replace(master, jobs=jobs,
                               scenario_id=f"{master.scenario_id}#A{m}")


def make_b_cell(seed: int) -> TerminalScenario:
    return generate_block(seed, B_SIZE)


# ------------------------------------------------------------------ 자격시험용 지문
def _v(x):
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    return getattr(x, "value", None) or str(x)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def layout_digest(scn: TerminalScenario) -> str:
    """초기 적재 배치 — 이송·부하와 무관하게 셀 간 동일해야 한다."""
    return _digest(sorted((c.container_id, _v(c.size), c.bay, c.row, c.tier)
                          for c in scn.containers.values()))


def vessel_digest(scn: TerminalScenario) -> str:
    """본선 일정·마감 — start/planned completion/ETD/moves/cadence/물리하한."""
    rows = []
    for v in scn.vessels:
        p = v.plan
        rows.append((v.vessel_id, _v(v.work_type), p.planned_start_s,
                     p.planned_completion_s, getattr(p, "etd_s", None),
                     p.total_moves, p.sts_move_interval_s,
                     getattr(p, "phys_min_completion_s", None)))
    return _digest(sorted(rows, key=lambda r: r[0]))


def ext_job_rows(scn: TerminalScenario) -> list:
    return sorted((j.job_id, _v(j.flow), j.actual_gate_in, j.actual_block_arrival,
                   j.provided_eta, getattr(j, "estimated_block_arrival", None),
                   getattr(j, "appointment_gate_time", None),
                   getattr(j, "exit_travel_s", None), j.target_container,
                   _v(j.inbound_size)) for j in scn.jobs if j.is_external_truck)


def ext_job_digest(scn: TerminalScenario) -> str:
    return _digest(ext_job_rows(scn))


def flow_ratio(scn: TerminalScenario) -> float:
    ext = [j for j in scn.jobs if j.is_external_truck]
    if not ext:
        return 0.0
    return sum(1 for j in ext if _v(j.flow) == "GATE_OUT") / len(ext)


def config_digest(scn: TerminalScenario) -> dict:
    """셀 간 대조용 config 지문 묶음 (spec 자격시험 요구 항목)."""
    ext = [j for j in scn.jobs if j.is_external_truck]
    return {"layout": layout_digest(scn), "vessels": vessel_digest(scn),
            "ext_jobs": ext_job_digest(scn), "n_external": len(ext),
            "flow_ratio_gate_out": round(flow_ratio(scn), 4),
            "n_vessel_jobs": sum(1 for j in scn.jobs if not j.is_external_truck),
            "horizon_s": scn.horizon_s, "drain_window_s": scn.drain_window_s}
