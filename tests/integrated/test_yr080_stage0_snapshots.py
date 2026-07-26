"""YR-080 단계 0 — 트럭 트랙 비파급 스냅샷 안전망.

본선(양하 방향·인과 연결) 수정이 **본선 없는 시나리오의 트럭 트랙을 1바이트도 바꾸지
않아야 한다**는 계약을 수정 전 실측값으로 동결한다 (구현 계획 단계 0, 2026-07-22).
- A/C: n_vessels=0 혼합 트럭 — 시나리오 생성(작업·컨테이너 배치) 해시 + SF_SPT 에피소드 지표.
- B: GATE_IN 전용(gate_out_share=0) — 반입(STORE) 경로 등가 리팩터(단계 1)의 기준.
값이 깨지면: RNG 스트림 누수(트럭 축 오염) 또는 리팩터 비등가 — 즉시 중단 신호.
"""
from __future__ import annotations

import hashlib

import pytest

from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario


def _snapshot(profile, params, seed):
    scn = generate_terminal_scenario(profile, seed, params)
    jhash = hashlib.sha256(repr([
        (j.job_id, j.flow.value, round(j.release_time, 3), j.actual_gate_in,
         j.actual_block_arrival, j.provided_eta, j.deadline, j.target_container,
         j.inbound_size, j.inbound_load) for j in scn.jobs]).encode()).hexdigest()[:16]
    chash = hashlib.sha256(repr(sorted(
        (c.container_id, c.bay, c.row, c.tier)
        for c in scn.containers.values())).encode()).hexdigest()[:16]
    sim = TerminalSimulator(profile, scn, check_invariants=True)
    r = run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                          RewardCalculator.assumed_default(),
                          generator=CandidateGenerator())
    return (jhash, chash, round(r["total_cost"], 6), r["n_decisions"],
            round(r["completion_rate"], 4), round(r["mean_wait_min"], 4))


# 실측 동결값 — YR-091/092 재동결 (2026-07-26, 외부감사 물리 정정): ①초기 pile 규격
# 1회 추첨(YR-092)이 배치·후속 draw 열 이동 ②비통과 초기 분산·idle 장벽(YR-091)이
# 이동/대기 동역학 변경. 세 케이스 완주 1.0 유지 — 물리 강화 후에도 트랙 건전.
CASES = [
    ("poc_novessel", build_integrated_profile,
     dict(n_external=20, n_vessels=0), 880001,
     ("3515de849b093c8b", "42bd0b524a01a7c9", 32.179371, 16, 1.0, 0.0)),
    ("poc_gatein_only", build_integrated_profile,
     dict(n_external=16, n_vessels=0, gate_out_share=0.0), 880002,
     ("b41e53dbf2b728d3", "c815a4c3392a710b", 27.536664, 13, 1.0, 0.0)),
    ("calibrated_novessel", build_calibrated_profile,
     dict(n_external=20, n_vessels=0), 880003,
     ("a7cd939e17852fac", "d6a902c9fa746053", 53.021938, 22, 1.0, 0.0)),
]


@pytest.mark.parametrize("label,profile_fn,params_kw,seed,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_truck_track_frozen(label, profile_fn, params_kw, seed, expected):
    got = _snapshot(profile_fn(), TerminalGenParams(**params_kw), seed)
    assert got == expected, (
        f"{label}: 트럭 트랙 스냅샷 불일치 — 본선 수정이 본선 없는 시나리오를 오염. "
        f"기대 {expected} / 실측 {got}")
