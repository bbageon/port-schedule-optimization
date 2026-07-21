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


# 수정 전 실측 동결값 (2026-07-22, 커밋 직전 산출 — 재산출 스크립트는 docstring 참조)
CASES = [
    ("poc_novessel", build_integrated_profile,
     dict(n_external=20, n_vessels=0), 880001,
     ("779eb22e8f271e83", "8f7987ff59701ef9", 45.435235, 16, 1.0, 0.0)),
    ("poc_gatein_only", build_integrated_profile,
     dict(n_external=16, n_vessels=0, gate_out_share=0.0), 880002,
     ("33a470e437499507", "1a7f58729b772e9c", 27.160938, 13, 1.0, 0.145)),
    ("calibrated_novessel", build_calibrated_profile,
     dict(n_external=20, n_vessels=0), 880003,
     ("37a0f9eb482e5599", "13acb11db1e7d2cf", 77.03312, 22, 1.0, 0.0203)),
]


@pytest.mark.parametrize("label,profile_fn,params_kw,seed,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_truck_track_frozen(label, profile_fn, params_kw, seed, expected):
    got = _snapshot(profile_fn(), TerminalGenParams(**params_kw), seed)
    assert got == expected, (
        f"{label}: 트럭 트랙 스냅샷 불일치 — 본선 수정이 본선 없는 시나리오를 오염. "
        f"기대 {expected} / 실측 {got}")
