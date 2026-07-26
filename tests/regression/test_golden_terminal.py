"""통합 시뮬레이터 회귀 golden (YR-036).

고정 fixture + 참조 디스패처의 결과를 동결한다. 의존성·정렬·예약·비용배선 변경으로
이 값이 흔들리면 원인을 검토한 뒤에만 갱신한다 (test_golden.py 관습, tol 1e-3).
"""
from pathlib import Path

from yard_rl.contract import SCHEMA_VERSION, dumps
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (CraneAssignment, ReferenceDispatcher, TerminalSimulator,
                               build_integrated_profile, build_minimal_terminal_scenario,
                               record_episode)
from yard_rl.contract.schema import CandidateKind

_HERE = Path(__file__).parent
_RECORD_GOLDEN = _HERE / f"golden_terminal_record_{SCHEMA_VERSION}.json"

GOLDEN = {
    # YR-091/092 재동결 (2026-07-26, 외부감사 물리 정정) — 항목별 사유:
    # 초기 위치 분산(동일 bay 시작 제거)·idle 상시 장벽으로 이동 동선이 바뀜:
    #   empty_m 266.5→243.75 (크레인들이 구간 중앙 분산 시작 — 공차이동 감소),
    #   truck_wait 184.878→114.19 (가까운 크레인이 먼저 잡음), sts_wait 172.76→240.66·
    #   interference 943.55→636.18·lane_cong 파생 변화. n_events·n_decisions·완료수·
    #   rehandles 불변 = 사건 구조 보존(이동 기하만 변화). fixture 는 전 pile FT40 이라
    #   YR-092 영향 없음. 이전 재동결 사유(YR-080 단계3)는 git 이력 참조.
    "n_events": 36, "n_decisions": 7, "hash": "cf563bc19ab43fa7",
    "completed_external": 3, "completed_vessel": 4,
    "empty_m": 243.75, "rehandles": 1,
    "episode_raw": {
        "truck_wait": 114.190278, "long_wait": 0.0, "crane_travel": 0.0,
        "empty_travel": 243.75, "rehandle": 1.0, "sts_wait": 240.661111,
        "transfer_wait": 0.0, "vessel_delay": 0.0, "depart_delay": 0.0,
        "lane_cong": 567.204167, "interference": 636.177778,
        "resequence": 0.0, "imbalance": 0.039389},
}


def _drive():
    sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario())
    disp = ReferenceDispatcher()
    n_dec = 0
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        n_dec += 1
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=disp.select(sim, cid, cands))
                       if cands else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    return sim, n_dec


def test_golden_terminal():
    sim, n_dec = _drive()
    assert len(sim.event_log) == GOLDEN["n_events"]
    assert n_dec == GOLDEN["n_decisions"]
    assert sim.event_stream_hash() == GOLDEN["hash"]
    assert sim.kpis.completed_external == GOLDEN["completed_external"]
    assert sim.kpis.completed_vessel == GOLDEN["completed_vessel"]
    assert abs(sim.kpis.empty_gantry_m - GOLDEN["empty_m"]) < 1e-3
    assert sim.kpis.rehandle_count == GOLDEN["rehandles"]
    er = sim.cost.episode_raw()
    for k, v in GOLDEN["episode_raw"].items():
        assert abs(er[k] - v) < 1e-3, f"{k}: {er[k]} != {v}"


def test_record_serialization_frozen():
    """대표 TransitionRecord 1건 bit 동결 — 어댑터·직렬화 회귀 감지."""
    sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario())
    recs = record_episode(sim, ReferenceDispatcher(),
                          info_level=InformationLevel.PRE_ADVICE, episode_id="EP-GOLDEN")
    want = dumps(recs[0])
    got = _RECORD_GOLDEN.read_text(encoding="utf-8").rstrip("\n")
    assert want == got, "통합 record 직렬화가 golden 과 불일치 — 변경이면 검토 후 재생성"
