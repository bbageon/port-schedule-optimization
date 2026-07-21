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
    # YR-080 단계2 재동결 (양하=신규 반입 STORE·물리 도착 해제) — 항목별 사유:
    # n_decisions 4→5: J-VES-D 해제가 600(시간)→924(STS 144+이송 180 물리 도착)로 늦어져
    #   결정 시점 1회 증가. n_events 115 불변(JOB_RELEASED −1, VESSEL_RELEASED +1).
    # empty_m 513.5→253.5: 양하가 bay 10 반출(빈 주행 왕복)에서 인계점 인근 적재로 바뀜.
    # lane_cong 527.7→454.3·interference 420.4→691.2·imbalance 0.0282→0.0316: 결정
    #   시점·크레인 동선 변화의 파생 (rate 항 적분 구간 이동).
    # truck_wait·long_wait·sts_wait·transfer_wait 완전 불변 — 트럭 트랙·본선 프로세스
    #   (STS/이송 rate)는 단계 2 미접촉 증거 (단계 3 에서 LOAD 게이팅 시 변경 예정).
    "n_events": 115, "n_decisions": 5, "hash": "d0e73d88245c8304",
    "completed_external": 3, "completed_vessel": 2,
    "empty_m": 253.5, "rehandles": 1,
    "episode_raw": {
        "truck_wait": 184.878, "long_wait": 0.0, "crane_travel": 0.0, "empty_travel": 253.5,
        "rehandle": 1.0, "sts_wait": 768.0, "transfer_wait": 10620.0, "vessel_delay": 0.0,
        "depart_delay": 0.0, "lane_cong": 454.319, "interference": 691.206,
        # YR-043: imbalance 재정의 (누적 완료건수 pstdev → 작업부하 I(t)∈[0,1] / T_shift).
        "resequence": 0.0, "imbalance": 0.03155},
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
