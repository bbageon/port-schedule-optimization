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
    # YR-080 단계3 재동결 (인과 연결 + fixture 1:1 정합) — 항목별 사유:
    # n_events 115→36: fixture 선박 moves 20/15→2/2 (STS·이송 이벤트 대량 감소 —
    #   1박스=1야드작업 전량 정합, 단계0 결정 2). n_decisions 5→7: 본선 야드 job 2→4건.
    # completed_vessel 2→4: 정합된 job 수. V-LOAD 완료 1660.76 — **야드 반출→이송→
    #   안벽 사슬을 실제로 통과**한 완료 (이전엔 야드와 무관하게 진행).
    # sts_wait 768→172.76: 유령 pre-fill 제거 — 적하 STS 가 야드 완료를 기다린
    #   시간만 계상 (진짜 인과 신호). transfer_wait 10620→0: 이송이 실요청만 처리.
    # truck_wait 184.878 **완전 불변** — 트럭 트랙 무결 증거 (스냅샷 계약과 정합).
    # empty_m 253.5→266.5·lane_cong·interference·imbalance: job 수·동선 변화 파생.
    "n_events": 36, "n_decisions": 7, "hash": "63556f0e932dcdfd",
    "completed_external": 3, "completed_vessel": 4,
    "empty_m": 266.5, "rehandles": 1,
    "episode_raw": {
        "truck_wait": 184.878, "long_wait": 0.0, "crane_travel": 0.0, "empty_travel": 266.5,
        "rehandle": 1.0, "sts_wait": 172.761, "transfer_wait": 0.0, "vessel_delay": 0.0,
        "depart_delay": 0.0, "lane_cong": 580.981, "interference": 943.55,
        # YR-043: imbalance 재정의 (누적 완료건수 pstdev → 작업부하 I(t)∈[0,1] / T_shift).
        "resequence": 0.0, "imbalance": 0.040346},
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
