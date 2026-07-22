"""YR-080 단계 3 — 인과 연결 계약 (핵심 판정: "야드를 늦추면 배가 늦는가").

사슬: 적하 = YC 반출 완료 → 이송 → 안벽버퍼 → STS 처리 가능 → 선박 완료.
      양하 = STS → 이송 → 야드 도착 → job 해제(VESSEL_RELEASED) → YC 적재.
이게 안 붙어 있으면 크레인 행동이 본선 비용을 못 바꾼다 (YR-080d: 가중치 무효의 원인).
"""
from __future__ import annotations

from yard_rl.contract.schema import CandidateKind
from yard_rl.domain.enums import JobStatus
from yard_rl.integrated import (CraneAssignment, TerminalSimulator,
                                build_integrated_profile,
                                build_minimal_terminal_scenario)


def _drive(sim, allow_vessel):
    """SERVE 후보 중 (필터 통과) 첫 후보 선택 — 결정론 (candidates_for 순서)."""
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        for cid in dp.crane_ids:
            pick = None
            for ref in sim.candidates_for(cid):
                if ref.is_vessel and not allow_vessel(sim):
                    continue
                pick = ref
                break
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=pick)
                       if pick else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    return sim


def _fresh():
    return TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario())


def test_load_starvation_blocks_vessel():
    """본선 작업을 전혀 서비스하지 않으면: 적하 선박은 버퍼가 영원히 비어 미완."""
    sim = _drive(_fresh(), allow_vessel=lambda s: False)
    vl = sim.vessels["V-LOAD"]
    assert not vl.done, "야드가 반출을 안 했는데 적하 선박이 완료 — 인과 미연결"
    assert vl.buffer_level == 0
    assert vl.truth.actual_completion_s is None
    # 양하 선박(배→야드)은 야드 서비스와 무관하게 하역 자체는 진행 (의도된 비대칭)
    assert sim.vessels["V-DISCH"].done
    # 미서비스 본선 야드 job 은 미완으로 남는다 (엄격 계상 — 단계0 결정 5)
    assert all(sim.jobs[j].status != JobStatus.DONE
               for j in ("J-VES-L0", "J-VES-L1", "J-VES-D0", "J-VES-D1"))


def test_yard_delay_delays_vessel_completion():
    """핵심 단조성: 야드가 본선 반출을 늦출수록 적하 선박 완료가 늦어진다.

    임계 2500 = fixture 의 EQUIPMENT_UP(2600) 직전 — yielded 크레인을 깨우는
    이벤트가 있어야 지연 후 서비스가 재개된다 (임계를 마지막 wake 이벤트 뒤로
    두면 영영 미서비스 — 하네스 아티팩트, 인과 검정과 무관).
    """
    fast = _drive(_fresh(), allow_vessel=lambda s: True)
    slow = _drive(_fresh(), allow_vessel=lambda s: s.now >= 2500.0)
    t_fast = fast.vessels["V-LOAD"].truth.actual_completion_s
    t_slow = slow.vessels["V-LOAD"].truth.actual_completion_s
    assert t_fast is not None and t_slow is not None, "적하 선박 미완 — fixture 정합 확인"
    assert t_slow > t_fast, (
        f"야드 지연이 선박 완료에 전달되지 않음: fast={t_fast} slow={t_slow}")


def test_conservation_and_event_order():
    """보존·순서: 양하 job 해제 ≤ 야드 도착(prefix 불변식) · 적하 이송 = 반출 완료 후."""
    sim = _drive(_fresh(), allow_vessel=lambda s: True)
    arr_d = rel = 0
    for _t, kind, payload in sim.event_log:
        if kind == "TRANSFER_ARRIVE" and payload == "V-DISCH":
            arr_d += 1
        elif kind == "VESSEL_RELEASED":
            rel += 1
            assert rel <= arr_d, "양하 job 해제가 박스 도착보다 많음 — 유령 해제"
    assert rel == 2, f"양하 job 2건 전부 해제돼야 함 (실제 {rel})"
    # 적하: 안벽 도착(TRANSFER_ARRIVE V-LOAD)은 야드 반출 완료 이후에만
    l_ends = sorted(sim.jobs[j].service_end for j in ("J-VES-L0", "J-VES-L1"))
    l_arrs = sorted(t for t, k, p in sim.event_log
                    if k == "TRANSFER_ARRIVE" and p == "V-LOAD")
    assert len(l_arrs) == 2, f"적하 이송 도착 2건이어야 함 (실제 {len(l_arrs)})"
    for end, arr in zip(l_ends, l_arrs):
        assert arr >= end, f"박스가 반출 완료({end}) 전에 안벽 도착({arr}) — 유령 이송"


def test_determinism_two_runs_identical():
    sim1 = _drive(_fresh(), allow_vessel=lambda s: True)
    sim2 = _drive(_fresh(), allow_vessel=lambda s: True)
    assert sim1.event_stream_hash() == sim2.event_stream_hash()
