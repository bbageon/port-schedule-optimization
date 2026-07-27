"""YR-099 창중 review G0 — 이식 원자성·소유권·큐 중화·원장 이관·fail-closed."""
import pytest

from yard_rl.domain.enums import JobStatus
from yard_rl.experiments.yr099_midrun_review import (CELL_A, CELL_B, BASE_A, BASE_B,
                                                     move_inflight_job, run_pair, _sim)


@pytest.fixture()
def pair():
    return _sim(CELL_A, BASE_A + 900), _sim(CELL_B, BASE_B + 900)


def _first_inflight(sim):
    for jid, j in sorted(sim.jobs.items()):
        if (j.is_external_truck and j.status == JobStatus.PLANNED
                and getattr(j, "actual_gate_in", None) is not None
                and j.flow.name == "GATE_IN"):
            return jid
    raise RuntimeError("no candidate")


def test_g0_move_atomic_owner_queue_ledger(pair):
    src, dst = pair
    jid = _first_inflight(src)
    n0 = len(src.jobs) + len(dst.jobs)
    new_id = move_inflight_job(src, dst, jid, "t")
    assert len(src.jobs) + len(dst.jobs) == n0                       # 보존
    assert jid not in src.jobs and new_id in dst.jobs                # 소유권 유일
    assert not any(e.kind_name == "BLOCK_ARRIVAL" and e.payload == jid
                   for e in src.queue._heap)                         # 소스 큐 중화
    assert sum(1 for e in dst.queue._heap
               if e.kind_name == "BLOCK_ARRIVAL" and e.payload == new_id) == 1
    if src.time_ledger is not None:
        assert jid not in src.time_ledger.records                    # 원장 이관
        assert new_id in dst.time_ledger.records
    j = dst.jobs[new_id]
    assert j.actual_block_arrival > j.actual_gate_in + 180.0 + 179.0  # 신규 draw + route


def test_g0_move_deterministic_draw(pair):
    src, dst = pair
    jid = _first_inflight(src)
    src2, dst2 = _sim(CELL_A, BASE_A + 900), _sim(CELL_B, BASE_B + 900)
    a1 = move_inflight_job(src, dst, jid, "t")
    a2 = move_inflight_job(src2, dst2, jid, "t")
    assert dst.jobs[a1].actual_block_arrival == dst2.jobs[a2].actual_block_arrival


def test_g0_fail_closed_states(pair):
    src, dst = pair
    jid = _first_inflight(src)
    src.jobs[jid].status = JobStatus.WAITING                         # 이미 도착 = 창 밖
    with pytest.raises(ValueError):
        move_inflight_job(src, dst, jid, "t")
    src.jobs[jid].status = JobStatus.PLANNED
    dst.clock = 10 ** 9                                              # 타깃 시계 경과 (내부 필드)
    with pytest.raises(ValueError):
        move_inflight_job(src, dst, jid, "t")
    assert jid in src.jobs                                           # 실패 = 원상태 (KEEP)


def test_integration_paired_run_completes():
    a0 = run_pair(900 - BASE_A + BASE_A and 0, transfer=False)       # seed 0
    c = run_pair(0, transfer=True)
    assert a0["n_jobs"] == c["n_jobs"]                               # 보존 (터미널 합)
    assert a0["compl_min"] > 0.9 and c["compl_min"] > 0.9
    assert c["n_moved"] >= 0
