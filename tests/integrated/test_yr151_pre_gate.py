"""YR-151 0A 계약 고정 — PRE_GATE 원자성·창·상한·stale·누출 0 (성능 판정 아님)."""
from __future__ import annotations

import pytest

from yard_rl.integrated.load_cells import make_a_cell, make_b_cell
from yard_rl.integrated.multiblock import MultiBlockTerminal, TransferError
from yard_rl.integrated.pre_gate import (iter_pre_gate_candidates, probe_public_info_only,
                                         public_block_eta)
from yard_rl.experiments.yr149_load_cells import _sim_from

A_SEED, B_SEED = 907_000, 907_100


@pytest.fixture()
def mbt():
    return MultiBlockTerminal({"A": _sim_from(make_a_cell(A_SEED, 100)),
                               "B": _sim_from(make_b_cell(B_SEED))})


def _first_candidate(m, src="A"):
    c = iter_pre_gate_candidates(m, src)
    if not c:
        pytest.skip("이 시점 PRE_GATE 후보 없음")
    return c[0][0]


def test_candidates_are_pre_gate_and_in_window(mbt):
    for jid, eta in iter_pre_gate_candidates(mbt, "A"):
        rec = mbt.ledger.records[jid]
        assert rec.a_gate_in > mbt.now                      # 아직 미진입
        assert 0.0 < eta - mbt.now <= 1800.0                # 30분 창
        assert public_block_eta(mbt.blocks["A"].jobs[jid]) == eta


def test_no_future_realized_leak(mbt):
    """실현 미래값을 흔들어도 후보가 불변이어야 한다 (0A 핵심 계약)."""
    r = probe_public_info_only(mbt, "A")
    assert r["n_perturbed"] > 0, "흔들 미래 작업이 없어 검사가 공허하다"
    assert r["identical"], f"실현 미래 의존 = 누출: {r}"


def test_pre_gate_commit_is_atomic_and_moves_owner(mbt):
    jid = _first_candidate(mbt)
    n_a, n_b = len(mbt.blocks["A"].jobs), len(mbt.blocks["B"].jobs)
    route0 = mbt.route_cost_s
    assert mbt.try_pre_gate_transfer(jid, "B", travel_s=300.0, route_delta_s=0.0)
    rec = mbt.ledger.records[jid]
    assert rec.owner == "B" and rec.transfer_count == 1
    assert rec.origin_block == "A"                          # TOS 최초배정 이력 보존
    assert jid not in mbt.blocks["A"].jobs and jid in mbt.blocks["B"].jobs
    assert (len(mbt.blocks["A"].jobs), len(mbt.blocks["B"].jobs)) == (n_a - 1, n_b + 1)
    assert mbt.route_cost_s == route0                       # delta 0 이면 계상도 0
    mbt.check_invariants()                                  # 소유자 1·orphan 0


def test_a_gate_in_unchanged_and_arrival_from_gate(mbt):
    """PRE_GATE 는 A 를 바꾸지 않는다 — 도착만 게이트→새 블록으로 재계산."""
    jid = _first_candidate(mbt)
    a0 = mbt.ledger.records[jid].a_gate_in
    assert mbt.try_pre_gate_transfer(jid, "B", travel_s=300.0)
    assert mbt.ledger.records[jid].a_gate_in == a0
    assert mbt.blocks["B"].jobs[jid].actual_block_arrival == pytest.approx(a0 + 300.0)


def test_transfer_cap_blocks_second_move(mbt):
    jid = _first_candidate(mbt)
    assert mbt.try_pre_gate_transfer(jid, "B", travel_s=300.0)
    assert not mbt.try_pre_gate_transfer(jid, "A", travel_s=300.0)   # 작업당 1회
    assert jid not in [c[0] for c in iter_pre_gate_candidates(mbt, "B")]


def test_stale_version_rejected(mbt):
    jid = _first_candidate(mbt)
    txn = mbt.prepare_pre_gate_transfer(jid, "B", travel_s=300.0)
    mbt.ledger.records[jid].version += 1                    # 다른 결정이 먼저 확정된 상황
    with pytest.raises(TransferError):
        mbt.commit(txn)
    mbt.rollback(txn)
    assert mbt.ledger.records[jid].owner == "A"


def test_rollback_restores_and_double_commit_rejected(mbt):
    jid = _first_candidate(mbt)
    txn = mbt.prepare_pre_gate_transfer(jid, "B", travel_s=300.0)
    mbt.rollback(txn)
    assert mbt.ledger.records[jid].owner == "A" and jid in mbt.blocks["A"].jobs
    with pytest.raises(TransferError):
        mbt.commit(txn)                                     # 닫힌 txn 재commit 금지
    mbt.check_invariants()


def test_post_gate_job_rejected_by_pre_gate_path(mbt):
    """이미 게이트를 지난 작업은 PRE_GATE 창 밖 — 기존 경로와 정확히 대칭."""
    jid = _first_candidate(mbt)
    rec = mbt.ledger.records[jid]
    saved = rec.a_gate_in
    rec.a_gate_in = mbt.now - 1.0                           # 진입 사건 발생 상황 모사
    try:
        with pytest.raises(TransferError):
            mbt.prepare_pre_gate_transfer(jid, "B", travel_s=300.0)
    finally:
        rec.a_gate_in = saved
