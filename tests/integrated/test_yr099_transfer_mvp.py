"""YR-099 MVP G0 — 소유권 유일·보존·결정론·정보안전·fail-closed (엔진 무변경 계약)."""
import copy

import pytest

from yard_rl.domain.enums import JobFlow
from yard_rl.experiments.yr099_transfer_mvp import (
    CELL_A, CELL_B, REVIEW_MARGIN_S, _scen, eligible_inbound, move_job, to_predicted)


@pytest.fixture(scope="module")
def pair():
    return _scen(831_900, CELL_A), _scen(832_900, CELL_B)


def test_g0_conservation_and_owner_unique(pair):
    # 주의: 블록별 시나리오는 id 네임스페이스가 독립(둘 다 J-IN-005 존재 가능) —
    # 터미널 소유권 유일성은 (block, job_id) 로 판정한다 (spec canonical JobRegistry 의 MVP 형).
    sa, sb = pair
    jid = eligible_inbound(sa)[0]
    n0 = len(sa.jobs) + len(sb.jobs)
    sa2, sb2 = move_job(sa, sb, jid)
    assert len(sa2.jobs) + len(sb2.jobs) == n0                     # 보존
    for blk in (sa2, sb2):                                         # 블록 내 id 유일
        ids = [j.job_id for j in blk.jobs]
        assert len(ids) == len(set(ids))
    moved = f"{jid}@X"                                             # 이동 작업 = 정확히 한 블록
    assert not any(j.job_id in (jid, moved) for j in sa2.jobs)     # source 에 없음
    assert sum(1 for j in sb2.jobs if j.job_id == moved) == 1      # receiver 에 정확히 1
    # 원본 불변 (원자성: 실패 시 KEEP 이 가능해야 하므로 입력은 절대 오염 금지)
    assert any(j.job_id == jid for j in sa.jobs)


def test_g0_route_shift_and_ledger_continuity(pair):
    sa, sb = pair
    jid = eligible_inbound(sa)[0]
    j0 = next(j for j in sa.jobs if j.job_id == jid)
    _, sb2 = move_job(sa, sb, jid, route_s=180.0)
    j1 = next(j for j in sb2.jobs if j.job_id == f"{jid}@X")
    assert j1.actual_block_arrival == pytest.approx(j0.actual_block_arrival + 180.0)
    assert j1.estimated_block_arrival == pytest.approx(j0.estimated_block_arrival + 180.0)
    assert j1.actual_gate_in == pytest.approx(j0.actual_gate_in)   # 게이트 진입시각 유지(장부 연속)


def test_g0_move_missing_job_raises_keep(pair):
    sa, sb = pair
    with pytest.raises(ValueError):
        move_job(sa, sb, "J-NOPE-999")                             # 실패 → 호출부 KEEP


def test_g0_determinism(pair):
    sa, _ = pair
    assert eligible_inbound(sa) == eligible_inbound(copy.deepcopy(sa))


def test_info_safety_predicted_has_no_realized_draw(pair):
    sa, _ = pair
    p = to_predicted(sa)
    for j in p.jobs:
        est = getattr(j, "estimated_block_arrival", None)
        if est is not None:
            assert j.actual_block_arrival == pytest.approx(est)    # 실현 draw 소거
            assert j.actual_gate_in == pytest.approx(j.appointment_gate_time)


def test_fail_closed_eligibility(pair):
    sa, _ = pair
    elig = eligible_inbound(sa)
    for jid in elig:
        j = next(x for x in sa.jobs if x.job_id == jid)
        assert j.flow == JobFlow.GATE_IN                           # 반입만
        assert j.estimated_block_arrival >= REVIEW_MARGIN_S        # 임박 fail-closed
    # 반출(GATE_OUT)은 위치 고정 — 절대 eligible 아님
    outs = {j.job_id for j in sa.jobs if j.flow == JobFlow.GATE_OUT}
    assert not (set(elig) & outs)
