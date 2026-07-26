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


def test_ens_sample_deterministic_and_varied(pair):
    from yard_rl.experiments.yr099_transfer_mvp import _params, to_predicted_sample
    sa, _ = pair
    p = _params(CELL_A)
    s1 = to_predicted_sample(sa, p, "t:0")
    s2 = to_predicted_sample(sa, p, "t:0")
    s3 = to_predicted_sample(sa, p, "t:1")
    a1 = [j.actual_block_arrival for j in s1.jobs if getattr(j, "appointment_gate_time", None)]
    a2 = [j.actual_block_arrival for j in s2.jobs if getattr(j, "appointment_gate_time", None)]
    a3 = [j.actual_block_arrival for j in s3.jobs if getattr(j, "appointment_gate_time", None)]
    assert a1 == a2                                                # 같은 key = 결정론
    assert a1 != a3                                                # 다른 k = 표본 다양성


def test_ens_sample_info_safe(pair):
    # 실현 draw 를 오염시켜도 표본 출력 불변 = 실현 미참조 (누출 0)
    from yard_rl.experiments.yr099_transfer_mvp import _params, to_predicted_sample
    sa, _ = pair
    p = _params(CELL_A)
    ref = to_predicted_sample(sa, p, "t:9")
    tam = copy.deepcopy(sa)
    for j in tam.jobs:
        if getattr(j, "appointment_gate_time", None) is not None:
            j.actual_block_arrival = 99_999.0                      # 실현값 오염
            j.actual_gate_in = 88_888.0
    out = to_predicted_sample(tam, p, "t:9")
    r1 = [j.actual_block_arrival for j in ref.jobs if getattr(j, "appointment_gate_time", None)]
    r2 = [j.actual_block_arrival for j in out.jobs if getattr(j, "appointment_gate_time", None)]
    assert r1 == r2


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
