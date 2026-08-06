"""YR-150 0단계 — 배치·목적지별 route·N블록 matching 계약 테스트."""
import pytest

from yard_rl.integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S,
                                             GATE_BLOCK_MIN_S)
from yard_rl.integrated.transfer_quote import TransferQuoteResolver
from yard_rl.integrated.yard_layout import YardLayout, terminal_layout


# ------------------------------------------------------------------ 배치·앵커
def test_full_terminal_mean_equals_anchor():
    """전 블록 평균 게이트 주행이 기존 계약 중심값과 같아야 한다 (앵커 보존)."""
    assert terminal_layout().mean_gate_time_s() == pytest.approx(GATE_BLOCK_MEAN_S)


def test_all_blocks_within_supported_range():
    lo, hi = terminal_layout().gate_time_range_s()
    assert GATE_BLOCK_MIN_S <= lo and hi <= GATE_BLOCK_MAX_S


def test_route_matrix_is_metric_and_nonzero():
    layout = terminal_layout()
    ids = layout.ids
    assert all(layout.block_to_block_s(b, b) == 0.0 for b in ids)
    off = [layout.block_to_block_s(a, b) for a in ids for b in ids if a != b]
    assert min(off) > 0.0                       # ★목적지별 비용이 실제로 존재
    assert all(layout.block_to_block_s(a, b) == layout.block_to_block_s(b, a)
               for a in ids for b in ids)


def test_pre_gate_delta_is_signed_and_antisymmetric():
    """가까운 블록으로 보내면 주행이 **줄어야** 한다 — 0A 의 'route 차이 0' 해소."""
    layout = terminal_layout()
    a, b = layout.ids[0], layout.ids[-1]
    assert layout.pre_gate_route_delta_s(a, b) > 0
    assert layout.pre_gate_route_delta_s(b, a) < 0
    assert layout.pre_gate_route_delta_s(a, b) == pytest.approx(
        -layout.pre_gate_route_delta_s(b, a))


def test_subset_keeps_absolute_geometry():
    """부분집합은 전체 좌표를 그대로 물려받아야 한다(재중심화 금지)."""
    full = terminal_layout()
    sub = full.subset(("Y01", "Y11", "Y21"))
    assert all(sub.gate_to_block_s(b) == full.gate_to_block_s(b) for b in sub.ids)


def test_layout_outside_support_is_rejected():
    """앵커 지원범위를 벗어나면 조용히 자르지 않고 예외를 던진다."""
    with pytest.raises(ValueError):
        YardLayout(("A", "B"), (100.0, 5_000.0), 5.0)


# ------------------------------------------------------------------ N블록 matching
class _Rec:
    def __init__(self, owner, a, ver=0):
        self.owner, self.a_gate_in, self.version = owner, a, ver
        self.reassignable, self.transfer_count = True, 0


class _Ledger:
    def __init__(self, records):
        self.records = records


class _FakeMBT:
    """resolver 의 matching 계약만 시험하기 위한 최소 대역 — 엔진 물리는 쓰지 않는다."""

    def __init__(self, records, *, accept=True, blocks=None):
        ids = blocks if blocks is not None else sorted({r.owner for r in records.values()})
        self.blocks = {b: object() for b in ids}
        self.ledger = _Ledger(records)
        self.accept = accept
        self.calls = []

    def try_transfer(self, job_id, dst, *, route_s, travel_s):
        self.calls.append((job_id, dst, route_s, travel_s))
        if not self.accept:
            return False
        rec = self.ledger.records[job_id]
        rec.owner, rec.version, rec.transfer_count = dst, rec.version + 1, rec.transfer_count + 1
        return True


def _resolver(cap, *, layout=None, relief=None, burden=None):
    """예측 함수를 대역으로 바꿔 matching 순수 로직만 본다."""
    import yard_rl.integrated.transfer_quote as tq

    r = TransferQuoteResolver(
        object(), travel_fn=lambda s, d, j: 300.0,
        route_fn=(layout.post_gate_route_s if layout else None),
        gain_margin=0.0, terminal_epoch_cap=cap)
    r._orig = (tq.predict_keep_cost, tq.predict_move_cost)
    tq.predict_keep_cost = lambda sim, jid, kf: (relief or {}).get(jid, 10.0)
    tq.predict_move_cost = lambda s, d, jid, kf, **kw: (burden or {}).get(jid, 1.0)
    return r


def _restore(r):
    import yard_rl.integrated.transfer_quote as tq
    tq.predict_keep_cost, tq.predict_move_cost = r._orig


def test_nblock_allows_simultaneous_transfers_from_distinct_sources():
    """N블록 계약에서는 한 epoch 에 여러 소스가 동시에 확정될 수 있다."""
    records = {"jA": _Rec("A", 100.0), "jB": _Rec("B", 100.0), "jC": _Rec("C", 100.0)}
    mbt = _FakeMBT(records)
    r = _resolver(None)
    try:
        r.review(mbt, 100.0)
    finally:
        _restore(r)
    assert r.n_transferred == 3
    assert {c[0] for c in mbt.calls} == {"jA", "jB", "jC"}


def test_two_block_contract_still_caps_at_one_per_epoch():
    """2블록 기능계약(cap=1)은 구판 그대로 epoch당 1건만 시도한다."""
    records = {"jA": _Rec("A", 100.0), "jB": _Rec("B", 100.0), "jC": _Rec("C", 100.0)}
    mbt = _FakeMBT(records)
    r = _resolver(1)
    try:
        r.review(mbt, 100.0)
    finally:
        _restore(r)
    assert len(mbt.calls) == 1 and r.n_transferred == 1


def test_cap_one_does_not_promote_next_when_first_fails():
    """구판 보존: 첫 제안이 실패해도 다음 제안을 대신 올리지 않는다."""
    records = {"jA": _Rec("A", 100.0), "jB": _Rec("B", 100.0)}
    mbt = _FakeMBT(records, accept=False)
    r = _resolver(1)
    try:
        r.review(mbt, 100.0)
    finally:
        _restore(r)
    assert len(mbt.calls) == 1 and r.n_transferred == 0


def test_one_offer_per_source_per_epoch():
    """같은 소스의 작업이 여럿이어도 한 epoch 에 그 소스는 1건만 확정한다."""
    records = {"jA1": _Rec("A", 100.0), "jA2": _Rec("A", 100.0), "jB": _Rec("B", 100.0)}
    mbt = _FakeMBT(records)
    r = _resolver(None)
    try:
        r.review(mbt, 100.0)
    finally:
        _restore(r)
    srcs = [c[0][:2] for c in mbt.calls]
    assert len(mbt.calls) == 2 and len(srcs) == len(set(srcs))


def test_destination_route_is_recorded_and_minimised():
    """목적지별 route 가 원장에 남고, 부담+주행이 최소인 목적지를 고른다."""
    layout = terminal_layout().subset(("Y01", "Y11", "Y21"))
    # 소스는 Y01 — 두 수신 후보의 주행이 서로 다르다(Y11 110s vs Y21 220s).
    # (Y11 을 소스로 두면 양옆이 등거리라 route 가 선택을 가르지 못해 시험이 무의미해진다.)
    records = {"j1": _Rec("Y01", 100.0)}
    mbt = _FakeMBT(records, blocks=("Y01", "Y11", "Y21"))
    r = _resolver(None, layout=layout)
    try:
        r.review(mbt, 100.0)
    finally:
        _restore(r)
    quoted = [x for x in r.ledger if x.get("bids")]
    assert quoted, "견적 원장이 비었다"
    bids = quoted[0]["bids"]
    assert {b["dst"] for b in bids} == {"Y11", "Y21"}
    assert all(b["route_s"] == layout.post_gate_route_s("Y01", b["dst"]) for b in bids)
    assert len({b["route_s"] for b in bids}) == 2      # 주행이 실제로 갈린다
    # 부담이 같으므로 **더 가까운 쪽**이 뽑혀야 한다 — route 가 선택에 실제로 쓰인 증거
    assert quoted[0]["dst"] == "Y11"
    assert quoted[0]["route_s"] == layout.post_gate_route_s("Y01", "Y11")
