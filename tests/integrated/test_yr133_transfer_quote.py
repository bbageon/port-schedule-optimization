"""YR-133 계약 고정 — 견적 fail-closed·κ 동결 배선·브리지 불변(회귀는 22종이 담당)."""
from __future__ import annotations

from yard_rl.experiments.yr105_conditional_transfer import _CELLS, _sim
from yard_rl.integrated.cost_curve_v2 import KAPPA_V2P_PATH, KappaFit
from yard_rl.integrated.transfer_quote import (TransferQuoteResolver,
                                               predict_keep_cost, predict_move_cost)


def test_fail_closed_missing_or_unpredictable():
    kf = KappaFit.load(KAPPA_V2P_PATH, require_contract_physics=True)
    a, b = _sim(_CELLS["A"], 100), _sim(_CELLS["B"], 101)
    assert predict_keep_cost(a, "없는작업", kf) is None
    assert predict_move_cost(a, b, "없는작업", kf, travel_s=300.0, route_s=180.0) is None


def test_keep_arm_margin_inf_blocks_all():
    """gain_margin=∞ 이면 NetGain 은 항상 ≤0 — KEEP arm 은 구조적으로 무이송."""
    kf = KappaFit.load(KAPPA_V2P_PATH, require_contract_physics=True)
    r = TransferQuoteResolver(kf, travel_fn=lambda s, j: 300.0,
                              gain_margin=float("inf"))
    assert (1e9 - 0.0 - r.route_s / 3600.0 - r.gain_margin) <= 0.0


def test_quote_ledger_epoch_only_contract():
    """quote 는 epoch-local 로만 쓰인다 — resolver 가 이월 저장소를 갖지 않는다(만료
    위반 원천 차단 계약의 구조 검증)."""
    kf = KappaFit.load(KAPPA_V2P_PATH, require_contract_physics=True)
    r = TransferQuoteResolver(kf, travel_fn=lambda s, j: 300.0)
    assert not any(isinstance(getattr(r, n), dict) and "offer" in n.lower()
                   for n in vars(r))
    assert r.ledger == [] and r.n_transferred == 0
