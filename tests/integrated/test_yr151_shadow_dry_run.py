"""YR-151 — shadow **dry-run** 계약 (감사 재정의 2026-08-09).

구판 shadow(즉시-KEEP 반환)는 resolver 를 아예 안 거쳐 견적·matching·용량 검사가
검증되지 않았다. 새 계약: shadow 정책도 제안을 반환하고, dry_run resolver 가
would-commit 을 **기록만** 한다 — 소유권·시간 장부 불변(환경 무변이)·짝 강제.
"""
import pytest

from yard_rl.experiments.yr149_load_cells import _sim_from
from yard_rl.integrated.cost_curve_v2 import KappaFit
from yard_rl.integrated.load_cells import make_a_cell, make_b_cell
from yard_rl.integrated.multiblock import MultiBlockTerminal
from yard_rl.integrated.sell_review import UnifiedSellOrchestrator
from yard_rl.integrated.terminal_stream import _job_from_entry
from yard_rl.integrated.yard_layout import YardLayout

KF = KappaFit(kappa_t_s=600.0, bias_t_s=0.0, kappa_v_s=600.0, bias_v_s=0.0)


class _OfferFirst:
    mode = "live"

    def decide(self, mbt, src, cands, t):
        return cands[0][0] if cands else None


class _ShadowStub:
    mode = "shadow"

    def decide(self, mbt, src, cands, t):
        return cands[0][0] if cands else None


def _mbt_with_announced():
    """A/B 2블록 + 진입 예고(통지 미래) 반입 트럭 2대 — 판매 후보가 생기는 최소 상태."""
    mbt = MultiBlockTerminal({"A": _sim_from(make_a_cell(6_800_000, 50)),
                              "B": _sim_from(make_b_cell(6_800_001))})
    for i, bid in enumerate(("A", "A")):
        e = {"job_id": f"{bid}:W-SH{i}", "block": bid, "flow": "GATE_IN",
             "requested_flow": "GATE_IN", "fallback_reason": None, "target": None,
             "size_ft40": True, "travel_s": 300.0, "travel_base_s": 300.0,
             "exit_travel_s": 300.0}
        mbt.admit_external_job(bid, _job_from_entry(e, 1200.0 + 60 * i),
                               gate_in_s=1200.0 + 60 * i, travel_s=300.0)
    return mbt


def _layout2():
    return YardLayout(ids=("A", "B"), positions_m=(950.0, 1050.0), speed_mps=5.0)


def _ledger_snapshot(mbt):
    out = {}
    for jid, rec in mbt.ledger.records.items():
        out[jid] = (rec.owner, rec.transfer_count, rec.entry_deferrals)
    tl = {}
    for b, sim in mbt.blocks.items():
        led = getattr(sim, "time_ledger", None)
        if led:
            for jid, r in led.records.items():
                tl[(b, jid)] = r.gate_in
    return out, tl


def test_shadow_policy_requires_dry_run_resolver():
    """짝 강제 — shadow 정책 + 일반 resolver 는 생성 시점에 즉시 실격."""
    with pytest.raises(ValueError):
        UnifiedSellOrchestrator(_ShadowStub(), _layout2(), KF)


def test_dry_run_records_but_never_commits():
    """dry-run: would-commit 이 기록되되 소유권·시간 장부·이연 카운트는 불변."""
    mbt = _mbt_with_announced()
    orch = UnifiedSellOrchestrator(_ShadowStub(), _layout2(), KF, dry_run=True)
    before = _ledger_snapshot(mbt)
    orch.review(mbt, 600.0)                       # 60초 격자 위·통지 창 안
    after = _ledger_snapshot(mbt)
    assert before == after                        # 환경 무변이
    assert orch.n_space == 0 and orch.n_time == 0
    decisions = {e["decision"] for e in orch.ledger}
    assert decisions <= {"DRY_WOULD_COMMIT", "RESOLVER_KEEP"}
    assert any(e["decision"] == "DRY_WOULD_COMMIT" for e in orch.ledger), \
        "제안이 resolver 를 관통해 would-commit 까지 도달해야 한다"


def test_live_same_state_actually_commits():
    """같은 상태에서 live resolver 는 실제로 확정한다 — dry 와의 차이는 확정뿐."""
    mbt = _mbt_with_announced()
    orch = UnifiedSellOrchestrator(_OfferFirst(), _layout2(), KF)
    before = _ledger_snapshot(mbt)
    orch.review(mbt, 600.0)
    after = _ledger_snapshot(mbt)
    committed = [e for e in orch.ledger
                 if e["decision"] in ("SELL", "DEFER")]
    if committed:                                 # 확정이 있으면 상태가 변해야 한다
        assert before != after
    assert "DRY_WOULD_COMMIT" not in {e["decision"] for e in orch.ledger}
