"""비용 인과 ledger — 항등식·cause 화이트리스트·guardrail 분리·off parity (YR-038)."""
import pytest

from yard_rl.contract.schema import COST_TERMS
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (ReferenceDispatcher, TerminalSimulator,
                               build_integrated_profile, build_minimal_terminal_scenario,
                               record_episode)
from yard_rl.integrated.ledger import (RATE_CAUSE, TERM_CAUSES, CostCause,
                                      assert_ledger_identity, build_ledger_report)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE


def _run(enable_ledger):
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(),
                            info_level=LEVEL, enable_cost_ledger=enable_ledger)
    record_episode(sim, ReferenceDispatcher(), info_level=LEVEL, episode_id="E")
    return sim


def test_ledger_episode_identity():
    """Σledger[t] == episode_raw[t] ∀13 (중복계상 0 — 단일 write path 구성상)."""
    sim = _run(True)
    assert_ledger_identity(sim.cost)
    lt = sim.cost.ledger.term_totals()
    ep = sim.cost.episode_raw()
    for t in COST_TERMS:
        assert abs(lt[t] - ep[t]) < 1e-6


def test_ledger_off_parity():
    """ledger off(기본) vs on: episode_raw·event_stream_hash 동일 (golden 불변)."""
    off, on = _run(False), _run(True)
    assert off.cost.episode_raw() == on.cost.episode_raw()
    assert off.event_stream_hash() == on.event_stream_hash()


def test_ledger_cause_whitelist():
    """모든 entry 의 cause 가 항별 화이트리스트 안 (vessel_delay 는 2-cause)."""
    sim = _run(True)
    for e in sim.cost.ledger.all_entries():
        assert e.cause in TERM_CAUSES[e.term]
    assert TERM_CAUSES["vessel_delay"] == frozenset({CostCause.VESSEL_FINISH, CostCause.CLEAROUT})


def test_ledger_guardrail_excluded():
    """ledger 는 13 cost 항만 — 안전/mandatory(mask)는 원천 부재."""
    sim = _run(True)
    for e in sim.cost.ledger.all_entries():
        assert e.term in COST_TERMS


def test_ledger_interval_identity():
    """구간별 ledger 합 == 그 구간 cut() raw (파티션 정합)."""
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(),
                            info_level=LEVEL, enable_cost_ledger=True)
    led = sim.cost.ledger
    cut_totals = []
    dp = sim.run_until_decision()
    sim.cost.cut()   # pre-first 폐기 (record_episode 관습)
    disp = ReferenceDispatcher()
    from yard_rl.integrated import CraneAssignment
    from yard_rl.contract import CandidateKind
    while dp is not None:
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=disp.select(sim, cid, cands))
                       if cands else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
        dp = sim.run_until_decision()
        k_before = len(led.sealed)
        raw = sim.cost.cut()
        seg_tot = led.interval_term_totals(k_before)
        for t in COST_TERMS:
            assert abs(seg_tot[t] - raw[t]) < 1e-6


def test_rate_cause_map_complete():
    from yard_rl.integrated.cost import RATE_TERMS
    assert set(RATE_CAUSE) == set(RATE_TERMS)


def test_ledger_report_deterministic():
    a, b = _run(True), _run(True)
    assert a.cost.ledger.digest() == b.cost.ledger.digest()
    rep = build_ledger_report(a.cost)
    assert rep["report_id"] == "cost-ledger-report-v1"
    assert all(abs(v["residual"]) < 1e-9 for v in rep["identity_per_term"].values())
