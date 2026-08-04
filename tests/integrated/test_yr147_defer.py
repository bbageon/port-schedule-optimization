"""YR-147 2단계 — 유한 DEFER 계약 고정 (A 불변·B 유한화·C trigger 구분·엔진 재개방)."""
from __future__ import annotations

import pytest

from yard_rl.experiments.yr088_joint_rl import LEVEL
from yard_rl.experiments.yr090_dense_vessel import BASE
from yard_rl.experiments.yr136_softplus_contract import _sim_contract
from yard_rl.integrated import candidates as cand_mod
from yard_rl.integrated.baselines import _apply, _untriggered_defer, _wait_of
from yard_rl.integrated.candidates import CandidateGenerator


@pytest.fixture()
def sim():
    return _sim_contract("mid-tight", BASE["mid-tight"])


def _wait_cand(sim, cid):
    gen = CandidateGenerator()
    return [g for g in gen.generate(sim, cid, LEVEL).items
            if g.kind.name == "WAIT"][-1]


def test_mode_a_default_invariant(sim):
    assert cand_mod.WAIT_MODE == "WAIT"          # 기본값 = 현행 계약
    cid = sim.profile.cranes[0].crane_id
    g = _wait_cand(sim, cid)
    assert g.defer_until is None and g.defer_trigger is None


def test_mode_b_finite_defer(sim):
    cid = sim.profile.cranes[0].crane_id
    prev = cand_mod.WAIT_MODE
    cand_mod.WAIT_MODE = "DEFER_ALL"
    try:
        g = _wait_cand(sim, cid)
        now = sim.now
        assert g.defer_until is not None
        assert now < g.defer_until <= now + cand_mod.DEFER_T_MAX + 1e-6
        t, k, _jid = CandidateGenerator()._defer_trigger_time(sim, now, LEVEL)
        if t is not None:
            assert g.defer_until == pytest.approx(min(t, now + cand_mod.DEFER_T_MAX))
            assert g.defer_trigger == k
    finally:
        cand_mod.WAIT_MODE = prev


def test_mode_c_untriggered_marking(sim):
    cid = sim.profile.cranes[0].crane_id
    prev = cand_mod.WAIT_MODE
    cand_mod.WAIT_MODE = "DEFER_TRIGGER"
    try:
        g = _wait_cand(sim, cid)
        assert g.defer_until is not None         # C 도 만료는 항상 있음 (fallback 유한)
        if g.defer_trigger is None:
            assert _untriggered_defer(g)
        else:
            assert not _untriggered_defer(g)
    finally:
        cand_mod.WAIT_MODE = prev
    assert not _untriggered_defer(_wait_cand(sim, cid))   # A 모드 WAIT 는 제외 대상 아님


def test_engine_defer_wake_reopens(sim):
    gen = CandidateGenerator()
    dp = sim.run_until_decision()
    assert dp is not None
    t0 = sim.now
    _apply(sim, {c: _wait_of(gen.generate(sim, c, LEVEL)) for c in dp.crane_ids})
    sim.schedule_defer_wake(t0 + 0.5)
    dp2 = sim.run_until_decision()
    assert any(e[1] == "DEFER_WAKE" and abs(e[0] - (t0 + 0.5)) < 1e-6
               for e in sim.event_log)
    if dp2 is not None:
        assert dp2.time > t0                      # 결정시각 엄격 증가 (YR-050)


def test_schedule_ignores_past_and_beyond_end(sim):
    sim.run_until_decision()
    n0 = len(sim._defer_wakes)
    sim.schedule_defer_wake(sim.now - 10.0)
    sim.schedule_defer_wake(sim.end + 999.0)
    assert len(sim._defer_wakes) == n0
