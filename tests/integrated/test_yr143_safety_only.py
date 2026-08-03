"""YR-143 계약 고정 — SAFETY_ONLY(C0) 모드: 능동 위치조정 미발행·안전기능만."""
from __future__ import annotations

import pytest

from yard_rl.experiments.yr088_joint_rl import LEVEL
from yard_rl.experiments.yr090_dense_vessel import BASE
from yard_rl.experiments.yr136_softplus_contract import _sim_contract
from yard_rl.integrated import candidates as cand_mod
from yard_rl.integrated.candidates import CandidateGenerator


@pytest.fixture(scope="module")
def sim():
    return _sim_contract("high-tight", BASE["high-tight"])


def _kinds(sim, safety_only, bound):
    prev = cand_mod.SAFETY_ONLY, cand_mod.BOUND_REPO
    cand_mod.SAFETY_ONLY, cand_mod.BOUND_REPO = safety_only, bound
    try:
        gen = CandidateGenerator()
        out = []
        for c in [x.crane_id for x in sim.profile.cranes]:
            for g in gen.generate(sim, c, LEVEL).items:
                out.append((g.kind.name, g.job_ref.job_id if g.job_ref else None))
        return out
    finally:
        cand_mod.SAFETY_ONLY, cand_mod.BOUND_REPO = prev


def test_safety_only_no_active_repositioning(sim):
    """C0: 위치조정만 제거되고 그 외 후보(선재조작·대기 등)는 완전 불변."""
    base = _kinds(sim, safety_only=False, bound=False)
    c0 = _kinds(sim, safety_only=True, bound=False)
    assert all(k != "REPOSITION" for k, _ in c0)
    assert [x for x in base if x[0] != "REPOSITION"] == c0   # 그 외 후보 불변
    assert any(k == "WAIT" for k, _ in c0)


def test_safety_only_overrides_bound(sim):
    ks = _kinds(sim, safety_only=True, bound=True)      # 오조합이어도 C0 우선
    assert all(k != "REPOSITION" for k, _ in ks)


def test_flag_off_unchanged(sim):
    base = _kinds(sim, safety_only=False, bound=False)
    assert any(k == "REPOSITION" for k, _ in base)      # 기존 경로 불변 (비교 대조)
