"""YR-141 — 구속적 PREPOSITION 계약 고정 (opt-in·결속·근접 소멸·탈출 분리)."""
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


def _repo_ids(sim, bound: bool) -> list[str]:
    prev = cand_mod.BOUND_REPO
    cand_mod.BOUND_REPO = bound
    try:
        gen = CandidateGenerator()
        ids = []
        for c in [x.crane_id for x in sim.profile.cranes]:
            for g in gen.generate(sim, c, LEVEL).items:
                if g.kind.name == "REPOSITION" and g.job_ref is not None:
                    ids.append((c, g.job_ref.job_id, g.job_ref.reposition_target_bay))
        return ids
    finally:
        cand_mod.BOUND_REPO = prev


def test_flag_off_no_prepo_ids(sim):
    ids = _repo_ids(sim, bound=False)
    assert all(j.startswith("REPO:") for _, j, _ in ids)     # 기존 경로 불변


def test_flag_on_bound_ids_and_proximity(sim):
    ids = _repo_ids(sim, bound=True)
    prepo = [(c, j, tb) for c, j, tb in ids if j.startswith("PREPO:")]
    if not prepo:
        pytest.skip("이 시점 ETA 결속 후보 없음")
    for c, j, tb in prepo:
        jid = j.split(":")[1]
        assert jid in sim.jobs                                # 결속 작업 실재
        assert sim.jobs[jid].status.name == "PLANNED"         # 미도착만 (만료 내재)
        pos = sim.fleet.get(c).state.position_bay
        assert abs(tb - pos) > 1.0                            # 근접 소멸 규칙


def test_flag_on_no_unbound_normal_repo(sim):
    ids = _repo_ids(sim, bound=True)
    # 정상 상태(교착 아님)에서 비구속 REPO: 는 발행되지 않아야 한다 (탈출 전용 분리)
    unbound = [j for _, j, _ in ids if j.startswith("REPO:")]
    assert unbound == []


def test_determinism(sim):
    assert _repo_ids(sim, bound=True) == _repo_ids(sim, bound=True)


def test_execution_history_blocks_reissue(sim):
    """YR-142: 실행 이력에 오른 결속 작업은 재발행이 규칙으로 차단된다."""
    ids = _repo_ids(sim, bound=True)
    prepo = [(c, j, tb) for c, j, tb in ids if j.startswith("PREPO:")]
    if not prepo:
        pytest.skip("결속 후보 없음")
    target_jid = prepo[0][1].split(":")[1]
    saved = getattr(sim, "_prepo_history", None)
    try:
        sim._prepo_history = {target_jid}
        after = _repo_ids(sim, bound=True)
        assert all(j.split(":")[1] != target_jid
                   for _, j, _ in after if j.startswith("PREPO:"))
    finally:
        if saved is None:
            del sim._prepo_history
        else:
            sim._prepo_history = saved
