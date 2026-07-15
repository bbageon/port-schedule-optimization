"""동적 후보 생성기 — 4종·mandatory·padding·feasible 노출·누출 0 (YR-037)."""
import pytest

from yard_rl.contract import CandidateKind
from yard_rl.contract.validate import validate_candidates
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (CandidateGenerator, TerminalSimulator,
                               build_integrated_profile, build_minimal_terminal_scenario)
from yard_rl.integrated.adapter import _build_candidate_set
from yard_rl.integrated.candidates import GenCandidate
from yard_rl.integrated.reservation import Corridor, Reservation
from yard_rl.sim.constraints import ConstraintViolation

PROF = build_integrated_profile()
GEN = CandidateGenerator()


def _sim():
    return TerminalSimulator(PROF, build_minimal_terminal_scenario(),
                             info_level=InformationLevel.PRE_ADVICE)


def _first_decision(sim):
    dp = sim.run_until_decision()
    assert dp is not None
    return dp


def test_generated_candidate_set_valid():
    """생성기 → CandidateSet 이 validate_candidates 통과, id==index, WAIT 마지막 실후보."""
    sim = _sim()
    dp = _first_decision(sim)
    cid = dp.crane_ids[0]
    gen = GEN.generate(sim, cid, InformationLevel.PRE_ADVICE)
    cs = _build_candidate_set(sim, cid, gen, sim.now, InformationLevel.PRE_ADVICE, (), GEN.k_max)
    validate_candidates(cs)
    assert len(cs.items) == GEN.k_max
    assert [c.candidate_id for c in cs.items] == list(range(GEN.k_max))
    # 실후보의 마지막 kind 는 WAIT
    real = [c for c, p in zip(cs.items, cs.pad_mask) if p]
    assert real[-1].kind == CandidateKind.WAIT
    assert cs.feasible_mask[len(real) - 1] is True   # WAIT 항상 feasible


def test_padding_zeroed():
    sim = _sim()
    dp = _first_decision(sim)
    cid = dp.crane_ids[0]
    gen = GEN.generate(sim, cid, InformationLevel.PRE_ADVICE)
    cs = _build_candidate_set(sim, cid, gen, sim.now, InformationLevel.PRE_ADVICE, (), GEN.k_max)
    for c, pad, reason in zip(cs.items, cs.pad_mask, cs.mask_reason):
        if not pad:
            assert not any(c.features.known)
            assert reason == "PADDING"


def test_is_mandatory_and_preserved():
    """SLA 임박 트럭은 mandatory — 실행불가여도 set 잔존 (YR-029, pruning 금지)."""
    sim = _sim()
    _first_decision(sim)
    # 대상 트럭을 오래 대기(진실 도착 = now-2000s)로 만들어 mandatory 유발
    j = sim.jobs["J-OUT-A"]
    j.actual_block_arrival = sim.now - 2000.0
    from yard_rl.domain.enums import JobStatus
    j.status = JobStatus.WAITING
    cid = next(c for c in sim.fleet.ids() if sim.fleet.get(c).idle)
    serves = GEN._serve(sim, cid, sim.now)
    mand = [g for g in serves if g.job_ref.job_id == "J-OUT-A"]
    assert mand and mand[0].mandatory is True
    # 실행불가(가짜 예약으로 레인 점유)여도 잔존
    sim.reservations.reserve(Reservation("PHANTOM", None, Corridor(-9, -9),
                                         frozenset(), sim._lane_for(5), 0.0))
    serves2 = GEN._serve(sim, cid, sim.now)
    g2 = [g for g in serves2 if g.job_ref.job_id == "J-OUT-A"][0]
    assert g2.mandatory is True
    assert g2.feasible is False and g2.mask_reason == "LANE_CONFLICT"


def test_mandatory_plan_failed_no_crash():
    """혼잡으로 계획 불가한 mandatory SERVE(plan=None, PLAN_FAILED)도 어댑터에서 크래시 없이
    보존·검증 통과 (SLA-임박 순간의 하드크래시 회귀 가드, 적대리뷰 확정건)."""
    from yard_rl.domain.enums import JobStatus
    sim = _sim()
    _first_decision(sim)
    j = sim.jobs["J-OUT-A"]
    j.actual_block_arrival = sim.now - 2000.0
    j.status = JobStatus.WAITING
    geom = sim.profile.block
    allslots = frozenset((b, r) for b in range(1, geom.bay_count + 1)
                         for r in range(1, geom.row_count + 1))
    sim.reservations.reserve(Reservation("PHANTOM", None, Corridor(-9, -9), allslots, None, 0.0))
    gen = GEN.generate(sim, "YC-A", InformationLevel.PRE_ADVICE)
    failed = [g for g in gen.items if g.mandatory and g.plan is None]
    assert failed and failed[0].feasible is False and failed[0].mask_reason == "PLAN_FAILED"
    cs = _build_candidate_set(sim, "YC-A", gen, sim.now, InformationLevel.PRE_ADVICE, (), GEN.k_max)
    validate_candidates(cs)     # 크래시 없이 계약 통과


def test_prune_k_too_small():
    """mandatory 수가 budget 초과 → K_TOO_SMALL (조용한 유실 금지)."""
    gen = CandidateGenerator(k_max=3)   # budget = 2
    fake = [GenCandidate(0, CandidateKind.SERVE, None, None, True, True, None, 0.0)
            for _ in range(4)]
    with pytest.raises(ConstraintViolation, match="K_TOO_SMALL"):
        gen._prune(fake)


def test_feasibility_matches_reservation():
    """1차 mask = ReservationTable 판정 (동일 소스): _committed_reason None ⟺ can_reserve."""
    sim = _sim()
    _first_decision(sim)
    cid = sim.fleet.ids()[0]
    for g in GEN._serve(sim, cid, sim.now):
        if g.plan is None:
            continue
        r = sim._reservation(g.plan)
        assert (GEN._committed_reason(sim, g.plan) is None) == sim.reservations.can_reserve(r)


def test_four_kinds_representable():
    """SERVE/REPOSITION/WAIT (+PRE_REHANDLE 조건부) 가 계약 후보로 표현."""
    sim = _sim()
    dp = _first_decision(sim)
    kinds = set()
    for cid in dp.crane_ids:
        gen = GEN.generate(sim, cid, InformationLevel.PRE_ADVICE)
        kinds |= {g.kind for g in gen.items}
    assert CandidateKind.SERVE in kinds
    assert CandidateKind.WAIT in kinds
    assert CandidateKind.REPOSITION in kinds   # 미래 본선작업 근접


def test_pre_rehandle_gated_by_info_level():
    """PRE_REHANDLE 은 PRE_ADVICE + provided_eta 에서만 (누출 0)."""
    sim = _sim()
    _first_decision(sim)
    cid = sim.fleet.ids()[0]
    # BLOCK_ARRIVAL 레벨에서는 미발행
    assert GEN._pre_rehandle(sim, cid, sim.now, InformationLevel.BLOCK_ARRIVAL) == []


def test_no_leak_actual_arrival_independent():
    """provided_eta 고정·actual_block_arrival 만 다른 두 시나리오의 후보 존재집합 동일 (누출 0)."""
    def cand_tokens(shift):
        sc = build_minimal_terminal_scenario()
        for j in sc.jobs:
            if j.is_external_truck and j.actual_block_arrival is not None:
                j.actual_block_arrival += shift      # 진실 도착만 이동 (정책 비가시)
                j.actual_gate_in = max(0.0, j.actual_block_arrival - 600.0)
        s = TerminalSimulator(PROF, sc, info_level=InformationLevel.PRE_ADVICE)
        s.run_until_decision()
        cid = s.fleet.ids()[0]
        return {(g.kind.value, g.job_ref.job_id if g.job_ref else None)
                for g in GEN.generate(s, cid, InformationLevel.PRE_ADVICE).items}
    # provided_eta 는 전부 None(동일)이므로 PRE_REHANDLE 은 어차피 미발행 — 존재집합이 도착시프트에 불변
    assert cand_tokens(0.0) == cand_tokens(50.0)


def test_generator_determinism():
    s1, s2 = _sim(), _sim()
    d1, d2 = _first_decision(s1), _first_decision(s2)
    for cid in d1.crane_ids:
        g1 = GEN.generate(s1, cid, InformationLevel.PRE_ADVICE)
        g2 = GEN.generate(s2, cid, InformationLevel.PRE_ADVICE)
        assert [(g.kind, g.job_ref.job_id if g.job_ref else None, g.feasible) for g in g1.items] == \
               [(g.kind, g.job_ref.job_id if g.job_ref else None, g.feasible) for g in g2.items]
