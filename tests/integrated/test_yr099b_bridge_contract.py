"""YR-099-b G0 — 브리지 계약 6종 + 골든 불변 (opt-in 훅이 기존 경로를 안 건드림)."""
import dataclasses

import pytest

from yard_rl.domain.enums import InformationLevel, JobFlow, JobStatus
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                          _wait_of)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.engine import ReviewEpoch, TerminalDecision
from yard_rl.integrated.multiblock import (MultiBlockTerminal, TransferError, _namespace_jobs)
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE


def _sim(level="mid", dm=2.0, seed=845_000):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(level, vessel_deadline_mult=dm),
                            time_contract_v2=True, gate_block_contract=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def _run_sf(sim) -> str:
    pol, gen = ResolverPolicy(ServiceFirstSPTPreference(), "SF"), CandidateGenerator()
    dp = sim.run_until_decision()
    while dp is not None:
        assert isinstance(dp, TerminalDecision)
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gen_by))
        except Exception:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    return sim.event_stream_hash()


# ---------------------------------------------------------------- 골든 불변 (opt-in 계약)
def test_golden_unchanged_when_hook_unused():
    a, b = _sim(), _sim()
    b.review_epochs = []                      # 명시적 빈 리스트 = 분기 비성립
    ha, hb = _run_sf(a), _run_sf(b)
    assert ha == hb
    assert a.kpis.berth_overrun_s == b.kpis.berth_overrun_s
    assert a.unfinished_backlog() == b.unfinished_backlog()


# ---------------------------------------------------------------- ① 정확한 결정시점
def test_review_epoch_lands_exactly():
    """현재 시각의 결정은 epoch 보다 우선(정상) — 그 뒤 epoch 은 **정확히** 그 시각에 발화."""
    s = _sim()
    s.review_epochs = [1234.0]
    pol, gen = ResolverPolicy(ServiceFirstSPTPreference(), "SF"), CandidateGenerator()
    out = s.run_until_decision()
    hits = 0
    while out is not None and not isinstance(out, ReviewEpoch):
        gb = {c: gen.generate(s, c, LEVEL) for c in out.crane_ids}
        try:
            _apply(s, pol.decide(s, out, gb))
        except Exception:
            _apply(s, {c: _wait_of(gb[c]) for c in out.crane_ids})
        out = s.run_until_decision()
        hits += 1
        assert hits < 5000
    assert isinstance(out, ReviewEpoch)
    assert out.time == pytest.approx(1234.0) and s.clock == pytest.approx(1234.0)
    assert s.review_epochs == []               # 소비됨


def test_event_before_epoch_wins():
    s = _sim()
    first_evt = s.queue.peek_time()
    s.review_epochs = [first_evt + 100.0]
    out = s.run_until_decision()
    assert not isinstance(out, ReviewEpoch) or out.time <= first_evt + 100.0 + 1e-9


def test_epoch_outside_window_discarded():
    s = _sim()
    s.review_epochs = [s.end + 10_000.0]
    out = s.run_until_decision()
    assert not isinstance(out, ReviewEpoch)
    assert s.review_epochs == []


# ---------------------------------------------------------------- ④ canonical id
def test_namespacing_consistent_and_runnable():
    s = _sim()
    n0, ids0 = len(s.jobs), set(s.jobs)
    _namespace_jobs(s, "A")
    assert len(s.jobs) == n0
    assert all(k.startswith("A:") for k in s.jobs)
    assert all(j.job_id == k for k, j in s.jobs.items())
    for e in s.queue._heap:
        if e.kind_name in ("BLOCK_ARRIVAL", "JOB_RELEASED"):
            assert e.payload in s.jobs
    if s.time_ledger is not None:
        assert all(k.startswith("A:") for k in s.time_ledger.records)
    assert all(jid in s.jobs for _, jid in s._eta_wakes)
    _run_sf(s)                                   # 개명 후에도 완주
    assert s.unfinished_backlog() == 0 or s.clock >= s.end


# ---------------------------------------------------------------- ②③ 공용 시계·전역 장부
def _mbt(seed=846_000):
    return MultiBlockTerminal({"A": _sim("high", 0.5, seed), "B": _sim("mid", 2.0, seed + 500)})


def test_shared_clock_all_blocks_parked_at_same_epoch():
    mbt = _mbt()
    seen = []

    def review(m, t):
        for s in m.blocks.values():
            if s.clock < s.end:
                assert s.clock == pytest.approx(t), "공용 시계 위반"
        seen.append(t)

    pol, gens = ResolverPolicy(ServiceFirstSPTPreference(), "SF"), {}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    mbt.run(policy, review)
    assert len(seen) > 5 and seen == sorted(seen)
    mbt.check_invariants()


def test_global_ledger_survives_transfer():
    mbt = _mbt()
    jid = next(j for j, r in mbt.ledger.records.items()
               if r.flow == JobFlow.GATE_IN.value and r.owner == "A" and r.a_gate_in)
    rec = mbt.ledger.records[jid]
    a0, ver0 = rec.a_gate_in, rec.version
    mbt.blocks["A"].clock = mbt.blocks["B"].clock = a0 + 1.0
    assert mbt.try_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    assert rec.owner == "B" and rec.version == ver0 + 1 and rec.transfer_count == 1
    assert rec.a_gate_in == a0                          # A 는 터미널 보유 — 이송 무관
    assert rec.job_id == jid                            # canonical id 불변 (개명 없음)
    assert jid in mbt.blocks["B"].jobs and jid not in mbt.blocks["A"].jobs
    assert rec.transfer_history[-1][:2] == ("A", "B")
    mbt.check_invariants()


# ---------------------------------------------------------------- ⑤ 2단계·원자성
def test_prepare_validate_rollback_keeps_owner():
    mbt = _mbt()
    jid = next(j for j, r in mbt.ledger.records.items()
               if r.flow == JobFlow.GATE_IN.value and r.owner == "A" and r.a_gate_in)
    rec = mbt.ledger.records[jid]
    mbt.blocks["A"].clock = mbt.blocks["B"].clock = rec.a_gate_in + 1.0
    txn = mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    assert mbt._reserved_inbound["B"] == 1              # 예약됨
    rec.version += 1                                    # 경합 발생 (stale quote)
    with pytest.raises(TransferError):
        mbt.commit(txn)
    mbt.rollback(txn)
    assert mbt._reserved_inbound["B"] == 0              # 예약 해제
    assert rec.owner == "A" and jid in mbt.blocks["A"].jobs   # 소유권 원상 (KEEP)
    mbt.check_invariants()


def test_locked_after_block_in_rejected():
    mbt = _mbt()
    jid = next(j for j, r in mbt.ledger.records.items()
               if r.flow == JobFlow.GATE_IN.value and r.owner == "A" and r.a_gate_in)
    rec = mbt.ledger.records[jid]
    mbt.blocks["A"].clock = mbt.blocks["B"].clock = rec.a_gate_in + 1.0
    mbt.blocks["A"].jobs[jid].status = JobStatus.WAITING     # 이미 블록 도착
    with pytest.raises(TransferError):
        mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)


# ---------------------------------------------------------------- ⑥ 수신 용량
def test_capacity_guard_blocks_transfer():
    mbt = MultiBlockTerminal({"A": _sim("high", 0.5), "B": _sim("mid", 2.0, 846_500)},
                             capacity_margin=10 ** 9)
    jid = next(j for j, r in mbt.ledger.records.items()
               if r.flow == JobFlow.GATE_IN.value and r.owner == "A" and r.a_gate_in)
    rec = mbt.ledger.records[jid]
    mbt.blocks["A"].clock = mbt.blocks["B"].clock = rec.a_gate_in + 1.0
    with pytest.raises(TransferError, match="용량"):
        mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    assert mbt._reserved_inbound["B"] == 0
    assert mbt.free_slots("B") > 0                       # 실제로는 여유 있음(가드가 막은 것)
