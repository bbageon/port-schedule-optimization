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


def test_golden_unchanged_with_epochs_but_no_transfer():
    """**진짜 계약** (적대검증 critical-1 회귀가드): epoch 을 깔고 통과만 시켜도
    이송이 0건이면 런은 **바이트 동일**해야 한다. 정정 전에는 epoch 이 결정을 선점해
    크레인이 놀고 비용이 24~32% 부풀었다."""
    a = _sim()
    ha = _run_sf(a)
    b = _sim()
    b.review_epochs = sorted({round(j.actual_gate_in, 6) for j in b.jobs.values()
                              if getattr(j, "actual_gate_in", None) is not None and
                              j.flow == JobFlow.GATE_IN and 0 <= j.actual_gate_in <= b.end})
    assert len(b.review_epochs) > 10                       # 실제로 여러 개 깔림
    pol, gen = ResolverPolicy(ServiceFirstSPTPreference(), "SF"), CandidateGenerator()
    out, n_ep = b.run_until_decision(), 0
    while out is not None:
        if isinstance(out, ReviewEpoch):
            n_ep += 1                                      # 통과만 (이송 0)
        else:
            gb = {c: gen.generate(b, c, LEVEL) for c in out.crane_ids}
            try:
                _apply(b, pol.decide(b, out, gb))
            except Exception:
                _apply(b, {c: _wait_of(gb[c]) for c in out.crane_ids})
        out = b.run_until_decision()
    assert n_ep > 10                                       # epoch 이 실제로 발화
    assert b.event_stream_hash() == ha                     # 바이트 동일
    assert b.kpis.berth_overrun_s == a.kpis.berth_overrun_s
    if a.time_ledger is not None:
        assert b.time_ledger.block_area_s == pytest.approx(a.time_ledger.block_area_s)


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


def test_multiblock_schedules_gate_in_at_zero():
    """평가창 시작(t=0)에 gate-in 한 반입도 review 창에서 빠지면 안 된다."""
    a, b = _sim(seed=845_000), _sim(seed=845_001)
    inbound = next(j for j in a.jobs.values() if j.flow == JobFlow.GATE_IN)
    inbound.actual_gate_in = 0.0
    m = MultiBlockTerminal({"A": a, "B": b})
    assert 0.0 in a.review_epochs and 0.0 in b.review_epochs
    assert m.ledger.records[inbound.job_id].a_gate_in == 0.0


def test_event_before_epoch_wins():
    s = _sim()
    first_evt = s.queue.peek_time()
    s.review_epochs = [first_evt + 100.0]
    out = s.run_until_decision()
    assert not isinstance(out, ReviewEpoch) or out.time <= first_evt + 100.0 + 1e-9


def test_epoch_outside_window_discarded():
    """계약: 평가창 밖 epoch 은 **절대 발화하지 않고** 시계를 창 밖으로 밀지 않는다."""
    s = _sim()
    s.review_epochs = [s.end + 10_000.0]
    pol, gen = ResolverPolicy(ServiceFirstSPTPreference(), "SF"), CandidateGenerator()
    out = s.run_until_decision()
    while out is not None:
        assert not isinstance(out, ReviewEpoch), "창 밖 epoch 이 발화"
        gb = {c: gen.generate(s, c, LEVEL) for c in out.crane_ids}
        try:
            _apply(s, pol.decide(s, out, gb))
        except Exception:
            _apply(s, {c: _wait_of(gb[c]) for c in out.crane_ids})
        out = s.run_until_decision()
    assert s.review_epochs == []                 # 폐기됨
    assert s.clock <= s.end + 1e-6


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


# ------------------------------------------------- 게이트 D (YR-106-b) 원자성·멱등성 보완
def _pick(mbt, owner="A"):
    jid = next(j for j, r in mbt.ledger.records.items()
               if r.flow == JobFlow.GATE_IN.value and r.owner == owner and r.a_gate_in)
    rec = mbt.ledger.records[jid]
    for s in mbt.blocks.values():
        s.clock = rec.a_gate_in + 1.0
    return jid, rec


def test_commit_failure_leaves_no_orphan_job():
    """게이트 D 핵심 — 변경 도중 실패해도 작업이 **어느 블록에도 없는** 상태가 되면 안 된다.

    구판은 소스 장부 정합 검사가 `src.jobs.pop()` 뒤에 있어, 그 검사가 실패하면
    작업이 사라지고 터미널 점유가 영구히 어긋났다.
    """
    mbt = _mbt()
    jid, rec = _pick(mbt)
    tl = mbt.blocks["A"].time_ledger
    assert tl is not None
    a = mbt.blocks["A"].jobs[jid].actual_gate_in
    tl._a_sorted.remove(a)                      # 장부 정합을 인위적으로 깨뜨린다
    n_before = len(tl._a_sorted)
    assert mbt.try_transfer(jid, "B", route_s=180.0, travel_s=300.0) is False
    assert jid in mbt.blocks["A"].jobs and jid not in mbt.blocks["B"].jobs
    assert rec.owner == "A" and rec.version == 0 and rec.transfer_count == 0
    assert len(tl._a_sorted) == n_before        # 장부도 손대지 않았다
    assert mbt._reserved_inbound["B"] == 0
    assert mbt.route_cost_s == 0.0              # 주행비도 계상되지 않았다
    mbt.check_invariants()                      # 고아 없음


def test_commit_restores_state_on_unexpected_exception():
    """예상 못 한 예외(TransferError 아님)에도 원상복구 후 재발생해야 한다."""
    mbt = _mbt()
    jid, rec = _pick(mbt)
    txn = mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    boom = RuntimeError("주입 실패")

    class _Explode(dict):
        def __setitem__(self, k, v):
            raise boom

    real = mbt.blocks["B"].jobs
    mbt.blocks["B"].jobs = _Explode(real)
    try:
        with pytest.raises(RuntimeError):
            mbt.commit(txn)
    finally:
        mbt.blocks["B"].jobs = real
    assert jid in mbt.blocks["A"].jobs and rec.owner == "A" and rec.version == 0
    assert mbt.route_cost_s == 0.0
    mbt.check_invariants()


def test_rolled_back_txn_cannot_be_committed():
    """rollback 한 txn 재-commit 금지 — 구판은 **예약 없이** 이송이 성사됐다."""
    mbt = _mbt()
    jid, _ = _pick(mbt)
    txn = mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    mbt.rollback(txn)
    with pytest.raises(TransferError, match="닫힌 트랜잭션"):
        mbt.commit(txn)
    assert jid in mbt.blocks["A"].jobs


def test_same_time_reprepare_does_not_leak_reservation():
    """같은 (job,dst,시각) 재-prepare 시 예약이 새지 않는다 (구판은 키 충돌로 영구 누수)."""
    mbt = _mbt()
    jid, _ = _pick(mbt)
    t1 = mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    t2 = mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    assert t1.txn_id != t2.txn_id and mbt._reserved_inbound["B"] == 2
    mbt.rollback(t1)
    mbt.rollback(t2)
    assert mbt._reserved_inbound["B"] == 0


def test_double_commit_rejected():
    mbt = _mbt()
    jid, _ = _pick(mbt)
    txn = mbt.prepare_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    mbt.commit(txn)
    with pytest.raises(TransferError):
        mbt.commit(txn)
    mbt.check_invariants()


def test_transfer_shifts_provided_eta_with_actual():
    """정책이 읽는 예측 도착(provided_eta)도 경로비용만큼 함께 밀린다."""
    mbt = _mbt()
    jid, _ = _pick(mbt)
    j = mbt.blocks["A"].jobs[jid]
    eta0, est0 = j.provided_eta, getattr(j, "estimated_block_arrival", None)
    if eta0 is None:
        pytest.skip("이 시나리오는 provided_eta 미사용")
    assert mbt.try_transfer(jid, "B", route_s=180.0, travel_s=300.0)
    moved = mbt.blocks["B"].jobs[jid]
    assert moved.provided_eta == pytest.approx(eta0 + 180.0)
    if est0 is not None:
        assert moved.estimated_block_arrival == pytest.approx(est0 + 180.0)
