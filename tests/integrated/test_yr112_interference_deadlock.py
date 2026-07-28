"""YR-112 — 크레인 간섭 교착: 결정 기회가 열리지 않아 작업이 사라지던 결함.

증상(실측 seed 902013): 유휴 크레인 2기가 대상 bay 를 **사이에 두고** 안전간격 안에 서면
어느 쪽도 그 bay 를 잡을 수 없는데(물리적으로 옳다), 비켜설 **결정 기회 자체가 열리지
않아** 런이 그대로 끝났다. 미완 작업은 비용에 계상되지 않으므로 표본이 조용히 왜곡된다.

수정 방향: 물리(유휴 크레인 관통 금지, YR-091)는 그대로 두고 **결정 기회만** 연다.
발화 조건이 "이벤트도 wake 도 없고 · 실행가능 SERVE 도 없고 · 간섭으로만 막힌 작업이 있다"
셋 모두일 때라, 정상 런에는 후보도 결정도 하나 늘지 않는다(골든 보존).
"""
import dataclasses

import pytest

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                          _wait_of)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.reservation import Corridor
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE
DEADLOCK_SEED = 902_013          # 실측 재현 시드 (고load·정합마감)


def _sim(seed, *, level="high", dm=0.5, achievable=True, **kw):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(level, vessel_deadline_mult=dm),
                            time_contract_v2=True, gate_block_contract=True,
                            vessel_deadline_achievable=achievable, **kw)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def _run(sim):
    pol, gen = ResolverPolicy(ServiceFirstSPTPreference(), "SF"), CandidateGenerator()
    dp = sim.run_until_decision()
    while dp is not None:
        gb = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    done = sum(1 for j in sim.jobs.values() if j.status.name == "DONE")
    escapes = [e for e in sim.event_log if e[1] == "DEADLOCK_ESCAPE"]
    return done / len(sim.jobs), sim.unfinished_backlog(), escapes


def test_deadlock_seed_now_completes():
    """회귀 고정 — 이 시드가 다시 미완으로 끝나면 결함이 돌아온 것이다."""
    compl, backlog, escapes = _run(_sim(DEADLOCK_SEED))
    assert compl == 1.0 and backlog == 0
    assert len(escapes) == 1, "탈출이 정확히 한 번 필요했다"


def test_escape_never_fires_in_healthy_runs():
    """정상 런에는 발화 0 — 이게 깨지면 골든이 바뀐다(추가적 변경이 아니게 된다)."""
    for level in ("mid", "high"):
        for dm in (0.5, 2.0):
            for i in range(3):
                compl, backlog, escapes = _sim(830_000 + i * 7, level=level, dm=dm,
                                               achievable=False), None, None
                compl, backlog, escapes = _run(compl)
                assert escapes == [], f"{level}/{dm}/{i} 에서 예기치 않은 탈출"
                assert compl == 1.0 and backlog == 0


def test_predicate_is_false_in_healthy_state():
    """술어는 정상 상태에서 거짓이어야 한다 (생성기 후보도 늘지 않는다)."""
    sim = _sim(830_000, level="mid", dm=2.0, achievable=False)
    sim.run_until_decision()
    assert sim.interference_deadlock_corridors() == ()


def test_escape_reposition_actually_clears_the_corridor():
    """탈출 재배치는 **반대편 크레인의 통로를 실제로 연다** (해석식 검증).

    실측 구조: YC-L 22 · YC-W 24 · 대상 23 · 안전간격 2.
      · 현 상태 — 어느 쪽도 23 을 못 잡는다 (corridor 가 상대 점 장벽과 겹침)
      · YC-L 을 20 으로 물리면 YC-W 의 (23,24) 통로가 열린다
    """
    gap = 2.0
    assert Corridor(22.0, 23.0).overlaps(Corridor(24.0, 24.0), gap)   # YC-L 이 23 → 막힘
    assert Corridor(23.0, 24.0).overlaps(Corridor(22.0, 22.0), gap)   # YC-W 가 23 → 막힘
    assert not Corridor(20.0, 22.0).overlaps(Corridor(24.0, 24.0), gap)   # 물러나기 자체는 가능
    assert not Corridor(23.0, 24.0).overlaps(Corridor(20.0, 20.0), gap)   # 물러난 뒤 통로 열림


def test_escape_uses_corridor_not_single_bay():
    """두 번째 실측 형태 — 재조작이 끼어 통로가 (17,19) 인데 막는 크레인이 bay 16 에 있다.

    대상 bay(19) 하나만 보고 물러날 거리를 재면 16 이 나와 **아무도 안 움직인다**(현 위치).
    통로의 **가까운 끝**(lo=17) 기준이라야 15 가 나오고, 그때 비로소 통로가 열린다.
    """
    gap = 2.0
    assert Corridor(16.0, 16.0).overlaps(Corridor(17.0, 19.0), gap)       # 16 은 막는다
    assert not Corridor(15.0, 15.0).overlaps(Corridor(17.0, 19.0), gap)   # 15 면 열린다
    assert 19.0 - (gap + 1.0) == 16.0                                     # 구 계산 = 제자리
    assert 17.0 - gap == 15.0                                             # 통로 기준 = 1 bay 이동


def test_one_bay_escape_candidate_is_actually_generated():
    """해석식이 아니라 **생성기가 실제로 1 bay 후보를 내는지** 본다.

    앞 테스트는 산술만 확인한다 — 생성기가 그 목표를 발행하지 않으면 아무 소용이 없다.
    여기서는 교착 상태를 인위적으로 만들고(크레인 위치를 통로 (lo,hi) 옆에 세움)
    `_escape_bays` 의 중간값이 아니라 최종 `_reposition` 후보까지 통과하는지 확인한다.
    """
    sim = _sim(DEADLOCK_SEED)
    sim.run_until_decision()
    gen = CandidateGenerator()
    cid = sorted(sim.fleet.ids())[0]
    gap = sim.reservations.safety_gap_bay
    yc = sim.fleet.get(cid)
    yc.state.position_bay = 16.0
    # 술어를 실측 교착 형태로 고정 (엔진 상태 재현 대신 계약 경계만 검증)
    sim.interference_deadlock_corridors = lambda: ((17.0, 19.0),)
    bays = gen._escape_bays(sim, cid)
    assert 17.0 - gap in bays, f"통로 기준 목표(15)가 없다: {bays}"
    assert min(abs(b - 16.0) for b in bays) == pytest.approx(1.0), "1 bay 이동이 후보에 있어야"
    # 일반 REPOSITION 의 '이동가치 하한(>1 bay)' 이 탈출에 적용되면 이 후보가 사라진다
    assert all(not Corridor(b, b).overlaps(Corridor(17.0, 19.0), gap) for b in bays)
    final = gen._reposition(sim, cid, sim.now, LEVEL)
    one_bay = [g for g in final if g.job_ref.reposition_target_bay == pytest.approx(15.0)]
    assert one_bay, "통로 기준 1 bay 목표가 최종 REPOSITION 후보에서 사라졌다"
    assert any(g.feasible for g in one_bay), "1 bay 탈출 후보가 실행 불가능하게 발행됐다"


def test_generator_emits_feasible_escape_in_real_deadlock():
    """실제 교착 시드에서 생성기가 **실행가능한(feasible)** 재배치를 내는지 — 계약 종단 검증."""
    sim = _sim(DEADLOCK_SEED)
    gen = CandidateGenerator()
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    dp = sim.run_until_decision()
    hit = []
    while dp is not None:
        gb = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        if sim.interference_deadlock_corridors():
            for c in dp.crane_ids:
                for g in gb[c].items:
                    if g.kind.value == "REPOSITION" and g.feasible:
                        hit.append((c, sim.fleet.get(c).state.position_bay,
                                    g.job_ref.reposition_target_bay))
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    assert hit, "교착 시점에 실행가능한 탈출 재배치가 하나도 발행되지 않았다"


def test_escape_count_is_exposed_for_audit():
    """arm 별로 저장할 수 있게 발화 횟수가 노출된다 (결과 개입 여부 감사용)."""
    sim = _sim(DEADLOCK_SEED)
    assert sim.deadlock_escape_count == 0
    _run(sim)
    assert sim.deadlock_escape_count == 1


def test_escape_respects_strictly_increasing_decision_times():
    """YR-050 계약 — 결정 시각은 **엄격히 증가**한다(wake 1회성).

    탈출이 "이미 결정을 연 시각"에 다시 열면 그 계약을 깬다(실측으로 한 번 깼다).
    정책이 이미 답한 시각에 재질문하지 않고 다음 시각으로 미룬다.
    """
    sim = _sim(DEADLOCK_SEED)
    gen = CandidateGenerator()
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    times = []
    dp = sim.run_until_decision()
    while dp is not None:
        times.append(dp.time)
        gb = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    assert all(b > a for a, b in zip(times, times[1:])), "같은 시각 재결정 발생"


def test_escape_mode_contract_is_explicit():
    """탈출 시점 계약이 기본값으로 못 박혀 있어야 한다 (암묵 동작 금지).

    정본 = `immediate`. 근거(실측): 기존 판정 대역 32런에서 즉시 모드 발화 0회·해시 전부
    동일(골든 안전)이면서, 신규 대역에서는 지연 모드가 놓치는 교착을 더 잡는다.
    """
    sim = _sim(DEADLOCK_SEED)
    assert sim.escape_mode == "immediate"


def test_delayed_mode_still_available_for_comparison():
    """비교군(초판 계약)을 지울 수 없어야 한다 — 판정의 계약 민감도를 재려면 필요하다."""
    sim = _sim(DEADLOCK_SEED)
    sim.escape_mode = "delayed"
    compl, backlog, escapes = _run(sim)
    assert compl == 1.0 and backlog == 0 and len(escapes) == 1


def test_immediate_mode_preserves_legacy_band_runs():
    """기존 판정 대역(정합마감 OFF, 830k)에서 즉시 모드가 런을 바꾸지 않는다.

    이게 깨지면 과거 판정이 전부 재실행 대상이 된다 — 계약 채택의 전제 조건.
    """
    for level in ("mid", "high"):
        for dm in (0.5, 2.0):
            for i in range(2):
                a = _sim(830_000 + i * 7, level=level, dm=dm, achievable=False)
                b = _sim(830_000 + i * 7, level=level, dm=dm, achievable=False)
                a.escape_mode, b.escape_mode = "delayed", "immediate"
                _run(a)
                _run(b)
                assert a.event_stream_hash() == b.event_stream_hash()
                assert b.deadlock_escape_count == 0


def test_multiblock_transfer_seed_completes():
    """다중블록 + 이송 경로에서만 나타나던 두 번째 교착의 회귀 고정."""
    from yard_rl.experiments import yr105_conditional_transfer as y
    prev = y.ACHIEVABLE_DEADLINE
    y.ACHIEVABLE_DEADLINE = True
    try:
        r = y.run_arm(17, "confirm", vessel_guard=False, seeds={"A": 902_017, "B": 902_017})
    finally:
        y.ACHIEVABLE_DEADLINE = prev
    assert r["compl"] == 1.0 and r["backlog"] == 0


def test_escape_offers_reposition_candidate_when_deadlocked():
    """교착 상태에서 생성기가 '비켜서는' 재배치를 실제로 발행한다."""
    sim = _sim(DEADLOCK_SEED)
    gen = CandidateGenerator()
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    dp = sim.run_until_decision()
    seen_escape_candidate = False
    while dp is not None:
        gb = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        if sim.interference_deadlock_corridors():
            for c in dp.crane_ids:
                if any(g.kind.value == "REPOSITION" and g.feasible for g in gb[c].items):
                    seen_escape_candidate = True
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    assert seen_escape_candidate, "교착 시점에 실행가능한 재배치 후보가 있어야 한다"


def test_escape_does_not_loop_forever():
    """무한루프 방지의 **정확한 불변식**: 같은 시각에는 두 번 열리지 않는다.

    즉시 모드에서 정책이 계속 거부(WAIT)하면 시계가 진행할 때마다 다시 묻는다 — 이건
    결함이 아니라 설계다(교착이 남아 있는 한 결정 기회를 계속 주는 것이 옳다). 보장해야
    할 것은 "같은 시각 재개방 없음 = 시계가 반드시 전진한다" 뿐이다.
    """
    sim = _sim(DEADLOCK_SEED)
    gen = CandidateGenerator()
    dp = sim.run_until_decision()
    n = 0
    while dp is not None:
        n += 1
        assert n < 5000, "결정 루프가 종료되지 않음"
        gb = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})   # 항상 WAIT
        dp = sim.run_until_decision()
    times = [e[0] for e in sim.event_log if e[1] == "DEADLOCK_ESCAPE"]
    assert len(times) == len(set(times)), "같은 시각에 두 번 열렸다 (무한루프 위험)"


def test_real_policy_needs_only_one_escape():
    """정상 정책은 한 번만 열어 주면 교착을 푼다 (반복 발화는 거부의 결과일 뿐)."""
    sim = _sim(DEADLOCK_SEED)
    compl, backlog, escapes = _run(sim)
    assert compl == 1.0 and backlog == 0 and len(escapes) == 1


def test_unfinished_work_would_have_been_silently_dropped():
    """결함의 **연구적 위험**을 고정: 미완 작업은 비용에 안 잡힌다 = 조용한 왜곡."""
    sim = _sim(DEADLOCK_SEED)
    compl, backlog, _ = _run(sim)
    assert backlog == 0, "backlog 가 남으면 그 시드의 비용 비교는 무효로 취급해야 한다"
    assert compl == pytest.approx(1.0)
