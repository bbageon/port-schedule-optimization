"""YR-048 — 통합 시나리오 제공 ETA 주입 + PRE_REHANDLE(선제 재조작) 발생율 계약.

배경: 후보 생성기는 PRE_ADVICE + `job.provided_eta` 를 요구하는데 통합 생성기가 ETA 를
만들지 않아 선제 재조작 후보가 실험에서 **전혀** 발생하지 않았다 (H2 축 통째 비활성 —
YR-047 적대 리뷰 파생 발견). 모델은 단일야드 관행과 동일: eta = 실제도착 ± uniform(300s).
"""
from dataclasses import replace

from yard_rl.contract.schema import CandidateKind
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (BaselinePreference, CandidateGenerator, CentralResolver,
                                TerminalSimulator, build_integrated_profile, record_episode)
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario

PROF = build_integrated_profile()
GEN = CandidateGenerator()
SEED = 310000


def _kind_counts_for_scenario(scenario, level, max_decisions=200):
    """resolver 로 에피소드를 진행하며 생성된 후보 kind 를 계수."""
    sim = TerminalSimulator(PROF, scenario, info_level=level)
    r = CentralResolver(BaselinePreference())
    counts: dict[str, int] = {}
    for _ in range(max_decisions):
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: GEN.generate(sim, c, level) for c in dp.crane_ids}
        for g in gen_by.values():
            for gc in g.items:
                counts[gc.kind.value] = counts.get(gc.kind.value, 0) + 1
        r.apply(sim, r.resolve(sim, dp, gen_by), gen_by)
    return counts


def _kind_counts(seed, level, max_decisions=200):
    return _kind_counts_for_scenario(generate_terminal_scenario(PROF, seed), level,
                                     max_decisions)


def test_external_trucks_have_bounded_eta_and_vessel_jobs_none():
    sc = generate_terminal_scenario(PROF, SEED)
    ext = [j for j in sc.jobs if j.is_external_truck]
    ves = [j for j in sc.jobs if not j.is_external_truck]
    assert ext and ves
    for j in ext:                                     # 반입·반출 전부, 오차는 ±300s 안
        assert j.provided_eta is not None and j.provided_eta >= 0.0
        assert abs(j.provided_eta - j.actual_block_arrival) <= 300.0
    assert all(j.provided_eta is None for j in ves)   # 본선 연계는 ETA 비대상


def test_eta_stream_isolated_scenario_structure_unchanged():
    """전용 RNG 스트림 계약 — eta_error_s 를 바꿔도 구조(도착·대상·컨테이너·본선)는 동일."""
    a = generate_terminal_scenario(PROF, SEED)
    b = generate_terminal_scenario(PROF, SEED, TerminalGenParams(eta_error_s=0.0))
    assert a.containers == b.containers
    assert len(a.jobs) == len(b.jobs)
    for ja, jb in zip(a.jobs, b.jobs):
        assert (ja.job_id, ja.actual_block_arrival, ja.actual_gate_in,
                ja.target_container) == (jb.job_id, jb.actual_block_arrival,
                                         jb.actual_gate_in, jb.target_container)
    for jb in b.jobs:                                 # PERFECT: eta == 실제 도착
        if jb.is_external_truck:
            assert jb.provided_eta == jb.actual_block_arrival
    assert [v.vessel_id for v in a.vessels] == [v.vessel_id for v in b.vessels]


def test_generation_deterministic():
    a = generate_terminal_scenario(PROF, 7)
    b = generate_terminal_scenario(PROF, 7)
    assert a.containers == b.containers and a.jobs == b.jobs


def test_eta_values_pinned_golden():
    """ETA 스트림 파생을 박제 — 전용 스트림(f"eta:{seed}")이 주 스트림으로 회귀하거나
    파생식이 바뀌면 여기서 즉시 발화한다 (리뷰 반영: 격리 테스트만으로는 그 회귀를 못 잡음).
    """
    sc = generate_terminal_scenario(PROF, 310000)
    eta = {j.job_id: j.provided_eta for j in sc.jobs if j.is_external_truck}
    for jid, expect in (("J-IN-003", 1897.9592478515692),
                        ("J-IN-005", 2616.5992022176697),
                        ("J-IN-009", 3999.1432854349496)):
        assert abs(eta[jid] - expect) < 1e-9, f"{jid}: eta 스트림 파생 변경 감지"
    assert sc.meta["eta_error_s"] == 300.0            # arm 정체성 박제 (YR-019 대비)


def test_pre_rehandle_occurs_at_pre_advice_only():
    """발생율 계약 (YR-048 의 목적) + 정보 게이트 (누출 0)."""
    pre = _kind_counts(SEED, InformationLevel.PRE_ADVICE)
    assert pre.get("PRE_REHANDLE", 0) > 0, \
        "PRE_ADVICE 인데 선제 재조작 후보 0건 — H2 축 비활성 (YR-048 재발)"
    for level in (InformationLevel.GATE_IN, InformationLevel.BLOCK_ARRIVAL):
        assert _kind_counts(SEED, level).get("PRE_REHANDLE", 0) == 0, \
            f"{level}: ETA 불가시 레벨에서 PRE_REHANDLE 발행 — 정보 누출"


def test_eta_activates_reposition_channel_as_separate_ablation():
    """ETA는 선제 재조작뿐 아니라 미래 도착 bay 위치선점도 연다 — H2 귀속 분리 계약."""
    full = generate_terminal_scenario(PROF, SEED)
    no_eta = replace(full, jobs=[replace(j, provided_eta=None) for j in full.jobs],
                     meta={**full.meta, "eta_error_s": None})
    with_eta = _kind_counts_for_scenario(full, InformationLevel.PRE_ADVICE)
    without_eta = _kind_counts_for_scenario(no_eta, InformationLevel.PRE_ADVICE)
    assert with_eta.get("PRE_REHANDLE", 0) > 0
    assert without_eta.get("PRE_REHANDLE", 0) == 0
    assert with_eta.get("REPOSITION", 0) > without_eta.get("REPOSITION", 0), \
        "ETA 위치선점 경로가 사라짐 — YR-045 3-arm 기여율 분리가 무효"


def test_record_episode_validates_with_pre_rehandle():
    """계약 통합 — PRE_REHANDLE 이 실제로 후보에 실린 에피소드가 validate_all 을 통과."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED),
                            info_level=InformationLevel.PRE_ADVICE)
    recs = record_episode(sim, info_level=InformationLevel.PRE_ADVICE, episode_id="YR048")
    kinds = {c.kind for r in recs for ob in r.observations for c in ob.candidates.items}
    assert recs and CandidateKind.PRE_REHANDLE in kinds
