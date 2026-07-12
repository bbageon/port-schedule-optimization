"""합성 시나리오 생성기 테스트 — 결정론·유효성·물량."""
from yard_rl.domain.enums import JobFlow
from yard_rl.domain.validators import validate_scenario
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams, generate

PROFILE = load_profile("configs/terminals/poc_single_crane.yaml")


def test_deterministic_same_seed():
    a = generate(PROFILE, seed=7)
    b = generate(PROFILE, seed=7)
    assert a.scenario_id == b.scenario_id
    assert [j.job_id for j in a.jobs] == [j.job_id for j in b.jobs]
    assert [(j.actual_block_arrival, j.target_container) for j in a.jobs] == \
           [(j.actual_block_arrival, j.target_container) for j in b.jobs]
    assert sorted(a.containers) == sorted(b.containers)


def test_different_seed_differs():
    a = generate(PROFILE, seed=1)
    b = generate(PROFILE, seed=2)
    assert [j.actual_block_arrival for j in a.jobs] != [j.actual_block_arrival for j in b.jobs]


def test_scenario_is_valid_and_within_horizon():
    sc = generate(PROFILE, seed=3, params=GenParams(peak=True))
    validate_scenario(sc.jobs, sc.containers, PROFILE)  # 예외 없어야 함
    for j in sc.jobs:
        if j.actual_block_arrival is not None:
            assert 0.0 <= j.actual_block_arrival <= sc.horizon_s
            assert j.provided_eta is not None and j.provided_eta >= 0.0  # Exp-3 외생 입력
        if j.flow == JobFlow.VESSEL_LOAD:
            assert j.deadline is not None and j.deadline > j.release_time


def test_peak_short_horizon_arrivals_within_bounds():
    """리뷰 확정건 회귀 가드: peak + 짧은 horizon 에서도 도착이 [0, horizon] 안."""
    p = GenParams(n_external=40, n_vessel=0, peak=True, horizon_s=3600.0)
    sc = generate(PROFILE, seed=5, params=p)
    arrivals = [j.actual_block_arrival for j in sc.jobs if j.actual_block_arrival is not None]
    assert arrivals and all(0.0 <= a <= 3600.0 for a in arrivals)


def test_job_mix_counts():
    p = GenParams(n_external=80, gate_out_share=0.5, n_vessel=6, fill_ratio=0.5)
    sc = generate(PROFILE, seed=11, params=p)
    outs = [j for j in sc.jobs if j.flow == JobFlow.GATE_OUT]
    ins = [j for j in sc.jobs if j.flow == JobFlow.GATE_IN]
    vessels = [j for j in sc.jobs if j.flow == JobFlow.VESSEL_LOAD]
    assert len(outs) + len(ins) == 80
    assert len(vessels) == 6
    # 반출 대상 중복 없음, 본선 대상과 분리
    targets = [j.target_container for j in outs + vessels]
    assert len(targets) == len(set(targets))
