"""YR-150 1단계 — 터미널 전체 유입 스트림·배분·관측창 계약 테스트."""
import pytest

from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.repro import code_dirty, git_dirty
from yard_rl.integrated.scenario_gen import GATE_BLOCK_MAX_S, GATE_BLOCK_MIN_S
from yard_rl.integrated.terminal_stream import (LOAD_WINDOW_S, ObservationContract,
                                                TerminalStreamParams, allocate,
                                                build_terminal, distribution_vector)
from yard_rl.integrated.yard_layout import terminal_layout

SEED = 4_242_000


@pytest.fixture(scope="module")
def built():
    return build_terminal(build_calibrated_profile(), SEED,
                          params=TerminalStreamParams(load_4h=100))


# ------------------------------------------------------------------ 관측창 계약
def test_observation_rejects_snapshot_outside_5_to_10_min():
    with pytest.raises(ValueError):
        ObservationContract(snapshot_s=60.0)
    with pytest.raises(ValueError):
        ObservationContract(snapshot_s=900.0)


def test_snapshot_grid_covers_measurement_window():
    obs = ObservationContract()
    ts = obs.snapshot_times()
    assert ts[0] == obs.warmup_s and ts[-1] == obs.observe_s
    assert len(ts) == int(obs.measure_s // obs.snapshot_s) + 1


# ------------------------------------------------------------------ 배분
def test_allocation_sums_exactly_and_is_deterministic():
    layout = terminal_layout()
    p = distribution_vector(layout, TerminalStreamParams(load_4h=100))
    assert sum(p.values()) == pytest.approx(1.0)
    for n in (21, 100, 137, 188):
        counts = allocate(p, n)
        assert sum(counts.values()) == n
        assert allocate(p, n) == counts


def test_hotspot_block_receives_more():
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=100, hotspot_blocks=("Y05",),
                                  hotspot_weight=3.0)
    p = distribution_vector(layout, params)
    assert p["Y05"] > p["Y06"] * 2.5


# ------------------------------------------------------------------ master stream
def test_stream_is_split_not_multiplied(built):
    """21블록 합계가 터미널 스트림과 정확히 같아야 한다(21배 부풀림 금지)."""
    total = sum(len([j for j in s.jobs if j.is_external_truck])
                for s in built["scenarios"].values())
    assert total == built["n_total"]


def test_arrival_rate_matches_load_definition():
    """L 은 **4시간당 터미널 전체 도착량**이다 — 관측창이 길면 같은 비율로 늘어난다."""
    obs = ObservationContract()
    b = build_terminal(build_calibrated_profile(), SEED,
                       params=TerminalStreamParams(load_4h=100), obs=obs)
    assert b["n_total"] == round(100 * obs.observe_s / LOAD_WINDOW_S)


def test_arrivals_continue_to_observation_end(built):
    """도착이 관측창 끝까지 이어져야 한다 — 비우기 구조가 아니다."""
    obs_end = built["observation"]["observe_s"]
    last = max(j.actual_gate_in for s in built["scenarios"].values()
               for j in s.jobs if j.is_external_truck)
    assert last >= 0.9 * obs_end


def test_scenario_ends_at_observation_time(built):
    """관측시간에서 종료 — 비우기 구간을 두지 않는다."""
    obs_end = built["observation"]["observe_s"]
    assert all(s.end_time == obs_end for s in built["scenarios"].values())
    assert all(s.drain_window_s == 0.0 for s in built["scenarios"].values())


def test_travel_depends_on_destination_block_and_stays_in_support(built):
    """게이트→블록 주행이 **블록마다 다르고** 계약 지원범위 안에 있어야 한다."""
    means = {}
    for b, s in built["scenarios"].items():
        tv = [j.actual_block_arrival - j.actual_gate_in
              for j in s.jobs if j.is_external_truck]
        assert tv, f"{b}: 배정된 트럭 없음"
        assert all(GATE_BLOCK_MIN_S <= v <= GATE_BLOCK_MAX_S for v in tv)
        means[b] = sum(tv) / len(tv)
    assert means["Y21"] - means["Y01"] > 100.0     # 양 끝 블록의 주행이 확실히 다르다


def test_predicted_arrival_does_not_use_realised_values(built):
    """예측 블록도착은 **예약+기대 주행**이며 실현 편차를 참조하지 않는다(누출 0)."""
    layout = terminal_layout()
    for b, s in built["scenarios"].items():
        for j in s.jobs:
            if not j.is_external_truck:
                continue
            expect = j.appointment_gate_time + layout.gate_to_block_s(b)
            assert j.estimated_block_arrival == pytest.approx(expect)
            assert j.provided_eta == pytest.approx(expect)


def test_vessels_spread_over_blocks_and_window(built):
    obs_end = built["observation"]["observe_s"]
    starts = [v.plan.planned_start_s for s in built["scenarios"].values()
              for v in s.vessels]
    blocks = [b for b, s in built["scenarios"].items() if s.vessels]
    assert len(blocks) >= 5
    assert min(starts) < 0.2 * obs_end and max(starts) > 0.7 * obs_end


def test_build_is_deterministic():
    a = build_terminal(build_calibrated_profile(), SEED,
                       params=TerminalStreamParams(load_4h=75))
    b = build_terminal(build_calibrated_profile(), SEED,
                       params=TerminalStreamParams(load_4h=75))
    assert a["assignment"] == b["assignment"]


# ------------------------------------------------------------------ 재현 스탬프
def test_code_dirty_sees_untracked_sources(tmp_path, monkeypatch):
    """`git_dirty()` 가 놓치는 **미추적 신규 소스**를 `code_dirty()` 는 잡아야 한다."""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert git_dirty() is False          # 미추적 파일은 안 보인다 (구 검사의 구멍)
    assert code_dirty() is True          # 새 검사는 잡는다
