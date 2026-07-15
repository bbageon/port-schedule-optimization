"""YR-031-b — 이탈 추출·AUC·beam 궤적 계약 테스트."""
import json

import pytest

torch = pytest.importorskip("torch")

from yard_rl.experiments.oracle_gap import (OracleGapConfig, _beam_day,  # noqa: E402
                                            _greedy_day, beam_day_with_trace)
from yard_rl.experiments.oracle_pattern import (_auc, PatternConfig,  # noqa: E402
                                                collect_day_events,
                                                quick_pattern_config,
                                                run_oracle_pattern)
from yard_rl.experiments.coverage_ablation import _gen_params  # noqa: E402
from yard_rl.experiments.direct_job_runner import _scenario  # noqa: E402
from yard_rl.io.profile_loader import load_profile  # noqa: E402

PROFILE = "configs/terminals/hjnc_armg.yaml"


def test_auc_rank_statistic():
    assert _auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert _auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert _auc([1, 0], [0.2, 0.9]) == pytest.approx(0.0)


def _setup(seed=160_000, n=10, width=4):
    cfg = OracleGapConfig(test_episodes=1, test_seed0=seed, beam_width=width,
                          n_external=n, quick=True)
    profile = load_profile(PROFILE)

    class Shim:
        n_external = n
        drain_window_s = 86_400.0

    scenario = _scenario(profile, seed, _gen_params(Shim()), n)
    return cfg, profile, scenario


def test_beam_trace_reproduces_beam_value_and_replays_to_same_cost():
    cfg, profile, scenario = _setup()
    greedy_mean, trace = _greedy_day(profile, scenario, cfg)
    best, best_trace = beam_day_with_trace(profile, scenario, trace, cfg)
    assert best == _beam_day(profile, scenario, trace, cfg)  # 값 동일 (원 알고리즘)
    assert best <= greedy_mean + 1e-9 and len(best_trace) == len(trace)
    pcfg = PatternConfig(test_episodes=1, test_seed0=160_000, beam_width=4,
                         n_external=10, quick=True)
    rows = collect_day_events(profile, scenario, best_trace, pcfg)
    assert len(rows) == len(best_trace)
    if best < greedy_mean - 1e-9:            # 개선일이면 이탈이 존재해야 함
        assert any(r["diverged"] for r in rows)
    for r in rows:
        assert len(r["features_context"]) == 22
        if r["diverged"]:
            assert len(r["pair_diff"]) == 5


def test_greedy_trace_replay_has_zero_divergence():
    cfg, profile, scenario = _setup()
    _mean, trace = _greedy_day(profile, scenario, cfg)
    pcfg = PatternConfig(test_episodes=1, test_seed0=160_000, beam_width=4,
                         n_external=10, quick=True)
    rows = collect_day_events(profile, scenario, tuple(trace), pcfg)
    assert not any(r["diverged"] for r in rows)   # greedy 궤적 대조 = 이탈 0


def test_pattern_quick_smoke(tmp_path):
    out = tmp_path / "pattern"
    report = run_oracle_pattern(PROFILE, str(out), quick_pattern_config(),
                                progress=lambda _msg: None)
    text = report.read_text(encoding="utf-8")
    assert "H-A" in text and "H-B" in text
    payload = json.loads((out / "oracle_pattern_results.json")
                         .read_text(encoding="utf-8"))
    assert payload["hypothesis_a"]["verdict"] in ("SUPPORTED", "PARTIAL", "REJECTED")
    assert payload["hypothesis_b"]["verdict"] in (
        "SUPPORTED", "REJECTED", "INSUFFICIENT_DATA")
    assert (out / "divergence_events.json").exists()
