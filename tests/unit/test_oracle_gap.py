"""YR-031 — oracle gap beam search 계약 테스트."""
import json

import pytest

from yard_rl.experiments.oracle_gap import (OracleGapConfig, _beam_day,
                                            _greedy_day, quick_oracle_config,
                                            run_oracle_gap)
from yard_rl.experiments.coverage_ablation import _gen_params
from yard_rl.experiments.direct_job_runner import _scenario
from yard_rl.io.profile_loader import load_profile

PROFILE = "configs/terminals/hjnc_armg.yaml"


def _setup(n=10, seed=160_001):
    cfg = OracleGapConfig(test_episodes=1, beam_width=4, n_external=n, quick=True)
    profile = load_profile(PROFILE)

    class _Shim:
        n_external = cfg.n_external
        drain_window_s = cfg.drain_window_s

    scenario = _scenario(profile, seed, _gen_params(_Shim()), n)
    return cfg, profile, scenario


def test_beam_never_worse_than_greedy():
    """greedy 트랙 상시 유지 → best_found ≤ greedy (구성적 보장)."""
    cfg, profile, scenario = _setup()
    greedy_mean, trace = _greedy_day(profile, scenario, cfg)
    assert len(trace) == 10  # 결정 수 = 작업 수 (lockstep 전제)
    best = _beam_day(profile, scenario, trace, cfg)
    assert best <= greedy_mean + 1e-9


def test_beam_deterministic():
    cfg, profile, scenario = _setup()
    _g, trace = _greedy_day(profile, scenario, cfg)
    assert (_beam_day(profile, scenario, trace, cfg)
            == _beam_day(profile, scenario, trace, cfg))


def test_quick_run_end_to_end(tmp_path):
    report = run_oracle_gap(out_dir=str(tmp_path / "out"),
                            cfg=quick_oracle_config(), progress=lambda _m: None)
    text = report.read_text(encoding="utf-8")
    assert "총상금" in text and "판정" in text
    payload = json.loads(
        (tmp_path / "out" / "oracle_gap_results.json").read_text(encoding="utf-8"))
    assert payload["verdict"] in ("CLOSED", "OPEN", "INTERMEDIATE")
    assert all(r["improvement"] >= -1e-9 for r in payload["per_day"])
