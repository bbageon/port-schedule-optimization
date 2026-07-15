"""YR-033 — 선택 프로토콜·winner's curse 진단 계약 테스트."""
import json

import pytest

torch = pytest.importorskip("torch")

from yard_rl.experiments.setfeat_selection import (SelectConfig,  # noqa: E402
                                                   _spearman,
                                                   quick_select_config,
                                                   run_selection_experiment)

PROFILE = "configs/terminals/hjnc_armg.yaml"


def test_spearman_basic():
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert abs(_spearman([1, 2, 3, 4], [1, 1, 1, 1])) < 1e-9   # 무분산 → 0


def test_config_test_band_disjoint_but_reuses_train_val_for_reproduction():
    cfg = SelectConfig()
    # test=fresh 240k, train/val 은 YR-012-c 재현 위해 의도적 동일
    assert cfg.test_seed0 == 240_000
    assert cfg.train_seed0 == 200_000 and cfg.validation_seed0 == 210_000
    assert not (set(cfg.test_seeds) & set(range(220_000, 220_100)))  # YR-012-c test 회피
    with pytest.raises(ValueError):
        SelectConfig(test_seed0=220_000)   # YR-012-c test 재사용 금지
    with pytest.raises(ValueError):
        SelectConfig(val30_episodes=200)   # val 초과
    with pytest.raises(ValueError):
        SelectConfig(smooth_window=2)      # 짝수 금지


def test_selection_quick_smoke_test_not_used_for_selection(tmp_path):
    out = tmp_path / "sel"
    report = run_selection_experiment(PROFILE, str(out), quick_select_config(),
                                      progress=lambda _m: None)
    text = report.read_text(encoding="utf-8")
    assert "winner's curse" in text and "Spearman" in text
    payload = json.loads((out / "selection_results.json").read_text(encoding="utf-8"))
    # 세 프로토콜 전부 산출·선택 ep 는 checkpoint 집합 내
    assert set(payload["protocols"]) == {"P1_val30", "P2_val90", "P3_val90_smooth3"}
    recs = json.loads((out / "checkpoint_records.json").read_text(encoding="utf-8"))
    eps = {r["episode"] for r in recs}
    assert all(ep in eps for ep in payload["protocols"].values())
    # 진단·formal_win 필드 존재
    d = payload["diagnostics"]
    assert "spearman_val90_test" in d and "best_achievable_delta_vs_greedy" in d
    assert all("formal_win" in e for e in payload["paired"].values())
    # 선택은 val 만 — P1 이 val30 argmin 인지 (test 미사용 계약)
    p1_ep = payload["protocols"]["P1_val30"]
    v30 = {r["episode"]: r["val30_mean"] for r in recs}
    assert v30[p1_ep] == min(v30.values())
