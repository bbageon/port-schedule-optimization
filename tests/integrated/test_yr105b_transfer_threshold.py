from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yard_rl.experiments import yr105_conditional_transfer as y5
from yard_rl.experiments import yr105b_transfer_threshold as y105b


def test_gap_threshold_is_explicit_and_does_not_leak_global_state():
    old_deadline = y5.ACHIEVABLE_DEADLINE
    old_threshold = y5.THRESH
    y5.ACHIEVABLE_DEADLINE = True
    seeds = {"A": 920_000, "B": 920_001}
    try:
        implicit = y5.run_arm(0, "pilot", vessel_guard=False, seeds=seeds)
        explicit = y5.run_arm(
            0, "pilot", vessel_guard=False, seeds=seeds, gap_threshold=old_threshold)
        disabled = y5.run_arm(
            0, "pilot", vessel_guard=False, seeds=seeds, gap_threshold=float("inf"))
    finally:
        y5.ACHIEVABLE_DEADLINE = old_deadline

    assert explicit == implicit
    assert disabled["n_moved"] == 0
    assert disabled["gap_threshold"] == float("inf")
    assert explicit["a2o_mean_min_raw"] is not None
    assert explicit["n_a2o"] > 0
    assert explicit["total_raw"] == pytest.approx(explicit["total"], abs=0.001)
    assert y5.THRESH == old_threshold


def test_pilot_select_confirm_realizations_are_disjoint():
    pilot, _ = y105b._band("pilot", 2)
    select, _ = y105b._band("select", 2, exclude=pilot.all_hashes)
    confirm, report = y105b._band(
        "confirm", 2, exclude=pilot.all_hashes | select.all_hashes)
    assert report["ok"]
    assert not (pilot.all_hashes & select.all_hashes)
    assert not (pilot.all_hashes & confirm.all_hashes)
    assert not (select.all_hashes & confirm.all_hashes)


def test_action_digest_ignores_threshold_number_but_detects_actual_action():
    base = [{
        "t": 10.0, "job": "j1", "src": "A", "dst": "B",
        "gap": 0.11, "gap_threshold": 0.05,
        "fired": True, "blocked_by_vessel": False,
        "transferred": True, "rejected": False,
    }]
    threshold_only = [{**base[0], "gap_threshold": 0.20, "gap": 0.31}]
    changed = [{**base[0], "transferred": False, "dst": None}]
    assert y105b._canonical_trace(base) == y105b._canonical_trace(threshold_only)
    assert y105b._canonical_trace(base) != y105b._canonical_trace(changed)


def test_select_candidate_requires_both_metrics_and_uses_maximin():
    winner, label, scores = y105b._select_candidate({
        "0.05": {"total": 12.0, "a2o_min": 0.4},
        "0.20": {"total": 6.0, "a2o_min": 0.8},
    })
    assert label == "CANDIDATE"
    assert winner == 0.20
    assert scores["0.05"] == pytest.approx(0.4)
    assert scores["0.20"] == pytest.approx(0.6)

    winner, label, _ = y105b._select_candidate({
        "0.05": {"total": 2.0, "a2o_min": -0.1},
        "0.20": {"total": -1.0, "a2o_min": 0.2},
    })
    assert winner is None
    assert label == "NO_CANDIDATE"


def test_select_candidate_exact_tie_prefers_conservative_threshold():
    winner, label, _ = y105b._select_candidate({
        "0.05": {"total": 5.0, "a2o_min": 0.5},
        "0.20": {"total": 5.0, "a2o_min": 0.5},
    })
    assert label == "CANDIDATE"
    assert winner == 0.20


@pytest.mark.parametrize(
    ("total_ci", "a2o_ci", "equiv", "power_ok", "expected"),
    [
        ((10.1, 15.0), (1.1, 2.0), False, True, "JOINT_PRACTICAL_IMPROVEMENT"),
        ((0.1, 5.0), (0.1, 0.8), False, True, "JOINT_CONFIRMED_SMALL"),
        ((0.1, 5.0), (-0.8, -0.1), False, True, "TRADEOFF_FAIL"),
        ((-5.0, -0.1), (-0.5, 0.5), False, True, "HARMFUL"),
        ((-2.0, 2.0), (-0.2, 0.2), True, True, "EQUIVALENT"),
        ((-2.0, 2.0), (-2.0, 2.0), False, True, "INCONCLUSIVE"),
        ((0.1, 5.0), (0.1, 0.8), False, False, "POWER_FAIL"),
    ],
)
def test_confirm_joint_classification(total_ci, a2o_ci, equiv, power_ok, expected):
    primary = {
        "total": {"ci": list(total_ci), "equivalent": equiv},
        "a2o_min": {"ci": list(a2o_ci), "equivalent": equiv},
    }
    assert y105b._classification(primary, True, power_ok) == expected
    assert y105b._classification(primary, False, power_ok) == "INVALID"


def test_head_artifact_rejects_uncommitted_content(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    path.write_text('{"local": 1}', encoding="utf-8")
    monkeypatch.setattr(y105b, "_relative", lambda _path: "artifact.json")
    monkeypatch.setattr(y105b, "_head_blob", lambda _path: b'{"head": 1}')
    with pytest.raises(RuntimeError, match="HEAD와 다르다"):
        y105b._require_head_artifact(path)


def test_source_contract_change_is_fail_closed(monkeypatch):
    frozen = {"source_contract": {"files": {"a": "old"}, "digest": "old"}}
    monkeypatch.setattr(
        y105b, "_source_snapshot",
        lambda: {"files": {"a": "new"}, "digest": "new"})
    with pytest.raises(RuntimeError, match="계약이 바뀌었다"):
        y105b._require_source_contract(frozen)


def test_guard_fails_closed_on_missing_a2o_and_identical_actions():
    arms = {}
    traces = {}
    for key in ("0.05", "0.10"):
        arms[key] = {
            "compl": 1.0, "backlog": 0, "policy_exceptions": 0,
            "total": 3.0, "total_raw": 3.0,
            "a2o_mean_min_raw": None, "n_a2o": 0,
            "chan": {"truck": 1.0, "vessel": 1.0, "move": 1.0,
                     "other": 0.0, "total": 3.0},
        }
        traces[key] = {"action_digest": "same"}
    guard = y105b._guard([{"arms": arms, "traces": traces}], (0.05, 0.10))
    assert not guard.ok
    assert any("A→O" in failure for failure in guard.failures)
    assert any("조작 미발화" in failure for failure in guard.failures)


def _pilot_rows() -> list[dict]:
    rows = []
    for i in range(16):
        wave = (i % 5) - 2
        arms = {}
        for key, total_shift, a2o_shift in (
            ("0.05", -2.0 + 0.4 * wave, -0.2 + 0.03 * wave),
            ("0.10", 0.0, 0.0),
            ("0.20", 1.0 - 0.2 * wave, 0.1 - 0.02 * wave),
        ):
            total = 100.0 + i * 0.2 + total_shift
            a2o = 20.0 + i * 0.01 + a2o_shift
            arms[key] = {
                "total": total, "total_raw": total,
                "a2o_mean_min": round(a2o, 2), "a2o_mean_min_raw": a2o,
                "n_a2o": 20, "n_moved": 1, "n_rejected": 0,
                "compl": 1.0, "backlog": 0, "policy_exceptions": 0,
                "chan": {"truck": 1.0, "vessel": 1.0, "move": 1.0,
                         "other": 0.0, "total": total},
            }
        rows.append({"arms": arms, "traces": {
            key: {"action_digest": f"{key}-{i}"} for key in arms}})
    return rows


def test_pilot_output_seals_means_ci_and_raw_rows(tmp_path, monkeypatch):
    manifest_path = tmp_path / "prereg_manifest.json"
    power_path = tmp_path / "power_note.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(y105b, "OUT", tmp_path)
    monkeypatch.setattr(y105b, "MANIFEST", manifest_path)
    monkeypatch.setattr(y105b, "POWER_NOTE", power_path)
    monkeypatch.setattr(y105b, "_require_clean", lambda: None)
    manifest = {
        "pilot_band": {"seeds": {"A": list(range(16)), "B": list(range(16, 32))},
                       "realization_hashes": {"A": ["a"] * 16, "B": ["b"] * 16}},
        "source_contract": {"digest": "d", "files": {}},
    }
    monkeypatch.setattr(y105b, "_manifest", lambda: manifest)
    monkeypatch.setattr(y105b, "_historical_hashes", lambda: frozenset())
    monkeypatch.setattr(y105b, "_sha256", lambda _path: "manifest-sha")
    rows = _pilot_rows()
    monkeypatch.setattr(
        y105b, "_run_rows",
        lambda *args, **kwargs: (
            rows, {"band": manifest["pilot_band"], "independence": {"ok": True}}))
    monkeypatch.setattr(
        y105b, "_guard", lambda *_args: SimpleNamespace(ok=True, failures=[]))

    class FakeBand:
        def freeze_json(self):
            return {"seeds": {"A": [1], "B": [2]},
                    "realization_hashes": {"A": ["x"], "B": ["y"]}}

    monkeypatch.setattr(
        y105b, "_band", lambda *args, **kwargs: (FakeBand(), {"ok": True}))
    result = y105b.run_pilot()
    encoded = json.dumps(result, ensure_ascii=False)
    assert len(result["power_by_pair"]) == 6
    assert '"rows"' not in encoded
    assert '"mean"' not in encoded
    assert '"ci"' not in encoded
    assert result["sealed"].startswith("pilot arm 평균")
