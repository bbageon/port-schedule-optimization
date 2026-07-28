from __future__ import annotations

import json

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


@pytest.mark.parametrize(
    ("truck_ci", "vessel_ci", "total_ci", "expected"),
    [
        ((3.1, 5.0), (-1.0, 2.0), (-1.0, 3.0), "PRACTICAL_IMPROVEMENT"),
        ((0.1, 2.0), (-1.0, 2.0), (-1.0, 3.0), "SMALL_CONFIRMED"),
        ((0.1, 2.0), (-11.0, 2.0), (-1.0, 3.0), "TRADEOFF_FAIL"),
        ((-1.0, 2.0), (-1.0, 2.0), (-1.0, 3.0), "INCONCLUSIVE"),
        ((-3.0, -0.1), (-1.0, 2.0), (-1.0, 3.0), "HARMFUL"),
    ],
)
def test_confirm_classification_contract(truck_ci, vessel_ci, total_ci, expected):
    channels = {
        "truck": {"ci": list(truck_ci), "equivalent": False},
        "vessel": {"ci": list(vessel_ci), "equivalent": False},
        "total": {"ci": list(total_ci), "equivalent": False},
    }
    assert y105b._classification(channels, True) == expected
    assert y105b._classification(channels, False) == "INVALID"


def test_select_rejects_sample_size_not_frozen_by_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(y105b, "OUT", tmp_path)
    monkeypatch.setattr(y105b, "_require_clean", lambda: None)
    (tmp_path / "power_note.json").write_text(
        json.dumps({"frozen_sample_plan": {"n_select": 24}, "guards": {"ok": True}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="표본계약"):
        y105b.run_select(23)
