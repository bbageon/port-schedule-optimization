"""Failure evidence must distinguish lack of stock from existing claims/reservations."""
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/v3"
sys.path.insert(0, str(SCRIPTS))
from analyze_admission_replay import failure_kind
from run_admission_replay import parse_cpus


def snapshot():
    return dict(available=True, free_formula_matches=True, inventory_boxes=10,
                unclaimed_inventory_boxes=0, physical_free=90, admission_free=90,
                capacity_margin=2)


@pytest.mark.parametrize("changes,reason,expected", [
    ({"inventory_boxes": 0}, "NO_TARGET", "block_empty"),
    ({}, "NO_TARGET", "all_inventory_claimed"),
    ({"unclaimed_inventory_boxes": 3}, "NO_TARGET", "target_missing_despite_unclaimed_inventory"),
    ({"admission_free": -5}, "capacity", "reservations_consume_available_space"),
    ({"physical_free": 1, "admission_free": -5}, "capacity", "physical_space_at_margin"),
    ({"free_formula_matches": False}, "capacity", "inventory_formula_mismatch"),
    ({"available": False}, "NO_TARGET", "missing_snapshot"),
])
def test_failure_classification_keeps_observation_limits(changes, reason, expected):
    evidence = snapshot()
    evidence.update(changes)
    assert failure_kind({"reason": reason, "inventory_at_failure": evidence}) == expected


def test_missing_observation_does_not_become_inventory_depletion():
    assert failure_kind({"reason": "NO_TARGET"}) == "missing_snapshot"
    assert failure_kind({"reason": "TAIL"}) == "outside_horizon"


@pytest.mark.parametrize("text", ["0,0,1", "0,1", "-1,1,2", "0,1,2,3"])
def test_cpu_contract_rejects_duplicate_or_wrong_worker_count(text):
    with pytest.raises(ValueError):
        parse_cpus(text)


def test_three_distinct_cpus():
    assert parse_cpus("0,1,2") == (0, 1, 2)
