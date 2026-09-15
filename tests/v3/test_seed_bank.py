"""Input independence, preserved curve, and tamper detection for revision seeds."""
import copy
import math
import pytest

from yard_rl.v3.eval.seed_bank import (create_month_payload, digest, generator_contract,
    load_bundle, seed_domains, validate_payload, validate_seeds, write_bundle)
from yard_rl.v3.stage.month import plan_days
from yard_rl.v3.world.integrated.terminal_stream import diurnal_rate


def test_run_and_background_domains_are_disjoint():
    seeds = [20_000_000 + 100_000 * i for i in range(20)]
    domains = validate_seeds(seeds)
    all_days = [d for item in domains for d in item["days"]]
    assert len(set(all_days)) == 600
    assert domains[-1]["root"] == 21_900_000


@pytest.mark.parametrize("seeds", [[], [20_000_000, 20_000_000],
    [20_000_000, 20_001_000], [9_900_000], [-1], [True]])
def test_reused_or_invalid_seed_families_are_rejected(seeds):
    with pytest.raises(ValueError):
        validate_seeds(seeds)


def test_collision_with_prior_background_is_rejected():
    with pytest.raises(ValueError, match="collision"):
        validate_seeds([20_000_000], prior_seeds=[seed_domains(20_000_000)["background_numeric_seeds"][-1]])


def test_fixed_curve_matches_paper_parameters():
    for hour in (0, 6, 10, 15, 21, 24):
        peaks = ((10, 1.5, .317), (15, 2.5, .633), (21, 1, .05))
        expected_per_hour = .38 * 7500 / 24 + .62 * 7500 * sum(
            w * math.exp(-.5 * ((hour - mu) / sig)**2) / (sig * math.sqrt(2 * math.pi))
            for mu, sig, w in peaks)
        assert diurnal_rate(hour * 3600, total=7500) == pytest.approx(expected_per_hour / 3600)
    assert [w[1] for w in generator_contract()["load_weights"]] == [.30, .30, .25, .10, .05]


@pytest.fixture
def small_payload():
    return create_month_payload(9_900_700, days=plan_days(9_900_700, (300, 300)))


def test_daily_continuity_curve_and_source_repeat(small_payload):
    metrics = validate_payload(small_payload, require_month=False)
    repeated = create_month_payload(9_900_700, days=plan_days(9_900_700, (300, 300)))
    assert digest(repeated) == digest(small_payload)
    assert metrics["requested_trucks"] == 600
    assert metrics["curve_exactly_preserved"]
    assert len(small_payload["initial_scenarios"]) == 21
    assert all(not s["jobs"] and not s["vessels"] for s in small_payload["initial_scenarios"].values())
    assert small_payload["days"][1]["t0"] == 86400


def test_changed_arrival_time_is_detected(small_payload):
    bad = copy.deepcopy(small_payload)
    bad["schedule"][0]["arrival_s"] += 60
    with pytest.raises(ValueError, match="Arrival curve"):
        validate_payload(bad, require_month=False)


def test_short_fixture_cannot_be_claimed_as_a_month(small_payload):
    with pytest.raises(ValueError, match="Monthly load"):
        validate_payload(small_payload)


def test_changed_curve_contract_is_detected(small_payload):
    bad = copy.deepcopy(small_payload)
    bad["generator_contract"]["night_fraction"] = .15
    with pytest.raises(ValueError, match="contract changed"):
        validate_payload(bad, require_month=False)


def test_bundle_roundtrip_and_corruption(small_payload, tmp_path):
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    expected = write_bundle(a, small_payload)
    assert write_bundle(b, small_payload) == expected
    assert load_bundle(a, expected_sha256=expected) == small_payload
    a.write_bytes(a.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(a, expected_sha256=expected)
