from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def _load(name: str) -> dict:
    path = ROOT / "configs" / "terminals" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_dgt_profile_preserves_real_landside_waterside_split():
    p = _load("dgt_public_topology.yaml")
    assert p["yard"]["orientation"] == "PERPENDICULAR_TO_QUAY"
    assert p["yard"]["crane_count"] == 46
    assert p["yard"]["cranes_per_block"] == 2
    assert p["yard"]["block_count"] == 23
    points = {x["type"]: x for x in p["transfer_points"]}
    assert points["LSTP"]["allowed_vehicle_types"] == ["EXTERNAL_TRUCK"]
    assert points["LSTP"]["approach_maneuver"] == "REVERSE_IN"
    assert points["WSTP"]["allowed_vehicle_types"] == ["AGV"]
    assert p["road_network"]["hard_vehicle_separation"] is True
    assert p["road_network"]["main_road_direction_confirmed"] is False


def test_hjnc_profile_does_not_invent_lane_count_or_direction_edges():
    p = _load("hjnc_public_topology.yaml")
    assert p["yard"]["orientation"] == "PARALLEL_TO_QUAY"
    assert p["yard"]["block_count"] == 21
    assert p["yard"]["crane_count"] == 42
    assert p["yard"]["cranes_per_block"] == 2
    assert p["yard"]["block_length_teu"] == 52
    assert p["road_network"]["directed_circulation_depicted"] is True
    assert p["road_network"]["directed_edges"] is None
    assert all(x["service_capacity"] is None
               for x in p["block_template"]["service_sides"])


def test_public_profiles_use_operational_ids_not_fixture_l1_l2():
    dgt = _load("dgt_public_topology.yaml")
    hjnc = _load("hjnc_public_topology.yaml")
    ids = [x["id_template"] for x in dgt["transfer_points"]]
    ids += [x["id_template"] for x in hjnc["block_template"]["service_sides"]]
    assert all(x not in {"L1", "L2"} for x in ids)
