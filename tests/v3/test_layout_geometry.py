"""Physical route/end-interface checks for the new synthetic vertical geometry."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from yard_rl.v3.layouts.geometry import (OrthogonalRoadNetwork, VerticalLayoutSpec,
                                        build_vertical_layout)


def spec(**overrides):
    """Small arithmetic fixture only; these are not experiment parameters."""
    values = dict(n_blocks=3, bay_count=4, row_count=2, tier_max=2,
                  bay_length_m=10., row_width_m=3., block_gap_m=4.,
                  land_road_clearance_m=6., water_road_clearance_m=8.,
                  gate_approach_m=20., quay_approach_m=30., speed_mps=2.)
    return VerticalLayoutSpec(**(values | overrides))


def test_routes_follow_separate_orthogonal_roads():
    layout = build_vertical_layout(spec())
    assert [layout.gate_to_block_s(b) for b in layout.ids] == [13., 18., 23.]
    assert [layout.quay_to_block_s(b) for b in layout.ids] == [29., 24., 19.]
    # Returning to the landside spine costs two clearance legs, not just dx.
    assert layout.block_to_block_s("Y01", "Y03") == 16.
    assert layout.block_to_block_s("Y03", "Y01") == 16.
    assert layout.block_to_block_s("Y02", "Y02") == 0.
    path = layout.landside_road.shortest_path("LANDSIDE:E00", "LANDSIDE:E02")
    assert path.nodes == ("LANDSIDE:E00", "LANDSIDE:J00", "LANDSIDE:J01",
                          "LANDSIDE:J02", "LANDSIDE:E02")
    assert path.distance_m == 32.
    with pytest.raises(KeyError):
        layout.landside_road.distance_m("GATE", "WATERSIDE:E00")


def test_interfaces_match_engine_bay_zero_and_n_plus_one_contract():
    layout = build_vertical_layout(spec())
    land, water = (layout.interface("Y02", side) for side in ("LANDSIDE", "WATERSIDE"))
    assert (land.bay, land.row, land.xy_m) == (0., 1.5, (10., 0.))
    assert (water.bay, water.row, water.xy_m) == (5., 1.5, (10., 50.))
    assert layout.local_to_xy_m("Y02", 1, 1) == (8.5, 10.)
    assert layout.local_to_xy_m("Y02", 4, 2) == (11.5, 40.)
    assert water.xy_m[1]-land.xy_m[1] == (spec().bay_count+1)*spec().bay_length_m
    assert layout.spec.slots_per_block == 16


def test_existing_21_block_capacity_without_changing_profile_values():
    layout = build_vertical_layout(spec(n_blocks=21, bay_count=24, row_count=10,
                                        tier_max=6, bay_length_m=6.5, row_width_m=3.1))
    assert layout.n == 21 and layout.spec.slots_per_block == 1440
    assert layout.local_interface("WATERSIDE") == (25., 5.5)
    assert layout.spec.block_length_m == 162.5


def test_geometry_parameters_change_real_route_distance_not_only_labels():
    original = build_vertical_layout(spec())
    longer = build_vertical_layout(spec(land_road_clearance_m=16.))
    assert longer.gate_to_block_s("Y02")-original.gate_to_block_s("Y02") == 5.
    assert longer.block_to_block_s("Y01", "Y02")-original.block_to_block_s("Y01", "Y02") == 10.
    assert longer.quay_to_block_s("Y02") == original.quay_to_block_s("Y02")
    slower = build_vertical_layout(spec(speed_mps=1.))
    assert slower.yt_round_trip_s("Y03") == 2*original.yt_round_trip_s("Y03")


def test_subset_and_rename_preserve_full_geometry_and_route_contract():
    original = build_vertical_layout(spec())
    subset = original.subset(["Y03", "Y01"]).renamed({"Y03": "A", "Y01": "B"})
    assert subset.ids == ("A", "B")
    assert subset.interface("A", "LANDSIDE").xy_m == (20., 0.)
    assert subset.quay_to_block_s("A") == original.quay_to_block_s("Y03")
    assert subset.gate_to_block_s("B") == original.gate_to_block_s("Y01")
    assert subset.post_gate_route_s("A", "B") == 16.
    assert subset.pre_gate_route_delta_s("A", "B") == -10.
    assert subset.gate_time_range_s() == (13., 23.)
    assert subset.mean_gate_time_s() == 18.
    assert subset.matrix_s() == {"A": {"A": 0., "B": 16.}, "B": {"A": 16., "B": 0.}}


def test_graph_shortest_path_respects_edges_and_direction():
    graph = OrthogonalRoadNetwork(dict(A=(0., 0.), B=(0., 3.), C=(4., 3.), D=(4., 0.)),
                                  (("A", "B"), ("B", "C"), ("C", "D")), directed=True)
    assert graph.distance_m("A", "D") == 10.  # Euclidean distance would be 4.
    with pytest.raises(ValueError, match="No road route"):
        graph.distance_m("D", "A")
    assert graph.distance_m("A", "A") == 0.


@pytest.mark.parametrize("kwargs", [dict(speed_mps=0), dict(speed_mps=math.nan),
    dict(block_gap_m=math.inf), dict(land_road_clearance_m=-1), dict(n_blocks=0),
    dict(bay_count=True), dict(row_count=2.5), dict(gate_approach_m=0),
    dict(bay_length_m=1e308)])
def test_invalid_parameters_fail_before_building(kwargs):
    with pytest.raises(ValueError):
        spec(**kwargs)


@pytest.mark.parametrize("nodes,edges", [
    ((("A", (0., 0.)), ("A", (1., 0.))), ()),
    ((("A", (0., 0.)), ("B", (1., 1.))), (("A", "B"),)),
    ((("A", (0., 0.)), ("B", (0., 0.))), (("A", "B"),)),
    ((("A", (math.inf, 0.)),), ()),
    ((("A", (0., 0.)),), (("A", "missing"),)),
    ((("A", (0., 0.)), ("B", (1., 0.))), (("A", "B"), ("B", "A"))),
])
def test_invalid_road_geometry_is_rejected(nodes, edges):
    with pytest.raises(ValueError):
        OrthogonalRoadNetwork(nodes, edges)


def test_invalid_blocks_interfaces_and_disconnected_inputs_are_rejected():
    layout = build_vertical_layout(spec())
    for operation in (lambda: layout.subset([]), lambda: layout.subset(["Y01", "Y01"]),
                      lambda: layout.renamed({"Y01": "Y02"}),
                      lambda: layout.local_interface("SHARED"),
                      lambda: layout.local_to_xy_m("Y01", 6, 1),
                      lambda: build_vertical_layout(spec(), ["one"]),
                      lambda: replace(layout, centres_x_m=(1., 10., 20.))):
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(KeyError):
        layout.gate_to_block_s("missing")
    disconnected = OrthogonalRoadNetwork(layout.landside_road.nodes, ())
    with pytest.raises(ValueError, match="No road route"):
        replace(layout, landside_road=disconnected)
    with pytest.raises(ValueError, match="two-way roads"):
        replace(layout, landside_road=replace(layout.landside_road, directed=True))
    with pytest.raises(ValueError, match="footprints overlap"):
        replace(layout, centres_x_m=(0., 1., 20.))


def test_serializable_auditable_geometry_and_snapshot_copy():
    layout = build_vertical_layout(spec())
    payload = json.loads(json.dumps(layout.as_dict(), allow_nan=False))
    assert payload["synthetic"] is True
    assert payload["orientation"] == "PERPENDICULAR_TO_QUAY"
    assert payload["interfaces"]["Y01"]["WATERSIDE"]["bay"] == 5.
    assert payload["parameters"]["speed_mps"] == 2.
    assert payload["landside_road"]["nodes"]["GATE"] == [-20., -6.]
    assert deepcopy(layout).as_dict() == layout.as_dict()


def test_policy_time_queries_do_not_search_roads_again(monkeypatch):
    layout = build_vertical_layout(spec())
    smaller = layout.subset(["Y03", "Y01"]).renamed({"Y03": "A", "Y01": "B"})
    snapshots = [layout, deepcopy(layout), smaller, deepcopy(smaller)]
    expected = [item.as_dict() for item in snapshots]
    road_geometry = (layout.landside_road.as_dict(), layout.waterside_road.as_dict())

    def unexpected_search(*args, **kwargs):
        raise AssertionError("A repeated policy time query traversed the road graph")

    monkeypatch.setattr(OrthogonalRoadNetwork, "distance_m", unexpected_search)
    monkeypatch.setattr(OrthogonalRoadNetwork, "shortest_path", unexpected_search)
    for item, payload in zip(snapshots, expected):
        for _ in range(10):
            for bid in item.ids:
                assert item.gate_to_block_s(bid) == payload["gate_to_block_s"][bid]
                assert item.quay_to_block_s(bid) == payload["quay_to_block_s"][bid]
                assert item.yt_round_trip_s(bid) == 2*payload["quay_to_block_s"][bid]
            assert item.matrix_s() == payload["block_to_block_s"]
        assert item.as_dict() == payload
    assert (layout.landside_road.as_dict(), layout.waterside_road.as_dict()) == road_geometry
