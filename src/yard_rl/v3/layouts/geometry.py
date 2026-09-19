"""Explicit synthetic vertical geometry; no legacy-world files are changed.

Storage bay centres are 1..N. End interfaces are bay 0 and N+1, at the
middle trolley row. The extra pitch between each end bay centre and its
interface is a synthetic assumption, not a measured terminal dimension.
External trucks use the landside road; internal YT use the separate
waterside road. These roads model route distance, not traffic congestion.
All physical parameters are required. A horizontal-layout builder is not
implemented here, so this module alone does not qualify an H/V comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import heapq
import math
from typing import Mapping

Point = tuple[float, float]
SIDES = ("LANDSIDE", "WATERSIDE")


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RoadPath:
    nodes: tuple[str, ...]
    distance_m: float


@dataclass(frozen=True)
class OrthogonalRoadNetwork:
    """An explicit road graph. Crossings connect only at shared node IDs.

    Edges must be nonzero and axis aligned. An undirected graph is the
    synthetic two-way-road assumption; a directed graph can be supplied
    explicitly. No direct Euclidean shortcut is added between endpoints.
    """

    nodes: tuple[tuple[str, Point], ...] | Mapping[str, Point]
    edges: tuple[tuple[str, str], ...]
    directed: bool = False
    _points: dict = field(init=False, repr=False, compare=False)
    _adj: dict = field(init=False, repr=False, compare=False)
    _paths: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        source = self.nodes.items() if isinstance(self.nodes, Mapping) else self.nodes
        nodes = tuple((name, tuple(point)) for name, point in source)
        if not nodes or any(not isinstance(n, str) or not n for n, _ in nodes):
            raise ValueError("Road nodes require nonempty string IDs")
        points = dict(nodes)
        if len(points) != len(nodes):
            raise ValueError("Duplicate road node ID")
        for point in points.values():
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError("Road coordinates must be finite 2D points")
        if not isinstance(self.directed, bool):
            raise ValueError("directed must be a bool")
        adj, seen = {n: [] for n in points}, set()
        edges = tuple(tuple(edge) for edge in self.edges)
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("A road edge must contain two node IDs")
            a, b = edge
            if a not in points or b not in points:
                raise ValueError(f"Unknown road endpoint: {edge}")
            key = edge if self.directed else tuple(sorted(edge))
            if key in seen:
                raise ValueError(f"Duplicate road edge: {edge}")
            seen.add(key)
            dx, dy = (abs(x-y) for x, y in zip(points[a], points[b]))
            if (dx > 0 and dy > 0) or dx+dy <= 0 or not math.isfinite(dx+dy):
                raise ValueError(f"Road edge must be finite, nonzero and orthogonal: {edge}")
            adj[a].append((b, dx+dy))
            if not self.directed:
                adj[b].append((a, dx+dy))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_adj", {n: tuple(sorted(v)) for n, v in adj.items()})
        object.__setattr__(self, "_paths", {})

    def point(self, node: str) -> Point:
        return self._points[node]

    def shortest_path(self, source: str, destination: str) -> RoadPath:
        if source not in self._points or destination not in self._points:
            raise KeyError(f"Unknown road endpoint: {source}, {destination}")
        key = source, destination
        if key in self._paths:
            return self._paths[key]
        distances, previous, queue = {source: 0.0}, {}, [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            if node == destination:
                path = [node]
                while path[-1] != source:
                    path.append(previous[path[-1]])
                answer = RoadPath(tuple(reversed(path)), distance)
                self._paths[key] = answer
                return answer
            for other, length in self._adj[node]:
                candidate = distance+length
                if candidate < distances.get(other, math.inf):
                    distances[other], previous[other] = candidate, node
                    heapq.heappush(queue, (candidate, other))
        raise ValueError(f"No road route from {source} to {destination}")

    def distance_m(self, source: str, destination: str) -> float:
        return self.shortest_path(source, destination).distance_m

    def as_dict(self) -> dict:
        return dict(nodes={n: list(p) for n, p in self.nodes},
                    edges=[list(e) for e in self.edges], directed=self.directed)


@dataclass(frozen=True, kw_only=True)
class VerticalLayoutSpec:
    n_blocks: int
    bay_count: int
    row_count: int
    tier_max: int
    bay_length_m: float
    row_width_m: float
    block_gap_m: float
    land_road_clearance_m: float
    water_road_clearance_m: float
    gate_approach_m: float
    quay_approach_m: float
    speed_mps: float

    def __post_init__(self) -> None:
        for name in ("n_blocks", "bay_count", "row_count", "tier_max"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("bay_length_m", "row_width_m", "block_gap_m",
                     "land_road_clearance_m", "water_road_clearance_m",
                     "gate_approach_m", "quay_approach_m", "speed_mps"):
            _positive(name, getattr(self, name))
        for name, value in (("block_length_m", self.block_length_m),
                            ("block_width_m", self.block_width_m),
                            ("terminal_span_m", (self.n_blocks-1)*self.pitch_m
                             + self.gate_approach_m+self.quay_approach_m)):
            _positive(name, value)

    @property
    def block_length_m(self) -> float:
        """Distance between end interfaces, including the synthetic end gaps."""
        return (self.bay_count+1)*self.bay_length_m

    @property
    def block_width_m(self) -> float:
        return self.row_count*self.row_width_m

    @property
    def pitch_m(self) -> float:
        return self.block_width_m+self.block_gap_m

    @property
    def slots_per_block(self) -> int:
        return self.bay_count*self.row_count*self.tier_max


@dataclass(frozen=True)
class EndInterface:
    block_id: str
    side: str
    xy_m: Point
    bay: float
    row: float
    road_node: str


@dataclass(frozen=True)
class VerticalYardLayout:
    spec: VerticalLayoutSpec
    ids: tuple[str, ...]
    centres_x_m: tuple[float, ...]
    land_node_ids: tuple[str, ...]
    water_node_ids: tuple[str, ...]
    landside_road: OrthogonalRoadNetwork
    waterside_road: OrthogonalRoadNetwork
    _indices: dict = field(init=False, repr=False, compare=False)
    _gate_times: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _quay_times: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _block_times: tuple[tuple[float, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (not self.ids or len(set(self.ids)) != len(self.ids)
                or any(not isinstance(b, str) or not b for b in self.ids)):
            raise ValueError("Block IDs must be nonempty and unique")
        object.__setattr__(self, "_indices", {bid: i for i, bid in enumerate(self.ids)})
        if any(len(v) != self.n for v in (self.centres_x_m, self.land_node_ids,
                                          self.water_node_ids)):
            raise ValueError("Block coordinates and interfaces must match block IDs")
        if self.n > self.spec.n_blocks:
            raise ValueError("More represented blocks than the declared full layout")
        if self.landside_road.directed or self.waterside_road.directed:
            raise ValueError("Vertical layout currently requires explicit two-way roads")
        ordered = sorted(self.centres_x_m)
        if any(b-a < self.spec.block_width_m for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Block footprints overlap")
        for bid, x in zip(self.ids, self.centres_x_m):
            if not math.isfinite(x):
                raise ValueError("Block centre must be finite")
            for side, expected in (("LANDSIDE", (x, 0.0)),
                                   ("WATERSIDE", (x, self.spec.block_length_m))):
                interface = self.interface(bid, side)
                road = self.landside_road if side == "LANDSIDE" else self.waterside_road
                if interface.xy_m != expected or road.point(interface.road_node) != expected:
                    raise ValueError("Road endpoint does not coincide with block end")
        # Policy decisions only index these immutable tables. Graph route search
        # and unit conversion happen once during construction; replace/subset
        # rebuild tables for their own block order, without changing the roads.
        object.__setattr__(self, "_gate_times", tuple(
            self.landside_road.distance_m("GATE", node)/self.speed_mps
            for node in self.land_node_ids))
        object.__setattr__(self, "_quay_times", tuple(
            self.waterside_road.distance_m("QUAY", node)/self.speed_mps
            for node in self.water_node_ids))
        object.__setattr__(self, "_block_times", tuple(tuple(
            self.landside_road.distance_m(a, b)/self.speed_mps
            for b in self.land_node_ids) for a in self.land_node_ids))

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def speed_mps(self) -> float:
        return self.spec.speed_mps

    def _index(self, bid: str) -> int:
        try:
            return self._indices[bid]
        except KeyError as exc:
            raise KeyError(f"Unknown block: {bid}") from exc

    def local_interface(self, side: str) -> tuple[float, float]:
        if side not in SIDES:
            raise ValueError(f"Unknown interface side: {side}")
        return (0.0 if side == "LANDSIDE" else float(self.spec.bay_count+1),
                (self.spec.row_count+1)/2.0)

    def local_to_xy_m(self, bid: str, bay: float, row: float) -> Point:
        if (not math.isfinite(bay) or not math.isfinite(row)
                or not 0 <= bay <= self.spec.bay_count+1
                or not 0.5 <= row <= self.spec.row_count+0.5):
            raise ValueError("Coordinate outside block/interface geometry")
        centre = self.centres_x_m[self._index(bid)]
        return (centre+(row-(self.spec.row_count+1)/2)*self.spec.row_width_m,
                bay*self.spec.bay_length_m)

    def interface(self, bid: str, side: str) -> EndInterface:
        i = self._index(bid)
        bay, row = self.local_interface(side)
        node = self.land_node_ids[i] if side == "LANDSIDE" else self.water_node_ids[i]
        return EndInterface(bid, side, self.local_to_xy_m(bid, bay, row), bay, row, node)

    def gate_to_block_s(self, bid: str) -> float:
        return self._gate_times[self._index(bid)]

    def quay_to_block_s(self, bid: str) -> float:
        return self._quay_times[self._index(bid)]

    def block_to_block_s(self, a: str, b: str) -> float:
        return self._block_times[self._index(a)][self._index(b)]

    def yt_round_trip_s(self, bid: str) -> float:
        return 2*self.quay_to_block_s(bid)

    def pre_gate_route_delta_s(self, src: str, dst: str) -> float:
        return self.gate_to_block_s(dst)-self.gate_to_block_s(src)

    def post_gate_route_s(self, src: str, dst: str) -> float:
        return self.block_to_block_s(src, dst)

    def gate_time_range_s(self) -> tuple[float, float]:
        times = [self.gate_to_block_s(b) for b in self.ids]
        return min(times), max(times)

    def mean_gate_time_s(self) -> float:
        return math.fsum(self.gate_to_block_s(b) for b in self.ids)/self.n

    def subset(self, ids) -> VerticalYardLayout:
        chosen = tuple(ids)
        indices = [self._index(b) for b in chosen]
        return replace(self, ids=chosen,
                       centres_x_m=tuple(self.centres_x_m[i] for i in indices),
                       land_node_ids=tuple(self.land_node_ids[i] for i in indices),
                       water_node_ids=tuple(self.water_node_ids[i] for i in indices))

    def renamed(self, mapping: dict[str, str]) -> VerticalYardLayout:
        return replace(self, ids=tuple(mapping.get(b, b) for b in self.ids))

    def matrix_s(self) -> dict[str, dict[str, float]]:
        return {a: {b: self.block_to_block_s(a, b) for b in self.ids} for a in self.ids}

    def as_dict(self) -> dict:
        return dict(schema="yard_rl.v3.vertical-layout.v1", orientation="PERPENDICULAR_TO_QUAY",
                    synthetic=True, ids=list(self.ids), parameters=asdict(self.spec),
                    block_length_m=self.spec.block_length_m, block_width_m=self.spec.block_width_m,
                    slots_per_block=self.spec.slots_per_block,
                    interfaces={b: {side: asdict(self.interface(b, side)) for side in SIDES}
                                for b in self.ids},
                    landside_road=self.landside_road.as_dict(),
                    waterside_road=self.waterside_road.as_dict(),
                    gate_to_block_s={b: self.gate_to_block_s(b) for b in self.ids},
                    quay_to_block_s={b: self.quay_to_block_s(b) for b in self.ids},
                    block_to_block_s=self.matrix_s(),
                    assumptions=["Synthetic geometry, not a measured terminal",
                                 "Gate at left end of landside road; quay at right end of waterside road",
                                 "Two-way orthogonal roads with separate landside/waterside access",
                                 "End interfaces at bay 0 and N+1; stock bay centres 1..N",
                                 "Same explicit speed for external trucks and internal YT",
                                 "No road congestion, role enforcement, handover capacity or H/V runtime qualification"])


def build_vertical_layout(spec: VerticalLayoutSpec, ids=None) -> VerticalYardLayout:
    """Build two separated comb roads, connected to the respective block ends."""
    names = tuple(ids) if ids is not None else tuple(f"Y{i+1:02d}" for i in range(spec.n_blocks))
    if len(names) != spec.n_blocks:
        raise ValueError("Block ID count must equal spec.n_blocks; use subset after building")
    centres = tuple(i*spec.pitch_m for i in range(spec.n_blocks))
    roads, endpoints = [], []
    for side in SIDES:
        land = side == "LANDSIDE"
        end_y = 0.0 if land else spec.block_length_m
        road_y = -spec.land_road_clearance_m if land else end_y+spec.water_road_clearance_m
        anchor = "GATE" if land else "QUAY"
        anchor_x = -spec.gate_approach_m if land else centres[-1]+spec.quay_approach_m
        nodes, edges, ends = [(anchor, (anchor_x, road_y))], [], []
        for i, x in enumerate(centres):
            junction, endpoint = f"{side}:J{i:02d}", f"{side}:E{i:02d}"
            nodes.extend(((junction, (x, road_y)), (endpoint, (x, end_y))))
            edges.append((junction, endpoint))
            if i:
                edges.append((f"{side}:J{i-1:02d}", junction))
            ends.append(endpoint)
        edges.append((anchor, f"{side}:J{0 if land else spec.n_blocks-1:02d}"))
        roads.append(OrthogonalRoadNetwork(tuple(nodes), tuple(edges)))
        endpoints.append(tuple(ends))
    return VerticalYardLayout(spec, names, centres, endpoints[0], endpoints[1], roads[0], roads[1])
