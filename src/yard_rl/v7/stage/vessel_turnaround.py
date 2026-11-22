"""One turnaround clock per port call, including concurrently working quay cranes."""
from collections import defaultdict

from ..reward.krw import vessel_krw_per_hour


class VesselTurnaround:
    def __init__(self, built, terminal):
        self.calls = defaultdict(list)
        self.manifest = {v.vessel_id: v for v in built["fleet"]}
        for row in built["vessel_schedule"]:
            processes = list(terminal.blocks[row["block"]].vessels.values())
            if len(processes) != 1 or processes[0].plan.total_moves != row["moves"]:
                raise ValueError(f"Unmatched quay work manifest: {row}")
            self.calls[row["vessel_id"]].append((row, processes[0]))
        if self.calls.keys() != self.manifest.keys():
            raise ValueError("Missing port call in quay schedule")
        for vid, streams in self.calls.items():
            ship = self.manifest[vid]
            if (len(streams) != ship.cls.sts
                    or sum(row["moves"] for row, _ in streams) != ship.moves):
                raise ValueError(f"Incomplete port call manifest: {vid}")

    def rows(self, now):
        result = []
        for vid, streams in self.calls.items():
            starts = [r["start_s"] for r, _ in streams]
            if len(set(starts)) != 1:
                raise ValueError(f"Inconsistent service-ready times: {vid}")
            ends = [v.truth.actual_completion_s for _, v in streams]
            end = max(ends) if all(t is not None and t <= now for t in ends) else None
            hours = max(0., (now if end is None else end) - starts[0]) / 3600.
            ship = self.manifest[vid]
            result.append(dict(vessel_id=vid, gt=ship.gt, start_s=starts[0], finish_s=end,
                               hours=hours, cost=vessel_krw_per_hour(ship.gt)*hours,
                               moves=ship.moves, stream_finishes=ends,
                               remaining=sum((r["moves"] if not v.started else v.remaining_moves)
                                             for r, v in streams)))
        return result
