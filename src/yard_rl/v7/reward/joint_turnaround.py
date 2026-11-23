"""Elapsed truck and port-call minutes, valued at the registered fixed ratio."""
from bisect import bisect_right
from collections import defaultdict
from math import isfinite


VESSEL_TIME_WEIGHT = 67


class JointTurnaroundClock:
    """Reconstruct past exposure at a decision time from a completed physical ledger."""

    def __init__(self, truck_intervals, vessel_intervals):
        events = defaultdict(int)
        for intervals, weight in ((truck_intervals, 1), (vessel_intervals, VESSEL_TIME_WEIGHT)):
            for start, end in intervals:
                if end is None or not (isfinite(start) and isfinite(end) and 0 <= start <= end):
                    raise ValueError("Incomplete or invalid turnaround interval")
                events[start] += weight
                events[end] -= weight
        self.times = sorted(events)
        self.accrued_seconds = []
        self.active_weights = []
        previous, active, seconds = 0.0, 0, 0.0
        for t in self.times:
            seconds += (t - previous) * active
            active += events[t]
            self.accrued_seconds.append(seconds)
            self.active_weights.append(active)
            previous = t
        self.total_minutes = seconds / 60

    def minutes_at(self, t):
        if not isfinite(t) or t < 0:
            raise ValueError("Invalid decision time")
        i = bisect_right(self.times, t) - 1
        if i < 0:
            return 0.0
        seconds = self.accrued_seconds[i] + self.active_weights[i] * (t - self.times[i])
        return seconds / 60
