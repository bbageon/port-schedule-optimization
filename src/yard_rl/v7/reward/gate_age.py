"""Truck points from gate entry to job completion, in minutes."""
from bisect import bisect_right
from collections import defaultdict
from math import isfinite

from .joint_turnaround import JointTurnaroundClock


# Calibrated on BP's first training replay; fixed across all comparisons.
MAX_SURCHARGE = 0.12251520108471743
GRACE_MIN = 60.0
RAMP_MIN = 30.0


def truck_points(minutes):
    if not isfinite(minutes) or minutes < 0:
        raise ValueError("Invalid gate turnaround")
    excess = max(0.0, minutes - GRACE_MIN)
    ramp = min(excess, RAMP_MIN)
    return minutes + MAX_SURCHARGE * (ramp**2 / (2 * RAMP_MIN) + excess - ramp)


class GateAgeTruckClock:
    """Sum of accrued truck points, queried repeatedly at simulation times in seconds."""

    def __init__(self, intervals):
        intervals = list(intervals)
        self.base = JointTurnaroundClock(intervals, ())
        # Surcharge rate grows with slope 1 during each truck's ramp (scale applied at query).
        # At completion remove exactly that truck's rate. Active trucks can cross midnight.
        events = defaultdict(lambda: [0, 0.0, 0])
        for gate, finish in intervals:
            begin = gate + 60 * GRACE_MIN
            if finish <= begin:
                continue
            end = min(finish, begin + 60 * RAMP_MIN)
            events[begin][0] += 1
            events[end][0] -= 1
            events[finish][1] -= (end - begin) / 60
            events[begin][2] += 1
            events[finish][2] -= 1

        self.times = sorted(events)
        self.segments = []
        previous = area = rate = 0.0
        slope = active = 0
        for t in self.times:
            delta = (t - previous) / 60
            area += delta * (rate + 0.5 * slope * delta)
            change, removed, count = events[t]
            rate += slope * delta + removed
            slope += change
            active += count
            if active == 0:
                rate = 0.0
            self.segments.append((area, rate, slope))
            previous = t
        self.total_minutes = self.base.total_minutes + MAX_SURCHARGE * area / RAMP_MIN

    def minutes_at(self, t):
        base = self.base.minutes_at(t)
        i = bisect_right(self.times, t) - 1
        if i < 0:
            return base
        area, rate, slope = self.segments[i]
        delta = (t - self.times[i]) / 60
        return base + MAX_SURCHARGE / RAMP_MIN * (area + delta * (rate + 0.5 * slope * delta))


class StopGateAgeClock:
    """Gate-aged points only while a visit belongs to this stop's block."""

    def __init__(self, intervals):
        intervals = list(intervals)
        self.through_finish = GateAgeTruckClock((gate, finish) for gate, _, finish in intervals)
        self.before_stop = GateAgeTruckClock((gate, start) for gate, start, _ in intervals if start > gate)
        self.total_minutes = self.through_finish.total_minutes - self.before_stop.total_minutes

    def minutes_at(self, t):
        return self.through_finish.minutes_at(t) - self.before_stop.minutes_at(t)
