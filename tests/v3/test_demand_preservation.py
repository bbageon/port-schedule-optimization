"""Adversarial request-admission cases, without reducing the synthetic workload."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from yard_rl.v3.stage.demand_engine import DemandTerminal
from yard_rl.v3.stage.month import make_retarget
from yard_rl.v3.stage.month_engine import inject_vessel
from yard_rl.v3.stage.orders import V3Announcer
from yard_rl.v3.world.domain.enums import ContainerSize, JobFlow, JobStatus, LoadStatus
from yard_rl.v3.world.domain.models import Container, Job
from yard_rl.v3.world.integrated.multiblock import TransferError
from yard_rl.v3.world.integrated.profiles import build_h21_profile
from yard_rl.v3.world.integrated.time_contract import TimeLedger


class Queue:
    def __init__(self):
        self.events = []

    def push(self, *args):
        self.events.append(args)


class Ledger:
    def __init__(self):
        self.records = {}

    def register(self, item):
        self.records[item.job_id] = item


def terminal():
    m = DemandTerminal.__new__(DemandTerminal)
    box = Container("C1", ContainerSize.FT40, LoadStatus.FULL, "Y01", 1, 1, 1)
    sim = NS(clock=0, end=3600, scenario=NS(horizon_s=3600, drain_window_s=0),
        stacks=NS(containers={"C1": box}), jobs={}, vessels={}, queue=Queue(),
        time_ledger=TimeLedger(sla_s=3600), profile=build_h21_profile(), _refresh_rates=lambda: None)
    m.blocks, m.ledger = {"Y01": sim}, Ledger()
    m._reserved_inbound, m.capacity_margin = {"Y01": 10000}, 2
    return m, sim


def request(key="truck", flow="GATE_OUT", arrival=120):
    return dict(job_id=key, flow=flow, block="Y01", target="C1", arrival_s=arrival,
        lead_s=arrival, travel_s=30, travel_base_s=30, exit_travel_s=30, size_ft40=True)


def test_all_stock_claimed_does_not_drop_or_reschedule_truck():
    m, s = terminal()
    s.jobs["vessel"] = Job("vessel", JobFlow.VESSEL_LOAD, 0, None, None, target_container="C1")
    e = request()
    legacy = V3Announcer([e], retarget=make_retarget(1))
    legacy.review(m, 0)
    assert legacy.n_skipped == 1 and not m.ledger.records
    ann = V3Announcer([e], retarget=make_retarget(1), preserve_requests=True, record_admissions=True)
    ann.review(m, 0)
    j = s.jobs["truck"]
    assert ann.n_admitted == 1 and ann.n_skipped == 0 and j.target_container is None
    assert (j.actual_gate_in, j.actual_block_arrival) == (120, 150)
    assert s.time_ledger.records["truck"].gate_in == 120
    assert e == request()  # Original common demand remains immutable.
    s.stacks.containers["NEW"] = deepcopy(s.stacks.containers["C1"])
    m.bind_available_targets(120)
    assert j.target_container is None  # Work release is the block arrival, not notice.
    m.bind_available_targets(180)
    assert j.target_container == "NEW"
    assert s.jobs["vessel"].target_container == "C1"


def test_reservation_shortage_keeps_original_inbound_arrival():
    m, s = terminal()
    ann = V3Announcer([request(flow="GATE_IN")], preserve_requests=True)
    assert m.free_slots("Y01") < 0
    ann.review(m, 0)
    assert ann.n_admitted == 1 and ann.n_skipped == 0
    assert len(s.stacks.containers) == 1  # Registration did not place a physical box.
    assert s.jobs["truck"].actual_block_arrival == 150


def test_full_vessel_manifest_survives_inventory_shortage():
    m, s = terminal()
    row = dict(work="LOAD", start_s=600, cadence_s=60, moves=4)
    result = inject_vessel(m, "Y01", row, key="V1", size_seed="unit", defer_load_targets=True)
    assert result.asked_moves == result.moves == s.vessels["V1"].plan.total_moves == 4
    assert all(j.target_container is None for j in s.jobs.values())
    m.bind_available_targets(599)
    assert not any(j.target_container for j in s.jobs.values())
    m.bind_available_targets(600)
    assert sum(j.target_container is not None for j in s.jobs.values()) == 1
    assert len(s.jobs) == 4


def test_branch_copy_keeps_contract_and_has_independent_target_state():
    m, s = terminal()
    ann = V3Announcer([], preserve_requests=True)
    assert ann.clone_fresh().preserve_requests and ann.window(0, 60).preserve_requests
    inject_vessel(m, "Y01", dict(work="LOAD", start_s=600, cadence_s=60, moves=2),
        key="V1", size_seed="unit", defer_load_targets=True)
    branch = deepcopy(m)
    branch.bind_available_targets(600)
    assert all(j.target_container is None for j in s.jobs.values())
    assert sum(j.target_container is not None for j in branch.blocks["Y01"].jobs.values()) == 1


def test_future_announced_request_survives_short_counterfactual_cutoff():
    m, s = terminal()
    s.end = 60  # Snapshot is truncated, declared input horizon remains 3600.
    ann = V3Announcer([request(flow="GATE_IN")], preserve_requests=True)
    ann.review(m, 0)
    assert ann.n_admitted == 1 and s.time_ledger.records["truck"].gate_in == 120


def test_invalid_arrival_fails_instead_of_disappearing():
    m, s = terminal()
    ann = V3Announcer([request(arrival=4000)], preserve_requests=True, end_s=3600)
    with pytest.raises(TransferError, match="outside declared horizon"):
        ann.review(m, 0)
    assert not s.jobs and not m.ledger.records and ann.n_skipped == 0


def test_two_pending_jobs_cannot_claim_same_container():
    m, s = terminal()
    for key in ("A", "B"):
        s.jobs[key] = Job(key, JobFlow.GATE_OUT, 0, 0, 0, status=JobStatus.WAITING)
    m.bind_available_targets(0)
    assert s.jobs["A"].target_container == "C1" and s.jobs["B"].target_container is None
    m.bind_available_targets(60)
    assert s.jobs["B"].target_container is None
