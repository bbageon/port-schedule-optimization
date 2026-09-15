from types import SimpleNamespace as NS

from yard_rl.v3.stage.admission_audit import VesselWorkAudit, inventory_snapshot
from yard_rl.v3.stage.month import make_retarget
from yard_rl.v3.stage.orders import V3Announcer
from yard_rl.v3.world.domain.enums import JobFlow, JobStatus
from yard_rl.v3.world.integrated.multiblock import MultiBlockTerminal


def terminal(jobs=None, vessels=None):
    sim = NS(clock=0, stacks=NS(containers={"C1": object()}),
        profile=NS(block=NS(bay_count=4, row_count=1, tier_max=1)),
        jobs=jobs or {}, vessels=vessels or {})
    mbt = NS(blocks={"Y01": sim}, _reserved_inbound={"Y01": 0}, capacity_margin=0)
    mbt.free_slots = lambda bid: MultiBlockTerminal.free_slots(mbt, bid)
    return mbt, sim


def job(key, flow=JobFlow.VESSEL_LOAD, status=JobStatus.PLANNED, target=None, vessel="V1"):
    return NS(job_id=key, flow=flow, status=status, target_container=target, vessel_id=vessel)


def audit(asked=2, admitted=2, ok=True):
    a = VesselWorkAudit([dict(key="V1", block="Y01", work="LOAD", start_s=0, moves=asked)])
    a.admission(dict(key="V1", asked=asked, moves=admitted, ok=ok, why=""))
    return a


def vessel(done=False):
    return NS(plan=NS(total_moves=2), started=True, remaining_moves=0 if done else 2,
              done=done, truth=NS(actual_completion_s=60 if done else None))


def test_no_target_can_mean_claimed_inventory_instead_of_an_empty_block():
    mbt, sim = terminal({"ship_job": job("ship_job", target="C1")})
    entry = dict(job_id="truck", flow="GATE_OUT", block="Y01", target="C1",
                 arrival_s=120, lead_s=120, travel_s=30)
    outcomes = []
    for diagnose in (False, True):
        ann = V3Announcer([entry], retarget=make_retarget(1), record_admissions=True,
                          diagnose_admissions=diagnose)
        ann.review(mbt, 0)
        outcomes.append((ann.n_admitted, ann.n_skipped, ann.skips))
    assert outcomes[0] == outcomes[1]
    evidence = ann.admission_events[0]["inventory_at_failure"]
    assert evidence["inventory_boxes"] == evidence["claimed_inventory_boxes"] == 1
    assert evidence["unclaimed_inventory_boxes"] == 0
    assert ann.admission_events[0]["reason"] == "NO_TARGET"
    assert ann.clone_fresh().diagnose_admissions and ann.window(0, 60).diagnose_admissions
    assert list(sim.stacks.containers) == ["C1"] and list(sim.jobs) == ["ship_job"]


def test_negative_admission_free_is_distinguished_from_physical_overflow():
    mbt, sim = terminal({str(i): job(str(i), flow=JobFlow.VESSEL_DISCHARGE)
                         for i in range(4)})
    snapshot = inventory_snapshot(mbt, "Y01")
    assert snapshot["physical_free"] == 3
    assert snapshot["admission_free"] == -1
    assert snapshot["free_formula_matches"] is True
    assert len(sim.stacks.containers) == 1 and len(sim.jobs) == 4


def test_invalid_block_does_not_turn_an_admission_failure_into_a_diagnostic_exception():
    mbt, _ = terminal()
    assert inventory_snapshot(mbt, "absent")["available"] is False


def test_completed_jobs_survive_pruning_and_observation_does_not_double_count():
    a = audit()
    mbt, sim = terminal({"a": job("a", status=JobStatus.DONE), "b": job("b")},
                         {"V1": vessel()})
    a.observe(mbt)
    a.observe(mbt)
    del sim.jobs["a"]  # Daily pruning must not erase the completion evidence.
    rows, summary = a.finish(mbt)
    assert rows[0]["completed_yard_jobs"] == rows[0]["outstanding_yard_jobs"] == 1
    assert summary["recording_ok"] and not summary["all_requested_work_completed"]
    sim.jobs["b"].status = JobStatus.DONE
    sim.vessels["V1"] = vessel(done=True)
    a.observe(mbt)
    sim.jobs.clear()
    sim.vessels.clear()  # Retiring the vessel also preserves its last observed completion.
    rows, summary = a.finish(mbt)
    assert rows[0]["completed_yard_jobs"] == 2
    assert summary["recording_ok"] and summary["all_requested_work_completed"]


def test_sts_done_does_not_hide_unfinished_yard_jobs():
    a = audit()
    mbt, _ = terminal({"a": job("a", status=JobStatus.DONE), "b": job("b")},
                       {"V1": vessel(done=True)})
    _, summary = a.finish(mbt)
    assert summary["recording_ok"]
    assert summary["outstanding_yard_jobs"] == 1
    assert not summary["all_requested_work_completed"]


def test_a_missing_admitted_job_cannot_be_reported_as_complete():
    a = audit()
    mbt, _ = terminal({"a": job("a", status=JobStatus.DONE)}, {"V1": vessel(done=True)})
    _, summary = a.finish(mbt)
    assert summary["unaccounted_yard_jobs"] == 1
    assert not summary["recording_ok"] and not summary["all_requested_work_completed"]


def test_a_fully_rejected_vessel_keeps_its_original_requested_work():
    a = audit(asked=200, admitted=0, ok=False)
    mbt, _ = terminal()
    rows, summary = a.finish(mbt)
    assert rows[0]["attempted"] and summary["recording_ok"]
    assert summary["requested_moves"] == summary["unadmitted_moves"] == 200
    assert summary["admitted_moves"] == 0
    assert not summary["all_requested_work_completed"]


def test_missing_or_duplicate_admission_cannot_pass():
    a = VesselWorkAudit([dict(key="V1", block="Y01", work="LOAD", start_s=0, moves=2)])
    mbt, _ = terminal()
    assert not a.finish(mbt)[1]["recording_ok"]
    a.admission(dict(key="V1", asked=2, moves=0, ok=False))
    a.admission(dict(key="V1", asked=2, moves=0, ok=False))
    assert not a.finish(mbt)[1]["recording_ok"]
