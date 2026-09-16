"""Container supply bounds must distinguish policy routing from base demand."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v3"))
from analyze_inventory_balance import balances


def truck(flow, original="A", final="A", done=10):
    return dict(flow=flow, requested_block=original, final_block=final, job_done_s=done)


def test_redirected_supply_can_create_shortage_despite_feasible_original_counts():
    requests = [truck("GATE_IN", final="B"), truck("GATE_OUT"), truck("GATE_OUT", done=None)]
    original = balances({"A": 1, "B": 0}, requests, [], use_final_blocks=False)
    final = balances({"A": 1, "B": 0}, requests, [], use_final_blocks=True)
    assert original["A"]["minimum_unfinished_removals"] == 0
    assert original["A"]["end_stock_from_completed_flows"] is None
    assert final["A"]["minimum_unfinished_removals"] == 1
    assert final["A"]["end_stock_from_completed_flows"] == 0
    assert final["B"]["end_stock_from_completed_flows"] == 1


def test_late_unfinished_inbound_is_planned_supply_but_not_physical_stock():
    rows = [truck("GATE_IN", done=None), truck("GATE_OUT", done=None)]
    value = balances({"A": 0}, rows, [], use_final_blocks=True)["A"]
    assert value["minimum_unfinished_removals"] == 0
    assert value["end_stock_from_completed_flows"] == 0
    assert value["truck_in_unfinished"] == value["truck_out_unfinished"] == 1


def test_ship_side_completion_does_not_create_yard_inventory():
    vessels = [dict(block="A", work="DISCHARGE", asked=3, completed_yard_jobs=1, sts_done=True)]
    value = balances({"A": 2}, [], vessels, use_final_blocks=True)["A"]
    assert value["balance_if_all_requested_work_completed"] == 5
    assert value["end_stock_from_completed_flows"] == 3
    assert value["vessel_in_unfinished"] == 2
