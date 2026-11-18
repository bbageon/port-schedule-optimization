"""A server clock must not silently become a physical gate entry."""
import copy

import pytest

from build_gate_time_calibration import collect_pairs, flows_for
from gate_time_provenance import CONFIRMED, GateEvidence, load_evidence, parse_time


def request(*, at="20261104100000", log="2026-11-04 10:00:01", con="BOX1", truck="TRUCK1", flow="2", error="false"):
    return (f"{log} [INFO] REQUEST TO BLOCKCHAIN >> https://openapi.chainportal.co.kr/api/vbs/terminal/GateIn : "
            f"{{conNo={con}, truckNo={truck}, inOut={flow}, gateInTime={at}, error={error}}}")


def row(**kw):
    result = dict(con_no="BOX1", truck_no="TRUCK1", in_out="IN", gate_in_time="2026-11-04 10:00:00",
        doc_keys="d1", copino_first_time="2026-11-04 08:00:00")
    result.update(kw)
    return result


def test_log_time_alone_is_not_confirmed_entry():
    proof = GateEvidence(); proof.consume(request(at="", log="2026-11-04 10:00:00"))
    assert proof.classify(row()) == "unconfirmed_log_only"


def test_another_message_can_confirm_the_exact_same_entry():
    proof = GateEvidence(); proof.consume(request(at="", log="2026-11-04 10:00:00"))
    proof.consume(request(log="2026-11-04 10:00:21", flow="TO"))
    assert proof.classify(row()) == "confirmed_other_message"


@pytest.mark.parametrize("changed", [dict(con_no="BOX2"), dict(truck_no="TRUCK2"),
    dict(in_out="OUT"), dict(gate_in_time="2026-11-04 10:00:01")])
def test_no_join_across_container_truck_direction_or_time(changed):
    proof = GateEvidence(); proof.consume(request())
    assert proof.classify(row(**changed)) not in CONFIRMED


@pytest.mark.parametrize("error", ["true", "false"])
def test_error_flag_does_not_replace_time_evidence(error):
    explicit = GateEvidence(); explicit.consume(request(error=error))
    missing = GateEvidence(); missing.consume(request(at="", log="2026-11-04 10:00:00", error=error))
    assert explicit.classify(row()) in CONFIRMED
    assert missing.classify(row()) not in CONFIRMED


def test_later_real_entry_cannot_relabel_an_earlier_log_record():
    proof = GateEvidence(); proof.consume(request(at="", log="2026-11-04 10:00:00"))
    proof.consume(request(at="20261104100500", log="2026-11-04 10:05:01"))
    assert proof.classify(row()) == "unconfirmed_log_only"
    assert proof.classify(row(gate_in_time="2026-11-04 10:05:00")) == "confirmed_explicit"


def test_response_or_unrelated_log_does_not_certify_entry():
    proof = GateEvidence(); proof.consume(request().replace("/api/vbs/terminal/GateIn :", "/api/vbs/terminal/Other :"))
    assert proof.classify(row()) == "unconfirmed_no_request_match"


def test_conflicting_fields_fail_instead_of_picking_last_value():
    proof = GateEvidence()
    with pytest.raises(ValueError, match="Conflicting"):
        proof.consume(request().replace("gateInTime=", "conNo=DIFFERENT, gateInTime="))


@pytest.mark.parametrize("value", ["20261104990000", "2026-11-04T10:00:00+09:00", "unavailable"])
def test_broken_or_undeclared_clock_is_rejected(value):
    with pytest.raises(ValueError):
        parse_time(value)


def test_original_request_logs_are_required():
    with pytest.raises(ValueError, match="required"):
        load_evidence([])


def test_booking_rules_holdout_and_uncertain_partition():
    proof = GateEvidence()
    proof.consume(request())
    proof.consume(request(con="BOX2", flow="1"))
    proof.consume(request(con="BOX3", at="20261107100000", log="2026-11-07 10:00:01"))
    proof.consume(request(con="BOX4", at="", log="2026-11-04 10:00:00"))
    visits = [row(), row(con_no="BOX2", in_out="OUT", doc_keys="d2"),
        row(con_no="BOX3", gate_in_time="2026-11-07 10:00:00", copino_first_time="2026-11-07 08:00:00", doc_keys="d3"),
        row(con_no="BOX4", doc_keys="d4")]
    booking = dict(log_time="2026-11-04 08:00:00", in_out_reserve_time="2026-11-04 09:30:00",
        copino_err_type="NM", doc_key="d1", in_out="IN")
    orders = [booking, dict(booking, doc_key="d2", in_out="OUT", in_out_reserve_time="2026-11-04 11:00:00"),
        dict(booking,doc_key="d3",log_time="2026-11-07 08:00:00",in_out_reserve_time="2026-11-07 02:00:00"),
        dict(booking,doc_key="d4"),
        dict(booking,log_time="2026-11-04 11:00:00",in_out_reserve_time="2026-11-04 18:00:00")]
    groups,_ = collect_pairs(visits,orders,proof)
    flows = flows_for(groups["confirmed"])
    assert flows["GATE_IN"]["samples"] == [[1800,7200,1]]
    assert flows["GATE_OUT"]["samples"] == [[-3600,7200,1]]
    assert groups["unconfirmed"][("calibration","IN")] == [(1800,7200)]
    assert len(groups["confirmed"][("holdout","IN")]) == 1
    changed = copy.deepcopy(orders); changed[2]["in_out_reserve_time"] = "2026-11-06 01:00:00"
    changed_groups,_ = collect_pairs(visits,changed,proof)
    assert flows_for(changed_groups["confirmed"]) == flows
