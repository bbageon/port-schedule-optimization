from pathlib import Path
import hashlib
import json
import subprocess

from yard_rl.experiments.gate_harness import (
    AnchorEvidence,
    ClaimScope,
    GateOutcome,
    GateStatus,
    ResearchGateReport,
    WorkKind,
    attach_common_gates,
    audit_dashboard,
    authorize_task,
    combine_reliability,
    judge_claim_alignment,
    judge_performance,
    judge_runtime_evidence,
    judge_scenario_validity,
    revalidate_pass_evidence,
    report_from_dict,
    verify_committed_artifact,
)
from yard_rl.integrated.evalkit import PairedResult, judge_primary


def _paired(lo: float, hi: float, mean: float = -2.0, mde: float = 1.0) -> PairedResult:
    return PairedResult(10, mean, 1.0, 0.3, lo, hi, mde, "test", 10, 10)


def _pass(name: str) -> GateOutcome:
    return GateOutcome(name, GateStatus.PASS, "ok")


def _guards():
    return {
        "completion": True,
        "backlog_zero": True,
        "physical_valid": True,
        "vessel_protection": True,
    }


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _scenario_checks():
    internal = {
        "event_time_order": True,
        "information_boundary": True,
        "physical_constraints": True,
        "ledger_conservation": True,
        "achievable_vessel_deadline": True,
        "deterministic_replay": True,
    }
    flow = {
        "continuous_arrivals": True,
        "warmup_excluded": True,
        "fixed_measurement_window": True,
        "load_state_classified": True,
        "flow_balance_consistent_with_classification": True,
    }
    return internal, flow


def test_performance_uses_raw_ci_and_three_way_labels():
    passed = judge_performance(
        _paired(-2.0004, -0.0004), interest_effect=2.0, hard_guards=_guards(),
    )
    assert passed.status is GateStatus.PASS
    assert passed.evidence["candidate_minus_baseline_ci95_raw"][1] == -0.0004

    inconclusive = judge_performance(
        _paired(-2.0, 0.2), interest_effect=2.0, hard_guards=_guards(),
    )
    assert inconclusive.status is GateStatus.INCONCLUSIVE

    failed = judge_performance(
        _paired(0.1, 1.0, mean=0.5), interest_effect=2.0, hard_guards=_guards(),
    )
    assert failed.status is GateStatus.FAIL


def test_existing_joint_judge_also_uses_unrounded_ci():
    control = [{"total": 1.0, "a2o": 10.0}, {"total": 2.0, "a2o": 20.0}]
    treatment = [
        {"total": 1.0004, "a2o": 10.0004},
        {"total": 2.0004, "a2o": 20.0004},
    ]
    result = judge_primary(
        treatment,
        control,
        metrics=("total", "a2o_min"),
        metric_keys={"total": ("total",), "a2o_min": ("a2o",)},
        delta={"total": 0.0, "a2o_min": 0.0},
    )
    assert result["total"]["ci"][0] == 0.0  # 표시값은 반올림됨
    assert result["total"]["ci_raw"][0] > 0.0
    assert result["joint_and_pass"] is True  # 판정은 +0.0004 원정밀도


def test_performance_guard_and_power_fail_closed():
    guard = judge_performance(
        _paired(-3.0, -1.0), interest_effect=2.0,
        hard_guards={**_guards(), "completion": False},
    )
    assert guard.status is GateStatus.FAIL
    weak = judge_performance(
        _paired(-3.0, -1.0, mde=4.0), interest_effect=2.0, hard_guards=_guards(),
    )
    assert weak.status is GateStatus.INCONCLUSIVE


def test_runtime_evidence_requires_clean_commit_seeds_and_hashes(tmp_path: Path):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen", encoding="utf-8")
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    head = _init_repo(tmp_path)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    stamp = {
        "code": {"git_head": head, "git_dirty": False},
        "seeds": {"candidate": [101, 102]},
        "params": {"p": {"x": 1}},
        "prereg": "prereg.md",
    }
    assert judge_runtime_evidence(
        stamp, artifact_hashes={"result.json": digest}, root=tmp_path
    ).status is GateStatus.PASS
    stamp["code"]["git_dirty"] = True
    assert judge_runtime_evidence(stamp, artifact_hashes={}, root=tmp_path).status is GateStatus.FAIL


def test_reported_claims_must_match_raw_values():
    assert judge_claim_alignment({"cost": -1.25}, {"cost": -1.25}).status is GateStatus.PASS
    failed = judge_claim_alignment({"cost": -1.2}, {"cost": -1.25})
    assert failed.status is GateStatus.FAIL
    assert "수치 불일치" in failed.reasons[0]
    assert judge_claim_alignment({"cost": float("nan")}, {"cost": 1.0}).status is GateStatus.FAIL


def test_dashboard_audit_exactly_one_row_and_matching_spec(tmp_path: Path):
    board = tmp_path / ".claude" / "Dashboard"
    board.mkdir(parents=True)
    for name in ("in-progress.md", "ready.md", "backlog.md", "cancelled.md", "done.md"):
        (board / name).write_text("# board\n", encoding="utf-8")
    (board / "done.md").write_text("| YR-153 | Infra | gate |\n", encoding="utf-8")
    spec = tmp_path / ".claude" / "spec.md"
    spec.write_text("- **상태**: **done (2026-08-05)**\n", encoding="utf-8")
    evidence = tmp_path / "report.json"
    evidence.write_text("{}", encoding="utf-8")
    head = _init_repo(tmp_path)
    outcome = audit_dashboard(
        tmp_path, task_id="YR-153", expected_state="done", spec_path=".claude/spec.md",
        evidence_paths=["report.json"], evidence_commits=[head], remote_ref="HEAD",
    )
    assert outcome.status is GateStatus.PASS
    no_evidence = audit_dashboard(
        tmp_path, task_id="YR-153", expected_state="done", spec_path=".claude/spec.md",
        evidence_paths=[],
    )
    assert no_evidence.status is GateStatus.FAIL

    (board / "ready.md").write_text("| YR-153 | Infra | duplicate |\n", encoding="utf-8")
    duplicate = audit_dashboard(
        tmp_path, task_id="YR-153", expected_state="done", spec_path=".claude/spec.md",
        evidence_paths=["report.json"], evidence_commits=[head], remote_ref="HEAD",
    )
    assert duplicate.status is GateStatus.FAIL
    (board / "ready.md").write_text("# board\n", encoding="utf-8")
    (board / "done.md").write_text(
        "| YR-153 | Infra | gate |\n| YR-153 | Infra | duplicate same file |\n",
        encoding="utf-8",
    )
    duplicate_same_file = audit_dashboard(
        tmp_path, task_id="YR-153", expected_state="done", spec_path=".claude/spec.md",
        evidence_paths=["report.json"], evidence_commits=[head], remote_ref="HEAD",
    )
    assert duplicate_same_file.status is GateStatus.FAIL

    (board / "done.md").write_text("| YR-153 | Infra | gate |\n", encoding="utf-8")
    spec.write_text(
        "- **상태**: **done (2026-08-05)**\n\n로컬에서만 바꾼 거짓 설명\n",
        encoding="utf-8",
    )
    dirty_spec = audit_dashboard(
        tmp_path, task_id="YR-153", expected_state="done", spec_path=".claude/spec.md",
        evidence_paths=["report.json"], evidence_commits=[head], remote_ref="HEAD",
    )
    assert dirty_spec.status is GateStatus.FAIL
    assert any("spec.md" in reason for reason in dirty_spec.reasons)


def test_dashboard_audit_rejects_evidence_not_linked_to_declared_commit(tmp_path: Path):
    board = tmp_path / ".claude" / "Dashboard"
    board.mkdir(parents=True)
    for name in ("in-progress.md", "ready.md", "backlog.md", "cancelled.md", "done.md"):
        (board / name).write_text("# board\n", encoding="utf-8")
    (board / "done.md").write_text("| YR-153 | Infra | gate |\n", encoding="utf-8")
    spec = tmp_path / ".claude" / "spec.md"
    spec.write_text("- **상태**: **done (2026-08-05)**\n", encoding="utf-8")
    head = _init_repo(tmp_path)
    (tmp_path / "late-result.json").write_text("{}", encoding="utf-8")

    outcome = audit_dashboard(
        tmp_path,
        task_id="YR-153",
        expected_state="done",
        spec_path=".claude/spec.md",
        evidence_paths=["late-result.json"],
        evidence_commits=[head],
        remote_ref="HEAD",
    )
    assert outcome.status is GateStatus.FAIL
    assert any("late-result.json" in reason for reason in outcome.reasons)


def test_scenario_gate_limits_claim_scope_and_checks_continuous_flow(tmp_path: Path):
    internal, flow = _scenario_checks()
    source = tmp_path / "anchor-source.json"
    anchor_names = (
        "gate_to_block_time",
        "initial_yard_occupancy",
        "truck_arrival_rate",
        "crane_service_time",
        "vessel_workload",
    )
    source.write_text(json.dumps({
        "schema": "yard_rl.external_anchor.v1",
        "anchors": {
            name: {
                "metric": name,
                "unit": "test-unit",
                "observed_range": [0.0, 10.0],
                "source": {"title": "Test source", "locator": "table-1"},
            }
            for name in anchor_names
        },
    }), encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    anchors = {
        name: AnchorEvidence(
            observed_min=0.0,
            observed_max=10.0,
            simulated_min=1.0,
            simulated_max=9.0,
            unit="test-unit",
            source_path="anchor-source.json",
            source_sha256=source_hash,
        )
        for name in anchor_names
    }
    simulation = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=False,
        root=tmp_path,
    )
    assert simulation.status is GateStatus.PASS
    assert simulation.evidence["allowed_claim_scope"] == ClaimScope.SIMULATION_METHOD_ONLY.value

    real = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=True,
        root=tmp_path,
    )
    assert real.status is GateStatus.INCONCLUSIVE

    trace = tmp_path / "operational-trace.json"
    trace.write_text(json.dumps({
        "schema": "yard_rl.operational_trace.v1",
        "records": [
            {
                "job_id": f"J{index}",
                "job_type": "IMPORT",
                "block_id": "A",
                "gate_in_time_s": index * 100.0,
                "block_in_time_s": index * 100.0 + 180.0,
                "completion_time_s": index * 100.0 + 300.0,
                "gate_out_time_s": index * 100.0 + 420.0,
            }
            for index in range(30)
        ],
    }), encoding="utf-8")
    trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    verified_real = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=True,
        operational_trace_path="operational-trace.json",
        operational_trace_sha256=trace_hash,
        root=tmp_path,
    )
    assert verified_real.status is GateStatus.PASS
    assert verified_real.evidence["allowed_claim_scope"] == ClaimScope.REAL_TERMINAL.value

    empty_trace = tmp_path / "empty-trace.json"
    empty_trace.write_text("{}", encoding="utf-8")
    empty_trace_hash = hashlib.sha256(empty_trace.read_bytes()).hexdigest()
    rejected_trace = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=True,
        operational_trace_path="empty-trace.json",
        operational_trace_sha256=empty_trace_hash,
        root=tmp_path,
    )
    assert rejected_trace.status is GateStatus.INCONCLUSIVE
    assert rejected_trace.evidence["allowed_claim_scope"] == ClaimScope.SIMULATION_METHOD_ONLY.value

    broken = dict(internal)
    broken["information_boundary"] = False
    invalid = judge_scenario_validity(
        internal_checks=broken, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=False,
        root=tmp_path,
    )
    assert invalid.status is GateStatus.FAIL
    finite_only = judge_scenario_validity(
        internal_checks=internal, flow_checks={}, anchors=anchors,
        continuous_operation=False, request_real_terminal_claim=False,
        root=tmp_path,
    )
    assert finite_only.status is GateStatus.FAIL

    invalid_anchor = dict(anchors)
    invalid_anchor["truck_arrival_rate"] = AnchorEvidence(
        observed_min=0.0,
        observed_max=10.0,
        simulated_min=1.0,
        simulated_max=11.0,
        unit="test-unit",
        source_path="anchor-source.json",
        source_sha256=source_hash,
    )
    outside = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=invalid_anchor,
        continuous_operation=True, request_real_terminal_claim=False,
        root=tmp_path,
    )
    assert outside.status is GateStatus.FAIL

    fake_source = tmp_path / "fake-source.json"
    fake_source.write_text("not evidence", encoding="utf-8")
    fake_hash = hashlib.sha256(fake_source.read_bytes()).hexdigest()
    fake_anchors = {
        name: AnchorEvidence(0.0, 10.0, 1.0, 9.0, "test-unit", "fake-source.json", fake_hash)
        for name in anchor_names
    }
    fake = judge_scenario_validity(
        internal_checks=internal, flow_checks=flow, anchors=fake_anchors,
        continuous_operation=True, request_real_terminal_claim=False, root=tmp_path,
    )
    assert fake.status is GateStatus.FAIL


def test_composite_gate_only_allows_targeted_remediation_or_objective():
    report = ResearchGateReport(_pass("performance"), _pass("reliability"), _pass("scenario_validity"))
    assert report.all_pass
    assert report.authorize(WorkKind.OBJECTIVE)[0]
    assert not report.authorize(WorkKind.NEW_HYPOTHESIS)[0]

    blocked = ResearchGateReport(
        GateOutcome("performance", GateStatus.INCONCLUSIVE, "unknown"),
        _pass("reliability"),
        GateOutcome("scenario_validity", GateStatus.FAIL, "bad"),
    )
    assert blocked.authorize(WorkKind.REMEDIATION, targets=["scenario_validity"])[0]
    assert not blocked.authorize(WorkKind.REMEDIATION, targets=["performance"])[0]
    assert not blocked.authorize(
        WorkKind.REMEDIATION, targets=["performance", "scenario_validity"]
    )[0]
    assert not blocked.authorize(WorkKind.REMEDIATION, targets=["reliability"])[0]
    assert not blocked.authorize(WorkKind.NEW_HYPOTHESIS, targets=["performance"])[0]
    payload = attach_common_gates({"raw": [1, 2]}, blocked)
    assert payload["raw"] == [1, 2]
    assert payload["common_gates"]["next_work_mode"] == "REMEDIATION_ONLY"


def test_reliability_combines_runtime_and_dashboard():
    combined = combine_reliability(
        _pass("runtime_evidence"), GateOutcome("dashboard", GateStatus.FAIL, "bad", ("x",)),
        _pass("claim_alignment"),
    )
    assert combined.status is GateStatus.FAIL
    assert combined.reasons == ("x",)


def test_saved_gate_report_is_fail_closed_and_authorizes_one_target(tmp_path: Path):
    payload = {
        "performance": {"status": "INCONCLUSIVE", "summary": "unknown", "evidence": {}},
        "reliability": {"status": "FAIL", "summary": "bad", "evidence": {}},
        "scenario_validity": {"status": "INCONCLUSIVE", "summary": "unknown", "evidence": {}},
    }
    report = report_from_dict(payload)
    assert report.unresolved == ("performance", "reliability", "scenario_validity")
    assert report.authorize(WorkKind.REMEDIATION, targets=["reliability"])[0]
    assert not report.authorize(WorkKind.REMEDIATION, targets=[])[0]
    board = tmp_path / ".claude" / "Dashboard"
    board.mkdir(parents=True)
    for name in ("in-progress.md", "ready.md", "backlog.md", "cancelled.md", "done.md"):
        (board / name).write_text("# board\n", encoding="utf-8")
    (board / "ready.md").write_text("| YR-200 | Infra | fix |\n", encoding="utf-8")
    spec = tmp_path / "spec.md"
    spec.write_text(
        "# YR-200 — fix\n\n- **3대 게이트 보정 대상**: `reliability`\n",
        encoding="utf-8",
    )
    assert authorize_task(
        report, root=tmp_path, task_id="YR-200", spec_path="spec.md",
        kind=WorkKind.REMEDIATION, targets=["reliability"],
    )[0]
    assert not authorize_task(
        report, root=tmp_path, task_id="YR-999", spec_path="spec.md",
        kind=WorkKind.REMEDIATION, targets=["reliability"],
    )[0]


def test_arbitrary_all_pass_gate_json_is_rejected():
    payload = {
        name: {"status": "PASS", "summary": "forged", "evidence": {}}
        for name in ("performance", "reliability", "scenario_validity")
    }
    try:
        report_from_dict(payload)
    except ValueError as exc:
        assert "PASS evidence" in str(exc)
    else:
        raise AssertionError("빈 PASS evidence가 승인됨")


def test_saved_performance_pass_is_recomputed_not_trusted(tmp_path: Path):
    payload = {
        "performance": {
            "status": "PASS",
            "summary": "forged",
            "evidence": {
                "metric": "terminal_total_cost",
                "baseline": "SF-SPT",
                "n": 10,
                "candidate_minus_baseline_ci95_raw": [1.0, 2.0],
                "minimum_improvement": 0.0,
                "interest_effect": 2.0,
                "mde80": 1.0,
                "hard_guards": _guards(),
            },
        },
        "reliability": {"status": "INCONCLUSIVE", "summary": "unknown", "evidence": {}},
        "scenario_validity": {"status": "INCONCLUSIVE", "summary": "unknown", "evidence": {}},
    }
    report = report_from_dict(payload)
    valid, reason = revalidate_pass_evidence(report, root=tmp_path)
    assert not valid
    assert "performance" in reason


def test_gate_artifact_must_be_hashed_committed_and_unchanged(tmp_path: Path):
    gate = tmp_path / "gate.json"
    gate.write_text("{}", encoding="utf-8")
    head = _init_repo(tmp_path)
    digest = hashlib.sha256(gate.read_bytes()).hexdigest()
    assert verify_committed_artifact(
        tmp_path,
        artifact_path="gate.json",
        artifact_sha256=digest,
        commit=head,
        remote_ref="HEAD",
    ).status is GateStatus.PASS

    gate.write_text('{"forged":true}', encoding="utf-8")
    forged_digest = hashlib.sha256(gate.read_bytes()).hexdigest()
    assert verify_committed_artifact(
        tmp_path,
        artifact_path="gate.json",
        artifact_sha256=forged_digest,
        commit=head,
        remote_ref="HEAD",
    ).status is GateStatus.FAIL
