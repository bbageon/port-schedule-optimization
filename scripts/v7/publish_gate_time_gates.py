"""Pin the gate-time correction evidence without claiming policy performance."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from recheck_transport_queue import digest

OUT = Path("outputs/reports/yr332_gate_time_provenance")


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def write_gate(path, value):
    # Git stores LF here. Hash exactly those bytes, including on Windows.
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1,default=str),encoding="utf-8",newline="\n")
    path.with_name(path.name+".sha256").write_text(digest(path)+"\n",encoding="utf-8",newline="\n")


def publish(commit):
    from yard_rl.experiments.gate_harness import (
        audit_dashboard, judge_runtime_evidence, judge_claim_alignment, combine_reliability,
        judge_scenario_validity, GateOutcome, GateStatus, ResearchGateReport, attach_common_gates)
    verification,build,inputs = read("verification.json"),read("build-record.json"),read("input-check.json")
    assert verification["status"] == "PASS" and all(verification["checks"].values())
    assert digest(Path("scripts/v7/verify_gate_time_calibration.py")) == verification["verification_script_sha256"]
    paths = [OUT/name for name in ("README.md","profile-confirmed.json","provenance-summary.json",
        "calibration-comparison.json","arrival-impacts.json","build-record.json","input-check.json",
        "verification.json","source-manifest.json","input-bundle.json","tests-contract.xml",
        "visit-arrival-templates-confirmed.json","visit-patterns-confirmed.json")]
    paths.extend(Path("scripts/v7")/name for name in ("check_gate_time_inputs.py","verify_gate_time_calibration.py","publish_gate_time_gates.py"))
    artifacts = {**build["source_sha256"],**{p.as_posix():digest(p) for p in paths}}
    runtime = judge_runtime_evidence(build["repro"],artifact_hashes=artifacts)
    input_runtime = judge_runtime_evidence(inputs["repro"],artifact_hashes=artifacts)
    assert input_runtime.status is GateStatus.PASS,input_runtime.as_dict()
    board = audit_dashboard(".",task_id="YR-332-ba",expected_state="done",
        spec_path=Path(".claude/docs/dashboard-task-specs/YR-332-ba-gate-time-provenance.md"),
        evidence_paths=paths,evidence_commits=[commit],remote_ref="origin/강화학습-판매",pin_commit=commit)
    body = (OUT/"README.md").read_text(encoding="utf-8")
    cohort = read("calibration-comparison.json")["cohorts"]
    claims,truth = {},{}
    labels = {"보정 · 반입":"calibration_IN","보정 · 반출":"calibration_OUT",
              "분포 확인 · 반입":"holdout_IN","분포 확인 · 반출":"holdout_OUT"}
    for label,key in labels.items():
        line = next(line for line in body.splitlines() if line.startswith(f"| {label} |"))
        values = [float(v.strip().replace(",","")) for v in line.strip("|").split("|")[1:]]
        c = cohort[key]
        expected = [c["legacy"]["n"],c["confirmed"]["n"],c["unconfirmed"]["n"],
                    c["legacy"]["error_seconds"]["median"],c["confirmed"]["error_seconds"]["median"]]
        for i,(a,b) in enumerate(zip(values,expected)):
            claims[f"{key}:{i}"],truth[f"{key}:{i}"] = a,b
        assert len(values) == len(expected)
    patterns = {"unconfirmed":r"미확인 시각 \*\*([\d,]+)건\*\*",
        "input_cases":r"합성 입력 \*\*([\d,]+)개\*\*",
        "replays":r"반복 생성 \*\*([\d,]+)개\*\*",
        "tests":r"관련 시험 \*\*([\d,]+)개\*\*"}
    for key,pattern in patterns.items():
        claims[key] = int(re.search(pattern,body).group(1).replace(",",""))
    suites = ET.parse(OUT/"tests-contract.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get(k,"0")) == 0 for s in suites for k in ("errors","failures","skipped"))
    truth.update(unconfirmed=read("provenance-summary.json")["private_quarantine"]["rows"],
        input_cases=len(inputs["cases"]),replays=len(inputs["repeats"]),tests=sum(int(s.get("tests")) for s in suites))
    alignment = judge_claim_alignment(claims,truth,absolute_tolerance=0)
    reliability = combine_reliability(runtime,board,alignment)
    # Only input replay is measured here; do not fabricate runtime crane/event guards.
    scenario = judge_scenario_validity(internal_checks=dict(deterministic_replay=all(verification["replay_matches"].values())),
        flow_checks={},anchors={},continuous_operation=False,request_real_terminal_claim=False)
    performance = GateOutcome("performance",GateStatus.INCONCLUSIVE,
        "입력 보정만 검증했으며 새 학습·정책 성능은 측정하지 않음")
    result = attach_common_gates(dict(task="YR-332-ba",evidence_commit=commit,
        gate_time_provenance_axis="PASS",independent_verification=verification["checks"],
        scope="Source-confirmed reservation/notice calibration and input integration only; physical demand unchanged",
        unresolved_context=["YR-332-f queue definition","YR-332-i transport stages","YR-332-bb dependent demand recalibration"]),
        ResearchGateReport(performance,reliability,scenario))
    write_gate(OUT/"dashboard-audit.json",board.as_dict())
    write_gate(OUT/"current-gates.json",result)
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:v["status"] for k,v in result["common_gates"].items() if isinstance(v,dict) and "status" in v})


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("--commit",required=True)
    publish(p.parse_args().commit)
