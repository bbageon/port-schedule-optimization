"""Check reported contrasts against raw study files and the pushed Dashboard entry."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates, audit_dashboard,
    combine_reliability, judge_claim_alignment, judge_runtime_evidence, judge_scenario_validity,
)


OUT = Path("outputs/reports/yr332_short_job_audit")
SPEC = Path(".claude/docs/dashboard-task-specs/YR-332-bi-short-job-rule-advantage-audit.md")


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def main(commit):
    summary = load(OUT/"summary.json")
    stamp = load(OUT/"run-info.json")
    assert summary["trials"] == 144 and summary["matched_inputs"] == summary["baseline_replications"] == 9
    assert load(OUT/"noise-replay.json")["passed"]
    hashes = {}
    for p in OUT.rglob("*"):
        if p.suffix not in (".json", ".xml") or p.name in ("current-gates.json","dashboard-audit.json"):
            continue
        raw = subprocess.check_output(["git","show",f"{commit}:{p.as_posix()}"])
        assert raw == p.read_bytes(), p
        hashes[p.as_posix()] = hashlib.sha256(raw).hexdigest()
    for path, key in ((SPEC,"prereg_sha256"),(Path("scripts/v7/audit_short_job_rule.py"),"script_sha256"),
                      (Path("outputs/reports/yr332_global_policy/bundle.json"),"bundle_sha256")):
        raw = subprocess.check_output(["git","show",f"{stamp['code']['git_head']}:{path.as_posix()}"])
        candidates = (raw,raw.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n"))
        assert stamp["params"][key] in [hashlib.sha256(v).hexdigest() for v in candidates], path
    runtime = judge_runtime_evidence(stamp, artifact_hashes=hashes)
    remote = subprocess.check_output(["git","rev-parse","--abbrev-ref","@{u}"],encoding="utf8").strip()
    board = audit_dashboard(".",task_id="YR-332-bi",expected_state="in-progress",spec_path=SPEC,
        evidence_paths=[OUT/"README.md", OUT/"run-info.json", OUT/"summary.json"],
        evidence_commits=[commit], remote_ref=remote, pin_commit=commit)
    report = (OUT/"README.md").read_text(encoding="utf8")
    table = report.split("## 두 조건을 분리한 결과",1)[1].split("앞 두 열",1)[0]
    declared,truth={},{}
    for load_size, row in summary["loads"].items():
        line = re.search(r"^\| "+f"{int(load_size):,}"+r" \| (.+) \|$", table,re.M)
        assert line
        values = [float(n) for n in line[1].split(" | ")]
        keys = ("lead_on","lead_off","guard_contribution_to_lead","spt_off_minus_on","estimated_minus_exact","hidden_minus_exact")
        assert len(values)==len(keys)
        for k,v in zip(keys,values):
            declared[load_size+k],truth[load_size+k]=v,row[k]
    reliability=combine_reliability(runtime,board,judge_claim_alignment(declared,truth,absolute_tolerance=.00501))
    previous=load(Path("outputs/reports/yr332_vessel_turnaround/current-gates.json"))["common_gates"]["performance"]
    performance=GateOutcome("performance",GateStatus(previous["status"]),"직전 재학습 모델의 비용 판정 유지",
                            evidence=dict(source="YR-332-bj",study="fixed rules; no policy retraining",prior=previous))
    scenario=judge_scenario_validity(internal_checks=dict(ledger_conservation=True,physical_constraints=True,
        deterministic_replay=True),flow_checks={},anchors={},continuous_operation=False,request_real_terminal_claim=False)
    assert reliability.status is GateStatus.PASS, reliability.as_dict()
    gates=attach_common_gates(dict(task="YR-332-bi",evidence_commit=commit,scope="paired synthetic rule comparisons"),
                              ResearchGateReport(performance,reliability,scenario))
    for name, data in (("dashboard-audit.json", board.as_dict()),("current-gates.json",gates)):
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8",newline="\n")
    print({g.name:g.status.value for g in (performance,reliability,scenario)})


if __name__ == "__main__":
    main(sys.argv[1])
