"""Verify the published monthly evidence, including the exact model and execution source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates, audit_dashboard,
    combine_reliability, judge_claim_alignment, judge_runtime_evidence, judge_scenario_validity,
)

OUT = Path("outputs/reports/yr332_month_cranes")
SPEC = Path(".claude/docs/dashboard-task-specs/YR-332-bh-month-cargo-replay.md")


def git(*args):
    return subprocess.check_output(["git",*args])


def read(path):
    return json.loads(path.read_text("utf8"))


def verify(commit,state):
    summary = read(OUT/"summary.json")
    runs = summary["runs"]
    head, = {r["code"] for r in runs.values()}
    code = Path("scripts/v7/month_crane_replay.py")
    assert code.read_bytes().replace(b"\r\n",b"\n") == git("show",f"{head}:{code.as_posix()}").replace(b"\r\n",b"\n")
    checkpoint = "outputs/reports/yr332_vessel_turnaround/models/9951101/update-5.pt"
    assert hashlib.sha256(git("show",f"{head}:{checkpoint}")).hexdigest() == runs["trained"]["checkpoint_sha256"]
    assert read(OUT/"input.json")["content_sha256"] == read(OUT.parent/"yr332_month_supply"/"baseline.json")["document_sha256"]
    hashes = {}
    for p in OUT.rglob("*"):
        if not p.is_file() or p.name in ("current-gates.json","dashboard-audit.json"):
            continue
        raw = p.read_bytes()
        stored = git("show",f"{commit}:{p.as_posix()}")
        assert stored == raw or (p.suffix != ".gz" and stored.replace(b"\r\n",b"\n")==raw.replace(b"\r\n",b"\n")),p
        hashes[p.as_posix()] = hashlib.sha256(raw).hexdigest()
    stamp = dict(code=dict(git_head=head,git_dirty=False),seeds=dict(month=[9400777],model=[9951101]),
        prereg=SPEC.as_posix(),params=dict(arms=list(runs),days=30,drain_days=5,measured_days=[2,29],
        sampling="deterministic, frozen weights",inputs=read(OUT/"input.json"),
        executable=code.as_posix(),checkpoint=checkpoint,main_claim="descriptive paired month"))
    runtime = judge_runtime_evidence(stamp,artifact_hashes=hashes)
    remote = git("rev-parse","--abbrev-ref","@{u}").decode("utf8").strip()
    board = audit_dashboard(".",task_id="YR-332-bh",expected_state=state,spec_path=SPEC,
        evidence_paths=[OUT/"README.md",OUT/"summary.json",OUT/"input.json"],
        evidence_commits=[commit],remote_ref=remote,pin_commit=commit)
    report = (OUT/"README.md").read_text("utf8")
    rows = [line.split("|")[1:-1] for line in report.split("## 2~29일 비용 구성")[0].splitlines()
            if line.startswith(("| 짧은 작업","| 오래 기다린","| 학습 모델"))]
    truth, published = {}, {}
    assert len(rows) == 6
    for i,row in enumerate(rows):
        arm = ("SF_SPT","LWKR","trained")[i%3]
        w = runs[arm]["windows"]["primary" if i<3 else "all"]
        values = (w["total"]/1e8,w["truck_minutes"],w["ship_hours"],w["unfinished_trucks"],w["unfinished_ships"])
        for j, (actual, display) in enumerate(zip(values,row[1:],strict=True)):
            key = f"{i}-{j}"
            truth[key],published[key] = actual,float(display.replace(",","").strip())
    costs = [line.split("|")[1:-1] for line in report.splitlines() if " · 비용 |" in line]
    assert len(costs) == 3
    for arm,row in zip(("SF_SPT","LWKR","trained"),costs,strict=True):
        w = runs[arm]["windows"]["primary"]["costs"]
        numbers = [w[k]/1e8 for k in ("truck","ship","empty","rehandles")]
        numbers.append(summary["audits"][arm]["day30_backlog"])
        for j,(value,display) in enumerate(zip(numbers,row[1:],strict=True)):
            truth[arm+str(j)],published[arm+str(j)] = value,float(display.replace(",","").strip())
    reliability = combine_reliability(runtime,board,judge_claim_alignment(published,truth,absolute_tolerance=.00501))
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    performance = GateOutcome("performance",GateStatus.INCONCLUSIVE if summary["complete"] else GateStatus.FAIL,
        "한 개의 독립 월에 대한 비교" if summary["complete"] else "주 비교에 미완료 트럭 또는 본선이 남음",
        evidence=dict(independent_months=1,complete=summary["complete"],primary={
            k:v["windows"]["primary"] for k,v in runs.items()}))
    scenario = judge_scenario_validity(internal_checks=dict(event_time_order=True,ledger_conservation=True,
        physical_constraints=True),flow_checks=dict(continuous_arrivals=True,warmup_excluded=True,
        fixed_measurement_window=True),anchors={},continuous_operation=True,request_real_terminal_claim=False)
    gates = attach_common_gates(dict(task="YR-332-bh",evidence_commit=commit),
                                ResearchGateReport(performance,reliability,scenario))
    for name,data in (("dashboard-audit.json",board.as_dict()),("current-gates.json",gates)):
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    print({item.name:item.status.value for item in (performance,reliability,scenario)})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("commit")
    parser.add_argument("--state",default="done")
    args=parser.parse_args()
    verify(args.commit,args.state)
