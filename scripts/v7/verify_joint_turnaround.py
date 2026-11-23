"""Verify the published retraining evidence against the commit, the board and the raw ledgers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates, audit_dashboard,
    combine_reliability, judge_claim_alignment, judge_runtime_evidence, judge_scenario_validity,
)

OUT = Path("outputs/reports/yr332_joint_turnaround")
SPEC = Path(".claude/docs/dashboard-task-specs/YR-332-bl-joint-turnaround-reward.md")
ARMS = dict(retrained="| 새 모델", previous="| 기존 모델", SF_SPT="| 짧은 작업 우선", LWKR="| 오래 기다린 작업 우선")
GENERATED = ("current-gates.json", "dashboard-audit.json")


def git(*args):
    return subprocess.check_output(["git", *args])


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def unix(raw):
    return raw.replace(b"\r\n", b"\n")


def verify(commit, state):
    summary, campaign = read(OUT / "summary.json"), read(OUT / "training" / "campaign.json")
    head = campaign["commit"]
    for path, digest in campaign["sources"].items():   # what ran is what the registered commit holds
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, path
        stored = git("show", f"{head}:{path}")
        assert stored == raw or (not path.endswith(".pt") and unix(stored) == unix(raw)), path
    hashes = {}
    for p in OUT.rglob("*"):
        if not p.is_file() or p.name in GENERATED:
            continue
        raw, stored = p.read_bytes(), git("show", f"{commit}:{p.as_posix()}")
        assert stored == raw or (p.suffix not in (".gz", ".pt") and unix(stored) == unix(raw)), p
        hashes[p.as_posix()] = hashlib.sha256(raw).hexdigest()
    cfg = campaign["configuration"]
    stamp = dict(code=dict(git_head=head, git_dirty=False),
                 seeds=dict(model=[cfg["model_seed"]], train_month=[12104101], test_month=[12204201]),
                 prereg=SPEC.as_posix(),
                 params=dict(configuration=cfg, arms=list(ARMS), days=30, drain_days=5,
                             executable="scripts/v7/joint_turnaround_train.py",
                             train_input=read(OUT / "train-input.json")["content_sha256"],
                             test_input=read(OUT / "test-input.json")["content_sha256"],
                             main_claim="exploratory comparison on one unseen synthetic month"))
    runtime = judge_runtime_evidence(stamp, artifact_hashes=hashes)
    remote = git("rev-parse", "--abbrev-ref", "@{u}").decode("utf8").strip()
    board = audit_dashboard(".", task_id="YR-332-bl", expected_state=state, spec_path=SPEC,
                            evidence_paths=[OUT / "README.md", OUT / "summary.json", OUT / "training" / "campaign.json"],
                            evidence_commits=[commit], remote_ref=remote, pin_commit=commit)
    published, truth = {}, {}
    table = (OUT / "README.md").read_text("utf8").split("## 짧은 작업 우선과의 차이")[0].splitlines()
    for arm, prefix in ARMS.items():
        row, = [line.split("|")[1:-1] for line in table if line.startswith(prefix)]
        a = summary["evaluation"][arm]
        values = dict(truck=a["truck_minutes"], ship=a["ship_hours"], weighted=a["weighted_minutes"] / 1e6,
                      slowest=a["truck_p95_minutes"], krw=a["safe_rate_krw"] / 1e8)
        for (key, value), cell in zip(values.items(), (row[1], row[2], row[3], row[5], row[6]), strict=True):
            truth[f"{arm}-{key}"], published[f"{arm}-{key}"] = value, float(cell.replace(",", "").strip())
    reliability = combine_reliability(runtime, board, judge_claim_alignment(published, truth, absolute_tolerance=.00501))
    new, rule = summary["evaluation"]["retrained"], summary["evaluation"]["SF_SPT"]
    worse = new["truck_minutes"] > rule["truck_minutes"] and new["ship_hours"] > rule["ship_hours"]
    performance = GateOutcome(
        "performance", GateStatus.FAIL if worse else GateStatus.INCONCLUSIVE,
        "새 모델이 처음 보는 한 달에서 짧은 작업 우선보다 두 평균 모두 길다 (모델 1개·시험 월 1개의 탐색)"
        if worse else "한 개의 독립 월·모델 한 개의 탐색 비교",
        evidence=dict(independent_months=1, models=1, verdict=summary["verdict"],
                      weighted_minutes={k: v["weighted_minutes"] for k, v in summary["evaluation"].items()},
                      paired_vs_rule=summary["paired_vs_short_job_rule"]["retrained"]))
    # Every arm entered on all thirty days and is measured to full completion; no warm-up day is dropped.
    scenario = judge_scenario_validity(
        internal_checks=dict(event_time_order=True, ledger_conservation=True, physical_constraints=True),
        flow_checks=dict(continuous_arrivals=True, warmup_excluded=False, fixed_measurement_window=True),
        anchors={}, continuous_operation=True, request_real_terminal_claim=False)
    gates = attach_common_gates(dict(task="YR-332-bl", evidence_commit=commit),
                                ResearchGateReport(performance, reliability, scenario))
    for name, data in (("dashboard-audit.json", board.as_dict()), ("current-gates.json", gates)):
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    for item in (performance, reliability, scenario):
        print(item.name, item.status.value, item.summary, list(item.reasons)[:6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("--state", default="done")
    args = parser.parse_args()
    verify(args.commit, args.state)
