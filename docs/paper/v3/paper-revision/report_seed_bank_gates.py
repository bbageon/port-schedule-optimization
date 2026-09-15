"""Report input-preparation gates after the evidence/board commit is pushed."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from yard_rl.experiments.gate_harness import (GateOutcome, GateStatus, ResearchGateReport,
    attach_common_gates, audit_dashboard, combine_reliability, judge_claim_alignment,
    judge_runtime_evidence, revalidate_pass_evidence)
from audit_seed_bank import BASE, RUN, read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-commit", required=True)
    args = parser.parse_args()
    verified = read(BASE / "input-verification.json")
    if not verified["passed"] or not all(verified["checks"].values()):
        raise ValueError("Input reconciliation must pass before closing input preparation")
    manifest = read(RUN / "manifest.json")
    hashes = dict(verified["artifact_sha256"])
    for path in (BASE / "input-verification.json", Path(__file__),
            Path(__file__).with_name("audit_seed_bank.py"),
            Path(__file__).with_name("14-독립시드-합성곡선.md")):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    stamp = {"code": {"git_head": manifest["source_commit"], "git_dirty": manifest["code_dirty"]},
        "seeds": {"month_roots": manifest["spec"]["seeds"]}, "params": manifest,
        "prereg": (BASE / "seed-bank-prereg.json").relative_to(ROOT).as_posix()}
    runtime = judge_runtime_evidence(stamp, artifact_hashes=hashes, root=ROOT)
    dashboard = audit_dashboard(ROOT, task_id="YR-317-d1", expected_state="done",
        spec_path=".claude/docs/dashboard-task-specs/YR-317-d1-v3-independent-seed-bank.md",
        evidence_paths=hashes, evidence_commits=[args.board_commit],
        remote_ref="origin/강화학습-판매", pin_commit=args.board_commit)
    body = Path(__file__).with_name("14-독립시드-합성곡선.md").read_text(encoding="utf-8")
    numbers = re.search(r"(\d+)개 시드 × (\d+)일 = (\d+)일, 트럭 요청 ([\d,]+)건", body)
    if numbers is None:
        raise ValueError("Reported input counts missing")
    reported = dict(zip(("months", "days_per_month", "days", "total_requests"),
        (int(x.replace(",", "")) for x in numbers.groups())))
    raw = {"months": verified["months"], "days_per_month": 30,
        "days": verified["days"], "total_requests": verified["total_requests"]}
    alignment = judge_claim_alignment(reported, raw)
    reliability = combine_reliability(runtime, dashboard, alignment)
    performance = GateOutcome("performance", GateStatus.INCONCLUSIVE,
        "독립 월의 입력만 생성했으며 정책 성능은 아직 비교하지 않았다.",
        ("새 학습·독립 성능 평가 0회",), {"scope": "input preparation only"})
    scenario = GateOutcome("scenario_validity", GateStatus.FAIL,
        "합성곡선 보존은 확인했지만 전체 요청 처리의 현실성은 아직 해결되지 않았다.",
        ("기존 재관측의 트럭 미투입·선박 잔여 보정은 YR-317-g에 남는다.",),
        {"curve_preserved": True, "scenario_completion_validated": False,
         "replay_evidence": "outputs/reports/yr317_v3_request_audit/replay-completion-verified.json"})
    report = ResearchGateReport(performance, reliability, scenario)
    valid, reason = revalidate_pass_evidence(report, root=ROOT)
    payload = attach_common_gates({"schema": "yr317.seed-preparation-completion-gates.v1",
        "scope": "20 independent input bundles, not policy performance or real-terminal validation",
        "verification_passed": verified["passed"], "saved_pass_revalidated": valid,
        "revalidation_reason": reason, "claim_eligible": False}, report)
    (BASE / "completion-gates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reliability": reliability.status.value, "performance": performance.status.value,
        "scenario_validity": scenario.status.value, "saved_pass_revalidated": valid,
        "reasons": reliability.reasons, "revalidation_reason": reason}, ensure_ascii=True))
    return int(reliability.status is not GateStatus.PASS or not valid)


if __name__ == "__main__":
    raise SystemExit(main())
