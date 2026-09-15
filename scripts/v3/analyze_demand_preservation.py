"""After the fixed diagnosis, reconcile saved requests and generate a cost/throughput table."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze_admission_replay import ARMS, LABELS, read, summarize_arm
from analyze_request_audit import sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze(args):
    folder = args.run / "full"
    expected_commit = read(folder / "launch.json")["source_commit"]
    values = {a: summarize_arm(folder / a, args.previous, expected_commit) for a in ARMS}
    raw = {a: read(folder / a / "result.json") for a in ARMS}
    old = {a: read(args.previous / a / "result.json") for a in ARMS}
    checks = {"source_events_reconciled": all(r["ledger_reconciliation"]["passed"] for r in values.values()),
        "same_requests_across_policies": len({r["requested_identity_sha256"] for r in raw.values()}) == 1,
        "same_requests_as_original_run": all(r["prior_run_comparison"]["same_requested_identity"] for r in values.values()),
        "all_trucks_admitted": all(r["requests"]["requested"] == r["requests"]["admitted"] == 69000
            and r["requests"]["skipped"] == r["requests"]["unprocessed"] == 0 for r in values.values()),
        "original_vessel_work_preserved": all(r["vessel_work"]["requested_moves"] == 44322
            and r["vessel_work"]["unadmitted_moves"] == 0 for r in values.values()),
        "runtime_and_physical_checks": all(all(r["recording_checks"].values())
            and r["requests"]["physical_invariants_enabled"] for r in values.values())}
    completed = all(r["requests"]["states"] == {"COMPLETED": 69000}
        and r["vessel_work"]["all_requested_work_completed"] for r in values.values())
    baseline = values["NO_REALLOC"]["total_cost_krw"]
    for a, row in values.items():
        row["saving_percent"] = 100 * (baseline - row["total_cost_krw"]) / baseline
        row["old_total_cost_krw"] = sum(d["phi_krw"] for d in old[a]["days"])
        row["old_skipped"] = old[a]["request_summary"]["skipped"]
        row["old_vessel_unadmitted"] = old[a]["vessel_work_summary"]["unadmitted_moves"]
    artifacts = {str(p): sha(p) for a in ARMS for p in (folder / a / "result.json",
        folder / a / "manifest.json", folder / a / "requests.jsonl.gz", folder / a / "days.jsonl")}
    report = {"schema": "yr317.demand-preservation-comparison.v1", "checks": checks,
        "preservation_passed": all(checks.values()), "all_requested_completed": completed,
        "arms": values, "artifact_sha256": artifacts, "source_commit": expected_commit,
        "new_training_runs": 0, "independent_confirmation": False, "claim_eligible": False,
        "training_status": "not started; inspect remaining physical work and preregister training separately"}
    save(args.out / "comparison.json", report)
    lines = ["# 요청 보존 계약 — 같은 10일 진단 결과", "",
        "기존 시드·트럭 69,000건·본선 44,322건을 세 정책으로 비교했다. 새 학습·독립 성능 확증은 아니다.", "",
        "| 정책 | 기존 비용(억 원) | 보정 후 비용(억 원) | 보정 후 기준 대비 절감 | 기존→보정 미투입 | 보정 후 트럭 미완료 |",
        "|---|---:|---:|---:|---:|---:|"]
    for a, r in values.items():
        lines.append(f'| {LABELS[a]} | {r["old_total_cost_krw"]/1e8:.3f} | {r["total_cost_krw"]/1e8:.3f} | '
            f'{r["saving_percent"]:.2f}% | {r["old_skipped"]} → {r["requests"]["skipped"]} | '
            f'{r["requests"]["states"].get("CENSORED", 0)} |')
    lines += ["", "비용 변화에는 요청 보존과 대상 지정 시점 변경의 영향이 함께 있다. 학습 효과로 읽지 않는다.", "",
        "| 정책 | 기존→보정 본선 미투입 | 보정 후 본선 완료 | 보정 후 본선 잔여 |", "|---|---:|---:|---:|"]
    for a, r in values.items():
        v = r["vessel_work"]
        lines.append(f'| {LABELS[a]} | {r["old_vessel_unadmitted"]} → {v["unadmitted_moves"]} | '
            f'{v["completed_yard_jobs"]:,} | {v["outstanding_yard_jobs"]:,} |')
    lines += ["", f'입력·접수·비용 기록 대조 통과: {report["preservation_passed"]}. 전 요청 완료: {completed}.',
        "접수 성공과 실제 완주를 구분한다. 남은 작업이 있으면 원인을 확인하기 전 현실성 전체 통과로 승격하지 않는다.",
        "외부 일정 변경 비용·예약 정원·실측 대기 공간은 이번 보정에서 추가하지 않았다.",
        "다음은 YR-317-g에서 잔여와 물리 검사를 판정하고, 통과 범위에 맞춰 YR-317-c·d의 검증과 필요한 재학습을 진행하는 일이다.", ""]
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--launch", action="store_true")
    args = p.parse_args()
    if args.launch:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.with_suffix(".watch.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--run", str(args.run), "--previous", str(args.previous), "--out", str(args.out), "--watch"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(args.out.with_suffix(".watch.json"), {"pid": process.pid, "run": str(args.run),
            "analyzer_sha256": sha(__file__), "out": str(args.out)})
        print(json.dumps({"pid": process.pid, "out": str(args.out)}))
        return
    os.sched_setaffinity(0, {4}) if hasattr(os, "sched_setaffinity") else None
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        while True:
            if time.time() - (args.run / "progress.json").stat().st_mtime > 300:
                raise RuntimeError("Simulation supervisor heartbeat is stale; inspect processes")
            status = read(args.run / "progress.json")
            if status["state"] == "failed":
                raise RuntimeError("Simulation failed; inspect preserved run evidence")
            if status["state"] == "completed":
                break
            if not args.watch:
                raise RuntimeError("Simulation has not completed")
            save(args.out / "progress.json", {"state": "waiting", "simulation": status})
            time.sleep(10)
        report = analyze(args)
        save(args.out / "progress.json", {"state": "completed",
            "preservation_passed": report["preservation_passed"], "all_requested_completed": report["all_requested_completed"]})
        print(json.dumps({k: report[k] for k in ("preservation_passed", "all_requested_completed", "checks")}))
    except BaseException as exc:
        save(args.out / "progress.json", {"state": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
