"""Audit completed demand-preservation runs from saved evidence only.

This does not import the simulator, change costs, or run/train a policy. Vessel
yard completion and ship completion are different events and remain separate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import time

from analyze_admission_replay import ARMS, LABELS, read
from analyze_request_audit import reconcile, sha


def inspect(result, requests):
    end = result["request_summary"]["end_s"]
    vessels = result["vessel_work_ledger"]
    summary = result["vessel_work_summary"]
    issues = []

    def check(ok, reason):
        if not ok:
            issues.append(reason)

    check(len({v["key"] for v in vessels}) == len(vessels), "duplicate vessel stream")
    totals = Counter()
    pending = []
    for v in vessels:
        key = v["key"]
        asked, admitted = v["asked"], v["admitted"]
        done, active = v["completed_yard_jobs"], v["outstanding_yard_jobs"]
        numbers = [asked, admitted, done, active, v["unadmitted_moves"], v["unaccounted_yard_jobs"]]
        check(all(type(n) is int and n >= 0 for n in numbers), f"{key}: invalid counts")
        check(asked == admitted + v["unadmitted_moves"], f"{key}: requested work mismatch")
        check(admitted == done + active + v["unaccounted_yard_jobs"], f"{key}: admitted work mismatch")
        check(v["attempted"] is True and v["unaccounted_yard_jobs"] == 0, f"{key}: missing work")
        check(0 <= v["start_s"] <= end, f"{key}: invalid start")
        check(type(v["sts_done"]) is bool, f"{key}: ship completion unobserved")
        remaining = v["sts_remaining"]
        check(type(remaining) is int and 0 <= remaining <= admitted, f"{key}: invalid ship remaining")
        if v["sts_done"] is True:
            completion = v["sts_completion_s"]
            check(remaining == 0 and completion is not None and v["start_s"] <= completion <= end,
                  f"{key}: ship completion mismatch")
        totals.update(requested_moves=asked, admitted_moves=admitted,
            completed_yard_jobs=done, outstanding_yard_jobs=active,
            unadmitted_moves=v["unadmitted_moves"], unaccounted_yard_jobs=v["unaccounted_yard_jobs"])
        if active or not v["sts_done"] or v["unadmitted_moves"]:
            pending.append(dict(v, elapsed_since_start_h=(end - v["start_s"]) / 3600,
                                started_before_final_day=v["start_s"] < (len(result["plan"]) - 1) * 86400))
    for field, total in totals.items():
        check(summary[field] == total, f"vessel summary mismatch: {field}")
    check(summary["streams"] == len(vessels), "vessel stream count mismatch")
    all_vessels = bool(vessels) and not pending and not issues
    check(summary["all_requested_work_completed"] == all_vessels, "vessel completion flag mismatch")

    by_id = {r["job_id"]: r for r in requests}
    vessel_jobs = {f'{v["block"]}:J-{v["key"]}-{i:04d}': v
                   for v in vessels for i in range(v["admitted"])}
    seen = set()
    for b in result.get("demand_bindings", []):
        jid, t = b["job_id"], b["at_s"]
        check(jid not in seen, f"{jid}: repeated late binding")
        seen.add(jid)
        check(math.isfinite(t) and 0 <= t <= end and bool(b["target"]), f"{jid}: invalid binding")
        if b["flow"] == "GATE_OUT":
            r = by_id.get(jid)
            check(r is not None, f"{jid}: binding without requested truck")
            if r is not None:
                check(r["flow"] == "GATE_OUT" and r["block_in_s"] is not None
                      and r["block_in_s"] <= t + 1e-6, f"{jid}: binding before block arrival")
                check(r["job_done_s"] is None or t <= r["job_done_s"] + 1e-6,
                      f"{jid}: binding after completion")
        else:
            v = vessel_jobs.get(jid)
            check(b["flow"] == "VESSEL_LOAD" and v is not None, f"{jid}: unknown binding flow/job")
            if v is not None:
                check(v["work"] == "LOAD" and v["block"] == b["block"] and v["start_s"] <= t,
                      f"{jid}: binding inconsistent with vessel stream")
    check(len(seen) == result["request_summary"]["late_target_bindings"], "binding count mismatch")

    unfinished = [r for r in requests if r["state"] != "COMPLETED"]
    delayed = [max(0.0, r["gate_in_s"] - r["requested_arrival_s"])
               for r in requests if r["gate_in_s"] is not None]
    delayed = sorted(x for x in delayed if x > 1e-6)
    return {"passed": not issues, "issues": issues, "vessel_totals": dict(totals),
        "vessels_with_unfinished_work": pending, "all_vessel_work_completed": all_vessels,
        "all_trucks_completed": bool(requests) and not unfinished,
        "unfinished_trucks_by_state": dict(Counter(r["state"] for r in unfinished)),
        "unfinished_trucks": unfinished, "late_bindings_checked": len(seen),
        "unbound_jobs_at_end": result["request_summary"]["unbound_jobs_at_end"],
        "arrival_shift": {"later_gate_count": len(delayed), "total_hours": math.fsum(delayed) / 3600,
            "mean_minutes_among_shifted": (math.fsum(delayed) / len(delayed) / 60) if delayed else 0,
            "max_minutes": max(delayed, default=0) / 60,
            "scope": "observed gate time minus original requested time; not proof of actual driver waiting or a monetary cost"},
        "limits": ["Saved vessel rows do not contain each job's release time, physical inventory, or remaining service work.",
            "Binding checks prove recorded time consistency, not inventory availability at every binding.",
            "Remaining work is not assigned a cause from aggregate counters alone."],
        "claim_eligible": False, "new_training_runs": 0}


def audit_folder(folder):
    result, manifest = read(folder / "result.json"), read(folder / "manifest.json")
    with gzip.open(folder / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
        requests = [json.loads(line) for line in stream]
    days = [json.loads(line) for line in (folder / "days.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = inspect(result, requests)
    audit["request_reconciliation"] = reconcile(result, manifest, requests, days,
        expected_commit=manifest["repro"]["code"]["git_head"])
    audit["artifact_sha256"] = {name: sha(folder / name)
        for name in ("result.json", "manifest.json", "requests.jsonl.gz", "days.jsonl")}
    audit["passed"] &= (audit["request_reconciliation"]["passed"]
        and audit["artifact_sha256"]["requests.jsonl.gz"] == result["request_ledger_sha256"])
    audit["arm"], audit["source_commit"] = result["arm"], manifest["repro"]["code"]["git_head"]
    return audit


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Directory with policy subfolders")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {5})
    args.out.mkdir(parents=True, exist_ok=False)
    values = {}
    try:
        while len(values) < len(ARMS):
            for arm in ARMS:
                if arm in values or not (args.run / arm / "result.json").exists():
                    continue
                # A full-run result may still be in the middle of its final write.
                try:
                    value = audit_folder(args.run / arm)
                except json.JSONDecodeError:
                    if args.watch:
                        continue
                    raise
                save(args.out / f"{arm}.json", value)
                values[arm] = value
                print(json.dumps({"arm": arm, "passed": value["passed"],
                    "unfinished_vessels": len(value["vessels_with_unfinished_work"]),
                    "unfinished_trucks": len(value["unfinished_trucks"])}), flush=True)
            save(args.out / "progress.json", {"at": datetime.now(timezone.utc).isoformat(),
                "state": "completed" if len(values) == len(ARMS) else "waiting",
                "audited_arms": list(values), "all_available_passed": all(v["passed"] for v in values.values())})
            if len(values) == len(ARMS):
                break
            if not args.watch:
                raise RuntimeError("Some simulation results are not complete")
            status_path = args.run.parent / "progress.json"
            if time.time() - status_path.stat().st_mtime > 300 or read(status_path)["state"] == "failed":
                raise RuntimeError("Simulation stopped or supervisor heartbeat is stale")
            time.sleep(15)
        lines = ["# 요청 보존 이후 작업·시간 기록 검사", "",
            "원자료를 읽어 검산한 결과다. 새 시뮬레이션·학습·비용 부과는 하지 않았다.", "",
            "| 정책 | 기록 대조 | 트럭 미완료 | 미완료 선박 묶음 | 도착이 늦춰진 트럭 | 변경 시간 합계(시간) |",
            "|---|---|---:|---:|---:|---:|"]
        for arm, v in values.items():
            shift = v["arrival_shift"]
            lines.append(f'| {LABELS[arm]} | {v["passed"]} | {len(v["unfinished_trucks"])} | '
                f'{len(v["vessels_with_unfinished_work"])} | {shift["later_gate_count"]:,} | {shift["total_hours"]:,.2f} |')
        lines += ["", "도착 변경 시간은 원래 요청 시각과 실제 게이트 진입 시각의 차이다. 기사가 그 시간 동안 밖에서 기다렸다는 실측은 아니다.",
            "현재 비용은 게이트 진입 이후부터 계산하므로 이 시간 자체에 별도 변경 요금을 부과하지 않는다.",
            "선박의 야드 작업 완료와 선박 측 작업 완료는 각각 확인했다. 잔여의 발생 원인은 집계만으로 단정하지 않는다.",
            "미완료 작업 목록·선박 시작 후 경과시간·검사 한계는 정책별 JSON에 보존한다.", ""]
        (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
        if not all(v["passed"] for v in values.values()):
            raise RuntimeError("Saved outcome audit found mismatches; see policy JSON files")
    except BaseException as exc:
        save(args.out / "failure.json", {"error": repr(exc), "audited_arms": list(values)})
        raise


if __name__ == "__main__":
    main()
