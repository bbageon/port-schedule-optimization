"""Explain saved admission failures and preserve descriptive costs without rerunning."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path

from analyze_request_audit import reconcile, sha

ARMS = ("NO_REALLOC", "RL", "RL_NOVETO")
LABELS = {"NO_REALLOC": "배정 변경 없음", "RL": "전체 모형", "RL_NOVETO": "수락 거절 기능 제거"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def failure_kind(event):
    """Describe observed inventory, not an inferred upstream cause."""
    snapshot = event.get("inventory_at_failure", {})
    if event.get("reason") == "TAIL":
        return "outside_horizon"
    if not snapshot.get("available"):
        return "missing_snapshot"
    if not snapshot["free_formula_matches"]:
        return "inventory_formula_mismatch"
    if event.get("reason") == "NO_TARGET":
        if snapshot["inventory_boxes"] == 0:
            return "block_empty"
        if snapshot["unclaimed_inventory_boxes"] == 0:
            return "all_inventory_claimed"
        return "target_missing_despite_unclaimed_inventory"
    if snapshot["admission_free"] <= snapshot["capacity_margin"]:
        if snapshot["physical_free"] <= snapshot["capacity_margin"]:
            return "physical_space_at_margin"
        return "reservations_consume_available_space"
    return "other_admission_failure"


def summarize_arm(folder, previous, expected_commit):
    result, manifest = read(folder / "result.json"), read(folder / "manifest.json")
    with gzip.open(folder / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    live_days = [json.loads(line) for line in (folder / "days.jsonl").read_text(encoding="utf-8").splitlines()]
    ledger = reconcile(result, manifest, rows, live_days, expected_commit=expected_commit)
    assert sha(folder / "requests.jsonl.gz") == result["request_ledger_sha256"]
    assert manifest["contract"]["settings"]["diagnose_admissions"] is True
    counts, examples = Counter(), []
    for row in rows:
        for event in row["admission_events"]:
            if event["outcome"] == "SKIPPED":
                kind = failure_kind(event)
                counts[kind] += 1
                examples.append({"job_id": row["job_id"], "requested_day": row["requested_day"],
                                 "kind": kind, "event": event})
    assert sum(counts.values()) == result["request_summary"]["skipped"]
    clipped = [row for row in result["vessel_admissions"] if row["asked"] > row["moves"]]
    old = read(previous / result["arm"] / "result.json") if previous is not None else None
    comparison = None
    if old is not None:
        keys = ("phi_krw", "c_wait", "c_move", "c_rehandle", "c_vessel", "n_trucks",
                "truck_skipped", "n_space", "n_time", "traded", "decisions")
        differences = [{"day": new["index"], "field": key, "old": prior[key], "new": new[key]}
            for prior, new in zip(old["days"], result["days"]) for key in keys
            if not math.isclose(prior[key], new[key], rel_tol=1e-12, abs_tol=1e-4)]
        comparison = {"same_day_count": len(old["days"]) == len(result["days"]),
            "same_requested_identity": old["requested_identity_sha256"] == result["requested_identity_sha256"],
            "differences": differences, "compared_fields": list(keys)}
    return {"total_cost_krw": math.fsum(d["phi_krw"] for d in result["days"]),
        "components_krw": {k: math.fsum(d[k] for d in result["days"])
                           for k in ("c_wait", "c_move", "c_rehandle", "c_vessel")},
        "requests": result["request_summary"], "vessel_work": result["vessel_work_summary"],
        "failure_kinds": dict(counts), "failures": examples, "reduced_vessel_admissions": clipped,
        "recording_checks": result["recording_checks"], "ledger_reconciliation": ledger,
        "prior_run_comparison": comparison, "elapsed_s": result["elapsed_s"]}


def summarize(out, *, expected_commit, previous=None):
    results = {arm: summarize_arm(out / arm, previous, expected_commit) for arm in ARMS}
    raw = {arm: read(out / arm / "result.json") for arm in ARMS}
    same_requests = len({r["requested_identity_sha256"] for r in raw.values()}) == 1
    passed = same_requests and all(all(r["recording_checks"].values()) and
        r["ledger_reconciliation"]["passed"] for r in results.values())
    baseline = results["NO_REALLOC"]["total_cost_krw"]
    for result in results.values():
        result["saving_percent"] = 100 * (baseline - result["total_cost_krw"]) / baseline
    payload = {"schema": "yr317.admission-replay-summary.v1", "recording_passed": passed,
        "same_requested_work": same_requests, "arms": results,
        "seed": raw["NO_REALLOC"]["seed"], "days_per_arm": len(raw["NO_REALLOC"]["days"]),
        "new_training_runs": 0, "independent_confirmation": False, "claim_eligible": False}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 미투입 상태 재관측 결과", "", f'시드 {payload["seed"]}, 정책당 {payload["days_per_arm"]}일 관측 진단. 추가학습·독립 반복에 따른 성능 확증이 아니다.', "",
        "| 정책 | 총비용(억 원) | 절감률 | 트럭 미투입 | 본선 미투입 | 본선 완료 | 본선 잔여 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, result in results.items():
        v = result["vessel_work"]
        lines.append(f'| {LABELS[arm]} | {result["total_cost_krw"]/1e8:.3f} | {result["saving_percent"]:.2f}% | '
            f'{result["requests"]["skipped"]:,} | {v["unadmitted_moves"]:,} | {v["completed_yard_jobs"]:,} | {v["outstanding_yard_jobs"]:,} |')
    lines += ["", "비용에는 미투입 작업의 현실 부담·외부 대기·일정 변경 비용이 포함되지 않는다. 비용 차이를 정책만의 기여로 확정하지 않는다.", "",
        f"요청·비용·새 관측 기록의 대조 통과: {passed}. 검사 통과는 관측의 정합이며 실제 항만 성능의 통과가 아니다.", "",
        "## 미투입 순간의 관측", ""]
    labels = {"block_empty": "해당 블록의 실제 재고 없음", "all_inventory_claimed": "재고는 있으나 모두 기존 작업에 지정됨",
        "physical_space_at_margin": "실제 빈 공간이 용량 여유 기준 이하", "reservations_consume_available_space": "실제 여유는 있으나 미도착 작업·예약을 빼면 투입 기준 미달",
        "target_missing_despite_unclaimed_inventory": "미지정 재고가 있는데도 대상 없음: 추가 확인 필요",
        "other_admission_failure": "그 밖의 실패: 개별 기록 확인 필요",
        "missing_snapshot": "실패 상태 기록 없음", "inventory_formula_mismatch": "여유 계산 불일치", "outside_horizon": "종료창 밖"}
    for arm, result in results.items():
        lines.append(f"- {LABELS[arm]}: " + (", ".join(f"{labels[k]} {v}건" for k, v in result["failure_kinds"].items()) or "미투입 없음"))
    lines += ["", "위 분류는 실패 순간의 관측이다. 최초 원인이나 누락을 해결하는 정책을 입증하지 않는다. 자세한 재고·예약·시각은 summary.json과 요청별 기록에 보존한다.", "",
        "다음은 같은 YR-317-g(요청 처리 기록 검증)에서 재현 차이와 실패 경로를 검토하고 필요한 단일축 보정 범위를 정하는 작업이다.", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
