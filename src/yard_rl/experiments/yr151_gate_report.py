"""YR-151 0A → YR-153 3대 게이트 보고 (신뢰성 축 재판정).

0A 는 **신뢰성 게이트 단일축 보정**으로 인가받은 작업이다. 이 모듈은 0A 원자료
(`contract_0a.json`)를 하네스가 요구하는 스탬프 형식으로 옮겨 세 판정을 다시 돌린다.

  judge_runtime_evidence  실행 코드·시드·설정·사전등록·산출물 해시의 재현 사슬
  audit_dashboard         Dashboard row ↔ spec ↔ 증거 경로 ↔ push 된 commit
  judge_claim_alignment   report.md 에 사람이 옮겨적은 수치 ↔ 원자료 계산값
  → combine_reliability   위 셋을 신뢰성 게이트 하나로 결합

성능·현실성 축은 이번 보정 범위가 아니므로 저장된 판정을 **그대로 승계**한다
(단일축 보정 원칙 — 인가받지 않은 축을 같이 올리지 않는다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .gate_harness import (ResearchGateReport, audit_dashboard, combine_reliability,
                           judge_claim_alignment, judge_runtime_evidence, report_from_dict)

ROOT = Path(".")
CONTRACT = Path("outputs/reports/yr151_pre_gate_0a/contract_0a.json")
REPORT_MD = Path("outputs/reports/yr151_pre_gate_0a/report.md")
GATE = Path("outputs/reports/yr153_research_gates/current_gate.json")
SPEC = Path(".claude/docs/dashboard-task-specs/YR-151-block-ppo-sell-head.md")
# 구현 동결 · 판정 · spec 정합 · board 이동(BOARD_COMMIT 은 실행 시 인자로 덮어씀)
EVIDENCE_COMMITS = ["87d5688", "ae6005b", "0e6727c"]
REMOTE_REF = "origin/master"
# board 상태는 실행 시 인자로 받는다 — 저장된 신뢰성 PASS 는 board 상태에 고정되므로
# YR-151 row 가 이동하면(ready→backlog 등) 이 보고를 반드시 다시 돌려야 재계산이 통과한다.
BOARD_STATE_DEFAULT = "backlog"     # 2026-08-06 사용자 정정: 0B 는 YR-150 선결 대기

# report.md 본문에 사람이 옮겨적은 수치 — 원자료와 기계 대조할 대상.
# (표 "계약 8종 (20/20 셀 통과)" · "후보 수: 셀별 1~6건(20셀)")
REPORTED = {
    "n_cells": 20.0, "n_cells_exercised": 20.0,
    "C1_pass": 20.0, "C2_pass": 20.0, "C3_pass": 20.0, "C4_pass": 20.0,
    "C5_pass": 20.0, "C6_pass": 20.0, "C7_pass": 20.0, "C8_pass": 20.0,
    "min_candidates_A": 1.0, "max_candidates_A": 6.0,
}
_KEYS = ("C1_window_and_pre_gate", "C2_no_future_leak", "C3_atomic_commit",
         "C4_gate_in_invariant", "C5_transfer_cap", "C6_stale_rejected",
         "C7_rollback_restore", "C8_ledger_registration")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def raw_values(data: dict) -> dict[str, float]:
    """report.md 수치의 대조군을 원자료에서 직접 다시 계산한다."""
    cand = [r["n_candidates"]["A"] for r in data["cells"]]
    out = {"n_cells": float(data["verdict"]["n_cells"]),
           "n_cells_exercised": float(data["verdict"]["n_cells_exercised"]),
           "min_candidates_A": float(min(cand)), "max_candidates_A": float(max(cand))}
    for i, key in enumerate(_KEYS, start=1):
        out[f"C{i}_pass"] = float(data["summary"][key]["pass"])
    return out


def build(generated_at: str, board_commit: str,
          board_state: str = BOARD_STATE_DEFAULT) -> dict:
    commits = (*EVIDENCE_COMMITS, board_commit)
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rt = data["runtime"]

    # 하네스 스탬프 형식으로 이관. 산출물 해시는 **커밋된 파일의 실제 해시**를 쓴다 —
    # 원자료의 self_sha256 은 그 필드를 써넣기 전 파일의 해시라 자기검증이 불가능하다
    # (결함 기록: 아래 self_sha256_is_pre_write).
    stamp = {"code": {"git_head": rt["commit"], "git_dirty": rt["git_dirty"]},
             "seeds": rt["seeds"], "prereg": rt["prereg_file"], "params": rt["params"]}
    hashes = {str(CONTRACT.as_posix()): _sha256(CONTRACT)}

    runtime = judge_runtime_evidence(stamp, artifact_hashes=hashes, root=ROOT)
    dashboard = audit_dashboard(
        ROOT, task_id="YR-151", expected_state=board_state, spec_path=SPEC,
        evidence_paths=(CONTRACT, REPORT_MD), evidence_commits=commits,
        remote_ref=REMOTE_REF)
    alignment = judge_claim_alignment(REPORTED, raw_values(data))
    reliability = combine_reliability(runtime, dashboard, alignment)

    prior = report_from_dict(json.loads(GATE.read_text(encoding="utf-8")))
    report = ResearchGateReport(prior.performance, reliability, prior.scenario_validity)

    out = report.as_dict()
    out["generated_at"] = generated_at
    out["reliability"]["evidence"].update({
        "task": "YR-151 0A",
        "contract_all_pass": data["verdict"]["contract_all_pass"],
        "artifact_sha256": hashes,
        "self_sha256_is_pre_write": {
            "recorded": data.get("self_sha256"), "actual_file": hashes[CONTRACT.as_posix()],
            "note": "원자료의 self_sha256 은 그 필드를 덧쓰기 전 파일의 해시다 — "
                    "자기검증 불가. 게이트는 실제 파일 해시로 고정한다(후속 정정 대상)."},
        "prior_fail_reasons_closed": {
            "runtime_git_dirty": f"{rt['git_dirty']} (clean commit {rt['commit'][:7]})",
            "runtime_params": rt["params"],
            "runtime_prereg": f"{rt['prereg_file']} (sha256 {str(rt['prereg_sha256'])[:12]}…)"},
        "carried_over_note": "YR-149 의 신뢰성 FAIL 3사유를 0A 산출물이 직접 닫음"})
    out["currently_authorized"] = (
        "reliability PASS → YR-150: scenario_validity 단일축"
        if reliability.status.value == "PASS" else "YR-151 0A: reliability 단일축(미해소)")
    out["conditional_sequence"] = [
        "scenario_validity PASS 후 YR-151 0B: performance 단일축(학습 GO/STOP 은 0B 에서만)"]
    out["forbidden_next_scope"] = "위 미확정과 무관한 새 상태·보상·행동·가설 추가"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-at", required=True)
    ap.add_argument("--board-commit", required=True, help="현재 board·spec 상태의 commit")
    ap.add_argument("--board-state", default=BOARD_STATE_DEFAULT,
                    help="YR-151 row 가 있는 Dashboard 상태 파일 이름")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    out = build(a.generated_at, a.board_commit, a.board_state)
    if a.write:
        GATE.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"status": v["status"], "reasons": v["reasons"]}
                      for k, v in out.items() if isinstance(v, dict) and "status" in v}
                     | {"next_work_mode": out["next_work_mode"],
                        "unresolved_gates": out["unresolved_gates"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
