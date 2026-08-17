"""YR-156 — 게이트 증거를 commit 시점 내용에 고정한다.

구판은 board·spec 을 **현재 디스크 파일**로 읽고 "지정 commit 이후 한 글자도
안 바뀌었을 것"까지 요구했다. board 는 살아 있는 문서라, 판정과 무관한 정당한
편집(row 를 backlog→in-progress→done 으로 옮기는 등)만으로 과거 PASS 가 무효가
됐다(2026-08-06 실측 2회·2026-08-17 재발).

아래는 **오늘 실제로 일어난 사례**를 그대로 회귀로 박은 것이다.
"""
from __future__ import annotations

import subprocess

import pytest

from yard_rl.experiments.gate_harness import GateStatus, audit_dashboard

PIN = "3364e5a"          # YR-181 을 backlog 에 등록한 커밋
SPEC = ".claude/docs/dashboard-task-specs/YR-181-yr171bc-dirty-tree-rerun.md"


def _has_commit(repo=".") -> bool:
    try:
        r = subprocess.run(["git", "cat-file", "-e", f"{PIN}^{{commit}}"],
                           cwd=repo, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not _has_commit(), reason=f"회귀 기준 commit {PIN} 이 없는 클론")


def _kw(state="backlog"):
    return dict(task_id="YR-181", expected_state=state, spec_path=SPEC,
                evidence_paths=(), evidence_commits=(PIN,),
                remote_ref="origin/master", require_final_evidence=False)


def test_pin_commit_survives_legitimate_board_edits():
    """그 시점 내용으로 보면 통과하고, 이후 변경은 drift 로 남는다."""
    out = audit_dashboard(".", **_kw(), pin_commit=PIN)
    assert out.status is GateStatus.PASS, out.reasons
    drift = out.evidence["drift"]
    assert ".claude/Dashboard/backlog.md" in drift
    assert SPEC in drift                      # 실패가 아니라 기록이다


def test_without_pin_the_same_case_fails():
    """구 동작 보존 — pin 을 안 주면 현재 파일을 보므로 실패한다."""
    out = audit_dashboard(".", **_kw())
    assert out.status is GateStatus.FAIL
    assert out.evidence["pin_commit"] is None
    assert out.evidence["drift"] == []


@pytest.mark.parametrize("wrong", ["done", "ready", "in-progress"])
def test_pin_commit_still_detects_real_mismatch(wrong):
    """탐지력 — 그 시점의 실제 상태가 아닌 값을 우기면 여전히 잡는다.

    완화가 아니라 정정임을 보장하는 조항이다.
    """
    out = audit_dashboard(".", **_kw(wrong), pin_commit=PIN)
    assert out.status is GateStatus.FAIL
