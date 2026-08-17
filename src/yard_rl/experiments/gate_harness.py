"""연구 진행 3대 게이트 하네스 (YR-153).

다음 연구 단계로 넘어가기 전에 세 질문을 기계적으로 확인한다.

1. 성능: 규칙 기준선보다 운영비용이 실제로 개선됐는가?
2. 신뢰성: 실행 코드·재현정보·Dashboard·증거가 서로 일치하는가?
3. 현실성: 생성 시나리오가 주장 범위에 필요한 물리·정보·흐름 계약을 지키는가?

실패 결과도 연구 증거이므로 결과 파일은 먼저 저장한다. 이 모듈은 실패한 실험을
예외로 지우지 않고, **그 다음에 허용되는 작업의 범위만** 제한한다.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping, Sequence

from ..integrated.evalkit import PairedResult


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ClaimScope(str, Enum):
    """결과가 허용하는 주장 범위."""

    NONE = "NO_PERFORMANCE_CLAIM"
    SIMULATION_METHOD_ONLY = "LITERATURE_CALIBRATED_SIMULATION_METHOD_ONLY"
    REAL_TERMINAL = "REAL_TERMINAL_OPERATION"


class WorkKind(str, Enum):
    """다음 작업의 목적. 새 가설과 결함 보정을 구분한다."""

    OBJECTIVE = "OBJECTIVE"
    REMEDIATION = "REMEDIATION"
    NEW_HYPOTHESIS = "NEW_HYPOTHESIS"


@dataclass(frozen=True)
class GateOutcome:
    name: str
    status: GateStatus
    summary: str
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class AnchorEvidence:
    """문헌·공개자료 범위와 시뮬레이션 범위의 기계 대조 1건."""

    observed_min: float
    observed_max: float
    simulated_min: float
    simulated_max: float
    unit: str
    source_path: str
    source_sha256: str

    def as_dict(self) -> dict:
        return {
            "observed_range": [self.observed_min, self.observed_max],
            "simulated_range": [self.simulated_min, self.simulated_max],
            "unit": self.unit,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }


def judge_performance(
    result: PairedResult,
    *,
    minimum_improvement: float = 0.0,
    interest_effect: float,
    hard_guards: Mapping[str, bool],
) -> GateOutcome:
    """원정밀도 짝지은 CI로 성능 개선을 판정한다.

    `result`의 차이는 후보−기준선이다. 방향을 개선량으로 바꾼 뒤, 신뢰구간 전체가
    사전 최소개선량을 넘어야 PASS다. 반올림 값은 표시할 때만 쓴다.
    """
    if minimum_improvement < 0 or interest_effect <= 0:
        raise ValueError("minimum_improvement는 0 이상, interest_effect는 0 초과여야 함")
    required_guards = {"completion", "backlog_zero", "physical_valid", "vessel_protection"}
    missing_guards = sorted(required_guards - set(hard_guards))
    failed_guards = sorted(name for name, ok in hard_guards.items() if not ok)
    missing_or_failed = [f"미수집:{name}" for name in missing_guards] + failed_guards
    improvement = -result.mean
    improvement_lo, improvement_hi = -result.ci_hi, -result.ci_lo
    power_adequate = result.mde80 <= interest_effect

    evidence = {
        "metric": "terminal_total_cost",
        "baseline": "SF-SPT",
        "n": result.n,
        "candidate_minus_baseline": result.mean,
        "candidate_minus_baseline_ci95_raw": [result.ci_lo, result.ci_hi],
        "improvement": improvement,
        "improvement_ci95_raw": [improvement_lo, improvement_hi],
        "minimum_improvement": minimum_improvement,
        "interest_effect": interest_effect,
        "mde80": result.mde80,
        "power_adequate": power_adequate,
        "hard_guards": dict(hard_guards),
    }
    if missing_or_failed:
        return GateOutcome(
            "performance",
            GateStatus.FAIL,
            "성능 수치와 무관하게 안전·완주·보호 조건을 위반함",
            tuple(f"하드 가드 실패: {name}" for name in missing_or_failed),
            evidence,
        )
    if not power_adequate:
        return GateOutcome(
            "performance",
            GateStatus.INCONCLUSIVE,
            "현재 표본으로 목표 크기의 개선을 판별할 수 없음",
            ("사전 관심효과보다 최소검출가능효과가 큼",),
            evidence,
        )
    if improvement_lo > minimum_improvement:
        return GateOutcome(
            "performance",
            GateStatus.PASS,
            "규칙 기준선 대비 운영적으로 의미 있는 개선이 확인됨",
            evidence=evidence,
        )
    if improvement_hi < minimum_improvement:
        return GateOutcome(
            "performance",
            GateStatus.FAIL,
            "사전 최소개선량에 미달함이 충분한 표본에서 확인됨",
            evidence=evidence,
        )
    return GateOutcome(
        "performance",
        GateStatus.INCONCLUSIVE,
        "개선 가능성과 미달 가능성이 신뢰구간에 함께 남아 있음",
        ("고정된 확증 표본 또는 잠금평가가 필요함",),
        evidence,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> Mapping[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _anchor_source_error(path: Path, name: str, anchor: AnchorEvidence) -> str | None:
    """근거 registry가 실제 지표·단위·관측범위·출처를 담았는지 확인한다."""
    payload = _read_json_object(path)
    if payload is None or payload.get("schema") != "yard_rl.external_anchor.v1":
        return "근거 JSON schema 오류"
    records = payload.get("anchors")
    record = records.get(name) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        return "근거 JSON에 해당 지표 없음"
    if record.get("metric") != name:
        return "근거 지표명 불일치"
    if not isinstance(anchor.unit, str) or not anchor.unit.strip() or record.get("unit") != anchor.unit:
        return "근거 단위 불일치"
    observed = record.get("observed_range")
    if not isinstance(observed, (list, tuple)) or len(observed) != 2:
        return "근거 관측범위 누락"
    try:
        observed_values = (float(observed[0]), float(observed[1]))
    except (TypeError, ValueError):
        return "근거 관측범위 비수치"
    if observed_values != (float(anchor.observed_min), float(anchor.observed_max)):
        return "근거 관측범위 불일치"
    citation = record.get("source")
    if not isinstance(citation, Mapping):
        return "근거 출처 누락"
    if not all(isinstance(citation.get(key), str) and citation[key].strip()
               for key in ("title", "locator")):
        return "근거 출처 제목·위치 누락"
    return None


_TRACE_REQUIRED_FIELDS = {
    "job_id", "job_type", "block_id", "gate_in_time_s", "block_in_time_s",
    "completion_time_s", "gate_out_time_s",
}


def _operational_trace_error(path: Path, *, minimum_records: int) -> tuple[str | None, int]:
    """익명 운영이력이 사건 필드·순서·최소 표본 계약을 만족하는지 검사한다."""
    payload = _read_json_object(path)
    if payload is None or payload.get("schema") != "yard_rl.operational_trace.v1":
        return "실제 운영 이력 JSON schema 오류", 0
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < minimum_records:
        return f"실제 운영 이력 표본 부족({minimum_records}건 미만)", len(records) if isinstance(records, list) else 0
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or not _TRACE_REQUIRED_FIELDS.issubset(record):
            return f"실제 운영 이력 필수 사건 필드 누락: row {index}", len(records)
        identity = record.get("job_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            return f"실제 운영 이력 job_id 오류: row {index}", len(records)
        seen.add(identity)
        if not all(isinstance(record.get(key), str) and record[key]
                   for key in ("job_type", "block_id")):
            return f"실제 운영 이력 작업유형·블록 누락: row {index}", len(records)
        try:
            times = tuple(float(record[key]) for key in (
                "gate_in_time_s", "block_in_time_s", "completion_time_s", "gate_out_time_s"
            ))
        except (TypeError, ValueError):
            return f"실제 운영 이력 사건시각 비수치: row {index}", len(records)
        if not all(math.isfinite(value) for value in times) or list(times) != sorted(times):
            return f"실제 운영 이력 사건순서 위반: row {index}", len(records)
    return None, len(records)


def judge_runtime_evidence(
    stamp: Mapping[str, object],
    *,
    artifact_hashes: Mapping[str, str],
    root: str | Path = ".",
) -> GateOutcome:
    """실험 실행 시점의 재현정보를 fail-closed로 검사한다."""
    reasons: list[str] = []
    repo = Path(root).resolve()
    code = stamp.get("code")
    if not isinstance(code, Mapping):
        reasons.append("code 재현 스탬프 없음")
        code = {}
    head = code.get("git_head")
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", head):
        reasons.append("git commit 미기록 또는 형식 오류")
    elif not _git_ok(repo, ["cat-file", "-e", f"{head}^{{commit}}"]):
        reasons.append("기록한 git commit이 저장소에 없음")
    if code.get("git_dirty") is not False:
        reasons.append("판정 실행 코드가 clean commit이 아님")
    seeds = stamp.get("seeds")
    valid_seeds = isinstance(seeds, Mapping) and bool(seeds)
    if valid_seeds:
        valid_seeds = all(
            isinstance(values, (list, tuple))
            and bool(values)
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in values)
            for values in seeds.values()
        )
    if not valid_seeds:
        reasons.append("절대 시드 목록 미기록")
    prereg = stamp.get("prereg")
    if not isinstance(prereg, str) or not prereg.strip():
        reasons.append("사전등록 경로 미기록")
    elif not (repo / prereg).exists():
        reasons.append("사전등록 파일이 존재하지 않음")
    if not isinstance(stamp.get("params"), Mapping) or not stamp.get("params"):
        reasons.append("실행 파라미터 전문 미기록")
    if not artifact_hashes or any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
        for digest in artifact_hashes.values()
    ):
        reasons.append("산출물 sha256 미기록")
    else:
        for artifact, expected in artifact_hashes.items():
            path = repo / artifact
            if not path.is_file():
                reasons.append(f"산출물 없음: {artifact}")
            elif _sha256(path).lower() != expected.lower():
                reasons.append(f"산출물 sha256 불일치: {artifact}")
    status = GateStatus.FAIL if reasons else GateStatus.PASS
    return GateOutcome(
        "runtime_evidence",
        status,
        "실행 코드·시드·설정·산출물의 재현 사슬이 완전함" if not reasons else "실행 재현 사슬이 불완전함",
        tuple(reasons),
        {
            "git_head": head,
            "stamp": dict(stamp),
            "artifact_hashes": dict(artifact_hashes),
        },
    )


_BOARD_FILES = ("in-progress.md", "ready.md", "backlog.md", "cancelled.md", "done.md")


def _git_ok(root: Path, args: Sequence[str]) -> bool:
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _show_at(repo: Path, commit: str, relative: str) -> str | None:
    """그 commit 시점의 파일 내용. 없으면 None (YR-156)."""
    try:
        proc = subprocess.run(["git", "show", f"{commit}:{relative}"],
                              cwd=repo, capture_output=True, text=True,
                              encoding="utf-8", timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def audit_dashboard(
    root: str | Path,
    *,
    task_id: str,
    expected_state: str,
    spec_path: str | Path,
    evidence_paths: Iterable[str | Path],
    evidence_commits: Iterable[str] = (),
    remote_ref: str | None = None,
    require_final_evidence: bool = True,
    pin_commit: str | None = None,
) -> GateOutcome:
    """commit 뒤 Dashboard↔spec↔증거를 읽기 전용으로 감사한다.

    ★YR-156 (2026-08-17): `pin_commit` 을 주면 board·spec 을 **그 commit 시점
    내용**으로 읽는다. 판정이 근거로 삼아야 할 것은 판정 당시의 board 이고,
    그 뒤 목록을 정리했다고 실험이 달라지지 않는다. 구판은 현재 디스크 파일을
    읽고 "지정 commit 이후 한 글자도 안 바뀌었을 것"까지 요구해서,
    판정과 무관한 정당한 편집(row 를 ready→backlog 이동 등)만으로 과거 PASS 가
    무효가 됐다(2026-08-06 실측 2회·2026-08-17 재발). 이후 변경은 실패가 아니라
    `drift` 로 기록한다. `pin_commit=None` 이면 구 동작 그대로다.
    """
    repo = Path(root).resolve()
    evidence_path_list = tuple(evidence_paths)
    dashboard = repo / ".claude" / "Dashboard"
    reasons: list[str] = []
    matches: list[str] = []
    drift: list[str] = []
    row_pattern = re.compile(rf"^\|\s*{re.escape(task_id)}\s*\|", re.MULTILINE)
    for name in _BOARD_FILES:
        rel = f".claude/Dashboard/{name}"
        if pin_commit is not None:
            text = _show_at(repo, pin_commit, rel)
            if text is None:
                reasons.append(f"{pin_commit} 시점에 Dashboard 상태 파일 없음: {name}")
                continue
        else:
            path = dashboard / name
            if not path.exists():
                reasons.append(f"Dashboard 상태 파일 없음: {name}")
                continue
            text = path.read_text(encoding="utf-8")
        matches.extend([name] * len(row_pattern.findall(text)))
    expected_file = f"{expected_state}.md"
    if matches != [expected_file]:
        reasons.append(f"Dashboard row는 {expected_file}에 정확히 1개여야 함: {matches}")

    spec_rel = Path(spec_path).as_posix()
    if pin_commit is not None:
        text = _show_at(repo, pin_commit, spec_rel)
        if text is None:
            reasons.append(f"{pin_commit} 시점에 spec 없음: {spec_path}")
            text = ""
    else:
        spec = repo / spec_path
        if not spec.exists():
            reasons.append(f"spec 없음: {spec_path}")
            text = ""
        else:
            text = spec.read_text(encoding="utf-8")
    if text:
        state_pattern = re.compile(
            rf"\*\*상태\*\*\s*:\s*(?:\*\*)?{re.escape(expected_state)}\b",
            re.IGNORECASE,
        )
        if not state_pattern.search(text):
            reasons.append(f"spec 상태가 {expected_state}와 일치하지 않음")

    missing_paths = [str(path) for path in evidence_path_list if not (repo / path).exists()]
    if require_final_evidence and not evidence_path_list:
        reasons.append("완료 evidence 경로가 비어 있음")
    reasons.extend(f"증거 경로 없음: {path}" for path in missing_paths)
    commits = tuple(evidence_commits)
    if require_final_evidence and not commits:
        reasons.append("완료 evidence commit이 비어 있음")
    if require_final_evidence and not remote_ref:
        reasons.append("원격 반영을 검사할 remote_ref가 없음")
    for commit in commits:
        if not _git_ok(repo, ["cat-file", "-e", f"{commit}^{{commit}}"]):
            reasons.append(f"증거 commit 없음: {commit}")
        elif remote_ref and not _git_ok(repo, ["merge-base", "--is-ancestor", commit, remote_ref]):
            reasons.append(f"증거 commit이 {remote_ref}에 push되지 않음: {commit}")
    control_paths = (Path(".claude") / "Dashboard" / expected_file, Path(spec_path))
    linked_path_list = tuple(dict.fromkeys((*evidence_path_list, *control_paths)))
    for evidence_path in linked_path_list:
        path = (repo / evidence_path).resolve()
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError:
            reasons.append(f"증거 경로가 저장소 밖임: {evidence_path}")
            continue
        if not path.is_file():
            continue
        linked = False
        changed_only = False
        for commit in commits:
            exists_at_commit = _git_ok(repo, ["cat-file", "-e", f"{commit}:{relative}"])
            if not exists_at_commit:
                continue
            if _git_ok(repo, ["diff", "--quiet", commit, "--", relative]):
                linked = True
                break
            changed_only = True
        # ★YR-156: 지정 commit 에 **존재**하면 근거로 충분하다. 그 뒤 파일이
        # 바뀐 것은 실패가 아니라 drift 다 — 판정은 그 시점 내용을 본다.
        if commits and not linked and changed_only and pin_commit is not None:
            drift.append(relative)
        elif commits and not linked:
            reasons.append(f"현재 감사 대상 파일이 지정 commit 어느 것에도 포함되지 않음: {evidence_path}")

    status = GateStatus.FAIL if reasons else GateStatus.PASS
    return GateOutcome(
        "dashboard_code_alignment",
        status,
        "Dashboard·spec·코드·증거가 일치함" if not reasons else "Dashboard·코드·증거 사이 불일치가 있음",
        tuple(reasons),
        {
            "task_id": task_id,
            "expected_state": expected_state,
            "board_matches": matches,
            "spec_path": str(spec_path),
            "evidence_paths": [str(path) for path in evidence_path_list],
            "control_paths": [str(path) for path in control_paths],
            "evidence_commits": list(commits),
            "remote_ref": remote_ref,
            "pin_commit": pin_commit,
            # ★YR-156 — 판정 이후 바뀐 감사 대상 파일. 실패가 아니라 기록이다.
            "drift": drift,
        },
    )


_INTERNAL_REQUIRED = {
    "event_time_order",
    "information_boundary",
    "physical_constraints",
    "ledger_conservation",
    "achievable_vessel_deadline",
    "deterministic_replay",
}
_CONTINUOUS_REQUIRED = {
    "continuous_arrivals",
    "warmup_excluded",
    "fixed_measurement_window",
    "load_state_classified",
    "flow_balance_consistent_with_classification",
}
_ANCHOR_REQUIRED = {
    "gate_to_block_time",
    "initial_yard_occupancy",
    "truck_arrival_rate",
    "crane_service_time",
    "vessel_workload",
}


def judge_claim_alignment(
    reported_values: Mapping[str, float],
    raw_values: Mapping[str, float],
    *,
    absolute_tolerance: float = 0.0,
) -> GateOutcome:
    """보고서·Dashboard 핵심 수치를 결과 JSON 원값과 대조한다."""
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance는 0 이상이어야 함")
    reasons: list[str] = []
    if not reported_values or not raw_values:
        reasons.append("대조할 핵심 수치가 비어 있음")
    if set(reported_values) != set(raw_values):
        missing_report = sorted(set(raw_values) - set(reported_values))
        missing_raw = sorted(set(reported_values) - set(raw_values))
        reasons.extend(f"보고 누락: {name}" for name in missing_report)
        reasons.extend(f"원자료 키 없음: {name}" for name in missing_raw)
    differences: dict[str, float] = {}
    for name in sorted(set(reported_values) & set(raw_values)):
        try:
            reported = float(reported_values[name])
            raw = float(raw_values[name])
        except (TypeError, ValueError) as exc:
            reasons.append(f"비수치 주장: {name}")
            continue
        if not math.isfinite(reported) or not math.isfinite(raw):
            reasons.append(f"비유한 수치 주장: {name}")
            continue
        difference = abs(reported - raw)
        differences[name] = difference
        if difference > absolute_tolerance:
            reasons.append(f"수치 불일치: {name} (절대차 {difference})")
    status = GateStatus.FAIL if reasons else GateStatus.PASS
    return GateOutcome(
        "claim_alignment",
        status,
        "보고 핵심 수치가 원자료와 일치함" if not reasons else "보고 수치와 원자료가 일치하지 않음",
        tuple(reasons),
        {
            "absolute_tolerance": absolute_tolerance,
            "reported_values": dict(reported_values),
            "raw_values": dict(raw_values),
            "absolute_differences": differences,
        },
    )


def judge_scenario_validity(
    *,
    internal_checks: Mapping[str, bool],
    flow_checks: Mapping[str, bool],
    anchors: Mapping[str, AnchorEvidence],
    continuous_operation: bool,
    request_real_terminal_claim: bool,
    operational_trace_path: str | None = None,
    operational_trace_sha256: str | None = None,
    minimum_operational_trace_records: int = 30,
    root: str | Path = ".",
) -> GateOutcome:
    """시드 데이터의 내부타당성·운영흐름·자료등급을 분리해 판정한다."""
    if minimum_operational_trace_records <= 0:
        raise ValueError("minimum_operational_trace_records는 1 이상이어야 함")
    reasons: list[str] = []
    missing_internal = sorted(_INTERNAL_REQUIRED - set(internal_checks))
    failed_internal = sorted(name for name, ok in internal_checks.items() if not ok)
    reasons.extend(f"내부타당성 미수집: {name}" for name in missing_internal)
    reasons.extend(f"내부타당성 실패: {name}" for name in failed_internal)

    if continuous_operation:
        missing_flow = sorted(_CONTINUOUS_REQUIRED - set(flow_checks))
        failed_flow = sorted(name for name, ok in flow_checks.items() if not ok)
        reasons.extend(f"지속운영 검사 미수집: {name}" for name in missing_flow)
        reasons.extend(f"지속운영 검사 실패: {name}" for name in failed_flow)
    else:
        reasons.append("지속 유입 운영흐름 미검증")
    missing_anchors = sorted(_ANCHOR_REQUIRED - set(anchors))
    reasons.extend(f"필수 외부 앵커 미수집: {name}" for name in missing_anchors)
    repo = Path(root).resolve()
    for name, anchor in anchors.items():
        if not isinstance(anchor, AnchorEvidence):
            reasons.append(f"외부 앵커 증거 형식 오류: {name}")
            continue
        values = (anchor.observed_min, anchor.observed_max,
                  anchor.simulated_min, anchor.simulated_max)
        if not all(math.isfinite(value) for value in values):
            reasons.append(f"외부 앵커 비유한 수치: {name}")
        elif anchor.observed_min > anchor.observed_max or anchor.simulated_min > anchor.simulated_max:
            reasons.append(f"외부 앵커 범위 역전: {name}")
        elif (anchor.simulated_min < anchor.observed_min
              or anchor.simulated_max > anchor.observed_max):
            reasons.append(f"시뮬레이션 범위가 외부 앵커 밖임: {name}")
        source = repo / anchor.source_path
        if not source.is_file():
            reasons.append(f"외부 앵커 근거 파일 없음: {name}")
        elif (not re.fullmatch(r"[0-9a-fA-F]{64}", anchor.source_sha256)
              or _sha256(source).lower() != anchor.source_sha256.lower()):
            reasons.append(f"외부 앵커 근거 sha256 불일치: {name}")
        else:
            source_error = _anchor_source_error(source, name, anchor)
            if source_error:
                reasons.append(f"외부 앵커 {source_error}: {name}")

    trace_valid = False
    trace_error: str | None = None
    trace_records = 0
    if operational_trace_path and operational_trace_sha256:
        trace = repo / operational_trace_path
        if not trace.is_file():
            trace_error = "실제 운영 이력 파일 없음"
        elif (not re.fullmatch(r"[0-9a-fA-F]{64}", operational_trace_sha256)
              or _sha256(trace).lower() != operational_trace_sha256.lower()):
            trace_error = "실제 운영 이력 sha256 불일치"
        else:
            trace_error, trace_records = _operational_trace_error(
                trace, minimum_records=minimum_operational_trace_records,
            )
            trace_valid = trace_error is None
    elif operational_trace_path or operational_trace_sha256:
        trace_error = "실제 운영 이력 경로와 sha256 중 하나가 누락됨"

    hard_failure = bool(reasons)
    if hard_failure:
        status = GateStatus.FAIL
        scope = ClaimScope.NONE
        summary = "시나리오가 주장에 필요한 물리·정보·흐름 계약을 충족하지 못함"
    elif request_real_terminal_claim and not trace_valid:
        status = GateStatus.INCONCLUSIVE
        scope = ClaimScope.SIMULATION_METHOD_ONLY
        summary = "문헌 보정 시뮬레이션에는 적합하지만 실제 터미널 성능 주장은 자료 부족"
        reasons.append(trace_error or "실제 운영 이력(Level 2~3 이상) 없음")
    else:
        status = GateStatus.PASS
        scope = ClaimScope.REAL_TERMINAL if request_real_terminal_claim else ClaimScope.SIMULATION_METHOD_ONLY
        summary = "요청한 주장 범위의 시나리오 타당성 계약을 충족함"
    return GateOutcome(
        "scenario_validity",
        status,
        summary,
        tuple(reasons),
        {
            "continuous_operation": continuous_operation,
            "internal_checks": dict(internal_checks),
            "flow_checks": dict(flow_checks),
            "anchors": {name: anchor.as_dict() for name, anchor in anchors.items()
                        if isinstance(anchor, AnchorEvidence)},
            "operational_trace_path": operational_trace_path,
            "operational_trace_sha256": operational_trace_sha256,
            "operational_trace_verified": trace_valid,
            "operational_trace_records": trace_records,
            "minimum_operational_trace_records": minimum_operational_trace_records,
            "allowed_claim_scope": scope.value,
        },
    )


def judge_referenced_weights(root: str | Path = ".") -> GateOutcome:
    """**코드가 여는 가중치가 버전관리에 있나** (YR-182, 2026-08-17 신설).

    이 검사가 없어서 오늘 실제로 막혔다 — 모든 평가가 쓰는 채택 크레인 정책
    체크포인트가 추적되지 않아 다른 작업트리에서 실행이 즉시 실패했다. 그런데
    `runtime`·`dashboard` 는 commit 의 원격 반영만 보므로 **PASS 를 줬다**.

    방법: `src/` 안의 `outputs/reports/...` 경로 리터럴을 모아 그 아래 디스크에
    실재하는 `.pt` 중 **추적되지 않은 것**을 센다. 학습 데이터셋(`dataset_*.pt`)은
    시드에서 재생성되는 파생물이라 제외한다(YR-182 원장 §policy).
    """
    repo = Path(root).resolve()
    src = repo / "src"
    if not src.is_dir():
        return GateOutcome("referenced_weights", GateStatus.INCONCLUSIVE,
                           "src 디렉터리를 찾지 못함", ("src 없음",))
    text = "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                     for f in src.rglob("*.py"))
    dirs = {m.split("/")[2] for m in
            re.findall(r"outputs/reports/[A-Za-z0-9_\-/]+", text)}
    # 추적 목록을 **한 번에** 받아 집합으로 대조한다. 파일명에 `[ddqn]` 처럼
    # 대괄호가 들어가면 pathspec 이 glob 으로 해석해 오탐이 난다(2026-08-17 실측:
    # 추적 중인 125개가 전부 미추적으로 잡혔다).
    listed = subprocess.run(["git", "ls-files", "-z", "--", "outputs/reports"],
                            cwd=repo, capture_output=True, text=True)
    if listed.returncode != 0:
        return GateOutcome("referenced_weights", GateStatus.INCONCLUSIVE,
                           "git ls-files 실패", ("추적 목록을 읽지 못함",))
    tracked = {p for p in listed.stdout.split("\0") if p}
    missing, checked = [], 0
    for d in sorted(dirs):
        base = repo / "outputs" / "reports" / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.pt")):
            if f.name.startswith("dataset"):
                continue          # 재생성 가능한 파생 데이터 — 원장으로만 관리
            checked += 1
            rel = f.relative_to(repo).as_posix()
            if rel not in tracked:
                missing.append(rel)
    if missing:
        head = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        return GateOutcome(
            "referenced_weights", GateStatus.FAIL,
            "코드가 여는 가중치 중 추적되지 않은 것이 있음 — 다른 환경에서 재현 불가",
            (f"미추적 가중치 {len(missing)}개: {head}",),
            {"n_checked": checked, "n_missing": len(missing), "missing": missing})
    return GateOutcome(
        "referenced_weights", GateStatus.PASS,
        "코드가 여는 가중치가 모두 추적됨", (),
        {"n_checked": checked, "n_missing": 0})


def combine_reliability(
    runtime: GateOutcome,
    dashboard: GateOutcome,
    claim_alignment: GateOutcome,
    weights: GateOutcome | None = None,
) -> GateOutcome:
    """실행 스탬프와 commit 뒤 감사를 하나의 신뢰성 게이트로 묶는다.

    ★YR-182(2026-08-17): `weights` 축 추가 — 코드가 여는 가중치가 버전관리에
    있는지. 기본값 None 은 기존 호출부를 바이트 불변으로 둔다.
    """
    outcomes = tuple(x for x in (runtime, dashboard, claim_alignment, weights)
                     if x is not None)
    if any(item.status is GateStatus.FAIL for item in outcomes):
        status = GateStatus.FAIL
    elif any(item.status is GateStatus.INCONCLUSIVE for item in outcomes):
        status = GateStatus.INCONCLUSIVE
    else:
        status = GateStatus.PASS
    return GateOutcome(
        "reliability",
        status,
        "실행 재현정보와 Dashboard 증거가 모두 일치함" if status is GateStatus.PASS else "신뢰성 증거가 미완 또는 불일치함",
        tuple(reason for item in outcomes for reason in item.reasons),
        {
            "runtime": runtime.as_dict(),
            "dashboard": dashboard.as_dict(),
            "claim_alignment": claim_alignment.as_dict(),
            **({"referenced_weights": weights.as_dict()} if weights else {}),
        },
    )


@dataclass(frozen=True)
class ResearchGateReport:
    performance: GateOutcome
    reliability: GateOutcome
    scenario_validity: GateOutcome

    @property
    def all_pass(self) -> bool:
        return all(
            outcome.status is GateStatus.PASS
            for outcome in (self.performance, self.reliability, self.scenario_validity)
        )

    @property
    def unresolved(self) -> tuple[str, ...]:
        return tuple(
            outcome.name
            for outcome in (self.performance, self.reliability, self.scenario_validity)
            if outcome.status is not GateStatus.PASS
        )

    @property
    def allowed_claim_scope(self) -> str:
        if not self.all_pass:
            return ClaimScope.NONE.value
        return str(self.scenario_validity.evidence.get("allowed_claim_scope", ClaimScope.NONE.value))

    def as_dict(self) -> dict:
        return {
            "performance": self.performance.as_dict(),
            "reliability": self.reliability.as_dict(),
            "scenario_validity": self.scenario_validity.as_dict(),
            "claim_eligible": self.all_pass,
            "allowed_claim_scope": self.allowed_claim_scope,
            "next_work_mode": "ADVANCE_TO_OBJECTIVE" if self.all_pass else "REMEDIATION_ONLY",
            "unresolved_gates": list(self.unresolved),
        }

    def authorize(self, kind: WorkKind, *, targets: Iterable[str] = ()) -> tuple[bool, str]:
        """새 작업이 사용자 결정의 진행 규칙을 지키는지 판정한다."""
        target_set = set(targets)
        unresolved = set(self.unresolved)
        if self.all_pass:
            if kind is WorkKind.OBJECTIVE:
                return True, "3대 게이트 통과 — 연구 목표의 다음 확증·잠금 단계 진행"
            return False, "3대 게이트가 통과했으므로 관련 없는 새 가설·진단 우회는 허용하지 않음"
        if kind is not WorkKind.REMEDIATION:
            return False, "미통과 게이트가 있어 그 원인을 직접 해결하는 작업만 허용"
        if len(target_set) != 1 or not target_set.issubset(unresolved):
            return False, f"보정 작업은 현재 미통과 게이트 중 정확히 한 축만 지정해야 함: {sorted(unresolved)}"
        remediation_order = ("reliability", "scenario_validity", "performance")
        first_unresolved = next(name for name in remediation_order if name in unresolved)
        target = next(iter(target_set))
        if target != first_unresolved:
            return False, f"선결 게이트 {first_unresolved}부터 해결해야 함 (요청: {target})"
        return True, "미통과 게이트를 직접 해결하는 단일축 보정 작업"


def attach_common_gates(payload: Mapping[str, object], report: ResearchGateReport) -> dict:
    """결과 JSON에 3대 게이트를 붙인다. 원자료는 상태와 무관하게 항상 보존한다."""
    out = dict(payload)
    out["common_gates"] = report.as_dict()
    return out


def report_from_dict(payload: Mapping[str, object]) -> ResearchGateReport:
    """저장된 current/common gate JSON을 다음 작업 허용 판정으로 복원한다."""
    source = payload.get("common_gates", payload)
    if not isinstance(source, Mapping):
        raise ValueError("gate JSON 형식 오류")

    def read(key: str, name: str) -> GateOutcome:
        raw = source.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"gate JSON에 {key} 없음")
        try:
            status = GateStatus(str(raw["status"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{key} status 오류") from exc
        reasons = raw.get("reasons", ())
        evidence = raw.get("evidence", {})
        if not isinstance(reasons, (list, tuple)) or not isinstance(evidence, Mapping):
            raise ValueError(f"{key} evidence 형식 오류")
        if status is GateStatus.PASS:
            required = {
                "performance": {
                    "metric", "baseline", "n", "candidate_minus_baseline_ci95_raw",
                    "minimum_improvement", "interest_effect", "mde80", "hard_guards",
                },
                "reliability": {"runtime", "dashboard", "claim_alignment"},
                "scenario_validity": {
                    "continuous_operation", "internal_checks", "flow_checks", "anchors",
                    "operational_trace_verified", "allowed_claim_scope",
                },
            }[key]
            missing = required - set(evidence)
            if missing:
                raise ValueError(f"{key} PASS evidence 누락: {sorted(missing)}")
        return GateOutcome(
            name,
            status,
            str(raw.get("summary", "")),
            tuple(str(reason) for reason in reasons),
            dict(evidence),
        )

    return ResearchGateReport(
        read("performance", "performance"),
        read("reliability", "reliability"),
        read("scenario_validity", "scenario_validity"),
    )


def verify_committed_artifact(
    root: str | Path,
    *,
    artifact_path: str | Path,
    artifact_sha256: str,
    commit: str,
    remote_ref: str,
) -> GateOutcome:
    """저장 gate 파일 자체가 원격 commit에 고정됐는지 확인한다."""
    repo = Path(root).resolve()
    reasons: list[str] = []
    path = (repo / artifact_path).resolve()
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError:
        relative = ""
        reasons.append("gate 파일이 저장소 밖임")
    if not path.is_file():
        reasons.append("gate 파일 없음")
    elif (not re.fullmatch(r"[0-9a-fA-F]{64}", artifact_sha256)
          or _sha256(path).lower() != artifact_sha256.lower()):
        reasons.append("gate 파일 sha256 불일치")
    if not _git_ok(repo, ["cat-file", "-e", f"{commit}^{{commit}}"]):
        reasons.append("gate commit 없음")
    elif not _git_ok(repo, ["merge-base", "--is-ancestor", commit, remote_ref]):
        reasons.append(f"gate commit이 {remote_ref}에 push되지 않음")
    elif relative:
        if not _git_ok(repo, ["cat-file", "-e", f"{commit}:{relative}"]):
            reasons.append("gate 파일이 지정 commit에 없음")
        elif not _git_ok(repo, ["diff", "--quiet", commit, "--", relative]):
            reasons.append("현재 gate 파일이 지정 commit 뒤 변경됨")
    return GateOutcome(
        "gate_artifact",
        GateStatus.FAIL if reasons else GateStatus.PASS,
        "gate 파일 신뢰 사슬 통과" if not reasons else "gate 파일 신뢰 사슬 실패",
        tuple(reasons),
        {
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "commit": commit,
            "remote_ref": remote_ref,
        },
    )


def _nested_evidence(raw: object, name: str) -> Mapping[str, object] | None:
    if not isinstance(raw, Mapping) or raw.get("status") != GateStatus.PASS.value:
        return None
    evidence = raw.get("evidence")
    return evidence if isinstance(evidence, Mapping) and evidence else None


def _revalidate_performance_pass(evidence: Mapping[str, object]) -> bool:
    try:
        ci = evidence["candidate_minus_baseline_ci95_raw"]
        guards = evidence["hard_guards"]
        return bool(
            evidence["metric"] == "terminal_total_cost"
            and evidence["baseline"] == "SF-SPT"
            and isinstance(evidence["n"], int) and evidence["n"] >= 2
            and isinstance(ci, (list, tuple)) and len(ci) == 2
            and all(math.isfinite(float(value)) for value in ci)
            and float(ci[0]) <= float(ci[1])
            and float(evidence["minimum_improvement"]) >= 0.0
            and float(evidence["interest_effect"]) > 0.0
            and float(evidence["mde80"]) <= float(evidence["interest_effect"])
            and isinstance(guards, Mapping)
            and all(guards.get(name) is True for name in (
                "completion", "backlog_zero", "physical_valid", "vessel_protection"
            ))
            and -float(ci[1]) > float(evidence["minimum_improvement"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def revalidate_pass_evidence(
    report: ResearchGateReport,
    *,
    root: str | Path,
) -> tuple[bool, str]:
    """저장 JSON의 PASS를 원자료 경로와 공용 판정기로 다시 계산한다."""
    repo = Path(root).resolve()
    if report.performance.status is GateStatus.PASS:
        if not _revalidate_performance_pass(report.performance.evidence):
            return False, "저장된 performance PASS evidence 재계산 실패"

    if report.reliability.status is GateStatus.PASS:
        evidence = report.reliability.evidence
        runtime = _nested_evidence(evidence.get("runtime"), "runtime")
        dashboard = _nested_evidence(evidence.get("dashboard"), "dashboard")
        alignment = _nested_evidence(evidence.get("claim_alignment"), "claim_alignment")
        if runtime is None or dashboard is None or alignment is None:
            return False, "저장된 reliability PASS 하위 evidence 누락"
        stamp = runtime.get("stamp")
        hashes = runtime.get("artifact_hashes")
        if (not isinstance(stamp, Mapping) or not isinstance(hashes, Mapping)
                or judge_runtime_evidence(stamp, artifact_hashes=hashes, root=repo).status
                is not GateStatus.PASS):
            return False, "저장된 runtime PASS 재계산 실패"
        try:
            dashboard_result = audit_dashboard(
                repo,
                task_id=str(dashboard["task_id"]),
                expected_state=str(dashboard["expected_state"]),
                spec_path=str(dashboard["spec_path"]),
                evidence_paths=dashboard["evidence_paths"],
                evidence_commits=dashboard["evidence_commits"],
                remote_ref=str(dashboard["remote_ref"]),
                # ★YR-156 — 저장된 판정과 **같은 시점**으로 다시 계산한다.
                # 없으면(구 판정) None 이라 구 동작 그대로다.
                pin_commit=dashboard.get("pin_commit"),
            )
        except (KeyError, TypeError):
            return False, "저장된 dashboard PASS evidence 형식 오류"
        if dashboard_result.status is not GateStatus.PASS:
            return False, "저장된 dashboard PASS 재계산 실패"
        reported = alignment.get("reported_values")
        raw = alignment.get("raw_values")
        try:
            alignment_result = judge_claim_alignment(
                reported if isinstance(reported, Mapping) else {},
                raw if isinstance(raw, Mapping) else {},
                absolute_tolerance=float(alignment.get("absolute_tolerance", 0.0)),
            )
        except (TypeError, ValueError):
            alignment_result = GateOutcome("claim_alignment", GateStatus.FAIL, "형식 오류")
        if (not isinstance(reported, Mapping) or not isinstance(raw, Mapping)
                or alignment_result.status is not GateStatus.PASS):
            return False, "저장된 claim-alignment PASS 재계산 실패"

    if report.scenario_validity.status is GateStatus.PASS:
        evidence = report.scenario_validity.evidence
        raw_anchors = evidence.get("anchors")
        if not isinstance(raw_anchors, Mapping):
            return False, "저장된 scenario PASS anchor evidence 누락"
        anchors: dict[str, AnchorEvidence] = {}
        try:
            internal_checks = evidence["internal_checks"]
            flow_checks = evidence["flow_checks"]
            if not isinstance(internal_checks, Mapping) or not isinstance(flow_checks, Mapping):
                raise TypeError
            for name, raw in raw_anchors.items():
                if not isinstance(raw, Mapping):
                    raise TypeError
                observed = raw["observed_range"]
                simulated = raw["simulated_range"]
                anchors[str(name)] = AnchorEvidence(
                    float(observed[0]), float(observed[1]),
                    float(simulated[0]), float(simulated[1]),
                    str(raw["unit"]), str(raw["source_path"]), str(raw["source_sha256"]),
                )
            scope = str(evidence["allowed_claim_scope"])
            result = judge_scenario_validity(
                internal_checks=internal_checks,
                flow_checks=flow_checks,
                anchors=anchors,
                continuous_operation=bool(evidence["continuous_operation"]),
                request_real_terminal_claim=scope == ClaimScope.REAL_TERMINAL.value,
                operational_trace_path=evidence.get("operational_trace_path"),
                operational_trace_sha256=evidence.get("operational_trace_sha256"),
                minimum_operational_trace_records=int(
                    evidence.get("minimum_operational_trace_records", 30)
                ),
                root=repo,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return False, "저장된 scenario PASS evidence 형식 오류"
        if result.status is not GateStatus.PASS or result.evidence["allowed_claim_scope"] != scope:
            return False, "저장된 scenario PASS 재계산 실패"
    return True, "저장된 PASS evidence 재계산 통과"


def authorize_task(
    report: ResearchGateReport,
    *,
    root: str | Path,
    task_id: str,
    spec_path: str | Path,
    kind: WorkKind,
    targets: Iterable[str] = (),
    stage: str | None = None,
) -> tuple[bool, str]:
    """게이트 상태뿐 아니라 Dashboard row와 spec의 보정 대상 선언까지 확인한다."""
    allowed, reason = report.authorize(kind, targets=targets)
    if not allowed:
        return False, reason
    repo = Path(root).resolve()
    row_pattern = re.compile(rf"^\|\s*{re.escape(task_id)}\s*\|", re.MULTILINE)
    row_count = 0
    for name in _BOARD_FILES:
        path = repo / ".claude" / "Dashboard" / name
        if path.exists():
            row_count += len(row_pattern.findall(path.read_text(encoding="utf-8")))
    if row_count != 1:
        return False, f"Dashboard에 {task_id} row가 정확히 1개가 아님: {row_count}"
    spec = repo / spec_path
    if not spec.is_file():
        return False, f"task spec 없음: {spec_path}"
    text = spec.read_text(encoding="utf-8")
    if task_id not in text.splitlines()[0]:
        return False, "task ID와 spec 제목이 일치하지 않음"
    if kind is WorkKind.REMEDIATION:
        target = next(iter(set(targets)))
        if stage:
            declared = re.search(
                rf"`?{re.escape(stage)}\s*=\s*{re.escape(target)}`?",
                text,
                re.IGNORECASE,
            )
        else:
            declared = any(
                "3대 게이트 보정 대상" in line and target in line
                for line in text.splitlines()
            )
        if not declared:
            label = f"{stage}={target}" if stage else target
            return False, f"spec에 3대 게이트 보정 대상 선언이 없음: {label}"
    return True, reason


def _cli_audit_dashboard(args: argparse.Namespace) -> int:
    outcome = audit_dashboard(
        args.root,
        task_id=args.task,
        expected_state=args.state,
        spec_path=args.spec,
        evidence_paths=args.evidence_path,
        evidence_commits=args.evidence_commit,
        remote_ref=args.remote_ref,
    )
    print(json.dumps(outcome.as_dict(), ensure_ascii=False, indent=2))
    return 0 if outcome.status is GateStatus.PASS else 2


def _cli_authorize_next(args: argparse.Namespace) -> int:
    artifact = verify_committed_artifact(
        args.root,
        artifact_path=args.gate_file,
        artifact_sha256=args.gate_sha256,
        commit=args.gate_commit,
        remote_ref=args.remote_ref,
    )
    if artifact.status is not GateStatus.PASS:
        print(json.dumps(artifact.as_dict(), ensure_ascii=False, indent=2))
        return 2
    try:
        payload = json.loads((Path(args.root) / args.gate_file).read_text(encoding="utf-8"))
        report = report_from_dict(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"allowed": False, "reason": f"gate JSON 검증 실패: {exc}"},
                         ensure_ascii=False, indent=2))
        return 2
    valid, validation_reason = revalidate_pass_evidence(report, root=args.root)
    if not valid:
        print(json.dumps({"allowed": False, "reason": validation_reason},
                         ensure_ascii=False, indent=2))
        return 2
    kind = WorkKind(args.kind)
    allowed, reason = authorize_task(
        report,
        root=args.root,
        task_id=args.task,
        spec_path=args.spec,
        kind=kind,
        targets=args.target,
        stage=args.stage,
    )
    print(json.dumps({
        "task": args.task,
        "allowed": allowed,
        "reason": reason,
        "kind": kind.value,
        "targets": list(args.target),
        "stage": args.stage,
        "current_unresolved": list(report.unresolved),
    }, ensure_ascii=False, indent=2))
    return 0 if allowed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="연구 진행 3대 게이트 하네스")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-dashboard", help="Dashboard·spec·evidence 정합 감사")
    audit.add_argument("--root", default=".")
    audit.add_argument("--task", required=True)
    audit.add_argument("--state", choices=[name[:-3] for name in _BOARD_FILES], required=True)
    audit.add_argument("--spec", required=True)
    audit.add_argument("--evidence-path", action="append", default=[])
    audit.add_argument("--evidence-commit", action="append", default=[])
    audit.add_argument("--remote-ref")
    audit.set_defaults(handler=_cli_audit_dashboard)
    authorize = sub.add_parser("authorize-next", help="저장된 3대 게이트로 다음 작업 착수 허용 검사")
    authorize.add_argument("--gate-file", required=True)
    authorize.add_argument("--gate-sha256", required=True)
    authorize.add_argument("--gate-commit", required=True)
    authorize.add_argument("--remote-ref", required=True)
    authorize.add_argument("--root", default=".")
    authorize.add_argument("--task", required=True)
    authorize.add_argument("--spec", required=True)
    authorize.add_argument("--stage")
    authorize.add_argument("--kind", choices=[kind.value for kind in WorkKind], required=True)
    authorize.add_argument("--target", action="append", default=[])
    authorize.set_defaults(handler=_cli_authorize_next)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
