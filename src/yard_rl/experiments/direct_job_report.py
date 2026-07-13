"""YR-027 외부트럭 Direct-Job Cost-Q 결과 보고서.

입력 payload는 JSON 직렬화 가능한 dict이다. runner와 느슨하게 결합하기 위해
누락 필드는 ``—``로 표시하고, ``test_rows``와 ``test.rows``를 모두 받는다.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

REPORT_NAME = "exp1_direct_costq_report.md"
RESULTS_NAME = "exp1_direct_costq_results.json"
_ARMS = ("SLA_OFF", "SLA_ON")
_MAX_RESULT_ROWS = 50


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(source: Mapping[str, Any], *keys: str) -> float | None:
    nested = (_mapping(source.get("metrics")), _mapping(source.get("summary")), source)
    for item in nested:
        for key in keys:
            value = item.get(key)
            if isinstance(value, Real) and not isinstance(value, bool):
                return float(value)
    return None


def _ratio(source: Mapping[str, Any], direct: tuple[str, ...],
           numerator: tuple[str, ...], denominator: tuple[str, ...]) -> float | None:
    value = _number(source, *direct)
    if value is not None:
        return value / 100.0 if value > 1.0 else value
    num, den = _number(source, *numerator), _number(source, *denominator)
    return num / den if num is not None and den and den > 0 else None


def _raw_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw: Any = payload.get("test_rows")
    if raw is None:
        test = payload.get("test")
        raw = _mapping(test).get("rows", test)
    if raw is None:
        raw = payload.get("results", [])
    if not raw:
        raw = payload.get("summary", [])
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, Mapping)]
    if not isinstance(raw, Mapping):
        return []
    rows: list[Mapping[str, Any]] = []
    for outer, value in raw.items():
        if isinstance(value, list):
            rows.extend({**row, "policy": row.get("policy", outer)}
                        for row in value if isinstance(row, Mapping))
        elif isinstance(value, Mapping) and any(isinstance(v, Mapping) for v in value.values()):
            rows.extend({**row, "arm": row.get("arm", outer), "policy": row.get("policy", name)}
                        for name, row in value.items() if isinstance(row, Mapping))
        elif isinstance(value, Mapping):
            rows.append({**value, "policy": value.get("policy", outer)})
    return rows


def _aggregate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _raw_rows(payload):
        arm = str(row.get("arm") or row.get("sla_arm") or row.get("sla_mode") or "UNKNOWN")
        policy = str(row.get("policy") or row.get("policy_name") or row.get("name") or "UNKNOWN")
        groups[(arm, policy)].append(row)

    metric_keys = {
        "mean": ("mean_wait_min", "mean_wait", "avg_wait_min"),
        "p50": ("p50_wait_min", "median_wait_min", "p50_wait"),
        "p95": ("p95_wait_min", "p95_wait"),
        "backlog": ("backlog", "unfinished_backlog"),
    }
    out: list[dict[str, Any]] = []
    for (arm, policy), rows in groups.items():
        item: dict[str, Any] = {"arm": arm, "policy": policy}
        for label, keys in metric_keys.items():
            values = [value for row in rows if (value := _number(row, *keys)) is not None]
            item[label] = sum(values) / len(values) if values else None
        rates = {
            "sla": [(_ratio(row, ("sla_rate", "sla_over_rate", "sla_exceed_rate", "sla_violation_rate"),
                            ("sla_exceed_count",), ("n_config", "configured_external_jobs"))) for row in rows],
            "completion": [(_ratio(row, ("completion_rate", "completed_rate"),
                                   ("completed_external",), ("n_config", "configured_external_jobs"))) for row in rows],
            "fallback": [(_ratio(row, ("fallback_rate", "fallback_decision_rate"),
                                 ("fallback_decisions",), ("n_decisions", "decision_count"))) for row in rows],
        }
        for label, values in rates.items():
            present = [value for value in values if value is not None]
            item[label] = sum(present) / len(present) if present else None
        out.append(item)
    arm_rank = {name: i for i, name in enumerate(_ARMS)}
    return sorted(out, key=lambda row: (arm_rank.get(row["arm"], 99), row["arm"], row["policy"]))


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    number = value * 100.0
    return f"{number:+.1f}%" if signed else f"{number:.1f}%"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _selected_baseline(selection: Mapping[str, Any], arm: str) -> Any:
    arm_node = _mapping(_mapping(selection.get("arms")).get(arm) or selection.get(arm))
    for key in ("validation_selected_baseline", "selected_baseline", "best_baseline", "baseline"):
        if arm_node.get(key) is not None:
            value = arm_node[key]
            return _mapping(value).get("policy", value)
        value = selection.get(key)
        if isinstance(value, Mapping) and value.get(arm) is not None:
            selected = value[arm]
            return _mapping(selected).get("policy", selected)
    return "—"


def _selected_cost_q(selection: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    arm_node = _mapping(_mapping(selection.get("arms")).get(arm) or selection.get(arm))
    return _mapping(arm_node.get("cost_q") or arm_node.get("selected_cost_q"))


def _ci(node: Mapping[str, Any], prefix: str = "") -> tuple[float | None, float | None]:
    packed = node.get(f"{prefix}ci") or node.get(f"{prefix}bootstrap_ci")
    if isinstance(packed, (list, tuple)) and len(packed) >= 2:
        return float(packed[0]), float(packed[1])
    if isinstance(packed, Mapping):
        return _number(packed, "lo", "low", "lower"), _number(packed, "hi", "high", "upper")
    return (_number(node, f"{prefix}ci_lo", f"{prefix}ci_low", f"{prefix}lower"),
            _number(node, f"{prefix}ci_hi", f"{prefix}ci_high", f"{prefix}upper"))


def _primary_stats(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    summary = _mapping(payload.get("summary"))
    root = _mapping(payload.get("paired_statistics") or payload.get("paired_stats") or payload.get("paired") or
                    summary.get("paired_stats") or summary.get("primary_paired"))
    primary = _mapping(root.get("primary") or root.get("SLA_OFF") or root)
    mean = _mapping(primary.get("mean_wait_min") or primary.get("mean_wait") or primary)
    p95 = _mapping(primary.get("p95_percent_change") or primary.get("p95_wait_min") or
                   primary.get("p95_wait") or primary.get("p95_percent") or primary)
    return mean, p95


def _coverage_label(rate: float | None) -> str:
    if rate is None:
        return "판정 불가(값 없음)"
    if abs(rate) < 1e-12:
        return "pure Cost-Q"
    if rate <= 0.05 + 1e-12:
        return "hybrid Cost-Q + fallback"
    return "coverage insufficient"


def write_direct_job_payload(payload: Mapping[str, Any], out_dir: str | Path,
                             filename: str = RESULTS_NAME) -> Path:
    """보고서 입력 원자료를 UTF-8 JSON으로 보존한다."""
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def build_direct_job_report(payload: Mapping[str, Any], out_dir: str | Path,
                            filename: str = REPORT_NAME) -> Path:
    """방어적으로 YR-027 Markdown 보고서를 만들고 그 경로를 반환한다."""
    manifest = _mapping(payload.get("manifest"))
    selection = _mapping(payload.get("selection"))
    summary = _mapping(payload.get("summary"))
    acceptance = _mapping(payload.get("acceptance"))
    scenario = _mapping(manifest.get("scenario") or manifest.get("gen_params"))
    mode = str(manifest.get("mode") or manifest.get("run_mode") or "unspecified")
    quick = bool(acceptance.get("quick") or manifest.get("quick_run") or
                 summary.get("quick_run") or mode.lower() == "quick")
    rows = _aggregate_rows(payload)
    display_rows = rows[:_MAX_RESULT_ROWS]
    mean_stats, p95_stats = _primary_stats(payload)
    mean_lo, mean_hi = _ci(mean_stats, "difference_")
    if mean_lo is None and mean_hi is None:
        mean_lo, mean_hi = _ci(mean_stats)
    p95_lo, p95_hi = _ci(p95_stats, "percent_change_")
    if p95_lo is None and p95_hi is None:
        p95_lo, p95_hi = _ci(p95_stats, "pct_")
    if p95_lo is None and p95_hi is None:
        p95_lo, p95_hi = _ci(p95_stats)
    mean_diff = _number(mean_stats, "estimate", "mean_diff", "diff_mean", "difference")
    p95_pct = _number(p95_stats, "estimate", "pct_change", "mean_pct_change", "percent_change")

    lines = [
        "# Exp-1 외부트럭 Direct-Job Cost-Q 결과",
        "",
        "> ⚠ **assumed 프로파일 + 합성 시나리오** 결과다. 실측 도착·서비스 자료와",
        "> CURRENT_RULE 검증 전이므로 실제 부산항 운영 대비 개선율로 해석할 수 없다.",
        "",
        "## 실험 정의",
        "",
        "- 대상: 외부트럭-only, 선박 작업 제외(`n_vessel=0`).",
        "- 의사결정 시점: 실제 블록 진입인 `BLOCK_ENTRY`.",
        "- 정책: 직접 feasible job 선택형 Cost-Q(`argmin`); `SLA_OFF`가 primary, `SLA_ON`이 secondary arm.",
        f"- 실행 모드: `{_cell(mode)}`; manifest n_vessel: `{_cell(scenario.get('n_vessel', manifest.get('n_vessel', '—')))}`.",
        "",
        "## Validation 선택",
        "",
    ]
    for arm in _ARMS:
        selected_cost = _selected_cost_q(selection, arm)
        p_value = selected_cost.get("p", "—")
        episode = selected_cost.get("checkpoint_episode", "—")
        validation_wait = _number(selected_cost, "validation_mean_wait_min")
        lines.append(
            f"- `{arm}`: Cost-Q `p={_cell(p_value)}`, checkpoint `{_cell(episode)}` "
            f"(validation mean {_fmt(validation_wait)} min); selected baseline "
            f"`{_cell(_selected_baseline(selection, arm))}`."
        )
    lines += [
        "",
        "## Test 요약 (공통 test seed)",
        "",
        "| Arm | Policy | Mean wait (min) | P50 (min) | P95 (min) | SLA 초과율 | Completion | Backlog | Fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in display_rows:
        lines.append(
            f"| {_cell(row['arm'])} | {_cell(row['policy'])} | {_fmt(row['mean'])} | "
            f"{_fmt(row['p50'])} | {_fmt(row['p95'])} | {_pct(row['sla'])} | "
            f"{_pct(row['completion'])} | {_fmt(row['backlog'])} | {_pct(row['fallback'])} |")
    if not rows:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    if len(rows) > _MAX_RESULT_ROWS:
        lines.append(f"\n> 행 수 제한으로 {len(rows) - _MAX_RESULT_ROWS}개 정책 행을 생략했다.")

    paired = _mapping(payload.get("paired_statistics") or payload.get("paired_stats") or payload.get("paired"))
    primary_arm = str(acceptance.get("primary_arm") or "SLA_OFF")
    primary = _mapping(paired.get("primary") or paired.get(primary_arm) or paired)
    baseline = primary.get("baseline") or _selected_baseline(selection, "SLA_OFF")
    baseline = _mapping(baseline).get("policy", baseline)
    alternative = (primary.get("alternative") or primary.get("policy")
                   or "CostQ+GreedyFallback")
    lines += [
        "",
        f"## Primary paired bootstrap (`{_cell(primary_arm)}`)",
        "",
        f"- 비교: `{_cell(alternative)}` − validation-selected baseline `{_cell(baseline)}`.",
        f"- Mean wait 차이: {_fmt(mean_diff, signed=True)} min (95% bootstrap CI "
        f"[{_fmt(mean_lo, signed=True)}, {_fmt(mean_hi, signed=True)}]).",
        f"- P95 wait 변화: {_fmt(p95_pct, signed=True)}% (95% bootstrap percent CI "
        f"[{_fmt(p95_lo, signed=True)}%, {_fmt(p95_hi, signed=True)}%]).",
        "",
        "## Fallback coverage 해석",
        "",
        "- 0% fallback은 **pure Cost-Q**, 0% 초과 5% 이하는 **hybrid Cost-Q + fallback**, 5% 초과는 **coverage insufficient**다.",
    ]
    costq_rows = [row for row in display_rows
                  if "cost" in row["policy"].lower() and "q" in row["policy"].lower()]
    for row in costq_rows:
        lines.append(f"- `{row['arm']}/{_cell(row['policy'])}`: {_pct(row['fallback'])} — {_coverage_label(row['fallback'])}.")
    if not costq_rows:
        lines.append("- Cost-Q fallback 값이 없어 coverage 판정을 보류한다.")
    if acceptance.get("coverage_class") is not None:
        lines.append(f"- runner primary coverage class: `{_cell(acceptance['coverage_class'])}`.")

    verdict = acceptance.get("decision")
    if verdict is None:
        verdict = acceptance.get("overall")
    if verdict is None:
        verdict = summary.get("acceptance_verdict") or summary.get("verdict")
    lines += ["", "## 판정과 한계", ""]
    if quick:
        lines.append("- **Quick run은 배선·동작 점검용이므로 합격/불합격 판정을 금지한다.**")
    else:
        lines.append(f"- Full run 판정: `{_cell(verdict if verdict is not None else 'runner 판정 미제공')}`.")
        if acceptance:
            lines.append(
                "- 기준별 결과: 평균대기 개선 "
                f"`{_cell(acceptance.get('mean_improved', '—'))}`, P95 guardrail "
                f"`{_cell(acceptance.get('p95_guardrail', '—'))}`, completion "
                f"`{_cell(acceptance.get('completion_ok', '—'))}`, backlog "
                f"`{_cell(acceptance.get('backlog_ok', '—'))}`, coverage "
                f"`{_cell(acceptance.get('coverage_ok', '—'))}`."
            )
    lines += [
        "- assumed 터미널 프로파일과 합성 외부트럭 흐름의 상대 비교만 가능하다.",
        "- 선박·본선 deadline·실제 운영 배차 규칙은 이 Exp-1의 범위 밖이다.",
        "- paired 수치, completion/backlog, fallback coverage를 함께 보지 않은 단일 평균 비교는 금지한다.",
        "",
        "*원자료: exp1_direct_results.json*",
    ]
    if len(lines) > 200:
        raise ValueError(f"report exceeds 200 lines: {len(lines)}")
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
