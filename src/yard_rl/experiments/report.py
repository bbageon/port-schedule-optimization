"""Exp-1 예비 PoC 리포트 생성 — markdown + JSON 원자료.

표현 규칙(구현계획 03 §1.1): CURRENT_RULE 미확보 → '실제 운영 대비' 표현 금지.
모든 수치는 '가정 프로파일 + 합성 시나리오' 조건임을 머리말에 명시.
"""
from __future__ import annotations

import json
from pathlib import Path

from .runner import EpisodeResult
from .statistics import paired_diff

_KEY_METRICS = ["mean_wait_min", "p95_wait_min", "queue_area_h", "tail_area_h",
                "travel_km", "rehandles", "vessel_delay_min", "completed_external",
                "backlog", "sla_exceed_count"]


def _mean(rs: list[EpisodeResult], key: str) -> float:
    return sum(r.metrics[key] for r in rs) / len(rs)


def _series(rs: list[EpisodeResult], key: str) -> list[float]:
    return [r.metrics[key] for r in sorted(rs, key=lambda x: x.seed)]


def build_report(results: dict[str, list[EpisodeResult]], *, baseline: str,
                 meta: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp1_results.json").write_text(
        json.dumps({p: [r.__dict__ for r in rs] for p, rs in results.items()},
                   indent=1, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Exp-1 예비 PoC 결과 (합성 시나리오)")
    lines.append("")
    lines.append("> ⚠ **가정 프로파일(assumed) + 합성 시나리오** 기반 예비 PoC.")
    lines.append("> 실측 자료·CURRENT_RULE 미확보 상태로, 어떤 수치도 실제 운영 대비")
    lines.append("> 개선율이 아니다. 시뮬레이터 실측 validation(YR-009) 전의 알고리즘")
    lines.append("> 동작 검증 목적으로만 해석한다.")
    lines.append("")
    lines.append("## 실행 조건")
    lines.append("")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## 정책별 평균 (test seeds, 공통난수)")
    lines.append("")
    header = "| 지표 | " + " | ".join(results.keys()) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(results) + 1))
    for key in _KEY_METRICS:
        row = [f"{_mean(rs, key):.2f}" for rs in results.values()]
        lines.append(f"| {key} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"## Paired 비교 — 기준: {baseline} (도착순 Baseline)")
    lines.append("")
    lines.append("| 정책 | 지표 | 기준평균 | 대안평균 | Δ (95% CI) | 변화% | 방향일치 seed | 유의 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    base_rs = results[baseline]
    for pname, rs in results.items():
        if pname == baseline:
            continue
        for key in ["mean_wait_min", "p95_wait_min", "queue_area_h", "travel_km",
                    "rehandles", "vessel_delay_min"]:
            d = paired_diff(_series(base_rs, key), _series(rs, key))
            lines.append(
                f"| {pname} | {key} | {d['mean_base']:.2f} | {d['mean_alt']:.2f} "
                f"| {d['mean_diff']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}] "
                f"| {d['pct_change']:+.1f}% | {d['seeds_same_direction']}/{d['n']} "
                f"| {'✔' if d['significant'] else '—'} |")
    lines.append("")
    lines.append("## 잠정 합격기준 점검 (03 §5.1 — 예비 PoC 버전)")
    lines.append("")
    ql = next((n for n in results if n.startswith("QL")), None)
    if ql:
        d_wait = paired_diff(_series(base_rs, "mean_wait_min"), _series(results[ql], "mean_wait_min"))
        d_p95 = paired_diff(_series(base_rs, "p95_wait_min"), _series(results[ql], "p95_wait_min"))
        thr = _mean(results[ql], "completed_external") / max(1e-9, _mean(base_rs, "completed_external"))
        checks = [
            ("안전·물리 제약위반 0 (invariant 상시 검사)", "충족 — 위반 시 실행이 중단됨"),
            (f"처리량 ≥ 기준 99% (실측 {thr * 100:.1f}%)", "충족" if thr >= 0.99 else "**미충족**"),
            (f"평균대기 5% 이상 감소 (실측 {d_wait['pct_change']:+.1f}%)",
             "충족" if d_wait["pct_change"] <= -5.0 else "**미충족**"),
            (f"P95 대기 악화 없음 (실측 {d_p95['pct_change']:+.1f}%)",
             "충족" if d_p95["ci_hi"] <= max(0.0, 0.05 * d_p95["mean_base"]) else "확인 필요"),
            (f"복수 seed 방향 일관 ({d_wait['seeds_same_direction']}/{d_wait['n']})",
             "충족" if d_wait["seeds_same_direction"] >= d_wait["n"] * 0.7 else "**미충족**"),
        ]
        for name, verdict in checks:
            lines.append(f"- {name}: {verdict}")
    lines.append("")
    lines.append("*생성: yard_rl.experiments.report — 원자료 exp1_results.json*")
    path = out / "exp1_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
