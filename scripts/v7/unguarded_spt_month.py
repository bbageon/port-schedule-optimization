"""Replay the unguarded shortest-job rule on BP's published test month and compare ledgers."""
import argparse
import gzip
import json
from math import ceil
from pathlib import Path

import torch

from joint_turnaround_preflight import CACHE, OUT as BL, read, save
from month_crane_replay import run
from planned_time_comparison import read_month


OUT = Path("outputs/reports/yr332_unguarded_spt")
ARMS = {
    "SPT_UNGUARDED": OUT / "evaluation",
    "SF_SPT": BL / "evaluation" / "SF_SPT",
    "bp": Path("outputs/reports/yr332_short_window/evaluation/retrained"),
}
LABELS = {
    "SPT_UNGUARDED": "고전 최단 작업 우선",
    "SF_SPT": "24분 승격을 쓰는 기존 최단 작업 우선",
    "bp": "BP 학습 정책 (24분 승격 적용)",
}


def compare():
    manifest = read(BL / "test-input.json")["sha256"]
    metrics = {}
    for name, folder in ARMS.items():
        result = read(folder / "result.json")
        if result["manifest_sha256"] != manifest or result["days"] != 30:
            raise ValueError(f"Unexpected evaluation input for {name}")
        window = result["windows"]["all"]
        row = {k: window[k] for k in ("trucks", "ships", "unfinished_trucks", "unfinished_ships")}
        metrics[name] = row
        if row["unfinished_trucks"] or row["unfinished_ships"]:
            continue
        row.update(read_month(folder, manifest))
        with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as ledger:
            waits = sorted((r["start"] - r["block"]) / 60 for r in map(json.loads, ledger))
        row["block_wait_p95_minutes"] = waits[ceil(.95 * len(waits)) - 1]
        row["block_wait_max_minutes"] = waits[-1]
        row["trucks_waiting_over_24_minutes_at_block"] = sum(w > 24 for w in waits)
    complete = all("cohort" in row for row in metrics.values())
    if complete:
        if len({row["cohort"] for row in metrics.values()}) != 1:
            raise ValueError("Truck and ship identities or entry times differ")
        baseline = metrics["SPT_UNGUARDED"]["weighted_minutes"]
        for row in metrics.values():
            row["weighted_percent_vs_unguarded"] = 100 * (row["weighted_minutes"] / baseline - 1)
    save(OUT / "comparison.json", dict(manifest_sha256=manifest, complete=complete, arms=metrics))
    lines = [
        "# BP 시험 월의 최단 작업 규칙 재평가", "",
        "세 방식에 같은 트럭과 본선을 투입했다. 트럭 턴타임은 게이트 진입부터 작업 완료까지, "
        "본선 턴타임은 작업 시작부터 마지막 양적하 완료까지다.", "",
    ]
    if complete:
        lines += [
            "| 방식 | 트럭 평균(분/건) | 본선 평균(시간/척) | 월 가중합(백만 분) | 고전 규칙 대비 |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, r in metrics.items():
            lines.append(f"| {LABELS[k]} | {r['truck_minutes']:.2f} | {r['ship_hours']:.2f} | "
                         f"{r['weighted_minutes']/1e6:.3f} | {r['weighted_percent_vs_unguarded']:+.2f}% |")
        base = metrics["SPT_UNGUARDED"]
        bp = metrics["bp"]
        lines += [
            "", f"BP와 고전 규칙의 차이(BP − 고전)는 트럭당 {bp['truck_minutes']-base['truck_minutes']:+.2f}분, "
            f"본선당 {bp['ship_hours']-base['ship_hours']:+.2f}시간이다. 양수는 BP가 더 오래 걸렸다는 뜻이다. "
            f"트럭 {base['trucks']:,}건과 본선 {base['ships']}척이 모두 끝났고 개별 진입 시각이 일치했다.", "",
            "| 방식 | 턴타임 상위 5% 경계(분) | 블록 대기 상위 5% 경계(분) | 블록 대기 최장(분) | 블록 대기 24분 초과(건) |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, r in metrics.items():
            lines.append(f"| {LABELS[k]} | {r['truck_p95']:.2f} | {r['block_wait_p95_minutes']:.2f} | "
                         f"{r['block_wait_max_minutes']:.2f} | {r['trucks_waiting_over_24_minutes_at_block']:,} |")
        lines += ["", "블록 대기는 트럭이 블록에 도착한 뒤 작업을 시작할 때까지다. 상위 5% 경계는 트럭 100건 중 "
                  "오래 걸린 5건에 해당하는 경계 시간이다. 관측한 최대는 이 시험 월에서 나온 값이다."]
    else:
        lines += ["처리 연장 기간이 끝나도 미완료 물량이 있어 월간 우열 판정을 보류한다.", ""]
        for k, r in metrics.items():
            lines.append(f"- {LABELS[k]} 미완료 트럭 {r['unfinished_trucks']:,}건, 미완료 본선 {r['unfinished_ships']}척.")
    lines += [
        "", "## 비용과 배차 조건", "",
        "- 채점에는 모든 방식에 같은 월 가중합을 쓴다. 트럭 총 턴타임(분)에 본선 총 턴타임(분)의 67배를 더한다. "
        "67은 이 실험에 고정한 상대 가중치다.",
        "- BP 학습은 한 결정 뒤 30분 동안 해당 블록에 귀속되는 시간을 쓴다. 트럭 1대의 추가 1분에는 "
        "언제든 가중 1분이 붙는다. 24분 조건은 배정 가능한 작업의 선택 순서에 적용된다.",
        "- 고전 규칙은 실행 가능한 실작업의 계획 소요시간 순, 동률은 작업 ID 순이다. 기존 규칙은 "
        "24분 승격을 쓰고 같은 소요시간에서 본선 여부와 트럭 대기도 본다. 두 규칙 차이에는 동률 처리도 포함된다.",
        "- 결과는 한 개 모델과 한 합성 월에서의 비교다. BP의 기존 사전등록 판정은 당시 사용한 보호 규칙과의 성적이다.",
        "", "[집계 원값](comparison.json) · [월별 입력 및 실험 명세](../../../.claude/docs/dashboard-task-specs/YR-332-bw-unguarded-spt.md)",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "compare"))
    args = parser.parse_args()
    if args.action == "replay":
        torch.set_num_threads(1)
        run("SPT_UNGUARDED", False, info_path=BL / "test-input.json", manifest_dir=CACHE,
            folder=ARMS["SPT_UNGUARDED"])
    compare()
