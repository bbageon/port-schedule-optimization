"""Verify the pushed board, result table and raw measurements for YR-332-bj."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from statistics import mean

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates, audit_dashboard,
    combine_reliability, judge_claim_alignment, judge_runtime_evidence, judge_scenario_validity,
)


OUT = Path("outputs/reports/yr332_vessel_turnaround")
SPEC = Path(".claude/docs/dashboard-task-specs/YR-332-bj-vessel-turnaround.md")
ARMS = {"rule": "짧은 작업 우선", "previous": "기존 모델", "trained": "재학습 모델"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report():
    s = read(OUT/"summary.json")
    eligible = {tuple(pair) for pair in s["comparable_inputs"]}
    all_rows = [read(p) for p in (OUT/"evaluation").glob("*.json")]
    assert len(all_rows) == 63
    lines = ["# 본선 작업 턴타임 기준 모델 재학습 성적", "",
        f"공통 완주 입력 {len(eligible)}/9개. 학습과 시험 날짜를 분리하고, 같은 트럭·본선 작업 명단으로 세 방식을 평가했다.",
        "", "## 같은 입력의 비용과 시간", "",
        "입력마다 모델 세 개의 평균을 먼저 낸 뒤 각 날짜와 물량에 같은 비중을 줬다. 비용은 하루에 들어온 물량 기준이다.",
        "", "| 배정 방식 | 총비용(억원) | 본선 턴타임(시간/척) | 트럭 턴타임(분/건) | 자정 미완료 트럭(건) |",
        "|---|---:|---:|---:|---:|"]
    for arm, group in s["groups"].items():
        lines.append(f"| {ARMS[arm]} | {group['cost']/1e8:.2f} | {group['ship_hours']:.2f} | "
                     f"{group['truck_minutes']:.2f} | {group['unfinished_trucks_24h']:.1f} |")
    lines += ["", "양수는 비교 대상보다 재학습 모델의 비용 또는 시간이 늘었다는 뜻이다.", ""]
    for ref, change in s["changes"].items():
        line = (f"- {ARMS[ref]} 대비 총비용 {change['cost']:+.2f}%, 본선 {change['ship_hours']:+.2f}%, "
                f"트럭 {change['truck_minutes']:+.2f}%. 비용이 더 낮은 입력 {change['wins']}/{len(eligible)}개.")
        if "change_pct_ci95" in change:
            lo, hi = change["change_pct_ci95"]
            line += f" 날짜와 시작 모델을 짝지어 다시 뽑은 비용 변화의 95% 범위는 {lo:+.2f}~{hi:+.2f}%다."
        lines.append(line)
    lines += ["", "### 물량별 총비용", "",
        "| 하루 트럭 오더 | 공통 완주 날짜 | 짧은 작업 우선(억원) | 기존 모델(억원) | 재학습 모델(억원) |",
        "|---:|---:|---:|---:|---:|"]
    for load in (3500, 7500, 15000):
        days = [i for i in range(3) if (i, load) in eligible]
        cells = []
        for arm in ARMS:
            values = [r["cost"]/1e8 for r in s["cases"][arm] if r["load"] == load and r["index"] in days]
            cells.append(f"{mean(values):.2f}" if values else "—")
        lines.append(f"| {load:,} | {len(days)}/3 | " + " | ".join(cells) + " |")
    lines += ["", "3,500건은 실측 도착률 범위에 든 입력이고, 7,500건과 15,000건은 혼잡이 심해졌을 때의 부담 시험이다.",
        "", "### 비용 항목", "",
        "| 배정 방식 | 트럭 시간(만원) | 장비 빈 이동(만원) | 재조작(만원) | 본선 시간(만원) |",
        "|---|---:|---:|---:|---:|"]
    for arm, group in s["groups"].items():
        cells = [f"{group[k]/10000:,.1f}" for k in ("truck_cost", "move_cost", "rehandle_cost", "vessel_cost")]
        lines.append(f"| {ARMS[arm]} | " + " | ".join(cells) + " |")
    lines += ["", "본선은 하역 가능 시각부터 마지막 안벽 적하·양하까지, 총톤수 1톤당 시간당 35.88원으로 한 척에 한 번 계산했다.",
        "트럭은 게이트 진입부터 야드 작업 완료까지다. 자정에 미완료면 그때까지의 시간으로 평균과 비용을 계산한다.",
        "본선 항목의 효과를 분리하기 위해 트럭은 기존 식을 유지했다. 시간당 4만원에 1시간 초과분 시간당 4만원을 더한다.",
        "빈 이동은 시간당 6만원, 재조작은 건당 2천원이다. 이 두 항목도 종전 24시간을 집계했다. 총비용은 네 항목의 모형 추정액이다.",
        "", "세계은행의 항만 지표도 화물작업 시작·종료와 선석 출발을 따로 기록한다. 여기서는 첫 작업을 미루는 선택에도 비용이 생기도록 시작을 사전에 주어진 하역 가능 시각으로 고정했다. "
        "[세계은행 항만성과 지표 산정법](https://thedocs.worldbank.org/en/doc/aac122f6df85534428d66a7b9af4b7f6-0400012026/original/CPPI-Methodology-Note.pdf)",
        "미국교통연구소의 트럭 체류 연구는 직접비와 운행 기회 손실을 따로 산정한다. 국내 트럭이 한 시간 이후 겪는 추가 비용의 곡선은 이번 비교에서 실측 보정하지 않았다. "
        "[미국교통연구소 체류비용 연구](https://truckingresearch.org/2024/09/new-research-documents-substantial-financial-and-safety-impacts-from-truck-driver-detention/)",
        "", "## 완료 여부와 학습 범위", "",
        "| 배정 방식 | 모든 배가 끝난 실행/전체 | 끝난 배/전체 | 144시간에 남은 안벽 작업(개, 전체 실행 합계) |",
        "|---|---:|---:|---:|"]
    for arm, label in ARMS.items():
        rows = [r for r in all_rows if r["arm"] == arm]
        calls = [v for r in rows for v in r["vessel_calls"]]
        lines.append(f"| {label} | {sum(r['vessel_complete'] for r in rows)}/{len(rows)} | "
                     f"{sum(v['finish_s'] is not None for v in calls)}/{len(calls)} | {sum(v['remaining'] for v in calls)} |")
    lines += ["", "학습은 매번 같은 날짜 12개를 모두 실행했다. 모든 배가 끝난 날의 선택에는 자정 이후 마지막 배까지 남은 비용을 연결했다.",
        "144시간에도 작업이 남으면 최종비용을 알 수 없어 그날을 가중치 갱신에서 제외했다.", "",
        "| 시작 모델 | 완료한 학습일 | 실제 가중치 갱신/예정 | 완료일의 자정 이후 선택(회) |", "|---|---:|---:|---:|"]
    for seed, rec in s["training"].items():
        lines.append(f"| {seed} | {rec['complete_days']}/12 | {rec['optimizer_steps']//128}/6 | "
                     f"{rec['post_midnight_choices_on_complete_days']:,} |")
    lines += ["", "별도 동작확인 날짜에는 가장 유리하게 계산해도 지정 블록에 양하 최소 123개를 놓을 수 없었다.",
        "시험은 기존 합성 날짜 세 개를 다시 썼다. 자정 이후 신규 유입 없이 당일 배를 마무리했으며, 실제 항만의 연속운영 절감액은 검증 대상 밖이다.",
        "", "[원자료 집계](summary.json) · [시험 63회](evaluation/) · [학습일과 가중치](models/) · "
        "[학습 설정](campaign.json) · [블록별 수용 한계](capacity-check.json) · [미래비용 검산](smoke/return-wiring.json)", ""]
    (OUT/"README.md").write_text("\n".join(lines), encoding="utf8", newline="\n")


def publish(commit):
    summary = read(OUT / "summary.json")
    stamp = read(OUT / "run-info.json")
    campaign = read(OUT / "campaign.json")
    head = stamp["code"]["git_head"]
    assert head == campaign["code_commit"] and stamp["params"]["prereg_sha256"] == campaign["prereg_sha256"]
    frozen = subprocess.check_output(["git", "show", f"{head}:{SPEC.as_posix()}"])
    assert campaign["prereg_sha256"] in {
        hashlib.sha256(raw).hexdigest() for raw in (frozen, frozen.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    }
    remote = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "@{u}"], encoding="utf8").strip()
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.suffix in {".pt", ".json", ".xml"}
             and p.name not in {"current-gates.json", "dashboard-audit.json"}]
    artifacts = {p.as_posix(): sha(p) for p in paths}
    for name, digest in artifacts.items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"])
        assert hashlib.sha256(blob).hexdigest() == digest, name
    runtime = judge_runtime_evidence(stamp, artifact_hashes=artifacts)
    board = audit_dashboard(".", task_id="YR-332-bj", expected_state="done", spec_path=SPEC,
        evidence_paths=[OUT/"summary.json", OUT/"README.md", OUT/"run-info.json"],
        evidence_commits=[commit], remote_ref=remote, pin_commit=commit)
    report = (OUT/"README.md").read_text(encoding="utf8")
    declared, truth = {}, {}
    for arm, group in summary["groups"].items():
        match = re.search(r"^\| " + ARMS[arm] + r" \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", report, re.M)
        assert match, arm
        for pos, (metric, divisor) in enumerate((("cost", 1e8), ("ship_hours", 1), ("truck_minutes", 1)), 1):
            declared[arm + metric] = float(match[pos])
            truth[arm + metric] = group[metric] / divisor
    n = len(summary["comparable_inputs"])
    match = re.search(r"공통 완주 입력 (\d+)/9개", report)
    assert match
    declared["matched_inputs"], truth["matched_inputs"] = int(match[1]), n
    claims = judge_claim_alignment(declared, truth, absolute_tolerance=.00501)
    reliability = combine_reliability(runtime, board, claims)
    trials = [read(p) for p in (OUT/"evaluation").glob("*.json")]
    assert len(trials) == 63
    physical = all(r["txn_failed"] == 0 and r["policy_exceptions"] == 0 for r in trials)
    scenario = judge_scenario_validity(internal_checks=dict(
        event_time_order=all(v["finish_s"] is None or v["finish_s"] >= v["start_s"]
                             for row in trials for v in row["vessel_calls"]),
        physical_constraints=physical, ledger_conservation=True,
        deterministic_replay=read(OUT/"paired-rule-replay.json")["passed"],
        achievable_vessel_deadline=False), flow_checks={}, anchors={}, continuous_operation=False,
        request_real_terminal_claim=False)
    comparisons = summary["changes"].get("rule", {})
    guards = dict(completion=n == 9,
        backlog_zero=all(row["n_job_censored"] == 0 for row in trials if row["arm"] == "trained"),
        vessel_protection=bool(comparisons) and comparisons["ship_hours"] <= 0, physical_valid=physical)
    performance = GateOutcome("performance", GateStatus.FAIL if not all(guards.values()) else GateStatus.INCONCLUSIVE,
        "고정 시험의 완주·미완료·본선 보호 조건 미충족" if not all(guards.values())
        else "합성 날짜 세 개의 탐색 비교로 운영 성능 채택 보류",
        evidence=dict(metric="terminal_total_cost", baseline="SF-SPT", hard_guards=guards,
            paired=comparisons, comparable_inputs=n, synthetic_dates=3, diagnostic_only=True))
    gates = attach_common_gates(dict(task="YR-332-bj", evidence_commit=commit),
                                ResearchGateReport(performance, reliability, scenario))
    assert reliability.status is GateStatus.PASS, reliability.as_dict()
    for name, data in (("current-gates.json", gates), ("dashboard-audit.json", board.as_dict())):
        (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8", newline="\n")
    print({g.name: g.status.value for g in (performance, reliability, scenario)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("commit", nargs="?", help="Pushed evidence commit. Omit to render the result table.")
    args = parser.parse_args()
    if args.commit:
        publish(args.commit)
    else:
        write_report()
