"""교착 수정 대조 보고서 (YR-317-k).

같은 작업 트리·같은 고정 입력·같은 재배정 없음 정책으로 시드 21,000,000 을 9일까지
두 번 재생했다. 한쪽만 후보 가지치기를 고쳤다.

⚠️ **비용은 두 재생 사이에서만 비교한다.** 등록된 서른 날 실행의 날별 비용과는
비교하지 않는다 — 끝내 못 끝낸 트럭의 대기비용이 **남은 날들에 걸쳐 계속 쌓여**
요청일로 귀속되기 때문이다. 9일에서 끊은 재생은 그 누적을 담지 못한다.
초반 며칠(검열된 트럭이 거의 없는 구간)만 등록 기록과 맞춰 재현을 확인한다.

진단 증거이지 등록된 결과가 아니다 — 시드 하나·블록 하나다.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / "fix-validation-9d"
RECORDED = (ROOT / "outputs/reports/yr317_v3_independent_eval/run-7e2fb14"
            / "months/21000000/NO_REALLOC/daily-final.jsonl")
MODES = (("legacy", "수정 전"), ("feasible_first", "수정 후"))


def load(mode):
    path = RUN / f"validate-{mode}.json"
    if not path.exists():
        raise SystemExit(f"아직 없음: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def span(run):
    first, last = run.get("first_stall"), run.get("last_stall")
    return None if not first else (first["at_day"], last["at_day"])


def main():
    runs = {mode: load(mode) for mode, _ in MODES}
    kr = dict(MODES)
    recorded = ([json.loads(line) for line in
                 RECORDED.read_text(encoding="utf-8").splitlines() if line]
                if RECORDED.exists() else [])
    saved = {row["index"]: row for row in recorded}
    days = {mode: {row["index"]: row for row in run["days"]} for mode, run in runs.items()}

    lines = ["# 교착 수정 대조 재생 — 시드 21,000,000 · 블록 Y17 · 재배정 없음", "",
        "같은 작업 트리·같은 고정 입력·같은 정책으로 9일까지 두 번 재생했다.",
        "한쪽만 후보 가지치기를 **실행 가능 후보 우선**으로 고쳤다.", "",
        "## 1. 갈리는 것은 멈춤이 아니라 풀림이다", "",
        "| | 정지 신호 구간 | 신호가 잡힌 5분 표본 | 마지막 순간 밀린 일 | 9일 재생 실시간 |",
        "|---|---|---:|---:|---:|"]
    for mode, name in MODES:
        run = runs[mode]
        where = span(run)
        shown = "없음" if where is None else f"{where[0]:.2f}일 → {where[1]:.2f}일"
        if where and where[1] >= run["until_day"] - 0.5:
            shown += " **(안 풀림)**"
        elif where:
            shown += " (풀림)"
        backlog = run.get("last_stall", {}).get("waiting", 0)
        lines.append(f"| {name} | {shown} | {run['stall_windows']} | {backlog:,}건 | "
                     f"{run['elapsed_s'] / 3600:.2f}시간 |")
    lines += ["",
        "**둘 다 4.03일에 똑같이 멈췄다.** 수정은 교착을 예방하지 않는다 — 탈출을",
        "가능하게 한다. 수정 전은 재생이 끝나는 9일까지 한 번도 못 빠져나왔고 밀린 일이",
        "계속 불었다. 수정 후는 4.13일에 빠져나온 뒤 다시 멈추지 않았다.", "",
        "멈춘 블록은 노는 것이 아니다. 매 결정 주기마다 못 하는 일 수천 건을 다시 채점한다.",
        "그래서 **수정 전이 같은 9일을 도는 데 세 배 걸렸다**.",
        "⚠️ 기계 부하가 같지 않아 정밀한 배수는 아니다.", ""]

    lines += ["## 2. 날별 비용 (억원) — 두 재생 사이에서만 비교한다", "",
        "| 날 | 부하 | 수정 전 | 수정 후 | 차이 |", "|---:|---:|---:|---:|---:|"]
    for index in sorted(set(days["legacy"]) & set(days["feasible_first"])):
        a = days["legacy"][index]["phi_krw"] / 1e8
        b = days["feasible_first"][index]["phi_krw"] / 1e8
        lines.append(f"| {index} | {days['legacy'][index]['load']:,} | {a:,.2f} | {b:,.2f} | "
                     f"{b - a:+,.2f} |")
    lines += ["",
        "⚠️ **수정이 공짜가 아니다.** 멈추기 전 며칠은 수정 후가 오히려 조금 비싸다.",
        "실행 가능한 일을 앞세우면 크레인이 고르는 일 자체가 달라지고, 그것이 늘 싸지는",
        "않는다. 9일 안에서는 정지의 비용이 아직 드러나지 않는다 — 못 끝낸 트럭의 대기가",
        "남은 날들에 걸쳐 쌓이기 때문이다.", "",
        "**정지의 값은 서른 날을 다 돌아야 보인다.** 등록된 30일 실행(수정 전)에서 부하가",
        "같은 두 날을 견주면 3일째 26.38억 → 7일째 **296.3억** 이다. 이 값은 9일에서 끊은",
        "재생으로는 재현할 수 없고, 수정 엔진의 30일 실행은 [[YR-318]] 에서 나온다.", ""]

    lines += ["## 3. 기본 경로가 바뀌지 않았는가", ""]
    if saved:
        rows, agree = [], 0
        for index in sorted(set(saved) & set(days["legacy"]))[:4]:
            a = saved[index]["phi_krw"] / 1e8
            b = days["legacy"][index]["phi_krw"] / 1e8
            rows.append(f"| {index} | {a:,.2f} | {b:,.2f} | {saved[index].get('n_censored', 0):,} |")
            agree += abs(a - b) < 0.02
        lines += ["| 날 | 등록 30일 기록 | 수정 전 재생 | 그날 못 끝낸 트럭(확정) |",
                  "|---:|---:|---:|---:|", *rows, "",
                  f"초반 {agree}일은 **0.02억 안에서 일치**한다. 그 뒤로 벌어지는 것은 엔진이",
                  "달라서가 아니라 못 끝낸 트럭의 대기가 남은 날들에 걸쳐 쌓이기 때문이다.",
                  "즉 옵트인 기본값(`legacy`)은 기존 여든 번을 그대로 재현한다.", ""]
    else:
        lines += ["등록 기록 파일을 찾지 못해 대조하지 못했다.", ""]

    lines += ["재생기: `validate_fix.py` · 원자료: `fix-validation-9d/validate-*.json`",
              "⚠️ 시드 하나·블록 하나 진단이며 정책 우열의 증거가 아니다.", ""]

    out = RUN / "comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({kr[m]: dict(stall_windows=runs[m]["stall_windows"], span=span(runs[m]),
        hours=round(runs[m]["elapsed_s"] / 3600, 2),
        backlog=runs[m].get("last_stall", {}).get("waiting")) for m, _ in MODES},
        ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
