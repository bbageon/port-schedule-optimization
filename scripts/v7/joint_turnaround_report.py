"""Summarize the retraining run and the unseen-month comparison from the saved ledgers."""
import gzip
import json
from math import ceil, fsum, isclose, sqrt
from pathlib import Path
import random
import sys
from statistics import mean, pstdev

from review_joint_turnaround import DAY, VESSEL_TIME_WEIGHT, elapsed_seconds, times


OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/reports/yr332_joint_turnaround")
ARMS = dict(retrained="새 모델(1대67 보상)", previous="기존 모델", SF_SPT="짧은 작업 우선", LWKR="오래 기다린 작업 우선")
RULE = "SF_SPT"
RESAMPLES, SEED = 10000, 12204201


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def ledger(arm):
    folder = OUT / "evaluation" / arm
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        trucks = list(map(json.loads, stream))
    return trucks, read(folder / "ships.json"), read(folder / "result.json")


def evaluation():
    arms, entries = {}, None
    for arm in ARMS:
        trucks, ships, result = ledger(arm)
        window = result["windows"]["all"]
        row = dict(unfinished_trucks=window["unfinished_trucks"], unfinished_ships=window["unfinished_ships"],
                   code=result["code"], checkpoint_sha256=result["checkpoint_sha256"], seconds=result["seconds"])
        if row["unfinished_trucks"] or row["unfinished_ships"]:
            arms[arm] = dict(row, complete=False)   # reported, never ranked
            continue
        a, b = times(trucks, "job", "gate", 0, float("inf")), times(ships, "ship", "start", 0, float("inf"))
        same = [{k: s for k, (s, _) in cohort.items()} for cohort in (a, b)]
        if entries is None:
            entries = same
        elif same != entries:
            raise ValueError("The arms did not receive the same trucks, port calls and entry times")
        truck_s, ship_s = elapsed_seconds(a), elapsed_seconds(b)
        durations = sorted(end - start for start, end in a.values())
        days = {}
        for cohort, key in ((a, "truck"), (b, "ship")):
            for start, end in cohort.values():
                cell = days.setdefault(int(start // DAY), dict(truck=0.0, trucks=0, ship=0.0, ships=0))
                cell[key] += end - start
                cell[key + "s"] += 1
        arms[arm] = dict(row, complete=True, trucks=len(a), ships=len(b),
            truck_minutes=truck_s / len(a) / 60, ship_hours=ship_s / len(b) / 3600,
            truck_p95_minutes=durations[ceil(.95 * len(durations)) - 1] / 60,
            weighted_minutes=(truck_s + VESSEL_TIME_WEIGHT * ship_s) / 60,
            truck_share_of_weighted=truck_s / (truck_s + VESSEL_TIME_WEIGHT * ship_s),
            safe_rate_krw=fsum(max(d - 3600, 0) for d in durations) * 20000 / 1800,
            empty_travel_krw=window["costs"]["empty"], rehandle_krw=window["costs"]["rehandles"],
            decisions=result["model_decisions"], days=days)
    return arms


def paired(arms):
    """Whole days are redrawn together. One month and one model: a description, not a confirmation."""
    rule, result = arms[RULE], {}
    rng = random.Random(SEED)
    index = sorted(rule["days"])
    draws = [[rng.choice(index) for _ in index] for _ in range(RESAMPLES)]

    def metrics(arm, chosen):
        cells = [arm["days"][d] for d in chosen]
        truck, ship = fsum(c["truck"] for c in cells), fsum(c["ship"] for c in cells)
        ships = sum(c["ships"] for c in cells)
        return (truck / sum(c["trucks"] for c in cells) / 60, ship / ships / 3600 if ships else None,
                (truck + VESSEL_TIME_WEIGHT * ship) / 60)

    for name, arm in arms.items():
        if name == RULE or not arm["complete"]:
            continue
        samples = [[], [], []]
        for chosen in draws:
            x, y = metrics(arm, chosen), metrics(rule, chosen)
            if x[1] is None:
                continue
            samples[0].append(x[0] - y[0])
            samples[1].append(x[1] - y[1])
            samples[2].append(100 * (x[2] / y[2] - 1))
        point = [arm["truck_minutes"] - rule["truck_minutes"], arm["ship_hours"] - rule["ship_hours"],
                 100 * (arm["weighted_minutes"] / rule["weighted_minutes"] - 1)]
        keys = ("truck_minutes", "ship_hours", "weighted_percent")
        result[name] = {k: dict(difference=p, low=sorted(s)[int(.025 * len(s))], high=sorted(s)[int(.975 * len(s)) - 1])
                        for k, p, s in zip(keys, point, samples)}
        result[name]["days_worse_for_trucks"] = sum(
            arm["days"][d]["truck"] > rule["days"][d]["truck"] for d in index)
        result[name]["days"] = len(index)
    return result


def training():
    campaign = read(OUT / "training" / "campaign.json")
    cfg = campaign["configuration"]
    updates = []
    for u in range(cfg["updates"]):
        months = [read(OUT / "training" / f"rollout-{u}-{k}.json") for k in range(cfg["months_per_update"])]
        fit = read(OUT / "training" / f"fit-{u}.json")
        done = [m for m in months if m["complete"]]
        totals = [m["weighted_minutes"] for m in done]
        updates.append(dict(update=u, complete_months=len(done), months=len(months),
            weighted_minutes=totals, weighted_mean=mean(totals), weighted_sd=pstdev(totals),
            truck_minutes=mean(m["month"]["truck_minutes"] for m in done),
            ship_hours=mean(m["month"]["ship_hours"] for m in done),
            decisions=mean(m["decisions"] for m in done), samples=mean(m["samples"] for m in done),
            value_explained_at_collection=mean(m["explained"] for m in done),
            value_explained_before_update=fit["value_before_update"]["explained"],
            target_noise_weighted_minutes=fit["value_before_update"]["value_rmse"] * cfg["return_scale"],
            policy_change=fit["fit"]["means"]["approx_kl"], clip_fraction=fit["fit"]["means"]["clip_fraction"],
            entropy=fit["fit"]["means"]["entropy"], policy_samples=fit["policy_samples"],
            seconds=max(m["seconds"] for m in months)))
    return dict(commit=campaign["commit"], configuration=cfg, updates=updates,
                value_restart=read(OUT / "training" / "fit-0.json")["value_restart"])


def verdict(arms):
    new, rule, old = arms["retrained"], arms[RULE], arms["previous"]
    if not (new["complete"] and rule["complete"]):
        return "판정 불가: 시험 월을 끝내지 못한 방식이 있다"
    truck, ship = new["truck_minutes"] < rule["truck_minutes"], new["ship_hours"] < rule["ship_hours"]
    text = ("공동 목표 달성(탐색): 새 모델이 짧은 작업 우선보다 두 평균을 모두 줄였다" if truck and ship else
            "교환관계: 새 모델이 짧은 작업 우선보다 한쪽만 줄였다" if truck or ship else
            "미달: 새 모델이 짧은 작업 우선보다 두 평균 모두 길다")
    return text


def won(value):
    return f"{value / 1e8:,.2f}"


def report(summary):
    arms, train, pairs = summary["evaluation"], summary["training"], summary["paired_vs_short_job_rule"]
    rule = arms[RULE]
    lines = ["# 1대67 보상 재학습과 처음 보는 한 달 비교", "",
             "합성 환경 탐색이다. 현실성 판정은 미통과로 보존돼 있고 현장 성능이나 정책 채택을 뜻하지 않는다.", "",
             f"**판정: {summary['verdict']}.**", ""]
    new, old = arms["retrained"], arms["previous"]
    if new["complete"] and old["complete"]:
        lines += [f"새 모델은 기존 모델과 비교해 트럭 {new['truck_minutes'] - old['truck_minutes']:+.2f}분/작업, "
                  f"본선 {new['ship_hours'] - old['ship_hours']:+.2f}시간/척, "
                  f"가중 합계 {100 * (new['weighted_minutes'] / old['weighted_minutes'] - 1):+.2f}%였다.", ""]
    lines += ["## 시험 월 (학습에 쓰지 않은 30일)", "",
              "| 방식 | 트럭 분/작업 | 본선 시간/척 | 1대67 가중 합계(백만 분) | 규칙 대비 | 느린 트럭 5% 경계(분) | 안전운임 환산(억원) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in ARMS.items():
        a = arms[name]
        if not a["complete"]:
            lines.append(f"| {label} | 미완료 | 미완료 | — | — | — | — |")
            continue
        lines.append(f"| {label} | {a['truck_minutes']:.2f} | {a['ship_hours']:.2f} | {a['weighted_minutes'] / 1e6:.3f} | "
                     f"{100 * (a['weighted_minutes'] / rule['weighted_minutes'] - 1):+.2f}% | "
                     f"{a['truck_p95_minutes']:.1f} | {won(a['safe_rate_krw'])} |")
    lines += ["", f"트럭 {rule['trucks']:,}건과 본선 {rule['ships']}척이 네 방식 모두에서 끝났는지와 진입 시각이 같은지를 원장에서 확인했다. "
              "가중 합계는 낮을수록 좋다.", "",
              "## 짧은 작업 우선과의 차이 (날짜 단위 재표집, 탐색용 95% 구간)", "",
              "| 방식 | 트럭 분/작업 | 본선 시간/척 | 가중 합계 | 트럭이 더 오래 걸린 날 |", "|---|---:|---:|---:|---:|"]
    for name, p in pairs.items():
        cell = lambda k, unit: f"{p[k]['difference']:+.2f}{unit} ({p[k]['low']:+.2f}~{p[k]['high']:+.2f})"
        lines.append(f"| {ARMS[name]} | {cell('truck_minutes', '')} | {cell('ship_hours', '')} | "
                     f"{cell('weighted_percent', '%')} | {p['days_worse_for_trucks']}/{p['days']} |")
    lines += ["", "모델 하나와 시험 월 하나의 비교다. 구간은 진입 날짜를 다시 뽑아 만든 참고값이며 확증이 아니다.", "",
              "## 학습 과정 (학습 월을 현재 정책으로 4번씩 굴림)", "",
              "| 갱신 | 끝난 월 | 가중 합계 평균(백만 분) | 같은 정책 4개 월의 흔들림 | 트럭 분/작업 | 본선 시간/척 | 비용 예측 설명력 | 정책 변화량 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for u in train["updates"]:
        lines.append(f"| {u['update'] + 1} | {u['complete_months']}/{u['months']} | {u['weighted_mean'] / 1e6:.3f} | "
                     f"±{100 * u['weighted_sd'] / u['weighted_mean']:.2f}% | {u['truck_minutes']:.2f} | {u['ship_hours']:.2f} | "
                     f"{u['value_explained_before_update']:.3f} | {u['policy_change']:.6f} |")
    signal = summary["signal_to_noise"]
    lines += ["", "각 행은 그 갱신 직전 정책의 성적이다. 선택을 확률로 뽑으므로 시험 월의 고정 선택 성적과 직접 비교하지 않는다.", "",
              "## 학습 신호와 잡음", "",
              f"- 선택 하나의 학습 목표에 섞인 잡음은 평균 {signal['noise_per_decision']:,.0f} 가중 분이다. 비용 예측으로 설명하고 남은 크기다.",
              f"- 기존 모델과 짧은 작업 우선의 시험 월 격차를 선택 수로 나누면 선택 하나당 {signal['gain_per_decision']:,.1f} 가중 분이다. "
              "규칙만큼 잘하게 될 때 선택 하나가 평균적으로 벌 수 있는 크기다.",
              f"- 잡음이 이 이득의 약 {signal['noise_over_gain']:,.0f}배다. 갱신 한 번의 2,048표본을 평균해도 약 {signal['noise_over_gain_after_averaging']:,.0f}배가 남는다.",
              f"- 갱신 한 번의 정책 변화량은 평균 {signal['policy_change']:.6f}였다. 직전 하루 단위 학습은 0.001~0.003이었다.", "",
              "## 한계", "",
              "- 모델 1개·학습 월 1개·시험 월 1개다. 학습 난수에 따른 차이를 분리하지 못한다.",
              "- 고물량 날짜가 섞인 합성 월이며 실제 작업시간·본선 물량과의 대조는 남아 있다.",
              "- 안전운임 환산은 1시간 초과분만 계산한 작업당 근사다.", "",
              "[집계 원값](summary.json) · [학습 기록](training/) · [시험 월 원장](evaluation/) · [실행 전 등록](../../../.claude/docs/dashboard-task-specs/YR-332-bl-joint-turnaround-reward.md)"]
    return "\n".join(lines) + "\n"


def main():
    arms, train = evaluation(), training()
    summary = dict(vessel_time_weight=VESSEL_TIME_WEIGHT, evaluation=arms, verdict=verdict(arms),
                   paired_vs_short_job_rule=paired(arms), training=train)
    old, rule = arms["previous"], arms[RULE]
    noise = mean(u["target_noise_weighted_minutes"] for u in train["updates"])
    gain = abs(old["weighted_minutes"] - rule["weighted_minutes"]) / old["decisions"]
    if gain == 0:
        raise ValueError("The earlier model and the rule tie exactly; the per-decision gap is undefined")
    summary["signal_to_noise"] = dict(noise_per_decision=noise, gain_per_decision=gain,
        noise_over_gain=noise / gain, noise_over_gain_after_averaging=noise / gain / sqrt(train["configuration"]["max_samples"]),
        policy_change=mean(u["policy_change"] for u in train["updates"]))
    for arm in arms.values():      # daily cells are kept in the summary under string keys
        if arm["complete"]:
            arm["days"] = {str(d): c for d, c in sorted(arm["days"].items())}
            total = fsum(c["truck"] for c in arm["days"].values()) + VESSEL_TIME_WEIGHT * fsum(
                c["ship"] for c in arm["days"].values())
            if not isclose(total / 60, arm["weighted_minutes"], rel_tol=1e-12):
                raise ValueError("Daily cells do not add up to the month")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf8", newline="\n")
    (OUT / "README.md").write_text(report(summary), encoding="utf8", newline="\n")
    print(report(summary))


if __name__ == "__main__":
    main()
