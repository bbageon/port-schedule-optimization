"""YR-332-bn report: the retrained model on the unseen month beside the arms YR-332-bl ran on the same month."""
import gzip
import hashlib
import json
from math import ceil, fsum
from pathlib import Path
import random
from statistics import mean, pstdev

from review_joint_turnaround import DAY, VESSEL_TIME_WEIGHT, elapsed_seconds, times


OUT = Path("outputs/reports/yr332_predicted_tail")
BL = Path("outputs/reports/yr332_joint_turnaround")
ARMS = {"bn": ("새 모델(30분 + 예측값)", OUT / "evaluation" / "retrained"),
        "bl": ("bl 모델(월말까지 통째)", BL / "evaluation" / "retrained"),
        "previous": ("기존 모델", BL / "evaluation" / "previous"),
        "SF_SPT": ("짧은 작업 우선", BL / "evaluation" / "SF_SPT"),
        "LWKR": ("오래 기다린 작업 우선", BL / "evaluation" / "LWKR")}
RULE, RESAMPLES, SEED = "SF_SPT", 10000, 12204201


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def arm(folder):
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        trucks = list(map(json.loads, stream))
    ships, result = read(folder / "ships.json"), read(folder / "result.json")
    window = result["windows"]["all"]
    row = dict(code=result["code"], checkpoint_sha256=result["checkpoint_sha256"], decisions=result["model_decisions"],
               unfinished=window["unfinished_trucks"] + window["unfinished_ships"])
    if row["unfinished"]:
        return dict(row, complete=False), None
    a, b = times(trucks, "job", "gate", 0, float("inf")), times(ships, "ship", "start", 0, float("inf"))
    truck_s, ship_s = elapsed_seconds(a), elapsed_seconds(b)
    durations = sorted(end - start for start, end in a.values())
    days = {}
    for cohort, key in ((a, "truck"), (b, "ship")):
        for start, end in cohort.values():
            cell = days.setdefault(int(start // DAY), dict(truck=0., trucks=0, ship=0., ships=0))
            cell[key] += end - start
            cell[key + "s"] += 1
    return dict(row, complete=True, trucks=len(a), ships=len(b), truck_minutes=truck_s / len(a) / 60,
                ship_hours=ship_s / len(b) / 3600, truck_p95_minutes=durations[ceil(.95 * len(durations)) - 1] / 60,
                weighted_minutes=(truck_s + VESSEL_TIME_WEIGHT * ship_s) / 60,
                safe_rate_krw=fsum(max(d - 3600, 0) for d in durations) * 20000 / 1800,
                empty_travel_krw=window["costs"]["empty"], rehandle_krw=window["costs"]["rehandles"]), \
        ({k: s for k, (s, _) in a.items()}, {k: s for k, (s, _) in b.items()}, days)


def paired(arms, days):
    rng = random.Random(SEED)
    index = sorted(days[RULE])
    draws = [[rng.choice(index) for _ in index] for _ in range(RESAMPLES)]

    def metric(cells):
        truck, ship = fsum(c["truck"] for c in cells), fsum(c["ship"] for c in cells)
        ships = sum(c["ships"] for c in cells)
        return (truck / sum(c["trucks"] for c in cells) / 60, ship / ships / 3600 if ships else None,
                (truck + VESSEL_TIME_WEIGHT * ship) / 60)

    result = {}
    for name in arms:
        if name == RULE or not arms[name]["complete"]:
            continue
        samples = [[], [], []]
        for chosen in draws:
            x, y = metric([days[name][d] for d in chosen]), metric([days[RULE][d] for d in chosen])
            if x[1] is None:
                continue
            samples[0].append(x[0] - y[0])
            samples[1].append(x[1] - y[1])
            samples[2].append(100 * (x[2] / y[2] - 1))
        a, r = arms[name], arms[RULE]
        point = (a["truck_minutes"] - r["truck_minutes"], a["ship_hours"] - r["ship_hours"],
                 100 * (a["weighted_minutes"] / r["weighted_minutes"] - 1))
        result[name] = {k: dict(difference=p, low=sorted(s)[int(.025 * len(s))], high=sorted(s)[int(.975 * len(s)) - 1])
                        for k, p, s in zip(("truck_minutes", "ship_hours", "weighted_percent"), point, samples)}
        result[name]["days_worse_for_trucks"] = sum(days[name][d]["truck"] > days[RULE][d]["truck"] for d in index)
        result[name]["days"] = len(index)
    return result


def training():
    rows = []
    for u in range(4):
        fit = read(OUT / "training" / f"fit-{u}.json")
        runs = [read(OUT / "runs" / f"wave-{u}-{k}.json") for k in range(4)]
        months = [r["month"] for r in runs]
        rows.append(dict(update=u, noise=fit["noise_weighted_minutes"], time_share=fit["time_share"],
                         anchors=fit["anchors"], kl=fit["fit"]["means"]["approx_kl"],
                         clip=fit["fit"]["means"]["clip_fraction"], entropy=fit["fit"]["means"]["entropy"],
                         truck_minutes=mean(m["truck_minutes"] for m in months),
                         ship_hours=mean(m["ship_hours"] for m in months),
                         truck_spread=pstdev(m["truck_minutes"] for m in months)))
    return rows


def verdict(arms):
    new, rule = arms["bn"], arms[RULE]
    if not (new["complete"] and rule["complete"]):
        return "판정 불가: 시험 월을 끝내지 못한 방식이 있다"
    truck, ship = new["truck_minutes"] < rule["truck_minutes"], new["ship_hours"] < rule["ship_hours"]
    return ("공동 목표 달성(탐색): 새 모델이 짧은 작업 우선보다 두 평균을 모두 줄였다" if truck and ship else
            "교환관계: 새 모델이 짧은 작업 우선보다 한쪽만 줄였다" if truck or ship else
            "미달: 새 모델이 짧은 작업 우선보다 두 평균 모두 길다")


def main():
    arms, entries, days = {}, None, {}
    for name, (_, folder) in ARMS.items():
        arms[name], extra = arm(folder)
        if extra is None:
            continue
        if entries is None:
            entries = extra[:2]
        elif extra[:2] != entries:
            raise ValueError("The arms did not receive the same trucks, port calls and entry times")
        days[name] = extra[2]
    final = Path("build/yr332_joint_turnaround/predicted-tail/models/update-3.pt")
    if arms["bn"]["checkpoint_sha256"] != hashlib.sha256(final.read_bytes()).hexdigest():
        raise ValueError("The evaluated model is not the last registered update")
    summary = dict(vessel_time_weight=VESSEL_TIME_WEIGHT, evaluation=arms, verdict=verdict(arms),
                   paired_vs_short_job_rule=paired(arms, days), training=training(), measure=read(OUT / "measure.json"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf8", newline="\n")
    lines = ["# 30분 실제 + 예측값 학습과 처음 보는 한 달 비교 (bn)", "",
             "합성 환경 탐색이다. 현실성 판정은 미통과로 보존돼 있고 현장 성능이나 정책 채택을 뜻하지 않는다.", "",
             f"**판정: {summary['verdict']}.**", "", "## 시험 월 (학습에 쓰지 않은 30일)", "",
             "| 방식 | 트럭 분/작업 | 본선 시간/척 | 1대67 가중 합계(백만 분) | 규칙 대비 | 느린 트럭 5% 경계(분) |",
             "|---|---:|---:|---:|---:|---:|"]
    rule = arms[RULE]
    for name, (label, _) in ARMS.items():
        a = arms[name]
        if not a["complete"]:
            lines.append(f"| {label} | 미완료 | 미완료 | — | — | — |")
            continue
        lines.append(f"| {label} | {a['truck_minutes']:.2f} | {a['ship_hours']:.2f} | {a['weighted_minutes'] / 1e6:.3f} | "
                     f"{100 * (a['weighted_minutes'] / rule['weighted_minutes'] - 1):+.2f}% | {a['truck_p95_minutes']:.2f} |")
    lines += ["", "## 짧은 작업 우선과의 차이 (날짜 단위 재표집, 탐색용 95% 구간)", "",
              "| 방식 | 트럭 분/작업 | 본선 시간/척 | 가중 합계 | 트럭이 더 오래 걸린 날 |", "|---|---:|---:|---:|---:|"]
    for name, p in summary["paired_vs_short_job_rule"].items():
        cell = lambda k, unit: f"{p[k]['difference']:+.2f}{unit} ({p[k]['low']:+.2f}~{p[k]['high']:+.2f})"
        lines.append(f"| {ARMS[name][0]} | {cell('truck_minutes', '')} | {cell('ship_hours', '')} | "
                     f"{cell('weighted_percent', '%')} | {p['days_worse_for_trucks']}/{p['days']} |")
    lines += ["", "## 학습 과정 (학습 월을 직전 모델로 4번씩 굴림, 확률 선택)", "",
              "| 갱신 | 학습 신호 흔들림(가중 분) | 시간대 비율 | 정책 변화량 | 트럭 분/작업 | 본선 시간/척 |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary["training"]:
        lines.append(f"| {r['update'] + 1} | {r['noise']:,.0f} | {100 * r['time_share']:.1f}% | {r['kl']:.5f} | "
                     f"{r['truck_minutes']:.2f} | {r['ship_hours']:.2f} |")
    m = summary["measure"]
    lines += ["", f"2단계 측정: 흔들림 {m['noise_weighted_minutes']:,.0f} 가중 분(월말까지 셀 때 "
              f"{m['monte_carlo_noise_weighted_minutes']:,.0f}), 시간대 비율 {100 * m['time_share']:.1f}%, "
              f"갱신당 결정 {m['decisions_per_update']:,}개.", "",
              "## 한계", "", "- 모델 1개·학습 월 1개·시험 월 1개다. 학습 난수에 따른 차이를 분리하지 못한다.",
              "- 학습 월은 시험 월보다 훨씬 붐빈다. 합성 월이며 현장 대조는 남아 있다.", "",
              "[집계 원값](summary.json) · [2단계 측정](measure.json) · [실행 전 등록](../../../.claude/docs/dashboard-task-specs/YR-332-bn-predicted-tail-return.md)"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
