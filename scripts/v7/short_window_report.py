"""YR-332-bp report: the model trained on 30-minute block time, on the unseen month beside the arms YR-332-bl ran there.

Read-only on saved ledgers and training records. Writes summary.json and README.md in the bp report folder.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev

from predicted_tail_report import arm, paired
from review_joint_turnaround import VESSEL_TIME_WEIGHT


OUT = Path("outputs/reports/yr332_short_window")
BL = Path("outputs/reports/yr332_joint_turnaround")
MODELS = Path("build/yr332_joint_turnaround/short-window/models")
ARMS = {"bp": ("새 모델(30분 학습)", OUT / "evaluation" / "retrained"),
        "previous": ("기존 모델", BL / "evaluation" / "previous"),
        "bl": ("bl 모델(월말까지 통째)", BL / "evaluation" / "retrained"),
        "SF_SPT": ("짧은 작업 우선", BL / "evaluation" / "SF_SPT"),
        "LWKR": ("오래 기다린 작업 우선", BL / "evaluation" / "LWKR")}
RULE = "SF_SPT"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training():
    rows = []
    for u in range(4):
        runs = [read(OUT / "runs" / f"wave-{u}-{k}.json") for k in range(4)]
        months = [r["month"] for r in runs]
        fit = read(OUT / "training" / f"fit-{u}.json")
        if fit["checkpoint_sha256"] != sha(MODELS / f"update-{u}.pt"):
            raise ValueError(f"Update {u} checkpoint changed after training")
        rows.append(dict(update=u, collected_with=runs[0]["checkpoint"], collected_with_sha256=runs[0]["checkpoint_sha256"],
                         months_complete=sum(m["unfinished_trucks"] + m["unfinished_ships"] == 0 for m in months),
                         truck_minutes=mean(m["truck_minutes"] for m in months),
                         truck_spread=pstdev(m["truck_minutes"] for m in months),
                         ship_hours=mean(m["ship_hours"] for m in months), anchors=fit["anchors"],
                         noise_weighted_minutes=fit["noise_weighted_minutes"], time_share=fit["time_share"],
                         kl=fit["fit"]["means"]["approx_kl"], clip_fraction=fit["fit"]["means"]["clip_fraction"],
                         entropy=fit["fit"]["means"]["entropy"], critic_loss=fit["fit"]["means"]["critic_loss"],
                         replay_logp_max_error=fit["fit"]["replay_logp_max_error"],
                         value_fit=fit.get("value_fit", {}).get("normalized_rmse_after"),
                         rollout_seconds=mean(r["seconds"] for r in runs)))
    return rows


def main():
    arms, entries, days = {}, None, {}
    for name, (_, folder) in ARMS.items():
        arms[name], extra = arm(folder)
        if extra is None:
            raise ValueError(f"{name} left work unfinished on the unseen month")
        if entries is None:
            entries = extra[:2]
        elif extra[:2] != entries:
            raise ValueError("The arms did not receive the same trucks, port calls and entry times")
        days[name] = extra[2]
    if arms["bp"]["checkpoint_sha256"] != sha(MODELS / "update-3.pt"):
        raise ValueError("The evaluated model is not the last registered update")
    new, rule, old = arms["bp"], arms[RULE], arms["previous"]
    truck, ship = new["truck_minutes"] < rule["truck_minutes"], new["ship_hours"] < rule["ship_hours"]
    verdict = ("공동 목표 달성(탐색): 새 모델이 짧은 작업 우선보다 두 평균을 모두 줄였다" if truck and ship else
               "교환관계: 새 모델이 짧은 작업 우선보다 한쪽만 줄였다" if truck or ship else
               "미달: 새 모델이 짧은 작업 우선보다 두 평균 모두 길다")
    gap_closed = (old["weighted_minutes"] - new["weighted_minutes"]) / (old["weighted_minutes"] - rule["weighted_minutes"])
    campaigns = {p.name: read(p) for p in sorted(OUT.glob("train-campaign*.json"))}
    summary = dict(
        vessel_time_weight=VESSEL_TIME_WEIGHT, evaluation=arms, verdict=verdict,
        versus_previous=dict(truck_minutes=new["truck_minutes"] - old["truck_minutes"],
                             ship_hours=new["ship_hours"] - old["ship_hours"],
                             weighted_percent=100 * (new["weighted_minutes"] / old["weighted_minutes"] - 1),
                             share_of_rule_gap_closed=gap_closed),
        paired_vs_short_job_rule=paired(arms, days), training=training(),
        provenance=dict(campaigns={k: dict(commit=v["commit"], decisions_per_update=v["decisions_per_update"])
                                   for k, v in campaigns.items()},
                        note=("Wave 0 ran from 5b4fd2a6. The first update was lost twice without an error record "
                              "(b2129bd7 run gone before 20:26). Another session streamed the stored decisions "
                              "(c2260974, memory only) and ran updates 0-3, waves 1-3 and the evaluation."),
                        evaluation_code=read(OUT / "evaluation" / "retrained" / "result.json")["code"]),
        gates=dict(performance="FAIL (registered rule: both averages longer than the short-job rule on the unseen month)",
                   scenario_validity="unresolved, unchanged (synthetic exploration)",
                   reliability="see dashboard-audit.json"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf8", newline="\n")
    lines = ["# 30분 블록 시간 학습과 처음 보는 한 달 비교 (bp)", "",
             "합성 환경 탐색이다. 현실성 판정은 미통과로 보존돼 있고 현장 성능이나 정책 채택을 뜻하지 않는다.", "",
             f"**판정(실행 전 기준): {verdict}.** 다만 기존 모델보다 크게 나아져, 기존 모델과 규칙 사이 격차의 "
             f"{100 * gap_closed:.0f}%를 좁혔다.", "",
             "## 시험 월 (학습에 쓰지 않은 30일)", "",
             "| 방식 | 트럭 분/작업 | 본선 시간/척 | 1대67 가중 합계(백만 분) | 규칙 대비 | 느린 트럭 5% 경계(분) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, (label, _) in ARMS.items():
        a = arms[name]
        lines.append(f"| {label} | {a['truck_minutes']:.2f} | {a['ship_hours']:.2f} | {a['weighted_minutes'] / 1e6:.3f} | "
                     f"{100 * (a['weighted_minutes'] / rule['weighted_minutes'] - 1):+.2f}% | {a['truck_p95_minutes']:.2f} |")
    v = summary["versus_previous"]
    lines += ["", f"새 모델은 기존 모델보다 트럭 {v['truck_minutes']:+.2f}분/작업, 본선 {v['ship_hours']:+.2f}시간/척, "
              f"가중 합계 {v['weighted_percent']:+.2f}%였다. 트럭 {rule['trucks']:,}건과 본선 {rule['ships']}척이 다섯 방식 모두에서 "
              "끝났고 진입 시각이 같았다.", "",
              "## 짧은 작업 우선과의 차이 (날짜 단위 재표집, 탐색용 95% 구간)", "",
              "| 방식 | 트럭 분/작업 | 본선 시간/척 | 가중 합계 | 트럭이 더 오래 걸린 날 |", "|---|---:|---:|---:|---:|"]
    for name, p in summary["paired_vs_short_job_rule"].items():
        cell = lambda k, unit: f"{p[k]['difference']:+.2f}{unit} ({p[k]['low']:+.2f}~{p[k]['high']:+.2f})"
        lines.append(f"| {ARMS[name][0]} | {cell('truck_minutes', '')} | {cell('ship_hours', '')} | "
                     f"{cell('weighted_percent', '%')} | {p['days_worse_for_trucks']}/{p['days']} |")
    lines += ["", "## 학습 과정 (학습 월을 직전 모델로 4번씩 굴림, 확률 선택)", "",
              "| 물결 | 굴린 모델 | 트럭 분/작업 | 본선 시간/척 | 학습 신호 흔들림(가중 분) | 시간대 비율 | 정책 변화량 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["training"]:
        model = "시작 모델" if r["update"] == 0 else f"갱신 {r['update']}회 뒤"
        lines.append(f"| {r['update'] + 1} | {model} | {r['truck_minutes']:.2f} (±{r['truck_spread']:.2f}) | {r['ship_hours']:.2f} | "
                     f"{r['noise_weighted_minutes']:,.0f} | {100 * r['time_share']:.1f}% | {r['kl']:.5f} |")
    lines += ["", "각 물결은 그 시점 모델로 학습 월을 굴린 성적이다. 마지막 갱신 뒤 모델은 학습 월에서 굴리지 않고 시험 월에서만 평가했다.", "",
              "## 한계", "",
              "- 30분 점수에는 본선 효과가 1% 미만만 담기는데도 본선 시간이 줄었다. 트럭 처리가 빨라져 크레인 여유가 본선 작업으로 간 결과로 보이나 확인하지 않았다.",
              "- 갱신당 결정 수(2,048 → 331,520)와 갱신당 최적화 단계 수(128 → 20,720)가 함께 바뀌어 어느 쪽 효과인지 나눌 수 없다.",
              "- 모델 1개·학습 월 1개·시험 월 1개다. 학습 월은 시험 월보다 훨씬 붐빈다.",
              "- 첫 물결은 `5b4fd2a6`, 나머지 갱신·물결·평가는 결정 기록을 나눠 읽게만 바꾼 `c2260974`에서 돌았다. 계산 방법은 같다.", "",
              "[집계 원값](summary.json) · [실행 기록](train-campaign.json) · [학습 기록](training/) · "
              "[실행 전 등록](../../../.claude/docs/dashboard-task-specs/YR-332-bp-short-window-large-batch.md)"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
