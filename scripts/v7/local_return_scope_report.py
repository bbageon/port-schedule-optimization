"""Write the Korean summary of YR-332-bm step 1 from its saved measurement."""
import json
from pathlib import Path


OUT = Path("outputs/reports/yr332_local_return_scope")
SHOWN = ("0.5h", "1h", "2h", "4h", "8h", "24h", "48h", "month")
NAMES = {"0.5h": "30분", "month": "월말까지"}


def name(key):
    return NAMES.get(key, key.replace("h", "시간"))


def main():
    s = json.loads((OUT / "summary.json").read_text(encoding="utf8"))
    d, q, g, cap, noise = s["decision"], s["queue_hours"], s["gain"], s["capture"], s["noise"]
    best = d["recommended_window"]
    row = noise[f"block/{best}"]
    verdict = {"retrain": "추천 기간으로 재학습을 진행할 수 있다",
               "register the predicted-tail follow-up": "기간 뒤를 예측값으로 채우는 방식을 후속으로 등록한다",
               "close without retraining": "기간을 잘라 세는 방식으로는 재학습하지 않는다"}[d["outcome"]]
    lines = ["# 학습 목표 범위 측정 (bm 1단계, 재학습 없음)", "",
             "합성 환경 진단이다. bl이 남긴 학습 월 16개의 원장과 저장 결정만 다시 계산했다. "
             "새 시뮬레이션과 가중치 갱신은 없다.", "",
             f"**판정(실행 전 규칙): {verdict}.** 가장 나은 기간은 {name(best)}이다. 그때 남는 흔들림이 담긴 효과의 "
             f"{row['noise_over_signal']:,.0f}배로 기준({d['thresholds']['max_noise_over_signal']:.0f}배)을 넘고, "
             f"본선 포착률이 {100 * cap[best]['ship']:.1f}%로 기준({100 * d['thresholds']['min_ship_capture']:.0f}%)에 못 미친다. "
             f"시간대 비율은 {100 * row['time_share']:.1f}%로 기준({100 * d['thresholds']['max_time_share']:.0f}%) 안이다.", "",
             "## 선택의 효과가 나타나는 데 걸리는 시간", "",
             f"- 결정 시각에 그 블록 몫으로 들어와 있던 트럭은 끝나기까지 중간 {q['truck_p50_median']:.2f}시간, "
             f"느린 10%는 {q['truck_p90_median']:.2f}시간이 걸렸다. 결정마다 잰 값의 중앙값이다.",
             f"- 결정의 {100 * q['decisions_with_controlled_ship']:.1f}%에는 그 블록이 마지막까지 남아 완료를 좌우하는 배가 있었다. "
             f"그 배는 결정 뒤 중간 {q['controlled_ship_p50_median']:.2f}시간에 끝났다.", "",
             "## 기간별 결과 (블록 범위)", "",
             "| 기간 | 트럭 포착 | 본선 포착 | 남는 흔들림(가중 분) | 담긴 효과 대비 | 시간대 비율 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in SHOWN:
        n = noise[f"block/{key}"]
        lines.append(f"| {name(key)} | {100 * cap[key]['truck']:.1f}% | {100 * cap[key]['ship']:.1f}% | {n['noise']:,.0f} | "
                     f"{n['noise_over_signal']:,.0f}배 | {100 * n['time_share']:.1f}% |")
    terminal, trucks = noise[f"terminal/{best}"], noise[f"block_trucks/{best}"]
    month_terminal, month_block = noise["terminal/month"], noise["block/month"]
    lines += ["", f"담긴 효과 = {g['per_decision']:.2f} × ({g['truck_share']:.3f} × 트럭 포착 + {g['ship_share']:.3f} × 본선 포착) 가중 분이다. "
              f"시간대 비율의 우연 기대값은 {100 * row['chance_share']:.1f}%다. 흔들림은 비용 예측부만 새로 맞춘 뒤 "
              f"맞추지 않은 결정 {s['split']['held_out']:,}개에서 쟀다.", "",
              "## 어느 축이 흔들림을 줄였나 (판정에 쓰지 않음)", "",
              f"- 같은 {name(best)}에서 블록만 세면 흔들림이 약 {terminal['noise'] / row['noise']:.0f}분의 1로 줄었다 "
              f"(터미널 전체 {terminal['noise']:,.0f} → 블록 {row['noise']:,.0f}).",
              f"- 월말까지 세면 블록으로 좁혀도 흔들림이 크다(터미널 {month_terminal['noise']:,.0f}, 시간대 비율 "
              f"{100 * month_terminal['time_share']:.0f}% → 블록 {month_block['noise']:,.0f}). 터미널 월말 값은 bl 진단과 같은 크기다.",
              f"- 67 없이 트럭만 세면 {name(best)} 흔들림이 {trucks['noise']:,.0f}, 담긴 효과 대비 {trucks['noise_over_signal']:,.0f}배다. "
              "짧은 기간의 본선 항목은 대부분 '그 블록이 맡은 배 수 × 시간'이라 선택과 거의 무관한데, "
              "비용 예측부가 이를 정확히 맞히지 못해 흔들림으로 남는다.", "",
              "## 갱신에 쓰는 결정 수", "",
              f"- 한 물결(학습 월 4번)에서 선택지가 갈린 결정은 {min(s['collected_per_wave']):,}~{max(s['collected_per_wave']):,}개였고, "
              f"갱신에는 {s['used_per_update']:,}개만 썼다.",
              f"- 신호와 흔들림이 같아지려면 갱신 한 번에 (담긴 효과 대비 배수)²개가 필요하다. "
              f"블록 {name(best)} {row['decisions_for_equal_signal']:,.0f}개, 트럭만 {name(best)} {trucks['decisions_for_equal_signal']:,.0f}개, "
              f"트럭만 1시간 {noise['block_trucks/1h']['decisions_for_equal_signal']:,.0f}개다. 결과를 본 뒤 추가한 설명 값이다.", "",
              "## 한계", "",
              "- 트럭 포착률은 결정 시각에 그 블록에 있던 트럭이 기간 안에 끝나는 비율로 어림했다. 선택의 효과가 이보다 빨리 "
              "나타나면 짧은 기간의 트럭 쪽 수치는 좋아진다. 본선 쪽 결론은 같다.",
              "- 선택 하나의 효과는 bl 시험 월의 기존 모델과 짧은 작업 우선의 격차를 모든 결정 수로 나눈 값이다.",
              "- 모델 하나가 모은 학습 월 하나의 결정이다. 이 달은 시험 월보다 훨씬 붐빈다.", "",
              "[집계 원값](summary.json) · [착수 기록](authorization.json) · "
              "[실행 전 등록](../../../.claude/docs/dashboard-task-specs/YR-332-bm-local-return-scope.md)"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
