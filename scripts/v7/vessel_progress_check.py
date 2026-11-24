"""YR-332-bo step 2: does the quay-work number let the value model see port call progress?

Read-only on the four recorded months. Two checks registered before the run:
  C1  remaining open time of the deciding block's unfinished port calls, predicted by a fresh value head on fixed
      features, with versus without the number: held-out error at least 30% lower with it.
  C2  among decisions whose block's part finished last for a started call, the share where that part had the most
      quay work left at the decision, so that the block's own progress moves the number: at least 50%.
"""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from joint_turnaround_train import decode, setting
from local_return_scope import contexts, residual, sha, time_share
from vessel_progress_record import EPOCH_S, MONTHS, OUT, ROOT, save
from yard_rl.v7.stage.month import DAY_S


FIT, HELD_OUT = (0, 1), (2, 3)
MIN_ERROR_CUT, MIN_BOTTLENECK_SHARE = .30, .50
SEED = 9951101 + 7000
BEFORE = Path("outputs/reports/yr332_predicted_tail_return/vessel-visibility.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def manifest():
    cfg = setting(False)
    with gzip.open(cfg["manifests"] / read(cfg["train"])["path"], "rt", encoding="utf8") as stream:
        document = json.load(stream)
    order, blocks, block_of = defaultdict(list), defaultdict(set), {}
    for rows in document["vessels_by_day"].values():   # the replay lists stream finishes in this order
        for row in rows:
            order[row["ship"]].append(row["block"])
            blocks[row["ship"]].add(row["block"])
            block_of[row["key"]] = row["block"]
    return order, blocks, block_of


def month(k, order, blocks, block_of):
    folder = ROOT / str(k)
    record = read(OUT / "rollouts" / f"rollout-{k}.json")
    if (not record["complete"] or sha(folder / "samples.json.gz") != record["samples_sha256"]
            or sha(folder / "work-left.json.gz") != record["work_left_sha256"]):
        raise ValueError("Recorded month differs from its registration")
    by_block = defaultdict(list)
    for s in read(folder / "ships.json"):
        where = order[s["ship"]]
        if len(where) != len(s["stream_finishes"]) or s["finish"] != max(s["stream_finishes"]):
            raise ValueError(f"Stream ledger mismatch for {s['ship']}")
        own = defaultdict(float)
        for b, f in zip(where, s["stream_finishes"]):
            own[b] = max(own[b], f)
        for b in blocks[s["ship"]]:
            by_block[b].append((s["ship"], s["start"], s["finish"], s["day"] * DAY_S, own[b], own[b] == s["finish"]))
    with gzip.open(folder / "work-left.json.gz", "rt", encoding="utf8") as stream:
        epochs = {t: {call: (hours, key) for call, hours, key in rows} for t, rows in json.load(stream)["epochs"]}
    with gzip.open(folder / "samples.json.gz", "rt", encoding="utf8") as stream:
        kept = json.load(stream)["keep"]
    rows = []
    for item in kept:
        sample, obs = decode(item), item["global_observation"]
        t, b = obs["decision_time"], obs["decision_block"]
        calls = by_block[b]
        target = sum(finish - max(t, start) for _, start, finish, added, _, _ in calls if added <= t < finish) / 3600
        now = epochs.get(int(t // EPOCH_S) * EPOCH_S, {})
        later = epochs.get(int((t + 1800) // EPOCH_S) * EPOCH_S, {})
        controlled = [c for c, start, _, _, own, last in calls if last and start <= t < own]
        rows.append(dict(state=sample.state, month=k, t=t, number=item["work_left_h"], target=target,
                         controlled=len(controlled),
                         bottleneck=sum(c in now and block_of[now[c][1]] == b for c in controlled),
                         progressed=sum(c in now and (c not in later or later[c][0] < now[c][0] - 1e-9)
                                        for c in controlled)))
    return rows


def main():
    order, blocks, block_of = manifest()
    rows = [r for k in range(MONTHS) for r in month(k, order, blocks, block_of)]
    print("decisions", len(rows), flush=True)
    target = np.array([r["target"] for r in rows])
    month_of = np.array([r["month"] for r in rows])
    fit = np.isin(month_of, FIT) & (target > 0)
    test = np.isin(month_of, HELD_OUT) & (target > 0)
    context = contexts(rows)
    number = torch.tensor([[math.log1p(r["number"])] for r in rows], dtype=context.dtype)
    times = np.array([r["t"] for r in rows])[test]
    spread = float(target[test].std())
    fits = {}
    for name, x in (("without", context), ("with", torch.cat([context, number], 1))):
        left = residual(x, target, fit, test, SEED)
        rmse = float(np.sqrt(np.mean(left ** 2)))
        fits[name] = dict(rmse_hours=rmse, explained=1 - (rmse / spread) ** 2, time_share=time_share(left, times)[0])
    cut = 1 - fits["with"]["rmse_hours"] / fits["without"]["rmse_hours"]
    controlling = [r for r in rows if r["controlled"]]
    bottleneck = sum(r["bottleneck"] > 0 for r in controlling) / len(controlling)
    progressed = sum(r["progressed"] > 0 for r in controlling) / len(controlling)
    agree = sum((r["number"] > 0) == (r["target"] > 0) for r in rows) / len(rows)
    checks = dict(error_cut=cut >= MIN_ERROR_CUT, bottleneck_share=bottleneck >= MIN_BOTTLENECK_SHARE)
    summary = dict(
        question="Does the remaining quay work number let the value model see port call progress?",
        decisions=len(rows), fit=int(fit.sum()), held_out=int(test.sum()), target_sd_hours=spread,
        fits=fits, error_cut=cut, controlling_decisions=len(controlling), bottleneck_share=bottleneck,
        progressed_within_30_min=progressed, before_unclipped_share=read(BEFORE)["controlling_with_unclipped_slack"],
        number_and_target_agree_on_open_calls=agree,
        number_median_hours=float(np.median([r["number"] for r in rows if r["number"] > 0])),
        thresholds=dict(min_error_cut=MIN_ERROR_CUT, min_bottleneck_share=MIN_BOTTLENECK_SHARE),
        checks=checks, outcome="hand the number to bn" if all(checks.values()) else "do not hand the number to bn")
    save(OUT / "summary.json", summary)
    report(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "fits"}, ensure_ascii=False, indent=1))
    print(json.dumps(fits, indent=1))


def report(s):
    w, wo = s["fits"]["with"], s["fits"]["without"]
    passed = s["outcome"] == "hand the number to bn"
    lines = ["# 양적하 진척 숫자 확인 (bo, 학습 없음)", "",
             "합성 환경 진단이다. 고정한 시작 모델로 학습 월을 4번 굴리며 결정마다 '그 블록이 일부를 맡은 미완료 기항의 "
             "남은 안벽 작업 시간'(기항마다 가장 많이 남은 몫 기준)을 기록했다. 가중치 갱신은 없다.", "",
             f"**판정(실행 전 기준): {'두 기준 모두 통과 — bn에 이 숫자를 넣는다' if passed else '미달 — bn에 이 숫자를 넣지 않는다'}.**", "",
             "## 1. 배가 끝나기까지 남은 시간을 더 잘 맞히나", "",
             f"그 블록이 일부를 맡은 미완료 기항이 끝나기까지 남은 시간(평가 결정의 표준편차 {s['target_sd_hours']:.2f}시간)을 "
             "비용 예측부 머리만 새로 맞춰 예측했다. 2개 월로 맞추고 다른 2개 월로 평가했다.", "",
             f"- 숫자 없이: 오차 {wo['rmse_hours']:.2f}시간, 설명력 {100 * wo['explained']:.1f}%, 시간대 비율 {100 * wo['time_share']:.1f}%",
             f"- 숫자를 넣고: 오차 {w['rmse_hours']:.2f}시간, 설명력 {100 * w['explained']:.1f}%, 시간대 비율 {100 * w['time_share']:.1f}%",
             f"- 오차 감소 {100 * s['error_cut']:.1f}% (기준 {100 * s['thresholds']['min_error_cut']:.0f}% 이상) — "
             f"{'통과' if s['checks']['error_cut'] else '미달'}", "",
             "## 2. 블록의 양적하 진척이 숫자에 드러나나", "",
             f"블록 몫이 가장 늦게 끝나 기항 완료를 좌우한 결정 {s['controlling_decisions']:,}개를 봤다.", "",
             f"- 결정 순간 그 블록 몫이 가장 많이 남아 있던 비율: {100 * s['bottleneck_share']:.1f}% "
             f"(기준 {100 * s['thresholds']['min_bottleneck_share']:.0f}% 이상) — {'통과' if s['checks']['bottleneck_share'] else '미달'}. "
             "이때는 그 블록의 30분 양적하가 숫자를 줄인다.",
             f"- 비교: 지금 입력(여유 시간, ±2시간 상한)에서 30분 진척이 드러날 수 있던 비율은 {100 * s['before_unclipped_share']:.1f}%였다.",
             f"- 30분 안에 그 기항의 남은 작업이 실제로 줄어든 비율: {100 * s['progressed_within_30_min']:.1f}% (참고)", "",
             "## 참고", "",
             f"- 숫자가 0보다 큰지와 실제로 열린 기항이 있는지가 일치한 비율 {100 * s['number_and_target_agree_on_open_calls']:.2f}%. "
             f"0보다 클 때 중앙값 {s['number_median_hours']:.1f}시간.",
             "- 남은 작업 시간은 남은 안벽 작업 수 × 안벽 크레인 작업 간격이다. 막힘으로 늦어지는 시간은 들어 있지 않다.",
             "- 같은 학습 월을 맞춤과 평가에 썼다. 모델 하나의 기록이다.", "",
             "[집계 원값](summary.json) · [3일 재현 확인](smoke-check.json) · [실행 기록](campaign.json) · "
             "[명세](../../../.claude/docs/dashboard-task-specs/YR-332-bo-vessel-progress-visibility.md)"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
