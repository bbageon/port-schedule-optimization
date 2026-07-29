"""YR-122 — 3차 단일축: **할인 정합** γ→1.0 (사용자 선택 2026-07-29, 결과 미열람 동결).

■ 가설 (YR-118·119·121 세 결과를 하나로 설명)
평가 목적은 **비할인 총비용**인데 학습은 **할인 비용**(γ=0.99, 기준 600초)을 최소화한다.
할인 하에서는 비용을 뒤로 미루는 것 자체가 이득이다 — 1시간 시간 세탁의 할인 이득
≈ 남은 비용의 5.9% ≈ 3~7 numeraire(YR-121 사후 추정). 그래서 유휴 쏠림이
①WAIT 금지 시 재배치로(YR-118) ②허용 시 WAIT 로(YR-119) 나타나고
③1.0/h 벌점(이득의 1/3~1/7)으로는 안 막혔다(YR-121).
**γ=1.0 이면 미루기의 할인 이득이 0** — 학습 목적이 평가 목적과 정렬된다.
에피소드가 유한하므로(종결 보장) γ=1 은 정의상 문제없고, TD 목표는 이미 SCALE 로
정규화되어 크기 문제도 완화돼 있다.

■ 유일한 차이 하나
  · 대조군 WAITON  : YR-119 실험군 그대로 (γ=0.99·WAIT 허용·벌점 없음) — 동결 산물 재사용
  · 실험군 GAMMA1  : **γ=1.0** (WAIT 허용·벌점 없음 — 그 외 전부 WAITON 과 동일)
  · REPO_PENALTY 0.5·UNSERVED 30·lr·batch·500ep·시드 {88000,99000,123000} 불변.
  벌점(YR-121)과 동시 변경하지 않는다 — 이 축이 실패하면 "할인은 원인이 아니다"가 남는다.

■ 판정 (동결)
  L1 (주표적) 전략적 WAIT 비율: GAMMA1 < WAITON, **3/3 학습시드** (대조 0.433/0.490/0.513).
  L2 총비용 paired(GAMMA1−WAITON, 72쌍): 95% CI **상한 < 0** (유의 개선).
  L3 A→O paired: 방향 보고 (δ=1분).
  L4 풍선 재발 감시: 재배치 비중 < 0.30 ∧ 단일 비-SERVE 60% 장악 0건.
  L5 참고(판정 아님): WAITOFF·SF 대비 위치.
  통과 = L1 ∧ L2 ∧ L4. 미통과 시 기각 보고 — 같은 실험 안에서 γ 를 더 만지지 않는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.evalkit import paired
from ..integrated.repro import repro_stamp
from . import yr088_joint_rl as y88
from . import yr090_dense_vessel as y90
from . import yr119_wait_retrain as y119
from .yr119_wait_retrain import EVAL_SEEDS, TRAIN_SEEDS, _episode, _rl_policy_factory

OUT = Path("outputs/reports/yr122_gamma_alignment")
Y119 = Path("outputs/reports/yr119_wait_retrain")
GAMMA1 = 1.0
DELTA = {"total": 10.0, "a2o_min": 1.0}


def train_arm(seed: int, episodes: int = 500) -> Path:
    prev_out = y90.OUT
    prev_g88, prev_g90 = y88.GAMMA, y90.GAMMA
    y90.OUT = OUT
    y88.GAMMA = y90.GAMMA = GAMMA1            # 유일한 차이 (y90:132 가 읽는 값)
    y119._set_forbid_wait(False)              # WAIT 허용 — WAITON 과 동일
    try:
        return y90.train_one("GAMMA1", seed, episodes=episodes)
    finally:
        y90.OUT = prev_out
        y88.GAMMA, y90.GAMMA = prev_g88, prev_g90
        y119._set_forbid_wait(True)


def evaluate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((Y119 / "results.json").read_text(encoding="utf-8"))
    eps = [(c, s) for c in EVAL_SEEDS for s in EVAL_SEEDS[c]]
    arms: dict[str, list[dict]] = {}
    for ts in TRAIN_SEEDS:
        ck = OUT / f"gamma1_s{ts}" / "rl_net.pt"
        if not ck.exists():
            raise FileNotFoundError(f"체크포인트 없음: {ck} — 먼저 --train")
        print(f"[eval] GAMMA1:{ts}", flush=True)
        mk = _rl_policy_factory(ck, forbid=False)
        arms[f"GAMMA1:{ts}"] = [_episode(c, s, mk) for c, s in eps]
    res = _judge(prior, arms)
    res["repro"] = repro_stamp(
        experiment="YR-122 할인 정합 γ→1.0 — 3차 단일축",
        seeds={"train": list(TRAIN_SEEDS), **{c: EVAL_SEEDS[c] for c in EVAL_SEEDS}},
        profile_id="calibrated",
        prereg="유일 차이 = GAMMA 0.99→1.0 (대조군 = YR-119 WAITON 동결 산물). "
               "통과 = L1(전략적 WAIT 감소 3/3) ∧ L2(총비용 CI 상한<0) ∧ L4(풍선 재발 없음). "
               "미통과 시 기각 보고 — 같은 실험 안 γ 추가 조정 금지.",
        extra={"gamma": GAMMA1, "delta": DELTA,
               "control_source": "yr119_wait_retrain/results.json (결정론 평가 재사용)"})
    (OUT / "results.json").write_text(
        json.dumps({"arms": arms, **res}, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def _judge(prior: dict, arms: dict[str, list[dict]]) -> dict:
    waiton = {ts: prior["arms"][f"WAITON:{ts}"] for ts in TRAIN_SEEDS}
    out: dict = {"per_arm": {}, "judgment": {}}
    for key, rows in arms.items():
        ts = int(key.split(":")[1])
        ctrl = waiton[ts]
        out["per_arm"][key] = {
            "unhealthy": sum(1 for r in rows if not r["healthy"]), "n": len(rows),
            "repo_share_mean": round(fmean(r["repo_share"] for r in rows), 4),
            "strategic_wait_rate_mean": round(fmean(r["strategic_wait_rate"] for r in rows), 4),
            "wait_duration_share_mean": round(fmean(r["wait_duration_share"] for r in rows), 4),
            "compl_min": min(r["compl"] for r in rows),
            "d_total_vs_waiton": paired([a["total"] - b["total"] for a, b in zip(rows, ctrl)],
                                        delta_interest=DELTA["total"]).as_dict(),
            "d_total_vs_sf_mean": round(
                fmean(r["total"] for r in rows)
                - fmean(s["total"] for s in prior["sf"]), 3)}
    l1 = {}
    d_tot, d_a2o = [], []
    for ts in TRAIN_SEEDS:
        rows, ctrl = arms[f"GAMMA1:{ts}"], waiton[ts]
        swr_g = fmean(r["strategic_wait_rate"] for r in rows)
        swr_c = fmean(r["strategic_wait_rate"] for r in ctrl)
        l1[ts] = {"gamma1": round(swr_g, 4), "waiton": round(swr_c, 4),
                  "improved": swr_g < swr_c}
        d_tot += [a["total"] - b["total"] for a, b in zip(rows, ctrl)]
        if all(r["a2o_min"] is not None for r in rows + ctrl):
            d_a2o += [a["a2o_min"] - b["a2o_min"] for a, b in zip(rows, ctrl)]
    all_rows = [r for ts in TRAIN_SEEDS for r in arms[f"GAMMA1:{ts}"]]
    repo_mean = fmean(r["repo_share"] for r in all_rows)
    n_dom = sum(1 for r in all_rows
                if max((v for k, v in r["shares"].items() if k != "SERVE"), default=0.0) > 0.60)
    l2 = paired(d_tot, delta_interest=DELTA["total"]).as_dict()
    j = {"L1_strategic_wait": {"per_seed": l1,
                               "pass": all(v["improved"] for v in l1.values())},
         "L2_total_vs_waiton": l2,
         "L3_a2o_vs_waiton": (paired(d_a2o, delta_interest=DELTA["a2o_min"]).as_dict()
                              if d_a2o else None),
         "L4_balloon": {"repo_share_mean": round(repo_mean, 4),
                        "n_dominated_episodes": n_dom,
                        "pass": repo_mean < 0.30 and n_dom == 0}}
    j["pass"] = bool(j["L1_strategic_wait"]["pass"] and l2["ci"][1] < 0 and j["L4_balloon"]["pass"])
    out["judgment"] = j
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="", help="시드 쉼표목록 | all")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.train:
        seeds = TRAIN_SEEDS if a.train == "all" else [int(x) for x in a.train.split(",")]
        for ts in seeds:
            train_arm(ts, episodes=a.episodes)
    if a.eval:
        r = evaluate()
        print(json.dumps(r["judgment"], ensure_ascii=False, indent=1))
    print("DONE")
