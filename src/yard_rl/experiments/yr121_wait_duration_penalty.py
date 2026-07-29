"""YR-121 — 2차 단일축: WAIT **지속시간** 벌점 (YR-119 J5 트리거 발화의 사전등록 후속).

■ 배경
YR-119: WAIT 를 학습·실행에서 허용하니 재배치 대체는 사라졌지만(J4) **전략적 WAIT 퇴화가
부활**했다 — SERVE 가능한데 WAIT 선택 0.479(대조군의 24배), 총비용 +50.36 유의 악화.
사전등록 트리거 규칙: "WAIT 남용 관측 시에만 **지속시간 벌점 / 후보 제한 중 하나만** 추가.
REPO_PENALTY 동시 변경 금지." 사용자 선택(2026-07-29): **지속시간 벌점**.

■ 유일한 차이 하나
  · 대조군 WAITON  : YR-119 실험군 그대로 (WAIT 허용·벌점 없음) — **체크포인트 재사용**
    (같은 코드·같은 recipe 로 방금 학습·평가된 산물. 평가는 결정론이라 저장 행 재사용)
  · 실험군 DURPEN  : WAIT 허용 + `WAIT_TIME_PENALTY = 1.0` (numeraire/크레인·시간)
  · 그 외 전부 동일: FORBID_WAIT=False·REPO_PENALTY 0.5·UNSERVED 30·lr·batch·500ep·
    학습시드 {88000, 99000, 123000}·학습/검증/평가 대역.

■ 벌점 정의 (assumed 앵커 — 튜닝 아님)
  선택된 WAIT 크레인 수 × SMDP 지속시간(h) × 1.0. 즉 "크레인이 **의도적으로** 노는 1시간
  = 트럭 1대 대기 1시간"과 같은 값. 강제 WAIT(실행가능 조합 없음)는 선택이 아니므로 제외.
  유휴 쏠림의 할인 가설(대기로 시간을 흘리면 미래 비용이 γ^Δt 로 깎여 할인 기준 이득)을
  직접 겨냥한다 — 흘린 시간만큼 즉시 비용이 붙으면 시간 세탁이 공짜가 아니게 된다.

■ 판정 (결과 미열람 동결)
  K1 (주표적) 전략적 WAIT 비율: DURPEN < WAITON, **3/3 학습시드**에서.
  K2 총비용 paired(DURPEN−WAITON, 72쌍): 95% CI **상한 < 0** (유의 개선).
  K3 A→O paired: 방향 보고 (δ=1분).
  K4 풍선 재발 감시: 재배치 비중 평균 < 0.30 **그리고** 단일 비-SERVE 행동 60% 장악
     에피소드 0 (벌점이 WAIT 를 막자 다시 재배치로 새는지).
  K5 참고(판정 아님): WAITOFF·SF 대비 위치 (YR-119 동결 원자료).
  통과 = K1 ∧ K2 ∧ K4. 미통과 시 이 축 기각을 그대로 보고한다(벌점 상향 재시도는
  별도 사전등록 — 같은 실험 안에서 튜닝하지 않는다).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.evalkit import paired
from ..integrated.repro import repro_stamp
from . import yr090_dense_vessel as y90
from . import yr119_wait_retrain as y119
from .yr119_wait_retrain import EVAL_SEEDS, TRAIN_SEEDS, _episode, _rl_policy_factory

OUT = Path("outputs/reports/yr121_wait_duration_penalty")
Y119 = Path("outputs/reports/yr119_wait_retrain")
PENALTY = 1.0                      # numeraire / 크레인·시간 (assumed 앵커 — docstring)
DELTA = {"total": 10.0, "a2o_min": 1.0}


def train_arm(seed: int, episodes: int = 500) -> Path:
    prev_out, prev_pen = y90.OUT, y90.WAIT_TIME_PENALTY
    y90.OUT = OUT
    y90.WAIT_TIME_PENALTY = PENALTY
    y119._set_forbid_wait(False)                 # WAIT 허용 (YR-119 실험군과 동일)
    try:
        return y90.train_one("DURPEN", seed, episodes=episodes)
    finally:
        y90.OUT = prev_out
        y90.WAIT_TIME_PENALTY = prev_pen
        y119._set_forbid_wait(True)


def evaluate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((Y119 / "results.json").read_text(encoding="utf-8"))
    eps = [(c, s) for c in EVAL_SEEDS for s in EVAL_SEEDS[c]]
    arms: dict[str, list[dict]] = {}
    for ts in TRAIN_SEEDS:
        ck = OUT / f"durpen_s{ts}" / "rl_net.pt"
        if not ck.exists():
            raise FileNotFoundError(f"체크포인트 없음: {ck} — 먼저 --train")
        print(f"[eval] DURPEN:{ts}", flush=True)
        mk = _rl_policy_factory(ck, forbid=False)
        arms[f"DURPEN:{ts}"] = [_episode(c, s, mk) for c, s in eps]
    res = _judge(prior, arms)
    res["repro"] = repro_stamp(
        experiment="YR-121 WAIT 지속시간 벌점 — 2차 단일축",
        seeds={"train": list(TRAIN_SEEDS), **{c: EVAL_SEEDS[c] for c in EVAL_SEEDS}},
        profile_id="calibrated",
        prereg="유일 차이 = WAIT_TIME_PENALTY 1.0 (대조군 = YR-119 WAITON 동결 산물 재사용). "
               "통과 = K1(전략적 WAIT 감소 3/3) ∧ K2(총비용 CI 상한<0) ∧ K4(풍선 재발 없음). "
               "미통과 시 기각 보고 — 같은 실험 안 벌점 튜닝 금지.",
        extra={"penalty": PENALTY, "delta": DELTA,
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
                                        delta_interest=DELTA["total"]).as_dict()}
    k1 = {}
    d_tot, d_a2o = [], []
    for ts in TRAIN_SEEDS:
        rows, ctrl = arms[f"DURPEN:{ts}"], waiton[ts]
        swr_d = fmean(r["strategic_wait_rate"] for r in rows)
        swr_c = fmean(r["strategic_wait_rate"] for r in ctrl)
        k1[ts] = {"durpen": round(swr_d, 4), "waiton": round(swr_c, 4),
                  "improved": swr_d < swr_c}
        d_tot += [a["total"] - b["total"] for a, b in zip(rows, ctrl)]
        if all(r["a2o_min"] is not None for r in rows + ctrl):
            d_a2o += [a["a2o_min"] - b["a2o_min"] for a, b in zip(rows, ctrl)]
    all_rows = [r for ts in TRAIN_SEEDS for r in arms[f"DURPEN:{ts}"]]
    repo_mean = fmean(r["repo_share"] for r in all_rows)
    n_dominated = sum(1 for r in all_rows
                      if max((v for k, v in r["shares"].items() if k != "SERVE"),
                             default=0.0) > 0.60)
    j2 = paired(d_tot, delta_interest=DELTA["total"]).as_dict()
    j = {"K1_strategic_wait": {"per_seed": k1,
                               "pass": all(v["improved"] for v in k1.values())},
         "K2_total_vs_waiton": j2,
         "K3_a2o_vs_waiton": (paired(d_a2o, delta_interest=DELTA["a2o_min"]).as_dict()
                              if d_a2o else None),
         "K4_balloon": {"repo_share_mean": round(repo_mean, 4),
                        "n_dominated_episodes": n_dominated,
                        "pass": repo_mean < 0.30 and n_dominated == 0},
         "K5_reference": {
             ts: {"waitoff_d_total_vs_sf": prior["per_arm"][f"WAITOFF:{ts}"]["d_total_vs_sf"]["mean"],
                  "waiton_d_total_vs_sf": prior["per_arm"][f"WAITON:{ts}"]["d_total_vs_sf"]["mean"]}
             for ts in TRAIN_SEEDS}}
    j["pass"] = bool(j["K1_strategic_wait"]["pass"] and j2["ci"][1] < 0 and j["K4_balloon"]["pass"])
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
