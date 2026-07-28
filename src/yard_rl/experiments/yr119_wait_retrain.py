"""YR-119 — WAIT 을 허용한 재학습: 단일축 (사용자 설계 2026-07-28, 결과 미열람 동결).

■ YR-118 이 남긴 것과 남기지 못한 것 (사용자 감사 반영)
확정된 것: ①건전성 검사는 과민이 아니다(미건전 = 같은 시드 규칙 대비 +6.00 유의하게 비쌈)
②퇴화 행동은 REPOSITION 과다 선택 ③학습 안 한 WAIT 를 평가 때만 여는 것은 해법이 아니다
(행동분포는 정상처럼 보여도 비용 +34~41%).
**확정하면 안 되는 것**: "FORBID_WAIT 가 REPOSITION 퇴화의 근본 원인이다." 개입이 학습 안 한
WAIT 를 평가 시점에 주입했으므로, REPOSITION 감소가 "금지 해제 덕"인지 "신경망이 미학습
WAIT 의 가치를 엉터리로 낮게 매겨 무작정 고른 것"인지 분리되지 않았다.
정확한 현재 결론: **WAIT 가 제거된 행동공간에서 일부 학습정책이 REPOSITION 을 대체 행동처럼
과다 선택한다. WAIT 를 정상적으로 학습시키면 해결되는지는 미검증.** ← 이 실험이 그걸 검증한다.

추가 근거(YR-118 후속 분석): 실패 17건이 **학습시드 88000 에 10건 집중**
(모델 CALC:88000 단독 7/24 — 균일 가정 확률 0.018). 즉 행동공간 문제와 **학습 초기화
불안정이 공존**할 수 있다 → 판정에 초기화 재현성 축을 반드시 넣는다.

■ 설계 — 유일한 차이 하나 (사용자 지정)
  · 대조군 WAITOFF: 현재처럼 전략적 WAIT 를 **학습·실행 모두 제외** (FORBID_WAIT=True)
  · 실험군 WAITON : WAIT 를 **학습·실행 모두 허용** (FORBID_WAIT=False)
  · 나머지 전부 동일: 보상(REPO_PENALTY 0.5·UNSERVED 30 불변)·후보 생성기·학습률 5e-4·
    배치 64·에피소드 500·학습 시드 {88000, 99000, 123000}·학습/검증 대역·신경망 구조.
  · **대조군도 현행 코드로 재학습한다** — 기존 체크포인트는 YR-112(교착 탈출)·YR-116(후보
    계약) 이전 엔진에서 학습됐으므로 재사용하면 단일축이 깨진다.

■ 성공 기준 (사용자 지정 — 건전성 통과만 보지 않는다)
  ① 규칙(SF) 대비 짝지은 총비용
  ② 실제 게이트 진입→진출 시간 (A→O, v2 시간계약으로 측정)
  ③ 완주율 1.0 · backlog 0 (하드 guard)
  ④ REPOSITION 비율 감소
  ⑤ 실행 가능한 SERVE 가 있는데 고른 **전략적 WAIT 비율**
  ⑥ WAIT 의 횟수뿐 아니라 **지속시간** (결정 간격 기준)
  ⑦ 서로 다른 학습 초기화(3 시드)에서의 재현성
  판정 라벨은 evalkit(δ: total 10 · a2o_min 1분). 이 실험은 **진단·개발 단계**이며
  채택 판정이 아니다(채택은 별도 사전등록 후보평가).

■ 2차 실험 트리거 (사용자 지정)
  WAITON 에서 전략적 WAIT 가 과다(단일 행동 60% 장악 또는 SERVE 가능 시 WAIT 선택률 이
  대조군의 3배 초과)로 관측될 때만, **WAIT 지속시간 벌점 / 후보 제한 중 하나만** 추가하는
  후속 단일축을 연다. REPO_PENALTY 동시 변경 금지(원인 구분 불가).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

import torch

from ..integrated.baselines import (ActionMixError, ResolverPolicy, ServiceFirstSPTPreference,
                                    assert_healthy_action_mix, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.evalkit import paired
from ..integrated.repro import repro_stamp
from . import yr088_joint_rl as y88
from . import yr090_dense_vessel as y90
from .yr090_dense_vessel import BASE, CELLS, _sim
from .yr100_candidate_eval import RC_EVAL

OUT = Path("outputs/reports/yr119_wait_retrain")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
# YR-100-[3]/YR-118 과 같은 평가대역 — 실패를 재현했던 바로 그 표본에서 개선을 본다
EVAL_SEEDS = {c: [BASE[c] + 700 + i for i in range(6)] for c in CELLS}
DELTA = {"total": 10.0, "a2o_min": 1.0}
ARMS = {"WAITOFF": True, "WAITON": False}       # arm → FORBID_WAIT 값


def _set_forbid_wait(v: bool) -> None:
    """단일축 토글 — yr088 은 호출시점 전역을 읽고, yr090 은 값-import 라 **양쪽** 을 맞춘다."""
    y88.FORBID_WAIT = v
    y90.FORBID_WAIT = v


def train_arm(arm: str, seed: int, episodes: int = 500) -> Path:
    forbid = ARMS[arm]
    dst = OUT / f"{arm.lower()}_s{seed}"
    prev_out, prev88, prev90 = y90.OUT, y88.FORBID_WAIT, y90.FORBID_WAIT
    y90.OUT = OUT                                  # train_one 의 저장 경로만 우회
    _set_forbid_wait(forbid)
    try:
        t0 = time.time()
        p = y90.train_one(arm, seed, episodes=episodes)   # arm != "DENSE" → 대조 recipe
        print(f"[train {arm} s{seed}] {time.time() - t0:.0f}s → {p}", flush=True)
        return p
    finally:
        y90.OUT = prev_out
        y88.FORBID_WAIT = prev88
        y90.FORBID_WAIT = prev90


class _Recorder:
    """정책 래퍼 — 결정마다 (시각, 크레인별 선택, SERVE 실행가능 여부)를 기록.

    WAIT **지속시간**(사용자 기준 ⑥)은 같은 크레인의 다음 결정까지의 간격으로 계산한다.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.events: list[tuple[float, str, str, bool]] = []   # (t, crane, kind, serve_ok)

    def decide(self, sim, dp, gen_by):
        assign = self.inner.decide(sim, dp, gen_by)
        for c in dp.crane_ids:
            serve_ok = any(g.feasible and g.kind.value == "SERVE" for g in gen_by[c].items)
            self.events.append((dp.time, c, assign[c].kind.value, serve_ok))
        return assign

    def wait_metrics(self, end: float) -> dict:
        by_crane: dict[str, list[tuple[float, str, bool]]] = {}
        for t, c, k, s in self.events:
            by_crane.setdefault(c, []).append((t, k, s))
        n_wait = n_wait_serve_ok = n_serve_ok = 0
        wait_dur = 0.0
        for c, ev in by_crane.items():
            for i, (t, k, s) in enumerate(ev):
                if s:
                    n_serve_ok += 1
                if k == "WAIT":
                    n_wait += 1
                    if s:
                        n_wait_serve_ok += 1
                    nxt = ev[i + 1][0] if i + 1 < len(ev) else end
                    wait_dur += max(0.0, nxt - t)
        return {"n_wait": n_wait,
                "strategic_wait_rate": round(n_wait_serve_ok / n_serve_ok, 4) if n_serve_ok else 0.0,
                "wait_duration_s": round(wait_dur, 1),
                "wait_duration_share": round(wait_dur / max(1.0, end), 4)}


def _a2o_mean_min(sim) -> float | None:
    tl = getattr(sim, "time_ledger", None)
    if tl is None:
        return None
    vals = []
    for r in tl.records.values():
        a = getattr(r, "gate_in", None)
        if a is None:
            continue
        o = getattr(r, "gate_out", None)
        vals.append(((o if o is not None else sim.end) - a) / 60.0)   # 미출문 = 검열(end−A)
    return round(fmean(vals), 3) if vals else None


def _episode(cell: str, seed: int, mk_policy) -> dict:
    sim = _sim(cell, seed)
    rec = _Recorder(mk_policy())
    r = run_joint_episode(sim, rec, RC_EVAL, generator=CandidateGenerator())
    mix = r["_mix"]
    healthy = True
    try:
        assert_healthy_action_mix(mix, label=f"{cell}/s{seed}")
    except ActionMixError:
        healthy = False
    d = mix.as_dict()
    return {"cell": cell, "seed": seed, "healthy": healthy,
            "total": round(r["total_cost"], 3), "a2o_min": _a2o_mean_min(sim),
            "compl": r["completion_rate"], "backlog": r["backlog"],
            "shares": d["shares"], "serve_when_available": d["serve_when_available"],
            "repo_share": d["shares"].get("REPOSITION", 0.0),
            **rec.wait_metrics(sim.end),
            "deadlock_escapes": sim.deadlock_escape_count}


def _rl_policy_factory(ck_path: Path, forbid: bool):
    from ..integrated.encoding import StateNorm
    from ..integrated.joint_distill import JointPairNet
    ck = torch.load(ck_path, map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    norm = StateNorm(refs=ck["norm_refs"])

    def mk():
        _set_forbid_wait(forbid)                 # RLPolicy 는 생성 시점 전역을 읽는다
        return y88.RLPolicy(net, norm, name=f"RL[{ck_path.parent.name}]")
    return mk


def evaluate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [(c, s) for c in CELLS for s in EVAL_SEEDS[c]]
    print(f"[eval] SF 기준선 {len(eps)} 에피소드", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"))
          for c, s in eps]
    arms: dict[str, list[dict]] = {}
    for arm, forbid in ARMS.items():
        for ts in TRAIN_SEEDS:
            ck = OUT / f"{arm.lower()}_s{ts}" / "rl_net.pt"
            if not ck.exists():
                raise FileNotFoundError(f"학습 체크포인트 없음: {ck} — 먼저 --train")
            key = f"{arm}:{ts}"
            print(f"[eval] {key}", flush=True)
            mk = _rl_policy_factory(ck, forbid)
            arms[key] = [_episode(c, s, mk) for c, s in eps]
    res = _judge(sf, arms)
    res["repro"] = repro_stamp(
        experiment="YR-119 WAIT 허용 재학습 — 단일축 (학습·실행 모두)",
        seeds={"train": list(TRAIN_SEEDS), **{c: EVAL_SEEDS[c] for c in CELLS}},
        profile_id="calibrated",
        prereg="유일 차이 = FORBID_WAIT. 판정: J1 미건전 비율 감소(3/3 시드 비악화) · "
               "J2 짝지은 총비용 비악화(δ=10) · J3 A→O 비악화(δ=1분) · "
               "J4 REPOSITION 비중 감소 · J5 전략적 WAIT 남용 감시(2차 실험 트리거). "
               "진단·개발 단계 — 채택 판정 아님.",
        extra={"delta": DELTA, "arms": {k: ("FORBID" if v else "ALLOW") for k, v in ARMS.items()}})
    (OUT / "results.json").write_text(json.dumps({"sf": sf, "arms": arms, **res},
                                                 ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return res


def _judge(sf: list[dict], arms: dict[str, list[dict]]) -> dict:
    out: dict = {"per_arm": {}, "judgment": {}}
    for key, rows in arms.items():
        n_unh = sum(1 for r in rows if not r["healthy"])
        out["per_arm"][key] = {
            "unhealthy": n_unh, "n": len(rows),
            "repo_share_mean": round(fmean(r["repo_share"] for r in rows), 4),
            "strategic_wait_rate_mean": round(fmean(r["strategic_wait_rate"] for r in rows), 4),
            "wait_duration_share_mean": round(fmean(r["wait_duration_share"] for r in rows), 4),
            "compl_min": min(r["compl"] for r in rows),
            "backlog_max": max(r["backlog"] for r in rows),
            "d_total_vs_sf": paired([r["total"] - s["total"] for r, s in zip(rows, sf)],
                                    delta_interest=DELTA["total"]).as_dict(),
            "d_a2o_vs_sf": (paired([r["a2o_min"] - s["a2o_min"] for r, s in zip(rows, sf)],
                                   delta_interest=DELTA["a2o_min"]).as_dict()
                            if all(r["a2o_min"] is not None for r in rows + sf) else None)}
    # 공동 판정 — WAITON(실험군) − WAITOFF(대조군), 같은 (학습시드, 평가에피소드) 짝
    d_tot, d_a2o, d_repo = [], [], []
    unh = {"WAITON": 0, "WAITOFF": 0}
    per_seed = {}
    for ts in TRAIN_SEEDS:
        on, off = arms[f"WAITON:{ts}"], arms[f"WAITOFF:{ts}"]
        d_tot += [a["total"] - b["total"] for a, b in zip(on, off)]
        if all(r["a2o_min"] is not None for r in on + off):
            d_a2o += [a["a2o_min"] - b["a2o_min"] for a, b in zip(on, off)]
        d_repo += [a["repo_share"] - b["repo_share"] for a, b in zip(on, off)]
        u_on = sum(1 for r in on if not r["healthy"])
        u_off = sum(1 for r in off if not r["healthy"])
        unh["WAITON"] += u_on
        unh["WAITOFF"] += u_off
        per_seed[ts] = {"unhealthy_on": u_on, "unhealthy_off": u_off,
                        "non_worse": u_on <= u_off}
    j = {"J1_unhealthy": {"WAITON": unh["WAITON"], "WAITOFF": unh["WAITOFF"],
                          "per_seed": per_seed,
                          "pass": unh["WAITON"] < unh["WAITOFF"]
                                  and all(v["non_worse"] for v in per_seed.values())},
         "J2_total_on_minus_off": paired(d_tot, delta_interest=DELTA["total"]).as_dict(),
         "J3_a2o_on_minus_off": (paired(d_a2o, delta_interest=DELTA["a2o_min"]).as_dict()
                                 if d_a2o else None),
         "J4_repo_share_delta": round(fmean(d_repo), 4),
         "J5_second_experiment_trigger": None}
    on_all = [r for ts in TRAIN_SEEDS for r in arms[f"WAITON:{ts}"]]
    off_all = [r for ts in TRAIN_SEEDS for r in arms[f"WAITOFF:{ts}"]]
    swr_on = fmean(r["strategic_wait_rate"] for r in on_all)
    swr_off = fmean(r["strategic_wait_rate"] for r in off_all)
    wait_dom = any(r["shares"].get("WAIT", 0) > 0.60 for r in on_all)
    j["J5_second_experiment_trigger"] = {
        "strategic_wait_rate_on": round(swr_on, 4), "off": round(swr_off, 4),
        "wait_dominates_any": wait_dom,
        "triggered": bool(wait_dom or (swr_off > 0 and swr_on > 3 * swr_off))}
    out["judgment"] = j
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="", help="ARM:SEED 쉼표목록 | all")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.train:
        jobs = ([(arm, ts) for arm in ARMS for ts in TRAIN_SEEDS] if a.train == "all"
                else [(p.split(":")[0], int(p.split(":")[1])) for p in a.train.split(",")])
        for arm, ts in jobs:
            train_arm(arm, ts, episodes=a.episodes)
    if a.eval:
        r = evaluate()
        print(json.dumps(r["judgment"], ensure_ascii=False, indent=1))
    print("DONE")
