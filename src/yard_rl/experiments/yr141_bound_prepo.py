"""YR-141 — v4-B: 구속적 PREPOSITION 단일축 (15차 피드백 확장 판정 7항, 동결).

■ 유일 변경 = 위치조정 행동 정의 (candidates.BOUND_REPO=True):
  REPO:<crane>:<bay> (비구속) → PREPO:<jid>:<bay> (**결속 작업 명시**). 목표 근접(≤1 bay)
  시 후보 소멸(반복 1차 억제)·ETA 만료/도착 시 소멸(재생성 내재)·교착 탈출 REPO 는
  안전기능 불변 분리. 잔여 반복·만료 후 이동은 판정 ⑦이 계수(0 요구) — 스펙 고지.
  학습·평가 그 외 전부 YR-140 과 동일 (PPO 단위 정정판·표준 앵커·최종 정책).

■ 판정 (동결 — 비교군 3: SF / v4-A(YR-140 체크포인트 재사용) / v4-B. 신규 미열람 대역
  BASE+3200..3202·실현 지문 박제. v4-B 기준 7항 전부 = 성공):
  J1 완주 100% ∧ backlog 0 (전판)     J2 PREPOSITION(REPO 계열) 장악 0
  J3 vs SF v2 비용 방향 ≥2/3 (v4-A 이득 유지)   J4 (B−A) v2 짝 평균 ≤ 0 이 ≥2/3
  J5 (B−A) 본선 초과분 짝 평균 ≤ 0 이 ≥2/3 (비열등)
  J6 (B−A) v1 전체비용 짝 평균 ≤ 0 이 ≥2/3 (이동·재조작 악화 방지 대리)
  J7 같은 작업 반복 이동 = 0 ∧ 만료(도착) 후 이동 = 0 (실행 계수)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import torch

from ..integrated import candidates as cand_mod
from ..integrated.baselines import (ActionMixError, ResolverPolicy,
                                    ServiceFirstSPTPreference,
                                    assert_healthy_action_mix, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from ..integrated.seedbank import realization_hash
from . import yr088_joint_rl as y88
from .yr090_dense_vessel import BASE, CELLS
from .yr100_candidate_eval import RC_EVAL
from .yr119_wait_retrain import _Recorder, _a2o_mean_min
from .yr136_softplus_contract import _sim_contract
from .yr138_episode_pilot import _v2_hard_total
from .yr139_blockq_v4_ppo import OUT as OUT140_DEFAULT, train_one

OUT = Path("outputs/reports/yr141_bound_prepo")
OUT140 = Path("outputs/reports/yr140_ppo_unitfix")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
EVAL_EPS = [(c, BASE[c] + 3200 + i) for c in CELLS for i in range(3)]


class _PrepoRecorder(_Recorder):
    """판정 ⑦ 계수 + 텔레메트리 — 발행/실행/만료 후 실행. one-shot 이력 기록은
    PREPO_ONE_SHOT 플래그가 켜진 경우에만 한다 (YR-141 재현·B1 대조 보호, 18차 감사)."""

    def __init__(self, inner):
        super().__init__(inner)
        self.prepo_exec: dict[str, int] = {}
        self.prepo_offered = 0
        self.expired_exec = 0

    def decide(self, sim, dp, gen_by):
        assign = super().decide(sim, dp, gen_by)
        for c in dp.crane_ids:
            for g in gen_by[c].items:
                if g.job_ref is not None and \
                        cand_mod.prepo_bound_jid(g.job_ref.job_id or "") is not None:
                    self.prepo_offered += 1
        for c in dp.crane_ids:
            ref = getattr(assign[c], "job_ref", None)
            jid = cand_mod.prepo_bound_jid(getattr(ref, "job_id", "") or "")
            if jid is not None:
                self.prepo_exec[jid] = self.prepo_exec.get(jid, 0) + 1
                j = sim.jobs.get(jid)
                if j is None or j.status.name != "PLANNED":
                    self.expired_exec += 1          # 도착·소멸 후 실행 = 만료 위반
                if cand_mod.PREPO_ONE_SHOT:          # YR-142: one-shot 이력 기록
                    if not hasattr(sim, "_prepo_history"):
                        sim._prepo_history = set()
                    sim._prepo_history.add(jid)
        return assign


def _episode(cell, seed, mk_policy, bound: bool, one_shot: bool = False) -> dict:
    prev_b, prev_o = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = bound, one_shot
    try:
        sim = _sim_contract(cell, seed)
        rec = _PrepoRecorder(mk_policy())
        r = run_joint_episode(sim, rec, RC_EVAL, generator=CandidateGenerator())
    finally:
        cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = prev_b, prev_o
    healthy = True
    try:
        assert_healthy_action_mix(r["_mix"], label=f"{cell}/s{seed}")
    except ActionMixError:
        healthy = False
    d = r["_mix"].as_dict()
    # 원자료는 원정밀도 저장 — 반올림은 보고 출력 전용 (17차 감사)
    return {"cell": cell, "seed": seed, "healthy": healthy,
            "v2_total": _v2_hard_total(sim), "v1_total": r["total_cost"],
            "a2o_min": _a2o_mean_min(sim), "berth_over_min": r["berth_overrun_min"],
            "compl": r["completion_rate"], "backlog": r["backlog"],
            "shares": d["shares"],
            "prepo_repeat": sum(1 for v in rec.prepo_exec.values() if v >= 2),
            "prepo_expired": rec.expired_exec,
            "prepo_offered": rec.prepo_offered,
            "prepo_exec_total": sum(rec.prepo_exec.values()),
            "prepo_blocked": getattr(sim, "_prepo_blocked", 0),
            "prepo_dup_removed": getattr(sim, "_prepo_dup_removed", 0)}


def _mk_ppo(root: Path, ts: int):
    ck = torch.load(root / f"ppo_s{ts}" / "net.pt", map_location="cpu")
    actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])

    def mk():
        y88.FORBID_WAIT = True
        return y88.RLPolicy(actor, norm, name=f"{root.name}:{ts}")
    return mk


def train(ts: int, out_root: Path = OUT):
    prev = cand_mod.BOUND_REPO
    cand_mod.BOUND_REPO = True
    try:
        return train_one(ts, out_root=out_root)
    finally:
        cand_mod.BOUND_REPO = prev


def evaluate(out_root: Path = OUT, offset: int = 3200,
             b_root: Path | None = None) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    b_root = out_root if b_root is None else b_root
    eval_eps = [(c, BASE[c] + offset + i) for c in CELLS for i in range(3)]
    fingerprints = {f"{c}:{s}": realization_hash(_sim_contract(c, s).scenario)
                    for c, s in eval_eps}
    print(f"[eval] SF {len(eval_eps)}", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                   bound=False) for c, s in eval_eps]
    rows: dict[str, list[dict]] = {}
    for arm, root, bound in (("A", OUT140, False), ("B", b_root, True)):
        for ts in TRAIN_SEEDS:
            print(f"[eval] {arm}:{ts}", flush=True)
            mk = _mk_ppo(root, ts)
            rows[f"{arm}:{ts}"] = [_episode(c, s, mk, bound=bound) for c, s in eval_eps]
    per_seed = {}
    for ts in TRAIN_SEEDS:
        A, B = rows[f"A:{ts}"], rows[f"B:{ts}"]
        per_seed[ts] = {
            "B_vs_SF_v2": round(fmean(b["v2_total"] - s["v2_total"]
                                      for b, s in zip(B, sf)), 3),
            "B_minus_A_v2": round(fmean(b["v2_total"] - a["v2_total"]
                                        for b, a in zip(B, A)), 3),
            "B_minus_A_berth": round(fmean(b["berth_over_min"] - a["berth_over_min"]
                                           for b, a in zip(B, A)), 2),
            "B_minus_A_v1": round(fmean(b["v1_total"] - a["v1_total"]
                                        for b, a in zip(B, A)), 3),
            "compl_min": min(r["compl"] for r in B),
            "backlog_max": max(r["backlog"] for r in B),
            "repo_dom": sum(1 for r in B if r["shares"].get("REPOSITION", 0) > 0.60),
            "repo_share_mean": round(fmean(r["shares"].get("REPOSITION", 0)
                                           for r in B), 3),
            "prepo_repeat": sum(r["prepo_repeat"] for r in B),
            "prepo_expired": sum(r["prepo_expired"] for r in B)}
    v = per_seed.values()
    j = {"J1": all(x["compl_min"] >= 1.0 and x["backlog_max"] == 0 for x in v),
         "J2": all(x["repo_dom"] == 0 for x in v),
         "J3": sum(1 for x in v if x["B_vs_SF_v2"] < 0) >= 2,
         "J4": sum(1 for x in v if x["B_minus_A_v2"] <= 0) >= 2,
         "J5": sum(1 for x in v if x["B_minus_A_berth"] <= 0) >= 2,
         "J6": sum(1 for x in v if x["B_minus_A_v1"] <= 0) >= 2,
         "J7": all(x["prepo_repeat"] == 0 and x["prepo_expired"] == 0 for x in v)}
    j["success"] = all(j.values())
    judgment = {**j, "per_seed": per_seed}
    res = {"repro": repro_stamp(
               experiment=f"v4-B 구속적 PREPOSITION — 3군 판정 (미열람 BASE+{offset})",
               seeds={"train": list(TRAIN_SEEDS),
                      **{c: [BASE[c] + offset + i for i in range(3)] for c in CELLS}},
               profile_id="calibrated",
               prereg="유일 변경 = BOUND_REPO(결속·근접 소멸·만료 내재·탈출 분리). 판정 "
                      "7항: J1 완주·backlog / J2 장악 0 / J3 vs SF ≥2/3 / J4 B−A v2 ≤0 "
                      "≥2/3 / J5 본선 비열등 ≥2/3 / J6 v1 비열등 ≥2/3 / J7 반복·만료 후 "
                      "이동 0. 실현 지문 동봉.",
               extra={"n_eval": len(EVAL_EPS)}),
           "band_fingerprints": fingerprints,
           "sf": sf, "arms": rows, "judgment": judgment}
    (out_root / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(j, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--eval-offset", type=int, default=3200)
    a = ap.parse_args()
    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=True)
    if a.train:
        train(a.train, out_root=root)
    if a.eval:
        evaluate(out_root=root, offset=a.eval_offset)
    print("DONE")
