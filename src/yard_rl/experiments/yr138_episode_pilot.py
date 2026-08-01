"""YR-138 — 완주 파일럿: SF-SPT vs B(단일망+이중손실) vs D(V/A+이중손실) (12차 피드백).

■ 왜 (결과 미열람 동결)
YR-137: 순위(J1)·선택 손실(J3)이 최초 3/3 재현 — Block Q 의 실제 역할(후보 선택)과 직결.
J2(절대 크기)는 s88000 1성분 미달이나 후보 선택에는 순서가 우선이므로, 대리척도를 더
고치기보다 **실제 에피소드 비용을 줄이는지**를 바로 확인한다 (s88000 국소 진단 보류 —
실행에서 특정 초기화만 무너지면 그때 진단). J2 는 YR-133 견적 단계에서 재요구.

■ 설계
- 비교군 3: SF-SPT(규칙 기준선) / B(arm4_B — 단일망·회귀+순위) / D(arm4_D — V/A·회귀+순위).
  재학습 없음 — YR-137 체크포인트 그대로 (학습시드 3개씩).
- 물리 = 계약 물리(_sim_contract — 망이 학습한 세계). 평가 대역 = **BASE+2300..2302
  (4셀×3 = 12 에피소드, 미열람)**. 같은 에피소드에서 짝지어 끝까지 실행.
- 지표: **1차 = v2 실현 hard 총비용**(평가 정렬 조항 구현 — 트럭 Σ j_truck_realized(O,A,D_T)
  ·미출문 = end 검열 / 적하 Σ j_vessel_realized(F,P)·미완 = end 검열, ρ=10) ·
  부지표 = v1 RC_EVAL 총비용·평균 A→O·트럭 대기·본선 초과분 · guard = 완주·backlog·
  행동 건전성(WAIT/REPO 장악 재퇴화).
- 판정 (동결): G0 완주 1.0·backlog 0 (하드). 유의성 = 짝지은 차(arm−SF, 36쌍 =
  학습시드 3 × 에피소드 12)의 upper95 < 0.
  · B·D 모두 유의 개선 → **B 후보 지명(단순 우선)**
  · D 만 개선, 또는 D−B upper95 < 0 (실질 우위) → D 지명
  · 둘 다 개선 없음 → **오프라인 순위 개선이 운영성과로 연결 안 됨 — 정책 트랙 중단 선언**
  파일럿이므로 지명 = 진단·개발 단계 (최종 채택은 확대 확증 별도).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import torch
from torch import nn

from ..integrated.baselines import (ActionMixError, ResolverPolicy,
                                    ServiceFirstSPTPreference,
                                    assert_healthy_action_mix, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_curve_v2 import j_truck_realized, j_vessel_realized
from ..integrated.encoding import StateNorm
from ..integrated.evalkit import paired
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from ..integrated.vessel import VesselWorkType
from . import yr088_joint_rl as y88
from .yr090_dense_vessel import BASE, CELLS
from .yr100_candidate_eval import RC_EVAL
from .yr119_wait_retrain import _Recorder, _a2o_mean_min, _set_forbid_wait
from .yr135_advantage_q import OUT as OUT135, JointDuelingNet
from .yr136_softplus_contract import _sim_contract

OUT = Path("outputs/reports/yr138_episode_pilot")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
EVAL_EPS = [(c, BASE[c] + 2300 + i) for c in CELLS for i in range(3)]   # 12 — 미열람
SLA_ANCHOR = (300.0, 180.0, 300.0)   # 입문·서비스·출문 (L_T = 합 + profile SLA)


class _DuelWrap(nn.Module):
    """JointDuelingNet 을 RLPolicy 인터페이스(net(x) → (score, aux))로 — 결정 = 그룹 1개."""

    def __init__(self, d: JointDuelingNet):
        super().__init__()
        self.d = d

    def forward(self, x):
        q = self.d.q_groups(x, [list(range(len(x)))])
        return q, q


def _mk_policy(arm: str, ts: int):
    ck = torch.load(OUT135 / f"arm4_{arm}_s{ts}" / "net.pt", map_location="cpu")
    if ck["arch"] == "single":
        net = JointPairNet(ck["in_dim"])
        net.load_state_dict(ck["state"]); net.eval()
    else:
        d = JointDuelingNet(ck["in_dim"])
        d.load_state_dict(ck["state"]); d.eval()
        net = _DuelWrap(d)
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])

    def mk():
        _set_forbid_wait(False)
        return y88.RLPolicy(net, norm, name=f"{arm}:{ts}")
    return mk


def _v2_hard_total(sim) -> float:
    """평가 정렬 조항 — 실현 hard 총비용 (미완 = end 검열: 미완이 이득 보지 않게)."""
    sla = float(sim.profile.long_wait_sla_s)
    l_t = SLA_ANCHOR[0] + sla + SLA_ANCHOR[1] + SLA_ANCHOR[2]
    tot = 0.0
    tl = getattr(sim, "time_ledger", None)
    if tl is not None:
        for r in tl.records.values():
            a = getattr(r, "gate_in", None)
            if a is None:
                continue
            o = getattr(r, "gate_out", None)
            tot += j_truck_realized(o if o is not None else float(sim.end), a, a + l_t)
    for v in sim.vessels.values():
        if v.work_type != VesselWorkType.LOAD:
            continue
        p = v.plan.planned_completion_s
        if p is None:
            continue
        f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
        tot += j_vessel_realized(f if f is not None else float(sim.end), p)
    return round(tot, 4)


def _episode(cell: str, seed: int, mk_policy) -> dict:
    sim = _sim_contract(cell, seed)
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
            "v2_total": _v2_hard_total(sim),
            "v1_total": round(r["total_cost"], 3), "a2o_min": _a2o_mean_min(sim),
            "wait_min": round(r["mean_wait_min"], 2),
            "berth_over_min": round(r["berth_overrun_min"], 2),
            "compl": r["completion_rate"], "backlog": r["backlog"],
            "shares": d["shares"], "repo_share": d["shares"].get("REPOSITION", 0.0),
            **rec.wait_metrics(sim.end)}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] SF 기준선 {len(EVAL_EPS)}", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"))
          for c, s in EVAL_EPS]
    rows: dict[str, list[dict]] = {}
    for arm in ("B", "D"):
        for ts in TRAIN_SEEDS:
            print(f"[pilot] {arm}:{ts}", flush=True)
            mk = _mk_policy(arm, ts)
            rows[f"{arm}:{ts}"] = [_episode(c, s, mk) for c, s in EVAL_EPS]
    judgment: dict = {"per_arm": {}}
    diffs = {}
    for arm in ("B", "D"):
        allr = [r for ts in TRAIN_SEEDS for r in rows[f"{arm}:{ts}"]]
        d_v2 = [a["v2_total"] - s["v2_total"]
                for ts in TRAIN_SEEDS for a, s in zip(rows[f"{arm}:{ts}"], sf)]
        diffs[arm] = d_v2
        judgment["per_arm"][arm] = {
            "d_v2_vs_sf": paired(d_v2).as_dict(),
            "d_v1_vs_sf": paired([a["v1_total"] - s["v1_total"] for ts in TRAIN_SEEDS
                                  for a, s in zip(rows[f"{arm}:{ts}"], sf)]).as_dict(),
            "d_a2o_vs_sf": paired([a["a2o_min"] - s["a2o_min"] for ts in TRAIN_SEEDS
                                   for a, s in zip(rows[f"{arm}:{ts}"], sf)]).as_dict(),
            "compl_min": min(r["compl"] for r in allr),
            "backlog_max": max(r["backlog"] for r in allr),
            "unhealthy": sum(1 for r in allr if not r["healthy"]),
            "wait_dom": sum(1 for r in allr if r["shares"].get("WAIT", 0) > 0.60),
            "repo_dom": sum(1 for r in allr if r["shares"].get("REPOSITION", 0) > 0.60),
            "strategic_wait_mean": round(fmean(r["strategic_wait_rate"] for r in allr), 4)}
    d_db = [b - a for a, b in zip(diffs["B"], diffs["D"])]     # (D−SF)−(B−SF) = D−B
    judgment["d_D_minus_B_v2"] = paired(d_db).as_dict()
    g0 = {arm: judgment["per_arm"][arm]["compl_min"] >= 1.0
          and judgment["per_arm"][arm]["backlog_max"] == 0 for arm in ("B", "D")}
    imp = {arm: judgment["per_arm"][arm]["d_v2_vs_sf"]["ci"][1] < 0.0 and g0[arm]
           for arm in ("B", "D")}
    d_beats_b = judgment["d_D_minus_B_v2"]["ci"][1] < 0.0
    if imp["B"] and imp["D"]:
        verdict = "D" if d_beats_b else "B"
    elif imp["D"]:
        verdict = "D"
    elif imp["B"]:
        verdict = "B"
    else:
        verdict = "NONE — 오프라인 순위 개선이 운영성과로 미연결 (정책 트랙 중단 후보)"
    judgment.update({"G0": g0, "improved_vs_sf": imp, "nominee": verdict})
    res = {"repro": repro_stamp(
               experiment="YR-138 완주 파일럿 — SF vs B vs D (계약 물리·미열람 12 에피소드)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      **{c: [BASE[c] + 2300 + i for i in range(3)] for c in CELLS}},
               profile_id="calibrated",
               prereg="재학습 없음. 1차 = v2 실현 hard 총비용(짝지은 upper95<0 = 유의 개선)·"
                      "G0 완주 1.0∧backlog 0 하드. 지명: B·D 모두 개선→B(단순 우선), "
                      "D−B upper95<0 이면 D, 둘 다 아니면 트랙 중단 후보. WAIT/REPO 장악 "
                      "재퇴화 감시. 파일럿 — 최종 채택 아님.",
               extra={"n_eval": len(EVAL_EPS)}),
           "sf": sf, "arms": rows, "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps({"G0": g0, "improved": imp, "nominee": verdict}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("DONE")
