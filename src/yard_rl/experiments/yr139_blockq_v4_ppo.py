"""YR-139 — BlockQ-v4-A: 중앙 공동후보 PPO (학습방식 단일축, 13차 피드백·동결).

■ 계약
- **유일 변경 = 학습방식** (오프라인 Q 회귀 → 자기 궤적 on-policy PPO). 상태 인코딩
  (250·build_rows)·후보 생성·안전 resolver(mask 는 열거 단계에서 선제거)·계약 물리·
  전략 WAIT 제외(FORBID 유지)는 현행 그대로.
- Actor: 공동후보 행 → logit → softmax 선택확률 (블록 전체 = 단일 에이전트 — 크레인별
  신용 분해 불필요). Critic: V(상태 구획 174차원) — 행동을 고르지 않는 공통분 기준선.
- 보상 = 결정 구간의 **실제 발생비용** r_k = −(Φ(t_{k+1}) − Φ(t_k)).
  Φ(t) = v2 실현 hard 총비용의 시점-t 검열판 (관측 gate-in 트럭: O 실현(≤t) 아니면 t
  검열 / 적하: F 실현 아니면 t 검열·ρ=10). **핵심 등식(테스트 고정)**:
  Σ_k 구간비용 = Φ(end) − Φ(0) = 에피소드 평가 총비용 — 학습 비용 ≡ 평가 비용.
  (정의 통일 고지: 출문·완료가 end 밖이면 end 검열 — YR-138 판과 미세 차이.)
- γ=1 (유한 에피소드 총비용 정합) · GAE λ=0.95 (Critic 잡음 감소 용) · clip 0.2 ·
  lr 3e-4 · 엔트로피 0.01 — PPO 표준 앵커(원 논문 계열, 튜닝 아님). Advantage 정규화.
  체크포인트 선택 없이 **최종 정책 사용** (선택 누출·교락 제거).
- 학습: 초기화 3 (88000/99000/123000) × 60 iteration × 8 에피소드 (4셀 혼합·train
  회전 시드 BASE+0..15). 평가: 미열람 BASE+2600..2602 (12 ep)·argmax 결정론.
- 판정(동결): ①완주 100%·backlog 0 ②3 초기화 중 ≥2 에서 v2 실현 총비용 짝 평균 < 0
  (방향 — 유의성은 잠금평가 몫) ③WAIT·REPO 장악 0. **신호 없으면 PPO 트랙도 중단.**
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import fmean

import torch
from torch import nn
from torch.distributions import Categorical

from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_curve_v2 import j_truck_realized, j_vessel_realized
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from ..integrated.vessel import VesselWorkType
from .yr088_joint_rl import LEVEL, RC as RC_TRAIN, build_rows
from .yr090_dense_vessel import BASE, CELLS, SCALE
from .yr135_advantage_q import CTX_A, CTX_B
from .yr136_softplus_contract import _sim_contract

OUT = Path("outputs/reports/yr139_blockq_v4_ppo")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
N_ITER, EPS_PER_ITER, SPC = 60, 8, 16
CLIP, LR, ENT, LAM = 0.2, 3e-4, 0.01, 0.95
SLA_ANCHOR_S = 300.0 + 180.0 + 300.0     # L_T = 이 값 + profile SLA


class Critic(nn.Module):
    def __init__(self, in_dim: int = (CTX_A[1] - CTX_A[0]) + (CTX_B[1] - CTX_B[0])):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, s):
        return self.v(s).squeeze(-1)


def _state_vec(row: list[float]) -> list[float]:
    return row[CTX_A[0]:CTX_A[1]] + row[CTX_B[0]:CTX_B[1]]


def phi_v2(sim, t: float | None = None) -> float:
    """시점-t 검열 v2 실현 hard 총비용 — 보상·평가 공용 (등식 계약의 정의)."""
    t = float(sim.now if t is None else t)
    sla = float(sim.profile.long_wait_sla_s)
    l_t = SLA_ANCHOR_S + sla
    tot = 0.0
    tl = getattr(sim, "time_ledger", None)
    if tl is not None:
        for r in tl.records.values():
            a = getattr(r, "gate_in", None)
            if a is None or a > t + 1e-9:
                continue
            o = getattr(r, "gate_out", None)
            o_eff = o if (o is not None and o <= t + 1e-9) else t
            tot += j_truck_realized(o_eff, a, a + l_t)
    for v in sim.vessels.values():
        if v.work_type != VesselWorkType.LOAD:
            continue
        p = v.plan.planned_completion_s
        if p is None:
            continue
        f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
        f_eff = f if (f is not None and f <= t + 1e-9) else t
        tot += j_vessel_realized(f_eff, p)
    return tot


def run_episode(actor, critic, norm, cell: str, seed: int, rng: random.Random,
                sample: bool = True):
    """1 에피소드 — transitions [(rows, act, logp, value, reward)] + 통계. 등식 보장 구조."""
    sim = _sim_contract(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=True)      # 전략 WAIT 제외 (계약)
    trans = []
    phi_prev = phi_v2(sim)
    phi0 = phi_prev
    pend = None
    dp = sim.run_until_decision()
    k = 0
    while dp is not None:
        phi_now = phi_v2(sim)
        if pend is not None:
            pend[4] = -(phi_now - phi_prev)
            trans.append(tuple(pend))
            pend = None
        phi_prev = phi_now
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            x = torch.tensor(rows, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = actor(x)
                logits = -logits                     # score 관례(작을수록 좋음) → logit
                dist = Categorical(logits=logits)
                act = (int(dist.sample()) if sample and len(assigns) > 1
                       else int(torch.argmax(logits)))
                logp = float(dist.log_prob(torch.tensor(act)))
                val = float(critic(torch.tensor([_state_vec(rows[0])],
                                                dtype=torch.float32))[0])
            pend = [rows, act, logp, val, 0.0]
            _apply(sim, assigns[act])
        dp = sim.run_until_decision()
        k += 1
    phi_end = phi_v2(sim, sim.end)
    if pend is not None:
        pend[4] = -(phi_end - phi_prev)
        trans.append(tuple(pend))
    else:
        if trans:
            r0, a0, l0, v0, rw = trans[-1]
            trans[-1] = (r0, a0, l0, v0, rw - (phi_end - phi_prev))
    jobs = list(sim.jobs.values())
    compl = (sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)) if jobs else 1.0
    backlog = sim.unfinished_backlog() if hasattr(sim, "unfinished_backlog") else \
        sum(1 for j in jobs if j.status.name != "DONE")
    return trans, {"total": phi_end - phi0, "compl": round(compl, 4),
                   "backlog": backlog, "phi0": phi0}


def _gae(trans):
    """γ=1·GAE(λ) — returns(총비용 정합)·advantage."""
    n = len(trans)
    adv, ret = [0.0] * n, [0.0] * n
    running_ret, running_adv, next_v = 0.0, 0.0, 0.0
    for i in range(n - 1, -1, -1):
        r, v = trans[i][4], trans[i][3]
        running_ret = r + running_ret
        ret[i] = running_ret
        delta = r + next_v - v
        running_adv = delta + LAM * running_adv
        adv[i] = running_adv
        next_v = v
    return adv, ret


def train_one(ts: int) -> Path:
    out = OUT / f"ppo_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])
    torch.manual_seed(ts)
    actor, critic = JointPairNet(250), Critic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)
    rng = random.Random(ts)
    cells = list(CELLS)
    t0 = time.time()
    curve = []
    for it in range(1, N_ITER + 1):
        batch, totals = [], []
        for e in range(EPS_PER_ITER):
            cell = cells[(it * EPS_PER_ITER + e) % len(cells)]
            seed = BASE[cell] + rng.randrange(SPC)
            trans, st = run_episode(actor, critic, norm, cell, seed, rng, sample=True)
            totals.append(st["total"])
            if trans:
                adv, ret = _gae(trans)
                for (rows, act, logp, _v, _r), a_, r_ in zip(trans, adv, ret):
                    batch.append((rows, act, logp, a_, r_))
        if not batch:
            continue
        advs = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)
        rets = torch.tensor([b[4] for b in batch], dtype=torch.float32) / SCALE
        idx_all = list(range(len(batch)))
        for _ in range(4):                                        # PPO epochs
            rng.shuffle(idx_all)
            for s0 in range(0, len(idx_all), 64):
                mb = idx_all[s0:s0 + 64]
                loss_pi, loss_v, ent = 0.0, 0.0, 0.0
                for i in mb:
                    rows, act, logp_old, _a, _r = batch[i]
                    x = torch.tensor(rows, dtype=torch.float32)
                    logits, _ = actor(x)
                    dist = Categorical(logits=-logits)
                    logp = dist.log_prob(torch.tensor(act))
                    ratio = torch.exp(logp - logp_old)
                    a_i = advs[i]
                    # 표준 PPO clipped surrogate — r=−비용 이므로 A 는 이미 보상 기준
                    loss_pi = loss_pi - torch.min(
                        ratio * a_i, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_i)
                    v = critic(torch.tensor([_state_vec(rows[0])], dtype=torch.float32))[0]
                    loss_v = loss_v + (v - rets[i]) ** 2
                    ent = ent + dist.entropy()
                nmb = len(mb)
                loss = loss_pi / nmb + 0.5 * loss_v / nmb - ENT * ent / nmb
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters())
                                         + list(critic.parameters()), 1.0)
                opt.step()
        curve.append({"iter": it, "mean_total": round(fmean(totals), 3)})
        if it % 5 == 0 or it == 1:
            print(f"[ppo s{ts} it{it}] mean v2 total {fmean(totals):.2f}", flush=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "in_dim": 250, "train_seed": ts}, out / "net.pt")
    (out / "curve.json").write_text(json.dumps(
        {"curve": curve, "wall_s": round(time.time() - t0, 1)}), encoding="utf-8")
    print(f"[ppo s{ts}] 완료 {time.time() - t0:.0f}s", flush=True)
    return out / "net.pt"


def evaluate() -> dict:
    from .yr138_episode_pilot import _episode
    from . import yr088_joint_rl as y88
    eval_eps = [(c, BASE[c] + 2600 + i) for c in CELLS for i in range(3)]
    print(f"[eval] SF {len(eval_eps)}", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"))
          for c, s in eval_eps]
    rows = {}
    for ts in TRAIN_SEEDS:
        ck = torch.load(OUT / f"ppo_s{ts}" / "net.pt", map_location="cpu")
        actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
        ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                         / "rl_net.pt", map_location="cpu")
        norm = StateNorm(refs=ck0["norm_refs"])

        def mk(a=actor, n=norm, t=ts):
            y88.FORBID_WAIT = True
            return y88.RLPolicy(a, n, name=f"PPO:{t}")
        print(f"[eval] PPO:{ts}", flush=True)
        rows[ts] = [_episode(c, s, mk) for c, s in eval_eps]
    per_seed = {}
    for ts in TRAIN_SEEDS:
        rr = rows[ts]
        d = [a["v2_total"] - s["v2_total"] for a, s in zip(rr, sf)]
        per_seed[ts] = {
            "d_v2_mean": round(fmean(d), 3),
            "compl_min": min(r["compl"] for r in rr),
            "backlog_max": max(r["backlog"] for r in rr),
            "wait_dom": sum(1 for r in rr if r["shares"].get("WAIT", 0) > 0.60),
            "repo_dom": sum(1 for r in rr if r["shares"].get("REPOSITION", 0) > 0.60),
            "d_a2o_mean": round(fmean(a["a2o_min"] - s["a2o_min"]
                                      for a, s in zip(rr, sf)), 3)}
    g0 = all(v["compl_min"] >= 1.0 and v["backlog_max"] == 0 for v in per_seed.values())
    n_improve = sum(1 for v in per_seed.values() if v["d_v2_mean"] < 0)
    dom0 = all(v["wait_dom"] == 0 and v["repo_dom"] == 0 for v in per_seed.values())
    judgment = {"G0_all": g0, "n_improve_direction": n_improve, "dom_zero": dom0,
                "success": bool(g0 and n_improve >= 2 and dom0), "per_seed": per_seed}
    res = {"repro": repro_stamp(
               experiment="YR-139 v4-A 중앙 공동후보 PPO — 판정 (미열람 2600 대역)",
               seeds={"train": list(TRAIN_SEEDS),
                      **{c: [BASE[c] + 2600 + i for i in range(3)] for c in CELLS}},
               profile_id="calibrated",
               prereg="유일 변경 = 학습방식(PPO). 보상 등식(Σ 구간비용 = 평가 총비용) "
                      "테스트 고정. 판정: 완주 100%∧backlog 0 ∧ ≥2/3 초기화 v2 총비용 "
                      "짝 평균 < 0 ∧ WAIT·REPO 장악 0. 신호 없으면 PPO 트랙 중단.",
               extra={"iters": N_ITER, "eps_per_iter": EPS_PER_ITER,
                      "clip": CLIP, "lam": LAM}),
           "sf": sf, "arms": {str(k): v for k, v in rows.items()}, "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(judgment, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_one(a.train)
    if a.eval:
        evaluate()
    print("DONE")
