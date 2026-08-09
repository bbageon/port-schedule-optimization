"""YR-151 — TransferHead PPO 학습 루프 (아키텍처 선구현 · 테스트 유예 2026-08-09).

■ 구조 (대화로 확정된 설계 전부 반영)
  환경: 고정 WIP 21블록(H-21 YT) + 사전 통지 lead=1800s (판매 창) + 60초 검토 격자
  결정: 각 블록 PPO(공유 가중치 1벌) — [KEEP, 후보1..K] 계획 진단, B-(b)
  집행: UnifiedSellOrchestrator — 축 저울(공간 20곳 ∪ 시간 +Δ)·비용 통화·순열 불변
  보상: (i) 실현 전역 증분 — Φ 를 **행동 직전**에 기록하고(감사 치명 2 정정 —
        구판은 판매 확정 뒤에 기록해 판매의 즉시 비용이 자기 보상에서 빠졌다)
        r_k = −(Φ(t_{k+1}⁻) − Φ(t_k⁻)), 마지막은 관측 종료 시점 Φ_final 로 닫는다.
        Φ = Σ블록 검열 v2 실현비용 + route/3600 + 기사 외부 대기/3600
        (떠넘긴 비용 자동 포함 = 이기적 판매 + 가격 청구 동치. 학습 잣대 ≡ 평가 잣대)
  갱신: PPO clip 0.2 · γ=1 · lr 3e-4 (yr139 앵커 승계). advantage = **MC 총수익 −
        critic 기준선**(= GAE λ=1) — 감사 치명 3 정정: 구 문서의 "GAE λ=0.95" 는
        코드에 없던 주장이라 삭제. λ<1 GAE·관측종료 bootstrap 은 잔여 설계
        (검열 Φ 가 종료 시점 미완 비용을 부분 반영하나 종료 후 파급은 미포함 —
        0B 사전등록 전 결정 필요).

■ lead 해석 가정 (0B 사전등록 동결 대상 — 아직 사용자 미확정)
  "확약 총량(내부 + 통지 pipeline) = L" 해석을 기본으로 한다. 물리적 내부 대수는
  진행 중 통지분만큼 L 보다 작다. 대안(L 상향)은 config 교체로 가능.

■ ExecutionHead = 채택 실행 PPO (2026-08-09 연결 — 사용자 지시)
  기본값이 **채택 구성 그대로**다: YR-143 confirm C0 체크포인트(JointPairNet 250) +
  YR-125 StateNorm + 대기 허가증(DeployGuard) + C0 후보 플래그(SAFETY_ONLY·DEFER_ALL·
  one-shot). 가중치는 **동결**(requires_grad=False)이고 매 iteration 전후 해시를 비교해
  불변을 검사한다. TransferHead 만 별도 optimizer 로 학습한다 — 연구 질문이
  "채택 실행 PPO **위에** SELL 을 얹으면 비용이 더 주는가"이므로, SF-SPT 로 돌리면
  다른 질문("SF 운영에서 SELL 이 유효한가")이 된다. SF 는 배선 디버그 전용 opt-in.
■ 미구현 정직 고지: 이 파일은 실행하지 않았다(빌드 우선) — 판정은 shadow → 0B 뒤에만.
  대기 허가증 rollout 이 블록 21개 × 결정마다 돌므로 실행 비용이 크다(실측 미확인).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from ..integrated import candidates as cand_mod
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_curve_v2 import KappaFit
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty, repro_stamp
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (ObservationContract,
                                          WipAdmissionController, admission_epochs,
                                          build_fixed_wip)
from ..integrated.time_sell import deferral_ledger
from ..integrated.transfer_head import (PpoSellPolicy, TransferActor, TransferCritic)
from ..integrated.yard_layout import terminal_layout
from . import yr088_joint_rl as y88
from .yr088_joint_rl import LEVEL, RLPolicy
from .yr100_candidate_eval import RC_EVAL
from .yr139_blockq_v4_ppo import phi_v2
from .yr143_no_repo import ARM_FLAGS, CONFIRM_TS, NORM_TS
from .yr146_deploy_guard import DeployGuard
from .yr149_load_cells import _sim_from

OUT = Path("outputs/reports/yr151_transfer_ppo")
KAPPA = Path("outputs/reports/yr136_softplus_contract/kappa_fit_v2p.json")
# 채택 실행 정책 (YR-148 판정 그대로): YR-143 confirm C0 + YR-125 정규화 통계.
EXEC_CKPT_DIR = Path("outputs/reports/yr143_no_repo/confirm/c0")
NORM_CKPT = Path("outputs/reports/yr125_diff_credit") / f"diff1_s{NORM_TS}" / "rl_net.pt"
EXEC_TS_DEFAULT = CONFIRM_TS[0]          # 221000 — 사전등록에서 초기화 선택을 동결한다
WIP_TARGET = 100                 # 학습 기본 셀 (사전등록에서 동결)
CLIP, LR, ENT, GAMMA = 0.2, 3e-4, 0.01, 1.0     # yr139 앵커 승계
# advantage = MC 총수익 − critic 기준선 (= GAE λ=1). λ<1 은 도입하지 않았다(감사 정정).
N_ITER, EPS_PER_ITER = 40, 4
TRAIN_SEEDS = (7_100_000, 7_200_000, 7_300_000)


def load_kf() -> KappaFit:
    d = json.loads(KAPPA.read_text(encoding="utf-8"))
    return KappaFit(**{k: v for k, v in d.items()
                       if k in KappaFit.__dataclass_fields__})


# ------------------------------------------------------------------ 채택 실행 정책 연결
def actor_state_hash(net) -> str:
    """가중치 불변 검사용 해시 — 같은 런 안에서 전/후 비교 목적(재현 스탬프에 기록)."""
    import hashlib
    import io as _io
    buf = _io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in sorted(net.state_dict().items())}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_adopted_execution_head(ts: int = EXEC_TS_DEFAULT):
    """채택 실행 PPO 를 **동결 로딩** — YR-146/148 판정 경로와 동일한 조립.

    JointPairNet(250) actor(YR-143 confirm C0) + StateNorm(YR-125). 가중치는
    requires_grad=False 로 동결하고, 불변 검사는 actor_state_hash 로 한다.
    체크포인트가 없으면 fail-closed — SF 로 조용히 대체하지 않는다.
    """
    ckpt = EXEC_CKPT_DIR / f"ppo_s{ts}" / "net.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"채택 실행 체크포인트 없음: {ckpt} — SF 로 조용히 대체하지 않는다"
            " (배선 디버그는 exec_head='sf' 를 명시적으로)")
    ck = torch.load(ckpt, map_location="cpu")
    actor = JointPairNet(250)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)                    # 동결 — TransferHead 만 학습
    ck0 = torch.load(NORM_CKPT, map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])
    return actor, norm


class AdoptedExecFleet:
    """블록별 실행 정책 인스턴스(가중치 공유·동결) — C0 + 대기 허가증.

    DeployGuard 는 허가증 원장 등 상태를 갖므로 **블록마다 별도 인스턴스**를 만들되
    actor·norm 은 한 벌을 공유한다(21블록 동일 정책 배포 구조 그대로).
    """

    def __init__(self, actor, norm):
        self.actor, self.norm = actor, norm
        self._by_sim: dict[int, DeployGuard] = {}

    def get(self, sim) -> DeployGuard:
        g = self._by_sim.get(id(sim))
        if g is None:
            inner = RLPolicy(self.actor, self.norm, name="exec:c0")
            jr = JointRolloutGreedy(RC_EVAL, horizon_s=1800.0,
                                    generator=CandidateGenerator(),
                                    forbid_strategic_wait=True)
            g = DeployGuard(inner, self.actor, self.norm, jr)
            self._by_sim[id(sim)] = g
        return g


# ------------------------------------------------------------------ Φ (전역·검열)
def phi_terminal(mbt, t: float) -> float:
    """터미널 전체 시점-t 검열 실현비용 — 블록 합 + 주행 + 기사 외부 대기.

    전역이므로 판매로 옮겨진 비용(수신 블록 부담·주행·외부 대기)이 전부 포함된다.
    """
    tot = sum(phi_v2(sim, t) for sim in mbt.blocks.values())
    tot += mbt.route_cost_s / 3600.0
    for row in deferral_ledger(mbt):
        appt, a = row["original_appointment_s"], row["actual_gate_in_s"]
        if appt is not None and a is not None:
            tot += max(0.0, min(a, t) - appt) / 3600.0     # 외부 대기 — 동일 단가·검열
    return tot


class PhiRecorder:
    """epoch 마다 Φ 를 기록 — 구간 보상 r_k = −(Φ_{k+1} − Φ_k) 의 원료."""

    def __init__(self):
        self.samples: list[tuple[float, float]] = []

    def review(self, mbt, t: float) -> None:
        self.samples.append((t, phi_terminal(mbt, t)))


class _Chain:
    """review 콜러블 합성 — 투입 → 판매 검토 → Φ 기록 순서."""

    def __init__(self, *parts):
        self.parts = parts

    def review(self, mbt, t: float) -> None:
        for p in self.parts:
            p.review(mbt, t)


# ------------------------------------------------------------------ 에피소드
def run_episode(seed: int, policy: PpoSellPolicy, kf: KappaFit, *,
                wip: int = WIP_TARGET,
                obs: ObservationContract | None = None,
                exec_head: str = "adopted",
                exec_fleet: "AdoptedExecFleet | None" = None) -> dict:
    """고정 WIP + lead 통지 환경에서 1 에피소드 — 정책 trail 과 구간 Φ 를 수집.

    exec_head:
      "adopted" (기본) — 채택 실행 PPO(C0+대기 허가증) 동결본. 연구 질문의 전제.
      "sf"              — 배선 디버그 전용(명시 opt-in). 이 결과로 성능 주장 금지.
    """
    obs = obs or ObservationContract()
    layout = terminal_layout()
    built = build_fixed_wip(build_h21_profile(), seed, wip_target=wip, obs=obs,
                            layout=layout)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    ctrl = WipAdmissionController(built["pool"], wip_target=wip,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=obs.observe_s)
    orch = UnifiedSellOrchestrator(policy, layout, kf)
    rec = PhiRecorder()

    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}
    if exec_head == "adopted":
        if exec_fleet is None:
            exec_fleet = AdoptedExecFleet(*load_adopted_execution_head())

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), CandidateGenerator())
            gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
            try:
                _apply(sim, exec_fleet.get(sim).decide(sim, dp, gb))
            except Exception:
                exc["n"] += 1
                _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
    else:                                       # "sf" — 배선 디버그 전용
        pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), CandidateGenerator())
            gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
            try:
                _apply(sim, pol.decide(sim, dp, gb))
            except Exception:
                exc["n"] += 1
                _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    # 채택 구성의 후보 플래그 (C0 + 유한 DEFER + one-shot) — YR-146 판정 경로와 동일.
    # 전역 monkey-patch 라 반드시 원상복구한다 (구조 결함 자체는 YR-160 이 고친다).
    f = ARM_FLAGS["c0"]
    prev = (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY,
            cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT)
    cand_mod.WAIT_MODE = "DEFER_ALL"
    cand_mod.SAFETY_ONLY = f["safety_only"]
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = f["bound"], True
    try:
        # ★Φ 는 **행동 직전** 기록(감사 치명 2): 투입(비용 무영향·통지 미래) → Φ 기록 →
        #   판매 확정 순서. 판매의 즉시 비용은 다음 구간에 잡혀 자기 보상에 귀속된다.
        mbt.run(exec_policy, review_fn=_Chain(ctrl, rec, orch).review)
    finally:
        (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY,
         cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT) = prev
    phi_final = phi_terminal(mbt, obs.observe_s)          # 관측 종료 시점 — 구간의 끝
    return {"phi": rec.samples, "phi_final": phi_final, "sell_ledger": orch.ledger,
            "n_space": orch.n_space, "n_time": orch.n_time,
            "policy_exceptions": exc["n"], "admitted": ctrl.n_admitted}


# ------------------------------------------------------------------ 전이 조립
def build_batch(trail: list[dict], phi: list[tuple[float, float]],
                phi_final: float) -> list[dict]:
    """결정별 (총수익 R, advantage 원료) 조립 — γ=1 전역 보상.

    Φ 표본은 **행동 직전** 값이므로(치명 2 정합), 결정 시각 t_k 의 reward-to-go =
    −(Φ_final − Φ(t_k⁻)) 는 그 결정의 즉시 비용부터 관측 종료까지를 포함한다.
    advantage = R − V(s) (MC − critic 기준선 = GAE λ=1. λ<1·bootstrap 은 잔여).
    """
    if not phi:
        return []
    times = [p[0] for p in phi]
    values = [p[1] for p in phi]
    out = []
    for tr in trail:
        idx = max(i for i, tt in enumerate(times) if tt <= tr["t"] + 1e-9)
        r2go = -(phi_final - values[idx])
        out.append({**tr, "ret": r2go, "adv": r2go - tr["value"]})
    return out


# ------------------------------------------------------------------ PPO 갱신
def ppo_update(actor: TransferActor, critic: TransferCritic,
               opt_a, opt_c, batch: list[dict], *, epochs: int = 4) -> dict:
    if not batch:
        return {"n": 0}
    advs = torch.tensor([b["adv"] for b in batch], dtype=torch.float32)
    adv_n = (advs - advs.mean()) / (advs.std() + 1e-6)
    stats = {"n": len(batch), "pi_loss": 0.0, "v_loss": 0.0}
    for _ in range(epochs):
        pi_loss = torch.tensor(0.0)
        v_loss = torch.tensor(0.0)
        for b, a_n in zip(batch, adv_n):
            logits = actor(b["rows"])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor(b["action"]))
            ratio = torch.exp(logp - b["logp"])
            pi_loss = pi_loss - torch.min(
                ratio * a_n, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_n)
            pi_loss = pi_loss - ENT * dist.entropy()
            v_loss = v_loss + (critic(b["critic_in"]) - b["ret"]) ** 2
        opt_a.zero_grad(); (pi_loss / len(batch)).backward(); opt_a.step()
        opt_c.zero_grad(); (v_loss / len(batch)).backward(); opt_c.step()
        stats["pi_loss"] = float(pi_loss.item() / len(batch))
        stats["v_loss"] = float(v_loss.item() / len(batch))
    return stats


def train_one(ts: int, *, out_root: Path = OUT,
              exec_ts: int = EXEC_TS_DEFAULT) -> Path:
    """초기화 1개 학습 — shadow 아님(on-policy). 실행은 디버깅 국면 후 판정 절차로만."""
    kf = load_kf()
    # ★재현성 정정(감사): 시드를 신경망 생성 **전**에 고정 — 초기 가중치까지 시드 귀속.
    torch.manual_seed(ts)
    actor, critic = TransferActor(), TransferCritic()
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    # 채택 실행 정책 — 동결 로딩 1회, 매 iteration 뒤 hash 불변 검사 (계약 가드).
    exec_actor, exec_norm = load_adopted_execution_head(exec_ts)
    exec_hash0 = actor_state_hash(exec_actor)
    hist = []
    for it in range(N_ITER):
        batch_all: list[dict] = []
        for e in range(EPS_PER_ITER):
            policy = PpoSellPolicy(actor, critic, mode="live", sample=True,
                                   seed=ts + it * 100 + e)
            ep = run_episode(ts + it * EPS_PER_ITER + e, policy, kf,
                             exec_fleet=AdoptedExecFleet(exec_actor, exec_norm))
            # critic 입력은 결정 시점에 policy.trail["critic_in"] 으로 저장됨(치명 1 정정)
            batch_all += build_batch(policy.trail, ep["phi"], ep["phi_final"])
        stats = ppo_update(actor, critic, opt_a, opt_c, batch_all)
        if actor_state_hash(exec_actor) != exec_hash0:      # 실행 head 불변 — 매 iter 검사
            raise RuntimeError("ExecutionHead 가중치가 변했다 — 학습 분리 계약 위반")
        hist.append({"iter": it, **stats})
    # ★저장 순서 정정(감사 치명 4): 재현 스탬프를 **먼저** 만들고(실패하면 아무 파일도
    #   안 남음) → 스탬프 포함 train.json → net.pt. 구판은 net.pt 를 먼저 쓰고
    #   repro_stamp(experiment= 누락)에서 TypeError — 재현 증거 없는 반쪽 산출물이 남았다.
    stamp = repro_stamp(
        experiment="YR-151 TransferHead PPO (골격 — 판정 실행은 shadow·0B 관문 뒤에만)",
        seeds={"train": [ts], "exec_head": [exec_ts]},
        params={"WIP_TARGET": WIP_TARGET, "N_ITER": N_ITER,
                "LEAD_S": ANNOUNCE_LEAD_S,
                "exec_head": f"adopted C0+guard (yr143 confirm ppo_s{exec_ts})",
                "exec_head_hash": exec_hash0,
                "anchors": "yr139 승계(clip/lr/ent/γ) · advantage = MC−V(λ=1)"},
        prereg=".claude/docs/dashboard-task-specs/YR-151-block-ppo-sell-head.md")
    out = out_root / f"ppo_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps(
        {"history": hist, "code_dirty": bool(code_dirty()), "stamp": stamp},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
               out / "net.pt")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ts", type=int, default=0, help="TRAIN_SEEDS 의 index")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        train_one(TRAIN_SEEDS[a.train_ts])
    print("DONE")
