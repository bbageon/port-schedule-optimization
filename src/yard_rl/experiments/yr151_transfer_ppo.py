"""YR-151 — TransferHead PPO 학습 루프 (아키텍처 선구현 · 테스트 유예 2026-08-09).

■ 구조 (대화로 확정된 설계 전부 반영)
  환경: 고정 WIP 21블록(H-21 YT) + 사전 통지 lead=1800s (판매 창) + 60초 검토 격자
  결정: 각 블록 PPO(공유 가중치 1벌) — [KEEP, 후보1..K] 계획 진단, B-(b)
  집행: UnifiedSellOrchestrator — 축 저울(공간 20곳 ∪ 시간 +Δ)·비용 통화·순열 불변
  보상: (i) 실현 전역 증분 — r_k = −(Φ(t_{k+1}) − Φ(t_k)),
        Φ = Σ블록 검열 v2 실현비용 + route/3600 + 기사 외부 대기/3600
        (떠넘긴 비용 자동 포함 = 이기적 판매 + 가격 청구 동치. 학습 잣대 ≡ 평가 잣대)
  갱신: PPO clip 0.2 · γ=1 · GAE λ=0.95 · lr 3e-4 (yr139 동결 앵커 승계 — 튜닝 아님)

■ lead 해석 가정 (0B 사전등록 동결 대상 — 아직 사용자 미확정)
  "확약 총량(내부 + 통지 pipeline) = L" 해석을 기본으로 한다. 물리적 내부 대수는
  진행 중 통지분만큼 L 보다 작다. 대안(L 상향)은 config 교체로 가능.

■ 미구현 정직 고지
  · ExecutionHead 는 계약상 채택 PPO(C0+대기허가증) 체크포인트 동결이어야 하나,
    이 골격은 **규칙(SF-SPT)** 을 자리에 두고 hash 검사 자리만 마련했다 — 체크포인트
    배선은 YR-160(채택 구성 단일 정의)과 함께 잔여 작업이다.
  · 이 파일은 실행하지 않았다(빌드 우선 지시) — 판정은 shadow → 0B 관문 뒤에만.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_curve_v2 import KappaFit
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
from .yr088_joint_rl import LEVEL
from .yr139_blockq_v4_ppo import phi_v2
from .yr149_load_cells import _sim_from

OUT = Path("outputs/reports/yr151_transfer_ppo")
KAPPA = Path("outputs/reports/yr136_softplus_contract/kappa_fit_v2p.json")
WIP_TARGET = 100                 # 학습 기본 셀 (사전등록에서 동결)
CLIP, LR, ENT, LAM, GAMMA = 0.2, 3e-4, 0.01, 0.95, 1.0     # yr139 앵커 승계
N_ITER, EPS_PER_ITER = 40, 4
TRAIN_SEEDS = (7_100_000, 7_200_000, 7_300_000)


def load_kf() -> KappaFit:
    d = json.loads(KAPPA.read_text(encoding="utf-8"))
    return KappaFit(**{k: v for k, v in d.items()
                       if k in KappaFit.__dataclass_fields__})


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
                obs: ObservationContract | None = None) -> dict:
    """고정 WIP + lead 통지 환경에서 1 에피소드 — 정책 trail 과 구간 Φ 를 수집."""
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

    # ExecutionHead 자리 — 계약상 채택 PPO 동결 체크포인트(hash 검사 포함)가 들어와야
    # 하나, 배선 미구현이라 규칙(SF-SPT)을 둔다(잔여 작업 — YR-160 과 함께).
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    mbt.run(exec_policy, review_fn=_Chain(ctrl, orch, rec).review)
    return {"phi": rec.samples, "sell_ledger": orch.ledger,
            "n_space": orch.n_space, "n_time": orch.n_time,
            "policy_exceptions": exc["n"], "admitted": ctrl.n_admitted}


# ------------------------------------------------------------------ 전이 조립
def build_batch(trail: list[dict], phi: list[tuple[float, float]]) -> list[dict]:
    """결정별 (보상 이후 합 R, advantage 원료) 조립 — γ=1 전역 보상.

    epoch 격자 Φ 에서 결정 시각 t 이후의 총 증분 −(Φ(end) − Φ(t)) 을 그 결정의
    reward-to-go 로 쓴다(γ=1 — 전 궤적 credit). advantage = R − V(s).
    """
    if not phi:
        return []
    times = [p[0] for p in phi]
    values = [p[1] for p in phi]
    phi_end = values[-1]
    out = []
    for tr in trail:
        idx = max(i for i, tt in enumerate(times) if tt <= tr["t"] + 1e-9)
        r2go = -(phi_end - values[idx])
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


def train_one(ts: int, *, out_root: Path = OUT) -> Path:
    """초기화 1개 학습 — shadow 아님(on-policy). 실행은 디버깅 국면 후 판정 절차로만."""
    kf = load_kf()
    actor, critic = TransferActor(), TransferCritic()
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    torch.manual_seed(ts)
    hist = []
    for it in range(N_ITER):
        batch_all: list[dict] = []
        for e in range(EPS_PER_ITER):
            policy = PpoSellPolicy(actor, critic, mode="live", sample=True,
                                   seed=ts + it * 100 + e)
            ep = run_episode(ts + it * EPS_PER_ITER + e, policy, kf)
            # critic 입력을 trail 에 보강(수집 시점 상태 — 학습 전용 중앙 관측)
            for tr in policy.trail:
                tr["critic_in"] = torch.zeros(14)      # 골격: 수집기 통합 시 교체(잔여)
            batch_all += build_batch(policy.trail, ep["phi"])
        stats = ppo_update(actor, critic, opt_a, opt_c, batch_all)
        hist.append({"iter": it, **stats})
    out = out_root / f"ppo_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
               out / "net.pt")
    (out / "train.json").write_text(json.dumps(
        {"history": hist, "code_dirty": bool(code_dirty()),
         "stamp": repro_stamp(seeds={"train": [ts]},
                              params={"WIP_TARGET": WIP_TARGET, "N_ITER": N_ITER,
                                      "LEAD_S": ANNOUNCE_LEAD_S,
                                      "anchors": "yr139 승계(clip/lr/ent/lam/γ)"})},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ts", type=int, default=0, help="TRAIN_SEEDS 의 index")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        train_one(TRAIN_SEEDS[a.train_ts])
    print("DONE")
