"""YR-151 — TransferHead: 블록 판매 PPO (계획 진단 head, 설계 확정 2026-08-09).

■ 의미 규정 (사용자 확정)
PPO 의 판단 = **"이 작업을 포함한 현 계획이 나쁘다"**. 행동 K+1 은 계획 변형 K+1개다:
  KEEP     = 현 계획(지금 안고 있는 작업 구성) 유지가 최선
  OFFER(j) = 현 계획에서 j 를 덜어낸 계획이 (실비용을 치르고도) 더 낫다
작업이 나쁜 게 아니라 맥락(계획) 속에서 나쁘다 — 볼록 비용의 직접 귀결.

■ 구조 확정 (B-(b))
  · PPO 는 **축(공간/시간)을 모른다** — "무엇을 덜어낼지"만. 어느 좌표(어느 블록/어느
    시각)로 옮길지는 resolver 가 순이득 저울로 정한다. 따라서 **가중치는 1벌**
    (축별 분리 불요 — 축 귀속은 resolver 원장이 사후 분해).
  · 공유 가중치 1벌을 21블록이 각자 자기 공개 정보로 실행(중앙학습·분산실행).
    경험 21배·일반 규칙 학습·동일 정책 배포가 근거.
  · 기존 크레인 PPO(JointPairNet — 동적 후보 행 점수화)와 같은 검증된 패턴.

■ 보상 (확정: (i) 실현 전역 증분)
r = −(결정 구간의 터미널 전체 실현 v2 증분비용 — 수신 부담·주행·기사 외부 대기 포함).
전역이므로 떠넘긴 비용도 계산서에 자동 포함(= 이기적 판매 + 가격 청구와 동치).
Σ 구간보상 = −(에피소드 총비용) 검열 등식 승계 — 학습 잣대 ≡ 평가 잣대.

■ KEEP 의 근거
명시 공식이 아니라 학습된 가치(그 뒤 벌어진 일 전부 반영) + 비가역성 비대칭(SELL 은
1회 잠금·KEEP 은 60초 뒤 재고). 신호 존재는 0B 가 실측(NO_LEARNABLE_SIGNAL 관문).

■ 계약 가드: ExecutionHead 동결·hash 불변(학습 루프에서 검사) / 별도 파라미터·optimizer
(드문 판매 신호 보존) / 입력은 공개 정보만(실현 미래값·타 블록 내부 금지).
■ 테스트 유예 (빌드 우선 지시) — 특징 스케일은 assumed 상수(사전등록 동결 대상).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


from .sell_review import block_inside, block_pipeline


ROW_DIM = 14                  # 블록 7 + KEEP 플래그 1 + 후보 6
BLOCK_DIM = 7
HID = 64


# ------------------------------------------------------------------ 특징 (전부 공개 정보)
def block_features(mbt, src: str, t: float, n_cands: int) -> list[float]:
    """블록 계획 요약 7차원 — "작업을 품은 계획"을 평가하기 위한 맥락."""
    sim = mbt.blocks[src]
    inside = block_inside(sim, t)
    pipeline = block_pipeline(mbt, src, t)
    crane_backlog = sum(max(0.0, sim.fleet.get(c.crane_id).state.available_at - t)
                        for c in sim.profile.cranes)
    g = sim.profile.block
    occupancy = len(sim.stacks.containers) / max(1, g.bay_count * g.row_count * g.tier_max)
    slack_vals = [v.slack_s(t) for v in sim.vessels.values()
                  if v.plan.planned_completion_s is not None and not v.done]
    vessel_slack = min([s for s in slack_vals if s is not None] + [2.0 * 3600.0])
    announced_30 = sum(1 for rec in mbt.ledger.records.values()
                       if rec.owner == src and rec.a_gate_in is not None
                       and t < rec.a_gate_in <= t + 1800.0)
    return [inside / 10.0, pipeline / 10.0, crane_backlog / 3600.0, occupancy,
            max(-2.0, min(2.0, vessel_slack / 3600.0)), announced_30 / 10.0,
            n_cands / 6.0]


def candidate_features(mbt, src: str, jid: str, t: float) -> list[float]:
    """후보 6차원 — 창 내 잔여시간(멈춤 문제의 시계)·이력 포함."""
    rec = mbt.ledger.records[jid]
    j = mbt.blocks[src].jobs[jid]
    eta = getattr(j, "estimated_block_arrival", None) or j.provided_eta or t
    gate_remain = (rec.a_gate_in - t) if rec.a_gate_in is not None else 0.0
    is_out = 1.0 if rec.flow == "GATE_OUT" else 0.0
    size40 = 1.0 if str(getattr(j, "inbound_size", "")).endswith("40") else 0.0
    return [max(0.0, min(1.0, (eta - t) / 1800.0)), is_out, size40,
            max(0.0, min(1.0, gate_remain / 1800.0)),
            float(rec.transfer_count), float(rec.entry_deferrals)]


def build_rows(mbt, src: str, cands: list, t: float) -> torch.Tensor:
    """[KEEP 행, 후보 행 ×K] — KEEP 은 학습 가능한 기준 행(플래그 1·후보부 0)."""
    bf = block_features(mbt, src, t, len(cands))
    rows = [bf + [1.0] + [0.0] * 6]
    for jid, *_ in cands:
        rows.append(bf + [0.0] + candidate_features(mbt, src, jid, t))
    return torch.tensor(rows, dtype=torch.float32)


def critic_input(mbt, src: str, t: float, n_cands: int) -> torch.Tensor:
    """중앙학습 전용 — source 요약 + **전 블록 순열불변 요약**(평균 pooling)."""
    bf = block_features(mbt, src, t, n_cands)
    allf = [block_features(mbt, b, t, 0) for b in mbt.blocks]
    pooled = [sum(col) / len(allf) for col in zip(*allf)]
    return torch.tensor(bf + pooled, dtype=torch.float32)


# ------------------------------------------------------------------ 신경망 (공유 1벌)
class TransferActor(nn.Module):
    def __init__(self, in_dim: int = ROW_DIM, hid: int = HID):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows).squeeze(-1)          # 행별 점수 → 밖에서 softmax


class TransferCritic(nn.Module):
    def __init__(self, in_dim: int = BLOCK_DIM * 2, hid: int = HID):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.v(s).squeeze(-1)


# ------------------------------------------------------------------ SellPolicy 어댑터
class PpoSellPolicy:
    """UnifiedSellOrchestrator 에 꽂히는 학습 정책 — decide() 1건/epoch/블록.

    mode:
      "live"   — 선택을 실제로 내놓는다(on-policy 학습·평가)
      "shadow" — 계산만 하고 **항상 KEEP 반환**(원장 기록 전용 — 정책경사 학습 금지 단계)
    trail 에 (행, 선택, log-prob, V) 전이를 쌓아 학습 루프가 소비한다.
    """

    def __init__(self, actor: TransferActor, critic: TransferCritic | None = None,
                 *, mode: str = "live", sample: bool = True, seed: int = 0):
        assert mode in ("live", "shadow")
        self.actor = actor
        self.critic = critic
        self.mode = mode
        self.sample = sample
        self.gen = torch.Generator().manual_seed(seed)
        self.trail: list[dict] = []

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        if not cands:
            return None
        rows = build_rows(mbt, src, cands, t)
        with torch.no_grad():
            logits = self.actor(rows)
            dist = Categorical(logits=logits)
            if self.sample:
                a = int(torch.multinomial(dist.probs, 1, generator=self.gen).item())
            else:
                a = int(torch.argmax(logits).item())
            logp = float(dist.log_prob(torch.tensor(a)).item())
            v = (float(self.critic(critic_input(mbt, src, t, len(cands))).item())
                 if self.critic is not None else 0.0)
        pick = None if a == 0 else cands[a - 1][0]
        self.trail.append({"t": t, "src": src, "rows": rows, "action": a,
                           "logp": logp, "value": v, "n_cands": len(cands),
                           "picked": pick})
        return None if self.mode == "shadow" else pick
