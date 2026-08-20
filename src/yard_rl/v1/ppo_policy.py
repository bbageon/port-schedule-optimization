"""v1 — PPO 확률 정책 (구 `integrated/transfer_head.py` 의 세대 전용분).

■ 세대 위치
  v1 = "이 트럭을 내놓을 확률"을 배우는 정책. 규칙 대비 **+97.94** 로 졌고
  critic 붕괴·보상 0 이 91% 라는 진단을 남겼다. → v2 는 확률 대신 **비용**을 배운다.

■ 의미 규정 (사용자 확정 2026-08-09)
  PPO 의 판단 = **"이 작업을 포함한 현 계획이 나쁘다"**. 행동 K+1 은 계획 변형 K+1개다.
    KEEP     = 현 계획 유지가 최선
    OFFER(j) = j 를 덜어낸 계획이 (실비용을 치르고도) 더 낫다
  작업이 나쁜 게 아니라 맥락(계획) 속에서 나쁘다 — 볼록 비용의 직접 귀결.

  · PPO 는 **작업 종류는 특징으로 알지만 좌표(목적지·시각)를 직접 고르지 않는다.**
    좌표는 resolver 가 순이득 저울로 정한다 → 가중치는 1벌.
  · 공유 가중치 1벌을 21블록이 각자 공개 정보로 실행(중앙학습·분산실행).

■ 보상
  r = −(결정 구간의 터미널 전체 실현 v2 증분비용). 전역이라 떠넘긴 비용도 자동 포함.
  Σ 구간보상 = −(에피소드 총비용) 검열 등식 승계 — 학습 잣대 ≡ 평가 잣대.

■ 계약 가드
  ExecutionHead 동결·hash 불변 / 별도 파라미터·optimizer / 입력은 공개 정보만.

■ 특징은 공용이다
  `block_features` · `candidate_features` 는 `v1/features.py` 에 있다.
  v2 도 같은 식을 쓰지만 **사본을 따로 갖는다** — 세대를 얼리기 위한 의도된 중복이다.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .features import BLOCK_DIM, block_features, candidate_features

__all__ = ["ROW_DIM", "CRITIC_DIM", "HID", "BLOCK_DIM", "build_rows", "critic_input",
           "TransferActor", "TransferCritic", "PpoSellPolicy"]

ROW_DIM = 14                  # 블록 7 + KEEP 플래그 1 + 후보 6
CRITIC_DIM = 15               # 소스 7 + 수신 시장 요약 6 + 전역 2 (★감사 강화판)
HID = 64


def build_rows(mbt, src: str, cands: list, t: float) -> torch.Tensor:
    """[KEEP 행, 후보 행 ×K] — KEEP 은 학습 가능한 기준 행(플래그 1·후보부 0)."""
    bf = block_features(mbt, src, t, len(cands))
    rows = [bf + [1.0] + [0.0] * 6]
    for jid, *_ in cands:
        rows.append(bf + [0.0] + candidate_features(mbt, src, jid, t))
    return torch.tensor(rows, dtype=torch.float32)


def critic_input(mbt, src: str, t: float, n_cands: int,
                 layout=None) -> torch.Tensor:
    """중앙학습 전용 — 소스 요약 + **수신 시장 요약** + 전역 (★감사 강화 2026-08-09).

    구판(전 블록 단순 평균)은 "빈 블록이 **가까이** 있는가"와 "수신처 **여유**가 실제로
    있는가"를 구분하지 못해 기준선(V)이 흐렸다. 수신별 (부하, 소스로부터의 주행 차이,
    용량 여유)를 평균·최소로 pooling(순열불변)해 resolver 가 만들 시장 상황을 critic 이
    설명할 수 있게 한다. layout 없이(2블록 테스트 등) 쓰면 주행 특징은 0.
    """
    from .sell_review import block_inside, block_pipeline
    bf = block_features(mbt, src, t, n_cands)
    qs, routes, heads = [], [], []
    for dst in mbt.blocks:
        if dst == src:
            continue
        qs.append(float(block_inside(mbt.blocks[dst], t)
                        + block_pipeline(mbt, dst, t)))
        heads.append(float(mbt.free_slots(dst)))
        r = 0.0
        if layout is not None:
            try:
                r = layout.pre_gate_route_delta_s(src, dst)
            except KeyError:
                r = 0.0
        routes.append(r)
    n = max(1, len(qs))
    recv = [sum(qs) / n / 10.0, (min(qs) if qs else 0.0) / 10.0,
            sum(routes) / n / 600.0, (min(routes) if routes else 0.0) / 600.0,
            sum(heads) / n / 1000.0, (min(heads) if heads else 0.0) / 1000.0]
    end = max(s.end for s in mbt.blocks.values())
    total_inside = sum(block_inside(s, t) for s in mbt.blocks.values())
    glob = [total_inside / 100.0, min(1.0, t / max(1.0, end))]
    return torch.tensor(bf + recv + glob, dtype=torch.float32)

# ------------------------------------------------------------------ 신경망 (공유 1벌)
class TransferActor(nn.Module):
    def __init__(self, in_dim: int = ROW_DIM, hid: int = HID):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows).squeeze(-1)          # 행별 점수 → 밖에서 softmax


class TransferCritic(nn.Module):
    def __init__(self, in_dim: int = CRITIC_DIM, hid: int = HID):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.v(s).squeeze(-1)


# ------------------------------------------------------------------ SellPolicy 어댑터
class PpoSellPolicy:
    """UnifiedSellOrchestrator 에 꽂히는 학습 정책 — decide() 1건/epoch/블록.

    mode:
      "live"   — 선택이 실제로 확정된다(on-policy 학습·평가)
      "shadow" — ★감사 재정의(2026-08-09): 선택을 **반환하되** dry_run resolver 와만
                 결합된다(orchestrator 생성자가 짝을 강제) — 제안이 견적·matching·
                 용량 검사까지 실제로 흐르고 **원자 확정만 생략**(would-commit 원장).
                 구판 "항상 KEEP 반환"은 resolver 검증 불가라 폐기.
    trail 에 (행, 선택, log-prob, V) 전이를 쌓아 학습 루프가 소비한다.
    """

    def __init__(self, actor: TransferActor, critic: TransferCritic | None = None,
                 *, mode: str = "live", sample: bool = True, seed: int = 0,
                 layout=None):
        assert mode in ("live", "shadow")
        self.actor = actor
        self.critic = critic
        self.mode = mode
        self.sample = sample
        self.layout = layout                 # critic 수신 시장 요약의 주행 특징용
        self.gen = torch.Generator().manual_seed(seed)
        self.trail: list[dict] = []

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        if not cands:
            return None
        rows = build_rows(mbt, src, cands, t)
        # ★감사 치명 1 정정: critic 입력을 **결정 시점에 저장**한다 — 구판은 학습 직전
        # zeros 로 덮어써 critic 이 항상 0 상태만 학습했다(가치학습·advantage 무효).
        ci = critic_input(mbt, src, t, len(cands), layout=self.layout)
        with torch.no_grad():
            logits = self.actor(rows)
            dist = Categorical(logits=logits)
            if self.sample:
                a = int(torch.multinomial(dist.probs, 1, generator=self.gen).item())
            else:
                a = int(torch.argmax(logits).item())
            logp = float(dist.log_prob(torch.tensor(a)).item())
            v = float(self.critic(ci).item()) if self.critic is not None else 0.0
        pick = None if a == 0 else cands[a - 1][0]
        self.trail.append({"t": t, "src": src, "rows": rows, "critic_in": ci,
                           "action": a, "logp": logp, "value": v,
                           "n_cands": len(cands), "picked": pick})
        # shadow 도 pick 을 반환한다 — 확정 생략은 dry_run resolver 의 몫(짝은
        # orchestrator 생성자가 강제). 구판 즉시-None 은 resolver 검증 불가(감사).
        return pick
