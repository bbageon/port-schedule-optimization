"""Candidate Q-network + resolver 주입용 QPreference (YR-039 §1~§2).

- CandidateQNet: `[global, yc, queue, candidate] → Q_cost` 후보별 공유망.
  출력 head zero-init → 미학습 전 후보 Q=0 (§1 계약).
- Dueling variant: Q_i = V(ctx) + A_i − mean_selectable(A) (spec 비교축).
- QPreference: rank = (Q값, *BaselinePreference.rank) — 미학습(전부 0) 시
  BaselinePreference 와 완전 동일 순서 (테스트 고정). mandatory-우선·feasibility·
  dry_run 오라클은 resolver 골격이 전담 (안전은 학습 밖, YR-034).

torch 는 optional [rl] — 본 모듈 임포트 시에만 요구.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .encoding import DecisionEncoding
from .resolver import BaselinePreference


@dataclass(frozen=True)
class QNetConfig:
    hidden: int = 128
    dueling: bool = False

    def __post_init__(self) -> None:
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")


class CandidateQNet(nn.Module):
    """공유망 — context(g‖yc‖queue) 임베딩 + 후보별 결합 채점. permutation-invariant
    (후보 간 정보 교환 없음 — 집합 맥락은 queue_summary 가 운반, YR-031-b)."""

    def __init__(self, dims: tuple[int, int, int, int], cfg: QNetConfig = QNetConfig()):
        super().__init__()
        fg, fy, fq, fc = dims
        self.cfg = cfg
        h = cfg.hidden
        self.ctx = nn.Sequential(nn.Linear(fg + fy + fq, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU())
        self.cand = nn.Sequential(nn.Linear(h + fc, h), nn.ReLU())
        self.adv_head = nn.Linear(h, 1)
        nn.init.zeros_(self.adv_head.weight)     # 미학습 Q≡0 (§1)
        nn.init.zeros_(self.adv_head.bias)
        if cfg.dueling:
            self.val_head = nn.Linear(h, 1)
            nn.init.zeros_(self.val_head.weight)
            nn.init.zeros_(self.val_head.bias)

    def forward(self, g: torch.Tensor, yc: torch.Tensor, queue: torch.Tensor,
                cand: torch.Tensor, selectable: torch.Tensor) -> torch.Tensor:
        """g/yc/queue: [B,F*] · cand: [B,K,Fc] · selectable: [B,K] bool → Q [B,K].

        비선택(패딩·infeasible) 위치도 채점은 하되, Dueling 평균과 사용처의
        argmin/backup 은 selectable 만 본다 (호출부 masked_fill 책임 분담).
        """
        ctx = self.ctx(torch.cat([g, yc, queue], dim=-1))            # [B,H]
        k = cand.shape[1]
        ctx_k = ctx.unsqueeze(1).expand(-1, k, -1)                   # [B,K,H]
        z = self.cand(torch.cat([ctx_k, cand], dim=-1))              # [B,K,H]
        adv = self.adv_head(z).squeeze(-1)                           # [B,K]
        if not self.cfg.dueling:
            return adv
        val = self.val_head(z.mean(dim=1))                           # [B,1]→[B]
        sel = selectable.float()
        denom = sel.sum(dim=1).clamp(min=1.0)
        adv_mean = (adv * sel).sum(dim=1) / denom                    # selectable 평균
        return val + adv - adv_mean.unsqueeze(1)


@torch.no_grad()
def score_decision(net: CandidateQNet, enc: DecisionEncoding,
                   device: torch.device | str = "cpu") -> dict[int, float]:
    """결정 1건 채점 → {candidate_id: Q}. selectable 후보만 반환."""
    net.eval()
    dev = torch.device(device)
    t = lambda x: torch.tensor([x], dtype=torch.float32, device=dev)  # noqa: E731
    q = net(t(list(enc.g)), t(list(enc.yc)), t(list(enc.queue)),
            torch.tensor([list(map(list, enc.cand))], dtype=torch.float32,
                         device=dev),
            torch.tensor([list(enc.selectable)], dtype=torch.bool, device=dev))[0]
    return {cid: float(q[i]) for i, cid in enumerate(enc.candidate_ids)
            if enc.selectable[i]}


class QPreference(BaselinePreference):
    """resolver Preference seam 주입체 — 결정마다 scores 갱신 후 resolve 호출.

    scores: {(crane_id, candidate_id): Q_cost}. 미등록/미학습(0.0) 후보는
    BaselinePreference tie-break 로 정렬 — zero-init 망이면 전 후보 0 이라
    기존 baseline resolver 와 결정이 완전히 일치한다 (계약 테스트).
    """

    def __init__(self) -> None:
        self.scores: dict[tuple[str, int], float] = {}

    def set_scores(self, scores: dict[tuple[str, int], float]) -> None:
        self.scores = scores

    def rank(self, sim, crane_id, gc) -> tuple:
        q = self.scores.get((crane_id, gc.candidate_id), 0.0)
        return (q,) + super().rank(sim, crane_id, gc)
