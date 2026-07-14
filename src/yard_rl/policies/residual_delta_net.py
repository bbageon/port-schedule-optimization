"""YR-012 — 잔차 연속-feature Δ 학습 (함수근사).

Q_total(s,j) = G(s,j) + Δθ(x(s,j)) — YR-030-c 잔차 골격 승계, Δ 저장소만
bucket 표 → MLP(연속 입력). 출력층 zero-init 으로 미학습 정책 ≡ greedy 를
정확히 보장한다 (사전등록 §1 계약).

torch 는 optional dependency ([rl]) — 본 모듈 임포트 시에만 요구.
사전등록: .claude/docs/strategy-history/2026-07-14-YR-012-residual-delta-net-prereg.md
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .cost_q import CandidateProtocol, CandidateT, EvaluationDiagnostics

N_FEATURES = 14
# z-score 적용 차원 (나머지는 [0,1] 스케일 — passthrough, 사전등록 §2)
_ZSCORE_DIMS = (2, 3, 4, 6, 7, 8, 9, 10, 11, 13)


def extract_features(candidate: CandidateProtocol) -> list[float]:
    """후보 → 14차원 연속 feature (env 가 부착한 global_raw/future_raw 사용)."""
    g = getattr(candidate, "global_raw", ())
    f = getattr(candidate, "future_raw", ())
    if len(g) != 5 or len(f) != 4:
        raise ValueError("연속 feature 미부착 후보 — v1_final env 필요 (YR-012)")
    return [
        g[0],                                   # 0 진행률 [0,1]
        g[1],                                   # 1 크레인 bay 정규화 [0,1]
        g[2],                                   # 2 대기 수
        g[3],                                   # 3 최장대기 s
        g[4],                                   # 4 30분초과 수
        1.0 if candidate.transfer_direction == "YARD_TO_TRUCK" else 0.0,  # 5
        float(candidate.wait_s),                # 6 자기 대기 s
        float(candidate.reach_s),               # 7 크레인 이동 s
        float(candidate.estimated_service_s),   # 8 예상 서비스 s
        float(candidate.blocker_count),         # 9 선행이동 수
        f[0],                                   # 10 남은 작업 수
        f[1],                                   # 11 남은 총 서비스 s
        f[2],                                   # 12 잔여 짧은작업 비율 [0,1]
        f[3],                                   # 13 최근접 잔여 bay 거리
    ]


@dataclass(frozen=True)
class FeatureScaler:
    """train FIFO 관측 mean/std z-score — fit 후 동결 (val/test 재조정 금지)."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    fitted: bool = False

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> "FeatureScaler":
        if not rows:
            raise ValueError("scaler fit 에 관측이 필요")
        cols = list(zip(*rows))
        if len(cols) != N_FEATURES:
            raise ValueError(f"feature 차원 불일치: {len(cols)} != {N_FEATURES}")
        mean, std = [], []
        for d, col in enumerate(cols):
            if d in _ZSCORE_DIMS:
                m = sum(col) / len(col)
                var = sum((v - m) ** 2 for v in col) / len(col)
                mean.append(m)
                std.append(max(math.sqrt(var), 1e-6))
            else:
                mean.append(0.0)
                std.append(1.0)
        return cls(tuple(mean), tuple(std), fitted=True)

    def transform(self, x: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mean, self.std)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["mean"]), tuple(d["std"]), bool(d["fitted"]))


class _DeltaMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)
        # 출력층 zero-init — 미학습 Δθ(x) ≡ 0 → 정책 ≡ greedy (사전등록 §1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


@dataclass(frozen=True)
class DeltaNetConfig:
    gamma: float = 0.95
    lr: float = 1e-3
    grad_clip: float = 1.0
    hidden: int = 64

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.lr <= 0 or self.grad_clip <= 0 or self.hidden <= 0:
            raise ValueError("lr/grad_clip/hidden must be positive")


@dataclass
class ResidualDeltaNetAgent:
    """CostQAgent 와 동일한 select/update 프로토콜 (러너 duck-typing 재사용)."""

    cfg: DeltaNetConfig = DeltaNetConfig()
    scaler: FeatureScaler | None = None
    seed: int = 0
    diagnostics: EvaluationDiagnostics = field(default_factory=EvaluationDiagnostics)

    def __post_init__(self) -> None:
        if self.scaler is None or not self.scaler.fitted:
            raise ValueError("fitted FeatureScaler 필요 (train 관측으로 동결)")
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)  # 결정론 (사전등록 §3)
        self.rng = random.Random(self.seed)
        self.net = _DeltaMLP(self.cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    @property
    def name(self) -> str:
        return "ResidualDeltaNet"

    # ------------------------------------------------------------- Q_total
    def _x(self, candidates: Sequence[CandidateProtocol]) -> torch.Tensor:
        rows = [self.scaler.transform(extract_features(c)) for c in candidates]
        return torch.tensor(rows, dtype=torch.float32)

    @torch.no_grad()
    def q_totals(self, candidates: Sequence[CandidateProtocol]) -> list[float]:
        deltas = self.net(self._x(candidates)).tolist()
        return [float(c.prior_cost) + d for c, d in zip(candidates, deltas)]

    @staticmethod
    def _tie_break(c: CandidateProtocol) -> tuple[float, float, float, str]:
        return (-float(c.wait_s), float(c.estimated_service_s),
                float(c.block_entry_s), str(c.job_id))

    def _argmin_total(self, candidates: list[CandidateT]) -> CandidateT:
        totals = self.q_totals(candidates)
        return min(zip(totals, candidates),
                   key=lambda tc: (tc[0], *self._tie_break(tc[1])))[1]

    # ------------------------------------------------------------- 행동 선택
    def act_train(self, global_state: object, candidates: Iterable[CandidateT],
                  epsilon: float) -> CandidateT:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        feasible = list(candidates)
        if not feasible:
            raise ValueError("action selection requires a feasible candidate")
        if self.rng.random() < epsilon:
            return self.rng.choice(feasible)
        return self._argmin_total(feasible)

    def act(self, global_state: object, candidates: Iterable[CandidateT]
            ) -> CandidateT:
        feasible = list(candidates)
        if not feasible:
            raise ValueError("action selection requires a feasible candidate")
        self.diagnostics.decisions += 1          # 연속 근사 — coverage 개념 없음
        self.diagnostics.fully_covered_decisions += 1
        self.diagnostics.signatures_checked += len(feasible)
        self.diagnostics.visited_signatures += len(feasible)
        return self._argmin_total(feasible)

    select_train = act_train
    select_eval = act

    # ---------------------------------------------------------------- 학습
    def update(self, global_state: object, candidate: CandidateProtocol,
               cost: float, next_global_state: object | None,
               next_candidates: Iterable[CandidateProtocol], done: bool) -> float:
        """online TD 회귀: loss = (Δθ(x) − (Y − G))² — 사전등록 §1·§3."""
        step_cost = float(cost)
        if not math.isfinite(step_cost) or step_cost < 0.0:
            raise ValueError("step cost must be finite and non-negative")
        if done:
            target_total = step_cost
        else:
            nxt = list(next_candidates)
            if not nxt:
                raise ValueError("non-terminal backup requires a feasible next key")
            target_total = step_cost + self.cfg.gamma * min(self.q_totals(nxt))
        target_delta = target_total - float(candidate.prior_cost)
        if not math.isfinite(target_delta):
            raise ValueError("residual target must be finite")
        pred = self.net(self._x([candidate]))[0]
        loss = (pred - torch.tensor(float(target_delta))) ** 2
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return float(pred.detach())

    def reset_diagnostics(self) -> None:
        self.diagnostics = EvaluationDiagnostics()

    # ------------------------------------------------------------- 저장/로드
    def save(self, path: str | Path) -> None:
        torch.save({
            "format": "yard-rl-residual-delta-net-v1",
            "config": asdict(self.cfg),
            "seed": self.seed,
            "scaler": asdict(self.scaler),
            "state_dict": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "diagnostics": asdict(self.diagnostics),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualDeltaNetAgent":
        payload = torch.load(str(path), weights_only=False)
        if payload.get("format") != "yard-rl-residual-delta-net-v1":
            raise ValueError("unsupported residual delta-net format")
        sc = payload["scaler"]
        agent = cls(cfg=DeltaNetConfig(**payload["config"]),
                    scaler=FeatureScaler(tuple(sc["mean"]), tuple(sc["std"]),
                                         bool(sc["fitted"])),
                    seed=int(payload["seed"]),
                    diagnostics=EvaluationDiagnostics(**payload["diagnostics"]))
        agent.net.load_state_dict(payload["state_dict"])
        agent.opt.load_state_dict(payload["optimizer"])
        return agent
