"""Tabular Q-learning (SMDP) — 구현계획 02 §4.3·03 §1.2, 실험설계안 §5.

- 경과시간 보정 할인: gamma_tau = gamma_ref ** (elapsed / ref_s)
- exploration 은 simulator 안에서만 (epsilon-greedy), 평가는 epsilon=0 greedy
- 미방문 상태 fallback: LONGEST_WAIT 우선 결정론 순서 (Phase 4 완료조건)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.enums import PriorityRule
from .baselines import _FALLBACK_ORDER


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float = 0.1
    gamma_ref: float = 0.98      # ref_s 당 할인 (assumed — validation 탐색 대상)
    ref_s: float = 60.0
    eps_start: float = 1.0
    eps_final: float = 0.05
    n_actions: int = 9


def _fallback_action(mask: list[bool]) -> int:
    for r in _FALLBACK_ORDER:
        if mask[r]:
            return int(r)
    for a, ok in enumerate(mask):  # 사전행동(EPA·PRE_REHANDLE)만 열린 상태
        if ok:
            return a
    raise RuntimeError("mask 전부 False")


class QTable:
    """Q 값 + 방문횟수. 보상이 항상 ≤0 이므로 0-초기화 미시도 action 이
    greedy 에서 학습값을 이기는 편향이 생긴다 → greedy/bootstrap 은
    **시도해 본 action 만** 대상으로 한다 (없으면 fallback rule)."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.q: dict[tuple, list[float]] = {}
        self.n: dict[tuple, list[int]] = {}

    def known(self, s) -> bool:
        return s in self.q

    def row(self, s) -> list[float]:
        if s not in self.q:
            self.q[s] = [0.0] * self.n_actions
            self.n[s] = [0] * self.n_actions
        return self.q[s]

    def visit(self, s, a: int):
        self.row(s)
        self.n[s][a] += 1

    def greedy(self, s, mask: list[bool]) -> int:
        """masked & 시도된 action 중 argmax (동률: 낮은 id). 없으면 fallback."""
        if s not in self.q:
            return _fallback_action(mask)
        row, cnt = self.q[s], self.n[s]
        best, best_v = None, None
        for a in range(self.n_actions):
            if mask[a] and cnt[a] > 0 and (best_v is None or row[a] > best_v):
                best, best_v = a, row[a]
        return _fallback_action(mask) if best is None else best

    def max_valid(self, s, mask: list[bool]) -> float:
        """bootstrap 용 — 시도된 action 이 없으면 0 (문서화된 중립값)."""
        if s not in self.q or not any(mask):
            return 0.0
        vals = [v for a, v in enumerate(self.q[s]) if mask[a] and self.n[s][a] > 0]
        return max(vals) if vals else 0.0

    # 저장/로드 (해석가능성·재현용)
    def save(self, path: str | Path):
        data = {"q": {",".join(map(str, k)): v for k, v in self.q.items()},
                "n": {",".join(map(str, k)): v for k, v in self.n.items()}}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, n_actions: int) -> "QTable":
        t = cls(n_actions)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in data["q"].items():
            t.q[tuple(int(x) for x in k.split(","))] = v
        for k, v in data["n"].items():
            t.n[tuple(int(x) for x in k.split(","))] = v
        return t


@dataclass
class QLearningAgent:
    cfg: QLearningConfig
    seed: int = 0
    policy_name: str = "QL"
    table: QTable = None
    rng: random.Random = None
    train_log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.table = self.table or QTable(self.cfg.n_actions)
        self.rng = random.Random(self.seed)

    # --- 행동 선택 ---
    def act_train(self, state, mask: list[bool], eps: float) -> int:
        if self.rng.random() < eps:
            valid = [a for a in range(self.cfg.n_actions) if mask[a]]
            return self.rng.choice(valid)
        return self.table.greedy(state, mask)

    def act(self, state, mask: list[bool]) -> int:  # 평가용 greedy (eps=0)
        return self.table.greedy(state, mask)

    @property
    def name(self) -> str:
        return self.policy_name

    # --- SMDP 업데이트 ---
    def update(self, s, a: int, r: float, s2, mask2: list[bool],
               elapsed_s: float, done: bool):
        gamma = 0.0 if done else self.cfg.gamma_ref ** (elapsed_s / self.cfg.ref_s)
        target = r + gamma * self.table.max_valid(s2, mask2)
        row = self.table.row(s)
        row[a] += self.cfg.alpha * (target - row[a])
        self.table.visit(s, a)


def train(agent: QLearningAgent, env, scenarios: list, epochs: int = 1) -> None:
    """episode = 운영 shift 1개. epsilon 은 전체 episode 에 걸쳐 선형 감소."""
    total = max(1, len(scenarios) * epochs)
    ep_idx = 0
    for _ in range(epochs):
        for sc in scenarios:
            frac = ep_idx / max(1, total - 1) if total > 1 else 1.0
            eps = agent.cfg.eps_start + (agent.cfg.eps_final - agent.cfg.eps_start) * frac
            state, info = env.reset(sc)
            ep_return, steps = 0.0, 0
            while state is not None:
                a = agent.act_train(state, info.action_mask, eps)
                s2, r, done, info2 = env.step(a)
                agent.update(state, a, r, s2, info2.action_mask, info2.elapsed_s, done)
                state, info = s2, info2
                ep_return += r
                steps += 1
            agent.train_log.append({"episode": ep_idx, "scenario": sc.scenario_id,
                                    "eps": round(eps, 3), "return": round(ep_return, 3),
                                    "steps": steps,
                                    "states_known": len(agent.table.q)})
            ep_idx += 1
