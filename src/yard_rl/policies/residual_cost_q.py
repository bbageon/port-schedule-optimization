"""YR-030-c Greedy 기반 잔차 Cost-Q (사용자 최종 전략, 2026-07-14).

Q_total(s,j) = G(s,j) + ΔQ(z(s,j)) · j* = argmin Q_total

- G(s,j): 후보의 정확한 greedy 즉시비용 (candidate.prior_cost — 초 단위 정밀,
  bucket 아님). **절대 학습값으로 대체되지 않는다** (YR-030-b prior 와의 차이).
- ΔQ: 테이블에는 "greedy 비용으로 설명되지 않는 보정분"만 저장 (음수 허용).
  미방문 z 는 ΔQ=0 → 학습 전/미경험 상황에서 정책 ≡ greedy.
- 학습식: Y = c + γ·min_j'[G(s',j') + ΔQ(z')] (종료 Y=c), Y_Δ = Y − G(s,j),
  ΔQ(z) ← ΔQ(z) + α[Y_Δ − ΔQ(z)], α = n^-p.
- z (key_mode):
  * "state_job": 기존 coarse (GlobalState, JobState) — Residual-only arm
  * "future":   future_situation 5-tuple 단독 — Residual+future arm (§3 원문)

사전등록: .claude/docs/strategy-history/2026-07-14-YR-030-c-residual-costq-prereg.md
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .cost_q import (CandidateProtocol, CandidateT, CostQAgent, CostQConfig,
                     CostQKey, CostQTable, EvaluationDiagnostics, _canonical,
                     _decode, _encode, make_cost_q_key)

KEY_MODES = ("state_job", "future")
_FORMAT = "yard-rl-residual-cost-q-agent-v1"


@dataclass
class ResidualCostQAgent(CostQAgent):
    """CostQAgent 프로토콜(select/update/save) 호환 — 러너·평가기 재사용."""

    key_mode: str = "state_job"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.key_mode not in KEY_MODES:
            raise ValueError(f"key_mode must be one of {KEY_MODES}: {self.key_mode}")
        if self.cfg.use_greedy_prior:
            raise ValueError("잔차 구조는 prior 대체 모드와 배타적 — "
                             "use_greedy_prior=False 여야 함 (G 는 Q_total 에 상존)")

    @property
    def name(self) -> str:
        return f"ResidualCostQ[{self.key_mode}]"

    # ------------------------------------------------------------------ key
    def key(self, global_state: object, candidate: CandidateProtocol) -> CostQKey:
        if self.key_mode == "future":
            future = getattr(candidate, "future_feature", ())
            if not future:
                raise ValueError(
                    "future key_mode 는 future_feature 부착 후보가 필요 (v1_final env)")
            return (("future",), _canonical(tuple(future)))
        return make_cost_q_key(global_state, candidate)

    # ------------------------------------------------------------- Q_total
    def q_total(self, global_state: object, candidate: CandidateProtocol) -> float:
        """G(정확 greedy 비용) + ΔQ. 미방문 ΔQ=0 → 정확히 greedy."""
        return float(candidate.prior_cost) + self.table.value(
            self.key(global_state, candidate))

    def _argmin_total(self, global_state: object,
                      candidates: list[CandidateT]) -> CandidateT:
        return min(candidates,
                   key=lambda c: (self.q_total(global_state, c),
                                  *self._tie_break(c)))

    # ------------------------------------------------------------- 행동 선택
    def act_train(self, global_state: object, candidates: Iterable[CandidateT],
                  epsilon: float) -> CandidateT:
        """ε-random / argmin Q_total (사전등록 §7 — 미방문 우선탐색 없음)."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        feasible = self._candidates(candidates)
        if self.rng.random() < epsilon:
            return self.rng.choice(feasible)
        return self._argmin_total(global_state, feasible)

    def act(self, global_state: object, candidates: Iterable[CandidateT]
            ) -> CandidateT:
        """평가 ε=0 — fallback 없음 (G 가 항상 유효한 하한 안내). coverage 진단만."""
        feasible = self._candidates(candidates)
        unique_keys = tuple(dict.fromkeys(self.key(global_state, c) for c in feasible))
        visited = sum(self.table.is_visited(k) for k in unique_keys)
        self.diagnostics.decisions += 1
        self.diagnostics.signatures_checked += len(unique_keys)
        self.diagnostics.visited_signatures += visited
        if visited == len(unique_keys):
            self.diagnostics.fully_covered_decisions += 1
        return self._argmin_total(global_state, feasible)

    select_train = act_train
    select_eval = act

    # ---------------------------------------------------------------- 학습
    def update(self, global_state: object, candidate: CandidateProtocol,
               cost: float, next_global_state: object | None,
               next_candidates: Iterable[CandidateProtocol], done: bool) -> float:
        """ΔQ(z) ← ΔQ(z) + α[(Y − G(s,j)) − ΔQ(z)] — Y_Δ 는 음수 허용."""
        step_cost = float(cost)
        if not math.isfinite(step_cost) or step_cost < 0.0:
            raise ValueError("step cost must be finite and non-negative")
        if done:
            target_total = step_cost
        else:
            if next_global_state is None:
                raise ValueError("non-terminal update requires next_global_state")
            nxt = list(next_candidates)
            if not nxt:
                raise ValueError("non-terminal backup requires a feasible next key")
            target_total = step_cost + self.cfg.gamma * min(
                self.q_total(next_global_state, c) for c in nxt)
        target_delta = target_total - float(candidate.prior_cost)
        if not math.isfinite(target_delta):
            raise ValueError("residual target must be finite")
        return self.table.update(self.key(global_state, candidate), target_delta,
                                 self.cfg.learning_rate_power)

    # ------------------------------------------------------------- 저장/로드
    def save(self, path: str | Path) -> None:
        payload = {
            "format": _FORMAT,
            "key_mode": self.key_mode,
            "config": asdict(self.cfg),
            "seed": self.seed,
            "table": self.table.to_payload(),
            "diagnostics": asdict(self.diagnostics),
            "rng_state": _encode(_canonical(self.rng.getstate())),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ResidualCostQAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != _FORMAT:
            raise ValueError("unsupported residual Cost-Q agent format")
        agent = cls(cfg=CostQConfig(**payload["config"]), seed=int(payload["seed"]),
                    table=CostQTable.from_payload(payload["table"]),
                    diagnostics=EvaluationDiagnostics(**payload["diagnostics"]),
                    key_mode=str(payload["key_mode"]))
        rng_state = _decode(payload["rng_state"])
        if not isinstance(rng_state, tuple):
            raise ValueError("invalid residual Cost-Q RNG state")
        agent.rng.setstate(rng_state)
        return agent
