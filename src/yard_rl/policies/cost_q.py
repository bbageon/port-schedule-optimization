"""Tabular direct-job Cost-Q policy for YR-027.

The policy intentionally depends on a tiny candidate protocol instead of the
direct-job environment.  A candidate supplies its shared feature signature and
the four raw fields needed for deterministic tie-breaking.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Protocol, TypeVar


class CandidateProtocol(Protocol):
    """Structural interface expected from a feasible direct-job candidate."""

    job_id: str
    feature: Hashable
    wait_s: float
    estimated_service_s: float
    block_entry_s: float


CandidateT = TypeVar("CandidateT", bound=CandidateProtocol)
CostQKey = tuple[Hashable, Hashable]


def _canonical(value: object) -> Hashable:
    """Convert common immutable state/feature objects to stable JSON keys."""
    if isinstance(value, Enum):
        cls = type(value)
        return ("enum", cls.__module__, cls.__qualname__, _canonical(value.value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cost-Q key cannot contain NaN or infinity")
        return value
    if isinstance(value, tuple):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, list):
        return ("list", tuple(_canonical(item) for item in value))
    if isinstance(value, Mapping):
        items = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        return ("mapping", tuple(sorted(items, key=repr)))
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        members = tuple((f.name, _canonical(getattr(value, f.name))) for f in fields(value))
        return ("dataclass", cls.__module__, cls.__qualname__, members)
    key_factory = getattr(value, "cost_q_key", None)
    if key_factory is not None:
        return _canonical(key_factory() if callable(key_factory) else key_factory)
    raise TypeError(
        "GlobalState and CandidateFeature must be tuples, dataclasses, mappings, "
        "or expose cost_q_key"
    )


def make_cost_q_key(global_state: object, candidate: CandidateProtocol) -> CostQKey:
    """Build the scalar-table key ``(GlobalState, CandidateFeature)``."""
    return (_canonical(global_state), _canonical(candidate.feature))


def _encode(value: Hashable) -> object:
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item) for item in value]}
    return {"type": "atom", "value": value}


def _decode(payload: object) -> Hashable:
    if not isinstance(payload, dict):
        raise ValueError("invalid Cost-Q key payload")
    if payload.get("type") == "tuple":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid Cost-Q tuple payload")
        return tuple(_decode(item) for item in items)
    if payload.get("type") == "atom":
        value = payload.get("value")
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
    raise ValueError("invalid Cost-Q atom payload")


class CostQTable:
    """Scalar Q values and visit counts with an implicit ``Q0 = 0``."""

    def __init__(self) -> None:
        self.q: dict[CostQKey, float] = {}
        self.n: dict[CostQKey, int] = {}

    def value(self, key: CostQKey) -> float:
        return self.q.get(key, 0.0)

    def visits(self, key: CostQKey) -> int:
        return self.n.get(key, 0)

    def is_visited(self, key: CostQKey) -> bool:
        return self.visits(key) > 0

    def min_value(self, keys: Iterable[CostQKey]) -> float:
        materialized = tuple(dict.fromkeys(keys))
        if not materialized:
            raise ValueError("non-terminal Cost-Q backup requires a feasible next key")
        # Deliberately include unseen keys at Q0=0: this is the standard min backup.
        return min(self.value(key) for key in materialized)

    def update(self, key: CostQKey, target: float, learning_rate_power: float) -> float:
        count = self.visits(key) + 1
        alpha = count ** (-learning_rate_power)
        old = self.value(key)
        self.q[key] = old + alpha * (target - old)
        self.n[key] = count
        return self.q[key]

    def to_payload(self) -> dict[str, object]:
        entries = []
        for key in set(self.q) | set(self.n):
            entries.append({
                "key": _encode(key),
                "q": self.value(key),
                "visits": self.visits(key),
            })
        entries.sort(key=lambda row: json.dumps(row["key"], sort_keys=True))
        return {"entries": entries}

    @classmethod
    def from_payload(cls, payload: object) -> "CostQTable":
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            raise ValueError("invalid Cost-Q table payload")
        table = cls()
        for row in payload["entries"]:
            if not isinstance(row, dict):
                raise ValueError("invalid Cost-Q table row")
            key = _decode(row.get("key"))
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("Cost-Q table key must contain state and candidate feature")
            q_value = float(row.get("q"))
            visits = int(row.get("visits"))
            if not math.isfinite(q_value) or visits < 0:
                raise ValueError("invalid Cost-Q value or visit count")
            table.q[key] = q_value
            table.n[key] = visits
        return table

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CostQTable":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CostQConfig:
    learning_rate_power: float = 0.6
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate_power <= 1.0:
            raise ValueError("learning_rate_power must be in (0, 1]")
        if self.gamma != 1.0:
            raise ValueError("YR-027 Cost-Q requires gamma exactly 1.0")


@dataclass
class EvaluationDiagnostics:
    decisions: int = 0
    fallback_count: int = 0
    fully_covered_decisions: int = 0
    signatures_checked: int = 0
    visited_signatures: int = 0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.decisions if self.decisions else 0.0

    @property
    def decision_coverage(self) -> float:
        return self.fully_covered_decisions / self.decisions if self.decisions else 0.0

    @property
    def signature_coverage(self) -> float:
        if not self.signatures_checked:
            return 0.0
        return self.visited_signatures / self.signatures_checked

    def as_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "fallback_rate": self.fallback_rate,
            "decision_coverage": self.decision_coverage,
            "signature_coverage": self.signature_coverage,
        }


@dataclass
class CostQAgent:
    cfg: CostQConfig = CostQConfig()
    seed: int = 0
    table: CostQTable | None = None
    diagnostics: EvaluationDiagnostics | None = None

    def __post_init__(self) -> None:
        self.table = self.table or CostQTable()
        self.diagnostics = self.diagnostics or EvaluationDiagnostics()
        self.rng = random.Random(self.seed)

    @property
    def name(self) -> str:
        return "CostQ+GreedyFallback"

    @property
    def fallback_count(self) -> int:
        return self.diagnostics.fallback_count

    @property
    def fallback_rate(self) -> float:
        return self.diagnostics.fallback_rate

    @property
    def coverage_rate(self) -> float:
        return self.diagnostics.signature_coverage

    def key(self, global_state: object, candidate: CandidateProtocol) -> CostQKey:
        return make_cost_q_key(global_state, candidate)

    @staticmethod
    def _candidates(candidates: Iterable[CandidateT]) -> list[CandidateT]:
        materialized = list(candidates)
        if not materialized:
            raise ValueError("Cost-Q action selection requires a feasible candidate")
        return materialized

    @staticmethod
    def _tie_break(candidate: CandidateProtocol) -> tuple[float, float, float, str]:
        return (
            -float(candidate.wait_s),
            float(candidate.estimated_service_s),
            float(candidate.block_entry_s),
            str(candidate.job_id),
        )

    def _argmin_q(self, state: object, candidates: list[CandidateT]) -> CandidateT:
        return min(
            candidates,
            key=lambda candidate: (self.table.value(self.key(state, candidate)),
                                   *self._tie_break(candidate)),
        )

    def _immediate_cost_greedy(self, candidates: list[CandidateT]) -> CandidateT:
        # (queue_length - 1) / (60*N) is common to every candidate, so the
        # deterministic argmin is shortest estimated service followed by the
        # registered Cost-Q tie-break order.
        return min(
            candidates,
            key=lambda candidate: (float(candidate.estimated_service_s),
                                   *self._tie_break(candidate)),
        )

    def act_train(
        self,
        global_state: object,
        candidates: Iterable[CandidateT],
        epsilon: float,
    ) -> CandidateT:
        """Prioritize unseen signatures, then epsilon exploration, then min-Q."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        feasible = self._candidates(candidates)
        candidates_by_key: dict[CostQKey, list[CandidateT]] = {}
        for candidate in feasible:
            candidates_by_key.setdefault(self.key(global_state, candidate), []).append(candidate)

        unseen = [key for key in candidates_by_key if not self.table.is_visited(key)]
        if unseen:
            selected_key = self.rng.choice(unseen)
            return min(candidates_by_key[selected_key], key=self._tie_break)
        if self.rng.random() < epsilon:
            return self.rng.choice(feasible)
        return self._argmin_q(global_state, feasible)

    def act(self, global_state: object, candidates: Iterable[CandidateT]) -> CandidateT:
        """Evaluate with min-Q only under full signature coverage."""
        feasible = self._candidates(candidates)
        unique_keys = tuple(dict.fromkeys(self.key(global_state, c) for c in feasible))
        visited = sum(self.table.is_visited(key) for key in unique_keys)

        self.diagnostics.decisions += 1
        self.diagnostics.signatures_checked += len(unique_keys)
        self.diagnostics.visited_signatures += visited
        if visited != len(unique_keys):
            self.diagnostics.fallback_count += 1
            return self._immediate_cost_greedy(feasible)

        self.diagnostics.fully_covered_decisions += 1
        return self._argmin_q(global_state, feasible)

    select_train = act_train
    select_eval = act

    def update(
        self,
        global_state: object,
        candidate: CandidateProtocol,
        cost: float,
        next_global_state: object | None,
        next_candidates: Iterable[CandidateProtocol],
        done: bool,
    ) -> float:
        """Apply ``Q <- Q + n^-p * (cost + min Q' - Q)`` with gamma=1."""
        step_cost = float(cost)
        if not math.isfinite(step_cost) or step_cost < 0.0:
            raise ValueError("Cost-Q step cost must be finite and non-negative")
        if done:
            target = step_cost
        else:
            if next_global_state is None:
                raise ValueError("non-terminal Cost-Q update requires next_global_state")
            next_keys = [self.key(next_global_state, item) for item in next_candidates]
            target = step_cost + self.table.min_value(next_keys)
        return self.table.update(
            self.key(global_state, candidate),
            target,
            self.cfg.learning_rate_power,
        )

    def reset_diagnostics(self) -> None:
        self.diagnostics = EvaluationDiagnostics()

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "yard-rl-cost-q-agent-v1",
            "config": asdict(self.cfg),
            "seed": self.seed,
            "table": self.table.to_payload(),
            "diagnostics": asdict(self.diagnostics),
            "rng_state": _encode(_canonical(self.rng.getstate())),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CostQAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "yard-rl-cost-q-agent-v1":
            raise ValueError("unsupported Cost-Q agent format")
        agent = cls(
            cfg=CostQConfig(**payload["config"]),
            seed=int(payload["seed"]),
            table=CostQTable.from_payload(payload["table"]),
            diagnostics=EvaluationDiagnostics(**payload["diagnostics"]),
        )
        rng_state = _decode(payload["rng_state"])
        if not isinstance(rng_state, tuple):
            raise ValueError("invalid Cost-Q RNG state")
        agent.rng.setstate(rng_state)
        return agent
