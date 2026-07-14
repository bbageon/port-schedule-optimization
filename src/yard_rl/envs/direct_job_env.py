"""Exp-1 external-truck-only dynamic-job cost environment.

The legacy simulator still calls the two truck flows ``GATE_IN``/``GATE_OUT``.
This module keeps those names behind the boundary and exposes the unambiguous
directions ``TRUCK_TO_YARD``/``YARD_TO_TRUCK``.  No gate, ETA, deadline, or
vessel feature is exposed to a policy.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Iterable, Literal, NamedTuple, TypeAlias

from ..domain.enums import JobFlow, JobStatus
from ..domain.models import Job, TerminalProfile
from ..domain.scenario import Scenario
from ..sim.engine import YardSimulator
from ..sim.stack import YardStacks
from ..sim.travel_time import (gantry_m, move_container, trolley_m)

TransferDirection: TypeAlias = Literal["TRUCK_TO_YARD", "YARD_TO_TRUCK"]
OperationPhase: TypeAlias = Literal["OPERATING", "CLEAR_OUT"]


class YardState(NamedTuple):
    """운영자가 읽을 수 있는 v1_rich 야드 상태.

    NamedTuple을 사용해 필드 이름을 제공하되, 값과 순서는 기존 5-tuple 그대로
    유지한다. 따라서 저장된 Cost-Q key와 hash/JSON 직렬화가 호환된다.
    """

    work_phase: int
    crane_area: int
    waiting_truck_level: int
    longest_wait_level: int
    over_30min_truck_count: int


class JobState(NamedTuple):
    """운영자가 읽을 수 있는 후보 작업 정보 (v1_final — 사용자 최종안 2026-07-14).

    구성은 v1_rich 후보 5-tuple 과 같되 입도가 다르다: truck_wait 4단계
    (짧음/보통/김/30분 이상), crane_travel_time 3단계(가까움/보통/멂).
    반입(TRUCK_TO_YARD)은 job_type 이 키에 있으므로 containers_to_move_first=0
    이 반출의 '없음'과 충돌하지 않는다 (반입 = 해당 없음).
    """

    job_type: TransferDirection
    truck_wait: int
    crane_travel_time: int
    total_work_time: int
    containers_to_move_first: int


# v2_minimal: (operation_phase, waiting_truck_level) / v1_rich·v1_final: YardState
MinimalYardState: TypeAlias = tuple[OperationPhase, int]
GlobalState: TypeAlias = YardState | MinimalYardState
CandidateFeature: TypeAlias = tuple
CostQKey: TypeAlias = tuple[GlobalState, CandidateFeature]
STATE_SCHEMAS = ("v2_minimal", "v1_rich", "v1_final")


class SLAMode(str, Enum):
    OFF = "SLA_OFF"
    ON = "SLA_ON"


class DirectJobEpisodeError(RuntimeError):
    """The scenario cannot satisfy the registered all-job clear-out contract."""


def _bucket(value: float, bounds: tuple[float, ...]) -> int:
    """Right-open bins: an exact edge belongs to the higher bucket.

    In particular, ``wait == SLA`` is encoded on the SLA-over side, matching
    the inclusive SLA action mask.
    """
    return bisect_right(bounds, value)


def _quartile_bounds(values: Iterable[float], hard_edges: tuple[float, ...] = ()) -> tuple[float, ...]:
    xs = sorted(float(v) for v in values if isfinite(float(v)) and float(v) >= 0.0)
    quantiles = () if not xs else tuple(xs[min(len(xs) - 1, int(len(xs) * q))]
                                          for q in (0.25, 0.5, 0.75))
    return tuple(sorted(set((*quantiles, *hard_edges))))


def _tertile_bounds(values: Iterable[float], hard_edges: tuple[float, ...] = ()) -> tuple[float, ...]:
    """v1_final 의 저입도 edge — 3분위(33/67%) + hard edge."""
    xs = sorted(float(v) for v in values if isfinite(float(v)) and float(v) >= 0.0)
    tertiles = () if not xs else tuple(xs[min(len(xs) - 1, int(len(xs) * q))]
                                         for q in (1 / 3, 2 / 3))
    return tuple(sorted(set((*tertiles, *hard_edges))))


_BUCKET_FIELDS = ("queue_len", "service_s", "oldest_wait_s", "own_wait_s", "reach_s",
                  "truck_wait_s", "crane_travel_s")


@dataclass(frozen=True)
class DirectJobBucketConfig:
    """Immutable bucket edges; ``fit`` must only receive training observations.

    v2_minimal 은 queue_len·service_s 만 사용한다. v1_rich(YR-028 ablation 복원)는
    oldest/own wait(30분 hard edge 포함)·reach edge 를 추가로 사용한다 — YR-027 v1
    구현(20a42cf)과 동일 정의.
    """

    queue_len: tuple[float, ...] = (1.0, 3.0, 6.0)
    service_s: tuple[float, ...] = (120.0, 300.0, 600.0)
    oldest_wait_s: tuple[float, ...] = (300.0, 900.0, 1800.0)
    own_wait_s: tuple[float, ...] = (300.0, 900.0, 1800.0)
    reach_s: tuple[float, ...] = (60.0, 150.0, 300.0)
    # v1_final 저입도 edge (사용자 최종안): truck_wait 4단계 / crane_travel 3단계
    truck_wait_s: tuple[float, ...] = (600.0, 1800.0)
    crane_travel_s: tuple[float, ...] = (90.0, 240.0)
    fitted: bool = False

    def __post_init__(self) -> None:
        for name in _BUCKET_FIELDS:
            edges = getattr(self, name)
            if any(not isfinite(x) or x < 0.0 for x in edges) or tuple(sorted(edges)) != edges:
                raise ValueError(f"{name} bucket edge는 유한한 비음수 오름차순이어야 함")

    @classmethod
    def fit(cls, *, queue_lengths: Iterable[float],
            service_times_s: Iterable[float],
            oldest_waits_s: Iterable[float] | None = None,
            own_waits_s: Iterable[float] | None = None,
            reaches_s: Iterable[float] | None = None,
            sla_s: float = 1800.0) -> "DirectJobBucketConfig":
        extra: dict[str, tuple[float, ...]] = {}
        if oldest_waits_s is not None:
            extra["oldest_wait_s"] = _quartile_bounds(oldest_waits_s, (float(sla_s),))
        if own_waits_s is not None:
            own = [float(w) for w in own_waits_s]
            extra["own_wait_s"] = _quartile_bounds(own, (float(sla_s),))
            extra["truck_wait_s"] = _tertile_bounds(
                [w for w in own if w < sla_s], (float(sla_s),))
        if reaches_s is not None:
            reaches = [float(r) for r in reaches_s]
            extra["reach_s"] = _quartile_bounds(reaches)
            extra["crane_travel_s"] = _tertile_bounds(reaches)
        return cls(
            queue_len=_quartile_bounds(queue_lengths),
            service_s=_quartile_bounds(service_times_s),
            fitted=True,
            **extra,
        )

    def save(self, path: str | Path) -> None:
        payload = {name: list(getattr(self, name)) for name in _BUCKET_FIELDS}
        payload["fitted"] = self.fitted
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DirectJobBucketConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = {name: tuple(float(x) for x in payload[name])
                    for name in _BUCKET_FIELDS if name in payload}
        return cls(**selected, fitted=bool(payload.get("fitted", False)))


@dataclass(frozen=True)
class DirectJobCandidate:
    """Policy-safe dynamic action and its deterministic tie-break inputs."""

    job_id: str
    feature: CandidateFeature
    transfer_direction: TransferDirection
    wait_s: float
    reach_s: float
    estimated_service_s: float
    end_crane_zone: int
    blocker_count: int
    block_entry_s: float
    # greedy 즉시비용 ĉ(j) = (대기대수-1)×service/(60N) — greedy-prior Q0 용 (분 단위)
    prior_cost: float = 0.0

    @property
    def service_s(self) -> float:
        return self.estimated_service_s

    @property
    def tie_break_key(self) -> tuple[float, float, float, str]:
        return (-self.wait_s, self.estimated_service_s, self.block_entry_s, self.job_id)

    def cost_q_key(self, state: GlobalState) -> CostQKey:
        return state, self.feature

    def key(self, state: GlobalState) -> CostQKey:
        return self.cost_q_key(state)


@dataclass(frozen=True)
class DirectJobRawGlobal:
    now_s: float
    horizon_s: float
    waiting_truck_count: int
    crane_position_bay: float = 0.0
    longest_wait_s: float = 0.0
    over_30min_truck_count: int = 0


@dataclass(frozen=True)
class DirectJobStepInfo:
    candidates: tuple[DirectJobCandidate, ...]
    feasible_candidates: tuple[DirectJobCandidate, ...]
    selected_job: str | None
    elapsed_s: float
    queue_area_delta_s: float
    step_cost: float
    cumulative_cost: float
    cumulative_queue_area_s: float
    n_config: int
    raw_global: DirectJobRawGlobal | None
    sla_mode: SLAMode
    sla_restricted: bool
    masked_job_ids: tuple[str, ...]
    backlog: int
    completed_external: int
    episode_success: bool | None

    @property
    def allowed_job_ids(self) -> tuple[str, ...]:
        return tuple(c.job_id for c in self.candidates)


class DirectJobEnv:
    """Dynamic Job action SMDP; each step returns a non-negative minute cost."""

    def __init__(self, profile: TerminalProfile, *, sla_mode: SLAMode | str = SLAMode.OFF,
                 bucket_cfg: DirectJobBucketConfig | None = None,
                 expected_n_config: int | None = None, check_invariants: bool = True,
                 strict_clear_out: bool = True, state_schema: str = "v2_minimal"):
        if state_schema not in STATE_SCHEMAS:
            raise ValueError(f"state_schema must be one of {STATE_SCHEMAS}: {state_schema}")
        self.profile = profile
        self.sla_mode = SLAMode(sla_mode)
        self.state_schema = state_schema
        self.buckets = bucket_cfg or DirectJobBucketConfig()
        self.expected_n_config = expected_n_config
        self.check_invariants = check_invariants
        self.strict_clear_out = strict_clear_out
        self.sim: YardSimulator | None = None
        self.n_config = 0
        self.n_steps = 0
        self._accounted_area_s = 0.0
        self.cumulative_cost = 0.0
        self._last_info: DirectJobStepInfo | None = None

    def reset(self, scenario: Scenario) -> tuple[GlobalState | None, DirectJobStepInfo]:
        forbidden = sorted(j.job_id for j in scenario.jobs if not j.is_external_truck)
        if forbidden:
            raise ValueError(f"Exp-1은 n_vessel=0 외부트럭 전용이어야 함: {forbidden}")
        configured_vessels = re.search(r"(?:^|_)v(\d+)(?:_|$)", scenario.scenario_id)
        if configured_vessels and int(configured_vessels.group(1)) != 0:
            raise ValueError(
                f"Exp-1 scenario_id는 n_vessel=0이어야 함: {scenario.scenario_id}")
        actual_n = len(scenario.jobs)
        if actual_n <= 0:
            raise ValueError("N_config는 1 이상이어야 함")
        if self.expected_n_config is not None and actual_n != self.expected_n_config:
            raise ValueError(f"N_config 불일치: expected={self.expected_n_config}, actual={actual_n}")
        if self.n_config and actual_n != self.n_config:
            raise ValueError(f"동일 env arm의 N_config 변경 금지: {self.n_config} -> {actual_n}")
        self.n_config = actual_n
        self.sim = YardSimulator(self.profile, scenario, check_invariants=self.check_invariants)
        self.n_steps = 0
        self._accounted_area_s = 0.0
        self.cumulative_cost = 0.0
        decision = self.sim.run_until_decision()
        state, info = self._observe(selected_job=None, elapsed_s=0.0,
                                    area_delta_s=0.0, step_cost=0.0,
                                    terminal=decision is None)
        self._last_info = info
        if decision is None:
            self._guard_terminal(info)
        return state, info

    def step(self, action: str | DirectJobCandidate
             ) -> tuple[GlobalState | None, float, bool, DirectJobStepInfo]:
        if self.sim is None or self.sim.terminal:
            raise RuntimeError("reset 이후 비종료 상태에서만 step 가능")
        current = self._last_info
        if current is None or not current.candidates:
            raise RuntimeError("현재 direct-job decision snapshot이 없음")
        job_id = action.job_id if isinstance(action, DirectJobCandidate) else str(action)
        if job_id not in current.allowed_job_ids:
            raise ValueError(f"현재 허용 후보가 아닌 job action: {job_id}")
        t0 = self.sim.now
        self.sim.execute_job(job_id)
        self.n_steps += 1
        state, info = self._advance_and_observe(selected_job=job_id,
                                                elapsed_s=self.sim.now - t0)
        return state, info.step_cost, state is None, info

    @property
    def terminal(self) -> bool:
        return bool(self.sim and self.sim.terminal)

    def _advance_and_observe(self, *, selected_job: str | None, elapsed_s: float
                             ) -> tuple[GlobalState | None, DirectJobStepInfo]:
        before = self.sim.now
        decision = self.sim.run_until_decision()
        elapsed = elapsed_s + (self.sim.now - before)
        area = self.sim.kpis.queue_area_s
        delta = area - self._accounted_area_s
        if delta < -1e-8:
            raise AssertionError("queue-area가 감소함")
        self._accounted_area_s = area
        cost = max(0.0, delta) / (60.0 * self.n_config)
        self.cumulative_cost += cost
        state, info = self._observe(selected_job=selected_job, elapsed_s=elapsed,
                                    area_delta_s=max(0.0, delta), step_cost=cost,
                                    terminal=decision is None)
        self._last_info = info
        if decision is None:
            self._guard_terminal(info)
        return state, info

    def _observe(self, *, selected_job: str | None, elapsed_s: float,
                 area_delta_s: float, step_cost: float,
                 terminal: bool = False) -> tuple[GlobalState | None, DirectJobStepInfo]:
        if terminal:
            backlog = self.sim.unfinished_backlog()
            success = backlog == 0 and self.sim.kpis.completed_external == self.n_config
            return None, DirectJobStepInfo(
                (), (), selected_job, elapsed_s, area_delta_s, step_cost,
                self.cumulative_cost, self.sim.kpis.queue_area_s, self.n_config,
                None, self.sla_mode,
                False, (), backlog, self.sim.kpis.completed_external, success)
        raw = self._raw_global()
        state = self._encode_global(raw)
        feasible = tuple(self._candidate(j, raw) for j in self._feasible_jobs())
        if self.state_schema == "v1_final":
            self._check_state_consistency(raw, feasible)
        allowed, restricted = self._apply_sla(feasible)
        masked = tuple(c.job_id for c in feasible if c not in allowed)
        return state, DirectJobStepInfo(
            allowed, feasible, selected_job, elapsed_s, area_delta_s, step_cost,
            self.cumulative_cost, self.sim.kpis.queue_area_s, self.n_config,
            raw, self.sla_mode,
            restricted, masked, self.sim.unfinished_backlog(),
            self.sim.kpis.completed_external, None)

    def _feasible_jobs(self) -> list[Job]:
        now = self.sim.now
        jobs = [j for j in self.sim.dispatchable_jobs()
                if j.is_external_truck and j.status == JobStatus.WAITING
                and j.actual_block_arrival is not None and j.actual_block_arrival <= now + 1e-9]
        jobs.sort(key=lambda j: j.job_id)
        return jobs

    def _waiting_jobs(self) -> list[Job]:
        return sorted((j for j in self.sim.jobs.values()
                       if j.is_external_truck and j.status == JobStatus.WAITING),
                      key=lambda j: j.job_id)

    def _raw_global(self) -> DirectJobRawGlobal:
        waiting = self._waiting_jobs()
        waits = [max(0.0, self.sim.now - j.actual_block_arrival) for j in waiting]
        return DirectJobRawGlobal(
            now_s=self.sim.now, horizon_s=self.sim.scenario.horizon_s,
            waiting_truck_count=len(waiting),
            crane_position_bay=self.sim.crane.position_bay,
            longest_wait_s=max(waits, default=0.0),
            over_30min_truck_count=sum(
                w >= self.profile.long_wait_sla_s for w in waits))

    def _encode_global(self, raw: DirectJobRawGlobal) -> GlobalState:
        if self.state_schema in ("v1_rich", "v1_final"):
            # YR-027 v1 값·순서 유지: 운영 4구간 + 도착 종료 후(4), 크레인 4구역.
            # v1_final(사용자 최종안 2026-07-14)의 YardState 정의도 이와 동일.
            if raw.now_s >= raw.horizon_s - 1e-9:
                work_phase = 4
            else:
                work_phase = min(
                    3, int(4.0 * raw.now_s / max(raw.horizon_s, 1e-9)))
            return YardState(
                work_phase=work_phase,
                crane_area=self._bay_zone(raw.crane_position_bay),
                waiting_truck_level=_bucket(
                    raw.waiting_truck_count, self.buckets.queue_len),
                longest_wait_level=_bucket(
                    raw.longest_wait_s, self.buckets.oldest_wait_s),
                over_30min_truck_count=min(3, raw.over_30min_truck_count),
            )
        phase: OperationPhase = ("CLEAR_OUT" if raw.now_s >= raw.horizon_s - 1e-9
                                 else "OPERATING")
        return phase, _bucket(raw.waiting_truck_count, self.buckets.queue_len)

    def _candidate(self, job: Job,
                   raw: DirectJobRawGlobal | None = None) -> DirectJobCandidate:
        direction: TransferDirection = ("TRUCK_TO_YARD" if job.flow == JobFlow.GATE_IN
                                        else "YARD_TO_TRUCK")
        wait = max(0.0, self.sim.now - job.actual_block_arrival)
        blockers = (len(self.sim.stacks.blockers_above(job.target_container))
                    if job.flow == JobFlow.GATE_OUT else 0)
        reach = self._reach_s(job)
        service, end_bay = self._estimate_service(job)
        end_zone = self._bay_zone(end_bay)
        if self.state_schema == "v1_rich":
            feature: CandidateFeature = (
                direction, _bucket(wait, self.buckets.own_wait_s),
                _bucket(reach, self.buckets.reach_s),
                _bucket(service, self.buckets.service_s), min(3, blockers))
        elif self.state_schema == "v1_final":
            feature = JobState(
                job_type=direction,
                truck_wait=_bucket(wait, self.buckets.truck_wait_s),
                crane_travel_time=_bucket(reach, self.buckets.crane_travel_s),
                total_work_time=_bucket(service, self.buckets.service_s),
                containers_to_move_first=min(3, blockers),
            )
        else:
            feature = (direction, _bucket(service, self.buckets.service_s), end_zone)
        waiting = (raw.waiting_truck_count if raw is not None
                   else len(self._waiting_jobs()))
        prior = max(0, waiting - 1) * service / (60.0 * max(1, self.n_config))
        return DirectJobCandidate(job.job_id, feature, direction, wait, reach, service,
                                  end_zone, blockers, float(job.actual_block_arrival),
                                  prior)

    def _check_state_consistency(self, raw: DirectJobRawGlobal,
                                 feasible: tuple[DirectJobCandidate, ...]) -> None:
        """v1_final 상태 일관성 규칙 (사용자 최종안 §5) — 위반 시 즉시 중단."""
        if raw.over_30min_truck_count > raw.waiting_truck_count:
            raise AssertionError("30분 초과 차량 수가 전체 대기 차량 수를 초과")
        for c in feasible:
            if c.wait_s > raw.longest_wait_s + 1e-6:
                raise AssertionError(
                    f"후보 대기({c.wait_s:.0f}s)가 최장 대기({raw.longest_wait_s:.0f}s) 초과")
            if c.transfer_direction == "TRUCK_TO_YARD" and c.blocker_count != 0:
                raise AssertionError("반입 작업에 선행 이동 컨테이너가 계상됨")

    def _apply_sla(self, candidates: tuple[DirectJobCandidate, ...]
                   ) -> tuple[tuple[DirectJobCandidate, ...], bool]:
        if self.sla_mode == SLAMode.OFF:
            return candidates, False
        overdue = tuple(c for c in candidates if c.wait_s >= self.profile.long_wait_sla_s)
        return (overdue, True) if overdue else (candidates, False)

    def _reach_s(self, job: Job) -> float:
        geom, spec, stacks, crane = (self.profile.block, self.profile.crane,
                                     self.sim.stacks, self.sim.crane)
        if job.flow == JobFlow.GATE_OUT:
            c = stacks.containers[job.target_container]
            bay, row = float(c.bay), float(c.row)
        else:
            slot = stacks.find_slot(job.inbound_size, spec, crane.position_bay, crane.trolley_row)
            if slot is None:
                raise AssertionError("dispatchable GATE_IN에 장치 슬롯이 없음")
            bay, row = float(slot[0]), float(geom.transfer_row)
        return (gantry_m(geom, crane.position_bay, bay) / spec.gantry_speed_mps
                + trolley_m(geom, crane.trolley_row, row) / spec.trolley_speed_mps)

    def _bay_zone(self, bay: float) -> int:
        spec = self.profile.crane
        width = max(1.0, float(spec.service_bay_max - spec.service_bay_min + 1))
        return min(3, max(0, int(4.0 * (bay - spec.service_bay_min) / width)))

    def _estimate_service(self, job: Job) -> tuple[float, float]:
        geom, spec, stacks, crane = (self.profile.block, self.profile.crane,
                                     self.sim.stacks, self.sim.crane)
        cb, cr = crane.position_bay, crane.trolley_row
        if job.flow == JobFlow.GATE_IN:
            db, dr = stacks.find_slot(job.inbound_size, spec, cb, cr)
            tier = stacks.top_tier(db, dr) + 1
            mv = move_container(spec, geom, cb, cr, (db, geom.transfer_row, 1),
                                (db, dr, tier))
            return mv.duration_s + spec.truck_positioning_time_s, float(db)
        end_bay = float(stacks.containers[job.target_container].bay)
        return self._estimate_outbound(job, stacks, cb, cr), end_bay

    def _estimate_outbound(self, job: Job, stacks: YardStacks,
                           cur_bay: float, cur_row: float) -> float:
        """Pure overlay of the engine's deterministic relocation planner."""
        geom, spec = self.profile.block, self.profile.crane
        target_id = job.target_container
        blockers = stacks.blockers_above(target_id)
        if not blockers:
            target = stacks.containers[target_id]
            mv = move_container(
                spec, geom, cur_bay, cur_row,
                (target.bay, target.row, target.tier),
                (target.bay, geom.transfer_row, 1),
            )
            return mv.duration_s + spec.truck_positioning_time_s
        piles = {slot: list(pile) for slot, pile in stacks._stacks.items()}
        positions = {cid: (c.bay, c.row, c.tier) for cid, c in stacks.containers.items()}
        sizes = {cid: c.size for cid, c in stacks.containers.items()}

        def find_slot(size, near_bay, near_row, exclude):
            best = None
            for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
                for row in range(1, geom.row_count + 1):
                    if (bay, row) in exclude:
                        continue
                    pile = piles.get((bay, row), [])
                    if len(pile) >= geom.tier_max or (pile and sizes[pile[-1]] != size):
                        continue
                    score = (gantry_m(geom, near_bay, bay)
                             + trolley_m(geom, near_row, row)
                             + len(pile) * geom.tier_height_m, bay, row)
                    if best is None or score < best:
                        best = score
            return None if best is None else (best[1], best[2])

        total = 0.0
        for blocker_id in blockers:
            bay, row, tier = positions[blocker_id]
            dest = find_slot(sizes[blocker_id], float(bay), float(row), {(bay, row)})
            if dest is None:
                raise AssertionError("dispatchable GATE_OUT에 재조작 슬롯이 없음")
            db, dr = dest
            piles[(bay, row)].pop()
            dtier = len(piles.get((db, dr), [])) + 1
            mv = move_container(spec, geom, cur_bay, cur_row, (bay, row, tier),
                                (db, dr, dtier))
            total += mv.duration_s
            cur_bay, cur_row = mv.end_bay, mv.end_row
            piles.setdefault((db, dr), []).append(blocker_id)
            positions[blocker_id] = (db, dr, dtier)
        bay, row, tier = positions[target_id]
        mv = move_container(spec, geom, cur_bay, cur_row, (bay, row, tier),
                            (bay, geom.transfer_row, 1))
        return total + mv.duration_s + spec.truck_positioning_time_s

    def _guard_terminal(self, info: DirectJobStepInfo) -> None:
        expected_area = sum(self.sim.kpis.wait_samples_s)
        if abs(self.sim.kpis.queue_area_s - expected_area) > 1e-6:
            raise AssertionError("queue-area와 외부트럭 대기합 불일치")
        if self.strict_clear_out and not info.episode_success:
            raise DirectJobEpisodeError(
                f"clear-out 실패: completed={info.completed_external}/{self.n_config}, "
                f"backlog={info.backlog}; scenario drain_window을 검토해야 함")
        if info.episode_success:
            if abs(self._accounted_area_s - self.sim.kpis.queue_area_s) > 1e-6:
                raise AssertionError("step cost queue-area 누락")
            area_cost = self.sim.kpis.queue_area_s / (60.0 * self.n_config)
            mean_wait_min = expected_area / (60.0 * self.n_config)
            if (abs(self.cumulative_cost - area_cost) > 1e-9
                    or abs(self.cumulative_cost - mean_wait_min) > 1e-9):
                raise AssertionError("Σstep_cost = queue_area/(60N) = mean_wait_min 항등식 위반")
