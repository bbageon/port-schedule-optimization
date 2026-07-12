"""공통 RL 환경 — 구현계획 02 §4.

모든 정책(Baseline·Q-learning)이 같은 env 를 통해 실행된다 → 정보·행동·보상 조건 동일.
step 흐름: mask 재확인 → rule → Job/사전행동 결정 → 2차 제약검증(engine) → 예약 →
다음 의사결정까지 진행 → 구간 보상.

Exp 확장:
- 정보수준(level)이 후보 가시성과 도착예상(eta_of)을 결정 (info_filter).
- control_scope 가 사전행동을 결정: plus_positioning → EPA 가 임박 미래작업으로
  빈 크레인 이동 가능, plus_pre_rehandle → PRE_REHANDLE rule 로 blocker 선처리.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.enums import ControlScope, InformationLevel, JobStatus, PriorityRule
from ..domain.models import Job, TerminalProfile
from ..domain.scenario import Scenario
from ..sim.engine import YardSimulator
from .action_mask import (N_ACTIONS, build_mask, future_job_bay,
                          positioning_targets, pre_rehandle_targets,
                          scope_allows_positioning)
from .info_filter import assert_no_leakage, is_visible, predicted_arrival
from .observations import BucketConfig, ObservationEncoder, StateKey
from .rewards import RewardCalculator, RewardConfig
from .rules import PriorityRuleExecutor


@dataclass
class StepInfo:
    action_mask: list[bool]
    elapsed_s: float
    selected_job: str | None
    raw_features: dict


class YardEnv:
    def __init__(self, profile: TerminalProfile, *,
                 info_level: InformationLevel = InformationLevel.BLOCK_ARRIVAL,
                 control_scope: ControlScope = ControlScope.SEQUENCE_ONLY,
                 reward_cfg: RewardConfig | None = None,
                 bucket_cfg: BucketConfig | None = None,
                 check_invariants: bool = False):
        self.profile = profile
        self.level = info_level
        self.scope = control_scope
        self.encoder = ObservationEncoder(profile, bucket_cfg or BucketConfig())
        self.executor = PriorityRuleExecutor(profile)
        self.reward = RewardCalculator(reward_cfg or RewardConfig())
        self._check = check_invariants
        self.sim: YardSimulator | None = None
        # 정보수준별 도착예상 — 실제 도착시각은 어떤 수준에서도 노출 금지
        self.eta_of = lambda j: predicted_arrival(j, self.level,
                                                  profile.gate_travel_estimate_s)

    # ------------------------------------------------------------------ api
    def reset(self, scenario: Scenario) -> tuple[StateKey | None, StepInfo]:
        self.sim = YardSimulator(self.profile, scenario, check_invariants=self._check)
        # 사전행동 scope: dispatch 후보가 없어도 idle 의사결정을 받아 포지셔닝 가능
        self.sim.yield_idle_decisions = scope_allows_positioning(self.scope)
        self.horizon = scenario.horizon_s
        self.n_steps = 0
        pools = self._advance_to_decision()
        self.reward.reset(self.sim.kpis.snapshot())
        return self._observe(pools, elapsed=0.0, job=None)

    def step(self, action: int) -> tuple[StateKey | None, float, bool, StepInfo]:
        cands, future = self._pools()
        mask = self._mask(cands, future)
        if not (0 <= action < N_ACTIONS and mask[action]):
            raise ValueError(f"mask 위반 action={action} mask={mask}")
        rule = PriorityRule(action)
        t0 = self.sim.now
        selected = self._execute(rule, cands, future)
        self.n_steps += 1
        nxt = self._advance_to_decision()
        elapsed = self.sim.now - t0
        r = self.reward.interval_reward(self.sim.kpis.snapshot())
        state, info = self._observe(nxt, elapsed=elapsed, job=selected)
        return state, r, nxt is None, info

    @property
    def terminal(self) -> bool:
        return self.sim.terminal

    # ------------------------------------------------------- 행동 실행 분기
    def _execute(self, rule: PriorityRule, cands: list[Job], future: list[Job]) -> str:
        kw = dict(crane=self.sim.crane, stacks=self.sim.stacks, now=self.sim.now,
                  eta_of=self.eta_of)
        if rule == PriorityRule.PRE_REHANDLE:
            pool = pre_rehandle_targets(future, now=self.sim.now, crane=self.sim.crane,
                                        stacks=self.sim.stacks, profile=self.profile,
                                        eta_of=self.eta_of)
            job = min(pool, key=lambda j: (self.eta_of(j), j.job_id))
            self.sim.execute_pre_rehandle(job.job_id)
            return f"PRE_REHANDLE:{job.job_id}"
        if rule == PriorityRule.EARLIEST_PROVIDED_ARRIVAL:
            cur = [j for j in cands if self.eta_of(j) is not None]
            pos = (positioning_targets(future, now=self.sim.now, crane=self.sim.crane,
                                       stacks=self.sim.stacks, profile=self.profile,
                                       eta_of=self.eta_of)
                   if scope_allows_positioning(self.scope) else [])
            best_pos = min(pos, key=lambda j: (self.eta_of(j), j.job_id)) if pos else None
            # 현재 후보가 없거나, 임박 미래작업의 예상도착이 현재 후보들보다 이르면 포지셔닝
            if best_pos is not None and (
                    not cur or self.eta_of(best_pos) < min(self.eta_of(j) for j in cur)):
                bay = future_job_bay(best_pos, self.sim.stacks, self.profile,
                                     self.sim.crane)
                self.sim.execute_positioning(bay)
                return f"POSITIONING:{best_pos.job_id}"
            job = self.executor.select(rule, cur, **kw)
            self.sim.execute_job(job.job_id)
            return job.job_id
        job = self.executor.select(rule, cands, **kw)
        self.sim.execute_job(job.job_id)
        return job.job_id

    # ------------------------------------------------------------- internals
    def _pools(self) -> tuple[list[Job], list[Job]]:
        """(dispatch 가능 후보, 공개된 미도착 미래작업) — 둘 다 누출 자동검사."""
        cands = [j for j in self.sim.dispatchable_jobs()
                 if is_visible(j, self.sim.now, self.level)]
        future = [j for j in self.sim.jobs.values()
                  if j.is_external_truck and j.status == JobStatus.PLANNED
                  and is_visible(j, self.sim.now, self.level)]
        future.sort(key=lambda j: j.job_id)
        assert_no_leakage(cands, self.sim.now, self.level)
        assert_no_leakage(future, self.sim.now, self.level)
        return cands, future

    def _mask(self, cands: list[Job], future: list[Job]) -> list[bool]:
        return build_mask(cands, level=self.level, scope=self.scope,
                          crane=self.sim.crane, stacks=self.sim.stacks,
                          profile=self.profile, future=future, eta_of=self.eta_of,
                          now=self.sim.now)

    def _advance_to_decision(self) -> tuple[list[Job], list[Job]] | None:
        """실행 가능한 행동이 있는 의사결정 시점까지 진행. 종료면 None.

        사전행동 scope(3B/3C)에서는 엔진이 idle 시점마다 의사결정을 내주고
        (yield_idle_decisions), 실행할 것이 없으면 이벤트 1개씩 소비한다.
        """
        while True:
            dp = self.sim.run_until_decision()
            if dp is None:
                return None  # run_until_decision None ⇒ finalize 완료(terminal)
            cands, future = self._pools()
            if any(self._mask(cands, future)):
                return cands, future
            self.sim.skip_to_next_event()
            if self.sim.terminal:
                return None

    def _observe(self, pools: tuple[list[Job], list[Job]] | None, *, elapsed: float,
                 job: str | None) -> tuple[StateKey | None, StepInfo]:
        if pools is None:
            return None, StepInfo([False] * N_ACTIONS, elapsed, job, {})
        cands, future = pools
        fut_etas = [e for e in (self.eta_of(j) for j in future) if e is not None]
        state = self.encoder.encode(cands, self.sim.crane, self.sim.stacks,
                                    self.sim.now, self.horizon, fut_etas)
        raw = self.encoder.raw_features(cands, self.sim.crane, self.sim.stacks,
                                        self.sim.now, self.horizon, fut_etas)
        return state, StepInfo(self._mask(cands, future), elapsed, job, raw)
