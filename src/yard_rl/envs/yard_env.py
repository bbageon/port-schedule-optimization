"""공통 RL 환경 — 구현계획 02 §4.

모든 정책(Baseline·Q-learning)이 같은 env 를 통해 실행된다 → 정보·행동·보상 조건 동일.
step 흐름: mask 재확인 → rule → Job 결정 → 2차 제약검증(engine) → 예약 →
다음 의사결정까지 진행 → 구간 보상. 모든 rule 이 mask 되거나 후보가 비공개면
다음 외부 이벤트까지 자동 진행 (02 §6).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.enums import ControlScope, InformationLevel, PriorityRule
from ..domain.models import Job, TerminalProfile
from ..domain.scenario import Scenario
from ..sim.engine import YardSimulator
from .action_mask import N_ACTIONS, build_mask
from .info_filter import assert_no_leakage, is_visible
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

    # ------------------------------------------------------------------ api
    def reset(self, scenario: Scenario) -> tuple[StateKey | None, StepInfo]:
        self.sim = YardSimulator(self.profile, scenario, check_invariants=self._check)
        self.horizon = scenario.horizon_s
        self.n_steps = 0
        cands = self._advance_to_decision()
        self.reward.reset(self.sim.kpis.snapshot())
        return self._observe(cands, elapsed=0.0, job=None)

    def step(self, action: int) -> tuple[StateKey | None, float, bool, StepInfo]:
        cands = self._visible_candidates()
        mask = self._mask(cands)
        if not (0 <= action < N_ACTIONS and mask[action]):
            raise ValueError(f"mask 위반 action={action} mask={mask}")
        job = self.executor.select(PriorityRule(action), cands,
                                   crane=self.sim.crane, stacks=self.sim.stacks,
                                   now=self.sim.now)
        t0 = self.sim.now
        self.sim.execute_job(job.job_id)   # 내부에서 2차 제약검증
        self.n_steps += 1
        nxt = self._advance_to_decision()
        elapsed = self.sim.now - t0
        r = self.reward.interval_reward(self.sim.kpis.snapshot())
        state, info = self._observe(nxt, elapsed=elapsed, job=job.job_id)
        done = nxt is None
        return state, r, done, info

    @property
    def terminal(self) -> bool:
        return self.sim.terminal

    # ------------------------------------------------------------- internals
    def _visible_candidates(self) -> list[Job]:
        cands = [j for j in self.sim.dispatchable_jobs()
                 if is_visible(j, self.sim.now, self.level)]
        assert_no_leakage(cands, self.sim.now, self.level)  # 미래정보 누출 자동검사
        return cands

    def _mask(self, cands: list[Job]) -> list[bool]:
        return build_mask(cands, level=self.level, scope=self.scope,
                          crane=self.sim.crane, stacks=self.sim.stacks,
                          profile=self.profile)

    def _advance_to_decision(self) -> list[Job] | None:
        """공개된 후보가 있는 의사결정 시점까지 진행. 종료면 None."""
        while True:
            dp = self.sim.run_until_decision()
            if dp is None:
                return None
            cands = self._visible_candidates()
            if cands and any(self._mask(cands)):
                return cands
            # 진실 dispatch 가능하지만 비공개(또는 전 rule mask) → 이벤트 1개 소비
            self.sim.skip_to_next_event()
            if self.sim.terminal:
                return None

    def _observe(self, cands: list[Job] | None, *, elapsed: float,
                 job: str | None) -> tuple[StateKey | None, StepInfo]:
        if cands is None:
            return None, StepInfo([False] * N_ACTIONS, elapsed, job, {})
        state = self.encoder.encode(cands, self.sim.crane, self.sim.stacks,
                                    self.sim.now, self.horizon)
        raw = self.encoder.raw_features(cands, self.sim.crane, self.sim.stacks,
                                        self.sim.now, self.horizon)
        return state, StepInfo(self._mask(cands), elapsed, job, raw)
