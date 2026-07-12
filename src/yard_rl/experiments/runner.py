"""실험 runner — 공통난수 paired 실행 (구현계획 03 §2·§3).

- 같은 seed 의 시나리오(이벤트·초기상태 동일)를 모든 정책에 재사용 (paired).
- bucket·reward Scale 은 train 구간 FIFO 실행으로 fit 후 고정 (03 §2.1).
- seed 대역: train 101+, validation 201+, test 301+ (일자 단위 분리의 합성 대응).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.enums import InformationLevel, PriorityRule
from ..domain.models import TerminalProfile
from ..domain.scenario import Scenario
from ..envs.observations import BucketConfig
from ..envs.rewards import RewardConfig
from ..envs.yard_env import YardEnv
from ..io.scenario_gen import GenParams, generate
from ..policies.baselines import FixedRulePolicy
from .statistics import percentile

TRAIN_SEED0, VAL_SEED0, TEST_SEED0 = 101, 201, 301
_BAND = 100  # seed 대역 폭 — 대역 침범(train-on-test) 방지 가드


def check_seed_bands(n_train: int, n_val: int, n_test: int) -> None:
    """train/val/test seed 대역 중첩 방지 (03 §2.1 분리 규율)."""
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        if not (0 < n <= _BAND):
            raise ValueError(f"{name} 시나리오 수 {n} — 1..{_BAND} 여야 대역 분리가 보장됨")


@dataclass
class EpisodeResult:
    policy: str
    scenario_id: str
    seed: int
    metrics: dict


def collect_metrics(env: YardEnv) -> dict:
    sim = env.sim
    k = sim.kpis
    waits_min = [w / 60.0 for w in k.wait_samples_s]
    sla_min = sim.profile.long_wait_sla_s / 60.0
    return {
        "mean_wait_min": sum(waits_min) / len(waits_min) if waits_min else 0.0,
        "p50_wait_min": percentile(waits_min, 0.50),
        "p90_wait_min": percentile(waits_min, 0.90),
        "p95_wait_min": percentile(waits_min, 0.95),
        "max_wait_min": max(waits_min) if waits_min else 0.0,
        "sla_exceed_count": sum(1 for w in waits_min if w > sla_min),
        "queue_area_h": k.queue_area_s / 3600.0,
        "tail_area_h": k.tail_area_s / 3600.0,
        "loaded_km": k.loaded_gantry_m / 1000.0,
        "empty_km": k.empty_gantry_m / 1000.0,
        "travel_km": (k.loaded_gantry_m + k.empty_gantry_m) / 1000.0,
        "rehandles": float(k.rehandle_count),
        "completed_external": float(k.completed_external),
        "completed_vessel": float(k.completed_vessel),
        "vessel_delay_min": k.vessel_delay_s / 60.0,
        "backlog": float(sim.unfinished_backlog()),
        "n_decisions": float(env.n_steps),
    }


def run_episode(policy, env: YardEnv, scenario: Scenario) -> EpisodeResult:
    state, info = env.reset(scenario)
    while state is not None:
        a = policy.act(state, info.action_mask)
        state, _r, _done, info = env.step(a)
    return EpisodeResult(policy.name, scenario.scenario_id, scenario.seed,
                         collect_metrics(env))


def make_scenarios(profile: TerminalProfile, seed0: int, n: int,
                   params: GenParams) -> list[Scenario]:
    return [generate(profile, seed0 + i, params) for i in range(n)]


def fit_buckets_and_scales(profile: TerminalProfile, train_scenarios: list[Scenario],
                           level: InformationLevel) -> tuple[BucketConfig, RewardConfig]:
    """train 구간 FIFO 실행에서 관측 분위수·비용 Scale 산정 (이후 고정)."""
    env = YardEnv(profile, info_level=level)  # 임시 bucket — raw feature 만 사용
    fifo = FixedRulePolicy(PriorityRule.FIFO)
    q_lens, oldests, reaches = [], [], []
    tot = {"queue": 0.0, "tail": 0.0, "move": 0.0, "re": 0.0, "vd": 0.0, "steps": 0}
    for sc in train_scenarios:
        state, info = env.reset(sc)
        while state is not None:
            f = info.raw_features
            q_lens.append(f["queue_len"])
            oldests.append(f["oldest_wait_s"])
            reaches.append(f["nearest_reach_s"])
            a = fifo.act(state, info.action_mask)
            state, _r, _d, info = env.step(a)
        k = env.sim.kpis
        tot["queue"] += k.queue_area_s
        tot["tail"] += k.tail_area_s
        tot["move"] += k.loaded_gantry_m + k.empty_gantry_m
        tot["re"] += k.rehandle_count
        tot["vd"] += k.vessel_delay_s
        tot["steps"] += env.n_steps
    buckets = BucketConfig.fit(q_lens, oldests, reaches)
    reward = RewardConfig()
    reward.fit_scales(total_queue_area=tot["queue"], total_tail_area=tot["tail"],
                      total_move_m=tot["move"], total_rehandles=tot["re"],
                      total_vessel_delay=tot["vd"], n_steps=tot["steps"])
    return buckets, reward


def evaluate_paired(policies: list, profile: TerminalProfile,
                    scenarios: list[Scenario], *, level: InformationLevel,
                    buckets: BucketConfig, reward: RewardConfig,
                    check_invariants: bool = True) -> dict[str, list[EpisodeResult]]:
    """모든 정책 × 같은 시나리오(공통난수) — 안전검증 켠 채 실행."""
    out: dict[str, list[EpisodeResult]] = {p.name: [] for p in policies}
    for sc in scenarios:
        for p in policies:
            env = YardEnv(profile, info_level=level, bucket_cfg=buckets,
                          reward_cfg=reward, check_invariants=check_invariants)
            out[p.name].append(run_episode(p, env, sc))
    return out
