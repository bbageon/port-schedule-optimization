"""공개정보 기반 현실형 예측 rollout (YR-087).

기존 rollout(`_rollout_cost`)은 deepcopy 후 **실제 엔진**으로 진행 → 큐의 `actual_block_arrival`
(진짜 미래 도착)을 소비 = **오라클(상한)**. 이 모듈의 `PredictiveRollout` 은 미래(미도착) 외부트럭의
도착 이벤트를 **`provided_eta ± Uniform(eta_error_s)` 예측**으로 교체한 K 개 시나리오에서 rollout 을
평가·평균한다 — 즉 진짜 미래 없이 공개정보(제공 ETA·본선 공개 cadence·현재 상태)만으로 결정한다.

단일 leak = 트럭 도착시각(본선 STS 는 이미 결정론 cadence 로 진행하므로 truth 아님). 검증(YR-087,
2026-07-25): 현실형 berth 이 SF 대비 유의개선(−16.4)·오라클과 구분 안 됨 → 하이브리드 근거.
"""
from __future__ import annotations

import copy
import heapq
import random

from .baselines import (JointRolloutGreedy, _feasible_joint, _rollout_cost, _wait_of)
from .candidates import CandidateGenerator
from .events import _Entry

DEFAULT_ETA_ERROR_S = 300.0          # scenario_gen 기본 (eta = actual ± Uniform(eta_error_s))


def make_predicted_scratch(sim, rng: random.Random, eta_error_s: float = DEFAULT_ETA_ERROR_S):
    """미래(미도착) 외부트럭 도착을 `provided_eta ± Uniform(eta_error_s)` 예측으로 교체한 sim 사본.

    이미 도착한 트럭·본선·현재 장비/버퍼 상태는 그대로(알려진 것). 예측 도착으로 `actual_block_arrival`
    과 큐의 BLOCK_ARRIVAL 시각을 함께 갱신해 cum_wait 기준을 일치시킨다. deepcopy 라 원본 불변.
    """
    scratch = copy.deepcopy(sim)
    now = scratch.now
    pred: dict[str, float] = {}
    for jid, j in scratch.jobs.items():
        if (getattr(j, "is_external_truck", False) and j.actual_block_arrival is not None
                and j.actual_block_arrival > now):
            # 외부감사 결함4 정정: ETA 결측 트럭을 실제 도착으로 대체하던 조건부 누출 제거 —
            # fail-closed (예측 불가면 예외). ETA 완비 시나리오(현 실험 전부)는 영향 없음.
            # 잔여 누출(deepcopy 가 미통지 고장·계획변경 이벤트 보유)은 YR-093 전용 작업.
            eta = j.provided_eta
            if eta is None:
                raise ValueError(f"{jid}: provided_eta 없음 — 현실형 예측 rollout 은 "
                                 "공개 ETA 없는 미래 트럭을 다룰 수 없다 (YR-093)")
            pt = max(now + 1.0, eta + rng.uniform(-eta_error_s, eta_error_s))
            j.actual_block_arrival = pt
            pred[jid] = pt
    if pred:
        heap = [
            _Entry(pred[e.payload], e.priority, e.seq, e.payload, e.data, e.kind_name)
            if (e.kind_name == "BLOCK_ARRIVAL" and e.payload in pred) else e
            for e in scratch.queue._heap
        ]
        heapq.heapify(heap)
        scratch.queue._heap = heap
    return scratch


class PredictiveRollout:
    """현실형 rollout — 후보 열거·feasibility 는 실제(현재상태), 평가는 K 개 예측 시나리오 rollout 평균.

    reward_calc·horizon_s·generator·base_policy 는 `JointRolloutGreedy`(오라클)와 동일 규약이라
    동일 하네스(`run_joint_episode`)에서 오라클과 직접 비교 가능. `seed` 로 예측 표본 결정론.
    """

    name = "REALISTIC_ROLLOUT"

    def __init__(self, reward_calc, *, horizon_s: float = 1800.0, generator=None,
                 eta_error_s: float = DEFAULT_ETA_ERROR_S, k: int = 4, seed: int = 0,
                 base_policy=None, forbid_strategic_wait: bool = True):
        self.rc = reward_calc
        self.horizon_s = horizon_s
        self.generator = generator or CandidateGenerator()
        self.eta_error_s = eta_error_s
        self.k = k
        self.rng = random.Random(seed)
        self.jr = JointRolloutGreedy(reward_calc, horizon_s=horizon_s, generator=self.generator,
                                     base_policy=base_policy,
                                     forbid_strategic_wait=forbid_strategic_wait)
        self.base_policy = self.jr.base_policy

    def decide(self, sim, dp, gen_by) -> dict:
        preds = [make_predicted_scratch(sim, self.rng, self.eta_error_s) for _ in range(self.k)]
        best, best_key = None, None
        for combo in self.jr._admissible_combos(sim, dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            costs = [_rollout_cost(p, assign, self.rc, horizon_s=self.horizon_s,
                                   base_policy=self.base_policy, generator=self.generator)[0]
                     for p in preds]
            score = round(sum(costs) / len(costs), 9)
            key = (score, tuple((c, assign[c].candidate_id) for c in dp.crane_ids))
            if best_key is None or key < best_key:
                best, best_key = assign, key
        return best or {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
