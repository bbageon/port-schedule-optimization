"""정규화 Core Cost 보상 — 구현계획 02 §7.

Reward = -C, C = Σ w_k · (구간 증가분 / Scale_k).
Scale 은 train 기간 FIFO Baseline 실행 결과로 산정 후 고정 (test 재산정 금지).
탄소는 항 자체가 없다 (§14.6·§20). 안전은 Hard Constraint — 비용항 아님.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..sim.kpis import KpiSnapshot


@dataclass
class RewardConfig:
    # 가중치 (w_W=1 고정, 나머지는 {0,0.1,0.3,1.0} 후보 — validation 탐색 대상.
    # 예비 PoC 는 중간값 고정을 명시적 가정으로 사용)
    w_wait: float = 1.0
    w_tail: float = 0.3
    w_move: float = 0.1
    w_rehandle: float = 0.1
    w_vessel: float = 0.3
    eta_empty: float = 0.0        # 공주행 가중 (자료 없음 → 0, assumed)
    # Scale (train FIFO 로 fit)
    s_wait: float = 1.0
    s_tail: float = 1.0
    s_move: float = 1.0
    s_rehandle: float = 1.0
    s_vessel: float = 1.0
    fitted: bool = False

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RewardConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def fit_scales(self, *, total_queue_area: float, total_tail_area: float,
                   total_move_m: float, total_rehandles: float, total_vessel_delay: float,
                   n_steps: int):
        """FIFO train 합계 → 의사결정 1회당 평균 규모로 정규화 (0 이면 대체 기준)."""
        n = max(1, n_steps)
        self.s_wait = max(1e-6, total_queue_area / n)
        # tail 이 정상부하에서 0 이면 SLA 초과가 거의 없다는 뜻 — queue scale 로 대체하고
        # 활성화 여부를 리포트에 명시 (02 §7: epsilon 임의 주입 금지)
        self.s_tail = max(1e-6, total_tail_area / n) if total_tail_area > 0 else self.s_wait
        self.s_move = max(1e-6, total_move_m / n)
        self.s_rehandle = max(1e-6, total_rehandles / n) if total_rehandles > 0 else 1.0
        self.s_vessel = max(1e-6, total_vessel_delay / n) if total_vessel_delay > 0 else 900.0
        self.fitted = True


@dataclass
class RewardCalculator:
    cfg: RewardConfig
    last: KpiSnapshot | None = field(default=None)

    def reset(self, snap: KpiSnapshot):
        self.last = snap

    def interval_reward(self, snap: KpiSnapshot) -> float:
        prev = self.last
        d_wait = snap.queue_area_s - prev.queue_area_s
        d_tail = snap.tail_area_s - prev.tail_area_s
        d_move = ((snap.loaded_gantry_m - prev.loaded_gantry_m)
                  + (1.0 + self.cfg.eta_empty) * (snap.empty_gantry_m - prev.empty_gantry_m))
        d_re = snap.rehandle_count - prev.rehandle_count
        d_v = snap.vessel_delay_s - prev.vessel_delay_s
        cost = (self.cfg.w_wait * d_wait / self.cfg.s_wait
                + self.cfg.w_tail * d_tail / self.cfg.s_tail
                + self.cfg.w_move * d_move / self.cfg.s_move
                + self.cfg.w_rehandle * d_re / self.cfg.s_rehandle
                + self.cfg.w_vessel * d_v / self.cfg.s_vessel)
        self.last = snap
        return -cost
