"""KPI 집계 — queue-area 정확 적분 (구현계획 02 §7, 03 §4).

- queue_area: 외부트럭 대기대수 × 시간의 적분 (완료차량 평균이 아니라 area 우선)
- tail_area: 대기가 SLA 를 초과한 구간만의 적분 (경계 통과를 구간 내에서 정확 처리)
- 평가 종료시점 미처리 차량의 대기도 적분에 포함된다 (integrate 를 종료시각까지 호출).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KpiSnapshot:
    queue_area_s: float
    tail_area_s: float
    loaded_gantry_m: float
    empty_gantry_m: float
    rehandle_count: int          # 선재조작 포함 (비용은 동일 항목으로 계상)
    completed_external: int
    completed_vessel: int
    vessel_delay_s: float
    pre_rehandle_count: int = 0  # 이 중 선재조작(도착 전 처리)분 — 리포트 분해용
    positioning_count: int = 0


@dataclass
class KpiTracker:
    sla_s: float
    queue_area_s: float = 0.0
    tail_area_s: float = 0.0
    loaded_gantry_m: float = 0.0
    empty_gantry_m: float = 0.0
    rehandle_count: int = 0
    completed_external: int = 0
    completed_vessel: int = 0
    vessel_delay_s: float = 0.0
    pre_rehandle_count: int = 0
    positioning_count: int = 0
    wait_samples_s: list[float] = field(default_factory=list)   # 대기 표본 (검열 포함)
    # 내부 상태: 현재 대기 중 외부트럭 job_id -> block_arrival
    _waiting: dict[str, float] = field(default_factory=dict)

    # --- 대기열 상태 전이 ---
    def truck_arrived(self, job_id: str, at: float):
        self._waiting[job_id] = at

    def service_started(self, job_id: str, at: float):
        arrival = self._waiting.pop(job_id)
        self.wait_samples_s.append(at - arrival)

    def waiting_count(self) -> int:
        return len(self._waiting)

    def close_censored(self, end: float):
        """종료시점 미서비스 트럭을 검열 표본으로 포함 (완료-only 표본의
        생존편향 방지 — backlog 를 남기는 정책이 대기지표에서 이득 보지 않게).
        _waiting 은 비우지 않는다 (backlog 카운트·정합성 검증용)."""
        for arrival in self._waiting.values():
            self.wait_samples_s.append(max(0.0, end - arrival))

    def oldest_wait_s(self, now: float) -> float:
        if not self._waiting:
            return 0.0
        return now - min(self._waiting.values())

    # --- 시간 적분 ---
    def integrate(self, t0: float, t1: float):
        if t1 <= t0:
            return
        dt = t1 - t0
        self.queue_area_s += dt * len(self._waiting)
        for arrival in self._waiting.values():
            sla_cross = arrival + self.sla_s
            overlap = t1 - max(t0, sla_cross)
            if overlap > 0:
                self.tail_area_s += overlap

    # --- 이동·완료 기록 ---
    def add_travel(self, loaded_m: float, empty_m: float):
        self.loaded_gantry_m += loaded_m
        self.empty_gantry_m += empty_m

    def add_rehandles(self, n: int):
        self.rehandle_count += n

    def job_completed(self, *, external: bool, deadline: float | None, end: float):
        if external:
            self.completed_external += 1
        else:
            self.completed_vessel += 1
            if deadline is not None and end > deadline:
                self.vessel_delay_s += end - deadline

    def snapshot(self) -> KpiSnapshot:
        return KpiSnapshot(self.queue_area_s, self.tail_area_s, self.loaded_gantry_m,
                           self.empty_gantry_m, self.rehandle_count, self.completed_external,
                           self.completed_vessel, self.vessel_delay_s,
                           self.pre_rehandle_count, self.positioning_count)
