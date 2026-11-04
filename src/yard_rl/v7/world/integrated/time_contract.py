"""YR-089 트럭 시간계약 v2 장부 — 블록 처리시간(B→C)·터미널 턴타임(A→O) 독립 적분.

시간축: 예약 T_appt → 실제 게이트진입 A → 블록도착 B → 작업시작 S → 작업완료 C → 게이트진출 O.
- **블록 처리시간 C−B** = 단일 블록 정책의 1차 학습비용 (기존 S−B 대기에 서비스시간 포함 —
  "빨리 시작했지만 오래 처리한" 정책의 오채택 방지).
- **터미널 턴타임 O−A** = 최종 KPI·채택 판정 (게이트·내부도로 미모델 구간 B−A·O−C 는 외생 고정).
- **YC 순수 대기 S−B** = 병목 진단·설명용 (기존 KpiTracker 가 계속 계산 — 비용에서만 제외).

설계 결정: TRUCK_GATE_IN/OUT 을 **이벤트 큐에 넣지 않는다** — A 는 시나리오 외생(사전 공지),
O 는 완료 시 C+exit_travel(외생) 로 확정되므로 장부가 구간적분 중 경계를 직접 처리한다.
이유: ① event_log 골든 불변(호환 arm 바이트 동일) ② "gate 사건이 불필요한 YC 결정시점을
열지 않는다" 계약을 구조로 보장 (spec 의 새 이벤트 명명 규칙은 이벤트를 쓸 경우의 규칙).

지식경계: 이 장부는 비용·KPI 정산 전용(GROUND_TRUTH 소비) — 정책 입력이 아니다. 정책 가시는
appointment_*·estimated_block_arrival (Job 필드·스키마 경로) 뿐이다.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class TruckTimes:
    """외부트럭 1대의 실현 시각 기록 (A/B/S/C/O — None = 미발생)."""
    gate_in: float                      # A (외생 — 시나리오 확정)
    block_arrival: float | None = None  # B
    service_start: float | None = None  # S
    job_done: float | None = None       # C
    gate_out: float | None = None       # O = C + exit_travel (완료 시 확정)


@dataclass
class TimeLedger:
    """v2 비용·KPI 적분 장부. integrate() 는 엔진 _advance 와 같은 구간에서만 호출된다."""

    sla_s: float
    # --- 독립 적분 (등식 테스트 대상) ---
    block_area_s: float = 0.0        # ∫ N_block(t) dt — B≤t<C 점유 (v2 truck_wait 원료)
    block_tail_area_s: float = 0.0   # 그중 (t−B) > sla 구간 (v2 long_wait 원료)
    terminal_area_s: float = 0.0     # ∫ N_inside(t) dt — A≤t<O 점유 (최종 KPI 적분)
    # --- 내부 상태 ---
    records: dict[str, TruckTimes] = field(default_factory=dict)
    _in_block: dict[str, float] = field(default_factory=dict)   # job_id -> B (미완료분)
    _a_sorted: list[float] = field(default_factory=list)        # 전체 A 시각 (정렬, 사전 등록)
    _a_idx: int = 0                                             # A ≤ clock 포인터
    _o_heap: list[float] = field(default_factory=list)          # 확정된 O 시각 (min-heap)
    _n_inside: int = 0                                          # 현재 A≤t<O 트럭 수
    closed_end_s: float | None = None

    # ------------------------------------------------- 등록·사건
    def register(self, job_id: str, gate_in: float) -> None:
        self.records[job_id] = TruckTimes(gate_in=gate_in)
        self._a_sorted.append(gate_in)

    def seal(self) -> None:
        """등록 완료 후 1회 — A 시각 정렬 (reset 마지막에 호출)."""
        self._a_sorted.sort()

    def block_arrival(self, job_id: str, t: float) -> None:
        self.records[job_id].block_arrival = t
        self._in_block[job_id] = t

    def service_start(self, job_id: str, t: float) -> None:
        self.records[job_id].service_start = t

    def job_done(self, job_id: str, t: float, exit_travel_s: float) -> float:
        """완료 → O 확정·경계 등록. 반환 = actual_gate_out."""
        r = self.records[job_id]
        r.job_done = t
        r.gate_out = t + max(0.0, exit_travel_s)
        self._in_block.pop(job_id, None)
        heapq.heappush(self._o_heap, r.gate_out)
        return r.gate_out

    # ------------------------------------------------- 적분
    def integrate(self, lo: float, hi: float) -> None:
        if hi <= lo:
            return
        # 블록 점유 (B·C 는 이벤트 시각과 일치 — 구간 내 상수) + SLA 꼬리 overlap
        n_blk = len(self._in_block)
        if n_blk:
            self.block_area_s += (hi - lo) * n_blk
            for b in self._in_block.values():
                ov = hi - max(lo, b + self.sla_s)
                if ov > 0:
                    self.block_tail_area_s += ov
        # 터미널 점유 — A(외생 사전등록)·O(완료시 확정) 경계는 구간 내부에 올 수 있다 → 조각 적분
        t = lo
        while t < hi:
            next_a = (self._a_sorted[self._a_idx]
                      if self._a_idx < len(self._a_sorted) else float("inf"))
            next_o = self._o_heap[0] if self._o_heap else float("inf")
            nxt = min(next_a, next_o, hi)
            if nxt > t:
                self.terminal_area_s += (nxt - t) * self._n_inside
                t = nxt
            if nxt >= hi:
                break
            if next_a <= next_o:            # 동시각이면 진입 먼저 (보수적 — 순서 결정론)
                self._a_idx += 1
                self._n_inside += 1
            else:
                heapq.heappop(self._o_heap)
                self._n_inside -= 1

    def close(self, end: float) -> None:
        """에피소드 종료 — 이후 표본·검열 계산의 기준시각 고정."""
        self.closed_end_s = end

    # ------------------------------------------------- 표본 (분석·KPI)
    def block_turntime_samples_s(self) -> list[float]:
        """C−B (완료) + 검열 end−B (미완료·도착분) — 미완료가 이득 보지 않게."""
        end = self.closed_end_s
        out = []
        for r in self.records.values():
            if r.block_arrival is None:
                continue
            if r.job_done is not None:
                out.append(r.job_done - r.block_arrival)
            elif end is not None:
                out.append(max(0.0, end - r.block_arrival))
        return out

    def terminal_turntime_samples_s(self) -> list[float]:
        """O−A (완료·O≤end) + 검열 end−A (그 외) — terminal_area 등식과 같은 정의."""
        end = self.closed_end_s
        out = []
        for r in self.records.values():
            if r.gate_out is not None and (end is None or r.gate_out <= end):
                out.append(r.gate_out - r.gate_in)
            elif end is not None and r.gate_in < end:
                out.append(end - r.gate_in)
        return out

    def censored_exposure_s(self) -> float:
        """종료시점 터미널 내(O 미실현 또는 O>end) 차량의 end−A 합 (검열)."""
        end = self.closed_end_s
        if end is None:
            return 0.0
        return sum(end - r.gate_in for r in self.records.values()
                   if r.gate_in < end and (r.gate_out is None or r.gate_out > end))
