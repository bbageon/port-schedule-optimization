"""블록 기하·스펙 상수 — `jit` 의 **static 인자** ([[YR-327]] 조각 1 §2).

배열 세계에서 격자 (B,R,T) 는 **배열 모양 자체**라 컴파일 상수다. 그래서 기하는
배열이 아니라 파이썬 값(frozen dataclass)으로 들고 다니고, `jax.jit(f, static_argnums=…)`
에 넘긴다. 값이 바뀌면 다시 컴파일된다 — 프로파일 하나에 컴파일 한 번이다.

■ 무엇이 들어가나 (v5 출처)
    bay_count·row_count·tier_max       BlockGeometry (domain/models.py:110-118)
    bay_len·row_w·tier_h               같은 곳 — find_slot 비용 계수 (sim/stack.py:142, 164)
    transfer_row                        차선(트럭 인계) row — 크레인 초기 trolley_row 이고
                                        STORE 의 출발·RETRIEVE 의 도착 row (cranes.py:38, engine.py:598, 650)
    sla_s                               long_wait_sla_s — 꼬리 대기 적분 기준 (kpis.py:71-80)
    shift_len_s                         imbalance rate = I/shift_len (engine.py:822)
    gap                                 safety_gap_bay — 예약 통로 충돌 여유 (reservation.py:21-22)
    n_lanes                             lane = (bay−1) mod n_lanes (engine.py:271-273)

■ ★비용표 `row_cost`·`tier_cost` — FMA(곱셈-덧셈 융합) 를 막는 장치
  v5 find_slot 의 비용식은 `g + |Δrow|·row_w + top·tier_h` (stack.py:164) 이고, 파이썬은
  곱을 **먼저 반올림**한 뒤 더한다. XLA 는 `a·b + c` 를 한 번에 반올림(FMA)할 수 있어
  마지막 비트가 갈리고, 동률 판정이 달라져 **다른 칸을 고를 수 있다**(반박 검증 finding 2).
  |Δrow| 와 top 은 정수(0..R / 0..T)라, 곱을 **호스트에서 파이썬으로 미리 계산한 표**로
  두고 배열판은 gather 만 한다 — 곱셈 명령이 없으니 융합할 것도 없다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Geom:
    """블록 하나의 기하·스펙 상수. **hashable** 이라 static 인자로 쓸 수 있다."""

    bay_count: int          # B — 격자 첫 축 (find_slot 은 B·R 칸을 전부 계산)
    row_count: int          # R
    tier_max: int           # T — 계획당 이동 칸 M 도 이 값 (blocker ≤ T−1 + 대상 1)
    bay_len: float          # m — gantry 거리 = |Δbay|·bay_len
    row_w: float            # m — trolley 거리 = |Δrow|·row_w
    tier_h: float           # m — hoist 거리 = ((T+1)−tier)·tier_h (travel_time.py:44-69)
    transfer_row: float     # 차선 row 좌표 (v5 BlockGeometry.transfer_row, 기본 0)
    sla_s: float            # 장기 대기 SLA (초)
    shift_len_s: float      # 교대 길이 (초) — imbalance rate 분모
    gap: float              # safety_gap_bay — 크레인 간 최소 bay 간격
    n_lanes: int            # 레인 수 (0 이면 lane = -1)

    # ── 파생 크기 ────────────────────────────────────────────────
    @property
    def n_moves(self) -> int:
        """M — 계획 하나가 담는 이동 칸 수 (= tier_max, 명세 capacities)."""
        return int(self.tier_max)

    @property
    def cells(self) -> int:
        return int(self.bay_count) * int(self.row_count)

    # ── ★미리 반올림한 비용표 (머리말 참조) ──────────────────────
    @property
    def row_cost(self) -> tuple[float, ...]:
        """row_cost[|Δrow|] = |Δrow|·row_w — 파이썬 곱(한 번 반올림). 길이 R+1."""
        return tuple(float(d) * self.row_w for d in range(int(self.row_count) + 1))

    @property
    def tier_cost(self) -> tuple[float, ...]:
        """tier_cost[top] = top·tier_h — 파이썬 곱(한 번 반올림). 길이 T+1."""
        return tuple(float(t) * self.tier_h for t in range(int(self.tier_max) + 1))

    # ── v5 프로파일에서 만들기 (호스트 1회) ─────────────────────
    @classmethod
    def from_profile(cls, profile) -> "Geom":
        """v5 `IntegratedProfile` (integrated/profile.py:22-34) 에서 상수를 읽는다.

        타입을 import 하지 않고 속성만 읽는다 — gpu/ 는 world/ 에 의존하지 않는다.
        """
        b = profile.block
        return cls(
            bay_count=int(b.bay_count), row_count=int(b.row_count), tier_max=int(b.tier_max),
            bay_len=float(b.bay_length_m), row_w=float(b.row_width_m),
            tier_h=float(b.tier_height_m), transfer_row=float(b.transfer_row),
            sla_s=float(profile.long_wait_sla_s), shift_len_s=float(profile.shift_len_s),
            gap=float(profile.safety_gap_bay),
            n_lanes=int(len(profile.lane_graph.lane_ids)))
