"""YR-150 0단계 — 터미널 블록 배치와 **목적지별** 주행시간 matrix (합성·명시 선언).

■ 왜 필요한가
지금까지 게이트→블록 주행은 `travel_fn(src, job)` 로 **목적지를 받지 않았고**, 블록 간
우회는 상수 180초였다. 그래서 "어느 블록으로 보내든 주행비용이 같다" = 목적지 선택의
물리적 대가가 0 이었다(YR-151 0A 발견 ②). N블록 이송에서 이 상태로 성능을 논하면
"가까운 블록으로 보내는 이득"이 구조적으로 사라진다.

■ 무엇을 가정하는가 (합성 — 실측 아님)
블록 좌표·도로망 공개자료가 없으므로 **단일 야드축 위에 블록을 등간격으로 놓은 합성 배치**를
쓴다. 게이트는 축의 한쪽 끝에 있다. 규모 앵커(21블록)에 붙인 **가정**이며 실제 터미널
배치의 재현이 아니다 — 주장 범위에 반드시 함께 적는다.

■ 무엇을 지키는가 (앵커 보존 — 튜닝 아님)
게이트→블록 시간의 **전 블록 평균**을 기존 계약값 `GATE_BLOCK_MEAN_S`(300초)에 고정하고,
모든 블록이 기존 지원범위 `[GATE_BLOCK_MIN_S, GATE_BLOCK_MAX_S]`(180~420초) 안에 들어오게
한다. 즉 **평균은 기존 보정 그대로 두고 블록 간 차이만 새로 생긴다**. 범위를 벗어나면
조용히 자르지 않고 예외를 던진다(fail-closed).
"""
from __future__ import annotations

from dataclasses import dataclass

from .scenario_gen import GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S, GATE_BLOCK_MIN_S

TERMINAL_BLOCKS = 21          # 비-DGT 공개 규모 앵커(HJNC 21블록) — 크기만 빌린 값
BLOCK_PITCH_M = 55.0          # 인접 블록 중심 간격 (블록폭+차로, 합성 가정)
TRUCK_SPEED_MPS = 5.0         # 야드 내 트럭 평균 주행속도 ≈ 18 km/h (합성 가정)


def block_ids(n: int = TERMINAL_BLOCKS) -> tuple[str, ...]:
    """터미널 블록 이름 — 위치 순서와 이름 순서가 같다(감사 편의)."""
    if n <= 0:
        raise ValueError("블록 수는 1 이상")
    return tuple(f"Y{i + 1:02d}" for i in range(n))


@dataclass(frozen=True)
class YardLayout:
    """야드축 위 블록 위치와 주행시간. 위치는 미터 절대좌표로 **명시 보관**한다.

    부분집합(N=3 회귀시험)도 전체 터미널의 절대좌표를 그대로 물려받는다 — 3블록만 따로
    재중심화하면 N=3 과 N=21 의 기하가 달라져 회귀시험이 본 환경을 대표하지 못한다.
    """

    ids: tuple[str, ...]
    positions_m: tuple[float, ...]
    speed_mps: float = TRUCK_SPEED_MPS

    def __post_init__(self) -> None:
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("블록 id 는 비어 있지 않고 중복이 없어야 함")
        if len(self.ids) != len(self.positions_m):
            raise ValueError("id 와 위치 개수가 다름")
        if self.speed_mps <= 0:
            raise ValueError("속도는 양수")
        lo, hi = self.gate_time_range_s()
        if lo < GATE_BLOCK_MIN_S - 1e-9 or hi > GATE_BLOCK_MAX_S + 1e-9:
            raise ValueError(                       # 앵커 위반은 조용히 자르지 않는다
                f"게이트→블록 시간이 계약 지원범위를 벗어남: "
                f"[{lo:.1f}, {hi:.1f}] ⊄ [{GATE_BLOCK_MIN_S}, {GATE_BLOCK_MAX_S}]")

    # ------------------------------------------------------------------ 기하
    @property
    def n(self) -> int:
        return len(self.ids)

    def position_m(self, bid: str) -> float:
        try:
            return self.positions_m[self.ids.index(bid)]
        except ValueError as exc:
            raise KeyError(f"배치에 없는 블록: {bid}") from exc

    # ------------------------------------------------------------------ 주행시간
    def gate_to_block_s(self, bid: str) -> float:
        """게이트(축 원점) → 블록 주행시간."""
        return self.position_m(bid) / self.speed_mps

    def block_to_block_s(self, a: str, b: str) -> float:
        """블록 → 블록 주행시간 (대칭·삼각부등식 성립 — 단일축 거리)."""
        return abs(self.position_m(a) - self.position_m(b)) / self.speed_mps

    def gate_time_range_s(self) -> tuple[float, float]:
        times = [p / self.speed_mps for p in self.positions_m]
        return (min(times), max(times))

    def mean_gate_time_s(self) -> float:
        return sum(p / self.speed_mps for p in self.positions_m) / self.n

    # ------------------------------------------------------------------ 이송 비용
    def pre_gate_route_delta_s(self, src: str, dst: str) -> float:
        """게이트 **진입 전** 재배정의 주행 차이 = (게이트→dst) − (게이트→src).

        트럭이 아직 게이트에 있으므로 목적지만 바뀐다. **음수일 수 있다** —
        더 가까운 블록으로 보내면 실제로 주행이 줄어든다(그게 이 축의 이득이다).
        """
        return self.gate_to_block_s(dst) - self.gate_to_block_s(src)

    def post_gate_route_s(self, src: str, dst: str) -> float:
        """게이트 **진입 후** 재배정의 추가 주행 = 원 목적지에서 새 목적지까지.

        전환 시점을 알 수 없으므로 **가장 불리한 경우(원 블록까지 간 뒤 이동)** 로 잡는다.
        상수 180초를 쓰던 구판의 자리를 대신하며 항상 0 이상이다.
        """
        return self.block_to_block_s(src, dst)

    # ------------------------------------------------------------------ 부분집합
    def subset(self, ids) -> "YardLayout":
        chosen = tuple(ids)
        missing = [b for b in chosen if b not in self.ids]
        if missing:
            raise KeyError(f"배치에 없는 블록: {missing}")
        return YardLayout(chosen, tuple(self.position_m(b) for b in chosen),
                          self.speed_mps)

    def renamed(self, mapping: dict[str, str]) -> "YardLayout":
        """블록 이름만 바꾼 같은 기하 (실험 하네스가 쓰는 이름에 맞출 때)."""
        return YardLayout(tuple(mapping.get(b, b) for b in self.ids),
                          self.positions_m, self.speed_mps)

    # ------------------------------------------------------------------ 감사
    def matrix_s(self) -> dict[str, dict[str, float]]:
        """블록 간 주행시간 전량 — 원장에 그대로 남길 수 있는 형태."""
        return {a: {b: round(self.block_to_block_s(a, b), 6) for b in self.ids}
                for a in self.ids}

    def as_dict(self) -> dict:
        return {"ids": list(self.ids),
                "positions_m": [round(p, 6) for p in self.positions_m],
                "speed_mps": self.speed_mps,
                "gate_to_block_s": {b: round(self.gate_to_block_s(b), 6)
                                    for b in self.ids},
                "gate_time_range_s": [round(v, 6) for v in self.gate_time_range_s()],
                "mean_gate_time_s": round(self.mean_gate_time_s(), 6),
                "anchor_mean_s": GATE_BLOCK_MEAN_S,
                "anchor_support_s": [GATE_BLOCK_MIN_S, GATE_BLOCK_MAX_S],
                "synthetic": "단일 야드축 등간격 합성 배치 — 실제 터미널 좌표 아님"}


def terminal_layout(n: int = TERMINAL_BLOCKS, *, pitch_m: float = BLOCK_PITCH_M,
                    speed_mps: float = TRUCK_SPEED_MPS) -> YardLayout:
    """등간격 배치 — 게이트 오프셋은 **전 블록 평균 = 앵커**가 되도록 결정된다(자유변수 아님)."""
    ids = block_ids(n)
    offset = GATE_BLOCK_MEAN_S * speed_mps - pitch_m * (n - 1) / 2.0
    if offset <= 0:
        raise ValueError(f"블록 수 {n}·간격 {pitch_m}m 가 게이트 앵커와 양립 불가")
    layout = YardLayout(ids, tuple(offset + pitch_m * i for i in range(n)), speed_mps)
    if abs(layout.mean_gate_time_s() - GATE_BLOCK_MEAN_S) > 1e-6:
        raise AssertionError("앵커 평균 보존 실패")     # 구조적으로 불가능해야 함
    return layout
