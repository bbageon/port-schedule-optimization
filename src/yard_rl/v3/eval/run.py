"""판정 진입점 — **학생만 돈다. 교사는 아예 안 불린다.**

설계 정본: `.claude/docs/architecture/06-학습과-판정.md` §2 · `06b-대조군.md`

■ 판정 축이 셋이다
  · **부하** 3,500 · 5,000 · 7,500 — 셋 다 실측이라 **전부 통과해야** 한다
  · **배차** 규칙 `SF-SPT` · 계획법 `JointRolloutGreedy` · 얼린 망(이력)
  · **재배치 팔** No Realloc · FCFS · SPT · EDD · Nearest · NetGain · **RL**

■ 완전교차하지 않는다 (06b §3)
  부하 3 × 배차 3 × 재배치 7 × 16일 = 1,008 런. **부분요인 528 런**으로 나눈다 —
  주판정은 배차 1종(계획법) 위에서 재배치 7팔, 일반성은 재배치 2팔 × 배차 3종.
  두 설계가 겹치는 96 셀(2팔 × 계획법)은 **한 번만** 돈다.

■ 오라클은 팔이 아니다
  정보·계산 반칙으로 만든 **진단용 상한**이라 판정표에 넣지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..reward.counterfactual import reset_rollout_calls, rollout_calls
from .guards import GuardReport, check_cell

#: 부하 축 — 계약(`load_levels_target = 3`)
LOAD_LEVELS = (3_500, 5_000, 7_500)

#: 배차 축
DISPATCHERS = ("SF_SPT", "ROLLOUT_GREEDY", "FROZEN_NET")

#: 재배치 팔 — 고전 배차 규칙을 재배치 축으로 번역한 것 + 학습 팔
REALLOC_ARMS = ("NO_REALLOC", "FCFS", "SPT", "LEAST_SLACK", "NEAREST",
                "NETGAIN", "RL")

#: 주판정에 쓰는 배차 (계획법) · 일반성 확인에 쓰는 재배치 양 끝
MAIN_DISPATCHER = "ROLLOUT_GREEDY"
GENERALITY_ARMS = ("NO_REALLOC", "RL")


@dataclass
class CellResult:
    load: int
    dispatcher: str
    arm: str
    day: int
    phi_krw: float
    admitted: int
    traded_edges: int
    guards: GuardReport | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"load": self.load, "dispatcher": self.dispatcher, "arm": self.arm,
             "day": self.day, "phi_krw": self.phi_krw,
             "admitted": self.admitted, "traded_edges": self.traded_edges}
        d.update(self.extra)
        if self.guards is not None:
            d.update(self.guards.as_dict())
        return d


def plan_cells(*, days: int = 16) -> list[tuple[int, str, str, int]]:
    """부분요인 설계 — 완전교차(1,008)가 아니라 **528 런**.

    | 무엇을 재나 | 설계 | 런 |
    |---|---|---|
    | 주판정 — 재배치 팔 서열 | 재배치 7팔 × 배차 1종(계획법) × 부하 3 × 16일 | 336 |
    | 일반성 — "배차 무관" | 재배치 2팔 × 배차 3 × 부하 3 × 16일 | 288 |
    | 겹침 (두 설계가 같은 셀을 센다) | 2팔 × 계획법 × 부하 3 × 16일 | −96 |

    겹치는 96 셀은 주판정에 이미 있으므로 **다시 돌리지 않는다.** 문서의 "624" 는
    이 겹침을 두 번 센 수였다(2026-08-22 정정).
    """
    cells: list[tuple[int, str, str, int]] = []
    for load in LOAD_LEVELS:
        for day in range(days):
            for arm in REALLOC_ARMS:                       # 주판정
                cells.append((load, MAIN_DISPATCHER, arm, day))
            for disp in DISPATCHERS:                       # 일반성
                if disp == MAIN_DISPATCHER:
                    continue                               # 주판정에 이미 있다
                for arm in GENERALITY_ARMS:
                    cells.append((load, disp, arm, day))
    return cells


def run_cell(episode_fn, *, load: int, dispatcher: str, arm: str, day: int,
             seed: int, code_dirty, expected_admitted: int | None = None
             ) -> CellResult:
    """한 셀을 굴리고 하드가드까지 붙여 돌려준다.

    `episode_fn` 은 무대를 하루치 굴리고 요약 dict 을 주는 함수다. 여기서는
    **팔을 갈아끼우는 일과 가드 판정만** 한다.
    """
    # ★교사 누출 금지 — 판정 구간에서 rollout 이 한 번이라도 불리면 실격이다.
    reset_rollout_calls()
    ep = episode_fn(load=load, dispatcher=dispatcher, arm=arm, seed=seed)
    calls = rollout_calls()

    admitted = int(ep.get("admitted", 0))
    traded = int(ep.get("traded_edges", 0))
    guards = check_cell(
        admitted=admitted,
        expected_admitted=int(expected_admitted or load),
        policy_exceptions=int(ep.get("policy_exceptions", 0)),
        code_dirty=code_dirty,
        rollout_calls=calls,
        traded_edges=traded if arm != "NO_REALLOC" else 1,   # 안 팔기는 거래 0 이 정상
    )
    return CellResult(load=load, dispatcher=dispatcher, arm=arm, day=day,
                      phi_krw=float(ep.get("phi_krw", 0.0)),
                      admitted=admitted, traded_edges=traded, guards=guards,
                      extra={k: v for k, v in ep.items()
                             if k.startswith(("c_", "n_", "mean_", "over_"))})


def paired_gap(cells: list[CellResult], *, load: int, dispatcher: str,
               arm: str, baseline: str) -> list[float]:
    """같은 날·같은 부하·같은 배차끼리 짝지어 `arm − baseline` 을 낸다."""
    by_day: dict[int, dict[str, float]] = {}
    for c in cells:
        if c.load != load or c.dispatcher != dispatcher:
            continue
        by_day.setdefault(c.day, {})[c.arm] = c.phi_krw
    out = []
    for _, m in sorted(by_day.items()):
        if arm in m and baseline in m:
            out.append(m[arm] - m[baseline])
    return out
