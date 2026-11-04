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

#: 부하 축 — 계약(`load_levels_target = 3`). **판정은 셋 다** 돈다.
#: ★6차 계약(2026-08-26) — 곡선이 평평해져 옛 7,500 의 혼잡이 사라졌다.
#: 12,500 이 옛 7,500 과 같은 강도(능력 115%)이고 15,000 이 그 위다.
#: 17,500 은 **하루 안에 줄이 안 빠져** 제외한다(적체 3,556대 잔류).
LOAD_LEVELS = (3_500, 5_000, 7_500, 12_500, 15_000)

#: ★학습에 쓰는 부하 (사용자 결정 2026-08-23 · [[YR-217]])
#:
#: 7,500 은 턴타임 상위 10% 가 **263.9분**이라 반사실 창(3시간=180분)으로 못 덮는다 —
#: 창 안에서 못 나간 트럭은 무슨 결정을 하든 창 안 비용이 같아 **라벨이 0** 이 된다.
#: 덮으려면 H ≥ 5시간이고 그건 회차당 5시간 넘는 계산이라 감당이 안 된다.
#:
#: 그래서 **학습은 두 수준, 판정은 세 수준**이다. 7,500 판정은 자동으로
#: **"배운 적 없는 조건에서도 되는가"** 라는 일반화 시험이 된다 — 통과하면 더 강한
#: 주장이지만, 실패했을 때 "정책이 나쁜가 / 안 배운 조건이라 그런가" 를 못 가른다.
#: 그 한계는 판정 기록에 **반드시 명시**한다.
#: ★학습도 다섯 부하 전부에서 돈다 (사용자 지시 2026-08-26) —
#: 혼잡을 학습에 넣어야 그 조건에서 배운 정책이 된다.
TRAIN_LOADS = LOAD_LEVELS
EVAL_LOADS = LOAD_LEVELS

#: 검토 창 스윕 축 (사용자 결정 2026-08-23 · [[YR-190]])
#: 4시간·24시간은 뺐다 — 필요한 H 가 각각 5시간·25시간이라 계산이 감당 안 된다.
WINDOW_SWEEP_S = (1_800.0, 3_600.0, 7_200.0)

#: 배차 축
DISPATCHERS = ("SF_SPT", "ROLLOUT_GREEDY", "FROZEN_NET")

#: 재배치 팔 — 고전 배차 규칙을 재배치 축으로 번역한 것 + 학습 팔.
#: ⚠️ **설계상의 목록**이다. 지금 무대에 꽂혀 실제로 도는 팔은
#: `stage.ARMS`(= `NO_REALLOC`·`RL`) 뿐이고 나머지는 [[YR-211]] 이다 —
#: 안 만든 팔은 조용히 대체되지 않고 `NotImplementedError` 로 즉시 실패한다.
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


def default_episode_fn(*, load: int, dispatcher: str, arm: str, seed: int, **kw):
    """무대에 실제로 꽂힌 v3 에피소드 ([[YR-215]]).

    지연 임포트다 — `eval` 이 `stage` 를 껴안으면 판정만 쓰려는 쪽까지 엔진을
    끌어오게 된다.
    """
    from ..stage.episode import run_episode
    return run_episode(load=load, dispatcher=dispatcher, arm=arm, seed=seed,
                       **kw).as_dict()


def run_cell(episode_fn=None, *, load: int, dispatcher: str, arm: str, day: int,
             seed: int, code_dirty, expected_admitted: int | None = None
             ) -> CellResult:
    """한 셀을 굴리고 하드가드까지 붙여 돌려준다.

    `episode_fn` 은 무대를 하루치 굴리고 요약 dict 을 주는 함수다. 여기서는
    **팔을 갈아끼우는 일과 가드 판정만** 한다.
    """
    # ★교사 누출 금지 — 판정 구간에서 rollout 이 한 번이라도 불리면 실격이다.
    reset_rollout_calls()
    ep = (episode_fn or default_episode_fn)(
        load=load, dispatcher=dispatcher, arm=arm, seed=seed)
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
