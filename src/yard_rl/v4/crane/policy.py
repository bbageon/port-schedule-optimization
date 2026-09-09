"""크레인 정책 — 비용을 읽고 다음 일감을 고른다 ([[YR-248]] 2단계).

■ 지금까지 크레인은 규칙이었다
  `SF_SPT` 는 정렬키가 다섯 자리지만 실제 승부는 **소요시간 61.6% · 이름순 33.6%**
  로 갈린다([[YR-242]] 실측). 즉 **사실상 최단작업 우선이고 이름순이 3분의 1을 정한다.**
  비용을 한 줄도 안 읽는다.

■ 무엇이 바뀌나 — `rank()` 하나만
  `Preference.rank(sim, crane_id, gc) -> tuple` 이 규약이고, 작을수록 먼저다.
  규칙 바닥은 `(실작업 우선, 정렬키, job_id)` 를 돌려주는데, 여기서는
  **망이 매긴 상대비용**을 그 자리에 넣는다. 나머지 구조는 그대로다.

■ v3 재배정층과 같은 방식으로 배운다
  자기 추정으로 자기를 가르치지 않는다(부트스트랩 없음). 시뮬레이터가 **실제로 굴려 잰**
  비용 차이를 정답으로 쓴다. 그래서 이 망은 *"이 후보를 고르면 그 뒤 비용이 평균보다
  얼마나 낮은가"* 를 낸다 — 절대 비용이 아니라 **짝 중심 상대값**이다.

■ ⚠️ 학습 전에는 규칙보다 나쁘다
  초기 망은 무작위라 `rank` 가 잡음이다. 그래서 `RuleBase` 처럼 **실작업 우선**을
  맨 앞 자리에 그대로 둔다 — 학습이 덜 돼도 *"일은 안 하고 자리이동만 고르는"*
  퇴화([[YR-039]] · 결정의 54~81%가 REPOSITION 이었다)로 무너지지 않는다.
  망은 **그 안에서 순서만** 정한다. 이 울타리를 걷는 것은 효과가 실증된 뒤에 한다.
"""
from __future__ import annotations

import torch
from torch import nn

from ..world.contract.schema import CandidateKind
from ..world.integrated.resolver import BaselinePreference
from .features import CRANE_DIM, crane_features

#: 학습 목표 눈금 — 원화를 망이 다루기 좋은 크기로 바꾼다.
#: ⚠️ 재배정층 `ADV_SCALE` 과 **같은 이유·같은 규약**이다: 조건이 바뀌면 재실측 후 동결.
#:    아직 크레인 결정의 라벨 격차를 실측한 적이 없어 **잠정값**이다 — 첫 학습에서
#:    실제 격차를 재고([[YR-304]] 계측) 그 값으로 다시 못박아야 한다.
CRANE_ADV_SCALE = 100_000.0

#: WAIT 자리 — `BaselinePreference` 규약과 같다(항상 최하위).
_WAIT_KEY: tuple = (2, 0.0, "")


class CraneNet(nn.Module):
    """후보 하나 → 상대비용 하나. 재배정층 망과 같은 모양(2층 64 ReLU).

    같은 모양을 쓰는 이유는 **비교를 깨끗하게** 하기 위해서다 — 크레인 축의 효과를
    보려는 것이지 망 구조의 효과를 보려는 게 아니다.
    """

    def __init__(self, dim: int = CRANE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CranePolicy(BaselinePreference):
    """학습 망으로 일감을 고르는 크레인 정책.

    `dispatch.make_preference` 가 만드는 규칙 바닥들과 **같은 자리에 꽂힌다**.
    """

    USES_FUTURE_INFORMATION = False       # 공개 정보만 본다 (계약)
    name = "RL_CRANE"

    def __init__(self, net: CraneNet | None = None, *, serve_first: bool = True):
        self.net = net if net is not None else CraneNet()
        self.net.eval()
        #: ★실작업 우선 울타리 — 위 머리말의 [[YR-039]] 퇴화 방지.
        #:  끄면 망이 SERVE/PRE_REHANDLE/REPOSITION 을 완전히 자유롭게 고른다.
        #:  **효과가 실증되기 전에는 켜 둔다.**
        self.serve_first = bool(serve_first)
        #: 학습용 흔적 — 결정마다 (특징, 점수)를 남긴다. 교사가 라벨을 붙일 자리다.
        self.trail: list[dict] = []

    def rank(self, sim, crane_id, gc) -> tuple:
        ref = gc.job_ref
        if ref is None:
            return _WAIT_KEY
        x = crane_features(sim, crane_id, gc)
        with torch.no_grad():
            score = float(self.net(torch.tensor([x], dtype=torch.float32))[0])
        tier = 0 if (not self.serve_first or gc.kind == CandidateKind.SERVE) else 1
        #: `job_id` 를 끝에 두는 것은 **결정론**을 위해서다 — 점수가 같으면 항상 같은 배정.
        return (tier, score, ref.job_id)

    # ------------------------------------------------------------------ 학습 훅
    def record(self, sim, crane_id, gc, chosen: bool) -> None:
        """이 결정의 특징을 남긴다. 교사가 나중에 라벨을 붙인다."""
        self.trail.append({
            "t": sim.clock, "crane": crane_id,
            "job": getattr(gc.job_ref, "job_id", None),
            "kind": str(gc.kind), "x": crane_features(sim, crane_id, gc),
            "chosen": bool(chosen)})


def to_advantage(phi: float, base: float) -> float:
    """`(Φ − 그 결정의 기준선) / CRANE_ADV_SCALE` — 학습 목표.

    기준선은 그 결정에서 굴린 세계들의 평균이다. 양쪽에서 같은 값을 빼므로
    **argmin 순서는 안 바뀐다**(상수 차감). 재배정층 `nets.to_advantage` 와 같은 규약.
    """
    return (float(phi) - float(base)) / CRANE_ADV_SCALE


def from_advantage(x: float) -> float:
    """망 출력 → 원화 (기준선 대비). 보고·진단용."""
    return float(x) * CRANE_ADV_SCALE
