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
#:
#: ★실측으로 동결했다 ([[YR-308]] · 2026-09-10 · 179건 · 창 3시간):
#:
#:      부하    라벨   격차 중앙    격차 평균      격차 최대    차이0 비율
#:     3,500     64      6,721원     27,606원     162,286원      18.8%
#:     7,500     59      8,589원     71,730원     649,111원      20.3%
#:    15,000     56     15,472원    121,393원   1,085,555원      14.3%
#:                    ─────────
#:                전체 중앙 9,470원  →  **10,000원** 으로 못박는다
#:
#:   부하별 목표 중앙이 0.67 · 0.86 · 1.55 로 전부 적정대(0.3~3.0) 안이다.
#:
#: ■ 왜 중앙값인가 — 분포가 **한쪽으로 길다**
#:   평균이 중앙의 4~8배다(6,721 vs 27,606). 대부분의 크레인 선택은 3시간 뒤 비용을
#:   거의 안 바꾸고(차이가 정확히 0 인 결정이 **약 18%**), 드물게 백만원짜리가 있다.
#:   평균에 맞추면 그 꼬리가 눈금을 끌어올려 **평범한 결정이 전부 0 근처로 뭉갠다.**
#:   꼬리는 Huber(β=1)가 선형 구간에서 받아 준다 — 눈금이 아니라 손실함수의 몫이다.
#:
#: ■ ⚠️ 잠정값 100,000 이었을 때 무슨 일이 났나
#:   재배정층에서 빌려온 값이라 목표 중앙이 **0.095** 였다 — 신호가 0 근처로 뭉개져
#:   학습이 사실상 안 됐을 것이다. [[YR-299]] B 가 이 계측이 없어 눈금 문제를
#:   검증조차 못 했던 그 실패를 여기서는 계측이 막았다.
CRANE_ADV_SCALE = 10_000.0

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
        #: ★강제 손잡이 — `{크레인: job_id}`. 반사실 세계에서 *"대신 저걸 했다면"* 을
        #:  만드는 유일한 방법이다. 판매·구매의 `force_once` 와 같은 규약이되,
        #:  **비우는 책임은 부르는 쪽**에 있다 (`crane/rollout.py` 가 결정 하나를
        #:  손으로 돌린 직후 비운다) — 시계로 맞추면 분기 시각이 결정 시각과
        #:  1e-6 만 어긋나도 강제가 통째로 날아간다.
        self.force_once: dict[str, str] = {}

    def rank(self, sim, crane_id, gc) -> tuple:
        ref = gc.job_ref
        if ref is None:
            return _WAIT_KEY
        if self.force_once.get(crane_id) == ref.job_id:
            #: tier −1 — 어떤 후보보다도 앞. 단 resolver 가 `mandatory` 를 더 앞에
            #: 두므로 **필수 작업은 못 제친다**(엔진 계약). 강제가 실제로 먹었는지는
            #: 부르는 쪽이 결과를 읽어 확인한다.
            return (-1, 0.0, ref.job_id)
        x = crane_features(sim, crane_id, gc)
        with torch.no_grad():
            score = float(self.net(torch.tensor([x], dtype=torch.float32))[0])
        tier = 0 if (not self.serve_first or gc.kind == CandidateKind.SERVE) else 1
        #: `job_id` 를 끝에 두는 것은 **결정론**을 위해서다 — 점수가 같으면 항상 같은 배정.
        return (tier, score, ref.job_id)


def to_advantage(phi: float, base: float) -> float:
    """`(Φ − 그 결정의 기준선) / CRANE_ADV_SCALE` — 학습 목표.

    기준선은 그 결정에서 굴린 세계들의 평균이다. 양쪽에서 같은 값을 빼므로
    **argmin 순서는 안 바뀐다**(상수 차감). 재배정층 `nets.to_advantage` 와 같은 규약.
    """
    return (float(phi) - float(base)) / CRANE_ADV_SCALE


def from_advantage(x: float) -> float:
    """망 출력 → 원화 (기준선 대비). 보고·진단용."""
    return float(x) * CRANE_ADV_SCALE
