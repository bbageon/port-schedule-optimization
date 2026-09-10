"""크레인 교사 — 정답을 만든다 ([[YR-305]] · [[YR-248]] 2단계).

■ 무엇을 하나 — 쉬운 말로
  크레인이 *"A트럭을 먼저 할까, B트럭을 먼저 할까"* 를 고르는 순간마다,
  **두 세계를 굴려** 어느 쪽이 실제로 쌌는지 잰다. 그 차이가 정답이다.

      세계 A   실제로 고른 일감을 한다        → H시간 뒤 비용 Φ_A
      세계 B   대신 다른 일감을 했다면        → H시간 뒤 비용 Φ_B
      정답     (그 세계 비용 − 두 세계 평균) ÷ 눈금

  망이 자기 추정으로 자기를 가르치지 않는다(부트스트랩 없음) — **시뮬레이터가
  실제로 재서** 가르친다. 재배정층([[YR-205]])과 같은 방식이다.

■ 왜 "두 세계 평균" 을 빼나
  안 빼면 망이 *"이 상황은 원래 비싸다"* 를 배우느라 정작 **선택의 효과**를 못 배운다.
  양쪽에서 같은 값을 빼므로 **어느 쪽이 싼지는 안 바뀐다**(상수 차감).

■ ⚠️ 후보가 여럿인데 둘만 굴린다
  크레인 앞에 일감이 중앙 3개·최대 18개 놓인다. 그중 **고른 것과 대안 하나**만
  굴린다. 셋 이상 굴리면 계산이 그만큼 는다(결정 1건 = rollout 2회).

  그래서 배우는 것은 *"이 선택이 저 선택보다 나았나"* 이지 *"이게 최선이었나"* 가
  아니다. 재배정층도 같은 한계를 갖는다 — 논문 한계 절에 적을 것.

■ 대안을 어떻게 고르나 — **차점자**
  무작위로 고르면 뻔한 나쁜 후보와 비교해 배울 게 없다. 망이 **두 번째로 좋다고 본
  것**과 비교한다. 그래야 *"1등과 2등 중 진짜 1등은 누구였나"* 를 배운다.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

from .policy import CRANE_ADV_SCALE, CRANE_SCALE_FLOOR, to_advantage


@dataclass
class CraneSample:
    """학습 표본 하나 — 특징 8칸과 정답."""

    row: list[float]
    target: float
    crane: str
    job: str | None
    action: str          # "PICKED" | "ALT"
    #: ★원화 원본 ([[YR-309]]). 눈금이 **그날 라벨 전체를 봐야** 정해지므로,
    #:  `add` 때는 원본만 담고 `result` 에서 한꺼번에 목표로 바꾼다.
    phi: float = 0.0
    base: float = 0.0


@dataclass
class CraneLabelSet:
    samples: list[CraneSample] = field(default_factory=list)
    worlds: int = 0                      # 굴린 반사실 세계 수
    zero: int = 0                        # 차이가 0 이던 결정 수
    #: ★이 날에 실제로 쓴 눈금(원) — 보고·역환산에 이 값을 써야 한다
    scale: float = CRANE_ADV_SCALE
    median_gap_krw: float = 0.0          # 그날 라벨의 격차 중앙 (하한 적용 전)
    floor_bound: bool = False            # 하한이 걸렸나 (= 그날은 비용 중립에 가까웠다)

    def __len__(self) -> int:
        return len(self.samples)

    def tensors(self):
        """(특징, 정답) 텐서. 표본이 없으면 `None`."""
        import torch
        if not self.samples:
            return None
        x = torch.tensor([s.row for s in self.samples], dtype=torch.float32)
        y = torch.tensor([s.target for s in self.samples], dtype=torch.float32)
        return x, y


class CraneLabelCollector:
    """굴린 결과를 표본으로 바꾼다. 재배정층 `LabelCollector` 와 같은 규약."""

    def __init__(self) -> None:
        self.out = CraneLabelSet()

    def add(self, *, picked_row: list[float], alt_row: list[float],
            phi_factual: float, phi_alt: float,
            crane: str, job: str | None) -> None:
        """한 결정 → 표본 **둘**. 고른 것과 대안 각각."""
        base = 0.5 * (float(phi_factual) + float(phi_alt))    # ★그 결정의 기준선
        if abs(float(phi_factual) - float(phi_alt)) < 1e-6:
            self.out.zero += 1
        #: ★목표는 여기서 안 정한다 — 눈금이 그날 라벨 **전체**의 중앙 격차라
        #:  마지막 한 건까지 와야 계산된다. `result()` 가 한꺼번에 채운다.
        self.out.samples.append(CraneSample(
            row=[float(v) for v in picked_row], target=0.0,
            crane=crane, job=job, action="PICKED",
            phi=float(phi_factual), base=base))
        self.out.samples.append(CraneSample(
            row=[float(v) for v in alt_row], target=0.0,
            crane=crane, job=job, action="ALT",
            phi=float(phi_alt), base=base))

    def note_worlds(self, n: int) -> None:
        self.out.worlds += int(n)

    def result(self) -> CraneLabelSet:
        """★눈금을 **그날 라벨로 정하고** 목표를 채운다 ([[YR-309]]).

        전에는 전 부하 공통 상수(10,000원)를 썼다. 그러면 부하 12,500 인 날의
        목표가 3.99, 부하 3,500 인 날이 0.30 으로 **13배** 벌어져, 학습률이 고정인데
        날마다 미는 세기가 13배 달라진다 ([[YR-308]] 4시간 학습이 요동친 원인 가설).
        """
        ss = self.out.samples
        gaps = [abs(ss[i].phi - ss[i + 1].phi) for i in range(0, len(ss) - 1, 2)]
        self.out.median_gap_krw = st.median(gaps) if gaps else 0.0
        self.out.scale = max(CRANE_SCALE_FLOOR, self.out.median_gap_krw)
        self.out.floor_bound = self.out.median_gap_krw < CRANE_SCALE_FLOOR
        for smp in ss:
            smp.target = to_advantage(smp.phi, smp.base, self.out.scale)
        return self.out


def pick_alternative(scored: list[tuple[float, int]]) -> int | None:
    """대안 후보의 자리 — **차점자**를 고른다.

    `scored` 는 `(점수, 자리)` 목록이고 점수가 작을수록 좋다. 망이 1등으로 본 것을
    실제로 했으니, 2등과 견줘야 *"진짜 1등은 누구였나"* 를 배운다.

    후보가 하나뿐이면 비교할 대상이 없으므로 `None` — 그 결정은 라벨을 못 만든다.
    """
    if len(scored) < 2:
        return None
    ordered = sorted(scored, key=lambda p: (p[0], p[1]))
    return ordered[1][1]
