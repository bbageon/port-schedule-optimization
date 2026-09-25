"""교사 라벨 수집 — 행위자별 이진 반사실을 모은다.

설계 정본: `.claude/docs/architecture/06-학습과-판정.md` §1-1 · `04b` §3

■ 라벨에는 유효기간이 있다
  상대가 학습하면 내 라벨이 낡는다. v2 의 150,000 FIFO 버퍼는 라벨이 **직접
  관측**이라 안 상했기에 가능했다 — 반사실 라벨은 **회차마다 다시 만든다.**
  그래서 버퍼가 아니라 **회차 단위 수집기**다.

■ ★목표는 **그 결정의 기준선 대비 차이**다 ([[YR-220]])
  누적 Φ 를 그대로 목표로 주면 두 세계가 공유하는 과거가 목표의 99.9% 를 차지해
  행동 효과가 망 잔차에 묻힌다([[YR-218]] 실패: 잔차가 신호의 43~94배).
  그 결정에서 굴린 세계들의 **평균을 빼고** 가르친다 — argmin 은 결정 안에서만
  비교하므로 상수를 빼도 **선택이 안 바뀐다.**

■ 두 학생의 라벨을 더하지 않는다
  각자 "내가 행동을 바꿨다면" 을 재므로 겹치는 몫이 양쪽에 다 들어간다.
  Φ 에는 일어난 일만 한 번 기록한다(합치면 이중 계상).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..actors.nets import to_advantage


@dataclass
class Sample:
    """학생 하나가 배울 한 줄 — (입력 행, 목표 Φ)."""

    row: list[float]
    target_scaled: float
    doc_key: str
    action: str


@dataclass
class LabelSet:
    """한 회차에서 모은 라벨. **회차가 끝나면 버린다**(유효기간)."""

    seller: list[Sample] = field(default_factory=list)
    buyer: list[Sample] = field(default_factory=list)
    worlds: int = 0

    def __len__(self) -> int:
        return len(self.seller) + len(self.buyer)

    def tensors(self, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seller if which == "SELLER" else self.buyer
        if not s:
            return (torch.empty(0), torch.empty(0))
        x = torch.tensor([v.row for v in s], dtype=torch.float32)
        y = torch.tensor([v.target_scaled for v in s], dtype=torch.float32)
        return x, y


class LabelCollector:
    """행위자 trail 과 교사 결과를 붙여 학습 표본을 만든다.

    Seller 의 trail 에는 그 결정에서 채점한 **모든 행**이 들어 있다. 그중
    **실제로 고른 행**과 **교사가 굴린 대안 행** 두 개에만 라벨이 붙는다 —
    나머지 행은 굴리지 않았으므로 목표가 없다(굴리면 rollout 이 그만큼 는다).
    """

    def __init__(self) -> None:
        self.out = LabelSet()

    def add_seller(self, trail_entry: dict, *, phi_factual: float,
                   phi_alt: float, alt_index: int | None = None) -> None:
        rows = trail_entry["rows"]
        picked = int(trail_entry["picked"])
        base = 0.5 * (float(phi_factual) + float(phi_alt))   # ★그 결정의 기준선
        self.out.seller.append(Sample(
            row=[float(v) for v in rows[picked]],
            target_scaled=to_advantage(phi_factual, base),
            doc_key=trail_entry["doc_key"], action=trail_entry["action"]))
        if alt_index is not None and 0 <= alt_index < len(rows) and alt_index != picked:
            self.out.seller.append(Sample(
                row=[float(v) for v in rows[alt_index]],
                target_scaled=to_advantage(phi_alt, base),
                doc_key=trail_entry["doc_key"], action="ALT"))

    def add_buyer(self, trail_entry: dict, *, phi_factual: float,
                  phi_alt: float) -> None:
        chose_buy = trail_entry["action"] == "BUY"
        fact_row = trail_entry["row_buy"] if chose_buy else trail_entry["row_reject"]
        alt_row = trail_entry["row_reject"] if chose_buy else trail_entry["row_buy"]
        base = 0.5 * (float(phi_factual) + float(phi_alt))   # ★판매와 같은 규약
        self.out.buyer.append(Sample(
            row=list(fact_row), target_scaled=to_advantage(phi_factual, base),
            doc_key=trail_entry["doc_key"], action=trail_entry["action"]))
        self.out.buyer.append(Sample(
            row=list(alt_row), target_scaled=to_advantage(phi_alt, base),
            doc_key=trail_entry["doc_key"], action="ALT"))

    def note_worlds(self, n: int) -> None:
        self.out.worlds += int(n)

    def result(self) -> LabelSet:
        return self.out
