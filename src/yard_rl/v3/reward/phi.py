"""Φ — 터미널 하루 총비용, 원화 네 항.

설계 정본: `.claude/docs/architecture/04-비용과-보상.md`

    Φ = C_wait + C_move + C_rehandle + C_vessel        [원]

구 Φ 는 "비용시간(cost-hour)" 단위였고 블록별 합·주행·기사 외부 대기가 흩어져
있었다. 네 항·원화로 통일한다.

■ 이중 계상 금지 (04 §1-4)
  · 재조작 lift 시간은 **항3에만** — 항2의 "추가 이동" 에서 뺀다
  · 트럭 대기는 **턴타임 하나로만** — 블록 체류·순수 줄서기는 진단 열
  · 고시 그대로의 법정 재현값은 진단 열. Φ 에 더하지 않는다
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schema.lifecycle import censored_turn_time_s
from .krw import (rehandle_krw, truck_wait_krw, vessel_idle_krw, yc_move_krw)


@dataclass(frozen=True)
class PhiBreakdown:
    """네 항과 합계. 진단 열은 Φ 에 안 들어간다."""

    wait: float
    move: float
    rehandle: float
    vessel: float

    # ── 진단 열 (보고용 · Φ 에 더하지 않는다)
    n_trucks: int = 0
    n_censored: int = 0
    mean_turn_time_s: float = 0.0
    over_ratio: float = 0.0          # 턴타임이 1시간을 넘은 트럭의 비율

    @property
    def total(self) -> float:
        return self.wait + self.move + self.rehandle + self.vessel

    def as_dict(self) -> dict:
        return {
            "phi_krw": self.total,
            "c_wait": self.wait, "c_move": self.move,
            "c_rehandle": self.rehandle, "c_vessel": self.vessel,
            "n_trucks": self.n_trucks, "n_censored": self.n_censored,
            "mean_turn_time_s": self.mean_turn_time_s,
            "over_ratio": self.over_ratio,
        }


def terminal_cost_krw(records, *, end_s: float,
                      vessel_idle: dict[str, tuple[float, float]] | None = None
                      ) -> PhiBreakdown:
    """Φ 를 네 항으로 계산한다.

    `records`      : {doc_key: ExecutionRecord}
    `end_s`        : 평가 종료 시각 — 미완료 트럭을 여기서 **검열**한다
    `vessel_idle`  : {vessel_id: (GT, 선박 유휴 초)} — 창 안의 유휴만 넘긴다

    **미완료 검열**: 완료차는 `O−A`, 미완료차는 `T−A`. 그래야 에피소드 끝에 남은
    트럭이 비용을 피하지 못한다.
    """
    wait = move = rehab = 0.0
    n = censored = 0
    tt_sum = 0.0
    over = 0

    for rec in records.values():
        tt = censored_turn_time_s(rec, end_s)
        if tt is None:
            continue                       # 아직 게이트도 안 지났다 = 무대 밖
        n += 1
        tt_sum += tt
        if rec.gate_out_s is None:
            censored += 1
        if tt > 3600.0:
            over += 1
        wait += truck_wait_krw(tt)
        move += yc_move_krw(rec.yc_extra_move_s)
        rehab += rehandle_krw(rec.rehandles)

    vessel = 0.0
    for gt, idle_s in (vessel_idle or {}).values():
        vessel += vessel_idle_krw(gt, idle_s)

    return PhiBreakdown(
        wait=wait, move=move, rehandle=rehab, vessel=vessel,
        n_trucks=n, n_censored=censored,
        mean_turn_time_s=(tt_sum / n if n else 0.0),
        over_ratio=(over / n if n else 0.0),
    )
