"""1분 epoch 토큰 — RUDDER 가 **보는 것** ([[YR-223]] 사전등록).

■ 왜 1분인가
  시장이 실제로 60초 격자로 열린다(`v3.stage.orders.EPOCH_S`). 같은 시각의 오더에
  임의 순서를 붙이면 모델이 **없는 순서**를 학습한다. epoch 을 통째로 한 토큰으로
  묶으면 그 가짜 순서가 사라진다.

  ⚠️ 대가: 기여도는 **epoch 몫**이지 개별 행동 몫이 아니다. 한 epoch 안에서 누가
  얼마나 기여했는지는 이 모델이 답하지 못한다 — 그건 반사실이 답한다(분업).

■ 12칸으로 시작한다 — **핵심 정보 우선** (AGENTS.md 설계원칙 2)
  한꺼번에 많이 넣으면 어디서 막혔는지 못 가린다. 시계 3 · 상태 2 · **행동 4** ·
  비용증분 3 으로 시작하고, 관문을 통과한 뒤에만 늘린다.

      시계    w_frac tod_sin tod_cos       창 안 위치 · 하루 시각(원형)
      상태    n_dec  n_inside             이 epoch 결정 건수 · 야드 재적
      ★행동   n_sell n_buy n_space n_time   제안 · 동의 · 확정(공간/시간)
      비용    d_wait d_move d_rehandle d_vessel   ★Φ 네 항의 **원화 증분**

  비용 칸을 초·횟수가 아니라 **원화**로 두는 이유: 맞혀야 하는 값 `Y = ΔΦ` 도
  원화라 단위가 같아진다. 창 전체의 증분 합이 곧 창 Φ 증가분이라 모델이
  "지금까지 얼마 썼나" 를 바로 읽는다.

■ ★행동 칸이 따로 있는 이유 — 관문 B
  이 넷을 0 으로 지우고 다시 학습해 **예측이 나빠지지 않으면**, 모델은 행동이
  아니라 "몇 시냐"만 외운 것이다. 그러면 기여도는 시각대 평균일 뿐 신용이 아니다.

■ 비용 증분은 **과거만** 본다
  `d_*` 는 epoch [t−60, t) 에 이미 일어난 일이다. 미래를 넣으면 라벨이 아니라
  정답지가 된다(causal mask 와 같은 이유).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..v3.features.block import inside_count
from ..v3.reward.phi import terminal_cost_krw
from ..v3.stage.episode import rehandles_of, vessel_idle_of, yc_empty_travel_s

#: 토큰 칸 이름. **순서가 곧 열 번호**다 — 바꾸면 저장된 모델과 안 맞는다.
FEATURES: tuple[str, ...] = (
    "w_frac", "tod_sin", "tod_cos",
    "n_dec", "n_inside",
    "n_sell", "n_buy", "n_space", "n_time",
    "d_wait", "d_move", "d_rehandle", "d_vessel",
)
#: 관문 B(행동 제거)가 0 으로 지우는 칸.
ACTION_FEATURES: tuple[str, ...] = ("n_sell", "n_buy", "n_space", "n_time")
ACTION_IDX: tuple[int, ...] = tuple(FEATURES.index(f) for f in ACTION_FEATURES)
DIM = len(FEATURES)
DAY_S = 86_400.0


@dataclass
class Token:
    """epoch 하나. `t` 는 진단용이고 학습에는 `vec` 만 들어간다."""

    t: float
    vec: list[float]

    def as_dict(self) -> dict:
        return {"t": self.t, **dict(zip(FEATURES, self.vec))}


@dataclass
class EpochTape:
    """창을 굴리는 동안 epoch 마다 한 번 `snap()` 을 부르면 토큰이 쌓인다.

    직전 epoch 과의 **차이**를 쓰므로 첫 호출은 기준점만 잡고 토큰을 안 남긴다.
    """

    t0: float
    horizon_s: float
    block_vessel: dict = field(default_factory=dict)
    tokens: list[Token] = field(default_factory=list)
    #: 직전 스냅 값 (누적량)
    _prev: dict = field(default_factory=dict, repr=False)
    _prev_dec: int = field(default=0, repr=False)
    _prev_sell: int = field(default=0, repr=False)
    _prev_buy: int = field(default=0, repr=False)

    # ------------------------------------------------------------------ 누적량
    def _cum(self, mbt, records, t: float) -> dict:
        """지금까지의 누적 비용 재료. `terminal_cost_krw` 와 **같은 재료**를 쓴다."""
        phi = terminal_cost_krw(
            records, end_s=t,
            vessel_idle=vessel_idle_of(mbt, t, self.block_vessel),
            yc_extra_move_s=yc_empty_travel_s(mbt),
            rehandles=rehandles_of(mbt))
        return {"wait": float(phi.wait), "move": float(phi.move),
                "reh": float(phi.rehandle), "vessel": float(phi.vessel),
                "phi": float(phi.total)}

    # ------------------------------------------------------------------ 한 epoch
    def snap(self, mbt, t: float, *, market, records, bridge) -> None:
        cum = self._cum(mbt, records, t)
        n_dec_all = len(market.seller.trail)
        n_sell_all = sum(1 for e in market.seller.trail if e["action"] != "KEEP")
        n_buy_all = sum(1 for e in market.buyer.trail if e["action"] == "BUY")
        n_conf = bridge.n_space + bridge.n_time

        if not self._prev:                      # 첫 호출 = 기준점만
            self._prev = dict(cum, space=bridge.n_space, time=bridge.n_time,
                              conf=n_conf)
            self._prev_dec, self._prev_sell, self._prev_buy = (
                n_dec_all, n_sell_all, n_buy_all)
            return

        p = self._prev
        inside = sum(inside_count(mbt, b, t, records) for b in mbt.blocks)
        vec = [
            (t - self.t0) / max(1.0, self.horizon_s),
            math.sin(2 * math.pi * (t % DAY_S) / DAY_S),
            math.cos(2 * math.pi * (t % DAY_S) / DAY_S),
            float(n_dec_all - self._prev_dec),
            float(inside),
            float(n_sell_all - self._prev_sell),
            float(n_buy_all - self._prev_buy),
            float(bridge.n_space - p["space"]),
            float(bridge.n_time - p["time"]),
            cum["wait"] - p["wait"],
            cum["move"] - p["move"],
            cum["reh"] - p["reh"],
            cum["vessel"] - p["vessel"],
        ]
        self.tokens.append(Token(t=float(t), vec=vec))
        self._prev = dict(cum, space=bridge.n_space, time=bridge.n_time,
                          conf=n_conf)
        self._prev_dec, self._prev_sell, self._prev_buy = (
            n_dec_all, n_sell_all, n_buy_all)

    def matrix(self) -> list[list[float]]:
        return [tk.vec for tk in self.tokens]
