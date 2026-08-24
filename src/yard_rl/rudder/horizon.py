"""지평 스윕 — **창을 얼마나 길게 보느냐에 따라 부호가 뒤집히는가** ([[YR-225]]).

■ 왜 이걸 재나 ([[YR-223]] 이 우연히 드러낸 것)
  창 128개에서 `Y = Φ(정책) − Φ(안팔기)` 가 **77% 음수**였다 — 3시간 창 안에서는
  재배치가 이득이다. 그런데 [[YR-220]] 의 **하루 전체** 판정은 반대다:
  RL 이 안팔기보다 비싼 회차가 **15/24**, 평균 **+267만원**.

      3시간 창    재배치가 이득  (중앙 −122,233원)
      하루 전체   재배치가 손해  (평균 +2,671,656원)

  **같은 정책인데 방향이 반대다.** 이게 사실이면 지금 라벨(H=3시간)은 정책에게
  *"팔수록 좋다"* 를 가르치는데 하루로 보면 그게 틀린 것이 된다 — 근시안이다.

■ ⚠️ 두 비교는 **완전히 같지 않다** (그래서 이 측정이 필요하다)
      창 비교      t0 까지는 정책이 굴린 상태에서 출발 · t0 이후만 안 판다
      하루 비교    처음부터 한 번도 안 판다
  즉 창의 "안팔기" 는 이미 정책이 만든 세상을 물려받는다. 두 수치의 차이가
  **지평 탓인지 출발점 탓인지** 구분하려면 같은 출발점에서 지평만 늘려 봐야 한다.

■ 측정 방법 — 출발점을 고정하고 **지평만** 늘린다
      t0 에서 스냅샷 하나를 뜬다 (하루를 t0 까지 굴린다 · 169초)
      그 하나에서 H = 3h · 6h · 12h · 하루끝 으로 짝비교를 굴린다
      → 부호가 뒤집히는 H 가 있으면 **지평이 원인**이다
      → 끝까지 음수면 원인은 지평이 아니라 **출발점**이다 (다른 축)

  스냅샷을 재사용하므로 하루 굴리기 값을 H 개수로 나눠 낸다.
"""
from __future__ import annotations

import time

from ..v3.world.integrated.terminal_stream import OBS_24H
from .runner import run_branch, snapshot_at

#: 사전등록 지평 — 3시간(지금 라벨) · 2배 · 4배 · 하루 끝까지.
HORIZONS_S = (10_800.0, 21_600.0, 43_200.0, None)   # None = 하루 끝까지


def sweep_at(*, load: int, seed: int, t0: float, seller_net=None, buyer_net=None,
             horizons=HORIZONS_S, obs=None, on_row=None) -> list:
    """t0 에서 스냅샷 하나를 떠서 **지평만 바꿔** 짝비교를 굴린다."""
    obs = obs or OBS_24H
    box = snapshot_at(load=load, seed=seed, t0=t0, seller_net=seller_net,
                      buyer_net=buyer_net, obs=obs)
    ctx = box["ctx"]
    common = dict(mbt=box["mbt"], orders=box["orders"], records=box["records"],
                  decided=box["decided"], t0=box["t0"], record=False)
    out = []
    for h in horizons:
        hs = float(obs.observe_s - box["t0"]) if h is None else float(h)
        if hs <= 0:
            continue
        t = time.time()
        pol = run_branch(ctx, horizon_s=hs, freeze=False, **common)
        keep = run_branch(ctx, horizon_s=hs, freeze=True, **common)
        row = {"load": load, "seed": seed, "t0": box["t0"], "horizon_s": hs,
               "to_end": h is None, "phi_policy": pol["phi"],
               "phi_keep": keep["phi"], "y_krw": pol["phi"] - keep["phi"],
               "traded": pol["traded"], "secs": time.time() - t}
        out.append(row)
        if on_row is not None:
            on_row(row)
    return out
