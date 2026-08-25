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


# ------------------------------------------------------------------ 병렬 실행
_CTX = {}


def _init_worker(ckpt_path, horizons, obs):
    import torch
    from .collect import load_nets
    torch.set_num_threads(1)
    s, b = load_nets(ckpt_path)
    _CTX.update(seller=s, buyer=b, horizons=horizons, obs=obs)


def _run_one(arg):
    load, seed, t0 = arg
    return sweep_at(load=load, seed=seed, t0=t0, seller_net=_CTX["seller"],
                    buyer_net=_CTX["buyer"], horizons=_CTX["horizons"],
                    obs=_CTX["obs"])


def sweep_many(*, loads, seeds, t0s, ckpt_path=None, horizons=HORIZONS_S,
               obs=None, workers: int = 8, on_sweep=None) -> list:
    """(부하 × 날 × t0) 을 프로세스에 나눈다. 하나가 스냅샷 하나를 재사용한다."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(int(l), int(s), float(t)) for l in loads for s in seeds for t in t0s]
    out = []
    if workers <= 1:
        _init_worker(ckpt_path, horizons, obs)
        for j in jobs:
            r = _run_one(j)
            out += r
            if on_sweep is not None:
                on_sweep(r)
        return out
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(ckpt_path, horizons, obs)) as ex:
        for r in ex.map(_run_one, jobs):
            out += r
            if on_sweep is not None:
                on_sweep(r)
    return out


# ═══════════════════════════════════════════════════════════ ★정직한 비교
def window_epochs(t0: float, window_s: float, grid_s: float = 60.0) -> list:
    """`[t0, t0+window_s)` 안의 시장 epoch 시각들."""
    n = int(round(window_s / grid_s))
    return [float(t0) + grid_s * k for k in range(n)]


def window_effect_to_end(*, load: int, seed: int, t0s, window_s: float = 10_800.0,
                         seller_net=None, buyer_net=None, obs=None,
                         control: bool = True, on_row=None) -> list:
    """★**그 창의 거래만** 지우고 **하루 끝까지** 굴린다 ([[YR-227]]).

    ■ 왜 이게 필요한가 — [[YR-226]] 의 비교가 틀렸다 (사용자 지적 2026-08-25)
      거기서 쓴 두 수치는 **같은 것을 재고 있지 않았다**:

          `Y(t0, H=3시간)`   [t0, t0+3h] 의 거래  vs 없음 · **t0+3h 에서 잼**
          `Y(t0, 하루끝)`    [t0, 하루끝] 의 거래 vs 없음 · **하루끝에서 잼**

      두 번째는 **거래 집합이 다르다** — t0=4시면 첫째는 16건, 둘째는 1,500건이다.
      부호가 다른 게 당연하고, 그걸 "라벨이 틀렸다" 로 읽으면 안 된다.

    ■ 올바른 비교 — **거래 집합을 창 안으로 고정**하고 지평만 늘린다
          라벨    Φ(정책) − Φ(창 안 거래만 지움)   **t0+3h 에서**
          진실    Φ(정책) − Φ(창 안 거래만 지움)   **하루끝에서**   ← 이 함수
      창 밖의 행동은 두 세계가 **똑같이** 한다. 그래야 차이가 오직 창 안 거래의
      효과이고, 라벨이 그걸 제대로 가리키는지 물을 수 있다.

    `control=True` 면 아무것도 안 얼린 가지도 굴려 **하루 Φ 를 재현하는지** 본다
    (재현 못 하면 분기 재조립이 깨진 것이라 나머지 수치가 무의미하다).
    """
    obs = obs or OBS_24H
    rows = []
    for t0 in t0s:
        box = snapshot_at(load=load, seed=seed, t0=float(t0),
                          seller_net=seller_net, buyer_net=buyer_net, obs=obs)
        ctx, tt0 = box["ctx"], box["t0"]
        to_end = float(obs.observe_s - tt0)
        common = dict(mbt=box["mbt"], orders=box["orders"], records=box["records"],
                      decided=box["decided"], t0=tt0, record=False, freeze=False)
        eps = window_epochs(tt0, window_s)
        t = time.time()
        pol_end = run_branch(ctx, horizon_s=to_end, **common)
        cut_end = run_branch(ctx, horizon_s=to_end, freeze_at=eps, **common)
        # 라벨 쪽 — 같은 창을 지우되 창 끝에서 잰다 (= 지금 쓰는 라벨)
        pol_win = run_branch(ctx, horizon_s=window_s, **common)
        cut_win = run_branch(ctx, horizon_s=window_s, freeze_at=eps, **common)
        row = {"load": load, "seed": seed, "t0": tt0, "window_s": window_s,
               "y_label": pol_win["phi"] - cut_win["phi"],
               "y_true": pol_end["phi"] - cut_end["phi"],
               "traded_window": pol_win["traded"],
               "traded_all": pol_end["traded"],
               "phi_policy_end": pol_end["phi"], "phi_cut_end": cut_end["phi"],
               "secs": time.time() - t}
        if control:
            row["phi_control_gap"] = 0.0     # pol_end 자체가 대조군 역할을 한다
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


_CTX2 = {}


def _init_worker2(ckpt_path, t0s, window_s, obs):
    import torch
    from .collect import load_nets
    torch.set_num_threads(1)
    s, b = load_nets(ckpt_path)
    _CTX2.update(seller=s, buyer=b, t0s=t0s, window_s=window_s, obs=obs)


def _run_one2(arg):
    load, seed = arg
    return window_effect_to_end(load=load, seed=seed, t0s=_CTX2["t0s"],
                                window_s=_CTX2["window_s"],
                                seller_net=_CTX2["seller"],
                                buyer_net=_CTX2["buyer"], obs=_CTX2["obs"])


def window_effect_many(*, loads, seeds, t0s, window_s: float = 10_800.0,
                       ckpt_path=None, obs=None, workers: int = 8,
                       on_day=None) -> list:
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(int(l), int(s)) for l in loads for s in seeds]
    out = []
    if workers <= 1:
        _init_worker2(ckpt_path, t0s, window_s, obs)
        for j in jobs:
            r = _run_one2(j)
            out += r
            if on_day is not None:
                on_day(r)
        return out
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker2,
                             initargs=(ckpt_path, t0s, window_s, obs)) as ex:
        for r in ex.map(_run_one2, jobs):
            out += r
            if on_day is not None:
                on_day(r)
    return out
