"""궤적 수집 — **하루 하나가 프로세스 하나** ([[YR-223]] 1단계).

■ 왜 이렇게 나누나
  실측(2026-08-24): **하루 굴리기 169초** · 창 짝 하나 약 54초(H=3시간).
  하루가 순차라 못 쪼갠다 — 그래서 하루를 **통째로** 작업자에게 준다. 창은 그
  작업자 안에서 순서대로 굴린다.

      작업자 하나 = 169초 + 창수 × 54초
      날 16개 · 창 13개 → 작업자당 871초 · 8코어면 두 묶음 ≈ 29분

■ ★스냅샷을 모아 두지 않는다
  창 13개를 다 복제해 들고 있으면 메모리가 터진다. 하루를 굴리다 예정 시각에
  닿으면 **그 자리에서** 창을 굴리고 스냅샷을 버린다(`runner.collect_day`).

■ ★검증은 날 단위로 가른다
  한 날의 창들은 서로 겹친다. 창 단위로 섞으면 held-out 이 무의미해진다
  (`train.split_by_seed`).
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from ..v3 import CF_HORIZON_S
from ..v3.actors import BuyerNet, SellerNet
from .runner import Window, collect_day

#: 사전등록 기본 — 남는 코어에 맞춘다([[YR-222]] 가 16개를 쓰는 중).
DEFAULT_WORKERS = 8

_CTX = {}


def load_nets(ckpt_path):
    """학습된 망을 읽는다. `None` 이면 초기 망(시드 고정)."""
    if ckpt_path is None:
        return None, None
    d = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    s, b = SellerNet(), BuyerNet()
    s.load_state_dict(d["seller"])
    b.load_state_dict(d["buyer"])
    s.eval()
    b.eval()
    return s, b


def _init_worker(ckpt_path, horizon_s, n_windows, explore):
    torch.set_num_threads(1)                 # 작업자끼리 스레드로 싸우지 않게
    s, b = load_nets(ckpt_path)
    _CTX.update(seller=s, buyer=b, horizon_s=horizon_s, n_windows=n_windows,
                explore=explore)


def _run_one(arg):
    load, seed = arg
    t0 = time.time()
    ws = collect_day(load=load, seed=seed, n_windows=_CTX["n_windows"],
                     horizon_s=_CTX["horizon_s"], seller_net=_CTX["seller"],
                     buyer_net=_CTX["buyer"], explore=_CTX["explore"])
    return {"load": load, "seed": seed, "secs": time.time() - t0,
            "windows": [w.__dict__ for w in ws]}


def collect(*, loads, seeds, n_windows: int, ckpt_path=None,
            horizon_s: float = CF_HORIZON_S, explore: float = 0.0,
            workers: int = DEFAULT_WORKERS, on_day=None) -> list:
    """(부하 × 날) 을 프로세스에 나눠 궤적을 모은다.

    `loads` 와 `seeds` 는 **짝지어** 돈다(zip 이 아니라 곱). 부하 2종 × 날 8 = 16 작업.
    """
    jobs = [(int(l), int(s)) for l in loads for s in seeds]
    out = []
    if workers <= 1:
        _init_worker(ckpt_path, horizon_s, n_windows, explore)
        for j in jobs:
            r = _run_one(j)
            out.append(r)
            if on_day is not None:
                on_day(r)
        return out
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(ckpt_path, horizon_s, n_windows,
                                       explore)) as ex:
        for r in ex.map(_run_one, jobs):
            out.append(r)
            if on_day is not None:
                on_day(r)
    return out


# --------------------------------------------------------------- 저장·불러오기
def save(days, path) -> pathlib.Path:
    """모은 궤적을 파일로. 다시 모으지 않고 학습만 다시 돌릴 수 있게."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(days, ensure_ascii=False), encoding="utf-8")
    return p


def load(path) -> list:
    """파일 → `Window` 목록(날 구분 없이 평평하게)."""
    days = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [Window(**w) for d in days for w in d["windows"]]


def summarize(windows) -> dict:
    """모은 것이 쓸 만한지 한눈에 — **Y 가 0 이면 배울 게 없다**."""
    ys = [float(w.y_krw) for w in windows]
    nz = [y for y in ys if abs(y) > 1.0]
    ys_s = sorted(abs(y) for y in ys)
    return {"n_windows": len(windows), "n_days": len({w.seed for w in windows}),
            "n_tokens": sum(len(w.tokens) for w in windows),
            "y_zero_ratio": 1.0 - len(nz) / max(1, len(ys)),
            "y_abs_median": ys_s[len(ys_s) // 2] if ys_s else 0.0,
            "y_abs_max": ys_s[-1] if ys_s else 0.0,
            "traded_median": sorted(w.traded for w in windows)[len(windows) // 2]
            if windows else 0}
