"""v6 배열 핵심이 GPU 에서 **같은 답을 내고 얼마나 빠른가** ([[YR-327]]).

두 가지를 잰다.
  ① **같은 답** — CPU 와 GPU 의 결과가 같아야 한다. 다르면 속도는 의미가 없다.
  ② **얼마나 빠른가** — 세계를 몇 개까지 동시에 밀 수 있나.

    PYTHONPATH=src python scripts/v6/bench_gpu.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from yard_rl.v6.gpu.policy import ORDER_FEATURES, init_params, q_values_batch


def make_batch(b: int, n: int, seed: int = 0):
    """세계 B 개 · 오더 N 개. 뒤 10% 는 빈 칸으로 둔다(실제와 같게)."""
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (b, n, ORDER_FEATURES), jnp.float32)
    keep = int(n * 0.9)
    mask = jnp.arange(n)[None, :] < keep
    return x, jnp.broadcast_to(mask, (b, n))


def timed(fn, *a, repeat: int = 20) -> float:
    """한 번 돌려 컴파일을 끝낸 뒤 `repeat` 회 평균(초)."""
    jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v6 배열 핵심 — 정확성·속도")
    ap.add_argument("--orders", type=int, default=64, help="결정 하나의 후보 수")
    ap.add_argument("--batches", default="1,16,128,1024,8192")
    ap.add_argument("--out", default="outputs/v6/bench_gpu.json")
    a = ap.parse_args(argv)

    dev = jax.devices()
    print(f"■ 장치: {dev}")
    gpu = [d for d in dev if d.platform == "gpu"]
    p = init_params(jax.random.PRNGKey(0))
    f = jax.jit(q_values_batch)

    # ── ① 같은 답인가 (GPU 가 있을 때만)
    same = None
    if gpu:
        x, m = make_batch(64, a.orders)
        on_gpu = jax.device_put(f(p, x, m)[0], jax.devices("cpu")[0])
        with jax.default_device(jax.devices("cpu")[0]):
            xc, mc = jax.device_put(x, jax.devices("cpu")[0]), jax.device_put(m, jax.devices("cpu")[0])
            on_cpu = q_values_batch(p, xc, mc)[0]
        gap = float(jnp.abs(on_gpu - on_cpu).max())
        scale = float(jnp.abs(on_cpu).max())
        #: ★판정 기준은 **값이 아니라 순서**다. 정책은 점수로 줄을 세워 고르므로,
        #:  점수가 소수점 아래에서 달라도 **같은 것을 고르면** 같은 정책이다.
        #:  (실측: 상대 오차 7.8e-4 인데 1등·상위 3등 순서가 64/64 일치)
        big = jnp.where(jax.device_put(m, jax.devices("cpu")[0]), 0.0, 1e30)
        rg = jnp.argsort(on_gpu + big, axis=1)
        rc = jnp.argsort(on_cpu + big, axis=1)
        top1 = float((rg[:, 0] == rc[:, 0]).mean())
        top3 = float((rg[:, :3] == rc[:, :3]).all(1).mean())
        same = top1 == 1.0
        print(f"  ① 같은 답: 값 최대차이 {gap:.2e} (상대 {gap/max(scale,1e-9):.1e}) · "
              f"**1등 일치 {top1:.0%} · 상위3 일치 {top3:.0%}** → "
              f"{'통과' if same else '★불일치'}")
    else:
        print("  ① 같은 답: GPU 가 없어 건너뜀")

    # ── ② 얼마나 빠른가
    print(f"\n■ 결정 하나에 후보 {a.orders}개 · 세계를 동시에 몇 개까지")
    print("   세계 수      1회 시간     초당 결정수      한 결정당")
    rows = []
    for b in (int(v) for v in a.batches.split(",")):
        x, m = make_batch(b, a.orders)
        s = timed(lambda X, M: f(p, X, M), x, m)
        per_s = b / s
        rows.append({"batch": b, "secs": s, "decisions_per_s": per_s})
        print(f"  {b:>8,}  {s*1e3:>9.3f}ms  {per_s:>12,.0f}  {s/b*1e6:>9.2f}µs")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"devices": [str(d) for d in dev], "orders": a.orders,
         "cpu_gpu_match": same, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n  → {out}")

    # v4 와의 대비 — 반사실 세계 하나가 3시간 시뮬(수 초)이었다
    best = max(rows, key=lambda r: r["decisions_per_s"])
    print(f"\n■ 참고: v4 는 반사실 세계 하나에 **수 초**가 들었다"
          f"(하루 128개). 여기 정책 평가는 초당 {best['decisions_per_s']:,.0f}건이다 — "
          f"단, **아직 세계를 굴리는 부분은 안 옮겼다**(정책망만).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
