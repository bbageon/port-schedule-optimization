"""부동소수점 **결합 순서·반올림 횟수를 v5(파이썬)와 똑같이** 지키는 도우미 ([[YR-327]] 조각 1).

파이썬은 `a*b + c` 를 **두 번** 반올림한다 — 곱을 먼저 반올림하고, 그 결과를 더하며 다시
반올림한다. XLA 는 같은 식을 곱셈-덧셈 융합(FMA, fused multiply-add)으로 **한 번만**
반올림할 수 있고, 그러면 마지막 비트가 갈린다. 동등성(v5 와 같은 답)은 마지막 비트에
달려 있으므로(동률 판정·누적 적분) 이 차이는 허용되지 않는다.

■ ★실측 (2026-09-25 · jax/jaxlib 0.11.2 · CPU x64) — "XLA_FLAGS 가 FMA 를 막는다" 는 틀렸다
    플래그 8종 (`--xla_allow_excess_precision=false`, `--xla_cpu_enable_fast_math=false`,
    `--xla_cpu_use_thunk_runtime=false`, `honor_*` …) 전부 → jit 한 `x*y+z` 는 FMA 값 (40/40)
    `--xla_backend_optimization_level=0` 만 0/40 이었으나 융합 커널 안(lax.scan 본문)에서는
    그것도 못 막았다 (4/4 FMA). **유일하게 유효한 방어는 `lax.optimization_barrier`** —
    모든 플래그 조합에서 0/40, 컴파일된 HLO 에 barrier 가 남아 곱이 먼저 실체화된다.
    (검출 탐침: a = b = 1 + 2^-27, c = −1 → 두 번 반올림 1.4901161193847656e-08,
     FMA 1.4901161249358807e-08. `tests/v6/test_gpu_core.py` 가 상시 시험으로 둔다.)
    GPU(WSL CUDA 1종)는 이 식을 융합하지 않았지만 장치·XLA 판에 따라 다를 수 있으므로
    규약은 백엔드와 무관하게 적용한다.

■ 규약 — 조각 1 이후 새로 쓰는 `a*b + c` 꼴은 **전부** 여기 함수를 거친다
    mul_exact(a, b)          곱을 장벽 뒤에 실체화 → 뒤따르는 덧셈과 융합되지 않는다
    sum_seq(xs)              파이썬 `sum()`/`+=` 와 같은 **왼쪽부터 차례로** 더하기 (정적 길이)
    div_const(a, c)          파이썬 상수로 나누기 — 역수 곱으로 바뀌지 않게 분모를 장벽 뒤에
    (vmap 아래 나눗셈은 travel.div_exact — broadcast 분모의 역수 곱 변환을 막는다)

  `jnp.sum` 은 축소 순서가 규정돼 있지 않다(CPU 는 지금 순차지만 GPU 는 트리). v5 가 `+=`
  로 하나씩 더하는 누적은 `sum_seq` (정적 길이) 또는 `lax.scan` (동적 마스크) 으로 적는다.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax

__all__ = ["mul_exact", "sum_seq", "div_const", "FMA_PROBE"]

#: FMA 검출용 상수 — (a, b, c, 두 번 반올림 값, 한 번 반올림(FMA) 값)
FMA_PROBE = (1.0 + 2.0 ** -27, 1.0 + 2.0 ** -27, -1.0,
             1.4901161193847656e-08, 1.4901161249358807e-08)


def mul_exact(a, b):
    """`a * b` 를 **한 번 반올림해 실체화**한다 — 뒤따르는 `+ c` 와 FMA 로 묶이지 않는다.

    파이썬의 `a * b` 와 비트 동일. 배열끼리·스칼라 어느 조합이든 된다 (elementwise).
    """
    return lax.optimization_barrier(a * b)


def sum_seq(xs, start=None):
    """`start + xs[0] + xs[1] + …` 를 **왼쪽부터 차례로** — 파이썬 `sum(xs, start)` 와 같은 순서.

    `xs` 는 파이썬 시퀀스(정적 길이)이거나 1차원 배열(모양이 정적)이다. `start` 가 None 이면
    첫 항에서 시작한다 (파이썬 `sum` 의 `0 + x0 == x0` 과 같다). 빈 시퀀스면 0.0.
    """
    if not isinstance(xs, (list, tuple)):
        n = int(xs.shape[0])
        xs = [xs[i] for i in range(n)]
    if not xs:
        return jnp.asarray(0.0) if start is None else start
    acc = xs[0] if start is None else start + xs[0]
    for x in xs[1:]:
        acc = acc + x
    return acc


def div_const(a, c: float, dtype=jnp.float64):
    """`a / c` (c 는 파이썬 상수) — XLA 가 `a * (1/c)` 로 바꾸지 못하게 분모를 장벽 뒤에 둔다."""
    d = lax.optimization_barrier(jnp.asarray(float(c), dtype))
    return a / d
