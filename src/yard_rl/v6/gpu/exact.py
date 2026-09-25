"""부동소수점 **결합 순서·반올림 횟수를 v5(파이썬)와 똑같이** 지키는 도우미 ([[YR-327]] 조각 1·5).

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

■ ★실측 2 (2026-09-26 · 파이썬 3.12.13) — **파이썬 3.12 `sum()` 은 순차 `+=` 가 아니다**
    `sum([0.1]*10)` 은 1.0 (3.11 까지는 0.9999999999999999). CPython 이 `float` 항을 Neumaier
    보정합으로 더하기 때문이다 (bltinmodule.c `builtin_sum_impl`). v5 가 `sum(list)` 로 실수를
    더하는 자리를 순차 scan 으로 옮기면 **3항부터 마지막 비트가 갈린다** → `sum_python`.
    또 XLA 는 jit 한 `0.0 + w` 를 `w` 로 접는다 (w = −0.0 이면 −0.0 이 새어 나옴; 파이썬
    `0 + (−0.0)` 은 +0.0). 그래서 `sum_python` 의 첫 항 0 도 장벽 뒤에 둔다.

■ 규약 — 조각 1 이후 새로 쓰는 `a*b + c` 꼴은 **전부** 여기 함수를 거친다
    mul_exact(a, b)          곱을 장벽 뒤에 실체화 → 뒤따르는 덧셈과 융합되지 않는다
    sum_seq(xs)              파이썬 `acc += x` 루프와 같은 **왼쪽부터 차례로** 더하기 (정적 길이, 보정 없음)
    sum_python(xs, mask)     파이썬 3.12 `sum(실수 목록)` 과 **비트 동일** (Neumaier 보정합, 동적 마스크)
    div_const(a, c)          파이썬 상수로 나누기 — 역수 곱으로 바뀌지 않게 분모를 장벽 뒤에
    (vmap 아래 나눗셈은 travel.div_exact — broadcast 분모의 역수 곱 변환을 막는다)

  어느 것을 쓰나 — v5 코드가 어떻게 더했는지로 고른다:
    v5 `acc += x` / `a + b + c` 식                     → sum_seq (또는 마스크가 동적이면 lax.scan `+=`)
    v5 `sum(...)` 이고 항이 파이썬 `float`             → sum_python  ★3.12 보정합
    v5 `sum(...)` 이고 항이 `int` (개수 세기)          → 정수 합 — 무엇으로 더해도 정확
    v5 `sum(...)` 이고 항이 `np.float64` (float 하위형) → sum_seq — CPython 이 그 항부터 보정 없는 일반 덧셈으로 떨어진다
  `jnp.sum` 은 축소 순서가 규정돼 있지 않다(CPU 는 지금 순차지만 GPU 는 트리) — 실수 합에는 쓰지 않는다.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax

__all__ = ["mul_exact", "sum_seq", "sum_python", "div_const", "FMA_PROBE"]

#: FMA 검출용 상수 — (a, b, c, 두 번 반올림 값, 한 번 반올림(FMA) 값)
FMA_PROBE = (1.0 + 2.0 ** -27, 1.0 + 2.0 ** -27, -1.0,
             1.4901161193847656e-08, 1.4901161249358807e-08)


def mul_exact(a, b):
    """`a * b` 를 **한 번 반올림해 실체화**한다 — 뒤따르는 `+ c` 와 FMA 로 묶이지 않는다.

    파이썬의 `a * b` 와 비트 동일. 배열끼리·스칼라 어느 조합이든 된다 (elementwise).
    """
    return lax.optimization_barrier(a * b)


def sum_seq(xs, start=None):
    """`start + xs[0] + xs[1] + …` 를 **왼쪽부터 차례로** — 파이썬 `acc += x` 루프(보정 없음)와 같은 순서.

    ⚠️ 파이썬 3.12 의 `sum(실수 목록)` 과는 **다르다** — 그쪽은 Neumaier 보정합이라 3항부터 마지막 비트가
    갈릴 수 있다 (머리말 실측 2). v5 가 `sum(...)` 으로 `float` 를 더하는 자리는 `sum_python` 을 쓴다.
    이 함수가 맞는 자리: v5 가 `+=` 로 쌓는 누적, `a + b + c` 식, `sum()` 의 항이 `int` 나 `np.float64` 인 곳.

    `xs` 는 파이썬 시퀀스(정적 길이)이거나 1차원 배열(모양이 정적)이다. `start` 가 None 이면
    첫 항 **그 자체**에서 시작한다 (`acc = xs[0]`; 파이썬 `acc = 0.0; acc += xs[0]` 과는 xs[0] = −0.0
    일 때만 부호가 다르다). 빈 시퀀스면 0.0.
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


def sum_python(xs, mask=None):
    """파이썬 3.12 **`sum(실수 목록)` 과 비트 동일** — Neumaier 보정합 (CPython bltinmodule.c `builtin_sum_impl`).

    ★조각 5 가 찾은 함정: 파이썬 3.12 부터 `sum()` 은 `float` 항을 **보정합**으로 더한다. 순차 `+=`(`sum_seq`)
    로 옮기면 3항부터 마지막 비트가 갈릴 수 있다 (2항까지는 같다 — 보정 c 가 `t` 의 반올림 오차 그 자체라
    `t + c` 를 반올림하면 도로 `t` 다). v5 가 `sum(...)` 으로 실수를 더하는 자리(time_contract.py:141
    `censored_exposure_s`, lane.py:50 평균 혼잡, engine.py:835 크레인 부하 합, adapter.py 특징 평균 …)는
    이 함수로 옮긴다. 배열 순서 = v5 가 더한 순서(사전 삽입 순서·목록 순서)여야 한다.

    알고리즘 (CPython 3.12 그대로 — `sum(xs)` 는 int 0 에서 시작한다):
        f = 0 + x0                                  첫 항: int 0 과 더하므로 −0.0 은 +0.0 이 된다
        이후 항마다  t = f + x ;  c += (|f| ≥ |x|) ? (f − t) + x : (x − t) + f ;  f = t
        끝에  (c ≠ 0 이고 c 유한) ? f + c : f        inf·nan 으로 넘친 합을 c 가 망치지 않게

    적용 범위: 항이 전부 **파이썬 `float`** 일 때. 항이 `int` 면 CPython 은 그 항을 보정 갱신 없이 `f += x`
    하고, `np.float64` 처럼 float 의 하위형이면 그 항부터 **보정 없는** 일반 덧셈으로 떨어진다 — 그런 자리는
    `sum_seq` 가 맞다 (통합자는 v5 자리의 항 타입을 확인한다).

    `xs`   (n,) 실수 배열 또는 파이썬 시퀀스 (정적 길이 n ≥ 0)
    `mask` (n,) bool — False 인 칸은 목록에 없는 것처럼 건너뛴다 (첫 항 규칙도 첫 True 칸에 적용). None = 전부
    돌려주는 값: () float64. jit·vmap 가능 — `lax.scan` 으로 원소 순서대로, 장벽으로 `0 + x0` 과 `t` 가 접히지
    않게 한다 (XLA 가 `0 + x → x`, `(f − (f + x)) + x → 0` 으로 단순화하면 답이 갈린다).
    """
    if isinstance(xs, (list, tuple)):
        xs = (jnp.stack([jnp.asarray(x, jnp.float64) for x in xs]) if xs
              else jnp.zeros((0,), jnp.float64))
    xs = jnp.asarray(xs, jnp.float64)
    n = int(xs.shape[0])
    mask = jnp.ones((n,), jnp.bool_) if mask is None else jnp.asarray(mask, jnp.bool_)
    zero = lax.optimization_barrier(jnp.zeros((), jnp.float64))    # `0 + x0` 의 0 — 접히지 않게 (−0.0 → +0.0)

    def body(carry, x):
        f, c, seen = carry
        w, on = x
        t = lax.optimization_barrier(f + w)                        # 실체화 — (f − t) 가 대수적으로 안 지워진다
        comp = jnp.where(jnp.abs(f) >= jnp.abs(w), (f - t) + w, (w - t) + f)
        nf = jnp.where(seen, t, zero + w)                          # 첫 항이면 0 + x0, 아니면 t
        nc = jnp.where(seen, c + comp, c)                          # 첫 항은 보정 갱신 없음
        return (jnp.where(on, nf, f), jnp.where(on, nc, c), seen | on), None

    (f, c, _), _ = lax.scan(body, (zero, zero, jnp.asarray(False)), (xs, mask))
    return jnp.where((c != 0.0) & jnp.isfinite(c), f + c, f)


def div_const(a, c: float, dtype=jnp.float64):
    """`a / c` (c 는 파이썬 상수) — XLA 가 `a * (1/c)` 로 바꾸지 못하게 분모를 장벽 뒤에 둔다."""
    d = lax.optimization_barrier(jnp.asarray(float(c), dtype))
    return a / d
