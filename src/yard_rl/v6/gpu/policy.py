"""**전체 오더를 한 번에** 보는 정책망 ([[YR-327]] · 사용자 지시 2026-09-25).

■ 무엇이 달라지나
  v4·v5 는 후보 하나씩 망에 넣어 점수를 받는다. 결정 한 번에 후보가 20~28개면
  망을 20~28번 부른다. v6 는 **오더 전체를 행렬 하나로** 넣어 점수를 한 번에 받는다.

      지금:  for 후보 in 후보들:  점수 = 망(후보)      ← 20~28회 호출
      v6  :  점수들 = 망(전 오더 행렬)                  ← **1회**

■ ★가치와 우위를 갈라서 낸다 (dueling · [[YR-326]] ①)
      Q(상태, 오더) = V(상태) + A(상태, 오더)
                       ↑ 공통 99.9%   ↑ 배워야 할 0.1%

  이 구조가 정확히 우리 문제 때문에 발명됐다. 원 논문 동기가 *"행동 격차 0.04 vs
  상태 가치 15"* (=0.27%) 인데 **우리 비는 0.003% 로 100배 더 심하다**. 그리고
  [[YR-218]] 이 겪은 *"망 잔차가 행동 효과의 43~94배"* 가 같은 현상이다.

  ⚠️ 우위는 **평균을 빼서** 못박는다(`A − mean(A)`). 안 빼면 V 와 A 가 같은 양을
  나눠 갖는 방법이 무한히 많아 학습이 흐른다(원 논문의 식별 문제).

■ ★반사실 기준선이 공짜로 따라온다 ([[YR-326]] ②)
  후보 전체의 점수가 한 번에 나오므로 *"내가 고른 것 − 후보 평균"* 을 **시뮬레이션
  없이** 계산할 수 있다. v4 가 세계 둘을 굴려 하던 일을 순전파 한 번으로 한다.

■ 마스크 — 없는 오더는 계산에서 뺀다
  칸이 고정이라 빈 칸(-1)이 섞인다. 점수에 −inf 를 씌워 고르기에서 빠지게 하고,
  평균에서도 뺀다. **빈 칸을 평균에 넣으면 기준선이 거짓이 된다.**
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

#: 오더 하나가 망에 주는 특징 수 — v4 크레인 8칸과 같은 뜻, 배열판
ORDER_FEATURES = 9
NEG_INF = -1e30


class PolicyParams(NamedTuple):
    """몸통 하나 + 머리 둘(가치·우위). 전부 평범한 조밀 층이다."""

    w1: jnp.ndarray; b1: jnp.ndarray      # 몸통 1
    w2: jnp.ndarray; b2: jnp.ndarray      # 몸통 2
    wv: jnp.ndarray; bv: jnp.ndarray      # 가치 머리 → ()
    wa: jnp.ndarray; ba: jnp.ndarray      # 우위 머리 → 오더마다 1


def init_params(key, *, dim: int = ORDER_FEATURES, hidden: int = 64) -> PolicyParams:
    """글로럿 초기화. v4 `CraneNet` 과 **같은 크기**(64·64)로 맞춰 비교를 깨끗하게."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    g = lambda k, i, o: jax.random.normal(k, (i, o), jnp.float32) * jnp.sqrt(2.0 / i)
    return PolicyParams(
        w1=g(k1, dim, hidden), b1=jnp.zeros((hidden,), jnp.float32),
        w2=g(k2, hidden, hidden), b2=jnp.zeros((hidden,), jnp.float32),
        wv=g(k3, hidden, 1), bv=jnp.zeros((1,), jnp.float32),
        wa=g(k4, hidden, 1), ba=jnp.zeros((1,), jnp.float32))


def _trunk(p: PolicyParams, x: jnp.ndarray) -> jnp.ndarray:
    """(N, 특징) → (N, 은닉). 오더 전체가 한 번에 지나간다."""
    h = jnp.maximum(0.0, x @ p.w1 + p.b1)
    return jnp.maximum(0.0, h @ p.w2 + p.b2)


def q_values(p: PolicyParams, x: jnp.ndarray, mask: jnp.ndarray):
    """오더 행렬 → `(Q, V, A)`.

    `x`    : (N, 특징)  — 오더 전체
    `mask` : (N,) bool  — 참인 칸만 진짜 오더

    ★V 는 **상태 하나의 값**이라 오더들을 가로질러 모은다(마스크 평균). A 는
      오더마다 하나이고 평균을 빼서 못박는다.
    """
    h = _trunk(p, x)                                   # (N, 은닉)
    m = mask.astype(jnp.float32)[:, None]
    n = jnp.maximum(m.sum(), 1.0)
    v = ((h * m).sum(0) / n) @ p.wv + p.bv             # (1,) — 상태 가치
    a = (h @ p.wa + p.ba)[:, 0]                        # (N,) — 오더별 우위
    a_mean = (a * mask).sum() / jnp.maximum(mask.sum(), 1)
    a = jnp.where(mask, a - a_mean, 0.0)               # ★평균 0 으로 못박는다
    return v[0] + a, v[0], a


def choose(p: PolicyParams, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """가장 좋은(= 비용이 가장 낮은) 오더의 번호. 빈 칸은 못 뽑힌다."""
    q, _, _ = q_values(p, x, mask)
    return jnp.argmin(jnp.where(mask, q, -NEG_INF))


def counterfactual_advantage(p: PolicyParams, x: jnp.ndarray, mask: jnp.ndarray,
                             picked: jnp.ndarray) -> jnp.ndarray:
    """**반사실 우위** — 고른 것이 *후보 평균*보다 얼마나 나은가.

    v4 는 이 값을 얻으려 세계 둘을 굴렸다(결정 1건에 3시간 시뮬 2회). 여기서는
    순전파 한 번이다 — 후보 전체의 Q 가 이미 나와 있으니 평균만 빼면 된다.

    ⚠️ 값이 맞으려면 Q 가 맞아야 한다. **그 검증이 [[YR-327]] 의 핵심 관문**이다.
    """
    q, _, _ = q_values(p, x, mask)
    base = (q * mask).sum() / jnp.maximum(mask.sum(), 1)
    return q[picked] - base


#: ★배치 판 — 세계 B 개의 결정을 한 번에. 이것이 GPU 가 빠른 이유다.
q_values_batch = jax.vmap(q_values, in_axes=(None, 0, 0))
choose_batch = jax.vmap(choose, in_axes=(None, 0, 0))
