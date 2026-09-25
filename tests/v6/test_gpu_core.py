"""v6 배열 세계의 핵심 계약 ([[YR-327]]).

■ 무엇을 지켜야 하나
  ① 사건 큐가 **힙과 같은 순서**를 낸다 — 시각순, 동시각은 넣은 순서
  ② 칸이 모자라면 **조용히 버리지 않고 표시**한다
  ③ 오더 전체가 **한 번의 순전파**로 점수를 받는다
  ④ 가치·우위 분해가 못박혀 있다 (우위 평균 0)
  ⑤ 빈 칸이 기준선을 오염시키지 않는다
  ⑥ 배치(vmap)가 하나씩 돌린 것과 **같은 답**을 낸다
"""
from __future__ import annotations

import heapq

import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")

from yard_rl.v6.gpu.events import (empty_queue, n_pending, next_event, peek_time,
                                   push_event)
from yard_rl.v6.gpu.policy import (ORDER_FEATURES, choose,
                                   counterfactual_advantage, init_params,
                                   q_values, q_values_batch)
from yard_rl.v6.gpu.state import (censored_turn_time_s, empty_world, turn_time_s)


# ───────────────────────────────────────────────── 사건 큐
def test_queue_matches_heap_order():
    """★힙과 같은 순서를 낸다 — 시각순, 동시각은 **넣은 순서**."""
    rows = [(30.0, 1, 7), (10.0, 2, 3), (10.0, 3, 4), (20.0, 4, 5), (10.0, 5, 6)]
    q = empty_queue(16)
    for t, k, tg in rows:
        q = push_event(q, t, k, tg)

    heap: list = []
    for i, (t, k, tg) in enumerate(rows):
        heapq.heappush(heap, (t, i, k, tg))

    got, want = [], []
    while heap:
        q, t, k, tg, alive = next_event(q)
        assert bool(alive)
        got.append((float(t), int(k), int(tg)))
        ht, _, hk, htg = heapq.heappop(heap)
        want.append((ht, hk, htg))
    assert got == want, f"힙과 순서가 다르다\n  배열 {got}\n  힙   {want}"


def test_empty_queue_reports_not_alive():
    """빈 큐에서 꺼내면 '없다' 고 한다 — 예외로 터지지 않는다."""
    q = empty_queue(4)
    _, _, _, _, alive = next_event(q)
    assert not bool(alive)
    assert float(peek_time(q)) == float("inf")


def test_overflow_is_reported_not_dropped():
    """★칸이 모자라면 **표시한다**. 조용히 버리면 세계가 달라진 것을 아무도 모른다."""
    q = empty_queue(2)
    for t in (1.0, 2.0, 3.0, 4.0):
        q = push_event(q, t, 0, 0)
    assert int(q.overflow) == 2, "넘친 수를 안 셌다"
    assert int(n_pending(q)) == 2


def test_popped_slot_is_reused():
    """꺼낸 칸은 다시 쓴다 — 밀어내지 않는다(크기가 변하면 컴파일이 안 된다)."""
    q = empty_queue(2)
    q = push_event(q, 5.0, 1, 1)
    q = push_event(q, 6.0, 2, 2)
    q, *_ = next_event(q)
    q = push_event(q, 7.0, 3, 3)          # 빈 자리에 들어가야 한다
    assert int(q.overflow) == 0
    assert int(n_pending(q)) == 2


# ───────────────────────────────────────────────── 정책망
def _xy(n=6, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (n, ORDER_FEATURES), jnp.float32)
    mask = jnp.array([True] * (n - 2) + [False] * 2)
    return x, mask


def test_all_orders_scored_in_one_pass():
    """★오더 전체가 **한 번의 순전파**로 점수를 받는다."""
    p = init_params(jax.random.PRNGKey(1))
    x, mask = _xy(8)
    q, v, a = q_values(p, x, mask)
    assert q.shape == (8,), "오더마다 점수가 하나씩 나와야 한다"
    assert v.shape == ()


def test_value_advantage_decomposition_is_pinned():
    """★우위의 평균이 0 — 안 그러면 가치와 우위가 같은 양을 나눠 갖는 방법이 무한하다."""
    p = init_params(jax.random.PRNGKey(2))
    x, mask = _xy(10)
    _, _, a = q_values(p, x, mask)
    assert abs(float((a * mask).sum() / mask.sum())) < 1e-5
    assert float(jnp.abs(a[~mask]).sum()) == 0.0, "빈 칸에 우위가 붙었다"


def test_masked_slots_never_chosen():
    """빈 칸은 절대 안 뽑힌다."""
    p = init_params(jax.random.PRNGKey(3))
    x, mask = _xy(7)
    assert bool(mask[int(choose(p, x, mask))])


def test_baseline_ignores_empty_slots():
    """★빈 칸이 기준선을 오염시키지 않는다 — 넣으면 반사실이 거짓이 된다."""
    p = init_params(jax.random.PRNGKey(4))
    x, mask = _xy(6)
    adv_a = counterfactual_advantage(p, x, mask, jnp.int32(0))
    # 빈 칸의 특징을 아무렇게나 바꿔도 우위가 안 변해야 한다
    x2 = x.at[-1].set(x[-1] * 1_000.0)
    adv_b = counterfactual_advantage(p, x2, mask, jnp.int32(0))
    assert abs(float(adv_a) - float(adv_b)) < 1e-4


def test_counterfactual_advantage_needs_no_simulation():
    """★반사실 우위가 순전파 한 번으로 나온다 — v4 는 여기서 세계 둘을 굴렸다."""
    p = init_params(jax.random.PRNGKey(5))
    x, mask = _xy(12)
    q, _, _ = q_values(p, x, mask)
    base = float((q * mask).sum() / mask.sum())
    for i in (0, 3, 7):
        got = float(counterfactual_advantage(p, x, mask, jnp.int32(i)))
        assert abs(got - (float(q[i]) - base)) < 1e-4


def test_batch_matches_one_at_a_time():
    """★배치(vmap)가 하나씩 돌린 것과 같은 답 — 이게 깨지면 GPU 결과를 못 믿는다."""
    p = init_params(jax.random.PRNGKey(6))
    xs = jnp.stack([_xy(5, s)[0] for s in range(4)])
    ms = jnp.stack([_xy(5, s)[1] for s in range(4)])
    qb, vb, ab = q_values_batch(p, xs, ms)
    for i in range(4):
        q1, v1, a1 = q_values(p, xs[i], ms[i])
        assert float(jnp.abs(qb[i] - q1).max()) < 1e-5
        assert abs(float(vb[i]) - float(v1)) < 1e-5


# ───────────────────────────────────────────────── 성과지표
def test_turn_time_and_censoring():
    """턴타임 = 게이트 아웃 − 게이트 인. **못 나간 오더는 검열**한다."""
    w = empty_world(3, 1, end_s=1_000.0)
    o = w.orders._replace(
        gate_in_s=jnp.array([100.0, 200.0, jnp.inf], jnp.float32),
        gate_out_s=jnp.array([460.0, jnp.inf, jnp.inf], jnp.float32))
    assert float(turn_time_s(o)[0]) == 360.0
    c = censored_turn_time_s(o, 1_000.0)
    assert float(c[0]) == 360.0
    assert float(c[1]) == 800.0, "못 나간 오더는 end − 게이트인 으로 세야 한다"
    assert bool(jnp.isnan(c[2])), "안 들어온 오더는 표본이 아니다"


def test_ranking_is_what_must_match_not_raw_values():
    """★정책은 점수로 **줄을 세워** 고른다 — 값이 소수점 아래에서 달라도 순서가 같으면 같은 정책이다.

    GPU 와 CPU 는 더하는 순서가 달라 부동소수점 결과가 미세하게 어긋난다
    (실측 상대 오차 7.8e-4). 이 시험은 **순서가 그 잡음에 안 흔들리는지**를 본다.

    ⚠️ 학습이 진행돼 후보 간 격차가 좁아지면 뒤집힐 수 있다 — 그래서 **판정은
    CPU 로** 다시 재는 것이 규약이다([[YR-327]]).
    """
    p = init_params(jax.random.PRNGKey(7))
    x, mask = _xy(12, seed=3)
    q, _, _ = q_values(p, x, mask)
    # 점수에 아주 작은 잡음을 섞어도 1등이 안 바뀌어야 한다
    noise = jax.random.normal(jax.random.PRNGKey(8), q.shape, jnp.float32) * 1e-4
    big = jnp.where(mask, 0.0, 1e30)
    assert int(jnp.argmin(q + big)) == int(jnp.argmin(q + noise + big)), (
        "1e-4 잡음에 1등이 뒤집혔다 — 후보 간 격차가 이미 잡음 수준이다")
