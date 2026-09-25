"""`gpu/exact.sum_python` 이 파이썬 3.12 `sum()` 과 **비트 동일**한가 ([[YR-327]] 조각 3·4 준비, key=sumfix).

■ 왜 — 조각 5 가 찾은 함정: 파이썬 3.12 부터 `sum(실수 목록)` 은 Neumaier 보정합이라, v5 가 `sum(...)` 으로
  더하는 자리를 순차 `+=` scan 으로 옮기면 3항부터 마지막 비트가 갈린다. 기대값은 손으로 적지 않는다 —
  **파이썬 `sum()` 을 실제로 불러** 같은 목록의 답을 받아 `==` 로 대조한다 (비트 비교: −0.0 ≠ +0.0, NaN 은 NaN 끼리).

■ 무엇을 보나
  ① 무작위 60항 × 50회 (자릿수 섞음) — `sum()` 과 비트 일치. 순차 `+=` 가 갈린 횟수도 보고한다 (시험이 진짜 보정을 건드렸다는 증거)
  ② 마스크 — False 칸을 뺀 목록의 `sum()` 과 일치 (첫 True 칸에 `0 + x0` 규칙)
  ③ 길이 0 / 1 / 2 — `sum([])`=0 · 한 항(−0.0 → +0.0, nan, ±inf) · 두 항(보정이 t 로 돌아오는 경계)
  ④ nan / inf / 넘침 경계 — `c` 가 비유한이면 보정을 건너뛰는 CPython 규칙
  ⑤ jit · vmap — eager 와 같은 비트, 배치는 한 줄씩 계산한 것과 같다
  ⑥ v5 자리 모양 — end−A (censored_exposure_s) · load/(1+deg) (lane.py:50) · 크레인 부하 K=3 (engine.py:835) 에서
     `sum()` ≠ 순차 `+=` 인 경우가 실제로 생기는지 보고한다 (통합자가 sum_seq → sum_python 판단에 쓴다)
  ⑦ `sum_seq` 는 `+=` 이지 `sum()` 이 아니다 — 갈리는 입력을 하나 못박아 둔다. np.float64 항은 `sum()` 도 순차다.

실행: WSL venv · CPU x64 (Windows 파이썬엔 jax 가 없다). 한 세션 80초 안에 끝난다.
"""
from __future__ import annotations

import math
import random
import struct
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — 마지막 비트가 시험 대상
jnp = jax.numpy

from yard_rl.v6.gpu.exact import sum_python, sum_seq   # noqa: E402

PY312 = sys.version_info >= (3, 12)


# ───────────────────────────────────────────────── 도우미
def _bits(x) -> int:
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def _same(a, b) -> bool:
    """비트 동일 (−0.0 ≠ +0.0) — NaN 은 NaN 끼리 같다고 본다 (부호·페이로드는 v5 가 쓰지 않는다)."""
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or _bits(a) == _bits(b)


def _seq(xs) -> float:
    """파이썬 `acc = 0.0; acc += x` — 보정 없는 순차합 (sum_seq 가 흉내내는 것)."""
    acc = 0.0
    for x in xs:
        acc += x
    return acc


def _rand_terms(rng: random.Random, n: int) -> list[float]:
    """자릿수·부호를 섞은 파이썬 float 목록 — 반올림 오차가 반드시 생기게 (1e-3 ~ 1e6 배율)."""
    return [float(rng.choice((1, -1)) * rng.random() * 10.0 ** rng.randint(-3, 6)) for _ in range(n)]


_JIT = jax.jit(sum_python)


def _check(xs, mask=None, fn=None, label=""):
    """파이썬 `sum()` 과 대조 — 갈리면 첫 갈린 항까지의 접두 합을 함께 낸다.

    기본은 jit 판(`_JIT`) — eager `lax.scan` 은 부를 때마다 다시 컴파일돼 80초 세션 창을 잡아먹는다.
    eager 판은 ⑤ 에서 따로 jit 과 대조한다.
    """
    kept = list(xs) if mask is None else [x for x, m in zip(xs, mask) if m]
    want = sum(kept)
    got = (fn or _JIT)(jnp.asarray(xs, jnp.float64), None if mask is None else jnp.asarray(mask, jnp.bool_))
    if not _same(got, want):
        # 처음 갈리는 접두 길이 — 어느 항부터 어긋나는지
        first = next((k for k in range(1, len(kept) + 1)
                      if not _same(sum_python(jnp.asarray(kept[:k], jnp.float64)), sum(kept[:k]))), None)
        pytest.fail(f"{label} sum_python={float(got)!r} ≠ sum()={want!r} (n={len(kept)}, 처음 갈리는 접두 길이={first})")


# ───────────────────────────────────────────────── ① 무작위 60항 × 50회
def test_random_60_terms_50_trials_match_python_sum(capsys):
    """★핵심 — 자릿수를 섞은 60항 목록 50개에서 `sum()` 과 비트 일치. 순차 `+=` 가 갈린 횟수도 보고한다."""
    rng = random.Random(9900777)
    n_seq_diff = 0
    for k in range(50):
        xs = _rand_terms(rng, 60)
        _check(xs, fn=_JIT, label=f"trial {k}")
        n_seq_diff += sum(xs) != _seq(xs)
    with capsys.disabled():
        print(f"\n[sum probe] python={sys.version_info[:3]} 60항×50회: sum()≠순차+= 인 회수 {n_seq_diff}/50 "
              f"(3.12 보정합이면 대부분 갈려야 정상)")
    if PY312:
        assert n_seq_diff > 0, "sum() 과 순차 += 가 한 번도 안 갈렸다 — 시험 입력이 보정을 못 건드린다"


def test_random_lengths_3_to_40_match_python_sum():
    """길이를 바꿔 가며 (3~40) — 길이가 정적 인자라 모양마다 다시 컴파일되지만 답은 같아야 한다."""
    rng = random.Random(327)
    for n in (3, 4, 5, 7, 11, 16, 23, 40):
        xs = _rand_terms(rng, n)
        _check(xs, label=f"n={n}")


# ───────────────────────────────────────────────── ② 마스크
def test_mask_skips_like_filtered_list():
    """False 칸은 목록에 없는 것과 같다 — 첫 True 칸이 `0 + x0` 자리다. 전부 False 면 0.0."""
    rng = random.Random(5)
    xs = _rand_terms(rng, 60)
    for k in range(20):
        mask = [rng.random() < 0.6 for _ in xs]
        _check(xs, mask, fn=_JIT, label=f"mask {k}")
    # 앞쪽이 꺼진 마스크 — 첫 항 규칙이 첫 True 칸에 적용되는지 (−0.0 을 그 칸에 둔다)
    xs2 = [-0.0] * 60
    xs2[7] = -0.0
    mask2 = [i >= 7 for i in range(60)]
    got = _JIT(jnp.asarray(xs2, jnp.float64), jnp.asarray(mask2))
    assert _same(got, sum(xs2[7:])) and _bits(got) == _bits(0.0), float(got)
    # 전부 False → 0.0 (파이썬 sum([]) == 0)
    z = _JIT(jnp.asarray(xs, jnp.float64), jnp.zeros((60,), jnp.bool_))
    assert _bits(z) == _bits(0.0)


# ───────────────────────────────────────────────── ③ 길이 0 / 1 / 2
def test_length_zero_one_two():
    """빈 목록 0 · 한 항은 `0 + x0`(−0.0 → +0.0) · 두 항은 보정이 도로 t 가 되어 순차와 같다."""
    assert _bits(_JIT(jnp.zeros((0,), jnp.float64), None)) == _bits(sum([])) == _bits(0.0)
    assert _bits(sum_python([])) == _bits(0.0)                                  # 파이썬 빈 시퀀스 (eager)
    for x in (3.5, -2.25, -0.0, 0.0, math.nan, math.inf, -math.inf, 1e308, 2.2250738585072014e-308):
        got = _JIT(jnp.asarray([x], jnp.float64), None)
        assert _same(got, sum([x])), (x, float(got), sum([x]))
    assert _bits(_JIT(jnp.asarray([-0.0], jnp.float64), None)) == _bits(0.0)   # ★ int 0 + (−0.0) = +0.0
    assert _bits(sum_python([-0.0])) == _bits(0.0)                              # eager 도
    pairs = [(1e16, 1.0), (1.0, 1e16), (0.1, 0.2), (1e308, 1e308), (math.inf, -math.inf), (math.inf, math.inf),
             (-0.0, -0.0), (1.0, -1.0), (-1.0, 1.0), (math.nan, 1.0), (1.0, math.nan)]
    for a, b in pairs:
        got = _JIT(jnp.asarray([a, b], jnp.float64), None)
        assert _same(got, sum([a, b])), ((a, b), float(got), sum([a, b]))
        if not (math.isnan(a) or math.isnan(b)):
            assert _same(got, _seq([a, b])), "두 항은 순차합과도 같아야 한다 (보정이 t 로 돌아온다)"


# ───────────────────────────────────────────────── ④ nan / inf / 넘침
@pytest.mark.parametrize("xs", [
    [1e308, 1e308, -1e308],                    # 넘침 → c = −inf → 보정 건너뜀 → inf
    [1e308, 1e308, -math.inf],                 # inf + (−inf) = nan
    [1e16, 1.0, math.inf],                     # 유한 c 가 있다가 inf 를 만나면 c 가 nan → inf
    [1e16, 1.0, -1e16, 1.0],                   # 보정이 실제로 값을 바꾸는 고전 예 (순차 1.0 · 보정 2.0)
    [math.nan, 1.0, 2.0], [1.0, math.nan, 2.0], [1.0, 2.0, math.nan],
    [math.inf, 1.0, 2.0], [1.0, -math.inf, 2.0], [-math.inf, math.inf, 1.0],
    [-0.0, -0.0, -0.0], [-0.0, 1.0, -1.0], [1.0, -1.0, -0.0],
    [0.1] * 10, [0.1] * 3, [1e100, 1.0, -1e100, 1.0, 1e-100],
    [1e-300, 1e-300, 1e-300], [2.2250738585072014e-308, 1.0, -1.0],    # 가장 작은 정규 수 — 비정규는 아래 탐침
], ids=lambda xs: repr(xs)[:40])
def test_nan_inf_and_overflow_edges(xs):
    """CPython 의 `if (c && Py_IS_FINITE(c)) f += c` 규칙 — inf·nan 으로 넘친 합을 보정이 망치지 않는다."""
    _check(xs, label=repr(xs))


def test_subnormal_probe(capsys):
    """★백엔드 탐침 — 비정규 수(< 2.2e-308)를 XLA CPU 실행기는 0 으로 씻는다 (FTZ/DAZ, `ScopedFlushDenormal`).

    실측 2026-09-26 CPU x64: `sum_python([1e-320])` = 0.0 (파이썬 1e-320) — 한 항짜리 `0 + x0` 부터 갈린다. 합산
    순서의 결함이 아니라 컴파일된 모든 연산의 성질이다. v5 값 영역(초·원·비율, 최소 1e-3 자릿수)엔 비정규 수가
    없으므로 동등성에 영향 없음 — 씻는 백엔드면 보고하고 건너뛰고, 안 씻는 백엔드(GPU 등)면 `sum()` 과 대조한다.
    """
    tiny = jnp.float64(5e-324)
    kept = float(jax.jit(lambda a, b: a + b)(tiny, jnp.float64(0.0)))
    mul = float(jax.jit(lambda a, b: a * b)(jnp.float64(1e-160), jnp.float64(1e-160)))   # 1e-320 (비정규)
    flush = kept == 0.0 or mul == 0.0
    with capsys.disabled():
        print(f"\n[denormal probe] backend={jax.default_backend()} 5e-324+0={kept!r} 1e-160*1e-160={mul!r} "
              f"→ flush_to_zero={flush}")
    cases = [[5e-324], [5e-324, -5e-324], [5e-324, 5e-324, 5e-324], [1e-320, -1e-320, 1e-320], [1e-300, -1e-300, 1e-320]]
    if flush:
        pytest.skip("이 백엔드는 비정규 수를 0 으로 씻는다 (XLA CPU FTZ/DAZ) — v5 값 영역엔 없음, 동등성 무관")
    for xs in cases:
        _check(xs, label=f"subnormal {xs!r}")


# ───────────────────────────────────────────────── ⑤ jit · vmap
def test_jit_and_vmap_match_eager_and_python():
    """jit == eager == sum(); vmap 배치 (B,n)+(B,n) 마스크는 한 줄씩 계산한 것과 잎마다 같다."""
    rng = random.Random(42)
    rows = [_rand_terms(rng, 32) for _ in range(12)]
    masks = [[rng.random() < 0.7 for _ in r] for r in rows]
    X = jnp.asarray(rows, jnp.float64)
    M = jnp.asarray(masks, jnp.bool_)
    got_v = jax.vmap(sum_python)(X, M)
    got_vj = jax.jit(jax.vmap(sum_python))(X, M)
    got_v_nomask = jax.jit(jax.vmap(lambda x: sum_python(x)))(X)
    for i, (r, m) in enumerate(zip(rows, masks)):
        want = sum([x for x, k in zip(r, m) if k])
        jit1 = _JIT(X[i], M[i])
        assert _same(jit1, want), (i, float(jit1), want)
        if i < 3:                                                   # eager 는 셋만 (호출마다 재컴파일)
            eager = sum_python(X[i], M[i])
            assert _same(eager, want), (i, float(eager), want)
        assert _same(got_v[i], want) and _same(got_vj[i], want), (i, float(got_v[i]), float(got_vj[i]), want)
        assert _same(got_v_nomask[i], sum(r)), (i, float(got_v_nomask[i]), sum(r))
    # −0.0 첫 항이 vmap·jit 아래서도 +0.0 — XLA 의 `0 + x → x` 접기가 장벽에 막혔는지
    Z = jnp.asarray([[-0.0, -0.0, -0.0], [-0.0, 1.0, -1.0]], jnp.float64)
    out = jax.jit(jax.vmap(lambda x: sum_python(x)))(Z)
    assert [_bits(v) for v in out] == [_bits(sum(r)) for r in Z.tolist()] == [_bits(0.0)] * 2


# ───────────────────────────────────────────────── ⑥ v5 자리 모양
def test_v5_site_shapes_report(capsys):
    """v5 가 `sum()` 하는 세 자리의 값 모양으로 — `sum()` 과 비트 일치, 순차 `+=` 가 갈리는 비율을 보고한다.

    · censored_exposure_s (time_contract.py:141): end − A, end = 86400, A 는 초 단위 실수 (밀리초 자리 섞음)
    · lane.py:50 평균 혼잡: load/(1+deg), load ∈ {0,1,2,3}, deg ∈ {1,2,3} — L = 3~8 레인
    · engine.py:835 크레인 부하 합: max(0, available_at − clock) — K = 3 (K = 2 는 두 항이라 항상 같다)
    """
    rng = random.Random(6668)
    report = {}
    # censored exposure — 트럭 40대
    n_diff = 0
    for k in range(20):
        end = 86400.0
        a = sorted(round(rng.random() * 86400.0, 3) for _ in range(40))
        xs = [end - x for x in a]
        _check(xs, label=f"exposure {k}")
        n_diff += sum(xs) != _seq(xs)
    report["exposure 40항"] = n_diff
    # lane mean — L 레인
    n_diff = n_tot = 0
    for L in range(3, 9):
        for k in range(10):
            xs = [float(rng.randint(0, 3)) / (1.0 + rng.randint(1, 3)) for _ in range(L)]
            _check(xs, label=f"lane L={L} {k}")
            n_tot += 1
            n_diff += sum(xs) != _seq(xs)
    report["lane L=3~8"] = f"{n_diff}/{n_tot}"
    # crane loads K=3
    n_diff = 0
    for k in range(30):
        clock = rng.random() * 86400.0
        xs = [max(0.0, clock + rng.random() * 900.0 - rng.random() * 300.0 - clock) for _ in range(3)]
        _check(xs, label=f"loads {k}")
        n_diff += sum(xs) != _seq(xs)
    report["crane K=3 30회"] = n_diff
    with capsys.disabled():
        print(f"\n[v5 site probe] sum()≠순차+= : {report}")


# ───────────────────────────────────────────────── ⑦ sum_seq 의 자리 · np.float64 항
def test_sum_seq_is_plain_left_fold_not_python312_sum():
    """`sum_seq` 는 `+=` 다 — [0.1]×10 에서 `sum()` 과 갈리고(3.12), `sum_python` 은 `sum()` 과 같다."""
    xs = [0.1] * 10
    seq = float(sum_seq(jnp.asarray(xs, jnp.float64)))
    assert seq == _seq(xs) == 0.9999999999999999
    if PY312:
        assert sum(xs) == 1.0 and seq != sum(xs)
    assert _same(_JIT(jnp.asarray(xs, jnp.float64), None), sum(xs))


def test_numpy_float64_items_make_python_sum_sequential():
    """★통합자 주의 — 항이 `np.float64` 면 CPython 이 보정 없는 일반 덧셈으로 떨어져 `sum()` 이 순차합이 된다.
    그런 v5 자리는 sum_seq 가 맞고 sum_python 은 틀린다."""
    xs = [np.float64(0.1)] * 10
    assert float(sum(xs)) == _seq([float(x) for x in xs]) == 0.9999999999999999
    assert float(sum(xs)) == float(sum_seq(jnp.asarray(xs, jnp.float64)))
    if PY312:
        assert float(sum(xs)) != float(_JIT(jnp.asarray(xs, jnp.float64), None))
