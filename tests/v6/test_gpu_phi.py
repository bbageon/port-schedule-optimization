"""배열 Φ (gpu/phi.py) 가 v5 `reward/phi.terminal_cost_krw` 와 **같은 답**을 내는가 ([[YR-327]] 조각 5).

■ 무엇을 지키나 — 기대값은 손으로 적지 않는다. **v5 를 실제로 불러** 같은 입력의 답을 받아 대조한다.
  ① 정수(n_trucks·n_censored)·분위수(p50/p90/p99)·비율·평균 — `==` (비트 일치)
  ② 원화 네 항·합계 — `==` (v5 `+=` 순서 그대로 더한다). 갈리면 |Δ| 와 함께 실패 메시지로 낸다
  ③ Φ 가 t 에 대해 비감소 (v5 runtime.boundary 의 `delta < −1e-5` 불변식과 같은 뜻)
  ④ 배치(vmap)가 하나씩 계산한 것과 잎 전부 같다

■ 무대
  A단계  손입력 6건 (A/O 조합: O>end 검열 · O 없음 · 완료 2 · A>end 제외 · A 없음) × end_s∈{3600,7200,10800},
         본선 2척, YC 123.4초, 재조작 3 — 명세의 손계산값(7200: wait 280,000 · total≈324,185.83 · n 4 ·
         검열 2 · p50 6700 · p90=p99 7000)은 v5 로 먼저 확인한 뒤 배열과 대조한다
  B단계  tests/v6/test_cargo_capacity_wait.world() 무대(반입 1건 · 반입+반출 2건)를 v5 로 굴려 검토 격자
         t 마다 원료(기록 A/O · YC 빈 주행 · 재조작 · 본선 유휴 표)를 뽑고 **같은 t 에 v5 Φ 를 부른 값**과
         배열 Φ 를 대조한다 (runtime.read_cost 가 쓴 cost_breakdown 과도 같아야 한다)
  C단계  무작위 기록 300건 × end_s 5종 (경계 A==end · A>end · O==end · O>end · None 섞음) — 합산 순서 감시

실행: WSL venv · x64 CPU. FMA·역수곱은 플래그가 아니라 gpu/exact.py 의 장벽 규약이 막는다.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import os
import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — 원화 1e7 규모에서 float32 는 1원 단위가 흔들린다
jnp = jax.numpy

from yard_rl.v6.gpu.phi import (PHI_KEYS, orders_from_records, terminal_cost_krw,   # noqa: E402
                                terminal_cost_krw_batch, truck_wait_krw,
                                vessel_idle_by_ship, vessel_idle_krw)
from yard_rl.v6.gpu.state import OrderArrays, empty_orders                            # noqa: E402
from yard_rl.v6.reward import krw                                                     # noqa: E402
from yard_rl.v6.reward.phi import terminal_cost_krw as phi_v5                        # noqa: E402
from yard_rl.v6.schema.record import ExecutionRecord                                  # noqa: E402

INT_KEYS = ("n_trucks", "n_censored")
KRW_KEYS = ("phi_krw", "c_wait", "c_move", "c_rehandle", "c_vessel")


# ───────────────────────────────────────────────── 도우미
def _rec(key: str, a, o) -> ExecutionRecord:
    return ExecutionRecord(doc_key=key, gate_in_s=a, gate_out_s=o)


def _v5(records, end_s, vessel=None, yc=0.0, rh=0) -> dict:
    """v5 를 실제로 부른다 — 기대값의 유일한 출처."""
    return phi_v5(records, end_s=end_s, vessel_idle=vessel, yc_extra_move_s=yc, rehandles=rh).as_dict()


def _vessel_arrays(vessel: dict | None):
    """v5 `{배: (GT, 유휴 초)}` → (gt, idle) 배열 — dict 순서 그대로 (합산 순서)."""
    if not vessel:
        return None, None
    rows = list(vessel.values())
    return (jnp.asarray([float(g) for g, _ in rows]), jnp.asarray([float(s) for _, s in rows]))


_phi_jit = jax.jit(terminal_cost_krw)
_phi_batch_jit = jax.jit(terminal_cost_krw_batch)


def _array(records, end_s, vessel=None, yc=0.0, rh=0) -> dict:
    """배열 Φ (jit) — 같은 입력을 배열로 옮겨 넣는다."""
    gt, idle = _vessel_arrays(vessel)
    out = _phi_jit(orders_from_records(records), float(end_s), vessel_gt=gt, vessel_idle_s=idle,
                   yc_extra_move_s=float(yc), rehandles=int(rh))
    return jax.block_until_ready(out).as_dict()


def _diff(want: dict, got: dict) -> list[str]:
    """v5(want) 와 배열(got) 을 키마다 `==` 로 — 갈린 키는 |Δ| 와 함께 적는다."""
    bad = []
    assert tuple(want) == PHI_KEYS == tuple(got), (tuple(want), tuple(got))
    for k in PHI_KEYS:
        w, g = want[k], got[k]
        if k in INT_KEYS:
            if int(w) != int(g):
                bad.append(f"{k}: v5={w} 배열={g}")
        elif not (w == g):
            bad.append(f"{k}: v5={w!r} 배열={g!r} |Δ|={abs(w - g):.3e}")
    return bad


def _assert_same(want: dict, got: dict, where: str) -> None:
    bad = _diff(want, got)
    assert not bad, f"[{where}] v5 와 갈린 항목 {len(bad)}개:\n  " + "\n  ".join(bad)


# ───────────────────────────────────────────────── A단계 — 손입력 6건
def stage_a_records() -> dict[str, ExecutionRecord]:
    """end=7200 에서 표본 {6700 검열, 7000 검열, 2000, 3000} · 대기 0 트럭 · 무대 밖 트럭."""
    return {
        "rA": _rec("rA", 500.0, 9000.0),    # O > end → 6700 검열 (명세 hard_parts 의 손계산 사례)
        "rB": _rec("rB", 200.0, None),      # O 없음 → 7000 검열
        "rC": _rec("rC", 1000.0, 3000.0),   # 완료 2000
        "rD": _rec("rD", 100.0, 3100.0),    # 완료 3000
        "rE": _rec("rE", 8000.0, None),     # A > end → 대기 0 · 표본 제외
        "rF": _rec("rF", None, None),       # 게이트 전 → 무대 밖
    }


STAGE_A_VESSEL = {"S1": (100_000.0, 300.0), "S2": (50_000.0, 270.0)}   # 24,916.67 + 11,212.50 원
STAGE_A_YC, STAGE_A_REHANDLES = 123.4, 3


def test_stage_a_hand_values_confirmed_by_v5():
    """명세의 손계산 기대값을 **v5 가 먼저** 내는지 — 이 시험이 깨지면 손계산·무대 설계가 틀린 것."""
    d = _v5(stage_a_records(), 7200.0, STAGE_A_VESSEL, STAGE_A_YC, STAGE_A_REHANDLES)
    assert d["c_wait"] == pytest.approx(280_000.0, rel=1e-9)
    assert d["c_move"] == pytest.approx(2056.67, abs=0.01)
    assert d["c_rehandle"] == 6000.0
    assert d["c_vessel"] == pytest.approx(36129.17, abs=0.01)
    assert d["phi_krw"] == pytest.approx(324185.83, abs=0.01)
    assert (d["n_trucks"], d["n_censored"]) == (4, 2)
    assert (d["p50_turn_time_s"], d["p90_turn_time_s"], d["p99_turn_time_s"]) == (6700.0, 7000.0, 7000.0)


@pytest.mark.parametrize("end_s", [3600.0, 7200.0, 10800.0])
def test_stage_a_matches_v5(end_s):
    """A단계 — 세 창 끝에서 열두 키 전부 `==`."""
    recs = stage_a_records()
    want = _v5(recs, end_s, STAGE_A_VESSEL, STAGE_A_YC, STAGE_A_REHANDLES)
    got = _array(recs, end_s, STAGE_A_VESSEL, STAGE_A_YC, STAGE_A_REHANDLES)
    _assert_same(want, got, f"A단계 end={end_s:g}")


def test_stage_a_batch_equals_single_and_v5():
    """④ 배치(vmap) — end_s 세 개를 한 번에 계산해도 하나씩 계산한 것·v5 와 잎 전부 같다."""
    recs = stage_a_records()
    ends = [3600.0, 7200.0, 10800.0]
    o = orders_from_records(recs)
    stacked = jax.tree_util.tree_map(lambda x: jnp.stack([x] * len(ends)), o)
    gt, idle = _vessel_arrays(STAGE_A_VESSEL)
    out = jax.block_until_ready(_phi_batch_jit(
        stacked, jnp.asarray(ends), vessel_gt=jnp.stack([gt] * 3), vessel_idle_s=jnp.stack([idle] * 3),
        yc_extra_move_s=jnp.full((3,), STAGE_A_YC), rehandles=jnp.full((3,), STAGE_A_REHANDLES, jnp.int32)))
    assert out.total.shape == (3,)
    for i, e in enumerate(ends):
        _assert_same(_v5(recs, e, STAGE_A_VESSEL, STAGE_A_YC, STAGE_A_REHANDLES), out.as_dict(i), f"batch[{i}] vs v5")
        _assert_same(_array(recs, e, STAGE_A_VESSEL, STAGE_A_YC, STAGE_A_REHANDLES), out.as_dict(i), f"batch[{i}] vs single")


def test_edge_gate_in_exactly_at_window_end():
    """★`A == end` 경계 — v5 Φ 는 턴타임 0 인 검열 표본으로 **센다** (state 판은 NaN). 배열 Φ 가 되살리는지."""
    recs = {"x": _rec("x", 60.0, None), "y": _rec("y", 60.0, 9000.0), "z": _rec("z", 10.0, 60.0)}
    for end in (60.0, 59.999, 60.001):
        _assert_same(_v5(recs, end), _array(recs, end), f"A==end 경계 end={end}")
    d = _array(recs, 60.0)
    assert (d["n_trucks"], d["n_censored"], d["p50_turn_time_s"]) == (3, 2, 0.0)


def test_empty_and_no_sample_cases():
    """표본 0 → 진단 열 전부 0 · 기록 0건 · 본선 표 없음 — v5 와 같다."""
    _assert_same(_v5({}, 100.0), _array({}, 100.0), "기록 0건")
    recs = {"f": _rec("f", None, None), "g": _rec("g", 500.0, None)}
    _assert_same(_v5(recs, 100.0, None, 10.0, 2), _array(recs, 100.0, None, 10.0, 2), "표본 0 · 계수기만")


# ───────────────────────────────────────────────── 단가 함수 (krw.py 배열판) — 비트 일치
def test_truck_wait_and_vessel_krw_elementwise_match_v5():
    """`K·tt/3600 + K·max(0,tt−3600)/3600` · `(2.99·gt)·idle/3600` — 무작위 1,000점에서 `==`."""
    rng = random.Random(5)
    tts = [rng.uniform(0.0, 20000.0) for _ in range(1000)] + [0.0, 3600.0, 3600.0000001, 1e-9, 86400.0]
    got = np.asarray(jax.jit(truck_wait_krw)(jnp.asarray(tts)))
    bad = [(t, krw.truck_wait_krw(t), g) for t, g in zip(tts, got) if krw.truck_wait_krw(t) != g]
    assert not bad, f"truck_wait_krw 갈림 {len(bad)}건, 예: {bad[:3]}"
    gts = [rng.choice([50_000.0, 100_000.0, 150_000.0]) for _ in range(500)]
    idles = [rng.uniform(0.0, 7200.0) for _ in range(500)]
    got_v = np.asarray(jax.jit(vessel_idle_krw)(jnp.asarray(gts), jnp.asarray(idles)))
    bad_v = [(g, s, krw.vessel_idle_krw(g, s), x) for g, s, x in zip(gts, idles, got_v)
             if krw.vessel_idle_krw(g, s) != x]
    assert not bad_v, f"vessel_idle_krw 갈림 {len(bad_v)}건, 예: {bad_v[:3]}"


def test_division_by_constant_probe(capsys):
    """★상수 재결합·역수 곱 감시 — `K·x/3600` 을 XLA 가 `x·(K/3600)` 나 `(K·x)·(1/3600)` 로 바꾸면 마지막 비트가 갈린다.

    실측 (CPU · jax 0.11.2): jit 한 평범한 식은 4,000점 중 **1,251점**이 파이썬과 갈린다 — 역수 곱(≈24점)이
    아니라 `(K·x)/3600 → x·(K/3600)` 상수 재결합 패턴이다. plain 판은 **보고만** 하고 단언은 장벽 판에만.
    """
    from yard_rl.v6.gpu.exact import mul_exact
    from yard_rl.v6.gpu.phi import _div_c
    rng = random.Random(11)
    xs = [rng.uniform(1.0, 1e7) for _ in range(4000)]
    K = krw.KRW_TRUCK_HOUR
    want = np.asarray([K * x / 3600.0 for x in xs])
    plain = np.asarray(jax.jit(lambda v: K * v / 3600.0)(jnp.asarray(xs)))
    recip = np.asarray([(K * x) * (1.0 / 3600.0) for x in xs])
    reassoc = np.asarray([x * (K / 3600.0) for x in xs])
    guarded = np.asarray(jax.jit(lambda v: _div_c(mul_exact(K, v), 3600.0))(jnp.asarray(xs)))
    n_plain, n_recip, n_re = int((plain != want).sum()), int((recip != want).sum()), int((reassoc != want).sum())
    with capsys.disabled():
        print(f"\n[div probe] backend={jax.default_backend()} plain-mismatch={n_plain}/{len(xs)} "
              f"(역수곱이면 ≈{n_recip} · 상수재결합이면 ≈{n_re} · plain==재결합: {int((plain == reassoc).sum())}) "
              f"guarded-mismatch={int((guarded != want).sum())}")
    assert n_recip > 0 and n_re > 0, "탐침 무효 — 두 변형이 이 입력에서 나눗셈과 안 갈린다"
    assert (guarded == want).all()


def test_vessel_idle_by_ship_matches_python_sum(capsys):
    """배별 유휴 = `sum(w) / max(1, sts)` — 파이썬 `sum()` 과 `==` (3.12 부터 Neumaier 보정합, phi.py 주석).

    스트림 40개·배 4척(STS 6·4·1·0)·표에 없는 배(-1) 섞음. 순차 `+=` 와 `sum()` 이 갈리는 배가 있는지도 보고한다.
    """
    import sys
    rng = random.Random(3)
    ships = {"L": 6, "M": 4, "S": 1, "Z": 0}
    names = list(ships)
    streams = [(rng.uniform(0, 5000.0) + 1e-3 * rng.random(), rng.randrange(-1, len(names)))
               for _ in range(40)]
    per: dict[int, list[float]] = {}
    for w, s in streams:
        if s >= 0:
            per.setdefault(s, []).append(w)
    want = [sum(per.get(i, [])) / max(1, ships[n]) for i, n in enumerate(names)]
    seq = []
    for i, n in enumerate(names):
        acc = 0.0
        for w in per.get(i, []):
            acc += w
        seq.append(acc / max(1, ships[n]))
    got = jax.jit(vessel_idle_by_ship, static_argnums=3)(
        jnp.asarray([w for w, _ in streams]), jnp.asarray([s for _, s in streams], jnp.int32),
        jnp.asarray(list(ships.values()), jnp.int32), len(names))
    with capsys.disabled():
        print(f"\n[sum probe] python={sys.version_info[:2]} sum()≠순차+= 인 배: "
              f"{sum(a != b for a, b in zip(want, seq))}/{len(names)}")
    assert [float(x) for x in got] == want
    # 배 하나·스트림 1개 (첫 항만) · 스트림 0개 → 0
    one = jax.jit(vessel_idle_by_ship, static_argnums=3)(jnp.asarray([123.456]), jnp.asarray([0], jnp.int32),
                                                          jnp.asarray([4], jnp.int32), 2)
    assert [float(x) for x in one] == [123.456 / 4, 0.0]


# ───────────────────────────────────────────────── B단계 — v5 무대에서 뽑은 원료
def _cargo_module():
    """tests/v6/test_cargo_capacity_wait.py 의 world()·run() — 시험 모듈이라 경로로 싣는다."""
    try:
        return importlib.import_module("test_cargo_capacity_wait")
    except ImportError:
        path = os.path.join(os.path.dirname(__file__), "test_cargo_capacity_wait.py")
        spec = importlib.util.spec_from_file_location("test_cargo_capacity_wait", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _harvest(pickup: bool) -> list[dict]:
    """무대를 v5 로 굴리며 검토 격자마다 원료와 **그 시점의 v5 Φ** 를 찍는다."""
    from yard_rl.v6.stage.episode import rehandles_of, yc_empty_travel_s
    from yard_rl.v6.stage.month import month_vessel_idle
    mod = _cargo_module()
    terminal, sim, ann, bridge, rt = mod.world(pickup=pickup)
    rows: list[dict] = []

    def check(m, t):
        recs = copy.deepcopy(bridge.records)                 # 이 시점의 기록 (뒤에 바뀐다)
        vessel = month_vessel_idle(m, rt.meta, rt.archive)
        yc, rh = yc_empty_travel_s(m), rehandles_of(m)
        want = _v5(recs, t, vessel, yc, rh)
        assert want == rt.cost_breakdown, "원료가 runtime.read_cost 가 쓴 것과 다르다"
        rows.append(dict(t=float(t), records=recs, vessel=vessel, yc=yc, rh=rh, want=want))

    mod.run(terminal, ann, bridge, rt, check)
    return rows


@pytest.mark.parametrize("pickup", [False, True], ids=["inbound-1", "inbound+pickup-2"])
def test_stage_b_cargo_stage_matches_v5_at_every_review(pickup, capsys):
    """B단계 — 검토 격자 t 마다 열두 키 `==` · Φ 비감소 · 배치판(전 t 를 한 번에)도 같다."""
    rows = _harvest(pickup)
    assert len(rows) >= 40, len(rows)
    bad_all: list[str] = []
    for r in rows:
        got = _array(r["records"], r["t"], r["vessel"], r["yc"], r["rh"])
        bad = _diff(r["want"], got)
        if bad:
            bad_all.append(f"t={r['t']:g}: " + "; ".join(bad))
    assert not bad_all, f"[B단계 pickup={pickup}] 갈린 시점 {len(bad_all)}/{len(rows)}:\n  " + "\n  ".join(bad_all[:8])

    # ③ Φ 비감소 — 검토 시각 순으로
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts)
    phis = [_array(r["records"], r["t"], r["vessel"], r["yc"], r["rh"])["phi_krw"] for r in rows]
    drops = [(a, b, x, y) for (a, x), (b, y) in zip(zip(ts, phis), zip(ts[1:], phis[1:])) if y < x]
    assert not drops, f"Φ 가 줄었다: {drops[:3]}"

    # ④ 배치 — 시점마다 다른 기록을 (T,N) 으로 쌓아 한 번에
    n = max(len(r["records"]) for r in rows)
    os_ = [orders_from_records(r["records"], n_max=n) for r in rows]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *os_)
    out = jax.block_until_ready(_phi_batch_jit(
        stacked, jnp.asarray(ts), yc_extra_move_s=jnp.asarray([r["yc"] for r in rows]),
        rehandles=jnp.asarray([r["rh"] for r in rows], jnp.int32)))
    assert all(not r["vessel"] for r in rows), "이 무대는 본선이 없어야 한다 (표 없이 배치)"
    for i, r in enumerate(rows):
        _assert_same(r["want"], out.as_dict(i), f"B단계 batch t={r['t']:g}")
    with capsys.disabled():
        last = rows[-1]["want"]
        print(f"\n[B단계 pickup={pickup}] 검토 {len(rows)}회 전부 == · 마지막 t={ts[-1]:g} "
              f"Φ={last['phi_krw']:.2f}원 n={last['n_trucks']} 검열={last['n_censored']}")


# ───────────────────────────────────────────────── C단계 — 무작위 기록 (합산 순서 감시)
def _random_records(rng: random.Random, n: int, end_grid: list[float]) -> dict[str, ExecutionRecord]:
    """A/O 를 소수점 시각으로 뽑고 경계(정확히 end · end 뒤 · None)를 섞는다."""
    recs = {}
    for i in range(n):
        kind = rng.random()
        e = rng.choice(end_grid)
        if kind < 0.08:
            a, o = None, None
        elif kind < 0.16:
            a, o = e, None                                    # A == end
        elif kind < 0.24:
            a, o = e + rng.uniform(0.0, 3000.0), None         # A > end
        elif kind < 0.40:
            a = rng.uniform(0.0, e)
            o = e                                             # O == end
        elif kind < 0.60:
            a = rng.uniform(0.0, 20000.0)
            o = a + rng.uniform(0.0, 12000.0)                 # 완료 (O 는 end 앞뒤 섞임)
        else:
            a, o = rng.uniform(0.0, 20000.0), None            # 미완료
        recs[f"r{i:04d}"] = _rec(f"r{i:04d}", a, o)
    return recs


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stage_c_random_records_match_v5(seed, capsys):
    """C단계 — 300건 × end 5종: 합 300항의 `+=` 순서가 어긋나면 마지막 비트에서 잡힌다."""
    rng = random.Random(seed)
    ends = [1800.0, 3600.0, 7200.5, 10800.0, 21600.25]
    recs = _random_records(rng, 300, ends)
    vessel = {f"V{i}": (rng.choice([50_000.0, 100_000.0, 150_000.0]), rng.uniform(0, 5000.0)) for i in range(5)}
    yc, rh = rng.uniform(0, 3000.0), rng.randrange(0, 50)
    seen = 0
    for e in ends:
        want = _v5(recs, e, vessel, yc, rh)
        _assert_same(want, _array(recs, e, vessel, yc, rh), f"C단계 seed={seed} end={e}")
        seen += want["n_trucks"]
    with capsys.disabled():
        print(f"\n[C단계 seed={seed}] 기록 300건 × end 5종 == · 표본 합 {seen}")


def test_orders_from_records_layout():
    """호스트 변환 — 사전 순서가 배열 순서, None 은 +inf, 외부트럭 표시, 칸 부족은 예외."""
    recs = {"b": _rec("b", 5.0, None), "a": _rec("a", None, None), "c": _rec("c", 1.0, 2.0)}
    o = orders_from_records(recs, n_max=5)
    assert isinstance(o, OrderArrays) and o.n == 5
    assert [float(x) for x in o.gate_in_s] == [5.0, float("inf"), 1.0, float("inf"), float("inf")]
    assert [float(x) for x in o.gate_out_s] == [float("inf")] * 2 + [2.0] + [float("inf")] * 2
    assert [bool(x) for x in o.is_external] == [True, True, True, False, False]
    assert o.gate_in_s.dtype == jnp.float64
    with pytest.raises(ValueError):
        orders_from_records(recs, n_max=2)
    # 다른 열은 빈 오더 그대로
    ref = empty_orders(5)
    assert (o.status == ref.status).all() and (o.rehandles == ref.rehandles).all()
