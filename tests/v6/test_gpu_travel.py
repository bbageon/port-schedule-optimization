"""v6 배열 이동시간 ↔ v5 `travel_time.move_container` **비트 동일** 대조 ([[YR-327]] 조각 1).

■ 무엇을 지키나
  ① 무작위 500건에서 (dur, loaded_m, empty_m, end_bay, end_row) 다섯 값이 v5 와 `==` 로 같다
     — 기대값을 손으로 적지 않고 **v5 함수를 실제로 불러** 받는다. 근사(allclose) 가 아니다.
  ② jit 로 컴파일해도, 500건을 vmap 으로 한 번에 돌려도 같은 비트다
  ③ 출력 dtype 이 float64 다 (x64 가 꺼져 조용히 float32 로 내려앉는 사고 방지)
  ④ estimate_reach_s 도 같다 (덤)
  ⑤ 공용 gpu/geom.py Geom 을 넘겨도 같다 (그 파일이 있을 때만)

실행: WSL venv · x64 CPU. (XLA_FLAGS=--xla_allow_excess_precision=false 는 관례 — FMA 를 막지 못한다(실측);
이 파일의 곱은 나눗셈으로만 이어져 FMA 대상이 아니고, 곱→덧셈은 gpu/exact.py 규약을 따른다)
"""
from __future__ import annotations

import functools
import random

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — travel.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu.travel import (MoveOut, Spec7, TravelGeom, estimate_reach_s,
                                   move_container, spec7_from_crane_spec)
from yard_rl.v6.world.domain.models import BlockGeometry, CraneSpec
from yard_rl.v6.world.sim import travel_time as v5

N_CASES = 500
FIELDS = ("dur", "loaded_m", "empty_m", "end_bay", "end_row")


# ───────────────────────────────────────────────── 무작위 입력
def _geom(rng: random.Random) -> BlockGeometry:
    B = rng.randint(4, 40); R = rng.randint(2, 10); T = rng.randint(2, 6)
    return BlockGeometry(block_id="B", bay_count=B, row_count=R, tier_max=T,
                         bay_length_m=rng.uniform(6.0, 7.5),
                         row_width_m=rng.uniform(2.4, 3.2),
                         tier_height_m=rng.uniform(2.5, 3.2))


def _case(rng: random.Random, geoms):
    """(geom, spec, start_bay, start_row, src, dst) 한 건 — 실제 쓰이는 범위 안에서 무작위.
    기하는 풀(10개)에서 고른다 — 실제 엔진도 블록 하나에 기하 하나라, 같은 기하끼리 vmap 으로 묶어 시험한다."""
    geom = rng.choice(geoms)
    B, R, T = geom.bay_count, geom.row_count, geom.tier_max
    spec = CraneSpec(crane_id="YC", service_bay_min=1, service_bay_max=B,
                     gantry_speed_mps=rng.uniform(0.5, 3.0),
                     trolley_speed_mps=rng.uniform(0.5, 2.0),
                     hoist_speed_loaded_mps=rng.uniform(0.3, 1.0),
                     hoist_speed_empty_mps=rng.uniform(0.5, 1.5),
                     lock_time_s=rng.uniform(5.0, 30.0),
                     unlock_time_s=rng.uniform(5.0, 30.0),
                     truck_positioning_time_s=rng.uniform(10.0, 120.0))
    # 현위치 — 초기 배치는 lo+(k+0.5)(hi−lo)/n 같은 비정수, 완료 뒤엔 float(정수). 둘 다 섞는다.
    start_bay = rng.uniform(1.0, float(B)) if rng.random() < 0.5 else float(rng.randint(1, B))
    start_row = rng.uniform(0.0, float(R)) if rng.random() < 0.5 else float(rng.randint(0, R))
    # 슬롯 — row 0 = 차선(트럭 인계), tier 1..T
    src = (rng.randint(1, B), rng.randint(0, R), rng.randint(1, T))
    dst = (rng.randint(1, B), rng.randint(0, R), rng.randint(1, T))
    return geom, spec, start_bay, start_row, src, dst


def _v5(geom, spec, start_bay, start_row, src, dst):
    m = v5.move_container(spec, geom, start_bay, start_row, src, dst)
    return (m.duration_s, m.loaded_gantry_m, m.empty_gantry_m, m.end_bay, m.end_row)


def _mismatch_report(name, rows):
    """불일치 목록 → 사람이 읽을 요약 (개수 · 최대 절대차 · 앞 3건)."""
    if not rows:
        return f"{name}: 불일치 0"
    worst = max(abs(a - b) for _, _, a, b in rows)
    head = "; ".join(f"#{i} {f}: v5={a!r} v6={b!r}" for i, f, a, b in rows[:3])
    return f"{name}: 불일치 {len(rows)}건, 최대 절대차 {worst:.3e} — {head}"


@pytest.fixture(scope="module")
def cases():
    rng = random.Random(20260925)
    geoms = [_geom(rng) for _ in range(10)]
    return [_case(rng, geoms) for _ in range(N_CASES)]


# ───────────────────────────────────────────────── ① 한 건씩 (eager) 비트 동일
def test_move_container_bitwise_equal_eager(cases):
    bad = []
    for i, (geom, spec, sb, sr, src, dst) in enumerate(cases):
        want = _v5(geom, spec, sb, sr, src, dst)
        got = move_container(spec7_from_crane_spec(spec), sb, sr,
                             jnp.asarray(src, jnp.int32), jnp.asarray(dst, jnp.int32),
                             TravelGeom.from_block_geometry(geom))
        assert isinstance(got, MoveOut)
        for f, a, b in zip(FIELDS, want, got):
            assert b.dtype == jnp.float64, f"{f} dtype {b.dtype} — x64 가 꺼졌나"
            bv = float(b)
            if not (a == bv):                       # 비트 동일 — 근사 아님
                bad.append((i, f, a, bv))
    assert not bad, _mismatch_report("eager", bad)


# ───────────────────────────────────────────────── ② jit + vmap 으로 500건 한 번에
def test_move_container_bitwise_equal_jit_vmap(cases):
    """기하는 static 이라 (B,R,T,길이) 가 같은 건끼리 묶어 vmap 한다 — 실제 엔진도 블록 하나에 기하 하나."""
    groups: dict = {}
    for i, c in enumerate(cases):
        groups.setdefault(TravelGeom.from_block_geometry(c[0]), []).append(i)

    @functools.partial(jax.jit, static_argnames=("g",))   # 기하는 static — dataclass 라 hashable
    def batched(spec, sb, sr, src, dst, *, g):
        return jax.vmap(lambda s, a, b, x, y: move_container(s, a, b, x, y, g))(spec, sb, sr, src, dst)

    bad = []
    n_groups = 0
    for g, idx in groups.items():
        n_groups += 1
        sub = [cases[i] for i in idx]
        spec = Spec7(*[jnp.asarray([getattr(c[1], name) for c in sub], jnp.float64)
                       for name in ("gantry_speed_mps", "trolley_speed_mps",
                                    "hoist_speed_loaded_mps", "hoist_speed_empty_mps",
                                    "lock_time_s", "unlock_time_s", "truck_positioning_time_s")])
        sb = jnp.asarray([c[2] for c in sub], jnp.float64)
        sr = jnp.asarray([c[3] for c in sub], jnp.float64)
        src = jnp.asarray([c[4] for c in sub], jnp.int32)
        dst = jnp.asarray([c[5] for c in sub], jnp.int32)
        out = batched(spec, sb, sr, src, dst, g=g)
        for j, i in enumerate(idx):
            geom, sp, a_sb, a_sr, a_src, a_dst = cases[i]
            want = _v5(geom, sp, a_sb, a_sr, a_src, a_dst)
            for f, a, col in zip(FIELDS, want, out):
                assert col.dtype == jnp.float64
                bv = float(col[j])
                if not (a == bv):
                    bad.append((i, f, a, bv))
    assert 1 <= n_groups <= 10
    assert not bad, _mismatch_report("jit+vmap", bad)


# ───────────────────────────────────────────────── ③ 같은 기하로 (K,N) vmap 한 번 — 엔진이 부르는 모양
def test_move_container_single_geom_batch_matches_eager():
    """엔진은 기하 하나에 크레인×오더로 vmap 한다. 그 모양에서도 한 건씩 돈 것과 같은 비트."""
    rng = random.Random(7)
    geom = BlockGeometry(block_id="B", bay_count=24, row_count=10, tier_max=6,
                         bay_length_m=6.5, row_width_m=2.9, tier_height_m=2.9)
    g = TravelGeom.from_block_geometry(geom)
    K, N = 2, 16
    specs = [CraneSpec(crane_id=f"YC{k}", service_bay_min=1, service_bay_max=24,
                       gantry_speed_mps=rng.uniform(0.5, 3.0), trolley_speed_mps=rng.uniform(0.5, 2.0),
                       hoist_speed_loaded_mps=rng.uniform(0.3, 1.0), hoist_speed_empty_mps=rng.uniform(0.5, 1.5),
                       lock_time_s=rng.uniform(5, 30), unlock_time_s=rng.uniform(5, 30),
                       truck_positioning_time_s=rng.uniform(10, 120)) for k in range(K)]
    sb = [rng.uniform(1, 24) for _ in range(K)]
    sr = [rng.uniform(0, 10) for _ in range(K)]
    src = [(rng.randint(1, 24), rng.randint(0, 10), rng.randint(1, 6)) for _ in range(N)]
    dst = [(rng.randint(1, 24), rng.randint(0, 10), rng.randint(1, 6)) for _ in range(N)]

    spec_k = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[spec7_from_crane_spec(s) for s in specs])
    f_kn = jax.jit(jax.vmap(jax.vmap(lambda s, a, b, x, y: move_container(s, a, b, x, y, g),
                                     in_axes=(None, None, None, 0, 0)),
                            in_axes=(0, 0, 0, None, None)))
    out = f_kn(spec_k, jnp.asarray(sb, jnp.float64), jnp.asarray(sr, jnp.float64),
               jnp.asarray(src, jnp.int32), jnp.asarray(dst, jnp.int32))
    assert out.dur.shape == (K, N)
    bad = []
    for k in range(K):
        for n in range(N):
            want = _v5(geom, specs[k], sb[k], sr[k], src[n], dst[n])
            for f, a, col in zip(FIELDS, want, out):
                if not (a == float(col[k, n])):
                    bad.append((k * N + n, f, a, float(col[k, n])))
    assert not bad, _mismatch_report("(K,N) vmap", bad)


# ───────────────────────────────────────────────── ④ estimate_reach_s (덤)
def test_estimate_reach_bitwise_equal(cases):
    bad = []
    for i, (geom, spec, sb, sr, src, dst) in enumerate(cases[:100]):
        want = v5.estimate_reach_s(spec, geom, sb, sr, float(dst[0]), float(dst[1]))
        got = estimate_reach_s(spec7_from_crane_spec(spec), TravelGeom.from_block_geometry(geom),
                               jnp.float64(sb), jnp.float64(sr), jnp.float64(dst[0]), jnp.float64(dst[1]))
        if not (want == float(got)):
            bad.append((i, "reach", want, float(got)))
    assert not bad, _mismatch_report("reach", bad)


# ───────────────────────────────────────────────── ⑤ 공용 Geom (gpu/geom.py) 으로도 같은 비트
def test_move_container_with_shared_geom(cases):
    """엔진은 TravelGeom 이 아니라 공용 Geom 을 넘긴다 — 속성 이름이 맞는지 실제로 확인."""
    geom_mod = pytest.importorskip("yard_rl.v6.gpu.geom")
    bad = []
    for i, (geom, spec, sb, sr, src, dst) in enumerate(cases[:50]):
        g = geom_mod.Geom(bay_count=geom.bay_count, row_count=geom.row_count, tier_max=geom.tier_max,
                          bay_len=geom.bay_length_m, row_w=geom.row_width_m, tier_h=geom.tier_height_m,
                          transfer_row=0.0, sla_s=1800.0, shift_len_s=28800.0, gap=1.0, n_lanes=1)
        want = _v5(geom, spec, sb, sr, src, dst)
        got = move_container(spec7_from_crane_spec(spec), sb, sr,
                             jnp.asarray(src, jnp.int32), jnp.asarray(dst, jnp.int32), g)
        for f, a, b in zip(FIELDS, want, got):
            if not (a == float(b)):
                bad.append((i, f, a, float(b)))
    assert not bad, _mismatch_report("shared Geom", bad)
