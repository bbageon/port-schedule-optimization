"""배열 스택(gpu/stack_ops.py) 이 v5 `YardStacks` 와 **같은 답**을 내는가 ([[YR-327]] 조각 1).

■ 무엇을 지켜야 하나
  ① find_slot — 무작위 야드 5종 × 질의 300건에서 v5 와 (bay,row)|None **정확 일치**
     (불일치가 나면 v5 최선·차선 비용차 < 1e-9 인 '근접 동률' 을 따로 세어 보고한다)
  ② 배치(vmap)가 하나씩 부른 것과 같은 답
  ③ blockers_above · rehandle_capacity_ok — 같은 야드에서 100건 대조
  ④ place/remove — v5 와 같은 순서로 20회 적용한 뒤 격자·높이·맨위규격·좌표 일치
  ⑤ v5 가 예외를 내는 자리에서 위반 비트가 켜지고 상태는 안 변한다

기대값은 손으로 적지 않는다 — **v5 를 실제로 불러** 같은 입력의 답을 받는다.
x64 CPU 에서 돌린다. ★FMA(곱셈-덧셈 융합) 방어는 XLA_FLAGS 가 아니라 find_slot 안의 `exact.mul_exact`
(optimization_barrier) 다 — 플래그는 실측으로 무효 (gpu/exact.py 머리말).
"""
from __future__ import annotations

import importlib.util
import random
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★좌표·비용은 float64 — stack_ops.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu.stack_ops import (SIZE_INDEX, V_NOT_TOP, V_SIZE, V_TIER, Geom,  # noqa: E402
                                      blockers_above, find_slot, from_v5_stacks, place,
                                      rehandle_capacity_ok, remove, to_v5_piles)
from yard_rl.v6.world.domain.enums import ContainerSize, LoadStatus  # noqa: E402
from yard_rl.v6.world.domain.models import Container  # noqa: E402


def _load_yard_builder():
    """옆 파일 `test_find_slot_equiv.py` 의 `_yard(seed, fill)` — 경로로 읽는다 (패키지 아님)."""
    p = Path(__file__).with_name("test_find_slot_equiv.py")
    spec = importlib.util.spec_from_file_location("_v6_find_slot_equiv", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._yard


_yard = _load_yard_builder()
FILLS = [0.0, 0.30, 0.45, 0.75, 0.95]


def _setup(fill: float, seed_base: int = 7, *, n_reserve: int = 0):
    prof, stk = _yard(seed=seed_base + int(fill * 100), fill=fill)
    g = Geom.from_profile(prof)          # 공용 기하 (gpu/geom.py) — bay_len·row_w·tier_h
    ids = sorted(stk.containers)
    stacks, conts, ids = from_v5_stacks(stk, g, cont_ids=ids, n_cont=len(ids) + n_reserve)
    return prof, stk, g, stacks, conts, ids


def _excl_mask(g, excl):
    m = np.zeros((g.bay_count, g.row_count), bool)
    for b, r in excl:
        m[b - 1, r - 1] = True
    return m


def _v5_sorted_keys(stk, geom, size, spec, nb, nr, excl):
    """진단용 — v5 와 같은 식으로 모든 합법 칸의 (cost, bay, row) 를 나열해 정렬한다.
    기대값이 아니라 **불일치의 원인(근접 동률)** 을 가려내는 데만 쓴다."""
    keys = []
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        gc = abs(nb - bay) * geom.bay_length_m
        for row in range(1, geom.row_count + 1):
            if (bay, row) in excl:
                continue
            pile = stk._stacks.get((bay, row))
            top = len(pile) if pile else 0
            if top >= geom.tier_max:
                continue
            if pile and stk.containers[pile[-1]].size != size:
                continue
            keys.append((gc + abs(nr - row) * geom.row_width_m + top * geom.tier_height_m,
                         bay, row))
    keys.sort()
    return keys


def _queries(rng, g, n, *, max_excl=10):
    out = []
    for _ in range(n):
        size = rng.choice(list(ContainerSize))
        nb, nr = rng.uniform(0, 25), rng.uniform(0, 11)
        k = rng.randint(0, max_excl)
        excl = frozenset((rng.randint(1, g.bay_count), rng.randint(1, g.row_count))
                         for _ in range(k))
        out.append((size, nb, nr, excl))
    return out


def _int_queries(rng, g, n, *, max_excl=10):
    """엔진이 실제로 넣는 기준점 — blocker 자기 좌표(정수)·크레인 위치(정수 또는 반정수)·
    transfer_row(0). 연속 균등분포와 달리 **정확 동률**(좌우 대칭 칸)이 흔해 tie-break 를
    실제로 시험한다."""
    out = []
    for _ in range(n):
        size = rng.choice(list(ContainerSize))
        nb = float(rng.randint(1, g.bay_count)) + rng.choice((0.0, 0.0, 0.5))
        nr = float(rng.choice([g.transfer_row] + list(range(1, g.row_count + 1))))
        k = rng.randint(0, max_excl)
        excl = frozenset((rng.randint(1, g.bay_count), rng.randint(1, g.row_count))
                         for _ in range(k))
        out.append((size, nb, nr, excl))
    return out


def _count_exact_ties(stk, geom, spec, queries):
    n = 0
    for size, nb, nr, ex in queries:
        keys = _v5_sorted_keys(stk, geom, size, spec, nb, nr, ex)
        n += int(len(keys) >= 2 and keys[0][0] == keys[1][0])
    return n


def _batched_find_slot(g):
    f = partial(find_slot, g=g)
    return jax.jit(jax.vmap(f, in_axes=(None, None, 0, 0, None, None, 0, 0)))


def _compare_find_slot(stk, geom, g, stacks, spec, queries):
    """v5 와 배열판을 대조. (불일치 목록, 근접 동률 수) 를 돌려준다."""
    excl = jnp.asarray(np.stack([_excl_mask(g, q[3]) for q in queries]))
    sizes = jnp.asarray([SIZE_INDEX[q[0].value] for q in queries], jnp.int32)
    nbs = jnp.asarray([q[1] for q in queries], jnp.float64)
    nrs = jnp.asarray([q[2] for q in queries], jnp.float64)
    found, bay, row = _batched_find_slot(g)(
        stacks.height, stacks.top_size, excl, sizes,
        jnp.int32(spec.service_bay_min), jnp.int32(spec.service_bay_max), nbs, nrs)
    found, bay, row = np.asarray(found), np.asarray(bay), np.asarray(row)
    mism, near = [], 0
    for i, (size, nb, nr, ex) in enumerate(queries):
        want = stk.find_slot(size, spec, nb, nr, exclude=ex)
        got = (int(bay[i]), int(row[i])) if bool(found[i]) else None
        if got != want:
            keys = _v5_sorted_keys(stk, geom, size, spec, nb, nr, ex)
            tie = len(keys) >= 2 and (keys[1][0] - keys[0][0]) < 1e-9
            near += int(tie)
            mism.append((i, size.value, round(nb, 3), round(nr, 3), want, got,
                         "근접동률" if tie else "규칙차이", keys[:2]))
    return mism, near


# ───────────────────────────────────────────────── ① find_slot
@pytest.mark.parametrize("fill", FILLS)
def test_find_slot_matches_v5(fill):
    """★야드 5종 × 질의 300건 — (bay,row)|None 이 v5 와 **정확히** 같다."""
    prof, stk, g, stacks, conts, ids = _setup(fill)
    spec = prof.cranes[0]
    rng = random.Random(1000 + int(fill * 100))
    queries = _queries(rng, g, 300)
    mism, near = _compare_find_slot(stk, prof.block, g, stacks, spec, queries)
    n_none = sum(1 for q in queries if stk.find_slot(q[0], spec, q[1], q[2], exclude=q[3]) is None)
    print(f"\n[find_slot fill={fill}] 질의 300 · v5 None {n_none} · 불일치 {len(mism)} "
          f"(근접동률 {near})")
    assert not mism, (f"fill={fill}: 불일치 {len(mism)}건 (근접 동률 {near}건 포함)\n"
                      + "\n".join(str(m) for m in mism[:10]))


@pytest.mark.parametrize("fill", [0.30, 0.75, 0.95])
def test_find_slot_integer_reference_points(fill):
    """★정수·반정수 기준점 300건 — 정확 동률이 많아 (cost, bay, row) 사전식 tie-break 와
    비용의 비트 동일성이 실제로 시험된다 (연속 기준점 시험은 동률이 0건이라 못 본다)."""
    prof, stk, g, stacks, conts, ids = _setup(fill, seed_base=41)
    spec = prof.cranes[0]
    rng = random.Random(1500 + int(fill * 100))
    queries = _int_queries(rng, g, 300)
    ties = _count_exact_ties(stk, prof.block, spec, queries)
    mism, near = _compare_find_slot(stk, prof.block, g, stacks, spec, queries)
    print(f"\n[find_slot 정수기준 fill={fill}] 질의 300 · 정확동률 {ties} · 불일치 {len(mism)} "
          f"(근접동률 {near})")
    assert ties > 0, "동률이 없으면 이 시험은 tie-break 를 못 본다"
    assert not mism, (f"fill={fill}: 불일치 {len(mism)}건 (근접 동률 {near}건 포함)\n"
                      + "\n".join(str(m) for m in mism[:10]))


@pytest.mark.parametrize("fill", [0.45, 0.95])
def test_find_slot_with_random_service_window(fill):
    """담당 구간(bay_min..bay_max)이 좁을 때도 v5 와 같다 — H21 스펙은 1..24 라 따로 본다."""
    prof, stk, g, stacks, conts, ids = _setup(fill, seed_base=31)
    rng = random.Random(2000 + int(fill * 100))
    mism_all, near_all, n_none = [], 0, 0
    for _ in range(40):
        lo = rng.randint(1, g.bay_count)
        hi = rng.randint(lo, g.bay_count)
        spec = replace(prof.cranes[0], service_bay_min=lo, service_bay_max=hi)
        queries = _queries(rng, g, 5)
        mism, near = _compare_find_slot(stk, prof.block, g, stacks, spec, queries)
        n_none += sum(1 for q in queries
                      if stk.find_slot(q[0], spec, q[1], q[2], exclude=q[3]) is None)
        mism_all += mism
        near_all += near
    print(f"\n[find_slot 창 fill={fill}] 질의 200 · v5 None {n_none} · 불일치 {len(mism_all)} "
          f"(근접동률 {near_all})")
    assert not mism_all, f"불일치 {len(mism_all)}건 (근접 동률 {near_all})\n{mism_all[:10]}"


# ───────────────────────────────────────────────── ② 배치 = 낱개
def test_batch_matches_one_at_a_time():
    """★vmap 으로 한 번에 푼 답이 하나씩 부른 답과 같다 — 다르면 GPU 결과를 못 믿는다."""
    prof, stk, g, stacks, conts, ids = _setup(0.75)
    spec = prof.cranes[0]
    rng = random.Random(3)
    queries = _queries(rng, g, 40)
    excl = jnp.asarray(np.stack([_excl_mask(g, q[3]) for q in queries]))
    sizes = jnp.asarray([SIZE_INDEX[q[0].value] for q in queries], jnp.int32)
    nbs = jnp.asarray([q[1] for q in queries], jnp.float64)
    nrs = jnp.asarray([q[2] for q in queries], jnp.float64)
    lo, hi = jnp.int32(spec.service_bay_min), jnp.int32(spec.service_bay_max)
    fb, bb, rb = _batched_find_slot(g)(stacks.height, stacks.top_size, excl, sizes, lo, hi,
                                      nbs, nrs)
    one = jax.jit(find_slot, static_argnames="g")
    for i in range(len(queries)):
        f1, b1, r1 = one(stacks.height, stacks.top_size, excl[i], sizes[i], lo, hi,
                         nbs[i], nrs[i], g=g)
        assert (bool(fb[i]), int(bb[i]), int(rb[i])) == (bool(f1), int(b1), int(r1))


def test_find_slot_none_when_all_excluded():
    """합법 칸이 없으면 (False, -1, -1) — v5 는 None."""
    prof, stk, g, stacks, conts, ids = _setup(0.3)
    spec = prof.cranes[0]
    every = frozenset((b, r) for b in range(1, g.bay_count + 1) for r in range(1, g.row_count + 1))
    assert stk.find_slot(ContainerSize.FT40, spec, 3.0, 2.0, exclude=every) is None
    f, b, r = find_slot(stacks.height, stacks.top_size, jnp.ones_like(stacks.height, dtype=bool),
                        SIZE_INDEX["FT40"], spec.service_bay_min, spec.service_bay_max,
                        3.0, 2.0, g)
    assert (bool(f), int(b), int(r)) == (False, -1, -1)


# ───────────────────────────────────────────────── ③ blocker · 재조작 용량
@pytest.mark.parametrize("fill", [0.30, 0.45, 0.75, 0.95])
def test_blockers_and_capacity_match_v5(fill):
    """★같은 야드에서 100건 — blocker 목록(위→아래)·개수·재조작 가능 판정이 v5 와 같다."""
    prof, stk, g, stacks, conts, ids = _setup(fill, seed_base=11)
    idx = {cid: i for i, cid in enumerate(ids)}
    rng = random.Random(4000 + int(fill * 100))
    jb = jax.jit(blockers_above, static_argnames="g")
    jc = jax.jit(rehandle_capacity_ok, static_argnames="g")
    n_false = n_with_blockers = 0
    for _ in range(100):
        cid = rng.choice(ids)
        c = idx[cid]
        want = stk.blockers_above(cid)
        blk, n_block = jb(stacks.grid, stacks.height, conts.c_bay, conts.c_row, conts.c_tier,
                          jnp.int32(c), g=g)
        got = [ids[int(x)] for x in np.asarray(blk)[: int(n_block)]]
        assert got == want, f"{cid}: blocker 다름 v5 {want} vs 배열 {got}"
        assert all(int(x) == -1 for x in np.asarray(blk)[int(n_block):]), "무효 칸은 -1"
        n_with_blockers += int(bool(want))
        # 담당 구간: 전체(H21 1..24) 와 무작위 좁은 창 — 좁은 창에서 False 가 실제로 나온다
        cont = stk.containers[cid]
        windows = [(prof.cranes[0].service_bay_min, prof.cranes[0].service_bay_max)]
        lo = rng.randint(1, g.bay_count)
        windows.append((lo, rng.randint(lo, min(g.bay_count, lo + 1))))
        windows.append((cont.bay, cont.bay))
        for lo, hi in windows:
            spec = replace(prof.cranes[0], service_bay_min=lo, service_bay_max=hi)
            want_ok = stk.rehandle_capacity_ok(cid, spec)
            got_ok = bool(jc(stacks.height, stacks.top_size, jnp.int32(cont.bay),
                             jnp.int32(cont.row), n_block, jnp.int32(lo), jnp.int32(hi), g=g))
            assert got_ok == want_ok, f"{cid} 창 {lo}..{hi}: v5 {want_ok} vs 배열 {got_ok}"
            n_false += int(not want_ok)
    print(f"\n[blocker fill={fill}] 100건 · blocker 있음 {n_with_blockers} · 용량 판정 300건 중 "
          f"False {n_false}")


# ───────────────────────────────────────────────── ④ place / remove
def _assert_same_yard(stk, stacks, conts, ids, tag):
    v5_piles = {k: list(v) for k, v in stk._stacks.items() if v}
    assert to_v5_piles(stacks, ids) == v5_piles, f"{tag}: 격자가 다르다"
    height = np.asarray(stacks.height)
    top_size = np.asarray(stacks.top_size)
    for (b, r), pile in v5_piles.items():
        assert int(height[b - 1, r - 1]) == len(pile)
        assert int(top_size[b - 1, r - 1]) == SIZE_INDEX[stk.containers[pile[-1]].size.value]
    assert int(height.sum()) == sum(len(p) for p in v5_piles.values())
    assert int((top_size >= 0).sum()) == len(v5_piles), f"{tag}: 빈 칸의 맨위규격이 -1 이 아니다"
    c_bay, c_row, c_tier = (np.asarray(conts.c_bay), np.asarray(conts.c_row),
                            np.asarray(conts.c_tier))
    alive = np.asarray(conts.c_alive)
    for i, cid in enumerate(ids):
        cont = stk.containers.get(cid)
        if cont is None:
            assert not alive[i] and c_bay[i] == -1, f"{tag}: 없는 {cid} 가 살아 있다"
        else:
            assert alive[i] and (c_bay[i], c_row[i], c_tier[i]) == (cont.bay, cont.row, cont.tier), \
                f"{tag}: {cid} 좌표 다름"


@pytest.mark.parametrize("fill", [0.45, 0.95])
def test_place_remove_sequence_matches_v5(fill):
    """★v5 와 같은 순서로 20회 (재배치 remove→place · 반출 remove · 반입 place) 적용 → 격자 일치."""
    n_in = 8
    prof, stk, g, stacks, conts, ids = _setup(fill, seed_base=17, n_reserve=n_in)
    spec = prof.cranes[0]
    idx = {cid: i for i, cid in enumerate(ids)}
    rng = random.Random(5000 + int(fill * 100))
    jp = jax.jit(place)
    jr = jax.jit(remove)
    fs = jax.jit(find_slot, static_argnames="g")
    # 반입 예비칸 규격을 미리 정한다 (호스트가 reset 에 하는 일)
    in_sizes = [rng.choice(list(ContainerSize)) for _ in range(n_in)]
    conts = conts._replace(c_size=conts.c_size.at[len(ids):].set(
        jnp.asarray([SIZE_INDEX[s.value] for s in in_sizes], jnp.int32)))
    ids = ids + [f"IN_{i}" for i in range(n_in)]
    n_inbound = 0
    ops = []
    for step in range(20):
        kind = step % 3                                  # 0 재배치 · 1 반출 · 2 반입
        if kind in (0, 1):
            tops = [p[-1] for p in stk._stacks.values() if p]
            cid = rng.choice(tops)
            cont = stk.containers[cid]
            src = (cont.bay, cont.row)
            stk.remove(cid)
            stacks, conts, v = jr(stacks, conts, jnp.int32(idx[cid]))
            assert int(v) == 0, f"step {step} remove {cid}: 위반 {int(v)}"
            _assert_same_yard(stk, stacks, conts, ids, f"step {step} remove")
            if kind == 0:
                dest = stk.find_slot(cont.size, spec, float(src[0]), float(src[1]),
                                     exclude=frozenset({src}))
                excl = jnp.asarray(_excl_mask(g, {src}))
                f, b, r = fs(stacks.height, stacks.top_size, excl, SIZE_INDEX[cont.size.value],
                             spec.service_bay_min, spec.service_bay_max,
                             float(src[0]), float(src[1]), g=g)
                assert bool(f) and (int(b), int(r)) == dest, f"step {step}: 재배치 칸 다름"
                stk.place(cont, dest[0], dest[1])
                stacks, conts, v = jp(stacks, conts, jnp.int32(idx[cid]), b, r)
                assert int(v) == 0
                ops.append(("rehandle", cid, src, dest))
            else:
                ops.append(("depart", cid, src))
        else:
            i = n_inbound
            n_inbound += 1
            size = in_sizes[i]
            cont = Container(container_id=f"IN_{i}", size=size, load_status=LoadStatus.FULL,
                             block=prof.block.block_id, bay=0, row=0, tier=0)
            nb, nr = rng.uniform(1, g.bay_count), float(g.transfer_row)
            dest = stk.find_slot(size, spec, nb, nr)
            assert dest is not None
            stk.place(cont, dest[0], dest[1])
            stacks, conts, v = jp(stacks, conts, jnp.int32(len(idx) + i),
                                  jnp.int32(dest[0]), jnp.int32(dest[1]))
            assert int(v) == 0
            ops.append(("inbound", cont.container_id, dest))
        _assert_same_yard(stk, stacks, conts, ids, f"step {step} {ops[-1][0]}")
    print(f"\n[place/remove fill={fill}] 20회 적용 · 종류 " +
          ", ".join(f"{k}:{sum(1 for o in ops if o[0] == k)}"
                    for k in ("rehandle", "depart", "inbound")) + " · 격자 일치")


# ───────────────────────────────────────────────── ⑤ 위반 비트
def test_violations_flag_and_keep_state():
    """v5 가 `raise` 하는 자리 — 비트를 켜고 **상태는 그대로** 둔다."""
    prof, stk, g, stacks, conts, ids = _setup(0.95, seed_base=23, n_reserve=2)
    idx = {cid: i for i, cid in enumerate(ids)}
    height = np.asarray(stacks.height)
    top_size = np.asarray(stacks.top_size)
    full = [(b, r) for b in range(g.bay_count) for r in range(g.row_count)
            if height[b, r] >= g.tier_max]
    assert full, "만재 칸이 있어야 tier 위반을 볼 수 있다"
    fb, fr = full[0]
    same_size = int(top_size[fb, fr])
    # 예비칸 두 개: 하나는 만재 칸과 같은 규격, 하나는 다른 규격
    c_same, c_other = len(ids), len(ids) + 1
    conts = conts._replace(c_size=conts.c_size.at[c_same].set(same_size)
                           .at[c_other].set((same_size + 1) % 3))

    def unchanged(s2, c2):
        return all(bool(jnp.array_equal(a, b)) for a, b in zip(s2, stacks)) and \
            all(bool(jnp.array_equal(a, b)) for a, b in zip(c2, conts))

    # TIER — 만재 칸에 쌓기 (v5: RuntimeError 'tier 초과')
    with pytest.raises(RuntimeError):
        stk.place(Container("X", ContainerSize(list(SIZE_INDEX)[same_size]), LoadStatus.FULL,
                            "B1", 0, 0, 0), fb + 1, fr + 1)
    s2, c2, v = place(stacks, conts, c_same, fb + 1, fr + 1)
    assert int(v) == V_TIER and unchanged(s2, c2)
    # SIZE — 다른 규격 위에 쌓기 (tier 여유 있는 비만재 칸)
    part = [(b, r) for b in range(g.bay_count) for r in range(g.row_count)
            if 0 < height[b, r] < g.tier_max]
    pb, pr = part[0]
    other = (int(top_size[pb, pr]) + 1) % 3
    conts = conts._replace(c_size=conts.c_size.at[c_other].set(other))
    with pytest.raises(RuntimeError):
        stk.place(Container("Y", ContainerSize(list(SIZE_INDEX)[other]), LoadStatus.FULL,
                            "B1", 0, 0, 0), pb + 1, pr + 1)
    s2, c2, v = place(stacks, conts, c_other, pb + 1, pr + 1)
    assert int(v) == V_SIZE and unchanged(s2, c2)
    # NOT_TOP — 맨 위가 아닌 컨테이너 제거
    deep = [p[0] for p in stk._stacks.values() if len(p) >= 2]
    with pytest.raises(RuntimeError):
        stk.remove(deep[0])
    s2, c2, v = remove(stacks, conts, idx[deep[0]])
    assert int(v) == V_NOT_TOP and unchanged(s2, c2)
    # 무효 번호 — 야드에 없는 예비칸 제거 · 번호 -1
    s2, c2, v = remove(stacks, conts, c_same)
    assert int(v) == V_NOT_TOP and unchanged(s2, c2)
    s2, c2, v = remove(stacks, conts, -1)
    assert int(v) == V_NOT_TOP and unchanged(s2, c2)
    s2, c2, v = place(stacks, conts, -1, pb + 1, pr + 1)
    assert int(v) == V_NOT_TOP and unchanged(s2, c2)
    # 정상 경로는 0
    top_cid = next(p[-1] for p in stk._stacks.values() if p)
    s2, c2, v = remove(stacks, conts, idx[top_cid])
    assert int(v) == 0 and not unchanged(s2, c2)
