"""배열 예약표 ↔ v5 `ReservationTable` 동등성 ([[YR-327]] 조각 1 · key=reserve).

■ 무엇을 지켜야 하나
  ① 거절 코드가 v5 `reject_reason` 과 **정확히 같다** — 5-lock 순서(DOUBLE→DUP→LANE→INTERF→SLOT)
     까지. K∈{1,2,3} · 무작위 예약 상태 200건 × 질의 여러 건.
  ② reserve / release / set_idle_position 을 같은 순서로 적용하면 표 내용이 v5 와 같다.
  ③ 오더 축 vmap 판이 하나씩 부른 것과 같고, jit 아래서도 같다.
  기대값은 손으로 적지 않는다 — **v5 를 실제로 불러** 받은 답과 비교한다.
"""
from __future__ import annotations

import random

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu.reserve import (REASON_TO_CODE, CODE_TO_REASON, ReservationArrays,
                                    can_reserve, empty_reservations, lane_occupancy,
                                    orphan_count, reject_code, reject_code_over_orders,
                                    release, reserve, reserved_slots, set_idle_position)
from yard_rl.v6.world.integrated.reservation import (Corridor, Reservation,
                                                     ReservationTable)
from yard_rl.v6.world.sim.constraints import ConstraintViolation

B, R = 6, 3            # 격자
N = 6                  # 오더(토큰) 칸
LANES = ("L1", "L2")   # 레인 번호 0,1 ; None = -1
GAP = 2.0              # safety_gap_bay (fixture 값)


# ───────────────────────────────────────────────── v5 ↔ 배열 변환
def _cids(k):
    return tuple(f"YC-{chr(65 + i)}" for i in range(k))     # 문자열 정렬 = 번호 순


def _to_arrays(table: ReservationTable, cids) -> ReservationArrays:
    """v5 표의 내부 dict 를 배열로. 번호: 크레인 = cids 순위, 토큰 'J{n}' → n, 레인 = LANES 순위."""
    res = empty_reservations(len(cids), N, B, R)
    for k, cid in enumerate(cids):
        r = table._by_crane.get(cid)
        if r is not None:
            slots = jnp.zeros((B, R), bool)
            for (b, rr) in r.slots:
                slots = slots.at[b - 1, rr - 1].set(True)
            res = res._replace(
                active=res.active.at[k].set(True),
                token=res.token.at[k].set(-1 if r.job_token is None else int(r.job_token[1:])),
                lo=res.lo.at[k].set(r.corridor.lo), hi=res.hi.at[k].set(r.corridor.hi),
                lane=res.lane.at[k].set(-1 if r.lane_id is None else LANES.index(r.lane_id)),
                release_at=res.release_at.at[k].set(r.release_at),
                slots=res.slots.at[k].set(slots))
        if cid in table._idle_pos:
            res = res._replace(idle_pos=res.idle_pos.at[k].set(table._idle_pos[cid]))
    for tok, cid in table._tokens.items():
        res = res._replace(token_owner=res.token_owner.at[int(tok[1:])].set(cids.index(cid)))
    return res


def _snapshot(res: ReservationArrays):
    """비교용 파이썬 값 — 비활성 행은 active/token/lane/slots 만 본다(lo/hi 는 읽지 않는 칸)."""
    out = []
    for k in range(res.k):
        a = bool(res.active[k])
        row = (a, int(res.token[k]), int(res.lane[k]),
               sorted((b + 1, r + 1) for b in range(B) for r in range(R) if bool(res.slots[k, b, r])))
        if a:
            row += (float(res.lo[k]), float(res.hi[k]), float(res.release_at[k]))
        out.append(row)
    return (out, [int(x) for x in res.token_owner], [float(x) for x in res.idle_pos])


# ───────────────────────────────────────────────── 무작위 생성
def _rand_query(rng: random.Random, k_count: int):
    """정수 bay 로 통로를 뽑아 `hi + gap == lo` 같은 경계 동률이 자주 나오게 한다."""
    lo = rng.randint(1, B)
    hi = rng.randint(lo, B)
    if rng.random() < 0.3:                       # 소수 좌표도 섞는다
        lo, hi = lo - rng.random(), hi + rng.random()
    token = rng.choice([None] + [f"J{i}" for i in range(N)])
    lane = rng.choice([None] + list(LANES))
    n_slots = rng.randint(0, 4)
    slots = frozenset((rng.randint(1, B), rng.randint(1, R)) for _ in range(n_slots))
    return dict(crane=rng.randrange(k_count), token=token, lo=float(lo), hi=float(hi),
                lane=lane, slots=slots, release_at=float(rng.randint(0, 5000)))


def _v5_res(cids, q) -> Reservation:
    return Reservation(crane_id=cids[q["crane"]], job_token=q["token"],
                       corridor=Corridor(q["lo"], q["hi"]), slots=q["slots"],
                       lane_id=q["lane"], release_at=q["release_at"])


def _arr_args(q):
    slots = jnp.zeros((B, R), bool)
    for (b, r) in q["slots"]:
        slots = slots.at[b - 1, r - 1].set(True)
    tok = -1 if q["token"] is None else int(q["token"][1:])
    lane = -1 if q["lane"] is None else LANES.index(q["lane"])
    return dict(k=q["crane"], token=tok, lo=q["lo"], hi=q["hi"], lane=lane, slots=slots)


def _rand_table(rng: random.Random, cids, gap: float = GAP) -> ReservationTable:
    """무작위 예약 상태 — v5 표에 실제로 reserve 해서 만든다(거절되면 그 크레인은 빈 채로)."""
    table = ReservationTable(gap)
    for k, cid in enumerate(cids):
        if rng.random() < 0.85:                  # 대부분 등록, 일부는 미등록(+inf) 도 본다
            table.set_idle_position(cid, float(rng.randint(1, B)) + rng.choice([0.0, 0.5]))
    for k in rng.sample(range(len(cids)), len(cids)):
        if rng.random() < 0.55:
            q = _rand_query(rng, len(cids))
            q["crane"] = k
            try:
                table.reserve(_v5_res(cids, q))
            except ConstraintViolation:
                pass
    return table


# ───────────────────────────────────────────────── ① 거절 코드 정확 일치
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("gap", [GAP, 0.0])      # gap 0 이면 간섭이 줄어 LANE·SLOT 사례가 는다
def test_reject_code_matches_v5(K, gap):
    """★무작위 예약 상태 200건 × 질의 5건 — v5 `reject_reason` 과 코드 **정확 일치**."""
    rng = random.Random(1000 + K)
    cids = _cids(K)
    fn = jax.jit(reject_code)
    mismatches, counts, boundary = [], {c: 0 for c in range(6)}, 0
    for s in range(200):
        table = _rand_table(rng, cids, gap)
        res = _to_arrays(table, cids)
        for _ in range(5):
            q = _rand_query(rng, K)
            want = REASON_TO_CODE[table.reject_reason(_v5_res(cids, q))]
            got = int(fn(res, gap=gap, **_arr_args(q)))
            counts[want] += 1
            # 경계 동률: 어떤 활성 통로·장벽과 정확히 gap 만큼 떨어진 경우 (<= 의 등호가 갈리는 곳)
            for cid, r in table._by_crane.items():
                if r.corridor.hi + gap == q["lo"] or q["hi"] + gap == r.corridor.lo:
                    boundary += 1
            for cid, p in table._idle_pos.items():
                if cid not in table._by_crane and (p + gap == q["lo"] or q["hi"] + gap == p):
                    boundary += 1
            if got != want:
                mismatches.append((s, q, CODE_TO_REASON[want], CODE_TO_REASON[got]))
    assert not mismatches, (f"K={K} gap={gap}: 불일치 {len(mismatches)}건 / 1000 (경계 동률 {boundary}건)\n"
                            + "\n".join(map(str, mismatches[:5])))
    if K >= 2:      # 다섯 코드가 전부 실제로 나왔는지 (아니면 시험이 헛돈다)
        assert all(counts[c] > 0 for c in range(6)), f"K={K}: 코드 분포 {counts}"
    print(f"K={K} gap={gap} 코드 분포 {counts} 경계 동률 {boundary}")


def test_order_axis_vmap_matches_scalar():
    """③ 오더 축 vmap (K 크레인 × N 오더의 can_reserve 원천) 이 하나씩 부른 것과 같다."""
    rng = random.Random(7)
    cids = _cids(3)
    for _ in range(30):
        table = _rand_table(rng, cids)
        res = _to_arrays(table, cids)
        qs = [_rand_query(rng, 3) for _ in range(N)]
        k = rng.randrange(3)
        args = [_arr_args(q) for q in qs]
        stack = lambda key, dt: jnp.asarray([a[key] for a in args], dt)
        got = reject_code_over_orders(res, k, stack("token", jnp.int32), stack("lo", jnp.float64),
                                      stack("hi", jnp.float64), stack("lane", jnp.int32),
                                      jnp.stack([a["slots"] for a in args]), GAP)
        want = [int(reject_code(res, gap=GAP, **dict(a, k=k))) for a in args]
        assert [int(x) for x in got] == want


# ───────────────────────────────────────────────── ② 갱신 함수 동등성
@pytest.mark.parametrize("K", [1, 2, 3])
def test_reserve_release_sequence_matches_v5(K):
    """reserve / release / set_idle_position 을 같은 순서로 적용 → 표 내용이 v5 와 같다.

    거절도 같은 자리에서 나야 한다(v5 예외 ↔ 배열 코드≠0, 표는 불변).
    """
    rng = random.Random(500 + K)
    cids = _cids(K)
    n_ops, n_reject = 0, 0
    for s in range(60):
        table = ReservationTable(GAP)
        res = empty_reservations(K, N, B, R)
        for _ in range(25):
            op = rng.choice(["reserve", "reserve", "release", "idle"])
            if op == "reserve":
                q = _rand_query(rng, K)
                try:
                    table.reserve(_v5_res(cids, q))
                    want = 0
                except ConstraintViolation as e:
                    want = REASON_TO_CODE[e.code]
                res, code = reserve(res, gap=GAP, release_at=q["release_at"], **_arr_args(q))
                assert int(code) == want, f"s={s} op={q} v5={CODE_TO_REASON[want]} 배열={CODE_TO_REASON[int(code)]}"
                n_reject += want != 0
            elif op == "release":
                k = rng.randrange(K)
                table.release(cids[k])
                res = release(res, k)
            else:
                k, bay = rng.randrange(K), float(rng.randint(1, B))
                table.set_idle_position(cids[k], bay)
                res = set_idle_position(res, k, bay)
            n_ops += 1
            assert _snapshot(res) == _snapshot(_to_arrays(table, cids)), f"s={s} 뒤 표가 다르다"
            assert int(orphan_count(res)) == table.orphan_count()
            # 파생 읽기도 v5 와 같아야 한다
            want_slots = sorted(table.reserved_slots())
            got = reserved_slots(res)
            assert sorted((b + 1, r + 1) for b in range(B) for r in range(R) if bool(got[b, r])) == want_slots
            occ = frozenset(r.lane_id for r in table.active() if r.lane_id)
            assert [bool(x) for x in lane_occupancy(res, len(LANES))] == [l in occ for l in LANES]
    assert n_reject > 0, "거절 사례가 하나도 안 나왔다 — 시험이 헛돈다"
    print(f"K={K} 연산 {n_ops} 건 중 거절 {n_reject} 건")


def test_release_of_unreserved_crane_is_noop_and_negative_token_safe():
    """★예약 없는 크레인 release 는 무동작이고, 토큰 -1 이 마지막 오더 칸을 건드리지 않는다."""
    res = empty_reservations(2, N, B, R)
    res, code = reserve(res, 0, N - 1, 1.0, 2.0, -1, jnp.zeros((B, R), bool), 10.0, GAP)
    assert int(code) == 0 and int(res.token_owner[N - 1]) == 0
    res2 = release(res, 1)                       # 크레인 1 은 예약이 없다 (token -1)
    assert _snapshot(res2) == _snapshot(res), "무동작이어야 하는데 표가 바뀌었다"
    assert int(res2.token_owner[N - 1]) == 0, "토큰 -1 이 마지막 칸을 지웠다 (음수 색인)"
    res3 = release(res2, 0)
    assert int(res3.token_owner[N - 1]) == -1 and not bool(res3.active[0])
    assert bool(can_reserve(res3, 0, N - 1, 1.0, 2.0, -1, jnp.zeros((B, R), bool), GAP))


def test_dtypes_are_float64_and_jit_agrees():
    """좌표·시각은 float64 (x64) 이고 jit 판이 즉시실행과 같다."""
    res = empty_reservations(2, N, B, R)
    assert res.lo.dtype == jnp.float64 and res.release_at.dtype == jnp.float64
    assert res.idle_pos.dtype == jnp.float64
    res = set_idle_position(res, 1, 4.0)
    slots = jnp.zeros((B, R), bool)
    a = int(reject_code(res, 0, 0, 1.0, 2.0, 0, slots, GAP))          # 2+2 <= 4 → 겹침 아님
    b = int(jax.jit(reject_code)(res, 0, 0, 1.0, 2.0, 0, slots, GAP))
    assert a == b == 0
    assert int(reject_code(res, 0, 0, 1.0, 2.5, 0, slots, GAP)) == 4  # 2.5+2 > 4 → 간섭
