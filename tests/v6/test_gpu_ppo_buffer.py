"""배열 학습 버퍼·GAE (`gpu/ppo_buffer.py`) 가 v5 `ppo/buffer.py` 와 **같은 답**을 내는가 ([[YR-327]] 조각 8).

■ 무엇을 지키나 — 기대값은 손으로 적지 않는다. **v5 `Interval`·`gae` 를 실제로 불러** 대조한다.
  ① 포장(`pack_intervals`) → 복원(`unpack_choice`) 이 원본 `Choice` 여섯 칸과 `==` (비트 일치)
  ② 우위 `adv` 와 목표 `returns` 가 v5 와 `==` — **허용 오차 없음**. 갈리면 개수·|Δ|·처음 갈린 (구간, 블록)
     을 실패 메시지로 낸다
  ③ v5 가 `ValueError` 를 던지는 여덟 가지 입력에 대해 `raise_on_audit` 가 **같은 메시지**로 던진다
  ④ 넘침(칸 부족)은 조용히 버리지 않고 개수로 보고된다
  ⑤ jit == eager · vmap(세계 묶음) == 낱개 · 패딩 칸을 늘려도 답이 안 바뀐다

■ ★이 시험이 붙든 함정 — v5 의 `Interval.values` 는 **float32** 다
  `ppo/runtime.py:204` 이 torch 망의 출력을 담기 때문이다. 그래서 numpy 승격 규칙에 따라
  `delta = reward + discount*nxt − values` 가 **마지막 구간만 float64, 나머지는 float32** 로
  계산된다. `test_13_float32_branch_is_necessary` 가 그 반례를 실측한다 — 전부 float64 로
  계산하면 60/60 무대에서 갈리고 |Δ| 가 1.9e-07 까지 벌어진다.

■ 무대
  A  무작위 200조 — 구간 1~9개 · 블록 1·2·3·5·8 · 결정 0~3 · 후보 1~6 · dt 30/60/120초 ·
     종료 구간 섞음 · `values` **float32(실제 실행)·float64(시험이 손으로 만드는 것)** 양쪽
  B  실제 `PPORuntime` 테이프 — 경계 8회를 실제로 돌려 나온 `buffer` 를 그대로 포장해 대조
     (`test_ppo_continuous.py:18-45` 와 같은 대체 훅 방식 · 역할 셋 seller·buyer·crane 다 태운다)
  C  경계값 — 구간 0개 · 결정 0개 · 후보 1개 · 구간 1개(=마지막=부트스트랩) · 종료 · 넘침
  D  실측 보고 — `gamma**dt` 비트 일치 비율 · float32 갈래의 필요성

실행: Windows CPU x64 (`.venv-jax`) 또는 WSL venv. x64 가 꺼져 있으면 크게 실패한다.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — v5 `gae` 는 f64 배열이다
jnp = jax.numpy

from jax import lax                                                            # noqa: E402

from yard_rl.v6.gpu.exact import div_const, mul_exact                          # noqa: E402
from yard_rl.v6.gpu.ppo_buffer import (FEAT_DIM, ROLES, GaeAudit,              # noqa: E402
                                       IntervalOverflow, active_mask,
                                       block_samples, flat_entries, gae,
                                       gae_audit, gae_batch, micro_actions,
                                       pack_intervals, raise_on_audit,
                                       unpack_choice, valid_mask)
from yard_rl.v6.ppo.buffer import Choice, Interval                             # noqa: E402
from yard_rl.v6.ppo.buffer import gae as gae_v5                                # noqa: E402
from yard_rl.v6.ppo.model import BlockPolicy, encode                           # noqa: E402
from yard_rl.v6.ppo.runtime import PPOConfig, PPORuntime                       # noqa: E402

GAMMA, LAM, UNIT = 0.999, 0.95, 60.0          # PPOConfig 기본값 (runtime.py:31-33)
#: 무작위 무대에서 함께 도는 설정 — 감쇠 1.0(할인 없음)·λ 0·1 극단까지
CONFIGS = ((0.999, 0.95, 60.0), (1.0, 1.0, 60.0), (0.999, 0.0, 60.0),
           (0.95, 1.0, 30.0), (0.5, 0.5, 120.0))
#: 요약 보고에 쌓는 실측값 (`test_zz_report` 가 화면에 찍는다)
MEASURED: dict = {}


# ───────────────────────────────────────────────── 도우미
def _policy(seed: int = 17) -> BlockPolicy:
    torch.manual_seed(seed)
    return BlockPolicy()


def _state_rows(rng: random.Random, n_blocks: int) -> torch.Tensor:
    return encode([[rng.uniform(-2, 2) for _ in range(8)] for _ in range(n_blocks)], "state")


def _choice(rng: random.Random, t: float, role: str) -> Choice:
    """v5 `encode` 로 실제 (n,37) 후보 행렬을 만든다 — 역할별 규칙(buyer 는 2줄)을 지킨다."""
    n = 2 if role == "buyer" else rng.randint(1, 6)
    raw = [[rng.uniform(-3, 3) for _ in range(rng.randint(1, 12))] for _ in range(n)]
    width = max(len(r) for r in raw)
    rows = encode([r + [0.0] * (width - len(r)) for r in raw], role)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[rng.randrange(n)] = True                       # 적어도 하나는 열려 있어야 한다
    for i in range(n):
        mask[i] |= rng.random() < 0.7
    allowed = [i for i in range(n) if bool(mask[i])]
    return Choice(role, t, rows, mask, rng.choice(allowed), rng.uniform(-4.0, -1e-3))


def _rollout(rng: random.Random, policy: BlockPolicy, *, n: int, n_blocks: int,
             values_f32: bool, terminate: bool = False) -> list[Interval]:
    """v5 `Interval` 목록 하나 — 길이·블록·결정·후보 수를 섞는다."""
    out, t = [], float(rng.choice([0.0, 60.0, 3600.0]))
    for i in range(n):
        dt = float(rng.choice([60.0, 60.0, 60.0, 30.0, 120.0]))
        states = _state_rows(rng, n_blocks)
        if values_f32:                                  # ★실제 실행 경로 (runtime.py:204)
            with torch.no_grad():
                values = policy.value(states).numpy().copy()
        else:                                           # 시험이 손으로 만드는 경로
            values = np.array([rng.uniform(-5, 5) for _ in range(n_blocks)], dtype=np.float64)
        choices = [[_choice(rng, t, rng.choice(ROLES[:3])) for _ in range(rng.randint(0, 3))]
                   for _ in range(n_blocks)]
        term = terminate and i == n - 1
        out.append(Interval(t, t + dt, states, values, choices,
                            rng.uniform(-1.0, 1.0), terminated=term))
        t = t + dt
    return out


def _bootstrap(policy: BlockPolicy, rng: random.Random, n_blocks: int) -> np.ndarray:
    with torch.no_grad():
        return policy.value(_state_rows(rng, n_blocks)).numpy().copy()


def _compare(a5, r5, a6, r6, label: str) -> None:
    """`==` 를 먼저 — 갈리면 개수·|Δ|·**처음 갈린 지점**을 낸다."""
    a5 = np.asarray(a5, np.float64)
    r5 = np.asarray(r5, np.float64)
    a6 = np.asarray(a6, np.float64)[:len(a5)]
    r6 = np.asarray(r6, np.float64)[:len(r5)]
    for name, x5, x6 in (("adv", a5, a6), ("returns", r5, r6)):
        if np.array_equal(x5, x6):
            continue
        bad = np.argwhere(x5 != x6)
        i, b = int(bad[0][0]), int(bad[0][1])
        raise AssertionError(
            f"{label} {name} 불일치 {len(bad)}/{x5.size} 칸 · 최대 |Δ| {np.abs(x5 - x6).max():.3e}\n"
            f"  처음 갈린 지점 (구간 {i}, 블록 {b}): v5 {x5[i, b]!r} vs 배열 {x6[i, b]!r}\n"
            f"  원인 후보: 합산·곱 결합 순서 또는 float32/float64 갈래 (모듈 머리말 (2))")


def _both(intervals, bootstrap, *, gamma=GAMMA, lam=LAM, unit=UNIT):
    """v5 와 배열판을 같은 입력으로 한 번씩 부른다."""
    a5, r5 = gae_v5(intervals, bootstrap, gamma=gamma, lam=lam, time_unit_s=unit)
    batch, over = pack_intervals(intervals)
    assert not over.any, f"넘침이 있으면 답이 달라진다: {over}"
    raise_on_audit(gae_audit(batch, bootstrap, gamma=gamma, lam=lam, time_unit_s=unit))
    a6, r6 = gae(batch, bootstrap, gamma=gamma, lam=lam, time_unit_s=unit)
    return (a5, r5), (a6, r6), batch


# ═══════════════════════════════════════════ ① 포장 — 들쭉날쭉 → 고정 칸
def test_01_pack_roundtrip_matches_v5_choice_fields():
    """포장했다가 다시 꺼낸 결정이 v5 `Choice` 여섯 칸과 비트 일치한다."""
    rng, pol = random.Random(11), _policy()
    checked = 0
    for _ in range(40):
        n_blocks = rng.choice([1, 2, 3, 5])
        ivs = _rollout(rng, pol, n=rng.randint(1, 5), n_blocks=n_blocks, values_f32=True)
        batch, over = pack_intervals(ivs)
        assert not over.any and batch.r_max == len(ivs) and batch.n_blocks == n_blocks
        assert batch.feat_dim == FEAT_DIM == 37
        rows = np.asarray(batch.rows)
        mask = np.asarray(batch.mask)
        for i, row in enumerate(ivs):
            np.testing.assert_array_equal(np.asarray(batch.states[i]),
                                          row.states.detach().numpy())
            np.testing.assert_array_equal(np.asarray(batch.values[i]), row.values)
            assert float(batch.start_s[i]) == row.start_s
            assert float(batch.end_s[i]) == row.end_s
            assert float(batch.reward[i]) == row.reward
            assert bool(batch.terminated[i]) == row.terminated
            for b, lst in enumerate(row.choices):
                assert int(batch.n_choices[i, b]) == len(lst)
                for k, c in enumerate(lst):
                    got = unpack_choice(batch, i, b, k)
                    assert got["role"] == c.role
                    assert got["time_s"] == c.time_s
                    assert got["action"] == c.action
                    assert got["log_prob"] == c.log_prob
                    np.testing.assert_array_equal(got["rows"], c.rows.detach().numpy())
                    np.testing.assert_array_equal(got["mask"], c.mask.numpy())
                    n = len(c.rows)
                    # 남는 칸은 0 · False — 진짜 후보와 섞이지 않는다
                    assert not rows[i, b, k, n:].any() and not mask[i, b, k, n:].any()
                    checked += 1
    assert checked > 300, f"검사한 결정이 너무 적다 ({checked})"
    MEASURED["choices_roundtripped"] = checked


def test_02_pack_padding_conventions():
    """빈 칸 규약 — 정수 -1 · 시각 +inf · 실수·마스크 0·False. 실제 개수는 계수기가 든다."""
    rng, pol = random.Random(12), _policy()
    ivs = _rollout(rng, pol, n=2, n_blocks=2, values_f32=True)
    ivs[0].choices[0] = []                          # 결정 0개인 칸을 확실히 만든다
    batch, over = pack_intervals(ivs, r_max=5, c_max=4, a_max=9)
    assert not over.any
    assert batch.r_max == 5 and batch.c_max == 4 and batch.a_max == 9
    pad = slice(2, 5)
    assert np.all(np.asarray(batch.start_s)[pad] == 0.0)     # inf 를 넣으면 inf−inf=nan 이 번진다
    assert np.all(np.asarray(batch.end_s)[pad] == 0.0)
    assert np.all(np.asarray(batch.reward)[pad] == 0.0)
    assert not np.asarray(batch.terminated)[pad].any()
    assert np.all(np.asarray(batch.n_choices)[pad] == 0)
    assert np.all(np.asarray(batch.action)[pad] == -1)
    assert np.all(np.asarray(batch.role)[pad] == -1)
    assert np.all(np.isinf(np.asarray(batch.choice_time_s)[pad]))
    assert not np.asarray(batch.mask)[pad].any()
    assert int(batch.n_intervals) == 2
    assert int(batch.n_choices[0, 0]) == 0
    np.testing.assert_array_equal(np.asarray(valid_mask(batch)),
                                  np.array([True, True, False, False, False]))


def test_03_overflow_is_counted_not_silently_dropped():
    """칸이 모자라면 **개수로 보고**한다 — v5 에는 없는 상황이라 조용히 버리면 답이 달라진다."""
    rng, pol = random.Random(13), _policy()
    ivs = _rollout(rng, pol, n=4, n_blocks=2, values_f32=True)
    for b in range(2):
        ivs[0].choices[b] = [_choice(rng, 0.0, "crane") for _ in range(3)]

    _, over = pack_intervals(ivs, r_max=2)
    assert over.intervals == 2 and over.any

    _, over = pack_intervals(ivs, c_max=1)
    dropped = sum(max(0, len(lst) - 1) for r in ivs for lst in r.choices)
    assert over.choices == dropped > 0 and over.any

    _, over = pack_intervals(ivs, a_max=1)
    cut = sum(max(0, len(c.rows) - 1) for r in ivs for lst in r.choices for c in lst)
    assert over.cands == cut > 0 and over.any

    batch, over = pack_intervals(ivs)
    assert over == IntervalOverflow(0, 0, 0) and not over.any


def test_04_masks_and_counters_match_v5_update():
    """`active_mask`·`micro_actions`·`block_samples` 가 v5 `update.py:19-21, 81-82` 와 같다."""
    rng, pol = random.Random(14), _policy()
    for _ in range(12):
        n, nb = rng.randint(1, 6), rng.choice([1, 3, 5])
        ivs = _rollout(rng, pol, n=n, n_blocks=nb, values_f32=True)
        batch, _ = pack_intervals(ivs, r_max=n + 3)
        entries = [(i, b) for i, row in enumerate(ivs) for b in range(len(row.values))]
        active = [(i, b) for i, b in entries if ivs[i].choices[b]]
        want = np.zeros((n + 3, nb), bool)
        for i, b in active:
            want[i, b] = True
        np.testing.assert_array_equal(np.asarray(active_mask(batch)), want)
        assert int(micro_actions(batch)) == sum(len(c) for r in ivs for c in r.choices)
        assert block_samples(batch) == len(entries)
        # 미니배치 순열이 올라탈 평평한 순서 — v5 `update.py:19-20` 과 같아야 한다
        assert [tuple(map(int, p)) for p in flat_entries(batch)] == entries


# ═══════════════════════════════════════════ ② GAE — v5 와 == 대조
@pytest.mark.parametrize("values_f32", [True, False], ids=["values-f32", "values-f64"])
def test_05_gae_matches_v5_on_200_random_rollouts(values_f32):
    """★핵심 — 무작위 200조(길이·블록·후보·dt·종료·설정을 섞어) 에서 `adv`·`returns` 가 `==`."""
    rng, pol = random.Random(20 + int(values_f32)), _policy()
    trials = 0
    for k in range(200):
        n = rng.randint(1, 9)
        nb = rng.choice([1, 2, 3, 5, 8])
        gamma, lam, unit = CONFIGS[k % len(CONFIGS)]
        ivs = _rollout(rng, pol, n=n, n_blocks=nb, values_f32=values_f32,
                       terminate=rng.random() < 0.25)
        boot = _bootstrap(pol, rng, nb)
        (a5, r5), (a6, r6), batch = _both(ivs, boot, gamma=gamma, lam=lam, unit=unit)
        assert a5.dtype == np.float64 and r5.dtype == np.float64
        _compare(a5, r5, a6, r6, f"조 {k} (구간 {n} · 블록 {nb} · γ {gamma} · λ {lam} · 단위 {unit})")
        trials += 1
    assert trials == 200
    MEASURED[f"random_rollouts_f32={values_f32}"] = trials


def test_06_gae_boundary_cases():
    """경계 — 결정 0개 · 후보 1개 · 구간 1개(=부트스트랩만) · dt 가 단위와 다름."""
    rng, pol = random.Random(31), _policy()

    # 결정이 전혀 없는 조 (가치망만 학습하는 구간)
    ivs = _rollout(rng, pol, n=4, n_blocks=3, values_f32=True)
    for row in ivs:
        row.choices[:] = [[] for _ in row.choices]
    (a5, r5), (a6, r6), batch = _both(ivs, _bootstrap(pol, rng, 3))
    _compare(a5, r5, a6, r6, "결정 0개")
    assert int(micro_actions(batch)) == 0 and not np.asarray(active_mask(batch)).any()

    # 후보 1개뿐인 결정 (마스크 한 칸) — 배열 축이 1 로 줄어드는 경계
    ivs = _rollout(rng, pol, n=3, n_blocks=2, values_f32=True)
    one = Choice("crane", 0.0, encode([[1.0]], "crane"), torch.ones(1, dtype=torch.bool), 0, -0.5)
    ivs[1].choices[0] = [one]
    (a5, r5), (a6, r6), batch = _both(ivs, _bootstrap(pol, rng, 2))
    _compare(a5, r5, a6, r6, "후보 1개")
    assert batch.a_max >= 1 and int(batch.n_cands[1, 0, 0]) == 1

    # 구간 1개 — 첫 단계가 곧 마지막 단계다 (nxt = 부트스트랩 f64 갈래만 탄다)
    for _ in range(20):
        ivs = _rollout(rng, pol, n=1, n_blocks=rng.choice([1, 4]), values_f32=True)
        boot = _bootstrap(pol, rng, len(ivs[0].values))
        (a5, r5), (a6, r6), _ = _both(ivs, boot)
        _compare(a5, r5, a6, r6, "구간 1개(부트스트랩)")

    # dt ≠ time_unit_s — gamma**dt · lam**dt 가 1 이 아닌 값이 된다
    for dt in (15.0, 45.0, 60.0, 90.0, 240.0):
        ivs = _rollout(rng, pol, n=3, n_blocks=2, values_f32=True)
        t = 0.0
        for row in ivs:
            row.start_s, row.end_s, t = t, t + dt, t + dt
        (a5, r5), (a6, r6), _ = _both(ivs, _bootstrap(pol, rng, 2))
        _compare(a5, r5, a6, r6, f"dt={dt}")


def test_07_terminated_cuts_future_rewards_like_v5():
    """종료 구간에서 할인이 0 이 된다 — v5 회귀시험(`test_ppo_regressions.py:74-79`) 그대로."""
    states = encode([[0]], "state")
    first = Interval(0, 60, states, np.zeros(1), [[]], -1, terminated=True)
    nxt = Interval(0, 60, states, np.zeros(1), [[]], -10)
    (a5, r5), (a6, r6), _ = _both([first, nxt], [2], gamma=1.0, lam=1.0, unit=60.0)
    assert np.asarray(r6)[:, 0].tolist() == [-1.0, -8.0] == r5[:, 0].tolist()
    _compare(a5, r5, a6, r6, "종료 구간")

    # 종료가 중간에 있는 긴 조 — 새 에피소드가 이어 붙는다 (시각이 되감겨도 연속성 검사를 안 받는다)
    rng, pol = random.Random(41), _policy()
    for _ in range(20):
        ivs = _rollout(rng, pol, n=6, n_blocks=3, values_f32=True)
        cut = rng.randrange(1, 5)
        ivs[cut].terminated = True
        for j, row in enumerate(ivs[cut + 1:]):         # 새 에피소드는 0 초부터 다시
            row.start_s, row.end_s = j * 60.0, (j + 1) * 60.0
        (a5, r5), (a6, r6), _ = _both(ivs, _bootstrap(pol, rng, 3))
        _compare(a5, r5, a6, r6, f"중간 종료 (구간 {cut})")


def test_08_zero_intervals_is_flagged_not_computed():
    """구간 0개 — v5 는 `ValueError` 를 던진다. 배열판은 `empty` 비트로 알린다."""
    with pytest.raises(ValueError, match="Invalid rollout"):
        gae_v5([], [0.0], gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
    batch, over = pack_intervals([], n_blocks=2, r_max=3)
    assert not over.any and int(batch.n_intervals) == 0
    audit = gae_audit(batch, np.zeros(2), gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
    assert isinstance(audit, GaeAudit) and bool(audit.empty) and not audit.ok
    with pytest.raises(ValueError, match="Invalid rollout or discount configuration"):
        raise_on_audit(audit)
    # 계산 자체는 던지지 않고 0 을 낸다 (jit 안에서 부를 수 있어야 한다)
    a6, r6 = gae(batch, np.zeros(2), gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
    assert not np.asarray(a6).any() and not np.asarray(r6).any()


def test_09_audit_raises_exactly_what_v5_raises():
    """v5 가 던지는 여덟 가지 입력 — 배열판이 **같은 메시지**로 던진다."""
    st1 = encode([[0]], "state")
    st2 = encode([[0], [0]], "state")

    def v5_message(ivs, boot, **kw) -> str:
        with pytest.raises(ValueError) as e:
            gae_v5(ivs, boot, **kw)
        return str(e.value)

    def mine_message(ivs, boot, *, n_blocks=None, **kw) -> str:
        with pytest.raises(ValueError) as e:
            batch, _ = pack_intervals(ivs, n_blocks=n_blocks)
            raise_on_audit(gae_audit(batch, boot, **kw))
        return str(e.value)

    ok = dict(gamma=0.9, lam=0.9, time_unit_s=60.0)
    row = lambda s, e, rw=-1.0, term=False, v=None: Interval(          # noqa: E731
        s, e, st1, np.zeros(1) if v is None else v, [[]], rw, terminated=term)

    cases = [
        # (설명, 구간, 부트스트랩, 설정)
        ("시각 비유한 (end=nan)", [row(0, float("nan"))], [0.0], ok),
        ("end ≤ start", [row(60, 60)], [0.0], ok),
        ("start < 0", [row(-60, 0)], [0.0], ok),
        ("불연속 (60↛120)", [row(0, 60), row(120, 180)], [0.0], ok),
        ("보상 비유한", [row(0, 60, rw=float("inf"))], [0.0], ok),
        ("values 비유한", [row(0, 60, v=np.array([np.nan]))], [0.0], ok),
        ("부트스트랩 비유한", [row(0, 60)], [float("nan")], ok),
        ("부트스트랩 빈 벡터", [row(0, 60)], [], ok),
        ("부트스트랩 2차원", [row(0, 60)], [[0.0]], ok),
        ("블록 수 불일치", [row(0, 60)], [0.0, 0.0], ok),
        ("time_unit_s = 0", [row(0, 60)], [0.0], dict(gamma=.9, lam=.9, time_unit_s=0.0)),
        ("time_unit_s 비유한", [row(0, 60)], [0.0], dict(gamma=.9, lam=.9, time_unit_s=float("inf"))),
        ("gamma = 0", [row(0, 60)], [0.0], dict(gamma=0.0, lam=.9, time_unit_s=60.0)),
        ("gamma > 1", [row(0, 60)], [0.0], dict(gamma=1.5, lam=.9, time_unit_s=60.0)),
        ("lam < 0", [row(0, 60)], [0.0], dict(gamma=.9, lam=-0.1, time_unit_s=60.0)),
        ("lam > 1", [row(0, 60)], [0.0], dict(gamma=.9, lam=1.1, time_unit_s=60.0)),
    ]
    for label, ivs, boot, kw in cases:
        want = v5_message(copy.copy(ivs), boot, **kw)
        got = mine_message(copy.copy(ivs), boot, **kw)
        assert got == want, f"{label}: v5 '{want}' vs 배열 '{got}'"

    # v5 의 '모양 불일치' 한 갈래는 **포장 시점**에 걸린다 (values 1칸 vs choices 2칸)
    broadcast = Interval(0, 60, st2, np.zeros(1), [[], []], -1)
    with pytest.raises(ValueError, match="block dimensions"):
        gae_v5([broadcast], [0, 0], **ok)
    with pytest.raises(ValueError, match="block dimensions"):
        pack_intervals([broadcast])
    MEASURED["audit_cases"] = len(cases) + 1


def test_10_padding_length_does_not_change_the_answer():
    """같은 자료를 R 칸 크게 잡아도 실제 구간의 답은 그대로이고 패딩 칸은 0 이다."""
    rng, pol = random.Random(51), _policy()
    for _ in range(15):
        n, nb = rng.randint(1, 5), rng.choice([1, 3])
        ivs = _rollout(rng, pol, n=n, n_blocks=nb, values_f32=True,
                       terminate=rng.random() < 0.3)
        boot = _bootstrap(pol, rng, nb)
        tight, _ = pack_intervals(ivs)
        loose, _ = pack_intervals(ivs, r_max=n + 7, c_max=8, a_max=11)
        at, rt = gae(tight, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
        al, rl = gae(loose, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
        np.testing.assert_array_equal(np.asarray(at), np.asarray(al)[:n])
        np.testing.assert_array_equal(np.asarray(rt), np.asarray(rl)[:n])
        assert not np.asarray(al)[n:].any() and not np.asarray(rl)[n:].any()


def test_11_jit_equals_eager():
    """jit 해도 같은 비트 — 융합(FMA)·재결합은 `exact` 규약이 막는다."""
    rng, pol = random.Random(61), _policy()
    jitted = jax.jit(gae, static_argnames=("gamma", "lam", "time_unit_s"))
    for values_f32 in (True, False):
        for _ in range(8):
            ivs = _rollout(rng, pol, n=5, n_blocks=3, values_f32=values_f32)
            boot = _bootstrap(pol, rng, 3)
            a5, r5 = gae_v5(ivs, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
            batch, _ = pack_intervals(ivs, r_max=7)
            ae, re = gae(batch, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
            aj, rj = jitted(batch, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
            np.testing.assert_array_equal(np.asarray(ae), np.asarray(aj))
            np.testing.assert_array_equal(np.asarray(re), np.asarray(rj))
            _compare(a5, r5, aj, rj, f"jit (values_f32={values_f32})")


def test_12_gae_batch_equals_singles():
    """세계 묶음(vmap)이 낱개와 같다 — 세계마다 구간 수가 달라도 패딩 규약이 흡수한다."""
    rng, pol = random.Random(71), _policy()
    nb, rmax, worlds = 3, 8, 5
    packed, boots, singles = [], [], []
    for _ in range(worlds):
        ivs = _rollout(rng, pol, n=rng.randint(1, rmax), n_blocks=nb, values_f32=True,
                       terminate=rng.random() < 0.4)
        boot = _bootstrap(pol, rng, nb)
        batch, over = pack_intervals(ivs, r_max=rmax, c_max=4, a_max=6)
        assert not over.any
        packed.append(batch)
        boots.append(boot)
        a5, r5 = gae_v5(ivs, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
        a6, r6 = gae(batch, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
        _compare(a5, r5, a6, r6, "묶음 낱개")
        singles.append((np.asarray(a6), np.asarray(r6)))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *packed)
    ab, rb = gae_batch(stacked, np.stack(boots), gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
    for w, (a6, r6) in enumerate(singles):
        np.testing.assert_array_equal(np.asarray(ab[w]), a6)
        np.testing.assert_array_equal(np.asarray(rb[w]), r6)


# ═══════════════════════════════════════════ ③ 실제 런타임 테이프
def _runtime_tape(*, n_blocks=3, boundaries=8, seed=77, terminated=False):
    """실제 `PPORuntime` 을 굴려 `buffer` 를 얻는다 (`test_ppo_continuous.py:18-45` 대체 훅 방식).

    세계(엔진)는 필요 없다 — `states_at`·`read_cost` 만 대체하고 `_update` 로 갱신을 막아
    `buffer` 에 `Interval` 이 쌓이게 한다. `values` 는 **실제 torch 망의 float32 출력**이다.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rt = PPORuntime(BlockPolicy(), config=PPOConfig(rollout_intervals=10 ** 6),
                    seed=seed, training=True, sample_actions=False)
    rt.bids = [f"b{i:02d}" for i in range(n_blocks)]
    rt.index = {b: i for i, b in enumerate(rt.bids)}
    rt.states_at = lambda t: encode(
        [[t / 3600.0, i * 0.25, (t % 600) / 600.0] for i in range(n_blocks)], "state")
    rt.read_cost = lambda t: 1.0e6 + 3.0 * t + 0.5 * (t / 60.0) ** 2
    boots: list[np.ndarray] = []
    rt._update = lambda bootstrap: boots.append(np.asarray(bootstrap).copy())
    for step in range(boundaries):
        t = float(step * 60)
        rt.boundary(t)
        for bid in rt.bids:
            for _ in range(rng.randint(0, 2)):
                role = rng.choice(("seller", "buyer", "crane"))
                n = 2 if role == "buyer" else rng.randint(1, 5)
                rows = [[rng.uniform(-2, 2) for _ in range(6)] for _ in range(n)]
                mask = torch.ones(n, dtype=torch.bool)
                rt.select(role, bid, t + rng.uniform(0, 59), rows, mask)
    rt.finish(float(boundaries * 60), terminated=terminated)
    assert boots, "갱신 훅이 한 번도 안 불렸다 — 무대 구성이 틀렸다"
    return rt, boots[-1]


@pytest.mark.parametrize("terminated", [False, True])
def test_15_real_runtime_tape_matches_v5(terminated):
    """실제 런타임이 만든 `Interval` 목록 그대로 — `values` 가 float32 인 **진짜 경로**다."""
    rt, boot = _runtime_tape(terminated=terminated)
    ivs = rt.buffer
    assert len(ivs) >= 5, f"구간이 너무 적다 ({len(ivs)})"
    assert np.asarray(ivs[0].values).dtype == np.float32, "실제 경로는 float32 여야 한다"
    assert sum(len(c) for r in ivs for c in r.choices) > 0, "결정이 하나도 안 쌓였다"
    boot = np.zeros_like(boot) if terminated else boot
    (a5, r5), (a6, r6), batch = _both(ivs, boot)
    _compare(a5, r5, a6, r6, f"런타임 테이프 (terminated={terminated})")
    assert int(batch.n_intervals) == len(ivs)
    assert int(micro_actions(batch)) == sum(len(c) for r in ivs for c in r.choices)
    roles = {ROLES[int(batch.role[i, b, k])]
             for i in range(batch.r_max) for b in range(batch.n_blocks)
             for k in range(int(batch.n_choices[i, b]))}
    MEASURED[f"tape_terminated={terminated}"] = (len(ivs), sorted(roles))


# ═══════════════════════════════════════════ ④ 실측 — 왜 이렇게 썼나
def _gae_all_f64(batch, bootstrap, *, gamma, lam, time_unit_s):
    """반례용 — `delta` 를 **전부 float64** 로 계산하는 판 (v5 승격 규칙을 무시한 것)."""
    nxt0 = jnp.asarray(bootstrap, jnp.float64)
    g64, l64 = jnp.asarray(gamma, jnp.float64), jnp.asarray(lam, jnp.float64)
    idx = jnp.arange(batch.r_max, dtype=jnp.int32)

    def body(carry, x):
        trace, nxt = carry
        i, s, e, rw, tm, v = x
        on = i < batch.n_intervals
        dt = div_const(e - s, time_unit_s)
        cont = jnp.where(tm, jnp.zeros((), jnp.float64), jnp.ones((), jnp.float64))
        disc = mul_exact(jnp.power(g64, dt), cont)
        dlam = mul_exact(disc, jnp.power(l64, dt))
        delta = (rw + mul_exact(disc, nxt)) - v.astype(jnp.float64)
        nt = delta + mul_exact(dlam, trace)
        return ((jnp.where(on, nt, trace), jnp.where(on, v.astype(jnp.float64), nxt)),
                jnp.where(on, nt, jnp.zeros((), jnp.float64)))

    _, adv = lax.scan(body, (jnp.zeros_like(nxt0), nxt0),
                      (idx, batch.start_s, batch.end_s, batch.reward,
                       batch.terminated, batch.values), reverse=True)
    return adv


def test_13_float32_branch_is_necessary():
    """★반례 — `delta` 를 전부 float64 로 계산하면 v5 와 갈린다. 몇 조가 갈리고 |Δ| 가 얼마인가."""
    rng, pol = random.Random(91), _policy()
    bad, worst, trials = 0, 0.0, 40
    for _ in range(trials):
        ivs = _rollout(rng, pol, n=6, n_blocks=3, values_f32=True)
        boot = _bootstrap(pol, rng, 3)
        a5, _ = gae_v5(ivs, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
        batch, _ = pack_intervals(ivs)
        a6, _ = gae(batch, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT)
        np.testing.assert_array_equal(np.asarray(a6), a5)          # 우리 판은 비트 일치
        wrong = np.asarray(_gae_all_f64(batch, boot, gamma=GAMMA, lam=LAM, time_unit_s=UNIT))
        d = np.abs(wrong - a5)
        if d.max() > 0:
            bad += 1
            worst = max(worst, float(d.max()))
    assert bad == trials, ("float64 로만 계산해도 v5 와 같다면 이 갈래는 불필요하다 — "
                           f"갈린 조 {bad}/{trials}")
    assert worst > 1e-9, f"차이가 너무 작다 ({worst:.3e}) — 반례로 못 쓴다"
    MEASURED["f32_branch_needed"] = (bad, trials, worst)


#: 실제 60초 격자에서 나오는 dt — 검토 경계가 격자 위라 (end−start)/time_unit_s 는 이런 값만 된다
#: ★**실제로 나올 수 있는 dt** (구간 길이 / `time_unit_s=60`) — 손으로 고른 여덟 개가 아니다
#:  (검증 반박: "실제로 쓰이는 dt 가 그 여덟 개라는 보장이 없다 — 배수 구간은 dt=120 이고 마지막
#:   구간 길이는 `sim_end − month_s` 에 달렸다"). 검토 경계가 60초 격자이므로 dt 는 **양의 정수**이고,
#:  배수 창(`DIURNAL_DRAIN_S=7,200`)이 한 구간이 되면 dt=120 이다. 이연·반복 경계로 분수 dt 도 생길 수
#:  있어 이진분수를 함께 둔다. 하루 전체(dt=1,441)도 상한으로 넣는다.
GRID_DTS = (tuple(0.25 * k for k in range(1, 17))          # 0.25 ~ 4.0 (이진분수)
            + tuple(float(k) for k in range(1, 131))        # 정수 1~130 (배수 구간 120 포함)
            + (1441.0,))


@pytest.mark.parametrize("base", [0.999, 0.95, 1.0, 0.5])
def test_14_power_bit_match_rate(base):
    """`gamma ** dt` 의 비트 일치 비율 실측 — 파이썬 libm `pow` vs XLA 커널.

    ★**무대(백엔드)에 따라 다르다** (2026-09-27 실측):
      CPU(x64)  dt 무작위 2만점에서 0.999·0.95·1.0 은 **0점 불일치**, 0.5 만 5점 (최대 상대 1.8e-16)
      GPU(x64)  0.999·0.95 도 **약 24% 가 1 ulp 갈린다** (4,825/20,004 · 최대 상대 ≤ 4e-16)
                격자 dt 라도 밑이 0.5 면 1.5·4.0 에서 1 ulp 갈렸다
    그래서 **단언하는 것은 두 가지**다 — ① 어긋남이 어느 밑·어느 무대에서도 1 ulp 를 넘지 않는다
    ② **실제 설정의 밑**(`PPOConfig.gamma=0.999`·`gae_lambda=0.95`, 그리고 할인 없음 1.0)에서는
    실제 60초 격자의 dt(0.25·0.5·0.75·1·1.5·2·3·4)가 **CPU·GPU 둘 다 정확히 같다**.
    나머지는 재서 찍는다 (`gpu/v5net.py` 의 실측 규약과 같은 방식) — 실제 설정 밖의 밑이나
    격자 밖 dt 를 쓰는 무대에서는 GPU 에서 마지막 비트까지 같다고 주장할 수 없다.
    """
    rng = random.Random(101)
    grid = np.array(GRID_DTS, dtype=np.float64)
    dts = np.concatenate([grid, np.array([rng.uniform(1e-4, 80.0) for _ in range(20_000)])])
    want = np.array([base ** float(d) for d in dts], dtype=np.float64)
    got = np.asarray(jax.jit(lambda d: jnp.power(jnp.asarray(base, jnp.float64), d))(jnp.asarray(dts)))
    same = int((want == got).sum())
    rel = np.abs(want - got) / np.maximum(np.abs(want), 1e-300)
    n = len(grid)
    grid_bad = [float(d) for d, ok in zip(grid, want[:n] == got[:n]) if not ok]
    MEASURED[f"pow_{base}"] = (f"{same}/{len(dts)} 일치", f"최대 상대 {float(rel.max()):.3e}",
                               f"격자 갈림 {grid_bad}", jax.default_backend())
    assert rel.max() <= 4e-16, f"1 ulp 를 넘게 갈렸다 — 최대 상대 {rel.max():.3e}"
    if base in (0.999, 0.95, 1.0):      # ★실제 설정의 밑
        #: CPU(정본 무대)에서는 **실제로 나올 수 있는 dt 전부**가 비트 일치여야 한다.
        #:  GPU 는 `pow` 구현이 달라 1 ulp 갈리는 dt 가 있다 (실측: 밑 0.95 에서 dt=24·121·1441) —
        #:  그것을 **재서 찍고**(MEASURED) 크기만 1 ulp 로 묶는다. 조용히 넘기지 않으려고 개수도 남긴다.
        if jax.default_backend() == "cpu":
            assert not grid_bad, f"★CPU 에서 실제 dt {len(grid_bad)}개가 갈렸다 — {grid_bad[:8]}"
        else:
            MEASURED[f"pow_{base}_gpu_grid_bad"] = (len(grid_bad), grid_bad[:8])


def test_zz_report(capsys):
    """실측 요약 — 무엇을 얼마나 확인했나."""
    with capsys.disabled():
        print(f"\n── 배열 학습 버퍼·GAE 실측 (gpu/ppo_buffer.py · YR-327 조각 8)"
              f" · 무대 {jax.default_backend()} · x64 {jax.config.jax_enable_x64}")
        for key in sorted(MEASURED):
            print(f"   {key:32s} {MEASURED[key]}")
        print("   ★v5 `Interval.values` 는 float32 → `delta` 도 float32 (마지막 구간만 float64)")
    assert MEASURED.get("random_rollouts_f32=True") == 200
    assert MEASURED.get("random_rollouts_f32=False") == 200
    assert MEASURED.get("f32_branch_needed", (0,))[0] > 0
