"""배열판 v5 학습 정책망(`gpu/v5net.py`)이 v5 `ppo/model.BlockPolicy` 와 **같은 답**을 내는가 ([[YR-327]] 조각 7).

■ 무엇을 지키나 — 기대값은 손으로 적지 않는다. **torch 망을 실제로 만들어** 같은 입력의 답을 받아 대조한다.
  ① `encode` — 37칸 입력이 v5 `ppo/model.encode` 와 **비트 동일** (역할 4종 · 폭 1~32 · buyer BUY 표시)
  ② `load_v5_params` — torch `state_dict` 를 전치 적재. 값은 하나도 안 바뀌고(float32→float64 는 정확),
     왕복(state_dict → 배열 → state_dict)이 항등이며, 되돌린 것을 torch 에 다시 실어도 같은 답
  ③ ★actor 점수·critic 값 — 무작위 입력 500건. torch 를 `.double()` 로 올려야 의미가 있다
  ④ 결정(argmax)·로그확률·확률·엔트로피 — torch `Categorical` 과 대조. 빈 마스크는 v5 가 예외라 `mask_ok` 로
  ⑤ jit·vmap·배치 브로드캐스트가 하나씩 돌린 것과 **비트 동일** (결정·bool 잎은 예외 없이)
  ⑤-2 ★PPO 손실의 **기울기** 8장이 torch 자동미분과 같다 — 조각 8 이 이 망으로 학습할 수 있다는 근거
  ⑥ 실제 학습된 v5 체크포인트(`outputs/v5/*/policy.pt`)로 같은 대조 — 있는 것만 돌린다(git 무시 경로)

■ ⚠️ 이 층은 **비트 일치가 불가능**하다 (`gpu/v5net.py` 머리말 ★) — `tanh` 근사식과 GEMM 축소 순서가
  torch(MKL/libm) 와 XLA 에서 다르다. 그래서 여기서는 **결정이 같은가 + |Δ| 가 얼마인가**를 시험하고,
  측정값을 화면에 찍어 통합자가 허용오차를 고를 근거로 남긴다 (`-s` 로 돌릴 때 보인다).

실행 (Git Bash · Windows · CPU x64):
    PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 .venv-jax/Scripts/python.exe \
        -m pytest tests/v6/test_gpu_v5net.py -q -s -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import os

import numpy as np                       # ★torch 보다 먼저 — 아나콘다 MKL 의 libiomp 를 먼저 올려
import pytest                            #   torch 사본과 부딪히는 OMP #15 를 피한다

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — 망 대조는 x64 에서만 뜻이 있다
jnp = jax.numpy
torch = pytest.importorskip("torch")

from yard_rl.v6.gpu import v5net                                            # noqa: E402
from yard_rl.v6.ppo.model import INPUT_DIM, RAW_DIM, ROLES, BlockPolicy     # noqa: E402
from yard_rl.v6.ppo.model import encode as encode_v5                        # noqa: E402

#: 크레인 결정 행의 실제 폭 (`ppo/crane.py:39-56` — 블록 8 + 종류 4 + 값 7 + 직전크레인 4 + 1)
CRANE_RAW = 24
#: 실제 학습된 v5 가중치 (git 무시 경로라 없을 수 있다 — 있는 것만 돌린다).
#: ★2026-09-26 검증 반박: 전에 쓴 두 경로(yr302-final-train · yr303-final-smoke)는 **바이트 단위로 같은 파일**
#:   이었다 (md5 ba55cacb… · 95,257 바이트 · 8개 키 전부 `torch.equal` 참). 그래서 "실제 학습 체크포인트 2벌 ·
#:   체크포인트 200건" 은 실질 1벌 100건이었다. 지금은 **내용 해시로 중복을 지우고**, 실제로 다른 학습 결과
#:   (yr302-debug-01 — PPO 갱신 2회 · yr302-final-train 은 6회)를 후보에 더한다.
def _unique_ckpts(paths):
    seen, out = set(), []
    for q in paths:
        if not os.path.exists(q):
            continue
        h = hashlib.md5(open(q, "rb").read()).hexdigest()
        if h in seen:
            continue                        # 같은 가중치를 두 번 돌리지 않는다
        seen.add(h)
        out.append(q)
    return out


REAL_CKPTS = _unique_ckpts(("outputs/v5/yr302-final-train/policy.pt",      # PPO 갱신 6회
                            "outputs/v5/yr302-debug-01/policy.pt",         # PPO 갱신 2회 — 서로 다른 가중치
                            "outputs/v5/yr303-final-smoke/policy.pt"))     # 위와 같은 파일이면 걸러진다


def test_real_ckpts_are_distinct_weights():
    """★증거의 규모를 정직하게 — `REAL_CKPTS` 에 **같은 가중치**가 두 번 들어 있지 않은가.

    같은 파일을 두 경로로 넣어 두면 "체크포인트 2벌로 대조했다" 가 거짓이 된다 (2026-09-26 검증 반박).
    """
    if not REAL_CKPTS:
        pytest.skip("outputs/v5/*/policy.pt 없음 (git 무시 경로)")
    digests = [hashlib.md5(open(q, "rb").read()).hexdigest() for q in REAL_CKPTS]
    assert len(set(digests)) == len(digests), f"같은 가중치가 두 번 들어 있다 {list(zip(REAL_CKPTS, digests))}"


# ───────────────────────────────────────────────── 도우미
def _torch_policy(hidden: int = 64, seed: int = 0) -> BlockPolicy:
    """v5 망을 **실제로** 만든다 — 기대값의 유일한 출처. 고정 시드."""
    torch.manual_seed(seed)
    return BlockPolicy(hidden=hidden)


def _params(policy: BlockPolicy) -> v5net.V5PolicyParams:
    return v5net.load_v5_params(policy.state_dict())


def _rows(rng, m: int, d: int = CRANE_RAW) -> list[list[float]]:
    """크레인 특징 행처럼 생긴 무작위 행 — 0 이 섞이고 눈금이 O(0.1~3) 이다."""
    x = rng.standard_normal((m, d)) * rng.choice([0.1, 1.0, 3.0])
    x[rng.random((m, d)) < 0.35] = 0.0                     # one-hot·없는 계획의 0 칸
    return x.tolist()


def _mask(rng, m: int) -> np.ndarray:
    """적어도 하나는 참 (v5 `model.py:45-46` 의 전제)."""
    mk = rng.random(m) < 0.7
    if not mk.any():
        mk[int(rng.integers(m))] = True
    return mk


def _torch_forward(policy: BlockPolicy, rows, role: str, *, double: bool):
    """v5 순전파 — `double=True` 면 망과 입력을 float64 로 올린다 (float32 대조는 뜻이 약하다)."""
    net = policy.double() if double else policy.float()
    x = encode_v5(rows, role)
    x = x.double() if double else x
    with torch.no_grad():
        scores = net.actor(net.trunk(x)).squeeze(-1)
        value = net.value(x)
    policy.float()                                          # 원래대로 (다음 대조가 float32 를 볼 수 있게)
    return np.asarray(scores.numpy(), np.float64), np.asarray(value.numpy(), np.float64)


def _same(tag: str, got: np.ndarray, want: np.ndarray, drift: dict, *, exact: bool = False):
    """비트 일치가 기본. 실수 잎만, 갈려도 **몇 ulp 안**이면 통과시키고 |Δ| 를 `drift` 에 적어 둔다.

    결정(`action`)·`mask_ok` 는 `exact=True` — 여기가 갈리면 동등성이 깨진 것이다.
    """
    if np.array_equal(got, want):
        return
    assert not exact, f"{tag} — 정수/bool 잎이 갈렸다 (결정 동등성 깨짐)"
    live = np.isfinite(want)
    d = float(np.abs(got - want)[live].max())
    drift[tag] = d
    ulp = float(np.spacing(np.abs(want[live]).max()))
    assert d <= 8 * ulp, f"{tag} |Δ|={d:.3e} — {8 * ulp:.3e}(8 ulp) 를 넘는다"


def _report(tag: str, worst_abs: float, worst_rel: float, n: int, flips: int):
    print(f"    {tag:34s} n={n:4d}  최대|Δ|={worst_abs:.3e}  최대상대Δ={worst_rel:.3e}  argmax 불일치={flips}")


# ───────────────────────────────────────────────── ① encode — 37칸 입력
@pytest.mark.parametrize("role", ROLES)
def test_encode_matches_v5_bitwise(role):
    """v5 `encode` 와 **비트 동일**. 역할 한자리 위치·0 패딩·buyer BUY 표시까지."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        d = int(rng.integers(1, RAW_DIM + 1))
        m = 2 if role == "buyer" else int(rng.integers(1, 14))
        rows = _rows(rng, m, d)
        want = encode_v5(rows, role).numpy().astype(np.float64)
        got = np.asarray(v5net.encode(rows, role))
        assert got.shape == (m, INPUT_DIM) == want.shape
        assert np.array_equal(got, want), f"role={role} d={d} m={m} 에서 입력이 갈린다"


def test_encode_casts_features_to_float32_like_v5():
    """★v5 는 특징을 float32 로 깎아서 넣는다 — `v5_cast=True`(기본) 가 그 반올림을 재현한다."""
    raw = [[1.0 / 3.0, 0.1 + 1e-12, 1234.5678901234, -2.718281828459045]]
    cast = np.asarray(v5net.encode(raw, "crane"))
    keep = np.asarray(v5net.encode(raw, "crane", v5_cast=False))
    want = encode_v5(raw, "crane").numpy().astype(np.float64)
    assert np.array_equal(cast, want)                       # v5 와 같은 값
    assert not np.array_equal(keep, want)                   # 끄면 float64 원값이 남는다 (학습 모드)
    assert np.array_equal(keep[0, :4], np.asarray(raw[0], np.float64))


def test_float32_input_promotes_to_the_same_answer():
    """★통합 규약 — 특징을 **float32 배열로 넘겨도** `v5_cast=True` 와 **비트 동일**한 답이 나온다.

    다른 담당(key=feat)의 `gpu/v5feat.as_net_input` 은 행을 float32 로 내린다. 그 결과를 float64 가중치에
    넣으면 jnp 이 다시 float64 로 올리므로, 값은 'v5 가 망에 넣는 float32 값'이고 계산은 float64 다 —
    즉 두 경로가 만난다. 이 시험이 그 접합을 못박는다 (그쪽 파일을 import 하지 않고 성질만 확인).
    """
    pol = _torch_policy(seed=3)
    p = _params(pol)
    rows = _rows(np.random.default_rng(4), 9)
    x_cast = v5net.encode(rows, "crane")                              # f32 로 깎고 f64 로 올린 것
    x_f32 = jnp.asarray(v5net.encode(rows, "crane", v5_cast=False), jnp.float32)
    assert x_f32.dtype == jnp.float32
    a_cast = np.asarray(v5net.actor_scores(p, x_cast))
    a_f32 = np.asarray(v5net.actor_scores(p, x_f32))
    assert a_f32.dtype == np.float64, "float32 입력 × float64 가중치는 float64 로 올라가야 한다"
    assert np.array_equal(a_cast, a_f32)
    assert np.array_equal(np.asarray(v5net.state_values(p, x_cast)),
                          np.asarray(v5net.state_values(p, x_f32)))


def test_encode_rejects_what_v5_rejects():
    """모양 규약은 v5 와 같은 자리에서 거절한다 (`model.py:15,25`)."""
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        v5net.encode(np.zeros((3, RAW_DIM + 1)), "crane")   # 너무 넓다
    with pytest.raises(ValueError):
        v5net.encode(np.zeros((0, 4)), "crane")             # 빈 행렬
    with pytest.raises(ValueError):
        v5net.encode(np.zeros((3, 4)), "buyer")             # buyer 는 두 줄
    with pytest.raises(ValueError):
        v5net.encode(np.zeros((2, 4)), "trucker")           # 없는 역할
    with pytest.raises(ValueError):
        v5net.encode(np.zeros((2, 2, 4)), "crane")          # 2차원이 아니다
    # v5 도 같은 입력을 거절하는가 (대조)
    for bad, role in (((np.zeros((3, RAW_DIM + 1))).tolist(), "crane"),
                      ([], "crane"), ((np.zeros((3, 4))).tolist(), "buyer")):
        with pytest.raises(ValueError):
            encode_v5(bad, role)
    assert bool(v5net.all_finite(_rows(rng, 3)))
    assert not bool(v5net.all_finite([[0.0, float("inf")]]))
    with pytest.raises(ValueError):                         # v5 는 비유한을 예외로 막는다
        encode_v5([[0.0, float("inf")]], "crane")


# ───────────────────────────────────────────────── ② state_dict 적재·왕복
def test_load_v5_params_transposes_and_upcasts():
    """torch `(out, in)` → 배열 `(in, out)`. 값은 하나도 안 바뀐다 (float32→float64 는 정확)."""
    pol = _torch_policy()
    sd = {k: v.detach().numpy() for k, v in pol.state_dict().items()}
    p = v5net.load_v5_params(sd)
    assert (p.w1.shape, p.b1.shape) == ((INPUT_DIM, 64), (64,))
    assert (p.w2.shape, p.b2.shape) == ((64, 64), (64,))
    assert (p.w_actor.shape, p.b_actor.shape) == ((64, 1), (1,))
    assert (p.w_critic.shape, p.b_critic.shape) == ((64, 1), (1,))
    assert all(leaf.dtype == jnp.float64 for leaf in p)
    assert (p.hidden, p.input_dim) == (64, INPUT_DIM)
    assert np.array_equal(np.asarray(p.w1), sd["trunk.0.weight"].T.astype(np.float64))
    assert np.array_equal(np.asarray(p.w_actor), sd["actor.weight"].T.astype(np.float64))
    assert np.array_equal(np.asarray(p.b_critic), sd["critic.bias"].astype(np.float64))


def test_state_dict_roundtrip_is_identity():
    """왕복 — state_dict → 배열 → state_dict 가 **비트 동일**하고, 되돌린 것을 torch 에 다시 실어도 같은 답."""
    pol = _torch_policy(seed=5)
    sd = pol.state_dict()
    p = v5net.load_v5_params(sd)
    back = v5net.to_v5_state_dict(p)
    for k in v5net.V5_STATE_KEYS:
        assert np.array_equal(back[k], sd[k].detach().numpy().astype(np.float64)), k
    twin = BlockPolicy(hidden=64)
    twin.load_state_dict({k: torch.as_tensor(v.copy(), dtype=torch.float32) for k, v in back.items()},
                         strict=True)
    rng = np.random.default_rng(6)
    rows = _rows(rng, 9)
    a0, v0 = _torch_forward(pol, rows, "crane", double=True)
    a1, v1 = _torch_forward(twin, rows, "crane", double=True)
    assert np.array_equal(a0, a1) and np.array_equal(v0, v1)
    # 배열판도 왕복 뒤 같은 잎
    p2 = v5net.load_v5_params(back)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(p, p2))


def test_load_v5_params_guards():
    """체크포인트 포장은 풀어 주고, 키·모양이 다르면 **조용히 넘기지 않고** 거절한다."""
    pol = _torch_policy(seed=7)
    sd = pol.state_dict()
    p = v5net.load_v5_params({"format": "yard-v5-shared-block-ppo-1", "policy": sd})
    assert np.array_equal(np.asarray(p.w1), sd["trunk.0.weight"].detach().numpy().T.astype(np.float64))
    with pytest.raises(KeyError):
        v5net.load_v5_params({k: v for k, v in sd.items() if k != "critic.bias"})
    bad = {k: v.detach().numpy() for k, v in sd.items()}
    bad["trunk.0.weight"] = bad["trunk.0.weight"][:, :36]
    with pytest.raises(ValueError):
        v5net.load_v5_params(bad)
    bad2 = {k: v.detach().numpy() for k, v in sd.items()}
    bad2["actor.bias"] = np.zeros((2,), np.float32)
    with pytest.raises(ValueError):
        v5net.load_v5_params(bad2)


def test_load_accepts_hidden_other_than_64():
    """은닉 폭은 체크포인트가 정한다 (`checkpoint.py:32`) — 32 로 만든 망도 그대로 실린다."""
    pol = _torch_policy(hidden=32, seed=8)
    p = _params(pol)
    assert p.hidden == 32 and p.w2.shape == (32, 32)
    rows = _rows(np.random.default_rng(9), 5)
    want, _ = _torch_forward(pol, rows, "crane", double=True)
    got = np.asarray(v5net.actor_scores(p, v5net.encode(rows, "crane")))
    assert np.abs(got - want).max() < 1e-12


def test_init_params_matches_torch_init_bounds():
    """새 가중치는 torch 기본 초기화와 **같은 분포**다 (경계 1/√fan_in · 편향도 같은 경계)."""
    p = v5net.init_params(jax.random.PRNGKey(0), hidden=64)
    assert all(leaf.dtype == jnp.float64 for leaf in p)
    assert (p.w1.shape, p.w2.shape, p.w_actor.shape) == ((INPUT_DIM, 64), (64, 64), (64, 1))
    assert float(jnp.abs(p.w1).max()) <= 1.0 / np.sqrt(INPUT_DIM)
    assert float(jnp.abs(p.b1).max()) <= 1.0 / np.sqrt(INPUT_DIM)
    assert float(jnp.abs(p.w_actor).max()) <= 1.0 / np.sqrt(64)
    with pytest.raises(ValueError):
        v5net.init_params(jax.random.PRNGKey(0), hidden=0)


def test_constants_mirror_v5():
    """상수를 v5 에서 베껴 두었으니(gpu/ 는 torch 무의존) **v5 가 바뀌면 여기서 걸린다**."""
    assert (v5net.RAW_DIM, v5net.INPUT_DIM, v5net.ROLES) == (RAW_DIM, INPUT_DIM, ROLES)
    assert v5net.HIDDEN == BlockPolicy().hidden
    assert tuple(v5net.V5_STATE_KEYS) == tuple(BlockPolicy().state_dict().keys())
    assert v5net.NEG_INF == -float("inf")


def test_shapes_and_dtypes():
    """모양·dtype 계약 — 동등성은 float64, 학습 모드는 float32 로 내려도 모양이 그대로다."""
    pol = _torch_policy(seed=10)
    rows = _rows(np.random.default_rng(33), 7)
    p64 = v5net.load_v5_params(pol.state_dict())
    x64 = v5net.encode(rows, "crane")
    assert x64.shape == (7, INPUT_DIM) and x64.dtype == jnp.float64
    assert v5net.trunk(p64, x64).shape == (7, 64)
    for f in (v5net.actor_scores, v5net.state_values):
        out = f(p64, x64)
        assert out.shape == (7,) and out.dtype == jnp.float64
    d64 = v5net.decide(p64, x64, np.ones(7, bool))
    assert d64.action.dtype == jnp.int32 and d64.action.shape == ()
    assert d64.probs.shape == (7,) and d64.entropy.shape == ()
    assert d64.mask_ok.dtype == jnp.bool_
    # 학습 모드 — 가중치·입력을 float32 로 (동등성 대조용이 아니다)
    p32 = v5net.load_v5_params(pol.state_dict(), dtype=jnp.float32)
    x32 = v5net.encode(rows, "crane", dtype=jnp.float32)
    assert all(leaf.dtype == jnp.float32 for leaf in p32) and x32.dtype == jnp.float32
    s32 = v5net.actor_scores(p32, x32)
    assert s32.dtype == jnp.float32 and s32.shape == (7,)
    assert v5net.entropy(s32, np.ones(7, bool)).dtype == jnp.float32
    # 같은 가중치니 float32 는 float64 의 반올림판 — 1e-6 안
    assert np.abs(np.asarray(s32, np.float64) - np.asarray(v5net.actor_scores(p64, x64))).max() < 1e-6


# ───────────────────────────────────────────────── ③ 같은 수를 내는가 (무작위 500건)
def _sweep(policy, p, rng, n, *, double, role="crane"):
    worst_a = worst_r = 0.0
    flips = 0
    gaps = []
    for _ in range(n):
        m = int(rng.integers(1, 14))
        rows = _rows(rng, m)
        want_a, want_v = _torch_forward(policy, rows, role, double=double)
        x = v5net.encode(rows, role)
        got_a = np.asarray(v5net.actor_scores(p, x))
        got_v = np.asarray(v5net.state_values(p, x))
        for want, got in ((want_a, got_a), (want_v, got_v)):
            d = np.abs(want - got)
            worst_a = max(worst_a, float(d.max()))
            worst_r = max(worst_r, float((d / np.maximum(np.abs(want), 1e-12)).max()))
        if int(want_a.argmax()) != int(got_a.argmax()):
            flips += 1
        if m > 1:
            s = np.sort(want_a)[::-1]
            gaps.append(float(s[0] - s[1]))
    return worst_a, worst_r, flips, gaps


def test_actor_and_critic_match_torch_float64():
    """★무작위 500건 — torch 를 float64 로 올려 actor 점수·critic 값을 대조한다."""
    pol = _torch_policy(seed=11)
    p = _params(pol)
    rng = np.random.default_rng(12)
    wa, wr, flips, gaps = _sweep(pol, p, rng, 500, double=True)
    _report("torch f64 vs 배열 f64", wa, wr, 500, flips)
    print(f"    (1·2위 점수 격차 최소 {min(gaps):.3e} — 이보다 |Δ| 가 작으면 결정은 안 바뀐다)")
    assert flips == 0, "float64 에서 결정이 갈렸다 — 동률 사례를 보고해야 한다"
    assert wa < 1e-12, f"최대 |Δ| {wa:.3e} — tanh·GEMM 차이(≈1e-15)를 넘는다"
    assert wr < 1e-9


def test_float32_gap_is_measured_not_assumed():
    """v5 가 **실제로 돌리는** float32 와의 격차를 재서 남긴다 (학습 모드 참고용).

    v5 는 망·입력이 float32 다 (`model.py:14`). 배열판 float64 와는 원리상 1e-7 급 차이가 난다 —
    허용오차를 고를 근거이자, 결정이 갈리는 조건(점수 격차 < 1e-7)의 경고다.
    """
    pol = _torch_policy(seed=11)
    p = _params(pol)
    rng = np.random.default_rng(13)
    wa, wr, flips, gaps = _sweep(pol, p, rng, 200, double=False)
    _report("torch f32(v5 실제) vs 배열 f64", wa, wr, 200, flips)
    assert flips == 0, f"float32 대조에서 결정 {flips}건이 갈렸다 — 점수 격차 최소 {min(gaps):.3e}"
    assert wa < 1e-4, f"float32 격차 {wa:.3e} 가 너무 크다 — 적재·전치를 의심하라"


@pytest.mark.parametrize("role", ROLES)
def test_all_roles_match(role):
    """역할 4종 모두 (state 는 블록 상태 행렬 — critic 이 블록마다 값 하나)."""
    pol = _torch_policy(seed=14)
    p = _params(pol)
    rng = np.random.default_rng(15)
    for _ in range(25):
        m = 2 if role == "buyer" else int(rng.integers(1, 22))
        rows = _rows(rng, m, 8 if role == "state" else CRANE_RAW)
        want_a, want_v = _torch_forward(pol, rows, role, double=True)
        x = v5net.encode(rows, role)
        assert np.abs(np.asarray(v5net.actor_scores(p, x)) - want_a).max() < 1e-12
        assert np.abs(np.asarray(v5net.state_values(p, x)) - want_v).max() < 1e-12


# ───────────────────────────────────────────────── ④ 분포·결정·빈 마스크
def test_distribution_matches_torch_categorical():
    """마스크 씌운 분포 — 로그확률·확률·엔트로피·`log_prob(action)` 을 torch `Categorical` 과 대조."""
    pol = _torch_policy(seed=16).double()
    p = _params(pol)
    rng = np.random.default_rng(17)
    worst = {"logits": 0.0, "probs": 0.0, "entropy": 0.0, "logp": 0.0}
    flips = 0
    for _ in range(500):
        m = int(rng.integers(1, 14))
        rows = _rows(rng, m)
        mk = _mask(rng, m)
        x5 = encode_v5(rows, "crane").double()
        with torch.no_grad():
            dist = pol.distribution(x5, torch.as_tensor(mk))
            want_lp = dist.logits.numpy().astype(np.float64)
            want_p = dist.probs.numpy().astype(np.float64)
            want_e = float(dist.entropy())
            want_a = int(dist.probs.argmax())
            want_logp = float(dist.log_prob(torch.tensor(want_a)))
        x = v5net.encode(rows, "crane")
        d = v5net.decide(p, x, mk)
        got_lp = np.asarray(v5net.log_probs(v5net.actor_scores(p, x), mk))
        worst["logits"] = max(worst["logits"], float(np.abs(got_lp[mk] - want_lp[mk]).max()))
        worst["probs"] = max(worst["probs"], float(np.abs(np.asarray(d.probs) - want_p).max()))
        worst["entropy"] = max(worst["entropy"], abs(float(d.entropy) - want_e))
        worst["logp"] = max(worst["logp"], abs(float(d.log_prob) - want_logp))
        if int(d.action) != want_a:
            flips += 1
        assert bool(d.mask_ok) and bool(mk[int(d.action)])
        assert np.array_equal(np.asarray(d.probs)[~mk], np.zeros(int((~mk).sum())))
        assert np.isneginf(np.asarray(d.scores)[~mk]).all()
    pol.float()
    print("    분포 대조 최대 |Δ|: " + "  ".join(f"{k}={v:.3e}" for k, v in worst.items())
          + f"  argmax 불일치={flips}")
    assert flips == 0
    assert max(worst.values()) < 1e-12


def test_masked_rows_never_chosen_and_never_leak():
    """마스크 칸은 못 뽑히고, 그 칸 특징을 아무렇게나 바꿔도 **살아 있는 칸의 답이 안 바뀐다**."""
    pol = _torch_policy(seed=18)
    p = _params(pol)
    rng = np.random.default_rng(19)
    rows = np.asarray(_rows(rng, 10))
    mk = np.ones(10, bool)
    mk[[2, 5, 9]] = False
    a0 = v5net.decide(p, v5net.encode(rows, "crane"), mk)
    assert bool(mk[int(a0.action)])
    rows2 = rows.copy()
    rows2[[2, 5, 9]] = 1e6                                  # 죽은 칸을 극단값으로
    a1 = v5net.decide(p, v5net.encode(rows2, "crane"), mk)
    assert int(a1.action) == int(a0.action)
    assert np.array_equal(np.asarray(a1.probs)[mk], np.asarray(a0.probs)[mk])
    assert float(a1.entropy) == float(a0.entropy)
    assert np.array_equal(np.asarray(a1.scores)[mk], np.asarray(a0.scores)[mk])
    assert np.isneginf(np.asarray(a1.scores)[~mk]).all()
    # 죽은 칸에 nan/inf 가 들어와도 새어나오지 않는다 (v5 는 encode 에서 예외로 막는 자리)
    rows3 = rows.copy().astype(np.float64)
    rows3[2] = np.nan
    a2 = v5net.decide(p, v5net.encode(rows3, "crane"), mk)
    assert int(a2.action) == int(a0.action) and np.isfinite(float(a2.entropy))


def test_empty_mask_is_flagged_not_raised():
    """빈 마스크 — v5 는 예외를 던진다. 배열판은 `mask_ok=False`·행동 -1 로 알려 준다 (전원 WAIT 규칙)."""
    pol = _torch_policy(seed=20).double()
    p = _params(pol)
    rng = np.random.default_rng(21)
    rows = _rows(rng, 6)
    dead = np.zeros(6, bool)
    with pytest.raises(ValueError):                         # v5 거동 (model.py:45-46)
        pol.distribution(encode_v5(rows, "crane").double(), torch.as_tensor(dead))
    pol.float()
    d = v5net.decide(p, v5net.encode(rows, "crane"), dead)
    assert not bool(d.mask_ok)
    assert int(d.action) == -1
    assert np.array_equal(np.asarray(d.probs), np.zeros(6))
    assert np.isneginf(np.asarray(d.scores)).all()
    assert float(d.entropy) == 0.0                          # nan 이 아니어야 한다
    assert np.isneginf(float(d.log_prob))
    assert np.isfinite(np.asarray(d.value)).all()           # critic 은 마스크와 무관
    assert int(v5net.greedy_policy_fn(p, v5net.encode(rows, "crane")[None], dead[None])[0]) == -1


def test_sample_action_stays_inside_mask():
    """추첨은 v5 표본과 같을 수 없다(다른 난수기) — 최소한 **마스크 안**에서만 뽑아야 한다."""
    pol = _torch_policy(seed=22)
    p = _params(pol)
    rng = np.random.default_rng(23)
    rows = _rows(rng, 8)
    mk = np.zeros(8, bool)
    mk[[1, 4, 6]] = True
    scores = v5net.actor_scores(p, v5net.encode(rows, "crane"))
    drawn = {int(v5net.sample_action(jax.random.PRNGKey(i), scores, mk)) for i in range(200)}
    assert drawn <= {1, 4, 6} and len(drawn) > 1
    assert int(v5net.sample_action(jax.random.PRNGKey(0), scores, np.zeros(8, bool))) == -1


# ───────────────────────────────────────────────── ⑤ jit · vmap · 배치
def test_jit_matches_eager_bitwise():
    """jit 한 것과 안 한 것이 **비트 동일** (마스크는 자료, 모양은 정적)."""
    pol = _torch_policy(seed=24)
    p = _params(pol)
    rng = np.random.default_rng(25)
    fast = jax.jit(v5net.decide)
    for _ in range(20):
        m = int(rng.integers(1, 14))
        rows = _rows(rng, m)
        mk = _mask(rng, m)
        x = v5net.encode(rows, "crane")
        a, b = v5net.decide(p, x, mk), fast(p, x, mk)
        for name, u, v in zip(v5net.V5Decision._fields, a, b):
            assert np.array_equal(np.asarray(u), np.asarray(v)), name


def test_batch_matches_one_at_a_time():
    """vmap·브로드캐스트·낱개가 같은 답 — `policy_fn` 규약 `(params, (K,M,37), (K,M)) → (K,)`.

    ★**결정·마스크는 비트 동일**이 조건이다. 실수 잎은 XLA 가 배치 행렬곱을 낱개와 다른 커널로 묶으므로
    마지막 비트가 갈릴 수 있다 (실측: `value` 가 1 ulp) — 갈린 잎과 |Δ| 를 찍어 증거로 남긴다.
    """
    pol = _torch_policy(seed=26)
    p = _params(pol)
    rng = np.random.default_rng(27)
    K, M = 4, 11
    xs, mks = [], []
    for _ in range(K):
        rows = _rows(rng, M)
        xs.append(np.asarray(v5net.encode(rows, "crane")))
        mks.append(_mask(rng, M))
    X, MK = jnp.asarray(np.stack(xs)), jnp.asarray(np.stack(mks))
    one = [v5net.decide(p, xs[k], mks[k]) for k in range(K)]
    vm = v5net.decide_batch(p, X, MK)                       # vmap 판
    bc = v5net.decide(p, X, MK)                             # 마지막 축 규약 (브로드캐스트)
    drift = {}
    for f in v5net.V5Decision._fields:
        want = np.stack([np.asarray(getattr(o, f)) for o in one])
        _same(f"vmap.{f}", np.asarray(getattr(vm, f)), want, drift, exact=f in ("action", "mask_ok"))
        _same(f"bcast.{f}", np.asarray(getattr(bc, f)), want, drift, exact=f in ("action", "mask_ok"))
    pick = np.asarray(v5net.greedy_policy_fn(p, X, MK))
    assert pick.dtype == np.int32 and pick.shape == (K,)
    assert np.array_equal(pick, np.stack([np.asarray(o.action) for o in one]))
    _same("actor_scores_batch", np.asarray(v5net.actor_scores_batch(p, X)),
          np.stack([np.asarray(v5net.actor_scores(p, xs[k])) for k in range(K)]), drift)
    _same("state_values_batch", np.asarray(v5net.state_values_batch(p, X)),
          np.stack([np.asarray(v5net.state_values(p, xs[k])) for k in range(K)]), drift)
    _same("greedy_action_batch", np.asarray(v5net.greedy_action_batch(
        v5net.actor_scores_batch(p, X), MK)), pick, drift, exact=True)
    print("    배치 vs 낱개 — 비트 일치 아닌 잎: " + (", ".join(f"{k} |Δ|={v:.3e}" for k, v in drift.items())
                                                    or "없음"))


def test_rows_are_independent_across_cranes():
    """★`dispatch.py:25-28` 이 요구하는 성질 — 행 k 의 답은 **다른 행 마스크**에 의존하지 않는다."""
    pol = _torch_policy(seed=28)
    p = _params(pol)
    rng = np.random.default_rng(29)
    K, M = 3, 9
    X = jnp.asarray(np.stack([np.asarray(v5net.encode(_rows(rng, M), "crane")) for _ in range(K)]))
    MK = np.stack([_mask(rng, M) for _ in range(K)])
    base = np.asarray(v5net.greedy_policy_fn(p, X, jnp.asarray(MK)))
    for k in range(1, K):
        alt = MK.copy()
        alt[k] = False                                      # 다른 행을 죽여 본다
        alt[k, 0] = True
        got = np.asarray(v5net.greedy_policy_fn(p, X, jnp.asarray(alt)))
        assert got[0] == base[0], "행 0 의 답이 행 k 의 마스크에 끌려 바뀌었다"


# ───────────────────────────────────────────────── ⑤-2 기울기 (조각 8 학습 루프가 쓸 경로)
def test_gradients_match_torch_autograd():
    """★PPO 손실의 **기울기**도 torch 자동미분과 같다 — 조각 8 이 이 망으로 학습할 수 있다는 뜻.

    손실은 `ppo/update.py:63` 과 같은 꼴: `-logπ(a)·우위 + 0.5·(V−목표)² − 0.001·엔트로피`.
    (클리핑·미니배치는 조각 8 몫이고, 여기서는 **망을 지나는 기울기**만 대조한다.)
    """
    pol = _torch_policy(seed=40).double()
    p = _params(pol)
    rng = np.random.default_rng(41)
    rows, mk = _rows(rng, 9), _mask(rng, 9)
    act, adv, target = 0, 1.7, 0.25
    while not mk[act]:
        act += 1

    x5 = encode_v5(rows, "crane").double()
    dist = pol.distribution(x5, torch.as_tensor(mk))
    loss = (-dist.log_prob(torch.tensor(act)) * adv
            + 0.5 * (pol.value(x5)[0] - target) ** 2 - 0.001 * dist.entropy())
    pol.zero_grad()
    loss.backward()
    want = {k: v.grad.detach().numpy().astype(np.float64) for k, v in pol.named_parameters()}
    want_loss = float(loss.detach())
    pol.float()

    def jax_loss(params):
        x = v5net.encode(rows, "crane")
        s = v5net.actor_scores(params, x)
        return (-v5net.log_prob_of(s, mk, act) * adv
                + 0.5 * (v5net.state_values(params, x)[0] - target) ** 2
                - 0.001 * v5net.entropy(s, mk))

    assert abs(float(jax_loss(p)) - want_loss) < 1e-12
    g = jax.grad(jax_loss)(p)
    pairs = (("trunk.0.weight", g.w1.T), ("trunk.0.bias", g.b1), ("trunk.2.weight", g.w2.T),
             ("trunk.2.bias", g.b2), ("actor.weight", g.w_actor.T), ("actor.bias", g.b_actor),
             ("critic.weight", g.w_critic.T), ("critic.bias", g.b_critic))
    worst = 0.0
    for name, got in pairs:
        got = np.asarray(got)
        assert got.shape == want[name].shape, name
        assert np.isfinite(got).all(), name
        worst = max(worst, float(np.abs(got - want[name]).max()))
    print(f"    기울기 대조 (8장) 최대 |Δ|={worst:.3e}  · |∂loss/∂actor.w| 합={np.abs(want['actor.weight']).sum():.4f}")
    assert worst < 1e-11, f"기울기가 갈린다 최대 |Δ|={worst:.3e}"


# ───────────────────────────────────────────────── ⑥ 실제 학습된 v5 가중치
@pytest.mark.skipif(not REAL_CKPTS, reason="outputs/v5/*/policy.pt 없음 (git 무시 경로)")
@pytest.mark.parametrize("ckpt", REAL_CKPTS)
def test_real_v5_checkpoint_matches(ckpt):
    """★무작위 초기화가 아니라 **실제 학습된 v5 체크포인트**로 같은 대조 — 조각 8 이 쓸 경로 그대로."""
    from yard_rl.v6.ppo.checkpoint import load_policy
    pol = load_policy(ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    p_direct = v5net.load_v5_params(raw)                    # 체크포인트 통째로 (포장 풀기)
    p = _params(pol)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(p, p_direct))
    rng = np.random.default_rng(31)
    wa, wr, flips, gaps = _sweep(pol, p, rng, 200, double=True)
    _report(f"실제 체크포인트 f64 {os.path.basename(os.path.dirname(ckpt))}", wa, wr, 200, flips)
    print(f"    (1·2위 점수 격차 최소 {min(gaps):.3e})")
    assert flips == 0 and wa < 1e-12
    wa32, wr32, flips32, _ = _sweep(pol, p, np.random.default_rng(32), 200, double=False)
    _report(f"실제 체크포인트 f32 {os.path.basename(os.path.dirname(ckpt))}", wa32, wr32, 200, flips32)
    assert flips32 == 0
