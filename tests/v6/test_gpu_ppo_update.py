"""배열판 PPO 갱신(`gpu/ppo_update.py`)이 v5 `ppo/update.update` 와 **같은 갱신**을 하는가 ([[YR-327]] 조각 8).

■ 무엇을 지키나 — 기대값은 손으로 적지 않는다. **v5 `update` 를 실제로 불러** 같은 테이프로 굴린 답과 맞춘다.
  ① 최적화기 — torch `Adam` 한 걸음·`clip_grad_norm_` 과 **float64 에서 1e-14 안** (수식·순서를 그대로 옮겼다)
     ★기울기가 없는 파라미터(`p.grad is None`)를 건너뛰는 규칙까지 — Adam 걸음 수가 파라미터마다 다르다
  ② optax 대조 — `optax.adam` 은 수식이 같아 1e-15 안에서 같고, `optax.clip_by_global_norm` 은
     torch 의 분모 보정 1e-6 이 없어 **상대 2e-6 어긋난다** (그래서 손으로 썼다)
  ③ 이득 정규화 — v5 의 `len(active) > 1 and std > 1e-8` 조건과 numpy ddof 0 표준편차
  ④ ★갱신 한 번 전체 — 미니배치별 손실·KL·기울기 노름, 표본별 로그비·이득, 최종 파라미터
  ⑤ ★조기중단이 **실제로 발동하는** 테이프 — 첫 배치·중간 배치·둘째 epoch 세 곳에서
     미니배치 수·중단 여부·중단 배치의 KL 기록까지 v5 와 같은지
  ⑥ 활성 표본이 **하나도 없는** 미니배치 — 행동점수 머리가 얼어 있는지 (v5 의 `p.grad is None`)
  ⑦ 난수 규약 — 미니배치 순열은 호스트가 v5 난수열로 만들고, 조기중단 뒤에는 되감아 v5 상태와 같아진다

■ ⚠️ 비트 일치는 **불가능**하다 — v5 정책망은 float32 다 (`gpu/v5net.py` 머리말 ★ · tanh·GEMM 차이까지).
  그래서 손실·KL·노름은 rtol 1e-4, 최종 파라미터는 atol 1e-5 로 본다. 실측값은 `test_zz_report` 가
  매번 다시 재서 화면에 찍는다 (`-s` 로 볼 수 있다).

실행 (Git Bash · Windows · CPU x64):
    PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 .venv-jax/Scripts/python.exe \
        -m pytest tests/v6/test_gpu_ppo_update.py -q -s -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np                       # ★torch 보다 먼저 — 아나콘다 MKL 의 libiomp 먼저 (OMP #15 회피)
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — 갱신 대조는 x64 에서만 뜻이 있다
jnp = jax.numpy
torch = pytest.importorskip("torch")

from yard_rl.v6.gpu import ppo_update as PU                       # noqa: E402
from yard_rl.v6.gpu import v5net                                  # noqa: E402
from yard_rl.v6.gpu.v5net import V5PolicyParams                   # noqa: E402
from yard_rl.v6.ppo.buffer import Choice, Interval, gae           # noqa: E402
from yard_rl.v6.ppo.model import BlockPolicy, encode              # noqa: E402
from yard_rl.v6.ppo.runtime import PPOConfig                      # noqa: E402
from yard_rl.v6.ppo.update import update as v5_update             # noqa: E402

#: 화면에 찍을 실측 (마지막 시험이 모아 보여 준다)
MEASURED: dict[str, float] = {}

#: ★KL 의 **절대** 허용오차. v5 는 `exp(r) − 1 − r` 을 **float32** 로 계산한다 (update.py:53).
#:   로그비 r 이 작으면 `exp(r)` ≈ 1 이라 `−1` 에서 **상쇄**가 일어나 float32 한 칸(1.19e-07)의
#:   절반(≈6e-08)이 그대로 오차로 남는다 — 표본 수십 개를 평균해도 1e-08 대다.
#:   즉 작은 KL 은 **v5 쪽이 부정확**하고 배열판(float64)이 정확하다. 상대 오차로 재면
#:   KL 1.2e-05 에서 4e-04 까지 벌어지므로 절대값으로 본다 (실측 최대 2.0e-08).
KL_ATOL = 2e-7


def _note(key, value):
    MEASURED[key] = max(float(value), MEASURED.get(key, 0.0))
    return value


def _rel(a, b):
    """상대 어긋남 |a−b| / max(|b|, 1e-12)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))) if a.size else 0.0


# ═══════════════════════════════════════════════ ① 최적화기 — torch 와 같은 수인가
SHAPES = dict(w1=(37, 8), b1=(8,), w2=(8, 8), b2=(8,), w_actor=(8, 1), b_actor=(1,),
              w_critic=(8, 1), b_critic=(1,))


def _tree(rng, scale=1.0):
    return V5PolicyParams(**{k: jnp.asarray(rng.normal(size=s) * scale) for k, s in SHAPES.items()})


def _torch_pair(tree):
    """같은 값의 torch float64 파라미터 8장 (순서는 `V5PolicyParams` 와 같다)."""
    return [torch.tensor(np.asarray(v), dtype=torch.float64, requires_grad=True) for v in tree]


def test_adam_matches_torch_including_untouched_parameters():
    """torch `Adam` 과 같은 수. ★기울기가 없는 파라미터는 **아무것도 안 바뀐다** (걸음 수까지)."""
    rng = np.random.default_rng(11)
    hyper = PU.PPOHyper()
    start = _tree(rng, 0.3)
    tp = _torch_pair(start)
    opt = torch.optim.Adam(tp, lr=hyper.learning_rate)
    params, state = start, PU.init_adam(start)
    # 5 걸음 — 3·4번째 걸음에서는 행동점수 머리(색인 4·5)에 기울기가 없다
    for step in range(5):
        grads = _tree(rng, 0.2)
        on = step not in (2, 3)
        touched = V5PolicyParams(*[jnp.asarray(True)] * 4, jnp.asarray(on), jnp.asarray(on),
                                 jnp.asarray(True), jnp.asarray(True))
        for k, (t, g) in enumerate(zip(tp, grads)):
            t.grad = None if (k in (4, 5) and not on) else torch.tensor(np.asarray(g),
                                                                       dtype=torch.float64)
        opt.step()
        params, state = PU.adam_step(params, state, grads, touched, hyper)
        for k, (t, p) in enumerate(zip(tp, params)):
            d = _rel(np.asarray(p), t.detach().numpy())
            _note("adam_rel", d)
            assert d < 1e-13, f"걸음 {step} 파라미터 {k}: 상대 {d:.3e}"
    # 걸음 수: 행동점수 머리만 2번 건너뛰었다
    counts = [int(c) for c in state.count]
    assert counts == [5, 5, 5, 5, 3, 3, 5, 5]
    assert [int(opt.state[t]["step"]) for t in tp] == counts


def test_adam_matches_optax_and_optax_clip_differs_from_torch():
    """optax 대조 — `adam` 은 같고 `clip_by_global_norm` 은 torch 의 1e-6 보정이 없어 다르다."""
    optax = pytest.importorskip("optax")
    rng = np.random.default_rng(12)
    hyper = PU.PPOHyper()
    params = _tree(rng, 0.3)
    mine, state = params, PU.init_adam(params)
    tx = optax.adam(hyper.learning_rate, b1=hyper.adam_b1, b2=hyper.adam_b2, eps=hyper.adam_eps)
    tx_state, tx_params = tx.init(params), params
    on = V5PolicyParams(*[jnp.asarray(True)] * 8)
    for _ in range(4):
        grads = _tree(rng, 0.2)
        mine, state = PU.adam_step(mine, state, grads, on, hyper)
        upd, tx_state = tx.update(grads, tx_state, tx_params)
        tx_params = optax.apply_updates(tx_params, upd)
        d = max(_rel(np.asarray(a), np.asarray(b)) for a, b in zip(mine, tx_params))
        _note("adam_vs_optax_rel", d)
        assert d < 1e-14, f"optax.adam 과 상대 {d:.3e}"
    # 클리핑: 노름이 max 를 **살짝** 넘게 맞춰(0.51) 실제 학습에서 나오는 크기로 비교한다
    raw = _tree(rng, 0.2)
    scale = 0.51 / float(PU.global_norm(raw))
    big = jax.tree.map(lambda g: g * scale, raw)
    norm = PU.global_norm(big)
    assert float(norm) > hyper.max_grad_norm
    torch_side = PU.clip_by_global_norm_torch(big, norm, hyper)
    optax_side, _ = optax.clip_by_global_norm(hyper.max_grad_norm).update(
        big, optax.clip_by_global_norm(hyper.max_grad_norm).init(big))
    gap = max(_rel(np.asarray(a), np.asarray(b)) for a, b in zip(torch_side, optax_side))
    _note("optax_clip_gap_rel", gap)
    # 차이의 정체는 torch 분모의 1e-6 하나다 — 식으로 못박는다
    want = hyper.clip_eps / (float(norm) + hyper.clip_eps)
    assert np.isclose(gap, want, rtol=1e-6), f"optax 클립 차이 {gap:.3e} vs 식 {want:.3e}"
    assert gap > 1e-6, f"노름 {float(norm):.3f} 에서 상대 차이가 ≈2e-6 이어야 한다: {gap:.3e}"
    # 그리고 torch 쪽 식은 실제 torch 와 같다
    tp = _torch_pair(big)
    for t, g in zip(tp, big):
        t.grad = torch.tensor(np.asarray(g), dtype=torch.float64)
    tn = torch.nn.utils.clip_grad_norm_(tp, hyper.max_grad_norm, error_if_nonfinite=True)
    _note("clip_norm_rel", _rel(float(norm), float(tn)))
    assert _rel(float(norm), float(tn)) < 1e-14
    for k, (t, g) in enumerate(zip(tp, torch_side)):
        d = _rel(np.asarray(g), t.grad.numpy())
        _note("clip_grad_rel", d)
        assert d < 1e-14, f"클립된 기울기 {k}: 상대 {d:.3e}"


def test_global_norm_matches_torch_on_unclipped_grads():
    """노름은 '파라미터별 노름을 다시 노름' 이다 — 클립이 필요 없는 작은 기울기에서도 같다."""
    rng = np.random.default_rng(13)
    small = _tree(rng, 1e-3)
    tp = _torch_pair(small)
    for t, g in zip(tp, small):
        t.grad = torch.tensor(np.asarray(g), dtype=torch.float64)
    tn = float(torch.nn.utils.clip_grad_norm_(tp, PU.PPOHyper().max_grad_norm))
    assert _rel(float(PU.global_norm(small)), tn) < 1e-14
    for t, g in zip(tp, small):                     # coef = 1 이라 값이 그대로다
        assert np.array_equal(t.grad.numpy(), np.asarray(g))


# ═══════════════════════════════════════════════ ③ 이득 정규화 (update.py:19-27)
@pytest.mark.parametrize("n_active,expect", [(0, False), (1, False), (7, True)])
def test_advantage_normalization_condition_and_value(n_active, expect):
    """활성 표본이 2개 이상이고 표준편차가 1e-8 을 넘을 때만 정규화한다 — numpy 와 같은 값으로."""
    rng = np.random.default_rng(14 + n_active)
    e = 20
    adv = rng.normal(size=e) * 2.0
    nch = np.zeros(e, np.int32)
    nch[rng.choice(e, size=n_active, replace=False)] = 1
    out, mean, std, applied = PU.normalize_advantage(jnp.asarray(adv), jnp.asarray(nch))
    assert bool(applied) is expect
    if expect:
        a = adv[nch > 0]
        _note("adv_mean_rel", _rel(float(mean), float(a.mean())))
        _note("adv_std_rel", _rel(float(std), float(a.std())))
        assert _rel(float(mean), float(a.mean())) < 1e-14
        assert _rel(float(std), float(a.std())) < 1e-14
        _note("adv_norm_rel", _rel(np.asarray(out), (adv - a.mean()) / a.std()))
        assert np.allclose(np.asarray(out), (adv - a.mean()) / a.std(), rtol=1e-13, atol=0)
    else:
        assert np.array_equal(np.asarray(out), adv)


def test_advantage_normalization_skipped_when_std_is_tiny():
    """표준편차가 1e-8 이하면 v5 는 정규화하지 않는다 (update.py:25)."""
    adv = np.full(9, 3.0) + np.arange(9) * 1e-12
    nch = np.ones(9, np.int32)
    out, _, std, applied = PU.normalize_advantage(jnp.asarray(adv), jnp.asarray(nch))
    assert float(std) <= 1e-8 and bool(applied) is False
    assert np.array_equal(np.asarray(out), adv)


# ═══════════════════════════════════════════════ 테이프 만들기 (v5 객체 → 배열)
def _fresh_policy(seed=17):
    torch.manual_seed(seed)
    return BlockPolicy()


def _build_intervals(rng, policy, *, n_intervals, n_blocks, log_ratio_of,
                     cmax_draw=3, amax_draw=5, active_of=None):
    """v5 `Interval` 목록 + 부트스트랩.

    `log_ratio_of(e)` 가 표본 e 의 **원하는 로그비**다 — 수집 로그확률을 `현재값 − 로그비` 로 적어 둔다
    (그러면 첫 epoch 의 `log_ratio = 현재값 − 기록값` 이 정확히 그 값이 된다).
    `active_of(e)` 가 거짓이면 그 표본은 결정이 하나도 없다 (가치만 배운다).
    """
    intervals, roles = [], ("seller", "crane", "buyer")
    for i in range(n_intervals):
        states = encode((rng.normal(size=(n_blocks, 12)) * 0.7).tolist(), "state")
        with torch.no_grad():
            values = policy.value(states).numpy().copy()
        per_block = []
        for b in range(n_blocks):
            e = i * n_blocks + b
            want = True if active_of is None else bool(active_of(e))
            n = int(rng.integers(1, cmax_draw + 1)) if want else 0
            if want and rng.random() < 0.25:
                n = 0                                   # 결정이 없는 블록도 섞는다 (v5 에 흔하다)
            cs = []
            for _ in range(n):
                role = roles[int(rng.integers(0, 3))]
                m = 2 if role == "buyer" else int(rng.integers(2, amax_draw + 1))
                x = encode((rng.normal(size=(m, 10)) * 0.8).tolist(), role)
                mk = torch.as_tensor(np.asarray(rng.random(m) > 0.25), dtype=torch.bool)
                mk[int(rng.integers(0, m))] = True       # 최소 한 칸은 열려 있다 (model.py:45-46)
                with torch.no_grad():
                    dist = policy.distribution(x, mk)
                    a = int(dist.probs.argmax())         # ★추첨 아님 — 동등성은 argmax 축 (규칙 ⑥)
                    lp = float(dist.log_prob(torch.tensor(a)))
                cs.append(Choice(role, i * 60.0, x, mk, a, lp - float(log_ratio_of(e)) / max(n, 1)))
            per_block.append(cs)
        intervals.append(Interval(i * 60.0, (i + 1) * 60.0, states, values, per_block,
                                  float(rng.normal() * 0.3), False))
    with torch.no_grad():
        bootstrap = policy.value(encode((rng.normal(size=(n_blocks, 12)) * 0.7).tolist(),
                                        "state")).numpy().copy()
    return intervals, bootstrap


def _pad_tape(intervals, *, mask_pad_true=True):
    """v5 `Interval` 목록 → `PU.Tape` (표본 칸 e = i*B + b — update.py:19-20 순서 그대로)."""
    r, b_n = len(intervals), len(intervals[0].values)
    cmax = max(1, max(len(c) for row in intervals for c in row.choices))
    amax = max(1, max(int(c.rows.shape[0]) for row in intervals for cs in row.choices for c in cs))
    e_n = r * b_n
    states = np.zeros((e_n, 37)); rows = np.zeros((e_n, cmax, amax, 37))
    mask = np.zeros((e_n, cmax, amax), bool); action = np.zeros((e_n, cmax), np.int32)
    old = np.zeros((e_n, cmax)); nch = np.zeros(e_n, np.int32)
    for i, row in enumerate(intervals):
        st = row.states.detach().numpy()
        for b in range(b_n):
            e = i * b_n + b
            states[e] = st[b]
            cs = row.choices[b]
            nch[e] = len(cs)
            for j in range(cmax):
                if j < len(cs):
                    c = cs[j]
                    m = int(c.rows.shape[0])
                    rows[e, j, :m] = c.rows.detach().numpy()
                    mask[e, j, :m] = c.mask.numpy()
                    action[e, j], old[e, j] = c.action, c.log_prob
                elif mask_pad_true:
                    mask[e, j, 0] = True                 # 패딩 규약 (머리말 ⚠️)
    return PU.Tape(states=jnp.asarray(states), rows=jnp.asarray(rows), mask=jnp.asarray(mask),
                   action=jnp.asarray(action), old_logp=jnp.asarray(old), n_choices=jnp.asarray(nch))


class _Probe:
    """v5 `update` 내부를 **가로채 기록**한다 — 실제 계산은 v5 그대로 지나간다.

    미니배치 경계는 `torch.autograd.backward` 호출로 갈린다 (미니배치마다 정확히 한 번).
    조기중단된 미니배치는 `backward` 가 없으므로 마지막 묶음이 그것이다.
    """

    def __init__(self):
        self.events = []

    def install(self, monkeypatch):
        import yard_rl.v6.ppo.update as mod
        real_surr, real_bwd = mod.clipped_surrogate, torch.autograd.backward
        real_clip = torch.nn.utils.clip_grad_norm_

        def surr(log_ratio, advantage, clip):
            self.events.append(("surr", float(log_ratio.detach()), float(advantage)))
            return real_surr(log_ratio, advantage, clip)

        def bwd(tensors, *a, **k):
            self.events.append(("bwd", float(tensors.detach()), 0.0))
            return real_bwd(tensors, *a, **k)

        def clip(params, max_norm, **k):
            n = real_clip(params, max_norm, **k)
            self.events.append(("clip", float(n), 0.0))
            return n

        monkeypatch.setattr(mod, "clipped_surrogate", surr)
        monkeypatch.setattr(torch.autograd, "backward", bwd)
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
        return self

    def minibatches(self):
        out, lrs, advs = [], [], []
        for kind, a, b in self.events:
            if kind == "surr":
                lrs.append(a); advs.append(b)
            elif kind == "bwd":
                out.append({"log_ratio": lrs, "adv": advs, "loss": a, "norm": None})
                lrs, advs = [], []
            else:
                out[-1]["norm"] = a
        if lrs:                                     # 조기중단된 미니배치 (backward 없음)
            out.append({"log_ratio": lrs, "adv": advs, "loss": None, "norm": None})
        return out


def _kl_of(log_ratios):
    a = np.asarray(log_ratios, np.float64)
    return float(np.mean(np.exp(a) - 1.0 - a)) if a.size else 0.0


def _array_active_slots(tape, orders_row):
    """배열 미니배치 한 줄에서 **v5 가 도는 순서**의 활성 칸 번호 (빈 칸·결정 없는 칸 제외)."""
    nch = np.asarray(tape.n_choices)
    return [k for k, e in enumerate(np.asarray(orders_row)) if e >= 0 and nch[e] > 0]


def _run_pair(monkeypatch, *, seed, n_intervals, n_blocks, log_ratio_of, config,
              active_of=None, v5_cast=True, mask_pad_true=True, rng_seed=505):
    """같은 테이프로 v5 와 배열판을 한 번씩 굴려 (v5 보고, 미니배치 기록, 배열 결과, 최종 파라미터들) 돌려준다."""
    rng = np.random.default_rng(seed)
    policy = _fresh_policy(seed)
    intervals, bootstrap = _build_intervals(rng, policy, n_intervals=n_intervals,
                                            n_blocks=n_blocks, log_ratio_of=log_ratio_of,
                                            active_of=active_of)
    tape = _pad_tape(intervals, mask_pad_true=mask_pad_true)
    start = v5net.load_v5_params(policy.state_dict())
    adv_raw, returns = gae(intervals, bootstrap, gamma=config.gamma, lam=config.gae_lambda,
                           time_unit_s=config.time_unit_s)
    hyper = PU.hyper_from_v5(config, v5_cast=v5_cast)
    e_n = n_intervals * n_blocks
    rng_arr = np.random.default_rng(rng_seed)
    state_before = rng_arr.bit_generator.state
    orders = PU.draw_orders(rng_arr, n_entries=e_n, hyper=hyper)
    # v5 쪽 — 같은 시드의 난수 발생기로 같은 순열을 뽑는다
    rng_v5 = np.random.default_rng(rng_seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    probe = _Probe().install(monkeypatch)
    rep = v5_update(policy, optimizer, intervals, bootstrap, config, rng_v5)
    after_v5 = {k: v.detach().numpy().copy() for k, v in policy.state_dict().items()}
    params, opt, out = jax.jit(PU.ppo_update, static_argnames=("hyper",))(
        start, PU.init_adam(start), tape, jnp.asarray(adv_raw.reshape(-1)),
        jnp.asarray(returns.reshape(-1)), orders, hyper=hyper)
    PU.rewind_orders_rng(rng_arr, state_before, n_entries=e_n,
                         epochs_entered=int(out.epochs_entered))
    return dict(rep=rep, mbs=probe.minibatches(), out=out, tape=tape, orders=orders,
                after_v5=after_v5, after_arr=v5net.to_v5_state_dict(params),
                rng_v5=rng_v5, rng_arr=rng_arr, opt=opt, e_n=e_n)


def _assert_same_update(got, *, rtol=1e-4, atol_param=1e-5, tag=""):
    """v5 보고·미니배치별 값·최종 파라미터를 한꺼번에 대조한다."""
    rep, out, mbs = got["rep"], got["out"], got["mbs"]
    assert int(out.violations) == 0, PU.violation_names(out.violations)
    # 정수는 정확히
    v5int = {k: rep[k] for k in ("minibatches", "block_samples", "active_block_samples",
                                 "micro_actions")}
    mine = PU.as_v5_report(out, n_intervals=rep["intervals"])
    assert {k: mine[k] for k in v5int} == v5int, f"{tag} 정수 보고: {mine} vs {rep}"
    assert mine["early_stopped"] == rep["early_stopped"], tag
    # 미니배치 수 — 배열판이 '들어간' 배치 수와 v5 가 실제로 돈 배치 수가 같아야 한다
    entered = int(np.sum(np.asarray(out.mb_recorded_kl)))
    assert entered == len(mbs), f"{tag} 들어간 미니배치 {entered} vs v5 {len(mbs)}"
    orders = np.asarray(got["orders"]).reshape(-1, np.asarray(got["orders"]).shape[-1])
    worst = dict(log_ratio=0.0, adv=0.0, loss=0.0, kl=0.0, norm=0.0)
    for k, ref in enumerate(mbs):
        slots = _array_active_slots(got["tape"], orders[k])
        assert len(slots) == len(ref["log_ratio"]) == int(out.mb_n_active[k]), (
            f"{tag} 미니배치 {k} 활성 표본 수: 배열 {len(slots)}/{int(out.mb_n_active[k])} "
            f"vs v5 {len(ref['log_ratio'])}")
        if slots:
            mine_lr = np.asarray(out.mb_log_ratio[k])[slots]
            mine_adv = np.asarray(out.mb_advantage[k])[slots]
            first = np.argmax(np.abs(mine_lr - np.asarray(ref["log_ratio"])))
            worst["log_ratio"] = max(worst["log_ratio"], _rel(mine_lr, ref["log_ratio"]))
            _note("mb_log_ratio_abs",
                  float(np.max(np.abs(mine_lr - np.asarray(ref["log_ratio"])))))
            worst["adv"] = max(worst["adv"], _rel(mine_adv, ref["adv"]))
            # 로그비는 float32 망이 바닥을 만든다 — 실측 절대 4.5e-07, 여유 11배
            assert np.allclose(mine_lr, ref["log_ratio"], rtol=1e-5, atol=5e-6), (
                f"{tag} 미니배치 {k} 로그비 — 처음 갈린 칸 {first}: "
                f"배열 {mine_lr[first]!r} vs v5 {ref['log_ratio'][first]!r}")
            assert np.allclose(mine_adv, ref["adv"], rtol=1e-6, atol=1e-9), (
                f"{tag} 미니배치 {k} 이득이 다르다 (정규화)")
        kl_gap = abs(float(out.mb_kl[k]) - _kl_of(ref["log_ratio"]))
        worst["kl"] = max(worst["kl"], kl_gap)
        assert kl_gap < KL_ATOL, f"{tag} 미니배치 {k} KL 차이 {kl_gap:.3e}" 
        if ref["loss"] is not None:
            worst["loss"] = max(worst["loss"], _rel(float(out.mb_loss[k]), ref["loss"]))
            assert bool(out.mb_applied[k]), f"{tag} 미니배치 {k} 는 v5 가 적용했다"
            assert np.isclose(float(out.mb_loss[k]), ref["loss"], rtol=rtol, atol=1e-9), (
                f"{tag} 미니배치 {k} 손실: {float(out.mb_loss[k])!r} vs {ref['loss']!r}")
            worst["norm"] = max(worst["norm"], _rel(float(out.mb_grad_norm[k]), ref["norm"]))
            assert np.isclose(float(out.mb_grad_norm[k]), ref["norm"], rtol=rtol, atol=1e-9), (
                f"{tag} 미니배치 {k} 기울기 노름: {float(out.mb_grad_norm[k])!r} vs {ref['norm']!r}")
        else:
            assert not bool(out.mb_applied[k]), f"{tag} 미니배치 {k} 는 v5 가 중단했다"
            assert bool(out.mb_recorded_kl[k]), f"{tag} 중단 배치의 KL 은 기록해야 한다"
    for key, val in worst.items():
        _note(f"mb_{key}_" + ("abs" if key == "kl" else "rel"), val)   # KL 만 절대값이다
    # 집계값 — KL 만 절대 허용오차가 크다 (KL_ATOL 머리말)
    for key, atol in (("loss", 1e-9), ("max_kl", KL_ATOL),
                      ("max_grad_norm_before_clip", 1e-9)):
        _note(f"{key}_rel", _rel(mine[key], rep[key]))
        assert np.isclose(mine[key], rep[key], rtol=rtol, atol=atol), (
            f"{tag} {key}: {mine[key]!r} vs {rep[key]!r}")
    # 최종 파라미터 — ★`actor.bias` 는 제외한다 (`test_action_head_bias_is_unidentifiable` 참조):
    #   행동점수 머리의 편향은 모든 후보 줄에 **똑같이** 더해지므로 마스크 softmax 에서 지워진다 →
    #   참 기울기가 정확히 0 이고, 남는 것은 반올림 잡음의 **부호**다. Adam 은 크기를 지워 버리므로
    #   (m/√v ≈ ±1) 잡음의 부호 하나가 걸음 하나(±3e-4) 를 가른다. 답에는 영향이 없다.
    diffs = {k: float(np.max(np.abs(got["after_arr"][k] - v))) for k, v in got["after_v5"].items()}
    got["param_diffs"] = diffs
    for key, d in diffs.items():
        if key == "actor.bias":
            _note("param_abs_actor_bias", d)
            continue
        _note("param_abs", d)
        assert d < atol_param, f"{tag} 최종 파라미터 {key}: 최대 절대차 {d:.3e} (전체 {diffs})"
    return worst


# ═══════════════════════════════════════════════ ④ 갱신 한 번 전체
@pytest.mark.parametrize("v5_cast", [True, False])
def test_update_matches_v5_without_early_stop(monkeypatch, v5_cast):
    """★본 대조 — 21블록·8구간 테이프로 갱신 한 번. 조기중단은 없다 (로그비 0.01)."""
    config = PPOConfig(epochs=2, minibatch_size=64)
    got = _run_pair(monkeypatch, seed=31, n_intervals=8, n_blocks=21,
                    log_ratio_of=lambda e: 0.01, config=config, v5_cast=v5_cast)
    assert got["rep"]["early_stopped"] is False and got["rep"]["minibatches"] == 6
    worst = _assert_same_update(got, tag=f"v5_cast={v5_cast}")
    if not v5_cast:
        # 이득은 정규화까지 float64 로만 계산된다 — numpy 와 거의 비트 일치여야 한다
        assert worst["adv"] < 1e-12, f"정규화 이득 상대 {worst['adv']:.3e}"


def test_update_with_a_single_minibatch_covering_everything(monkeypatch):
    """미니배치가 한 개(전 표본)일 때도 같다 — 순열 순서만 남는 경계 조건."""
    config = PPOConfig(epochs=1, minibatch_size=64)
    got = _run_pair(monkeypatch, seed=32, n_intervals=3, n_blocks=21,
                    log_ratio_of=lambda e: 0.02, config=config)
    assert got["rep"]["minibatches"] == 1
    _assert_same_update(got, tag="mb=all")


# ═══════════════════════════════════════════════ ⑤ 조기중단이 실제로 발동하는 테이프
def _stop_at(*, target_step, config, e_n, rng_seed=505):
    """(표본 → 로그비) 함수 — `target_step` 번째 미니배치에 들어가는 표본만 크게 어긋나게 한다."""
    hyper = PU.hyper_from_v5(config)
    orders = PU.draw_orders(np.random.default_rng(rng_seed), n_entries=e_n, hyper=hyper)
    flat = np.asarray(orders).reshape(-1, orders.shape[-1])
    hot = {int(x) for x in flat[target_step] if x >= 0}
    earlier = set()
    for k in range(target_step):
        earlier |= {int(x) for x in flat[k] if x >= 0}
    # 앞선 미니배치에도 들어가는 표본은 건드리지 않는다 — 그러면 거기서 먼저 멈춘다
    hot -= earlier
    assert hot, "목표 미니배치에만 들어가는 표본이 없다"
    return (lambda e: 1.2 if e in hot else 0.004), hot


@pytest.mark.parametrize("target_step", [0, 1, 2])
def test_early_stop_fires_at_the_same_minibatch_as_v5(monkeypatch, target_step):
    """★조기중단 발동 — 첫 배치(0)·중간(2)·둘째 epoch(4) 에서 v5 와 같은 지점에 멈추는가."""
    config = PPOConfig(epochs=2, minibatch_size=64)
    e_n = 8 * 21
    log_ratio_of, hot = _stop_at(target_step=target_step, config=config, e_n=e_n)
    got = _run_pair(monkeypatch, seed=33 + target_step, n_intervals=8, n_blocks=21,
                    log_ratio_of=log_ratio_of, config=config)
    rep = got["rep"]
    assert rep["early_stopped"] is True, f"이 테이프는 중단해야 한다 (hot {len(hot)}칸)"
    assert rep["minibatches"] == target_step, f"v5 는 {target_step} 배치 뒤에 멈춰야 한다: {rep}"
    assert rep["max_kl"] > config.target_kl, "중단 배치의 KL 이 보고에 있어야 한다 (update.py:59)"
    _assert_same_update(got, tag=f"stop@{target_step}")
    out = got["out"]
    assert int(out.epochs_entered) == 1, "epoch 1 에서 멈췄으면 순열은 한 번만 뽑는다"
    # 중단 뒤 미니배치는 계산도 기록도 하지 않는다
    assert not bool(np.any(np.asarray(out.mb_recorded_kl)[target_step + 1:]))


def _pick_late_spike(config, e_n, active_of, rng_seed=505):
    """epoch 1 에서는 **활성 표본이 많은** 미니배치, epoch 2 에서는 **적은** 미니배치에 들어가는 표본 하나.

    미니배치의 KL 은 활성 표본들의 평균이므로, 크게 어긋난 표본 하나의 기여는 `KL_hot / 활성표본수` 다.
    같은 표본이 epoch 마다 다른 배치에 들어가는 것을 이용해 **둘째 epoch 에서 처음 문턱을 넘게** 만든다.
    """
    orders = np.asarray(PU.draw_orders(np.random.default_rng(rng_seed), n_entries=e_n,
                                       hyper=PU.hyper_from_v5(config)))
    flat = orders.reshape(-1, orders.shape[-1])
    live = np.asarray([bool(active_of(e)) for e in range(e_n)])
    counts = [int(sum(1 for x in row if x >= 0 and live[x])) for row in flat]
    spots: dict[int, list[tuple[int, int]]] = {}
    for k, row in enumerate(flat):
        for x in row:
            if x >= 0 and live[x]:
                spots.setdefault(int(x), []).append((k, counts[k]))
    best = None
    for e, where in spots.items():
        if len(where) < 2:
            continue
        (k1, n1), (k2, n2) = where[0], where[1]
        if n1 > n2 and (best is None or n1 / max(n2, 1) > best["ratio"]):
            best = dict(entry=e, step1=k1, n1=n1, step2=k2, n2=n2, ratio=n1 / max(n2, 1))
    assert best and best["ratio"] > 1.5, f"쓸 만한 표본이 없다: {best}"
    return best


def test_early_stop_in_the_second_epoch_matches_v5(monkeypatch):
    """★둘째 epoch 에서 멈추는 경우 — 목표 KL 을 **실측 사이**에 놓아 그 분기를 강제한다.

    PPO 의 조기중단은 본래 '정책이 수집 때와 멀어지면 멈춘다' 는 장치라 epoch 를 거듭하면 KL 이 커진다.
    그래서 (ㄱ) 목표 KL 을 아주 크게 두고 한 번 굴려 미니배치별 KL 을 재고 (ㄴ) 그중 첫 epoch 의 최댓값과
    둘째 epoch 의 어떤 값 **사이**에 목표를 놓아 다시 굴린다. 테이프·순열이 같아 궤적은 중단 전까지
    똑같으므로 중단 지점은 **예측한 그 배치**여야 한다 — 그 예측이 맞는지 v5 에게 직접 물어본다.
    """
    seed, n_i, n_b, hot_lr = 45, 8, 21, 2.0
    e_n = n_i * n_b
    # 표본 하나만 크게 어긋나게 한다. 미니배치 KL 은 **활성 표본 수로 나눈 평균**이라, 그 표본이
    # epoch 1 에서는 활성 표본이 많은 배치에 섞여 묻히고 epoch 2 에서는 적은 배치에 도드라진다.
    only_third = lambda e: (e % 3) == 0                      # noqa: E731 — 활성 표본을 1/3 로 줄여 분산을 키운다
    hot = _pick_late_spike(PPOConfig(epochs=2, minibatch_size=16, target_kl=9.0), e_n, only_third)
    kw = dict(seed=seed, n_intervals=n_i, n_blocks=n_b, active_of=only_third,
              log_ratio_of=lambda e: hot_lr if e == hot["entry"] else 0.004)
    loose = _run_pair(monkeypatch, config=PPOConfig(epochs=2, minibatch_size=16, target_kl=9.0),
                      **kw)
    assert loose["rep"]["early_stopped"] is False
    kls = [_kl_of(m["log_ratio"]) for m in loose["mbs"]]
    n_mb = int(np.asarray(loose["orders"]).shape[1])
    hits = [k for k in range(n_mb, len(kls)) if kls[k] > max(kls[:k]) * 1.2]
    assert hits, (f"둘째 epoch 에 더 큰 KL 이 없다 — 고른 표본 {hot} · 미니배치별 KL "
                  f"{[round(v, 5) for v in kls]}")
    step = hits[0]
    target = 0.5 * (max(kls[:step]) + kls[step])
    got = _run_pair(monkeypatch, config=PPOConfig(epochs=2, minibatch_size=16,
                                                 target_kl=float(target)), **kw)
    assert got["rep"]["early_stopped"] is True
    assert got["rep"]["minibatches"] == step, (
        f"예측 {step} · v5 {got['rep']['minibatches']} (미니배치별 KL {kls})")
    assert int(got["out"].epochs_entered) == 2, "둘째 epoch 에 들어갔으면 순열을 두 번 뽑는다"
    _assert_same_update(got, tag="stop@epoch2")


def test_no_early_stop_when_divergence_stays_below_target(monkeypatch):
    """목표 바로 아래(로그비 0.23 = KL 0.02860)면 **한 번도** 멈추지 않는다 — 문턱이 제대로 걸리는지.

    ★문턱은 아슬아슬하다: 로그비 0.24 면 KL 0.03125 로 이미 넘는다 (목표 0.03). 즉 v5 의 float32 KL 과
      배열판의 float64 KL 이 문턱 양쪽에 갈릴 여지가 원리상 있다 (조각 7 의 결정 뒤집힘과 같은 종류).
      실측 여유는 KL 절대오차 ≈2e-08 대 문턱까지 1.4e-03 = **7만 배**다.
    """
    config = PPOConfig(epochs=1, minibatch_size=64)
    got = _run_pair(monkeypatch, seed=41, n_intervals=6, n_blocks=21,
                    log_ratio_of=lambda e: 0.23, config=config)
    assert got["rep"]["early_stopped"] is False
    assert 0.02 < got["rep"]["max_kl"] < config.target_kl
    assert config.target_kl - got["rep"]["max_kl"] > 1e-4 * config.target_kl
    _assert_same_update(got, tag="just-below")


def test_rng_stream_after_early_stop_matches_v5(monkeypatch):
    """조기중단 뒤 난수 상태 — 되감으면 v5 와 **같은 다음 순열**을 낸다 (여러 갱신을 이어 돌리기 위해)."""
    config = PPOConfig(epochs=2, minibatch_size=64)
    e_n = 6 * 21
    log_ratio_of, _ = _stop_at(target_step=1, config=config, e_n=e_n)
    got = _run_pair(monkeypatch, seed=44, n_intervals=6, n_blocks=21,
                    log_ratio_of=log_ratio_of, config=config)
    assert got["rep"]["early_stopped"] is True and int(got["out"].epochs_entered) == 1
    assert np.array_equal(got["rng_v5"].permutation(e_n), got["rng_arr"].permutation(e_n))


# ═══════════════════════════════════════════════ ⑥ 활성 표본이 없는 미니배치
def test_minibatch_without_any_decision_freezes_the_action_head(monkeypatch):
    """★결정이 하나도 없는 미니배치 — v5 는 행동점수 머리의 `p.grad` 가 None 이라 **건너뛴다**."""
    config = PPOConfig(epochs=1, minibatch_size=32)
    e_n = 4 * 21
    hyper = PU.hyper_from_v5(config)
    orders = PU.draw_orders(np.random.default_rng(505), n_entries=e_n, hyper=hyper)
    flat = np.asarray(orders).reshape(-1, orders.shape[-1])
    dead = {int(x) for x in flat[1] if x >= 0}          # 두 번째 미니배치만 전부 비운다
    got = _run_pair(monkeypatch, seed=51, n_intervals=4, n_blocks=21,
                    log_ratio_of=lambda e: 0.01, config=config,
                    active_of=lambda e: e not in dead)
    out = got["out"]
    assert int(out.mb_n_active[1]) == 0, "두 번째 미니배치에 결정이 남아 있다"
    assert int(out.mb_n_active[0]) > 0 and float(out.mb_kl[1]) == 0.0
    _assert_same_update(got, tag="dead-minibatch")
    # 행동점수 머리의 Adam 걸음 수가 몸통보다 하나 적다
    counts = [int(c) for c in got["opt"].count]
    assert counts[4] == counts[5] == counts[0] - 1, counts
    assert [int(c) for c in got["opt"].count] == [3, 3, 3, 3, 2, 2, 3, 3]


def test_two_updates_in_a_row_match_v5(monkeypatch):
    """★갱신을 **두 번 이어** 돌린다 — 최적화기 상태(m·v·걸음수)와 난수열이 v5 처럼 이어지는가.

    통합 단계의 실제 경로다 (`runtime._update` 가 60구간마다 부른다). 한 번만 맞아도 상태가 새면 둘째가 어긋난다.
    ⚠️ 둘째 회차의 **테이프는 v5 정책으로 모은다** (v5 가 정본이니까). 그런데 첫 회차 뒤 두 쪽 가중치가
       이미 ~1e-06 벌어져 있으므로 둘째 회차의 로그비는 float32 바닥(4e-07)보다 **더** 벌어진다 — 그래서
       둘째 회차의 미니배치 손실은 rtol 1e-3 으로 보고 실측을 찍는다. 판정선은 **최종 파라미터 atol 1e-5** 다.
    """
    config = PPOConfig(epochs=2, minibatch_size=64)
    policy = _fresh_policy(91)
    params, opt = v5net.load_v5_params(policy.state_dict()), None
    opt = PU.init_adam(params)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    hyper = PU.hyper_from_v5(config)
    rng_tape, rng_v5, rng_arr = (np.random.default_rng(91), np.random.default_rng(707),
                                 np.random.default_rng(707))
    step = jax.jit(PU.ppo_update, static_argnames=("hyper",))
    for round_no in (1, 2):
        intervals, bootstrap = _build_intervals(rng_tape, policy, n_intervals=5, n_blocks=21,
                                                log_ratio_of=lambda e: 0.02)
        tape = _pad_tape(intervals)
        adv, ret = gae(intervals, bootstrap, gamma=config.gamma, lam=config.gae_lambda,
                       time_unit_s=config.time_unit_s)
        e_n = 5 * 21
        before = rng_arr.bit_generator.state
        orders = PU.draw_orders(rng_arr, n_entries=e_n, hyper=hyper)
        probe = _Probe().install(monkeypatch)
        rep = v5_update(policy, optimizer, intervals, bootstrap, config, rng_v5)
        params, opt, out = step(params, opt, tape, jnp.asarray(adv.reshape(-1)),
                                jnp.asarray(ret.reshape(-1)), orders, hyper=hyper)
        PU.rewind_orders_rng(rng_arr, before, n_entries=e_n,
                             epochs_entered=int(out.epochs_entered))
        mine = PU.as_v5_report(out, n_intervals=rep["intervals"])
        assert int(out.violations) == 0
        assert {k: mine[k] for k in ("minibatches", "block_samples", "active_block_samples",
                                     "micro_actions", "early_stopped")} == {
            k: rep[k] for k in ("minibatches", "block_samples", "active_block_samples",
                                "micro_actions", "early_stopped")}, f"{round_no}회차 {mine} vs {rep}"
        rtol = 1e-4 if round_no == 1 else 1e-3
        for k, ref in enumerate(probe.minibatches()):
            if ref["loss"] is None:
                continue
            _note(f"round{round_no}_mb_loss_rel", _rel(float(out.mb_loss[k]), ref["loss"]))
            assert np.isclose(float(out.mb_loss[k]), ref["loss"], rtol=rtol, atol=1e-9), (
                f"{round_no}회차 미니배치 {k} 손실 {float(out.mb_loss[k])!r} vs {ref['loss']!r}")
        after_arr = v5net.to_v5_state_dict(params)
        for key, ref in policy.state_dict().items():
            d = float(np.max(np.abs(after_arr[key] - ref.detach().numpy())))
            if key == "actor.bias":
                _note("param_abs_actor_bias", d)
                continue
            _note(f"round{round_no}_param_abs", d)
            assert d < 1e-5, f"{round_no}회차 최종 파라미터 {key}: 절대차 {d:.3e}"
        # ★걸음 수가 두 쪽 모두 같게 늘어난다 (m·v 상태가 이어지는 증거)
        counts = [int(c) for c in opt.count]
        torch_counts = [int(optimizer.state[q]["step"]) for q in policy.parameters()
                        if q in optimizer.state]
        assert counts == torch_counts, f"{round_no}회차 Adam 걸음 수 {counts} vs {torch_counts}"
        assert counts[0] == rep["minibatches"] * round_no


# ═══════════════════════════════════════════════ ⑦ 난수 규약·패딩·jit
def test_draw_orders_reproduces_v5_permutations():
    """색인표는 v5 와 **같은 난수열**이다 — 배열 안에서 새로 뽑지 않는다 (규칙 ③)."""
    hyper = PU.PPOHyper(epochs=3, minibatch_size=7)
    orders = PU.draw_orders(np.random.default_rng(99), n_entries=17, hyper=hyper)
    ref = np.random.default_rng(99)
    assert orders.shape == (3, 3, 7)
    for k in range(3):
        want = np.asarray(ref.permutation(17), np.int32)
        got = orders[k].reshape(-1)
        assert np.array_equal(got[:17], want)
        assert np.all(got[17:] == -1)               # 남는 칸은 -1


def test_padding_variants_and_jit_agree(monkeypatch):
    """패딩 칸의 마스크가 전부 거짓이어도 결과는 같고, jit 과 eager 도 같다."""
    config = PPOConfig(epochs=1, minibatch_size=64)
    kw = dict(seed=61, n_intervals=4, n_blocks=21, log_ratio_of=lambda e: 0.03, config=config)
    a = _run_pair(monkeypatch, mask_pad_true=True, **kw)
    b = _run_pair(monkeypatch, mask_pad_true=False, **kw)
    for key in a["after_arr"]:
        assert np.array_equal(a["after_arr"][key], b["after_arr"][key]), key
    assert float(a["out"].loss) == float(b["out"].loss)
    _assert_same_update(a, tag="pad-true")
    # jit 없이 그대로 돌린 것과 비교
    rng = np.random.default_rng(61)
    policy = _fresh_policy(61)
    intervals, bootstrap = _build_intervals(rng, policy, n_intervals=4, n_blocks=21,
                                            log_ratio_of=lambda e: 0.03)
    tape = _pad_tape(intervals)
    start = v5net.load_v5_params(policy.state_dict())
    adv, ret = gae(intervals, bootstrap, gamma=config.gamma, lam=config.gae_lambda,
                   time_unit_s=config.time_unit_s)
    hyper = PU.hyper_from_v5(config)
    orders = PU.draw_orders(np.random.default_rng(505), n_entries=84, hyper=hyper)
    eager = PU.ppo_update(start, PU.init_adam(start), tape, jnp.asarray(adv.reshape(-1)),
                          jnp.asarray(ret.reshape(-1)), orders, hyper)
    jitted = jax.jit(PU.ppo_update, static_argnames=("hyper",))(
        start, PU.init_adam(start), tape, jnp.asarray(adv.reshape(-1)),
        jnp.asarray(ret.reshape(-1)), orders, hyper=hyper)
    gap = max(_rel(np.asarray(x), np.asarray(y)) for x, y in zip(eager[0], jitted[0]))
    _note("jit_vs_eager_rel", gap)
    assert gap < 1e-15, f"jit 과 eager 가 갈렸다: 상대 {gap:.3e}"


def test_nonfinite_divergence_sets_a_flag_instead_of_raising():
    """v5 가 `FloatingPointError` 를 던지는 자리는 **표시 비트**다 (jit 안에서는 못 던진다)."""
    rng = np.random.default_rng(71)
    params = v5net.init_params(jax.random.PRNGKey(3))
    e_n, cmax, amax = 8, 2, 4
    tape = PU.Tape(
        states=jnp.asarray(rng.normal(size=(e_n, 37))),
        rows=jnp.asarray(rng.normal(size=(e_n, cmax, amax, 37))),
        mask=jnp.asarray(np.ones((e_n, cmax, amax), bool)),
        action=jnp.zeros((e_n, cmax), jnp.int32),
        old_logp=jnp.asarray(np.full((e_n, cmax), -1e4)),   # exp(로그비) 가 넘친다 → KL 이 inf
        n_choices=jnp.ones(e_n, jnp.int32))
    hyper = PU.PPOHyper(epochs=1, minibatch_size=4)
    orders = PU.draw_orders(np.random.default_rng(1), n_entries=e_n, hyper=hyper)
    _, _, out = PU.ppo_update(params, PU.init_adam(params), tape, jnp.zeros(e_n),
                              jnp.zeros(e_n), orders, hyper)
    assert PU.violation_names(out.violations) == ["KL_NONFINITE"]
    assert int(out.minibatches) == 0 and bool(out.early_stopped)


def test_action_head_bias_is_unidentifiable_so_its_drift_is_harmless():
    """★`actor.bias` 는 **어떤 답도 바꾸지 못한다** — 그래서 그 칸만 v5 와 벌어져도 무해하다.

    행동점수 머리의 편향은 모든 후보 줄에 똑같이 더해지고, 결정은 마스크 softmax 라 상수항이 지워진다.
    그래서 ① 편향을 5.0 만큼 흔들어도 결정·로그확률·확률·엔트로피·상태가치가 그대로이고
          ② 손실의 편향 기울기가 정확히 0 에 가깝다 (남는 것은 반올림 잡음).
    Adam 은 크기를 지우므로(m/√v ≈ ±1) 그 잡음의 부호 하나가 걸음 하나(±lr)를 가른다.
    """
    rng = np.random.default_rng(81)
    p = v5net.load_v5_params(_fresh_policy(81).state_dict())
    x = jnp.asarray(rng.normal(size=(6, 37)))
    m = jnp.asarray([True, True, False, True, True, True])
    shifted = p._replace(b_actor=p.b_actor + 5.0)
    for name, fn in (("결정", lambda q: v5net.greedy_action(v5net.actor_scores(q, x), m)),
                     ("로그확률", lambda q: v5net.log_probs(v5net.actor_scores(q, x), m)),
                     ("확률", lambda q: v5net.probs(v5net.actor_scores(q, x), m)),
                     ("엔트로피", lambda q: v5net.entropy(v5net.actor_scores(q, x), m)),
                     ("상태가치", lambda q: v5net.state_values(q, x))):
        a, b = np.asarray(fn(p)), np.asarray(fn(shifted))
        keep = np.isfinite(a) & np.isfinite(b)          # 마스크 칸은 양쪽 다 -inf
        gap = float(np.max(np.abs(a[keep] - b[keep]))) if keep.any() else 0.0
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), f"{name} 의 -inf 자리가 달라졌다"
        _note("actor_bias_shift_gap", gap)
        assert gap < 1e-13, f"{name} 가 편향에 반응한다: {gap:.3e}"

    def loss(q):
        s = v5net.actor_scores(q, x)
        return (v5net.log_prob_of(s, m, jnp.int32(3)) + v5net.entropy(s, m)
                + jnp.sum(v5net.state_values(q, x)))

    g = jax.grad(loss)(p)
    _note("actor_bias_grad_abs", float(jnp.abs(g.b_actor[0])))
    assert float(jnp.abs(g.b_actor[0])) < 1e-14, "편향 기울기가 0 이 아니다"
    assert float(jnp.max(jnp.abs(g.w_actor))) > 1e-6, "가중치 기울기는 0 이 아니어야 한다 (대조)"


def test_zz_report():
    """실측을 한자리에 찍는다 — 통합 단계가 허용오차를 고를 근거 (`-s` 로 보인다)."""
    assert MEASURED, "앞 시험이 하나도 안 돌았다"
    print("\n  ── 배열판 PPO 갱신 vs v5 (float32 정책망이 바닥을 만든다) ──")
    for key in sorted(MEASURED):
        print(f"     {key:30s} {MEASURED[key]:.3e}")
    print("     (param_abs_actor_bias 는 답을 안 바꾸는 칸 — 위 시험이 증명한다)")
    # ★파수꾼 — 개별 시험의 통과 기준(rtol 1e-4)보다 **좁게** 잡아 정확도가 나빠지면 여기서 걸린다
    #   (괄호는 실측 대비 여유. 실측은 위에 찍힌 값이다.)
    for key, limit in (("adam_rel", 1e-13),            # 2.2e-16 → 450배
                       ("adam_vs_optax_rel", 1e-14),   # 2.0e-16 → 50배
                       ("clip_grad_rel", 1e-14),       # 4.4e-16 → 23배
                       ("clip_norm_rel", 1e-14),
                       ("adv_norm_rel", 1e-13),        # 정규화는 float64 로만 — 2.0e-16
                       ("param_abs", 1e-5),            # 1.7e-06 → 6배 (명세 기준값)
                       ("mb_loss_rel", 2e-5),          # 1.9e-06 → 10배
                       ("mb_norm_rel", 1e-5),          # 1.5e-07 → 66배
                       ("mb_kl_abs", KL_ATOL),         # KL 은 절대값 — 6.7e-08 → 3배
                       ("mb_log_ratio_abs", 5e-6),     # 4.5e-07 → 11배
                       ("actor_bias_grad_abs", 1e-14), # 참 기울기는 0 이다
                       ("jit_vs_eager_rel", 1e-15)):
        if key in MEASURED:
            assert MEASURED[key] < limit, f"{key} = {MEASURED[key]:.3e} ≥ {limit:.0e}"
