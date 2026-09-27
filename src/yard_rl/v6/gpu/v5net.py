"""v5 **학습 정책망**(`ppo/model.py` BlockPolicy)의 배열판 — 같은 가중치 → 같은 결정 ([[YR-327]] 조각 7).

■ 무엇인가 — 몸통 하나 + 머리 둘
    입력 37칸 ──[Linear 37→64]──tanh──[Linear 64→64]──tanh──┬──[Linear 64→1]── 행동 점수 (actor)
                                                             └──[Linear 64→1]── 상태 가치 (critic)

  v5 는 후보(또는 블록) 한 줄마다 점수 하나를 받는다. 줄끼리 섞이는 곳은 **마스크 softmax 하나**뿐이고
  (`Categorical(logits=…)`), 몸통·머리는 줄별로 독립이다 — 그래서 `gpu/dispatch.py` 의 `policy_fn` 규약
  (행 k 의 답이 다른 행 마스크에 의존하지 않는다)을 그대로 만족한다.

■ ⚠️ `gpu/policy.py` 와 **다른 망**이다 — 지우지도 섞지도 않는다
  | | `gpu/policy.py` ([[YR-326]] 새 축) | **이 파일** (동등성용) |
  |---|---|---|
  | 활성함수 | ReLU | **tanh** (v5 정본) |
  | 입력 | 오더 특징 9칸 | **37칸** (원특징 32 + 역할 4 + BUY 표시 1) |
  | 머리 | 가치 + 우위(dueling, 평균 0) | **행동 점수 + 상태 가치** (v5 정본) |
  | 목적 | 전 오더 한 번에 · 반사실 기준선 | **v5 체크포인트를 그대로 굴려 대조** |

■ ★입력 37칸의 구성 (`ppo/model.py:8-27` `encode`)
    0..31   원특징 (`RAW_DIM=32`) — 실제로 쓰는 칸만 채우고 **나머지는 0 패딩**
    32..35  역할 한자리(one-hot) — seller · buyer · crane · **state** 순서 고정
    36      buyer 의 BUY 줄 표시 — buyer 역할일 때 **0번 줄에만** 1 (BUY/REJECT 가 특징이 전부 0 이어도 갈리게)
  크레인 결정의 원특징은 24칸이다 (`ppo/crane.py:39-56` — 블록 8 + 종류 4 + 값 7 + 직전크레인 4 + 1).

■ ★비트 일치가 **불가능한 층**이다 — 측정으로 못박는다 (2026-09-26 · Windows CPU x64 · torch 2.14 · jax 0.11.2)
  조각 1~6 의 세계는 v5 와 마지막 비트까지 같았다. 망은 다르다. 이유는 둘 다 **우리 코드 밖**이다:

    ① `tanh` — torch(libm/SLEEF) 와 XLA 의 근사식이 다르다. 무작위 2만점에서 **57.6% 가 1~2 ulp 다르고**
       최대 |Δ| = 4.44e-16. 곱셈 순서(`exact.mul_exact`)로 막을 수 있는 종류가 아니다.
    ② 행렬곱 축소 순서 — torch 는 MKL GEMM, XLA 는 자기 커널이다. (13,37)·(37,64) 에서 |Δ| ≤ 5.3e-15.
       작은 모양(4,37)·(37,8) 에서는 **둘 다 순차 축소라 비트 일치**했다 — 즉 어긋남은 MKL 의 블록 커널 탓이다.
       `exact.sum_seq` + `mul_exact` 로 순차 합을 강제해도 **더 나빠진다** (측정: 순차 8.33e-16 vs 그냥 `@` 6.66e-16)
       → 규칙 1(v5 결합 순서 그대로)의 예외를 **측정 근거와 함께** 둔다: 이 파일은 평범한 `x @ W + b` 를 쓴다.

  실측 (`tests/v6/test_gpu_v5net.py` 가 매번 다시 재고 화면에 찍는다 — 무작위 후보 1~13줄):
      torch float64 vs 배열 float64   최대 |Δ| 2.22e-16   argmax 불일치 0/500   ← 무작위 가중치
      torch float32 vs 배열 float64   최대 |Δ| 1.21e-07   argmax 불일치 0/200   ← ★v5 실제 실행은 float32 다
      실제 학습 체크포인트 (yr302-final-train) f64 2.78e-16 · f32 1.24e-07   argmax 불일치 0/200
      분포: 로그확률 8.88e-16 · 확률 1.11e-16 · 엔트로피 1.33e-15 · log_prob 6.66e-16
  ★★결정이 갈릴 확률은 **0 이 아니다** — 이 층의 동등성은 *증명된 성질이 아니라 측정된 확률*이다
    (2026-09-26 검증 반박. 전에 여기 적혀 있던 "1·2위 최소 격차 2.5e-05 → 결정은 안 바뀐다" 는
     **망 두 벌(시드 101·1234)만 재서 나온 표본값**이었고, "갈릴 여지가 없다" 는 틀린 문장이었다.)

      v5 가 실제로 만든 후보 행 328결정분(무대 8종 · 행 합계 1,922줄)에 torch 기본 초기화 망 **2,600벌**을
      먹여 **결정 852,800건**을 대조 → **뒤집힘 4건 = 4.7e-06/결정**
      (직접 실행 2026-09-26 · `scripts/v6/probe_net_flip_rate.py --nets 2600` · 995초 ·
       결과 `outputs/v6/net_flip_rate.json` · 로그 `outputs/v6/verify/net_flip_rate.out`):
        · 전체 1·2위 **최소** 격차 1.583e-08 · 점수 |Δ| **최댓값** 9.418e-08 → **여유가 0.17배**로 뒤집힌다
        · 갈린 넷: 시드 1063 feat-crowded #21(후보 4 · v5=0 배열=1 · 격차 1.583e-08) ·
          2551 feat-vessel #57(후보 4 · v5=0 배열=1) · 2599 feat-crowded #5(후보 9 · v5=1 배열=2) ·
          2599 feat-crowded #14(후보 11 · v5=4 배열=5 · 격차 2.787e-08)
        · 배열을 float32 로 돌려도 **배열 쪽 답**이 나온다 → 원인은 정밀도가 아니라 tanh·GEMM 차이이고
          "배열도 float32 로 맞춘다" 로는 못 고친다
        · 근접 동점은 우연이 아니라 **구조적**이다 — 트럭 10대가 같은 시각에 도착하면 누적대기가 똑같아져
          후보 여러 개의 점수가 1e-05 안에 뭉친다 (feat-crowded 가 뒤집힘 3/4 를 낸 이유)
        · 규모 감각: 터미널 하루 망 호출 8,350회 → 하루당 기대 0.04건 · 하루가 갈릴 확률 ≈3.8% · 30일 ≈69%
      반면 **on-policy**(망이 실제로 세계를 굴리는 경로)에서는 지금까지 한 건도 안 갈렸고 최소 격차가
      2e-05 대였다 — 즉 재생 대조가 훨씬 가혹한 표본이다(그래서 이 확률은 상한에 가깝다).

  **그래서 이 층의 동등성 기준은 "표본에서 결정(argmax)이 같다 + |Δ| 가 허용오차 안 + 1·2위 최소 격차가
  |Δ| 보다 충분히 크다" 이다.** 통합 시험은 격차를 찍기만 하지 말고 **단언**해야 한다
  (`test_gpu_v5policy_equiv.test_zz_report` 가 `최소격차 ≥ 20 × |Δ|` 를 요구한다).
  그리고 조각 8 의 체크포인트 대조는 '하루 해시 일치' 가 아니라 **'첫 갈린 결정까지의 접두사 일치 +
  갈린 지점의 1·2위 격차 기록'** 으로 판정해야 한다 — 해시로 판정하면 한 달 대조가 ≈69% 확률로
  "불일치" 로 끝난다 (README 조각 8 항목).

  머리(actor·critic)의 `(M,64)·(64,1)` 은 XLA·MKL 이 **비트 일치**했다 — 어긋남은 몸통의 37→64·64→64
  GEMM 과 tanh 에서만 온다. vmap(배치)판은 낱개와 critic 값이 1 ulp(1.39e-17) 갈린다: 배치 행렬곱을
  다른 커널로 묶기 때문이다. `jnp.sum(mul_exact(...))` 로 바꾸면 vmap 이 낱개와 비트 일치하지만 **torch 와
  더 멀어진다**(0 → 5.33e-15) — v5 와 맞추는 쪽이 목적이라 `@` 를 쓴다.

■ ★v5 는 특징을 **float32 로 깎아서** 넣는다 (`model.py:14` `torch.as_tensor(rows, dtype=torch.float32)`)
  배열 세계의 특징은 float64 다. 그대로 넣으면 **입력부터 다르다**. 그래서 `encode(..., v5_cast=True)`(기본)
  이 float32 로 한 번 깎은 뒤 float64 로 올린다 — v5 가 망에 실제로 넣은 값과 **비트 동일**. 학습 모드
  (조각 8, 동등성 불필요)에서는 `v5_cast=False` 로 끈다.

■ 빈 마스크 — v5 는 예외를 던지고 드라이버가 **전원 WAIT** 로 바꾼다
  `model.py:45-46` 은 `not mask.any()` 면 ValueError 다. jit 안에서는 던질 수 없으므로 `mask_ok` 로 돌려준다:
  `greedy_action` 은 -1, 확률은 전부 0. 부르는 쪽이 `stage/episode.py:212-216` 의 규칙
  ("한 크레인 실패 = 그 결정 전원 WAIT") 대로 위반 비트를 켜고 전원 WAIT 로 바꾼다.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .exact import mul_exact

__all__ = ["RAW_DIM", "ROLES", "INPUT_DIM", "HIDDEN", "V5_STATE_KEYS", "NEG_INF",
           "V5PolicyParams", "V5Decision", "init_params", "load_v5_params",
           "to_v5_state_dict", "encode", "all_finite", "trunk", "actor_scores",
           "state_values", "masked_scores", "log_probs", "probs", "entropy",
           "log_prob_of", "greedy_action", "sample_action", "decide",
           "actor_scores_batch", "state_values_batch", "greedy_action_batch",
           "decide_batch", "greedy_policy_fn"]

#: ★칸 수·역할 순서는 **`gpu/v5feat.py` 가 정본**이다 (2026-09-26 통합 · 검증 반박 "이중 정의").
#:   여기서 다시 적지 않고 그대로 들여온다 — 통합 경로(`dispatch.make_v5net_pick`)가 쓰는 쪽이 v5feat 이다.
#:   RAW_DIM 32 (`model.py:8,15`) · ROLES 순서 고정 (`model.py:9,21`) · INPUT_DIM 37 (`model.py:10`)
from .v5feat import INPUT_DIM, RAW_DIM, ROLES  # noqa: E402  (순환 없음 — v5feat 는 이 파일을 안 쓴다)
#: v5 기본 은닉 폭 (`model.py:31`)
HIDDEN = 64
#: torch `state_dict` 키 — `nn.Sequential(Linear, Tanh, Linear, Tanh)` 이라 몸통은 0·2 번이다
V5_STATE_KEYS = ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                 "actor.weight", "actor.bias", "critic.weight", "critic.bias")
#: `masked_fill(~mask, -torch.inf)` 과 같은 값 (`model.py:47`) — -1e30 이 아니라 **진짜 -inf** 다
NEG_INF = -jnp.inf


class V5PolicyParams(NamedTuple):
    """v5 BlockPolicy 의 가중치 8장. **torch 와 달리 `(in, out)` 방향**이다 (`x @ w`).

    torch `nn.Linear` 는 `(out, in)` 으로 저장하고 `x @ W.T` 를 쓴다 — `load_v5_params` 가 전치한다.
    """

    w1: jnp.ndarray      # (37, H)  몸통 1
    b1: jnp.ndarray      # (H,)
    w2: jnp.ndarray      # (H, H)   몸통 2
    b2: jnp.ndarray      # (H,)
    w_actor: jnp.ndarray   # (H, 1) 행동 점수 머리
    b_actor: jnp.ndarray   # (1,)
    w_critic: jnp.ndarray  # (H, 1) 상태 가치 머리
    b_critic: jnp.ndarray  # (1,)

    @property
    def hidden(self) -> int:
        return int(self.b1.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.w1.shape[0])


class V5Decision(NamedTuple):
    """결정 한 건 — `ppo/runtime.select`(123-140행) 이 쓰는 값 전부."""

    action: jnp.ndarray     # () int32 — 고른 줄 번호, 마스크가 전부 거짓이면 -1
    log_prob: jnp.ndarray   # () 고른 줄의 로그확률 (torch `dist.log_prob`)
    scores: jnp.ndarray     # (M,) 마스크 씌운 원점수 (`logits`)
    probs: jnp.ndarray      # (M,) 확률 (마스크 칸은 0)
    entropy: jnp.ndarray    # () 분포 엔트로피 (PPO 손실 항)
    value: jnp.ndarray      # (M,) 줄마다 상태 가치 (critic)
    mask_ok: jnp.ndarray    # () bool — 거짓이면 v5 는 예외 → 부르는 쪽이 전원 WAIT


# ───────────────────────────────────────────────── 가중치 만들기·싣기
def init_params(key, *, hidden: int = HIDDEN, dim: int = INPUT_DIM,
                dtype=jnp.float64) -> V5PolicyParams:
    """torch `nn.Linear` **기본 초기화와 같은 분포**로 새 가중치 (난수 흐름은 다르다).

    torch 는 `kaiming_uniform_(a=√5)` → 경계가 `1/√fan_in` 인 균등분포이고, 편향도 같은 경계다
    (`torch/nn/modules/linear.py reset_parameters`). 동등성 대조에는 **이 함수를 쓰지 않는다** —
    같은 수를 내려면 가중치가 같아야 하므로 `load_v5_params` 로 v5 체크포인트를 싣는다.
    이 함수는 조각 8 이 v6 에서 **처음부터** 학습할 때 쓴다.
    """
    if hidden < 1:
        raise ValueError("hidden 은 1 이상이어야 한다 (model.py:33-34)")
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def lin(k, fan_in, fan_out):
        bound = 1.0 / np.sqrt(float(fan_in))
        kw, kb = jax.random.split(k)
        w = jax.random.uniform(kw, (fan_in, fan_out), dtype, -bound, bound)
        b = jax.random.uniform(kb, (fan_out,), dtype, -bound, bound)
        return w, b

    w1, b1 = lin(k1, dim, hidden)
    w2, b2 = lin(k2, hidden, hidden)
    wa, ba = lin(k3, hidden, 1)
    wc, bc = lin(k4, hidden, 1)
    return V5PolicyParams(w1, b1, w2, b2, wa, ba, wc, bc)


def _as_np(t) -> np.ndarray:
    """torch 텐서·numpy 배열·리스트를 numpy 로. torch 를 **import 하지 않는다** (gpu/ 는 torch 무의존)."""
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    return np.asarray(t)


def load_v5_params(state_dict: Mapping, *, dtype=jnp.float64) -> V5PolicyParams:
    """torch `state_dict` → `V5PolicyParams`. **전치**하고 float64 로 올린다.

    · torch `nn.Linear.weight` 는 `(out, in)` 이고 `x @ W.T` 를 계산한다. 배열판은 `x @ w` 라 `w = W.T` 다.
    · float32 → float64 승격은 **정확**하다 (가중치 값 자체는 하나도 안 바뀐다).
    · 체크포인트 전체(`torch.load(...)`)를 그대로 줘도 된다 — `"policy"` 칸을 찾아 쓴다 (`ppo/checkpoint.py:22`).
    """
    sd = state_dict
    if not all(k in sd for k in V5_STATE_KEYS) and "policy" in sd:
        sd = sd["policy"]                       # ppo/checkpoint.save_checkpoint 가 감싼 형태
    missing = [k for k in V5_STATE_KEYS if k not in sd]
    if missing:
        raise KeyError(f"v5 BlockPolicy state_dict 가 아니다 — 없는 키: {missing}")
    w1, b1, w2, b2, wa, ba, wc, bc = (_as_np(sd[k]) for k in V5_STATE_KEYS)
    if w1.shape[1] != INPUT_DIM:
        raise ValueError(f"입력 폭이 {w1.shape[1]} — v5 는 {INPUT_DIM} 이다 (model.py:10)")
    h = w1.shape[0]
    want = {"trunk.0.weight": (h, INPUT_DIM), "trunk.0.bias": (h,), "trunk.2.weight": (h, h),
            "trunk.2.bias": (h,), "actor.weight": (1, h), "actor.bias": (1,),
            "critic.weight": (1, h), "critic.bias": (1,)}
    got = dict(zip(V5_STATE_KEYS, (w1, b1, w2, b2, wa, ba, wc, bc)))
    bad = {k: (tuple(got[k].shape), v) for k, v in want.items() if tuple(got[k].shape) != v}
    if bad:
        raise ValueError(f"모양이 v5 BlockPolicy 와 다르다 (받은 것, 기대): {bad}")
    j = lambda a: jnp.asarray(a, dtype)
    return V5PolicyParams(w1=j(w1.T), b1=j(b1), w2=j(w2.T), b2=j(b2),
                          w_actor=j(wa.T), b_actor=j(ba),
                          w_critic=j(wc.T), b_critic=j(bc))


def to_v5_state_dict(p: V5PolicyParams) -> dict[str, np.ndarray]:
    """`V5PolicyParams` → torch `state_dict` 모양의 numpy 사전 (다시 전치해 `(out, in)`).

    `load_v5_params` 의 역이다 — 왕복이 값을 하나도 안 바꾸는지 시험이 확인한다.
    """
    vals = (np.asarray(p.w1).T, np.asarray(p.b1), np.asarray(p.w2).T, np.asarray(p.b2),
            np.asarray(p.w_actor).T, np.asarray(p.b_actor),
            np.asarray(p.w_critic).T, np.asarray(p.b_critic))
    return dict(zip(V5_STATE_KEYS, vals))


# ───────────────────────────────────────────────── 입력 만들기 (`model.py:13-27`)
def encode(raw, role, *, v5_cast: bool = True, dtype=jnp.float64) -> jnp.ndarray:
    """원특징 `(M, D)` → 망 입력 `(M, 37)`. v5 `encode` 와 같은 값 (머리말 ★입력 37칸).

    ★**시험·진단 전용이다** (2026-09-26 통합). 통합 경로(`dispatch.make_v5net_pick`)는 `v5feat.encode_rows`
      + `v5feat.as_net_input` 을 쓴다 — 37칸 인코딩의 정본은 `v5feat` 이고, 이 함수는 (ㄱ) v5 검증(빈 행렬·
      폭 초과 거절) 과 (ㄴ) v5feat 이 지원하지 않는 **buyer 의 BUY 줄 표시(칸 36)** 를 더한 겉옷이다.
      값 계산은 `v5feat.encode_rows` 한 곳에서만 한다 (사본 없음).

    `raw`     (M, D) — D ≤ 32. 남는 칸은 0 패딩이다.
    `role`    "seller"·"buyer"·"crane"·"state" (또는 0~3). **정적 인자**라 jit 안에서 그대로 풀린다.
    `v5_cast` v5 처럼 float32 로 한 번 깎는다 (머리말 ★). 동등성 대조는 True 가 필수다.

    ⚠️ 유한성 검사(`model.py:17-18`)는 자료 의존이라 여기서 안 한다 — `all_finite(raw)` 를 따로 부른다
       (통합 경로는 `dispatch.make_v5net_pick` 이 그것을 재서 위반 비트 `V_NET_NONFINITE` 를 켠다).
    ⚠️ buyer 는 v5 가 **줄 2개**(BUY, REJECT)만 받는다 (`model.py:24-26`).
    """
    from . import v5feat as VF
    idx = ROLES.index(role) if isinstance(role, str) else int(role)
    if not 0 <= idx < len(ROLES):
        raise ValueError(f"역할은 {ROLES} 중 하나여야 한다 — 받은 것: {role!r}")
    x = jnp.asarray(raw)
    if x.ndim != 2:
        raise ValueError(f"(M, D) 행렬이어야 한다 — 받은 모양: {x.shape} (model.py:15)")
    m, d = int(x.shape[0]), int(x.shape[1])
    if m == 0 or d > RAW_DIM:
        raise ValueError(f"줄이 1개 이상, 폭이 {RAW_DIM} 이하여야 한다 — 받은 모양: {x.shape}")
    if ROLES[idx] == "buyer" and m != 2:
        raise ValueError(f"buyer 는 BUY·REJECT 두 줄이다 — 받은 줄 수: {m} (model.py:25)")
    if v5_cast:
        x = jnp.asarray(x, jnp.float32)          # ★v5 가 망에 실제로 넣는 값
    #: buyer 는 v5feat 이 거절하므로 역할 칸을 뒤에서 옮겨 세운다 (칸 값 계산은 같은 함수 하나)
    base = VF.encode_rows(x, ROLES[idx] if ROLES[idx] != "buyer" else "crane")
    out = jnp.asarray(base, dtype)
    if ROLES[idx] == "buyer":
        out = out.at[:, RAW_DIM + ROLES.index("crane")].set(jnp.zeros((m,), dtype))
        out = out.at[:, RAW_DIM + idx].set(jnp.ones((m,), dtype))
        out = out.at[0, -1].set(jnp.ones((), dtype))   # BUY 줄 표시
    return out


def all_finite(raw) -> jnp.ndarray:
    """`model.py:17-18` 의 유한성 검사 — 거짓이면 v5 는 ValueError 다 (부르는 쪽이 전원 WAIT 로)."""
    return jnp.all(jnp.isfinite(jnp.asarray(raw)))


# ───────────────────────────────────────────────── 순전파 (`model.py:36-50`)
def trunk(p: V5PolicyParams, x: jnp.ndarray) -> jnp.ndarray:
    """공용 몸통 `(M, 37) → (M, H)`. Linear→tanh→Linear→tanh (`model.py:36-37`).

    ⚠️ `x @ w + b` 를 그냥 쓴다 — `exact.mul_exact`/`sum_seq` 를 **쓰지 않는다**. 근거는 머리말 ★ 측정
    (순차 합이 오히려 torch 와 더 멀다: 8.33e-16 vs 6.66e-16).
    """
    h = jnp.tanh(x @ p.w1 + p.b1)
    return jnp.tanh(h @ p.w2 + p.b2)


def actor_scores(p: V5PolicyParams, x: jnp.ndarray) -> jnp.ndarray:
    """줄마다 행동 점수 `(M,)` — v5 `distribution` 의 `logits` (`model.py:42`, 마스크 전)."""
    return (trunk(p, x) @ p.w_actor + p.b_actor)[..., 0]


def state_values(p: V5PolicyParams, x: jnp.ndarray) -> jnp.ndarray:
    """줄마다 상태 가치 `(M,)` — v5 `value` (`model.py:49-50`). 블록 상태 행렬을 주면 블록마다 하나."""
    return (trunk(p, x) @ p.w_critic + p.b_critic)[..., 0]


# ───────────────────────────────────────────────── 마스크·분포 (torch `Categorical` 그대로)
def masked_scores(scores: jnp.ndarray, mask) -> jnp.ndarray:
    """`logits.masked_fill(~mask, -inf)` (`model.py:47`). 마스크 칸의 nan·inf 도 여기서 사라진다."""
    return jnp.where(jnp.asarray(mask, bool), scores, NEG_INF)


def _normalized(scores: jnp.ndarray, mask) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`(정규화 로그확률, mask_ok)`. 축소는 **마지막 축**이라 `(M,)`·`(K,M)` 둘 다 된다.

    torch `Categorical(logits=z)` 는 `self.logits = z - z.logsumexp(-1, keepdim=True)` 를 들고 있고
    `log_prob` 은 그 값을 그대로 돌려준다 (`torch/distributions/categorical.py`). 여기도 같은 순서다.
    마스크가 전부 거짓이면 logsumexp 가 -inf 라 `-inf - (-inf) = nan` 이 되므로 -inf 로 가둔다.
    """
    z = masked_scores(scores, mask)
    ok = jnp.any(jnp.asarray(mask, bool), axis=-1, keepdims=True)
    mx = jnp.max(z, axis=-1, keepdims=True)
    mx = jnp.where(jnp.isfinite(mx), mx, jnp.zeros_like(mx))
    lse = jnp.log(jnp.sum(jnp.exp(z - mx), axis=-1, keepdims=True)) + mx
    return jnp.where(ok, z - lse, jnp.full_like(z, NEG_INF)), ok[..., 0]


def log_probs(scores: jnp.ndarray, mask) -> jnp.ndarray:
    """`(M,)` 정규화 로그확률 — torch `Categorical.logits`. 마스크 칸은 -inf."""
    return _normalized(scores, mask)[0]


def probs(scores: jnp.ndarray, mask) -> jnp.ndarray:
    """`(M,)` 확률 — torch `Categorical.probs` = `softmax(정규화 로그확률)`. 마스크 칸은 0."""
    lp, ok = _normalized(scores, mask)
    return jnp.where(ok[..., None], jax.nn.softmax(lp, axis=-1), jnp.zeros_like(lp))


def entropy(scores: jnp.ndarray, mask) -> jnp.ndarray:
    """`()` 엔트로피 — torch 구현 그대로: 로그확률을 `finfo.min` 으로 가둔 뒤 확률과 곱해 더한다.

    가두기(`clamp`)가 있어야 `-inf × 0 = nan` 이 안 난다 (`torch/distributions/categorical.py entropy`).
    곱은 `exact.mul_exact` 로 실체화한다 — 안 그러면 XLA 가 곱과 축소를 FMA 로 묶어 **jit 과 eager 의
    마지막 비트가 갈린다** (실측: 이 시험이 잡았다). torch 도 곱한 뒤 따로 더한다.
    """
    lp, _ = _normalized(scores, mask)
    lo = jnp.asarray(jnp.finfo(lp.dtype).min, lp.dtype)
    return -jnp.sum(mul_exact(jnp.maximum(lp, lo), probs(scores, mask)), axis=-1)


def log_prob_of(scores: jnp.ndarray, mask, action) -> jnp.ndarray:
    """`()` 고른 줄의 로그확률 — v5 `dist.log_prob(action)` (`runtime.py:136`).

    마스크가 전부 거짓이면 행동이 -1 이고 돌려주는 값은 -inf 다 (v5 는 그 전에 예외를 던진다).
    """
    lp = log_probs(scores, mask)
    a = jnp.clip(jnp.asarray(action, jnp.int32), 0, lp.shape[-1] - 1)
    return jnp.take_along_axis(lp, a[..., None], axis=-1)[..., 0]


def greedy_action(scores: jnp.ndarray, mask) -> jnp.ndarray:
    """`()` int32 — v5 고정운영 경로 `int(dist.probs.argmax())` (`runtime.py:135`).

    ★**점수가 아니라 확률의 argmax** 다. 단조 변환이라 보통 같지만, 두 점수가 부동소수점 한 칸 차이면
    확률이 같은 값으로 반올림돼 **앞 줄**이 이길 수 있다 — v5 와 같은 답을 내려면 이 순서를 지켜야 한다.
    마스크가 전부 거짓이면 -1 (v5 는 예외 → 부르는 쪽이 전원 WAIT).
    """
    ok = jnp.any(jnp.asarray(mask, bool), axis=-1)
    return jnp.where(ok, jnp.argmax(probs(scores, mask), axis=-1), -1).astype(jnp.int32)


def sample_action(key, scores: jnp.ndarray, mask) -> jnp.ndarray:
    """`()` int32 — 확률대로 추첨 (학습 수집 경로 `runtime.py:134`).

    ⚠️ **v5 와 같은 표본을 낼 수 없다** — v5 는 `torch.multinomial` 의 `torch.Generator` 흐름을 쓴다.
    동등성 대조는 `greedy_action`(추첨 끔) 으로 한다 ([[YR-319]] 가 가른 두 축 중 '최고점 선택').
    """
    ok = jnp.any(jnp.asarray(mask, bool), axis=-1)
    drawn = jax.random.categorical(key, log_probs(scores, mask), axis=-1)
    return jnp.where(ok, drawn, -1).astype(jnp.int32)


def decide(p: V5PolicyParams, x: jnp.ndarray, mask) -> V5Decision:
    """망 한 번 → 결정 한 건 (`V5Decision`). 몸통을 **한 번만** 지나간다.

    `x` `(M, 37)` · `mask` `(M,)` 이면 잎이 스칼라/`(M,)` 이고, 앞에 축을 더 붙이면 (예: `(K, M, 37)`)
    그 축이 그대로 앞에 붙는다 — 분포 계산이 전부 마지막 축 기준이다.

    행동은 **최고점 선택**이다 (`sample_actions=False` 인 고정운영·동등성 경로). 추첨으로 모으는 학습
    경로는 `scores = actor_scores(...)` → `sample_action(key, scores, mask)` → `log_prob_of(scores, mask, a)`
    로 따로 엮는다 — 그래야 어느 행동의 로그확률인지 헷갈리지 않는다.
    """
    h = trunk(p, x)
    scores = (h @ p.w_actor + p.b_actor)[..., 0]
    value = (h @ p.w_critic + p.b_critic)[..., 0]
    action = greedy_action(scores, mask)
    return V5Decision(action=action, log_prob=log_prob_of(scores, mask, action),
                      scores=masked_scores(scores, mask), probs=probs(scores, mask),
                      entropy=entropy(scores, mask), value=value,
                      mask_ok=jnp.any(jnp.asarray(mask, bool), axis=-1))


# ───────────────────────────────────────────────── 배치판 (크레인·세계를 쌓아 한 번에)
#: ★아래 다섯(`*_batch`·`greedy_policy_fn`)은 **시험·진단 전용**이다 — 생산 호출자가 없다
#:   (2026-09-26 grep 확인). 통합 경로는 `actor_scores` + `greedy_action` 을 한 줄에 직접 부르고,
#:   세계를 쌓는 것은 `jax.vmap(engine_step.run)` 이 바깥에서 한다 (망 가중치는 (B,…) 로 브로드캐스트).
#: `(점수 (K,M), 마스크 (K,M)) → (K,)`
greedy_action_batch = jax.vmap(greedy_action, in_axes=(0, 0))
#: `(params, x (K,M,37)) → (K,M)`
actor_scores_batch = jax.vmap(actor_scores, in_axes=(None, 0))
state_values_batch = jax.vmap(state_values, in_axes=(None, 0))
#: `(params, x (K,M,37), mask (K,M)) → V5Decision` 의 각 열 앞에 (K,)
decide_batch = jax.vmap(decide, in_axes=(None, 0, 0))


def greedy_policy_fn(params: V5PolicyParams, x: jnp.ndarray, mask) -> jnp.ndarray:
    """**망 층 편의 함수** — `(params, x (K,M,37), mask (K,M)) → (K,) int32`. 시험·진단이 쓴다.

    ★엔진 공동 규약이 **아니다** (2026-09-26 통합 · 검증 반박). 통합 접합점은 `dispatch.make_v5net_policy`
      이고 그 서명은 `(params, world, c3, fl, pr, open_) → (choice, lost, flags)` 다. 이 함수는
      `dispatch.py:25-28` 의 *순차* 규약 모양이지만 (ㄱ) 돌려주는 값이 오더 번호가 아니라 **후보 행 번호**이고
      (ㄴ) 37칸 중 19~23 이 **앞 크레인 선택에 의존**하므로 그 규약의 "행마다 독립" 을 만족하지 못한다.
      그대로 꽂으면 어긋난다 — 꽂지 마라.

    마스크가 전부 거짓인 행은 -1 → 부르는 쪽이 위반 비트를 켜고 그 결정을 전원 WAIT 로 바꾼다.

    ★"K² 중복 순전파" 에 대하여 (README 한계표 정정 · 2026-09-26 벡터화 렌즈 실측)
      (ㄱ) 통합 경로는 (K,N,F) 를 넘기지 **않는다** — `dispatch.make_v5net_pick` 이 `fo.x[k]` **한 줄
           (I=13, 37)** 만 망에 넣는다. (K,N,F) 를 넘기는 곳은 이 함수뿐이고 생산 호출자가 없다.
      (ㄴ) 망 순전파는 하루 계산의 **0.33~1.2%** (vmap 아래 2.6%) 다. 최적화할 자리가 아니다.
           정책 비용의 **90%** 는 `v5cond.joint_mask_items` 가 후보마다 `dry_run_joint` 을 부르는 것이다
           (결정당 K²·I 계획 — K=2·I=13 이면 52). 조각 8 의 (K,Amax) 실행가능 행렬이 그 자리다.
      (ㄷ) "결정 시작 때 (K,M) 점수를 미리 계산해 두고 마스크만 씌운다" 는 처방은 **원리상 성립하지 않는다**:
           37칸 중 19~23 이 '직전 크레인이 고른 것'(`ppo/crane.py:52-55`)이라 scan 단계 k≥1 의 행은 결정
           시작 시점에 아직 정해지지 않았고, 몸통이 37칸을 전부 섞는 조밀층이라 '그 열만 다시 계산' 이
           불가능하다. 실제로 가능한 절약은 **"K 줄을 만들지 말고 k 행 하나만 만든다"** 이고
           (정책 비용의 13~31%), 그쪽은 `make_v5net_pick` 머리말에 적혀 있다.
    """
    return greedy_action(actor_scores(params, x), mask)
