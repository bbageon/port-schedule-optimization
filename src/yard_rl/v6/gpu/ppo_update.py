"""v5 **PPO 갱신**(`ppo/update.py:16-82` `update`)의 배열판 — 같은 테이프 → 같은 갱신 ([[YR-327]] 조각 8).

■ 무엇인가 — 한 번의 갱신을 "고정 길이 scan" 하나로
    v5 는 파이썬 이중 루프다: epoch 2번 × 미니배치 20번 × 미니배치 안의 표본 64개 × 표본의 결정 1~수개.
    배열판은 **표본 칸을 미리 고정**해 두고 (epochs, n_mb, mb) 한 장의 색인표를 받아 `lax.scan` 한 번으로 돈다.

        v5 (파이썬)                                   배열판 (이 파일)
        ─────────────────────────────────────────────────────────────────────────────
        for epoch: order = rng.permutation(E)     →  호스트가 미리 뽑은 orders (E,) × epochs (★규칙 ③)
        for offset in range(0, E, 64)             →  (epochs, n_mb, mb) 색인표 · 남는 칸은 **-1**
        indices 마다 policy.distribution 재계산    →  (mb, Cmax, Amax, 37) 를 **한 번에** 순전파
        if kl > target_kl: break; break           →  `stopped` 캐리 + `lax.cond` (계산도 건너뛴다)
        optimizer.zero_grad/backward/step 제자리   →  `jax.value_and_grad` → 새 params 를 돌려주는 순수 함수
        raise FloatingPointError                  →  **U_* 표시 비트** (jit 안에서는 못 던진다)

■ ★v5 와 무엇이 같고 무엇이 못 같은가 (측정은 `tests/v6/test_gpu_ppo_update.py` 가 매번 다시 잰다)
  같은 것 (구조·정수·순서):
    · 미니배치 개수·경계·표본 순서 · 조기중단 지점 · **중단된 미니배치의 KL 은 기록**한다 (update.py:59)
    · 이득 정규화 조건 `len(active) > 1 and std > 1e-8` (update.py:23-27) · 정규화는 **전체** 칸에 적용
    · 손실 = `actor_loss + value_coef*value_loss − entropy_coef*entropy` (update.py:64)
    · 보고 정수 5종 (`minibatches`·`intervals`·`block_samples`·`active_block_samples`·`micro_actions`)
    · ★**Adam 의 걸음 수를 파라미터마다 따로 센다** — v5 는 `p.grad is None` 인 파라미터를 건너뛴다
      (`torch/optim/optimizer.py` — 상태를 만들지도, step 을 올리지도 않는다). 활성 표본이 **하나도 없는**
      미니배치에서는 `actor_loss = value_loss * 0` 이라 (update.py:54-55) 행동점수 머리(`actor.weight`·
      `actor.bias`)가 그래프에 안 붙는다 → 그 미니배치에서 actor 머리는 **m·v·걸음수·값 전부 그대로**다.
      `touched` 나무가 그 규칙을 그대로 옮긴다. (기울기가 0 인 것과 **다르다** — 0 이면 Adam 이 이전
       관성 m 으로 파라미터를 계속 움직인다.)
  못 같은 것 (우리 코드 밖 · 조각 7 머리말과 같은 이유):
    · v5 정책망은 **float32** 다. 배열판은 float64 → 점수부터 |Δ| ≈ 1.2e-07 이 깔린다 (`v5net.py` ★).
      그래서 손실·KL·기울기 노름은 **rtol 1e-4**, 최종 파라미터는 **atol 1e-5** 로 본다.
      `hyper.v5_cast=True` (기본) 는 v5 가 실제로 float32 로 깎아 넣는 두 자리를 그대로 깎는다:
        targets (`update.py:37` `torch.tensor(..., dtype=torch.float32)`) · 이득 (파이썬 float → float32 스칼라)
    · torch `.mean()`·`vector_norm` 의 축소 순서는 규정돼 있지 않다 (TensorIterator 벡터화 누산).
      미니배치 평균·노름은 `jnp.sum` 을 쓴다 — 순차 합을 강제해도 torch 와 가까워지지 않고 64·2368 단
      직렬 사슬만 남는다. 상대 오차 ~1e-16 으로 위의 1e-7 바닥에 묻힌다.
      반면 **결정 여러 개를 묶어 더하는 자리**(`torch.stack(new_logp).sum()` · `old_logp += c.log_prob`)는
      v5 의 왼쪽부터 차례 합을 그대로 지킨다 (`_seq_sum_masked`).
    · 기울기 클리핑 상수: torch 는 `max_norm / (norm + 1e-6)` 이다 (`torch/nn/utils/clip_grad.py:164`).
      **`optax.clip_by_global_norm` 은 그 1e-6 이 없다** (`max_norm / norm`) → 노름 0.5 에서 상대 2e-6 차이.
      그래서 클리핑은 손으로 쓴다 (`clip_by_global_norm_torch`). Adam 은 optax 와 수식이 같아 손으로 쓴
      쪽과 optax 쪽이 1e-15 안에서 같다 — 시험 `test_adam_matches_optax` 가 매번 증명한다.

■ ★실측 (2026-09-27 · 21블록·8구간 테이프 · 갱신 1회 · `tests/v6/test_gpu_ppo_update.py` 가 매번 다시 잰다)
      Windows CPU x64 (torch 2.14 · jax 0.11.2) / WSL GPU RTX 5090 (torch 2.13 · jax 0.11.2+cuda12)
      ─────────────────────────────────────────────────────────────────────────────────────
      Adam 한 걸음 vs torch (float64)        상대 2.2e-16 / 2.1e-16      ← 사실상 비트 일치
      Adam vs optax.adam                     상대 2.0e-16 / 2.0e-16
      클립된 기울기 vs torch                  상대 4.4e-16 / 0            ← 손으로 쓴 식이 맞다
      optax.clip_by_global_norm 과의 차이     상대 1.96e-06 (= 1e-6/노름) ← **그래서 안 쓴다**
      미니배치별 손실                         상대 1.6e-06 / 1.9e-06
      미니배치별 기울기 노름                  상대 1.5e-07 / 1.3e-07
      미니배치별 KL                           **절대** 6.7e-08 / 5.2e-08  ← 아래 ★KL 항 참조
      표본별 로그비                           절대 4.4e-07 / 4.5e-07
      최종 파라미터 (actor.bias 뺀 7장)       절대 1.3e-06 / 1.7e-06      ← 기준 1e-05 의 1/8
      갱신 **2회 연속** 뒤 파라미터           절대 2.2e-07 (걸음 수도 v5 와 같다)
      jit vs eager                            0 / 0

  ★KL 은 **상대 오차로 재면 안 된다** (3.9e-04 까지 벌어진다) — v5 가 `exp(r) − 1 − r` 을 **float32** 로
    계산하기 때문이다 (update.py:53). 로그비 r 이 작으면 `exp(r)` ≈ 1 이라 `− 1` 에서 **상쇄**가 일어나
    float32 한 칸의 절반(≈6e-08)이 오차로 남는다. 즉 작은 KL 은 **v5 쪽이 부정확**하고 배열판이 정확하다.
    ⚠️ 그래서 조기중단 문턱(target_kl 0.03)은 **원리상 갈릴 수 있다** — 조각 7 의 결정 뒤집힘과 같은 종류다.
       실측 여유: KL 절대오차 ≈2e-08 대 문턱까지의 거리 1.4e-03 → **7만 배**. 30일 학습에서 갱신이 700회면
       한 번쯤 갈릴 확률은 여전히 0 이 아니므로, 통합 시험은 '해시 일치' 가 아니라
       **'첫 갈린 지점까지의 접두사 일치 + 그 지점의 문턱 여유 기록'** 으로 판정해야 한다 (README 조각 8 항목).

  ★`actor.bias` 한 칸만은 atol 1e-05 를 못 맞춘다 (실측 6.3e-04) — 그리고 **맞출 필요가 없다**.
    행동점수 머리의 편향은 모든 후보 줄에 똑같이 더해지고 결정은 마스크 softmax 라 상수항이 지워진다 →
    **참 기울기가 정확히 0** 이고 (실측 8.3e-17) 남는 것은 반올림 잡음의 부호뿐이다. Adam 은 크기를 지우므로
    (m/√v ≈ ±1) 부호 하나가 걸음 하나(±lr = 3e-04)를 가른다. 그 칸은 v5 에서도 아무 답에 영향이 없다 —
    `test_action_head_bias_is_unidentifiable_so_its_drift_is_harmless` 가 편향을 5.0 흔들어도 결정·로그확률·
    확률·엔트로피·상태가치가 1.1e-15 안에서 그대로임을 증명한다. (v5 자체의 식별 불가 파라미터다.)

■ 속도 (2026-09-27 · v5 **실제 규모** R=60·B=21 = 표본 1,260 · 결정 1,919 · 미니배치 40 = epoch 2 × 20)
      Windows CPU x64   배열판 52ms   vs v5 1,696ms  → **32.5배** (컴파일 0.75초 1회)
      WSL GPU RTX 5090  배열판 11ms   vs v5 4,226ms  → **384배**  (컴파일 16.5초 1회)
  ⚠️ 이 값은 **갱신 부분만**이다 — 30일 학습 전체의 손익분기는 세계 굴리기가 지배한다 (통합 단계의 몫).
  ⚠️ 칸 크기가 메모리를 정한다: `rows` 는 E×Cmax×Amax×37×8 바이트다 (위 무대에서 5.6MB · Cmax 3·Amax 5).
     v5 실제 크레인 결정은 후보가 13줄까지 가고 한 구간에 결정이 더 많을 수 있으니, 통합 단계가 실제
     테이프로 Cmax·Amax 를 재야 한다 (Cmax 21·Amax 13 이면 같은 규모에서 ≈100MB).

■ 쓰는 법 (통합 단계 — 그대로 복사해 쓸 수 있다)
    hyper = hyper_from_v5(PPOConfig())            # v5 PPOConfig 를 오리처럼 읽는다 (torch 를 안 들인다)
    params = v5net.load_v5_params(torch.load(path))            # 또는 v5net.init_params(key)
    opt = init_adam(params)
    step = jax.jit(ppo_update, static_argnames=("hyper",))     # ★hyper 는 **정적 인자**다
    ...  # 갱신마다
    before = rng.bit_generator.state                           # np.random.default_rng(seed) — v5 와 같은 것
    orders = draw_orders(rng, n_entries=E, hyper=hyper)        # ★v5 와 같은 난수열 (E = R*B)
    params, opt, out = step(params, opt, tape, adv, returns, orders, hyper=hyper)
    rep = as_v5_report(out, n_intervals=R)                     # v5 update() 반환 사전과 같은 9개 키
    rewind_orders_rng(rng, before, n_entries=E,                # v5 처럼 '들어간 epoch 만큼' 만 소비
                      epochs_entered=int(out.epochs_entered))
    assert int(out.violations) == 0, violation_names(out.violations)   # v5 의 raise 자리

■ 들어오는 테이프 (`Tape`) — 칸을 고정한 v5 `buffer.Interval` 목록
    표본 칸 e 는 v5 의 `entries` 순서 그대로 **e = i*B + b** 다 (update.py:19-20 — 구간 i, 블록 b).
    `gae` 와 (R,B,…) 패딩은 `gpu/ppo_buffer.py`(조각 8 buffer 담당)의 몫이고, 이 파일은 그 결과를 받는다.
    ⚠️ 패딩 규약: 쓰지 않는 결정 칸(j ≥ n_choices)의 `mask` 도 **최소 한 칸은 True** 여야 한다
       (전부 False 면 정규화 로그확률이 -inf 뿐이라 값은 마스크로 지워지지만 읽기가 어렵다).
       값은 어차피 `n_choices` 마스크로 0 이 되므로 결과에는 영향이 없다 — 시험이 두 경우를 다 돌린다.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import v5net
from .exact import mul_exact
from .travel import div_exact
from .v5net import V5PolicyParams

__all__ = ["PPOHyper", "Tape", "AdamState", "MinibatchAux", "UpdateOut",
           "U_KL_NONFINITE", "U_LOSS_NONFINITE", "U_GRAD_NONFINITE", "U_NAMES",
           "hyper_from_v5", "init_adam", "adam_step", "global_norm",
           "clip_by_global_norm_torch", "clipped_surrogate", "normalize_advantage", "minibatch_loss",
           "ppo_update", "draw_orders", "rewind_orders_rng", "as_v5_report",
           "violation_names"]

F = jnp.float64

#: 갱신 표시 비트 — v5 가 `raise` 하는 세 자리 (`gpu/state.py` 의 세계 위반 비트와 **다른 집합**이다)
U_KL_NONFINITE = 1      # update.py:57-58  FloatingPointError("Non-finite PPO policy divergence")
U_LOSS_NONFINITE = 2    # update.py:65-66  FloatingPointError("Non-finite PPO loss")
U_GRAD_NONFINITE = 4    # update.py:69-70  clip_grad_norm_(..., error_if_nonfinite=True)
U_NAMES = {U_KL_NONFINITE: "KL_NONFINITE", U_LOSS_NONFINITE: "LOSS_NONFINITE",
           U_GRAD_NONFINITE: "GRAD_NONFINITE"}


def violation_names(bits) -> list[str]:
    """표시 비트 정수 → 이름 목록 (호스트 전용)."""
    b = int(bits)
    return [name for bit, name in U_NAMES.items() if b & bit]


# ───────────────────────────────────────────────── 하이퍼파라미터 (정적 인자)
class PPOHyper(NamedTuple):
    """v5 `PPOConfig` 중 **갱신이 쓰는 칸**만. 파이썬 수치라 `jax.jit(static_argnames=('hyper',))` 가 된다.

    기본값은 v5 `ppo/runtime.PPOConfig` 그대로다 (runtime.py:26-37). `adam_*`·`clip_eps` 는 v5 가
    torch 기본값으로 쓰는 값이다 (`torch.optim.Adam(betas=(0.9,0.999), eps=1e-8)` ·
    `clip_grad_norm_` 의 분모 보정 1e-6).
    """

    epochs: int = 2
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    clip_eps: float = 1e-6
    #: v5 가 float32 로 깎아 넣는 두 자리(targets·이득)를 똑같이 깎는다 (머리말 ★). 동등성 대조는 True.
    v5_cast: bool = True


def hyper_from_v5(config, *, v5_cast: bool = True) -> PPOHyper:
    """v5 `PPOConfig` (또는 같은 이름의 칸을 가진 무엇이든) → `PPOHyper`. **torch 를 들이지 않는다**."""
    return PPOHyper(epochs=int(config.epochs), minibatch_size=int(config.minibatch_size),
                    learning_rate=float(config.learning_rate), clip=float(config.clip),
                    value_coef=float(config.value_coef), entropy_coef=float(config.entropy_coef),
                    max_grad_norm=float(config.max_grad_norm), target_kl=float(config.target_kl),
                    v5_cast=bool(v5_cast))


# ───────────────────────────────────────────────── 들어오는 테이프
class Tape(NamedTuple):
    """칸을 고정한 갱신 입력. 표본 축 E = R×B 는 v5 `entries` 순서(e = i*B + b)와 같다.

    states    (E, 37)                블록 상태 줄 — v5 `intervals[i].states[b]` (float32 로 깎은 값)
    rows      (E, Cmax, Amax, 37)    결정마다 후보 행렬 — v5 `Choice.rows`. 빈 칸은 0
    mask      (E, Cmax, Amax) bool   후보 마스크 — v5 `Choice.mask`
    action    (E, Cmax) int32        v5 가 실제로 고른 줄 번호 — `Choice.action`
    old_logp  (E, Cmax)              수집 때의 로그확률 — `Choice.log_prob`
    n_choices (E,) int32             그 표본의 결정 수 (0 이면 **행동 학습에서 빠진다**, 가치는 배운다)

    ■ `gpu/ppo_buffer.IntervalBatch` 에서 오는 길 (통합 단계 — (R,B,…) 를 (E,…) 로 눕힌다)
        n = int(batch.n_intervals)          # ★패딩 구간은 빼야 한다 — v5 `entries` 는 실제 구간만이다
        e = n * batch.n_blocks
        tape = Tape(states=batch.states[:n].reshape(e, 37),
                    rows=batch.rows[:n].reshape(e, batch.c_max, batch.a_max, 37),
                    mask=batch.mask[:n].reshape(e, batch.c_max, batch.a_max),
                    action=batch.action[:n].reshape(e, batch.c_max),
                    old_logp=batch.log_prob[:n].reshape(e, batch.c_max),
                    n_choices=batch.n_choices[:n].reshape(e))
        adv, returns = <gae 출력>[:n].reshape(e)      # 정규화 **전** 값
      · `action` 의 패딩 -1 도, `mask` 가 전부 거짓인 패딩 결정도 그대로 받는다 (아래 ⚠️ 와 시험 참조).
      · f32 로 들어와도 된다 — 안에서 float64 로 올린다 (값은 안 바뀐다).
    """

    states: jnp.ndarray
    rows: jnp.ndarray
    mask: jnp.ndarray
    action: jnp.ndarray
    old_logp: jnp.ndarray
    n_choices: jnp.ndarray


class MinibatchAux(NamedTuple):
    """미니배치 하나의 부산물 — 시험이 v5 내부 값과 하나씩 맞춘다."""

    value_loss: jnp.ndarray     # ()
    actor_loss: jnp.ndarray     # ()
    entropy: jnp.ndarray        # ()
    kl: jnp.ndarray             # ()   v5 `float(torch.stack(approx_kls).mean())`
    log_ratio: jnp.ndarray      # (mb,) 표본별 로그비 — 활성 아닌 칸은 뜻이 없다
    advantage: jnp.ndarray      # (mb,) 정규화된 이득 (v5 `float(adv[i, b])`)
    n_active: jnp.ndarray       # () int32


class AdamState(NamedTuple):
    """Adam 상태. ★`count` 가 **파라미터마다 따로** 있다 — 머리말 ★ (torch 의 `p.grad is None` 규칙)."""

    mu: V5PolicyParams
    nu: V5PolicyParams
    count: V5PolicyParams       # 각 잎이 () int32


class UpdateOut(NamedTuple):
    """갱신 한 번의 결과. 앞 9칸은 v5 `update()` 반환 사전과 짝이 맞는다 (`as_v5_report`)."""

    loss: jnp.ndarray                        # () 적용된 미니배치 손실의 평균 (없으면 0.0)
    early_stopped: jnp.ndarray               # () bool
    max_kl: jnp.ndarray                      # () 기록된 KL 의 최댓값 (중단 배치 포함)
    max_grad_norm_before_clip: jnp.ndarray   # () 적용된 미니배치의 클립 전 노름 최댓값
    minibatches: jnp.ndarray                 # () int32 실제 적용된 미니배치 수
    block_samples: jnp.ndarray               # () int32 = E
    active_block_samples: jnp.ndarray        # () int32 = #(n_choices > 0)
    micro_actions: jnp.ndarray               # () int32 = Σ n_choices
    epochs_entered: jnp.ndarray              # () int32 ★v5 가 순열을 뽑은 epoch 수 (난수열 되감기용)
    violations: jnp.ndarray                  # () int32 U_* 비트
    adv_mean: jnp.ndarray                    # () 정규화에 쓴 평균
    adv_std: jnp.ndarray                     # () 정규화에 쓴 표준편차 (numpy ddof=0)
    adv_normalized: jnp.ndarray              # () bool 정규화를 적용했는가
    mb_loss: jnp.ndarray                     # (epochs*n_mb,) 미니배치별 손실
    mb_kl: jnp.ndarray                       # (epochs*n_mb,) 미니배치별 KL
    mb_grad_norm: jnp.ndarray                # (epochs*n_mb,) 미니배치별 클립 전 노름
    mb_applied: jnp.ndarray                  # (epochs*n_mb,) bool 파라미터를 실제로 고쳤는가
    mb_recorded_kl: jnp.ndarray              # (epochs*n_mb,) bool KL 을 기록했는가 (중단 배치 포함)
    mb_n_active: jnp.ndarray                 # (epochs*n_mb,) int32 활성 표본 수
    mb_log_ratio: jnp.ndarray                # (epochs*n_mb, mb) 표본별 로그비 — '처음 갈린 지점' 추적용
    mb_advantage: jnp.ndarray                # (epochs*n_mb, mb) 표본별 정규화 이득


def as_v5_report(out: UpdateOut, *, n_intervals: int) -> dict:
    """`UpdateOut` → v5 `update()` 가 돌려주는 사전과 **같은 키·같은 타입** (호스트 전용)."""
    return {"loss": float(out.loss), "early_stopped": bool(out.early_stopped),
            "max_kl": float(out.max_kl),
            "max_grad_norm_before_clip": float(out.max_grad_norm_before_clip),
            "minibatches": int(out.minibatches), "intervals": int(n_intervals),
            "block_samples": int(out.block_samples),
            "active_block_samples": int(out.active_block_samples),
            "micro_actions": int(out.micro_actions)}


# ───────────────────────────────────────────────── 호스트 난수 (★규칙 ③ — 배열 안에서 새로 뽑지 않는다)
def draw_orders(rng, *, n_entries: int, hyper: PPOHyper) -> np.ndarray:
    """v5 와 **같은 난수열**로 (epochs, n_mb, mb) 색인표. 남는 칸은 -1.

    v5 는 epoch 마다 `rng.permutation(len(entries))` 를 부르고 (update.py:32) 그 순서를
    `minibatch_size` 로 잘라 쓴다 (update.py:34-35). 마지막 배치는 짧으므로 -1 로 채운다.

    ⚠️ **난수 소비량이 다를 수 있다** — v5 는 조기중단하면 다음 epoch 의 순열을 **뽑지 않는다**.
       여기서는 (배열 길이가 정적이어야 하므로) `epochs` 개를 미리 다 뽑는다. 여러 갱신을 이어 돌려
       난수열까지 맞추려면 갱신 뒤 `rewind_orders_rng` 로 되감아야 한다.
    """
    e = int(n_entries)
    mb = int(hyper.minibatch_size)
    if e < 1 or mb < 1:
        raise ValueError("표본 수와 미니배치 크기는 1 이상이어야 한다")
    n_mb = -(-e // mb)                                  # ceil — v5 `range(0, E, mb)` 의 칸 수
    out = np.full((int(hyper.epochs), n_mb, mb), -1, dtype=np.int32)
    for k in range(int(hyper.epochs)):
        order = np.asarray(rng.permutation(e), dtype=np.int32)
        out[k].reshape(-1)[:e] = order                   # 남는 뒷칸은 -1 그대로
    return out


def rewind_orders_rng(rng, state_before, *, n_entries: int, epochs_entered) -> None:
    """`draw_orders` 가 더 뽑아 쓴 난수를 되감아 **v5 와 같은 상태**로 만든다 (제자리 수정).

    `state_before` 는 `draw_orders` 를 부르기 **전에** 떠 둔 `rng.bit_generator.state` 다.
    v5 는 실제로 들어간 epoch 수만큼만 `permutation` 을 부르므로 그만큼만 다시 뽑아 준다.
    """
    rng.bit_generator.state = state_before
    for _ in range(int(epochs_entered)):
        rng.permutation(int(n_entries))


# ───────────────────────────────────────────────── 작은 도우미
def _f32(x):
    """float32 로 한 번 깎고 float64 로 되올린다 — v5 가 텐서에 실제로 넣는 값 (`v5net.encode` ★ 와 같은 뜻)."""
    return jnp.asarray(jnp.asarray(x, jnp.float32), F)


def _masked_sum(x, m):
    """마스크 합 — `jnp.sum` (머리말 ★ '못 같은 것': torch `.mean()` 의 축소 순서는 규정돼 있지 않다)."""
    return jnp.sum(jnp.where(m, jnp.asarray(x, F), jnp.zeros((), F)))


def _masked_mean(x, m):
    """마스크 평균. 칸이 하나도 없으면 0.0 (v5 `value_loss * 0` — update.py:54-55).

    ⚠️ 나눗셈은 **평범한 `/`** 다 — `travel.div_exact` 는 `custom_vmap` 이라 **역전파가 안 된다**
       (실측: "Linearization failed to produce known values"). 여기는 손실 안이라 미분이 지나간다.
       `div_exact` 가 막는 것은 '배열 ÷ 펼쳐진 상수' 가 역수 곱으로 바뀌는 것인데, 여기 분모는
       **추적되는 스칼라 개수**라 그 재작성 대상이 아니다 (상수가 아니면 XLA 가 접지 못한다).
    """
    n = jnp.sum(jnp.asarray(m, jnp.int32)).astype(jnp.int32)
    s = _masked_sum(x, m)
    return jnp.where(n > 0, s / jnp.maximum(n, 1).astype(F), jnp.zeros((), F))


def _seq_sum_masked(x, m, axis_len: int):
    """마스크를 씌운 **왼쪽부터 차례 합** — v5 `0.0 + c0 + c1 + …` 순서 그대로 (`exact.sum_seq` 와 같은 규약).

    v5 의 두 자리를 함께 옮긴다: `old_logp += c.log_prob` (파이썬 float 누적, update.py:48) 와
    `torch.stack(new_logp).sum()` (update.py:50). 결정 수 Cmax 가 작으므로 정적 루프로 펼친다.
    첫 항의 `0.0 +` 는 장벽 뒤에 둬야 XLA 가 접지 않는다 (`exact.sum_python` 머리말 실측 2).
    """
    acc = lax.optimization_barrier(jnp.zeros(x.shape[:-1], F))
    for j in range(int(axis_len)):
        acc = jnp.where(m[..., j], acc + jnp.asarray(x[..., j], F), acc)
    return acc


def clipped_surrogate(log_ratio, advantage, clip: float):
    """v5 `update.clipped_surrogate` (update.py:10-13) 그대로 — 곱은 장벽 뒤에 두지 않아도 덧셈이 없다."""
    ratio = jnp.exp(log_ratio)
    lo, hi = jnp.asarray(1.0 - clip, F), jnp.asarray(1.0 + clip, F)
    return jnp.minimum(ratio * advantage, jnp.clip(ratio, lo, hi) * advantage)


# ───────────────────────────────────────────────── 이득 정규화 (update.py:19-27)
def normalize_advantage(adv, n_choices):
    """`(정규화된 이득, 평균, 표준편차, 적용했는가)`.

    v5 는 **행동 표본(`choices` 가 빈 칸이 아닌 것)** 만으로 평균·표준편차를 재고, 그 값으로
    **모든 칸**을 옮긴다 (빈 블록도 가치는 배우므로 — update.py:22). 조건은 두 개다:
    활성 표본이 2개 이상 (`len(active) > 1`) 이고 표준편차가 1e-8 을 넘을 때만.
    표준편차는 numpy `ndarray.std()` = ddof 0 (모집단) 이다.
    """
    adv = jnp.asarray(adv, F)
    active = jnp.asarray(n_choices, jnp.int32) > 0
    n = jnp.sum(active.astype(jnp.int32)).astype(jnp.int32)
    nf = jnp.maximum(n, 1).astype(F)
    mean = div_exact(_masked_sum(adv, active), nf)
    dev = adv - mean
    var = div_exact(_masked_sum(mul_exact(dev, dev), active), nf)
    std = jnp.sqrt(var)
    apply = (n > 1) & (std > 1e-8)
    return (jnp.where(apply, div_exact(dev, jnp.where(apply, std, jnp.ones((), F))), adv),
            mean, std, apply)


# ───────────────────────────────────────────────── 미니배치 하나 (update.py:34-64)
def minibatch_loss(params: V5PolicyParams, tape: Tape, adv_n, returns, idx, hyper: PPOHyper):
    """`(loss, MinibatchAux)`. `idx` `(mb,)` 는 표본 번호이고 **-1 은 빈 칸**이다.

    v5 순서 그대로:
      ① `value_loss = (policy.value(states) − targets)².mean()` — 미니배치의 **모든** 칸 (update.py:38)
      ② 결정이 있는 칸만 행동 항 — 결정마다 분포를 다시 만들어 로그확률·엔트로피 (update.py:40-53)
      ③ `actor_loss = −mean(objectives)` · `entropy = mean(Σ엔트로피)` · `kl = mean(exp(r)−1−r)`
      ④ `loss = actor_loss + value_coef·value_loss − entropy_coef·entropy` (update.py:64)
    """
    idx = jnp.asarray(idx, jnp.int32)
    valid = idx >= 0
    i = jnp.maximum(idx, 0)

    # ① 가치 머리 — targets 는 v5 가 float32 로 깎아 넣는다 (update.py:37)
    st = jnp.asarray(tape.states, F)[i]
    v = v5net.state_values(params, st)
    tgt = jnp.asarray(returns, F)[i]
    tgt = _f32(tgt) if hyper.v5_cast else tgt
    resid = v - tgt
    value_loss = _masked_mean(mul_exact(resid, resid), valid)   # torch `.square().mean()` 2단

    # ② 행동 머리 — (mb, Cmax, Amax, 37) 를 한 번에
    rows = jnp.asarray(tape.rows, F)[i]
    mask = jnp.asarray(tape.mask, bool)[i]
    scores = v5net.actor_scores(params, rows)                       # (mb, Cmax, Amax)
    lp = v5net.log_prob_of(scores, mask, jnp.asarray(tape.action, jnp.int32)[i])
    ent = v5net.entropy(scores, mask)                               # (mb, Cmax)
    cmax = int(tape.rows.shape[1])
    nch = jnp.asarray(tape.n_choices, jnp.int32)[i]
    cmask = jnp.arange(cmax, dtype=jnp.int32)[None, :] < nch[:, None]
    new_logp = _seq_sum_masked(lp, cmask, cmax)                     # torch.stack(new_logp).sum()
    old_logp = _seq_sum_masked(jnp.asarray(tape.old_logp, F)[i], cmask, cmax)
    ent_sum = _seq_sum_masked(ent, cmask, cmax)
    log_ratio = new_logp - old_logp

    # ③ 활성 표본 = 결정이 있는 칸 (update.py:42-43 `if not choices: continue`)
    active = valid & (nch > 0)
    adv = jnp.asarray(adv_n, F)[i]
    adv = _f32(adv) if hyper.v5_cast else adv         # v5 는 파이썬 float 를 float32 텐서와 곱한다
    obj = clipped_surrogate(log_ratio, adv, float(hyper.clip))
    actor_loss = -_masked_mean(obj, active)
    entropy_mean = _masked_mean(ent_sum, active)
    r = lax.stop_gradient(log_ratio)                 # update.py:53 `.detach()`
    kl = _masked_mean(jnp.exp(r) - 1.0 - r, active)

    # ④
    loss = (actor_loss + jnp.asarray(hyper.value_coef, F) * value_loss
            - jnp.asarray(hyper.entropy_coef, F) * entropy_mean)
    aux = MinibatchAux(value_loss=value_loss, actor_loss=actor_loss, entropy=entropy_mean,
                       kl=kl, log_ratio=log_ratio, advantage=adv,
                       n_active=jnp.sum(active.astype(jnp.int32)).astype(jnp.int32))
    return loss, aux


# ───────────────────────────────────────────────── 최적화기 (update.py:67-71)
def global_norm(grads: V5PolicyParams):
    """torch `clip_grad_norm_` 이 재는 전역 노름 — **파라미터별 노름을 다시 노름** 한다.

    `torch/nn/utils/clip_grad.py:105-107`: `vector_norm(stack([vector_norm(g) for g in grads]))`.
    수학적으로는 전체를 한 벡터로 본 2-노름과 같고, 반올림만 다르다 (그 구조를 그대로 따른다).
    """
    per = [jnp.sqrt(jnp.sum(mul_exact(jnp.asarray(g, F), jnp.asarray(g, F)))) for g in grads]
    sq = jnp.stack(per)
    return jnp.sqrt(jnp.sum(mul_exact(sq, sq)))


def clip_by_global_norm_torch(grads: V5PolicyParams, norm, hyper: PPOHyper):
    """torch 와 **같은 식**으로 클리핑: `coef = min(max_norm / (norm + 1e-6), 1)`.

    ⚠️ `optax.clip_by_global_norm` 은 이 1e-6 이 없다 (`optax/transforms/_clipping.py:105`
       — `(t / g_norm) * max_norm`). 노름 0.5·max 0.5 근방에서 상대 2e-6 차이가 나므로 손으로 쓴다.
       또 torch 는 **클립이 필요 없을 때도** 1.0 을 곱한다 (값은 안 바뀐다).
    """
    coef = jnp.minimum(div_exact(jnp.asarray(hyper.max_grad_norm, F),
                                 norm + jnp.asarray(hyper.clip_eps, F)), jnp.ones((), F))
    return jax.tree.map(lambda g: jnp.asarray(g, F) * coef, grads)


def init_adam(params: V5PolicyParams) -> AdamState:
    """`torch.optim.Adam(policy.parameters(), lr=…)` 의 초기 상태 (runtime.py:80). 걸음 수는 파라미터마다 0."""
    zero = jax.tree.map(lambda p: jnp.zeros_like(jnp.asarray(p, F)), params)
    count = jax.tree.map(lambda p: jnp.zeros((), jnp.int32), params)
    return AdamState(mu=zero, nu=zero, count=count)


def adam_step(params: V5PolicyParams, state: AdamState, grads: V5PolicyParams,
              touched: V5PolicyParams, hyper: PPOHyper):
    """`torch.optim.Adam` 한 걸음 — 식·순서를 `torch/optim/adam.py:456-545` 그대로.

        m ← m + (g − m)(1 − β₁)                    (`exp_avg.lerp_(grad, 1-β₁)`)
        v ← v·β₂ + g·g·(1 − β₂)                    (`mul_`+`addcmul_`)
        step_size = lr / (1 − β₁^t) ·  denom = √v / √(1 − β₂^t) + ε
        p ← p − step_size · m / denom              (`param.addcdiv_(exp_avg, denom, value=-step_size)`)

    ★`touched` 가 거짓인 파라미터는 **아무것도 바뀌지 않는다** (m·v·걸음수·값) — torch 가
      `p.grad is None` 인 파라미터를 건너뛰는 규칙이다 (머리말 ★).
    """
    b1, b2 = float(hyper.adam_b1), float(hyper.adam_b2)
    lr, eps = float(hyper.learning_rate), float(hyper.adam_eps)
    one_m_b1, one_m_b2 = 1.0 - b1, 1.0 - b2

    def leaf(p, m, v, c, g, on):
        p, m, v, g = (jnp.asarray(z, F) for z in (p, m, v, g))
        c2 = c + jnp.ones((), jnp.int32)
        step = c2.astype(F)
        m2 = m + mul_exact(g - m, one_m_b1)
        v2 = mul_exact(v, b2) + mul_exact(mul_exact(g, g), one_m_b2)
        bc1 = 1.0 - jnp.power(jnp.asarray(b1, F), step)
        bc2 = 1.0 - jnp.power(jnp.asarray(b2, F), step)
        step_size = div_exact(jnp.asarray(lr, F), bc1)
        denom = div_exact(jnp.sqrt(v2), jnp.sqrt(bc2)) + eps
        p2 = p - mul_exact(step_size, div_exact(m2, denom))
        keep = jnp.asarray(on, bool)
        return (jnp.where(keep, p2, p), jnp.where(keep, m2, m), jnp.where(keep, v2, v),
                jnp.where(keep, c2, c))

    out = jax.tree.map(leaf, params, state.mu, state.nu, state.count, grads, touched)
    flat = [out[k] for k in range(len(params))]
    new_params = V5PolicyParams(*[q[0] for q in flat])
    return new_params, AdamState(mu=V5PolicyParams(*[q[1] for q in flat]),
                                 nu=V5PolicyParams(*[q[2] for q in flat]),
                                 count=V5PolicyParams(*[q[3] for q in flat]))


def _touched(has_active) -> V5PolicyParams:
    """어느 파라미터가 이 미니배치의 그래프에 붙었나 (머리말 ★ · update.py:54-55).

    몸통·가치 머리는 `value_loss` 로 항상 붙는다. 행동점수 머리는 활성 표본이 있을 때만 붙는다.
    """
    yes = jnp.asarray(True)
    on = jnp.asarray(has_active, bool)
    return V5PolicyParams(w1=yes, b1=yes, w2=yes, b2=yes,
                          w_actor=on, b_actor=on, w_critic=yes, b_critic=yes)


# ───────────────────────────────────────────────── 갱신 한 번 (update.py:16-82)
def ppo_update(params: V5PolicyParams, opt_state: AdamState, tape: Tape, adv, returns,
               orders, hyper: PPOHyper = PPOHyper()):
    """PPO 갱신 한 번. `(새 params, 새 opt_state, UpdateOut)` 를 돌려주는 **순수 함수**.

    `adv`·`returns` `(E,)` 는 `gpu/ppo_buffer.gae` 의 **정규화 전** 출력이다 (정규화는 v5 처럼 이 안에서).
    `orders` `(epochs, n_mb, mb)` 는 `draw_orders` 가 만든 호스트 색인표 (-1 = 빈 칸).

    조기중단(KL) 은 자료 의존이므로 `stopped` 를 캐리로 들고 `lax.cond` 로 **계산째 건너뛴다**.
    v5 와 같은 두 가지를 지킨다: ① 중단된 미니배치는 파라미터를 **안 고치고** ② 그 KL 은 **기록한다**.
    """
    orders = jnp.asarray(orders, jnp.int32)
    if orders.ndim != 3:
        raise ValueError(f"orders 는 (epochs, n_mb, mb) 여야 한다 — 받은 모양: {orders.shape}")
    n_epochs, n_mb, mb = (int(v) for v in orders.shape)
    if (n_epochs, mb) != (int(hyper.epochs), int(hyper.minibatch_size)):
        raise ValueError(f"orders 모양 {orders.shape} 이 hyper(epochs={hyper.epochs}, "
                         f"minibatch_size={hyper.minibatch_size}) 와 다르다 — draw_orders 를 쓰라")
    steps = n_epochs * n_mb
    flat = orders.reshape(steps, mb)
    is_epoch_start = jnp.asarray(np.arange(steps) % n_mb == 0)

    adv_n, adv_mean, adv_std, adv_applied = normalize_advantage(adv, tape.n_choices)
    grad_fn = jax.value_and_grad(minibatch_loss, has_aux=True)

    def body(carry, step):
        params, opt, stopped, viol, n_done, loss_sum, max_kl, max_norm, n_epoch = carry
        idx, epoch_start = step
        # 들어간 epoch 수 — v5 는 순열을 epoch 머리에서 뽑는다 (중단 뒤 epoch 는 뽑지 않는다)
        n_epoch = n_epoch + (epoch_start & ~stopped).astype(jnp.int32)

        def skip(_):
            zero, row = jnp.zeros((), F), jnp.zeros((mb,), F)
            return (params, opt, stopped, viol, n_done, loss_sum, max_kl, max_norm,
                    zero, zero, zero, jnp.asarray(False), jnp.asarray(False),
                    jnp.zeros((), jnp.int32), row, row)

        def run(_):
            (loss, aux), grads = grad_fn(params, tape, adv_n, returns, idx, hyper)
            kl_bad = ~jnp.isfinite(aux.kl)
            loss_bad = ~jnp.isfinite(loss)
            over_kl = aux.kl > jnp.asarray(hyper.target_kl, F)
            norm = global_norm(grads)
            norm_bad = ~jnp.isfinite(norm)
            recorded = ~kl_bad                       # v5 는 유한할 때만 append (update.py:57-59)
            apply = recorded & ~over_kl & ~loss_bad & ~norm_bad
            bits = (jnp.where(kl_bad, U_KL_NONFINITE, 0)
                    | jnp.where(loss_bad & ~kl_bad, U_LOSS_NONFINITE, 0)
                    | jnp.where(norm_bad & ~kl_bad & ~loss_bad, U_GRAD_NONFINITE, 0))
            clipped = clip_by_global_norm_torch(grads, norm, hyper)
            stepped, opt2 = adam_step(params, opt, clipped, _touched(aux.n_active > 0), hyper)
            keep = jnp.asarray(apply, bool)
            new_params = jax.tree.map(lambda a, b: jnp.where(keep, a, b), stepped, params)
            new_opt = jax.tree.map(lambda a, b: jnp.where(keep, a, b), opt2, opt)
            return (new_params, new_opt,
                    stopped | over_kl | kl_bad | loss_bad | norm_bad,
                    viol | bits.astype(jnp.int32),
                    n_done + keep.astype(jnp.int32),
                    loss_sum + jnp.where(keep, loss, jnp.zeros((), F)),
                    jnp.where(recorded, jnp.maximum(max_kl, aux.kl), max_kl),
                    jnp.where(keep, jnp.maximum(max_norm, norm), max_norm),
                    loss, aux.kl, norm, keep, recorded, aux.n_active,
                    aux.log_ratio, aux.advantage)

        (params, opt, stopped, viol, n_done, loss_sum, max_kl, max_norm,
         mb_loss, mb_kl, mb_norm, mb_applied, mb_rec, mb_active,
         mb_lr, mb_adv) = lax.cond(stopped, skip, run, 0)
        carry = (params, opt, stopped, viol, n_done, loss_sum, max_kl, max_norm, n_epoch)
        return carry, (mb_loss, mb_kl, mb_norm, mb_applied, mb_rec, mb_active, mb_lr, mb_adv)

    zero_f, zero_i = jnp.zeros((), F), jnp.zeros((), jnp.int32)
    init = (params, opt_state, jnp.asarray(False), zero_i, zero_i, zero_f, zero_f, zero_f, zero_i)
    (params, opt_state, stopped, viol, n_done, loss_sum, max_kl, max_norm, n_epoch), ys = lax.scan(
        body, init, (flat, is_epoch_start))
    mb_loss, mb_kl, mb_norm, mb_applied, mb_rec, mb_active, mb_lr, mb_adv = ys

    n_choices = jnp.asarray(tape.n_choices, jnp.int32)
    out = UpdateOut(
        # v5 는 `np.mean(losses)` (numpy 쌍합) 이고 여기는 scan 누적합이다 — 상대 ~1e-16 차이로
        # float32 바닥(1e-06)에 묻힌다. 적용된 배치가 없으면 0.0 (update.py:77).
        loss=jnp.where(n_done > 0, div_exact(loss_sum, jnp.maximum(n_done, 1).astype(F)), zero_f),
        early_stopped=stopped, max_kl=max_kl, max_grad_norm_before_clip=max_norm,
        minibatches=n_done, block_samples=jnp.asarray(n_choices.shape[0], jnp.int32),
        active_block_samples=jnp.sum((n_choices > 0).astype(jnp.int32)).astype(jnp.int32),
        micro_actions=jnp.sum(n_choices).astype(jnp.int32), epochs_entered=n_epoch, violations=viol,
        adv_mean=adv_mean, adv_std=adv_std, adv_normalized=adv_applied,
        mb_loss=mb_loss, mb_kl=mb_kl, mb_grad_norm=mb_norm, mb_applied=mb_applied,
        mb_recorded_kl=mb_rec, mb_n_active=mb_active, mb_log_ratio=mb_lr, mb_advantage=mb_adv)
    return params, opt_state, out
