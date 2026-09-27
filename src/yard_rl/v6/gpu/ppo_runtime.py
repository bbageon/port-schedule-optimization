"""구간 보상과 수집 흐름의 **배열판** ([[YR-327]] 조각 8 · key=runtime).

v5 정본 `ppo/runtime.py:59-215` (`PPORuntime`) 에서 **학습 루프의 회계 부분**만 옮긴다 —
60초 동기화 경계마다 "비용이 그 60초 동안 얼마나 늘었나" 를 보상으로 바꾸고, 그 사이에 내린
결정을 블록별로 모아 두는 절차다. 망 순전파(조각 7 `v5net.py`)·Φ 계산(조각 5 `phi.py`)·
PPO 갱신(`update`)·GAE(`buffer`)는 **여기서 다시 구현하지 않는다**.

    경계 t 마다            Φ(t) 를 읽는다                     … read_cost (phi.py 재사용)
                           delta = Φ(t) − Φ(t−60)
                           reward = −delta / reward_scale_krw  … 팀 보상 하나 (블록 21개 공통)
                           구간 기록 = (t−60, t, 상태 (B,37), 가치 (B,), 보상, 결정 묶음)
    결정마다               후보 행렬·마스크·고른 번호를 그 블록 칸에 쌓는다   … select_record

■ ★v5 는 **공유 팀 보상**이다 — 그 사실을 그대로 재현한다
  보상은 터미널 **전체** 비용의 증가분 하나이고, 21개 블록이 **같은 값**을 받는다
  (`ppo/buffer.py:28` 주석 "Team reward, NOT the sum of 21 copies"). 그래서 어느 블록의
  어느 결정이 비용을 줄였는지 **가릴 수 없다** — [[YR-326]] 이 지적한 한계이고 v6 의
  반사실 신용배분(`gpu/policy.py`)이 노리는 자리다. 다만 조각 8 의 목표는 **동등성**이므로
  여기서는 고치지 않는다. 고치는 것은 동등성이 통과한 **뒤**의 별도 축이다.

■ 파이썬 `None`·예외를 배열에서 어떻게 다루나
  v5 는 `self.time_s is None` 으로 "아직 첫 경계 전" 을 나타내고, 불변식이 깨지면 예외를
  던진다. jit 안에서는 둘 다 불가능하므로
    · `None`  → `started` bool 플래그 (`time_s` 는 +inf 로 둔다)
    · 예외    → `flags` **위반 비트** 누적 (`F_*`) — 호스트가 뒤에 한 번 검사해 크게 실패한다
  `gpu/state.py` 의 `V_*` (세계 불변식) 와 **다른 이름표**를 쓴다: 여기 비트는 *학습 루프*
  쪽 계약이고 세계 상태와 섞이면 어느 층이 깨졌는지 못 가린다.
  ⚠️ **비트가 켜진 뒤의 상태값은 의미 없다.** v5 는 그 자리에서 예외로 런을 끝내므로 "예외 뒤의
  올바른 상태" 라는 것이 없다. 호스트는 갱신 단위마다 `flags` 를 보고 그 세계를 실격시킨다
  (조용히 이상한 답을 내지 않는 것이 목적이고, 이어서 정확히 굴러가는 것이 목적이 아니다).
  단 `select` 만은 예외로 **v5 가 던지는 네 자리에서 기록도 계수도 하지 않는다** — v5 가
  `role_counts` 증가(runtime.py:145) 보다 **앞서** 던지기 때문이고, 그 계수기는 보고에 나간다.

■ 부동소수점 규약 (`exact.py` 머리말)
  보상의 나눗셈은 `exact.div_const` — `−delta / 1e6` 이 역수 곱으로 접히면 마지막 비트가 갈린다.
  Φ 자체의 합산 순서는 `phi.py` 가 이미 지킨다 (조각 5 에서 v5 와 비트 일치 확인).

■ 이 파일이 **안 하는 것** (다른 담당·다른 조각)
  · 망 순전파·행동 고르기 → `v5net.py`·`dispatch.make_v5net_pick` (조각 7)
  · Φ 네 항 계산          → `phi.py` (조각 5)
  · GAE·미니배치·Adam     → 조각 8 의 `buffer`·`update` 담당
  · 블록 요약 8칸 만들기   → `v5feat.block_row` (조각 7) · 터미널 축 묶음은 `host_terminal.py`
  · `workload` 잠재함수 보상 성형(`runtime.py:167,180-184`) → **범위 밖**. v5 의 PPO 정본 경로는
    `workload=None` 이라 성형이 0 이다 (v6 시험 파일럿은 음성 결과였다). 훅 자리만 이름으로 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax.numpy as jnp

from .events import TIME_DTYPE
from .exact import div_const
from .phi import PhiArrays, terminal_cost_krw
from .state import EMPTY_TIME, OrderArrays

__all__ = [
    "EPS_TIME", "COST_FELL_EPS", "ROLES", "ROLE_SELLER", "ROLE_BUYER", "ROLE_CRANE", "ROLE_STATE",
    "CRANE_KINDS", "F_REVIEW_TIME_BAD", "F_CLOCK_BACKWARD", "F_COST_NONFINITE", "F_COST_FELL",
    "F_WINDOW_EDGE_MISSED", "F_DECISION_TIME_BAD", "F_DECISION_NO_BOUNDARY", "F_DECISION_EARLY",
    "F_PENDING_OVERFLOW", "F_MASK_EMPTY", "F_ACTION_OVERFLOW", "FLAG_NAMES", "flag_names",
    "FATAL_BITS", "raise_on_flags",
    "RuntimeConfig", "PendingTape", "RuntimeState", "IntervalRow", "BoundaryOut", "SelectOut",
    "CostOut", "CounterTape", "empty_pending", "new_state", "collecting_at", "read_cost",
    "select_record", "boundary", "refresh_values", "finish", "count_crane_action", "report",
    "cost_inputs_from_v5", "counter_tape_from_v5", "tape_read", "tape_diff",
]

F = TIME_DTYPE

# ───────────────────────────────────────────────── 상수 (v5 정본의 숫자 그대로)
#: 시각 비교 여유 — `runtime.py:134,161,172,197` 의 `1e-6`
EPS_TIME = 1e-6
#: "누적 비용이 줄었다" 판정 문턱 — `runtime.py:177` 의 `-1e-5`
COST_FELL_EPS = 1e-5

#: 역할 순서 = `ppo/model.py:9 ROLES` = 원핫 칸 번호
ROLES = ("seller", "buyer", "crane", "state")
ROLE_SELLER, ROLE_BUYER, ROLE_CRANE, ROLE_STATE = 0, 1, 2, 3
N_ROLES = len(ROLES)

#: 크레인 행동 종류 순서 = `ppo/crane.py:16-17 KINDS` (= `state.PK_*`)
CRANE_KINDS = ("SERVE", "PRE_REHANDLE", "REPOSITION", "WAIT")
N_KINDS = len(CRANE_KINDS)


# ───────────────────────────────────────────────── 위반 비트 (v5 가 예외를 던지는 자리)
#: `boundary`: 검토 시각이 비유한·음수 (`runtime.py:159-160` ValueError)
F_REVIEW_TIME_BAD = 1
#: `boundary`: 검토 시계가 거꾸로 갔다 (`runtime.py:161-162` RuntimeError)
F_CLOCK_BACKWARD = 2
#: `read_cost`/`boundary`: Φ 가 비유한 (`runtime.py:168-169` FloatingPointError)
F_COST_NONFINITE = 4
#: `boundary`: 누적 비용이 줄었다 = 회계 자료 유실 (`runtime.py:177-178` RuntimeError)
F_COST_FELL = 8
#: `boundary`: 학습창 경계가 구간 **안쪽**에 들어왔다 (`runtime.py:173-175` RuntimeError)
F_WINDOW_EDGE_MISSED = 16
#: `select`: 결정 시각이 비유한·음수 (`runtime.py:130-131` ValueError)
F_DECISION_TIME_BAD = 32
#: `select`: 초기 동기화 경계 전에 결정이 왔다 (`runtime.py:132-133` RuntimeError)
F_DECISION_NO_BOUNDARY = 64
#: `select`: 결정 시각이 수집 구간보다 앞이다 (`runtime.py:134-135` RuntimeError)
F_DECISION_EARLY = 128
#: `select`: 한 구간·한 블록의 결정 칸(`cmax`)이 부족해 못 담았다 — **조용한 유실 금지**
F_PENDING_OVERFLOW = 256
#: `select`: 행동 마스크가 전부 거짓 (`ppo/model.py:45-46` ValueError)
F_MASK_EMPTY = 512
#: `select`: 후보 수가 칸(`amax`)보다 많아 잘렸다 — 조용한 유실 금지
F_ACTION_OVERFLOW = 1024

FLAG_NAMES: dict[int, str] = {
    F_REVIEW_TIME_BAD: "REVIEW_TIME_BAD", F_CLOCK_BACKWARD: "CLOCK_BACKWARD",
    F_COST_NONFINITE: "COST_NONFINITE", F_COST_FELL: "COST_FELL",
    F_WINDOW_EDGE_MISSED: "WINDOW_EDGE_MISSED", F_DECISION_TIME_BAD: "DECISION_TIME_BAD",
    F_DECISION_NO_BOUNDARY: "DECISION_NO_BOUNDARY", F_DECISION_EARLY: "DECISION_EARLY",
    F_PENDING_OVERFLOW: "PENDING_OVERFLOW", F_MASK_EMPTY: "MASK_EMPTY",
    F_ACTION_OVERFLOW: "ACTION_OVERFLOW",
}


#: ★**호스트가 반드시 크게 실패해야 하는 비트** (2026-09-27 통합·수정 단계에서 추가).
#:  jit 안에서는 예외를 던질 수 없으니 비트로 모아 두고, 호스트로 돌아온 자리에서 이것을 검사한다.
#:  v5 는 이 자리에서 모두 예외를 던진다 — 비트만 켜고 지나가면 **잘린 테이프로 학습하고도 '완료'** 가 된다
#:  (검증 반박 '결정 칸 넘침이 조용히 지나간다'). `F_PENDING_OVERFLOW`·`F_ACTION_OVERFLOW` 는 v5 에
#:  대응물이 없는 배열판 고유 비트지만 **조용한 유실**이라 같은 등급으로 둔다.
FATAL_BITS = (F_REVIEW_TIME_BAD | F_CLOCK_BACKWARD | F_COST_NONFINITE | F_COST_FELL
              | F_WINDOW_EDGE_MISSED | F_DECISION_TIME_BAD | F_DECISION_NO_BOUNDARY
              | F_DECISION_EARLY | F_PENDING_OVERFLOW | F_MASK_EMPTY | F_ACTION_OVERFLOW)


def raise_on_flags(flags, *, where: str = "", fatal: int = FATAL_BITS) -> None:
    """켜진 비트가 `fatal` 에 걸리면 던진다 — v5 가 예외를 던지는 자리를 호스트에서 재현한다.

    `FloatingPointError` 로 던지는 이유: 부르는 쪽(`gpu/train`)이 이미 갱신 위반 비트를 그 예외로
    던지고 있어 한 가지로 잡을 수 있다. 비트 이름을 그대로 메시지에 싣는다.
    """
    bits = int(flags) & int(fatal)
    if bits:
        raise FloatingPointError(
            f"학습 회계 위반 비트 {bits} = {flag_names(bits)}{(' @' + where) if where else ''}")


def flag_names(v) -> tuple[str, ...]:
    """비트합 → 켜진 비트 이름들. `state.violation_names` 와 같은 꼴 (모르는 비트는 'BIT_<n>')."""
    v, out, bit = int(v), [], 1
    while v:
        if v & 1:
            out.append(FLAG_NAMES.get(bit, f"BIT_{bit}"))
        v >>= 1
        bit <<= 1
    return tuple(out)


# ───────────────────────────────────────────────── 설정 (jit static — 파이썬 값)
@dataclass(frozen=True)
class RuntimeConfig:
    """`PPOConfig` 중 **수집 흐름이 읽는 칸** + 고정 칸 크기.

    `PPOConfig` 의 나머지(할인·클립·손실계수)는 갱신 담당(`update`)이 읽는다. 여기 값은 전부
    파이썬 스칼라라 `jax.jit(static_argnums=...)` 로 넘기면 분기가 트레이싱 때 정적으로 풀린다.
    """

    n_blocks: int = 21                       #: 블록 수 B (= `len(runtime.bids)`)
    cmax: int = 64                           #: 한 구간·한 블록 결정 칸 (실측 최대 39 · 12시간 무대)
    amax: int = 29                           #: 후보 칸 (판매자 1+20+8 = 29 · 크레인 k_max+1)
    input_dim: int = 37                      #: 망 입력 폭 (`ppo/model.INPUT_DIM`)
    rollout_intervals: int = 60              #: 갱신 한 번에 모으는 구간 수 (`PPOConfig`)
    reward_scale_krw: float = 1_000_000.0    #: 보상 단위 (`PPOConfig`)
    training: bool = True                    #: `PPORuntime(training=)`
    stop_s: float | None = None              #: 디버그 절단 시각 (`DebugStop`)
    learning_window_s: tuple[float, float] | None = None    #: 학습창 [시작, 끝)

    def __post_init__(self):
        if min(self.n_blocks, self.cmax, self.amax, self.input_dim, self.rollout_intervals) < 1:
            raise ValueError("칸 크기와 구간 수는 1 이상이어야 한다")
        if self.reward_scale_krw <= 0:
            raise ValueError("reward_scale_krw 는 양수여야 한다")
        if self.stop_s is not None and not (math.isfinite(self.stop_s) and self.stop_s > 0):
            raise ValueError("stop_s 는 유한한 양수여야 한다 (runtime.py:62-63)")
        if self.learning_window_s is not None:
            start, end = (float(v) for v in self.learning_window_s)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("학습창은 유한·비음수·증가여야 한다 (runtime.py:74-78)")

    @classmethod
    def from_ppo_config(cls, config, *, n_blocks=21, cmax=64, amax=29, input_dim=37,
                        training=True, stop_s=None, learning_window_s=None) -> "RuntimeConfig":
        """v5 `PPOConfig` + `PPORuntime.__init__` 인자 → 이 설정. 이름을 손으로 옮기지 않는다."""
        window = (None if learning_window_s is None
                  else (float(learning_window_s[0]), float(learning_window_s[1])))
        return cls(n_blocks=int(n_blocks), cmax=int(cmax), amax=int(amax), input_dim=int(input_dim),
                   rollout_intervals=int(config.rollout_intervals),
                   reward_scale_krw=float(config.reward_scale_krw),
                   training=bool(training),
                   stop_s=(None if stop_s is None else float(stop_s)),
                   learning_window_s=window)


# ───────────────────────────────────────────────── 결정 묶음 (v5 `self.pending`)
class PendingTape(NamedTuple):
    """한 구간 안에 쌓인 결정 — `(B, cmax)` 고정 칸. v5 `pending[block].append(Choice(...))`.

    빈 칸은 `role = -1` · `action = -1` · `time_s = +inf` · `mask` 전부 거짓 · `log_prob = 0`
    (갱신 담당 `ppo_buffer.IntervalBatch` 의 패딩 규약과 같은 값). **`rows` 만은 안 지운다** —
    (B,cmax,amax,37) 을 경계마다 0 으로 다시 쓰면 11MB/경계다. 그래서 칸 `n[b]` 이상의 `rows` 에는
    지난 구간 값이 남아 있을 수 있고, 대신 그 칸의 `mask` 가 전부 거짓이라 **읽어도 후보가 없다**.
    """

    n: jnp.ndarray          # (B,) int32    블록마다 쌓인 결정 수 = len(pending[b])
    role: jnp.ndarray       # (B,cmax) int32  역할 코드 (-1 = 빈 칸)
    time_s: jnp.ndarray     # (B,cmax) f64    결정 시각 (+inf = 빈 칸)
    rows: jnp.ndarray       # (B,cmax,amax,37) f64  `encode(rows, role)` 결과
    mask: jnp.ndarray       # (B,cmax,amax) bool    행동 마스크 (빈 후보 칸은 False)
    n_actions: jnp.ndarray  # (B,cmax) int32  실제 후보 수 (= len(rows) — 패딩 전)
    action: jnp.ndarray     # (B,cmax) int32  고른 번호
    log_prob: jnp.ndarray   # (B,cmax) f64    고른 행동의 로그확률
    overflow: jnp.ndarray   # (B,) int32      칸이 없어 못 담은 결정 수 (0 이어야 정상)


def empty_pending(cfg: RuntimeConfig) -> PendingTape:
    b, c, a, d = cfg.n_blocks, cfg.cmax, cfg.amax, cfg.input_dim
    return PendingTape(
        n=jnp.zeros((b,), jnp.int32),
        role=jnp.full((b, c), -1, jnp.int32),
        time_s=jnp.full((b, c), EMPTY_TIME, F),
        rows=jnp.zeros((b, c, a, d), F),
        mask=jnp.zeros((b, c, a), jnp.bool_),
        n_actions=jnp.zeros((b, c), jnp.int32),
        action=jnp.full((b, c), -1, jnp.int32),
        log_prob=jnp.zeros((b, c), F),
        overflow=jnp.zeros((b,), jnp.int32),
    )


def _clear_pending(p: PendingTape) -> PendingTape:
    """`self.pending = [[] for _ in self.bids]` (runtime.py:205) — 색인 칸만 되돌린다.

    `overflow` 는 **누적**이다: 한 구간에서 칸이 넘쳤다는 사실을 경계가 지우면 조용한 유실이 된다.
    """
    return p._replace(n=jnp.zeros_like(p.n), role=jnp.full_like(p.role, -1),
                      time_s=jnp.full_like(p.time_s, EMPTY_TIME),
                      action=jnp.full_like(p.action, -1),
                      n_actions=jnp.zeros_like(p.n_actions),
                      mask=jnp.zeros_like(p.mask),            # 빈 칸은 후보가 없다 (거짓)
                      log_prob=jnp.zeros_like(p.log_prob))


# ───────────────────────────────────────────────── 루프 상태 (v5 인스턴스 칸)
class RuntimeState(NamedTuple):
    """`PPORuntime` 의 가변 칸 전부를 배열로. 순수 함수가 이 값을 받아 새 값을 돌려준다."""

    started: jnp.ndarray            # () bool   첫 경계를 지났나 (`time_s is not None`)
    time_s: jnp.ndarray             # () f64    마지막 경계 시각 (+inf = 아직)
    initial_cost: jnp.ndarray       # () f64    첫 경계의 Φ (`initial_cost`)
    cost_krw: jnp.ndarray           # () f64    마지막 경계의 Φ
    total_reward: jnp.ndarray       # () f64
    intervals: jnp.ndarray          # () int32
    learning_reward: jnp.ndarray    # () f64
    learning_intervals: jnp.ndarray  # () int32
    states: jnp.ndarray             # (B,37) f64  마지막 경계에서 찍은 상태 (구간 시작 상태)
    values: jnp.ndarray             # (B,) f64    그 상태의 가치
    pending: PendingTape            #             구간 안 결정 묶음
    n_buffered: jnp.ndarray         # () int32    `len(self.buffer)`
    updates: jnp.ndarray            # () int32    갱신 횟수 (`len(self.updates)`)
    truncated: jnp.ndarray          # () bool
    role_counts: jnp.ndarray        # (4,) int32  `self.role_counts`
    crane_actions: jnp.ndarray      # (4,) int32  `self.crane_actions` (KINDS 순)
    flags: jnp.ndarray              # () int32    위반 비트합


def new_state(cfg: RuntimeConfig) -> RuntimeState:
    """`PPORuntime.__init__` 직후 = 첫 경계 전 상태."""
    b = cfg.n_blocks
    z, zi = jnp.zeros((), F), jnp.zeros((), jnp.int32)
    return RuntimeState(
        started=jnp.asarray(False), time_s=jnp.asarray(EMPTY_TIME, F),
        initial_cost=z, cost_krw=z, total_reward=z, intervals=zi,
        learning_reward=z, learning_intervals=zi,
        states=jnp.zeros((b, cfg.input_dim), F), values=jnp.zeros((b,), F),
        pending=empty_pending(cfg), n_buffered=zi, updates=zi,
        truncated=jnp.asarray(False),
        role_counts=jnp.zeros((N_ROLES,), jnp.int32),
        crane_actions=jnp.zeros((N_KINDS,), jnp.int32),
        flags=zi)


# ───────────────────────────────────────────────── 산출 묶음
class IntervalRow(NamedTuple):
    """v5 `buffer.Interval` 한 줄의 배열판. `valid=False` 면 이 경계는 구간을 안 냈다.

    GAE·미니배치는 갱신 담당(`update`/`buffer`)이 이 줄을 `R` 개 쌓아서 쓴다 —
    여기서는 **모으기만** 한다.
    """

    start_s: jnp.ndarray     # () f64
    end_s: jnp.ndarray       # () f64
    states: jnp.ndarray      # (B,37) f64   구간 **시작** 상태
    values: jnp.ndarray      # (B,) f64
    reward: jnp.ndarray      # () f64       ★팀 보상 하나 — 21블록이 같은 값을 받는다
    terminated: jnp.ndarray  # () bool
    choices: PendingTape     #              그 구간에 쌓인 결정
    valid: jnp.ndarray       # () bool      buffer 에 실렸나 (`collecting_at(start)`)


class BoundaryOut(NamedTuple):
    """`boundary` 한 번의 결과 — 호스트가 읽어 기록·갱신을 부른다."""

    advanced: jnp.ndarray      # () bool   구간이 닫혔나 (`t > time_s + 1e-6`)
    ignored: jnp.ndarray       # () bool   반복 경계 → **아무것도 안 했다** (runtime.py:193-194)
    cost: jnp.ndarray          # () f64    이 경계의 Φ
    delta: jnp.ndarray         # () f64    Φ 증가분 (구간이 안 닫히면 0)
    reward: jnp.ndarray        # () f64    −delta / reward_scale_krw
    interval: IntervalRow      #           구간 기록 (valid 로 실렸는지 본다)
    do_update: jnp.ndarray     # () bool   `_update` 가 실제로 갱신을 하나
    bootstrap: jnp.ndarray     # (B,) f64  갱신에 넘길 부트스트랩 (terminated 면 0)
    should_stop: jnp.ndarray   # () bool   `stop_s` 에 닿았나
    stop: jnp.ndarray          # () bool   `DebugStop` 을 던질 자리 (should_stop & ~final)
    flags: jnp.ndarray         # () int32  이번 경계에서 **새로** 켜진 비트


class SelectOut(NamedTuple):
    """`select_record` 한 번의 결과."""

    recorded: jnp.ndarray   # () bool   학습창 안이라 pending 에 담겼나
    slot: jnp.ndarray       # () int32  담긴 칸 번호 (-1 = 안 담김)
    flags: jnp.ndarray      # () int32  이번 결정에서 새로 켜진 비트


class CostOut(NamedTuple):
    """`read_cost` 의 결과 — v5 는 `self.cost_breakdown` 에 부수효과로 남긴다."""

    total: jnp.ndarray   # () f64
    phi: PhiArrays       #        13항 내역 (`as_dict()` 가 v5 `PhiBreakdown.as_dict()`)
    flags: jnp.ndarray   # () int32


# ───────────────────────────────────────────────── 학습창 (runtime.py:96-98)
def collecting_at(cfg: RuntimeConfig, t) -> jnp.ndarray:
    """`PPORuntime.collecting_at` — 학습 중이고 `start ≤ t < end` 인가.

    창이 없으면 학습 중이면 항상 참. `training=False` 면 항상 거짓 (그래서 고정 운영은
    구간을 하나도 모으지 않는다).
    """
    t = jnp.asarray(t, F)
    if not cfg.training:
        return jnp.zeros(jnp.shape(t), jnp.bool_)
    if cfg.learning_window_s is None:
        return jnp.ones(jnp.shape(t), jnp.bool_)
    start, end = cfg.learning_window_s
    return (t >= start) & (t < end)


def _window_edge_inside(cfg: RuntimeConfig, t0, t1) -> jnp.ndarray:
    """`any(time_s < edge < t for edge in learning_window_s)` (runtime.py:173-174).

    학습창의 시작·끝이 구간 **안쪽**에 들어오면 그 구간의 보상이 창 안/밖에 잘못 붙는다 —
    v5 는 여기서 RuntimeError 를 던진다. 부등호는 **양쪽 다 강부등호**다.
    """
    if cfg.learning_window_s is None:
        return jnp.zeros((), jnp.bool_)
    t0, t1 = jnp.asarray(t0, F), jnp.asarray(t1, F)
    hit = jnp.zeros((), jnp.bool_)
    for edge in cfg.learning_window_s:
        hit = hit | ((t0 < edge) & (edge < t1))
    return hit


# ───────────────────────────────────────────────── ① 비용 읽기 (runtime.py:121-127)
def read_cost(orders: OrderArrays, t, *, vessel_gt=None, vessel_idle_s=None, vessel_mask=None,
              yc_extra_move_s=0.0, rehandles=0, truck_mask=None) -> CostOut:
    """`PPORuntime.read_cost` — 조각 5 `phi.terminal_cost_krw` 를 그대로 부른다 (재구현 금지).

    `orders` 는 **v5 기록 사전의 삽입 순서**여야 원화 합이 비트 일치한다 (`phi.orders_from_records`
    또는 터미널 배열 세계의 오더 축). `vessel_*` 는 `month_vessel_idle` 이 이미 배 단위로 줄인 표다
    (스트림 → 배 축소가 필요하면 `phi.vessel_idle_by_ship`).

    비유한 Φ 는 v5 가 `FloatingPointError` 를 던지는 자리 → `F_COST_NONFINITE` 비트.
    """
    phi = terminal_cost_krw(orders, t, vessel_gt=vessel_gt, vessel_idle_s=vessel_idle_s,
                            vessel_mask=vessel_mask, yc_extra_move_s=yc_extra_move_s,
                            rehandles=rehandles, truck_mask=truck_mask)
    flags = jnp.where(jnp.isfinite(phi.total), 0, F_COST_NONFINITE).astype(jnp.int32)
    return CostOut(total=phi.total, phi=phi, flags=flags)


# ───────────────────────────────────────────────── ② 결정 기록 (runtime.py:129-146)
def select_record(st: RuntimeState, cfg: RuntimeConfig, *, role, block, t, rows, mask,
                  action, log_prob, n_actions) -> tuple[RuntimeState, SelectOut]:
    """`PPORuntime.select` 의 **기록 부분**. 행동을 고르는 일은 망(`v5net`)이 이미 했다.

    v5 순서를 그대로 지킨다:
      1. 시각 검사 (비유한·음수 / 초기 경계 전 / 구간보다 앞)      → 비트
      2. `collecting_at(self.time_s)` 이면 그 블록 칸에 Choice 하나  ★창 판정은 **구간 시작 시각**
      3. `role_counts[role] += 1` — 담겼든 안 담겼든 **항상** 센다 (runtime.py:145)

    `rows` (amax,37) · `mask` (amax,) 는 **패딩 뒤** 폭이고, `n_actions` 가 v5 `len(rows)` 다 —
    마스크가 거짓인 후보도 후보 수에는 들어가므로 참 개수로 대신할 수 없다.
    """
    t = jnp.asarray(t, F)
    role = jnp.asarray(role, jnp.int32)
    block = jnp.asarray(block, jnp.int32)
    rows = jnp.asarray(rows, F)
    mask = jnp.asarray(mask, jnp.bool_)
    if rows.shape != (cfg.amax, cfg.input_dim):
        raise ValueError(f"후보 행렬 {rows.shape} != ({cfg.amax}, {cfg.input_dim}) — 칸을 넓혀라")
    if mask.shape != (cfg.amax,):
        raise ValueError(f"마스크 {mask.shape} != ({cfg.amax},)")

    new = jnp.zeros((), jnp.int32)
    new |= jnp.where(~jnp.isfinite(t) | (t < 0), F_DECISION_TIME_BAD, 0)
    new |= jnp.where(st.started, 0, F_DECISION_NO_BOUNDARY)
    new |= jnp.where(st.started & (t < st.time_s - EPS_TIME), F_DECISION_EARLY, 0)
    new |= jnp.where(mask.any(), 0, F_MASK_EMPTY)
    n_actions = jnp.asarray(n_actions, jnp.int32)
    new |= jnp.where(n_actions > cfg.amax, F_ACTION_OVERFLOW, 0)

    # ★v5 가 **던지는** 네 자리에서는 기록도 계수도 하지 않는다 (예외가 145행보다 앞이다)
    raised = (new & (F_DECISION_TIME_BAD | F_DECISION_NO_BOUNDARY
                     | F_DECISION_EARLY | F_MASK_EMPTY)) != 0
    collecting = collecting_at(cfg, st.time_s) & st.started & ~raised
    p = st.pending
    slot = p.n[block]
    room = slot < cfg.cmax
    write = collecting & room
    new |= jnp.where(collecting & ~room, F_PENDING_OVERFLOW, 0)

    idx = (block, jnp.minimum(slot, cfg.cmax - 1))
    keep = lambda old, val: jnp.where(write, old.at[idx].set(val), old)
    p2 = p._replace(
        n=jnp.where(write, p.n.at[block].add(1), p.n),
        role=keep(p.role, role),
        time_s=keep(p.time_s, t),
        rows=jnp.where(write, p.rows.at[idx].set(rows), p.rows),
        mask=jnp.where(write, p.mask.at[idx].set(mask), p.mask),
        n_actions=keep(p.n_actions, n_actions),
        action=keep(p.action, jnp.asarray(action, jnp.int32)),
        log_prob=keep(p.log_prob, jnp.asarray(log_prob, F)),
        overflow=jnp.where(collecting & ~room, p.overflow.at[block].add(1), p.overflow),
    )
    st2 = st._replace(pending=p2,
                      role_counts=st.role_counts.at[role].add(jnp.where(raised, 0, 1)),
                      flags=st.flags | new)
    return st2, SelectOut(recorded=write, slot=jnp.where(write, slot, -1), flags=new)


def count_crane_action(st: RuntimeState, kind) -> RuntimeState:
    """`rt.crane_actions[name] += 1` (`ppo/crane.py:82`) — 보고용 계수기.

    `kind` 는 `CRANE_KINDS` 순서의 정수 (`state.PK_SERVE` …). 이 계수기는 학습에 안 쓰이고
    `report()` 의 `crane_actions` 로만 나간다 (v5 는 퇴화 검출에 이 분포를 본다).
    """
    return st._replace(crane_actions=st.crane_actions.at[jnp.asarray(kind, jnp.int32)].add(1))


# ───────────────────────────────────────────────── ③ 경계 (runtime.py:158-210)
def boundary(st: RuntimeState, cfg: RuntimeConfig, t, cost, states, values,
             *, terminated=False, final=False) -> tuple[RuntimeState, BoundaryOut]:
    """`PPORuntime.boundary` — 60초 동기화 경계 한 번.

    호출부가 먼저 준비하는 것 (v5 는 `boundary` 안에서 스스로 불렀다):
      `states` (B,37) = `v5feat.state_rows(블록 요약)` · `values` (B,) = `v5net` 상태가치 ·
      `cost` = `read_cost(...).total`

    v5 분기 구조를 그대로 옮긴다 (runtime.py:170-194):
      · **첫 경계** (`initial_cost is None`) → `initial_cost = cost`, 구간 없음
      · **전진한 경계** (`t > time_s + 1e-6`) → 창 경계 검사 → delta·보상 → 누적 → 학습창이면 수집
      · **반복 경계** (같은 t, `final` 아님) → `return` — 비용·결정·상태를 **건드리지 않는다**
        (두 번 물었다고 비용을 두 번 계상하거나 결정을 지우면 안 된다)

    `final=True` (= `finish`) 는 반복 경계에서도 마지막 정리를 한다.
    """
    t = jnp.asarray(t, F)
    cost = jnp.asarray(cost, F)
    states = jnp.asarray(states, F)
    values = jnp.asarray(values, F)
    terminated = jnp.asarray(terminated, jnp.bool_)
    final = jnp.asarray(final, jnp.bool_)
    b = cfg.n_blocks
    if states.shape != (b, cfg.input_dim):
        raise ValueError(f"상태 {states.shape} != ({b}, {cfg.input_dim})")
    if values.shape != (b,):
        raise ValueError(f"가치 {values.shape} != ({b},)")

    # ── 1. 시각·비용 검사 (v5 가 예외를 던지는 세 자리)
    new = jnp.zeros((), jnp.int32)
    new |= jnp.where(~jnp.isfinite(t) | (t < 0), F_REVIEW_TIME_BAD, 0)
    new |= jnp.where(st.started & (t < st.time_s - EPS_TIME), F_CLOCK_BACKWARD, 0)
    new |= jnp.where(jnp.isfinite(cost), 0, F_COST_NONFINITE)

    # ── 2. 세 갈래 판정
    first = ~st.started                                   # initial_cost is None
    advanced = st.started & (t > st.time_s + EPS_TIME)     # elif t > self.time_s + 1e-6
    ignored = st.started & ~advanced & ~final              # elif not final: return
    apply = ~ignored

    # ── 3. 구간 보상 (전진한 경계만)
    new |= jnp.where(advanced & _window_edge_inside(cfg, st.time_s, t), F_WINDOW_EDGE_MISSED, 0)
    delta_raw = cost - st.cost_krw
    delta = jnp.where(advanced, delta_raw, 0.0)
    new |= jnp.where(advanced & (delta_raw < -COST_FELL_EPS), F_COST_FELL, 0)
    # ★나눗셈은 div_const — 역수 곱으로 접히면 마지막 비트가 갈린다 (exact.py)
    reward = jnp.where(advanced, div_const(-delta, cfg.reward_scale_krw, dtype=F), 0.0)

    collecting_start = collecting_at(cfg, st.time_s) & st.started
    collected = advanced & collecting_start

    interval = IntervalRow(start_s=st.time_s, end_s=t, states=st.states, values=st.values,
                           reward=reward, terminated=terminated, choices=st.pending,
                           valid=collected)

    total_reward = jnp.where(advanced, st.total_reward + reward, st.total_reward)
    intervals = jnp.where(advanced, st.intervals + 1, st.intervals)
    learning_reward = jnp.where(collected, st.learning_reward + reward, st.learning_reward)
    learning_intervals = jnp.where(collected, st.learning_intervals + 1, st.learning_intervals)
    n_buffered = jnp.where(collected, st.n_buffered + 1, st.n_buffered)

    initial_cost = jnp.where(first, cost, st.initial_cost)

    # ── 4. 시계·비용 전진 (runtime.py:195) — 반복 경계는 여기 안 온다
    time_s = jnp.where(apply, t, st.time_s)
    cost_krw = jnp.where(apply, cost, st.cost_krw)

    # ── 5. 갱신 방아쇠 (runtime.py:197-200)
    should_stop = (jnp.zeros((), jnp.bool_) if cfg.stop_s is None
                   else jnp.asarray(t >= cfg.stop_s - EPS_TIME))
    should_stop = should_stop & apply          # 반복 경계는 stop 판정 자체를 안 한다
    trigger = ((n_buffered >= cfg.rollout_intervals) | final | should_stop
               | ~collecting_at(cfg, t))
    # `_update` 는 학습 중이 아니거나 buffer 가 비면 아무것도 안 한다 (runtime.py:149-150)
    do_update = apply & trigger & (n_buffered > 0) & jnp.asarray(bool(cfg.training))
    bootstrap = jnp.where(terminated, jnp.zeros_like(values), values)

    n_buffered = jnp.where(do_update, 0, n_buffered)
    updates = jnp.where(do_update, st.updates + 1, st.updates)

    # ── 6. 상태·가치 재수집과 결정 칸 비우기 (runtime.py:202-205)
    #    ★v5 는 `bootstrap` 을 **갱신 전** 망으로, `self.values` 를 **갱신 뒤** 망으로 잰다.
    #      배열판은 망 파라미터를 들고 있지 않으므로 여기서는 받은 `values`(= 갱신 전 = bootstrap)
    #      를 넣어 두고, 갱신을 실제로 돌린 호출부가 `refresh_values(st, 갱신뒤가치)` 로 덮어쓴다.
    #      순서: ① 상태·가치 계산 → ② boundary → ③ do_update 면 갱신 → ④ refresh_values
    st2 = st._replace(
        started=st.started | apply,
        time_s=time_s, initial_cost=initial_cost, cost_krw=cost_krw,
        total_reward=total_reward, intervals=intervals,
        learning_reward=learning_reward, learning_intervals=learning_intervals,
        states=jnp.where(apply, states, st.states),
        values=jnp.where(apply, values, st.values),
        pending=_apply_clear(st.pending, apply),
        n_buffered=n_buffered, updates=updates,
        truncated=st.truncated | (should_stop & ~final),
        flags=st.flags | new,
    )
    out = BoundaryOut(advanced=advanced, ignored=ignored, cost=cost, delta=delta, reward=reward,
                      interval=interval, do_update=do_update, bootstrap=bootstrap,
                      should_stop=should_stop, stop=should_stop & ~final, flags=new)
    return st2, out


def _apply_clear(p: PendingTape, apply) -> PendingTape:
    cleared = _clear_pending(p)
    return PendingTape(*[jnp.where(apply, c, o) for c, o in zip(cleared, p)])


def refresh_values(st: RuntimeState, values) -> RuntimeState:
    """`runtime.py:201-204` — 갱신 **뒤에** 상태가치를 다시 찍는다.

    "Values MUST be recollected after an update, for the next on-policy batch." 갱신이 없었으면
    `boundary` 가 넣어 둔 값과 같으므로 불러도 안 불러도 같다. 반복 경계(`ignored`)에서는
    v5 가 일찍 돌아가므로 **부르지 않는다**.
    """
    values = jnp.asarray(values, F)
    if values.shape != st.values.shape:
        raise ValueError(f"가치 {values.shape} != {st.values.shape}")
    return st._replace(values=values)


def finish(st: RuntimeState, cfg: RuntimeConfig, t, cost, states, values,
           *, terminated=False) -> tuple[RuntimeState, BoundaryOut]:
    """`PPORuntime.finish` (runtime.py:212-214) — 마지막 경계(`final=True`) 뒤 `truncated` 확정.

    `terminated=False` 는 "수요 일정이 끝나 시간이 다 된 것" 이라 **잘린 것으로 본다**
    (`month_run.py` 가 그렇게 부른다: 유한한 수요 일정은 시간 제한이다).
    """
    st2, out = boundary(st, cfg, t, cost, states, values, terminated=terminated, final=True)
    return st2._replace(truncated=~jnp.asarray(terminated, jnp.bool_)), out


# ───────────────────────────────────────────────── 보고 (runtime.py:216-233)
def report(st: RuntimeState, cfg: RuntimeConfig, *, cost_breakdown=None, bridge=None,
           updates=None, workload=None) -> dict:
    """v5 `PPORuntime.report()` 와 **같은 키**의 파이썬 사전. 호스트 전용 (배열 아님).

    `bridge` 는 시장 다리의 계수기 네 개 `{traded_edges, txn_failed, n_space, n_time}` —
    시장 담당의 배열판이 내는 값을 그대로 넣는다. `updates` 는 갱신 담당이 모은 보고 목록.
    """
    window = cfg.learning_window_s
    return {
        "generation": "v6-array", "algorithm": "shared-block-PPO",
        "time_s": (None if not bool(st.started) else float(st.time_s)),
        "blocks": int(cfg.n_blocks),
        "intervals": int(st.intervals),
        "roles": {r: int(st.role_counts[i]) for i, r in enumerate(ROLES)
                  if int(st.role_counts[i])},
        "crane_actions": {k: int(st.crane_actions[i]) for i, k in enumerate(CRANE_KINDS)
                          if int(st.crane_actions[i])},
        "cost_krw": float(st.cost_krw),
        "initial_cost_krw": (None if not bool(st.started) else float(st.initial_cost)),
        "team_reward": float(st.total_reward),
        "learning_window_s": (None if window is None else (float(window[0]), float(window[1]))),
        "learning_intervals": int(st.learning_intervals),
        "learning_reward": float(st.learning_reward),
        "shaping_reward": 0.0, "learning_shaping_reward": 0.0,
        "learning_training_reward": float(st.learning_reward),
        "workload": workload,
        "truncated": bool(st.truncated),
        "updates": list(updates or []),
        "traded_edges": (bridge or {}).get("traded_edges"),
        "txn_failed": (bridge or {}).get("txn_failed"),
        "n_space": (bridge or {}).get("n_space"), "n_time": (bridge or {}).get("n_time"),
        "cost_breakdown": cost_breakdown,
        "flags": flag_names(int(st.flags)),
    }


# ───────────────────────────────────────────────── 호스트 다리 (v5 자료 → 배열)
def cost_inputs_from_v5(records, vessel_idle: dict, *, n_max: int | None = None) -> dict:
    """v5 `bridge.records` + `month_vessel_idle(...)` → `read_cost` 인자 묶음.

    · 기록 사전의 **삽입 순서**가 배열 순서다 (`phi.orders_from_records` 규약 — 원화 합의 비트)
    · 본선 표도 **사전 순서** 그대로 `(V,)` 로 (v5 `terminal_cost_krw` 가 그 순서로 더한다)
    비어 있는 본선 표는 칸 하나에 마스크 거짓으로 둔다 (`phi._fill` 규약).
    """
    from .phi import orders_from_records
    orders = orders_from_records(records, n_max=n_max)
    if vessel_idle:
        gt = jnp.asarray([float(g) for g, _ in vessel_idle.values()], F)
        idle = jnp.asarray([float(i) for _, i in vessel_idle.values()], F)
        mask = jnp.ones((len(vessel_idle),), jnp.bool_)
    else:
        gt = jnp.zeros((1,), F)
        idle = jnp.zeros((1,), F)
        mask = jnp.zeros((1,), jnp.bool_)
    return {"orders": orders, "vessel_gt": gt, "vessel_idle_s": idle, "vessel_mask": mask}


# ───────────────────────────────────────────────── 격자 사진 (_CounterTape · _MonthTape)
class CounterTape(NamedTuple):
    """엔진 **누적 계수기의 사진첩** — `stage/episode._CounterTape` · `stage/month_run._MonthTape`.

    왜 필요한가: Φ 의 항2(YC 빈 주행)·항3(재조작)·항4(본선 유휴)는 기록이 아니라 **누적 계수기**라
    지나간 뒤에는 "그때 얼마였나" 를 되짚을 수 없다. 트럭 대기(항1)는 기록의 시각을 `end_s` 로
    검열해 자를 수 있지만 계수기는 못 자른다 → **지나갈 때 찍어 둔다.** 하루치 Φ 는 날 경계
    두 사진의 **차분**이다 (`_MonthTape.diff`).

    칸: `time_s` 는 `round(t,6)` 오름차순(같은 값이면 **나중에 찍은 것이 뒤**여야 한다 —
    v5 는 사전이라 나중 값이 덮어쓴다). 본선 표는 사진마다 그 시점 사전 순서로 담고,
    `ship_id` 로 사진 사이를 맞춘다.
    """

    time_s: jnp.ndarray          # (S,) f64
    yc_extra_move_s: jnp.ndarray  # (S,) f64
    rehandles: jnp.ndarray       # (S,) int32
    vessel_gt: jnp.ndarray       # (S,V) f64
    vessel_idle_s: jnp.ndarray   # (S,V) f64
    vessel_id: jnp.ndarray       # (S,V) int32  전역 배 번호 (-1 = 빈 칸)
    #: 전역 배 수 — `tape_diff` 가 **모양**으로 쓰므로 정적 파이썬 int 여야 한다.
    #: `jit` 할 때는 `tape` 를 닫아 넣거나 `static_argnums` 로 넘긴다 (배열로 만들면 모양이 추적된다).
    n_ships: int


def counter_tape_from_v5(tape, *, n_ships: int | None = None) -> tuple[CounterTape, dict]:
    """v5 `_CounterTape`/`_MonthTape` → 배열판. 돌려주는 두 번째 값은 `배 이름 → 번호` 표다."""
    import numpy as np
    keys = sorted(tape.at)                              # round(t,6) 오름차순
    names: dict[str, int] = {}
    for k in keys:
        for name in tape.at[k][0]:
            names.setdefault(name, len(names))
    total = len(names) if n_ships is None else int(n_ships)
    if total < len(names):
        raise ValueError(f"배 칸 {total} < 실제 {len(names)} — 조용히 자르지 않는다")
    v_max = max([1] + [len(tape.at[k][0]) for k in keys])
    s = max(1, len(keys))
    time_s = np.full(s, np.inf)          # 사진이 하나도 없으면 어떤 t 로도 안 잡힌다
    yc = np.zeros(s)
    rh = np.zeros(s, np.int32)
    gt = np.zeros((s, v_max))
    idle = np.zeros((s, v_max))
    vid = np.full((s, v_max), -1, np.int32)
    for i, k in enumerate(keys):
        vessels, yc_i, rh_i = tape.at[k]
        time_s[i], yc[i], rh[i] = float(k), float(yc_i), int(rh_i)
        for j, (name, (g, idl)) in enumerate(vessels.items()):
            gt[i, j], idle[i, j], vid[i, j] = float(g), float(idl), names[name]
    return CounterTape(time_s=jnp.asarray(time_s, F), yc_extra_move_s=jnp.asarray(yc, F),
                       rehandles=jnp.asarray(rh), vessel_gt=jnp.asarray(gt, F),
                       vessel_idle_s=jnp.asarray(idle, F), vessel_id=jnp.asarray(vid),
                       n_ships=total), names


def _tape_index(tape: CounterTape, t) -> jnp.ndarray:
    """`t` 이하의 **가장 늦은** 사진 번호 (없으면 -1) — `_MonthTape.read` (month_run.py:154-158)."""
    key = jnp.asarray(t, F)
    ok = tape.time_s <= key
    n = tape.time_s.shape[0]
    idx = jnp.max(jnp.where(ok, jnp.arange(n, dtype=jnp.int32), -1))
    return idx


def tape_read(tape: CounterTape, t) -> dict:
    """사진 하나를 `read_cost` 인자 꼴로. 사진이 없으면 v5 처럼 `({}, 0.0, 0)` (빈 표·0·0)."""
    i = _tape_index(tape, t)
    have = i >= 0
    j = jnp.maximum(i, 0)
    return {
        "vessel_gt": jnp.where(have, tape.vessel_gt[j], 0.0),
        "vessel_idle_s": jnp.where(have, tape.vessel_idle_s[j], 0.0),
        "vessel_mask": have & (tape.vessel_id[j] >= 0),
        "yc_extra_move_s": jnp.where(have, tape.yc_extra_move_s[j], 0.0),
        "rehandles": jnp.where(have, tape.rehandles[j], 0),
    }


def tape_diff(tape: CounterTape, t0, t1) -> dict:
    """`t0 → t1` 에 **늘어난 만큼** — `_MonthTape.diff` (month_run.py:160-169).

    본선은 **t1 사진의 순서**로 담는다 (v5 가 `v1.items()` 를 훑는다). t0 에 없던 배는 0 에서
    시작한 것으로 본다. 세 항 모두 `max(0, 차)` — 계수기가 줄어드는 일은 없어야 하지만
    v5 가 방어를 두었으므로 그대로 둔다.
    """
    a, b = tape_read(tape, t0), tape_read(tape, t1)
    i0, i1 = _tape_index(tape, t0), _tape_index(tape, t1)
    j0, j1 = jnp.maximum(i0, 0), jnp.maximum(i1, 0)
    # t0 사진의 유휴를 **배 번호 축**으로 펼쳐 t1 순서로 다시 모은다
    lut = jnp.zeros((tape.n_ships + 1,), F)
    id0 = jnp.where((i0 >= 0) & (tape.vessel_id[j0] >= 0), tape.vessel_id[j0], tape.n_ships)
    lut = lut.at[id0].set(tape.vessel_idle_s[j0])
    id1 = jnp.where(tape.vessel_id[j1] >= 0, tape.vessel_id[j1], tape.n_ships)
    before = jnp.where(i1 >= 0, lut[id1], 0.0)
    return {
        "vessel_gt": b["vessel_gt"],
        "vessel_idle_s": jnp.maximum(0.0, b["vessel_idle_s"] - before),
        "vessel_mask": b["vessel_mask"],
        "yc_extra_move_s": jnp.maximum(0.0, b["yc_extra_move_s"] - a["yc_extra_move_s"]),
        "rehandles": jnp.maximum(0, b["rehandles"] - a["rehandles"]),
    }
