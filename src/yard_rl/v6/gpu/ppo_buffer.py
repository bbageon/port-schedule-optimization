"""학습 버퍼와 GAE(일반화 우위 추정) 배열판 — v5 `ppo/buffer.py:11-64` ([[YR-327]] 조각 8).

■ 무엇을 하나 — 두 가지다

  ① **들쭉날쭉한 결정 기록을 고정 칸 배열로 옮긴다** (`pack_intervals`)
     v5 는 60초 구간 하나를 `Interval` 객체로 담는데, 그 안의 `choices` 는 **길이가 매번 다른
     이중 리스트**다 (블록마다 결정 수가 다르고, 결정마다 후보 수가 다르다). 가속기는 길이가
     변하는 것을 컴파일할 수 없으므로 **가장 큰 칸을 잡고 남는 칸은 마스크로 덮는다**:

         구간 R × 블록 B × 결정 Cmax × 후보 Amax × 특징 37

     R    = 한 갱신 묶음의 구간 수 (v5 기본 60 — `PPOConfig.rollout_intervals`)
     B    = 블록 수 (터미널 21)
     Cmax = 한 구간·한 블록에서 나올 수 있는 **최대 결정 수** (판매자·구매자·크레인 합)
     Amax = 한 결정의 **최대 후보 수** (v5 는 판매자 1+20+8=29 · 크레인 k_max+1=13)
     37   = 정책망 입력 칸 (`ppo/model.py:10` — 원특징 32 + 역할 4 + BUY 표시 1)

     빈 칸 규약은 `gpu/` 관례를 따른다 — 정수는 `-1`, 시각은 `+inf`, 실수·마스크는 `0`·False.
     실제 개수는 `n_choices (R,B)` · `n_cands (R,B,Cmax)` · `n_intervals ()` 가 들고 있다.

     ★**칸이 모자라면 조용히 버리지 않는다** — `IntervalOverflow` 가 몇 개를 못 담았는지
     세어서 돌려준다 (`gpu/events.py`·`gpu/state.py` 의 넘침 표시와 같은 방식). v5 에는
     넘침이란 개념이 없으므로 **시험은 넘침 0 을 단언한다**; 넘쳤다는 것은 Cmax·Amax 를
     잘못 잡았다는 뜻이고 그대로 두면 답이 달라진다.

  ② **GAE 를 역방향 `lax.scan` 으로** (`gae`)
     v5 는 구간을 뒤에서 앞으로 훑는 파이썬 `for` 루프다 (`buffer.py:54-62`). 같은 식·같은
     결합 순서를 `lax.scan(..., reverse=True)` 로 옮긴다. 캐리는 두 개 — 우위 흔적 `trace`
     와 "다음 구간의 상태가치" `nxt` 다.

■ ★v5 와 같은 답을 내기 위해 반드시 지킨 것 넷

  (1) **연산 결합 순서** — 파이썬 연산자 우선순위 그대로 괄호를 박았다.
        delta = row.reward + discount * nxt - row.values
              = (reward + (discount * nxt)) - values          ← `*` 가 먼저, `+`·`-` 는 왼쪽부터
        trace = delta + discount * lam ** dt * trace
              = delta + ((discount * (lam ** dt)) * trace)    ← `**` 가 먼저
      곱은 전부 `exact.mul_exact` 로 실체화한다 — 안 그러면 XLA 가 뒤따르는 덧셈과 묶어
      (FMA, 곱셈-덧셈 융합) **반올림을 한 번만** 해서 마지막 비트가 갈린다.
      `dt = (end − start) / time_unit_s` 는 상수 나눗셈이라 `exact.div_const` 를 쓴다.

  (2) ★★**v5 의 `Interval.values` 는 float32 다** — 그래서 `delta` 도 float32 로 계산된다
      `ppo/runtime.py:204` 이 `self.policy.value(states).numpy()` 를 담는데, torch 망은
      float32 라 `values` 가 **float32 배열**이다. 반면 `buffer.py:37` 은 부트스트랩만
      `np.float64` 로 올린다. 그래서 numpy 승격 규칙(NEP 50)에 따라 이런 일이 벌어진다:

          i = 마지막 구간   nxt = 부트스트랩(f64)  →  discount*nxt 는 **float64** 연산
          i < 마지막 구간   nxt = values(f32)      →  discount*nxt 는 **float32** 연산
                                                     (파이썬 float `discount`·`reward` 가
                                                      먼저 float32 로 깎인다)

      `trace` 는 `np.zeros_like(부트스트랩)` 에서 시작하므로 **끝까지 float64** 이고,
      `adv` 도 float64 다 (`buffer.py:52`). 즉 float32 로 떨어지는 곳은 **`delta` 딱 하나**다.
      이 배열판은 `batch.values` 의 dtype 을 보고 그 갈래를 그대로 재현한다 —
      float32 면 두 갈래(첫 단계 f64 · 이후 f32)를 **둘 다 계산하고 `where` 로 고른다**.
      (시험이 만드는 `Interval`(`test_ppo_units.py:35`·`test_ppo_regressions.py`)은
       `np.zeros(1)` 같은 float64 라 f64 갈래만 탄다 — 두 dtype 모두 시험한다.)

  (3) ⚠️**거듭제곱 `gamma ** dt` 는 무대(백엔드)에 따라 1 ulp 갈린다** — 파이썬은 libm `pow`,
      XLA 는 자기 커널이다. 실측 (2026-09-27 · jax 0.11.2 · dt∈[1e-4, 80] 무작위):

          CPU x64 (Windows)  gamma 0.999 → 0/200,000 불일치 · 0.95 → 0/200,000 · 0.1 → 0/200,000
                             gamma 0.5   → 38/200,000 (최대 상대 2.1e-16)
          GPU x64 (RTX 5090) gamma 0.999·0.95 → **약 4,825/20,004 (24%) 가 1 ulp** (최대 상대 ≤4e-16)

      ★**실제 60초 격자에서는 문제가 없다** — 검토 경계가 60초 격자 위에 있으므로
      `dt = (end − start) / 60` 은 0.25·0.5·0.75·1·1.5·2·3·4 같은 **정수·이진분수**뿐이고,
      그 값에서는 CPU·GPU 둘 다 파이썬과 **정확히 같았다** (시험이 매번 단언한다).
      격자 밖 dt(예: 1/3)를 쓰는 무대에서는 GPU 에서 마지막 비트까지 같다고 주장할 수 없다.
      `tests/v6/test_gpu_ppo_buffer.py::test_14_power_bit_match_rate` 가 비율을 재서 찍는다.

  (4) **패딩 칸이 답을 오염시키지 않는다** — 역방향 scan 은 패딩 칸(`idx ≥ n_intervals`)을
      **먼저** 만난다. 그 칸에서는 캐리를 그대로 통과시키므로 (`where(on, 새값, 옛값)`)
      첫 실제 구간에 닿을 때 `trace = 0` · `nxt = 부트스트랩` 이 되어 v5 의 시작 상태와 같다.
      패딩 칸의 `start_s`·`end_s` 는 0 으로 채워 `dt = 0` 이 되게 한다 (inf 를 넣으면
      `inf − inf = nan` 이 캐리를 타고 번질 수 있다).

■ 예외 대신 **위반 비트** (절대 규칙 1 의 예외 항목)
  v5 `gae` 는 잘못된 입력에 `ValueError` 를 던진다 (`buffer.py:34-50`). jit 안에서는 던질 수
  없으므로 `gae_audit` 가 같은 검사를 **bool 배열**로 돌려주고, 호스트 도우미
  `raise_on_audit` 가 **v5 와 같은 순서·같은 메시지**로 던진다. 설정값(gamma·lam·time_unit_s)
  처럼 데이터가 아닌 것은 파이썬 상수이므로 `gae` 가 즉시 던진다 (fail-loud).

■ 쓰는 법 (통합 단계)
      batch, over = pack_intervals(runtime.buffer, r_max=60, c_max=C, a_max=A)
      assert not over.any, over                      # 칸이 모자라면 답이 달라진다
      raise_on_audit(gae_audit(batch, boot, gamma=g, lam=l, time_unit_s=60.0))
      adv, returns = jax.jit(gae, static_argnames=("gamma", "lam", "time_unit_s"))(
          batch, boot, gamma=g, lam=l, time_unit_s=60.0)
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .exact import div_const, mul_exact
from .v5feat import INPUT_DIM, ROLES

__all__ = ["FEAT_DIM", "ROLES", "IntervalBatch", "IntervalOverflow", "GaeAudit",
           "pack_intervals", "unpack_choice", "active_mask", "valid_mask",
           "micro_actions", "block_samples", "flat_entries",
           "gae", "gae_audit", "raise_on_audit",
           "gae_batch"]

#: 정책망 입력 칸 수 37 (`ppo/model.py:10`) — 정본은 `gpu/v5feat.INPUT_DIM`
FEAT_DIM = INPUT_DIM
F64 = jnp.float64
F32 = jnp.float32
I32 = jnp.int32

#: v5 `math.isclose(..., rel_tol=0, abs_tol=1e-6)` 의 절대 허용 (`buffer.py:45-46`)
CONTIGUITY_TOL = 1e-6


# ═══════════════════════════════════════════════ ① 고정 칸 배열
class IntervalBatch(NamedTuple):
    """v5 `Interval` 목록의 배열판 — 모든 칸이 고정 크기다.

    R = 구간 칸 수 · B = 블록 수 · C = Cmax(구간·블록당 최대 결정) · A = Amax(최대 후보)

    구간 축 (R,)
      start_s        구간 시작 시각 (초) — 패딩 0.0
      end_s          구간 끝 시각 (초) — 패딩 0.0
      reward         팀 보상 하나 (21 사본의 합이 **아니다**, `buffer.py:28`) — 패딩 0.0
      terminated     이 구간에서 에피소드가 끝났나 — 패딩 False
    구간×블록 (R,B)
      values         상태가치 — ★dtype 이 f32 인지 f64 인지가 `gae` 의 산술을 가른다 (머리말 (2))
      states         (R,B,37) 블록 상태 한 줄 (f32 — v5 `encode` 출력 그대로)
      n_choices      실제 결정 수 (0 이면 그 블록은 이 구간에 결정이 없었다)
    구간×블록×결정 (R,B,C)
      action         고른 후보 번호 — 패딩 -1
      log_prob       고를 때의 로그확률 (f64 — v5 는 파이썬 float) — 패딩 0.0
      role           역할 코드 0 seller · 1 buyer · 2 crane · 3 state (`ROLES`) — 패딩 -1
      choice_time_s  결정 시각 — 패딩 +inf
      n_cands        실제 후보 수 — 패딩 0
    구간×블록×결정×후보 (R,B,C,A)
      rows           후보 특징 (…,37) f32 — 패딩 0.0
      mask           고를 수 있는 후보인가 — 패딩 False (패딩 결정은 전 칸 False)
    스칼라
      n_intervals    실제 구간 수 (≤ R). 이 칸 뒤는 전부 패딩이다
    """

    start_s: jnp.ndarray
    end_s: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    values: jnp.ndarray
    states: jnp.ndarray
    n_choices: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    role: jnp.ndarray
    choice_time_s: jnp.ndarray
    n_cands: jnp.ndarray
    rows: jnp.ndarray
    mask: jnp.ndarray
    n_intervals: jnp.ndarray

    @property
    def r_max(self) -> int:
        return int(self.start_s.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.values.shape[1])

    @property
    def c_max(self) -> int:
        return int(self.action.shape[2])

    @property
    def a_max(self) -> int:
        return int(self.mask.shape[3])

    @property
    def feat_dim(self) -> int:
        return int(self.rows.shape[4])


class IntervalOverflow(NamedTuple):
    """고정 칸에 **못 담은 것의 개수** — v5 에는 없는 개념이므로 0 이 아니면 답이 달라진다.

    intervals  R 칸을 넘쳐 버려진 구간 수
    choices    Cmax 를 넘쳐 버려진 결정 수 (모든 (구간, 블록) 칸의 합)
    cands      Amax 를 넘쳐 잘린 후보 수 (모든 결정의 합)
    """

    intervals: int
    choices: int
    cands: int

    @property
    def any(self) -> bool:
        return bool(self.intervals or self.choices or self.cands)


def _np(x, dtype=None) -> np.ndarray:
    """torch 텐서·numpy 배열·리스트를 numpy 로 — torch 를 수입하지 않고 덕타이핑으로."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu") and not isinstance(x, np.ndarray):
        x = x.cpu()
    out = np.asarray(x)
    return out if dtype is None else out.astype(dtype, copy=False)


def _choice_fields(c):
    """v5 `Choice` (또는 같은 이름표를 가진 것) 에서 여섯 칸을 꺼낸다."""
    return (c.role, float(c.time_s), _np(c.rows), _np(c.mask, np.bool_),
            int(c.action), float(c.log_prob))


def pack_intervals(intervals: Sequence, *, n_blocks: int | None = None,
                   r_max: int | None = None, c_max: int | None = None,
                   a_max: int | None = None, feat_dim: int = FEAT_DIM,
                   ) -> tuple[IntervalBatch, IntervalOverflow]:
    """v5 `Interval` 목록 → `IntervalBatch` + 넘침 개수. **호스트(numpy) 함수**다.

    `r_max`·`c_max`·`a_max` 를 주지 않으면 자료에서 가장 큰 값을 골라 **넘침이 0** 이 되게
    잡는다 (시험 편의). 학습 루프에서는 컴파일 재사용을 위해 **고정값을 주는 것이 맞고**,
    그때 모자라면 `IntervalOverflow` 가 알려준다.

    `n_blocks` 는 구간이 0개일 때만 필수다 (그 밖에는 `len(values)` 로 안다).
    """
    rows_in = list(intervals)
    n_used = len(rows_in) if r_max is None else min(len(rows_in), int(r_max))
    kept = rows_in[:n_used]

    # ── 블록 수: 모든 구간이 같아야 한다 (v5 `buffer.py:48` 의 모양 검사와 같은 뜻)
    if kept:
        b = len(_np(kept[0].values))
        for r in kept:
            if len(_np(r.values)) != b or len(r.choices) != b:
                raise ValueError("구간마다 블록 수가 다르다 — values·choices 길이가 같아야 한다 "
                                 "(v5: matching block dimensions)")
        if n_blocks is not None and int(n_blocks) != b:
            raise ValueError(f"n_blocks={n_blocks} 인데 자료의 블록 수는 {b} 다")
    elif n_blocks is None:
        raise ValueError("구간이 0개면 n_blocks 를 알려 줘야 한다")
    else:
        b = int(n_blocks)

    # ── 칸 크기 정하기 + 넘침 세기
    counts = [[len(r.choices[j]) for j in range(b)] for r in kept]
    need_c = max((max(row) for row in counts), default=0)
    cm = max(1, need_c) if c_max is None else max(1, int(c_max))
    cand_counts = [len(_np(c.rows)) for r in kept for lst in r.choices for c in lst]
    need_a = max(cand_counts, default=0)
    am = max(1, need_a) if a_max is None else max(1, int(a_max))
    rm = max(1, n_used) if r_max is None else max(1, int(r_max))

    over_intervals = len(rows_in) - n_used
    over_choices = sum(max(0, n - cm) for row in counts for n in row)
    over_cands = 0

    # ── values dtype: v5 실행은 float32(torch), 시험이 손으로 만든 것은 float64
    v_dtype = (np.float32 if kept and all(_np(r.values).dtype == np.float32 for r in kept)
               else np.float64)

    start = np.zeros((rm,), np.float64)
    end = np.zeros((rm,), np.float64)
    reward = np.zeros((rm,), np.float64)
    term = np.zeros((rm,), np.bool_)
    values = np.zeros((rm, b), v_dtype)
    states = np.zeros((rm, b, feat_dim), np.float32)
    n_ch = np.zeros((rm, b), np.int32)
    action = np.full((rm, b, cm), -1, np.int32)
    logp = np.zeros((rm, b, cm), np.float64)
    role = np.full((rm, b, cm), -1, np.int32)
    ctime = np.full((rm, b, cm), np.inf, np.float64)
    n_cd = np.zeros((rm, b, cm), np.int32)
    feat = np.zeros((rm, b, cm, am, feat_dim), np.float32)
    mask = np.zeros((rm, b, cm, am), np.bool_)

    for i, r in enumerate(kept):
        start[i], end[i] = float(r.start_s), float(r.end_s)
        reward[i], term[i] = float(r.reward), bool(r.terminated)
        values[i] = _np(r.values, v_dtype)
        st = _np(r.states, np.float32)
        if st.shape != (b, feat_dim):
            raise ValueError(f"states 모양 {st.shape} — ({b}, {feat_dim}) 여야 한다")
        states[i] = st
        for j in range(b):
            lst = r.choices[j]
            n_ch[i, j] = min(len(lst), cm)
            for k, c in enumerate(lst[:cm]):
                rl, t_s, rws, msk, act, lp = _choice_fields(c)
                if rl not in ROLES:
                    raise ValueError(f"알 수 없는 역할 {rl!r} — {ROLES} 중 하나")
                if rws.ndim != 2 or rws.shape[1] != feat_dim:
                    raise ValueError(f"후보 행렬 모양 {rws.shape} — (n, {feat_dim}) 여야 한다")
                if msk.shape != (rws.shape[0],):
                    raise ValueError(f"마스크 모양 {msk.shape} 가 후보 수 {rws.shape[0]} 와 다르다")
                n = min(rws.shape[0], am)
                over_cands += rws.shape[0] - n
                role[i, j, k], ctime[i, j, k] = ROLES.index(rl), t_s
                action[i, j, k], logp[i, j, k] = act, lp
                n_cd[i, j, k] = n
                feat[i, j, k, :n] = rws[:n].astype(np.float32, copy=False)
                mask[i, j, k, :n] = msk[:n]

    batch = IntervalBatch(
        start_s=jnp.asarray(start), end_s=jnp.asarray(end), reward=jnp.asarray(reward),
        terminated=jnp.asarray(term), values=jnp.asarray(values), states=jnp.asarray(states),
        n_choices=jnp.asarray(n_ch), action=jnp.asarray(action), log_prob=jnp.asarray(logp),
        role=jnp.asarray(role), choice_time_s=jnp.asarray(ctime), n_cands=jnp.asarray(n_cd),
        rows=jnp.asarray(feat), mask=jnp.asarray(mask),
        n_intervals=jnp.asarray(n_used, jnp.int32))
    return batch, IntervalOverflow(int(over_intervals), int(over_choices), int(over_cands))


def unpack_choice(batch: IntervalBatch, r: int, b: int, c: int) -> dict:
    """진단용 — 패딩을 떼고 결정 하나를 v5 `Choice` 모양의 사전으로 돌려준다."""
    n = int(batch.n_cands[r, b, c])
    code = int(batch.role[r, b, c])
    return {"role": ROLES[code] if code >= 0 else None,
            "time_s": float(batch.choice_time_s[r, b, c]),
            "rows": np.asarray(batch.rows[r, b, c, :n]),
            "mask": np.asarray(batch.mask[r, b, c, :n]),
            "action": int(batch.action[r, b, c]),
            "log_prob": float(batch.log_prob[r, b, c])}


# ── 통합 단계(update.py)가 그대로 쓰는 마스크·계수
def valid_mask(batch: IntervalBatch) -> jnp.ndarray:
    """(R,) bool — 실제 구간 칸. 패딩은 False."""
    return jnp.arange(batch.r_max, dtype=I32) < batch.n_intervals


def active_mask(batch: IntervalBatch) -> jnp.ndarray:
    """(R,B) bool — **결정이 하나라도 있는** (구간, 블록) 칸.

    v5 `update.py:21` 의 `active = [(i,b) … if intervals[i].choices[b]]` 와 같다 —
    우위 정규화에는 이 칸만 들어가고, 빈 블록도 가치망은 학습한다 (`update.py:22`).
    """
    return valid_mask(batch)[:, None] & (batch.n_choices > 0)


def micro_actions(batch: IntervalBatch) -> jnp.ndarray:
    """v5 `update.py:82` 의 `micro_actions` — 실제 결정 총수 () int32."""
    return jnp.sum(jnp.where(valid_mask(batch)[:, None], batch.n_choices, 0), dtype=I32)


def block_samples(batch: IntervalBatch) -> int:
    """v5 `update.py:81` 의 `block_samples` = 구간 수 × 블록 수 (패딩 제외)."""
    return int(batch.n_intervals) * batch.n_blocks


def flat_entries(batch: IntervalBatch) -> np.ndarray:
    """v5 `update.py:19-20` 의 `entries` 순서 그대로 (구간 바깥·블록 안쪽) — (n·B, 2) int32.

        entries = [(i, b) for i, row in enumerate(intervals) for b in range(len(row.values))]

    미니배치 순열은 **호스트에서** 이 순서 위에 만든다 (`rng.permutation(len(entries))`) —
    numpy 난수열을 배열판에서 재현할 수 없으므로 v5 와 같은 순열을 그대로 넘겨야 한다.
    평평한 번호 k 는 `(k // B, k % B)` 다.
    """
    n, b = int(batch.n_intervals), batch.n_blocks
    ii, bb = np.divmod(np.arange(n * b, dtype=np.int32), b)
    return np.stack([ii, bb], axis=1)


# ═══════════════════════════════════════════════ ② GAE (역방향 scan)
class GaeAudit(NamedTuple):
    """v5 `gae` 가 `ValueError` 로 던지던 검사들 — jit 안에서는 비트로 돌려준다.

    empty           구간이 0개 (`buffer.py:34`)
    bad_bootstrap   부트스트랩에 비유한 값 (`buffer.py:38`)
    bad_time        (R,) 시각이 비유한·음수이거나 end ≤ start (`buffer.py:42-44`)
    non_contiguous  (R,) 직전 구간이 종료가 아닌데 시각이 안 이어짐 (`buffer.py:45-47`)
    bad_sample      (R,) values 또는 reward 가 비유한 (`buffer.py:48-50`)
    """

    empty: jnp.ndarray
    bad_bootstrap: jnp.ndarray
    bad_time: jnp.ndarray
    non_contiguous: jnp.ndarray
    bad_sample: jnp.ndarray

    @property
    def ok(self) -> bool:
        return not (bool(self.empty) or bool(self.bad_bootstrap)
                    or bool(np.asarray(self.bad_time).any())
                    or bool(np.asarray(self.non_contiguous).any())
                    or bool(np.asarray(self.bad_sample).any()))


def _check_config(gamma: float, lam: float, time_unit_s: float) -> None:
    """설정값은 파이썬 상수다 — v5 와 같은 자리에서 같은 메시지로 던진다 (`buffer.py:34-36`)."""
    if not (np.isfinite(time_unit_s) and time_unit_s > 0
            and 0 < gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("Invalid rollout or discount configuration")


def gae_audit(batch: IntervalBatch, bootstrap, *, gamma: float, lam: float,
              time_unit_s: float) -> GaeAudit:
    """v5 `gae` 의 입력 검사를 **배열로** — 던지지 않고 비트로 돌려준다 (순수·jit 가능)."""
    _check_config(gamma, lam, time_unit_s)
    nxt = jnp.asarray(bootstrap, F64)
    if nxt.ndim != 1 or not nxt.shape[0]:
        raise ValueError("Bootstrap must be a finite vector with one value per block")
    if nxt.shape[0] != batch.n_blocks:
        raise ValueError("Rollout must have finite rewards/values and matching block dimensions")

    on = valid_mask(batch)
    s, e = batch.start_s, batch.end_s
    bad_time = on & (~jnp.isfinite(s) | ~jnp.isfinite(e) | (s < 0) | (e <= s))

    prev_end = jnp.concatenate([jnp.zeros((1,), F64), e[:-1]])
    prev_term = jnp.concatenate([jnp.ones((1,), jnp.bool_), batch.terminated[:-1]])
    close = (prev_end == s) | (jnp.abs(prev_end - s) <= CONTIGUITY_TOL)   # math.isclose(abs_tol=1e-6)
    has_prev = jnp.arange(batch.r_max, dtype=I32) > 0
    non_contig = on & has_prev & ~prev_term & ~close

    finite_v = jnp.all(jnp.isfinite(batch.values), axis=1)
    bad_sample = on & (~finite_v | ~jnp.isfinite(batch.reward))
    return GaeAudit(empty=batch.n_intervals <= 0,
                    bad_bootstrap=~jnp.all(jnp.isfinite(nxt)),
                    bad_time=bad_time, non_contiguous=non_contig, bad_sample=bad_sample)


def raise_on_audit(audit: GaeAudit) -> None:
    """위반 비트를 **v5 와 같은 순서·같은 메시지**의 `ValueError` 로 바꾼다 (호스트 전용).

    v5 는 ① 설정 ② 부트스트랩 ③ 구간을 앞에서부터 훑으며 (시각 → 연속성 → 표본) 검사한다
    (`buffer.py:34-50`). 그 순서 그대로 첫 위반에서 던진다.
    """
    if bool(audit.empty):
        raise ValueError("Invalid rollout or discount configuration")
    if bool(audit.bad_bootstrap):
        raise ValueError("Bootstrap must be a finite vector with one value per block")
    bt = np.asarray(audit.bad_time)
    nc = np.asarray(audit.non_contiguous)
    bs = np.asarray(audit.bad_sample)
    for i in range(len(bt)):
        if bt[i]:
            raise ValueError("Synchronization intervals must advance finite physical time")
        if nc[i]:
            raise ValueError("Nonterminal collection intervals must be contiguous")
        if bs[i]:
            raise ValueError("Rollout must have finite rewards/values and matching block dimensions")


def gae(batch: IntervalBatch, bootstrap, *, gamma: float, lam: float,
        time_unit_s: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """물리시간 할인 GAE — v5 `buffer.gae` 와 같은 답 (`buffer.py:32-64`).

    돌려주는 값: `(adv (R,B) float64, returns (R,B) float64)`. 패딩 구간 칸은 둘 다 0 이다.

    ★v5 식 그대로 (`buffer.py:54-62`)
        dt        = (end − start) / time_unit_s
        discount  = gamma**dt × (0 종료 / 1 계속)
        delta     = reward + discount×nxt − values
        trace     = delta + discount × lam**dt × trace        ← 뒤 구간에서 앞으로
        adv[i]    = trace ;  nxt = values
        returns   = adv + values

    ⚠️ 이 함수는 **던지지 않는다** — 입력 검사는 `gae_audit` + `raise_on_audit` 가 맡는다
    (설정값만은 파이썬 상수라 여기서도 즉시 던진다).
    """
    _check_config(gamma, lam, time_unit_s)
    nxt0 = jnp.asarray(bootstrap, F64)
    if nxt0.ndim != 1 or not nxt0.shape[0]:
        raise ValueError("Bootstrap must be a finite vector with one value per block")
    if nxt0.shape[0] != batch.n_blocks:
        raise ValueError("Rollout must have finite rewards/values and matching block dimensions")

    g64, l64 = jnp.asarray(gamma, F64), jnp.asarray(lam, F64)
    #: ★values 가 float32 면 `delta` 도 float32 로 계산된다 (머리말 (2))
    f32_values = batch.values.dtype == jnp.dtype(F32)
    last = batch.n_intervals - 1
    idx = jnp.arange(batch.r_max, dtype=I32)

    def body(carry, x):
        trace, nxt = carry                              # (B,) f64 · (B,) f64
        i, start, end, reward, terminated, values = x
        on = i < batch.n_intervals                      # 패딩 칸이면 캐리를 그대로 통과
        first = i == last                               # ★첫 실제 단계만 nxt 가 부트스트랩(f64)

        dt = div_const(end - start, time_unit_s)        # (end − start) / 60.0 — 상수 나눗셈
        cont = jnp.where(terminated, jnp.zeros((), F64), jnp.ones((), F64))
        disc = mul_exact(jnp.power(g64, dt), cont)      # gamma**dt * continuation
        dlam = mul_exact(disc, jnp.power(l64, dt))      # discount * lam**dt  (파이썬 float 하나)

        # delta = (reward + discount*nxt) − values   ← 괄호는 파이썬 연산자 우선순위 그대로
        d64 = (reward + mul_exact(disc, nxt)) - values.astype(F64)
        if f32_values:
            v32 = values.astype(F32)
            d32 = (reward.astype(F32) + mul_exact(disc.astype(F32), nxt.astype(F32))) - v32
            delta = jnp.where(first, d64, d32.astype(F64))
        else:
            delta = d64

        # trace = delta + ((discount * lam**dt) * trace)
        new_trace = delta + mul_exact(dlam, trace)
        keep_trace = jnp.where(on, new_trace, trace)
        keep_nxt = jnp.where(on, values.astype(F64), nxt)
        return (keep_trace, keep_nxt), jnp.where(on, new_trace, jnp.zeros((), F64))

    init = (jnp.zeros_like(nxt0), nxt0)
    _, adv = lax.scan(body, init,
                      (idx, batch.start_s, batch.end_s, batch.reward,
                       batch.terminated, batch.values), reverse=True)
    on = valid_mask(batch)[:, None]
    returns = jnp.where(on, adv + batch.values, jnp.zeros((), F64))
    return jnp.where(on, adv, jnp.zeros((), F64)), returns


def gae_batch(batches: IntervalBatch, bootstraps, *, gamma: float, lam: float,
              time_unit_s: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """세계 W 개를 한 번에 — 잎마다 앞에 축 하나가 더 붙은 `IntervalBatch` 를 받는다.

    `n_intervals` 도 (W,) 라서 세계마다 구간 수가 달라도 된다 (패딩 규약이 그것을 흡수한다).
    돌려주는 값은 `(W,R,B)` 둘.
    """
    def one(b, boot):
        return gae(b, boot, gamma=gamma, lam=lam, time_unit_s=time_unit_s)

    # IntervalBatch 는 pytree 라 vmap 이 잎마다 0축을 벗긴다 — dtype 갈래는 정적으로 남는다
    return jax.vmap(one)(batches, jnp.asarray(bootstraps, F64))
