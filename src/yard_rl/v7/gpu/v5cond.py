"""조건부 규약 — 순차 조건부 joint_mask · 선호 정렬키 통일 · "한 크레인 실패 = 전원 WAIT"
([[YR-327]] 조각 7 · key=cond).

v5 정본 셋을 배열로 옮긴다. **학습 정책망(37특징 tanh)** 은 다른 담당(key=v5_policy)이고, 여기는
그 망이 서기 위한 *발판* — "지금 이 크레인이 고를 수 있는 후보는 무엇인가"(마스크)와 "고른 한 벌이
성립하지 않으면 어떻게 되는가"(전원 WAIT)다.

■ 옮긴 것 (v5 → 배열)
    `ppo/crane.py:20-36`   `joint_mask`  — 공동 결정에서 (크레인, 후보) 짝이 **앞 크레인의 조건부
                                           약속과 함께** 성립하는지 (K 번 순차로 묻는다)
    `ppo/crane.py:52-56`   `prior`       — "둘째 크레인은 첫째의 조건부 약속을 본다" (직전 **물은** 크레인)
    `ppo/crane.py:72-78`   순차 조건부 루프 (크레인 사전순 = 번호순)
    `stage/episode.py:208-216` `_rule_policy.exec_policy` 의 try/except — 결정 중 **어떤 예외든**
                                           그 결정을 **전 크레인 WAIT** 로 바꾸고 계수한다 (관측 거동)
    `baselines.py:28-60`   선호 `rank` 3-튜플 통일 (BaselinePreference · ServiceFirstSPT · FIFO)
    `baselines.py:134-149` `_feasible_joint` (token 중복 + dry_run 오라클)
    `baselines.py:159-169` `_apply` → **이미 있다**: `engine_step.apply_choices` (여기서 안 만든다)

■ ★겹침 확인 — `gpu/dispatch.py:resolve_central` 과 무엇이 같고 무엇이 다른가 (담당 지시 ①)
  `resolve_central` 은 v5 `CentralResolver.resolve` (전 쌍을 완전순서로 줄 세워 **그리디**) 다.
  `joint_mask` 는 v5 `ppo/crane.py` 의 **순차 조건부** (크레인마다 마스크를 주고 *정책이* 고른다) 로,
  알고리즘도 산출 모양도 다르다 — 겹치는 것은 **공동 실행가능 오라클** 하나뿐이다. 그래서 여기서는
  오라클을 **재구현하지 않고** `dispatch.dry_run_joint`(= v5 `engine.dry_run_commit` 738-763행) 를
  그대로 부른다. 새 물리 계산은 이 파일에 한 줄도 없다.
  선호 정렬키는 **여기가 정본**이다 (2026-09-26 통합 · 검증 반박 "사본이 둘"): `dispatch.resolve_central` 이
  자기 안에 같은 식을 펼쳐 들고 있었는데, 지금은 `pair_order` 하나를 부른다. 그래서 `pref_cols` ·
  `pair_key_cols` · `pair_order` 는 **생산 경로**(엔진 규칙 resolver) 가 쓰는 코드이고, 조각 1~6 골든
  (Y01 `6668fa4902c4efe4` 등)이 그 값을 잠근다.
  ⚠️ `pref_cols` 의 "fifo" 는 `resolve_central` 이 **거절**한다 — 실현 도착시각을 읽는 오라클(YR-107)이라
     배포 경로에 둘 수 없다. 상한 진단으로 쓰려면 `pair_order` 를 직접 부른다.

■ 이 파일의 **진단·시험 전용** 함수 (생산 호출자 없음 — 2026-09-26 grep 확인)
  생산이 쓰는 것은 넷이다: `joint_mask_items` · `item_max` · `sequential_conditional` · `guard_all_wait`
  (+ 위 정렬키 셋). 나머지 — `wait_col`·`item_cols`·`n_items`·`sel_tokens`·`taken_tokens`·`joint_mask_flat`·
  `pick_first_masked`·`pick_last_masked`·`make_seq_policy`·`feasible_joint`·`normalize_choice`·
  `decision_would_raise`·`first_failing_crane`·`all_wait_choice`·`substitute_all_wait` — 는 시험·진단이
  부르는 얇은 겉옷이다 (`normalize_choice`·`all_wait_choice`·`substitute_all_wait`·`decision_would_raise`
  는 `guard_all_wait` 안에서 쓰인다). 새 물리 계산은 한 줄도 없다.
  `feasible_joint` 의 token 중복 검사는 `reserve.reject_code` 의 DUP_JOB 과 **답이 같다** — v5 가 먼저
  검사하는 자리라 뜻을 남겨 둔 것이고(시험이 둘의 일치를 못박는다) 생산 경로는 `dry_run_joint` 만 쓴다.

■ 미이식 — 기록 (다음 담당이 같은 대조를 다시 하지 않게)
  · `baselines._untriggered_defer` (152-156행) — 유일한 호출자가 `baselines.py:304`(조합 열거 정책
    `JointRolloutGreedy` 계열) 라 **학습 경로 밖**이다. 명세 files_to_port 가 가리킨 구간(134-169) 안에
    있지만 이식 대상이 아니다.
  · DEFER wake(유한 대기 뒤 재개방) — 정책 반환에 `defer_until` 칸이 없다. 학습 경로는
    `LEGACY_DEFAULT`(wait_mode='WAIT')이고 `candidates.py:449-450` 이 그때 `defer_until=None` 을 주므로
    **구조상 안 밟힌다**. 후보 설정을 넓히는 조각(README 한계표)의 몫이다.

■ 전원 WAIT 규칙 — v5 가 **어느 예외에서** 그렇게 하나 (담당 지시 ②)
  `stage/episode.py:209-216` 의 try 는 `pol.decide(...)` 와 `_apply(...)` 를 함께 감싼다. 그 안에서
  실제로 터질 수 있는 것은
    (a) `_apply` → `sim.assign` (engine.py:694-697) 의 `_plan` 이 None → ConstraintViolation
        "NOT_DISPATCHABLE", 또는 `reservations.reserve` 의 5-lock 거절 → ConstraintViolation
    (b) `_plan` 안의 find_slot 후조건 RuntimeError (engine.py:636)
    (c) `close_decision` 의 불변식 위반
  이고, **(a) 가 이 파일이 덮는 자리**다. 배열판은 위반 비트가 아니라 같은 관측 거동을 낸다 —
  결정을 적용하기 **전에** `dry_run_joint` 로 "적용하면 터지는가"를 묻고, 터지면 물은 크레인 전원의
  답을 WAIT 열로 바꾼다 (`guard_all_wait`).
  ★한계 둘 (시험이 둘 다 논증한다):
    · v5 의 대체는 **아직 아무도 배정되지 않았을 때만** 관측 가능하다. `_apply` 가 앞 크레인을 배정한
      뒤 터지면 대체 `_apply` 가 그 크레인을 다시 배정하려다 DECISION_COVERAGE 로 **잡히지 않고**
      죽는다 (episode.py 의 except 는 한 번뿐). 그래서 "실패 → 전원 WAIT" 는 실패를 **선판정**하는
      배열판 쪽이 오히려 v5 의 의도대로 도는 경우다.
    · (b) 는 `PlanOut.viol`(V_PLAN_POSTCOND) 에 따로 남고 `dry_run_joint` 의 bool 에는 안 섞인다 —
      find_slot 이 옳으면 안 켜지는 엔진 버그 감시 비트다 (open_issue).

■ 같은 답을 내기 위한 규칙
  · 새 실수 연산 없음. 선호 열은 `resolve_central` 과 **같은 식**(−0.0 → +0.0 정규화까지)이고 float64 다.
  · 크레인 순서 = 번호 순 = v5 `sorted(dp.crane_ids)`. 후보 순서 = `prune.candidate_id` (= v5 items 위치).
  · 고정 크기: items 칸 I = k_max + 1 (실린 후보 ≤ k_max−1 + WAIT). 넘치면 조용히 자르지 않고
    `V_RESOLVER_TRUNC` 를 켠다 (`resolve_central` 과 같은 규약).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .cands3 import K_MAX
from .dispatch import ResolverParams, _pad_orders, dry_run_joint
from .events import EMPTY_TIME
from .geom import Geom
from .state import (EMPTY_ID, PK_PRE_REHANDLE, PK_REPOSITION, PK_SERVE, PK_WAIT, V_RESOLVER_TRUNC,
                    BlockWorld)

__all__ = [
    "PREF_NAMES", "item_max", "wait_col", "item_cols", "n_items", "sel_tokens", "taken_tokens",
    "pref_cols", "pair_key_cols", "pair_order",
    "prior_open", "joint_mask_items", "joint_mask_flat", "SeqOut", "sequential_conditional",
    "pick_first_masked", "pick_last_masked", "make_seq_policy",
    "feasible_joint", "normalize_choice", "decision_would_raise", "first_failing_crane",
    "all_wait_choice", "substitute_all_wait", "guard_all_wait",
]

#: 이름 순위표의 빈 칸 — 어떤 실제 이름보다 뒤 (dispatch.resolve_central 과 같은 값)
_NAME_PAD = 1 << 30
#: 옮긴 선호 — v5 `resolver.BaselinePreference`(27-33행) · `baselines.ServiceFirstSPTPreference`(34-37행) ·
#: `baselines.FIFOPreference`(51-60행). ⚠️ "fifo" 는 **실현 도착시각**을 읽는 오라클이다 (YR-107) — 진단 전용.
PREF_NAMES = ("baseline", "sf_spt", "fifo")


# ───────────────────────────────────────────────── 후보 칸 (items 공간 ↔ 평면 열)
def item_max(k_max: int = K_MAX) -> int:
    """items 칸 수 I — 실린 후보(≤ k_max−1) + mandatory 여유 + WAIT 1칸 (candidates.py:504-517 `_prune`)."""
    return int(k_max) + 1


def wait_col(fl) -> int:
    """평면 후보 행렬의 WAIT 열 번호 (cands3.flat_view: 마지막 열)."""
    return int(fl.raw.shape[1]) - 1


def item_cols(pr, k, *, k_max: int = K_MAX) -> jnp.ndarray:
    """items 색인 i → 평면 열 번호 (I,) int32, 빈 칸은 −1.

    v5 `generate().items` 의 위치가 곧 `candidate_id` 이므로, `prune.candidate_id` 를 거꾸로 흩어
    "i 번째 후보는 어느 열인가" 표를 만든다. I 를 넘는 id(= mandatory 초과)는 버려지고
    `sequential_conditional` 이 `V_RESOLVER_TRUNC` 로 알린다.
    """
    C = int(pr.keep.shape[1])
    I = item_max(k_max)
    cid = jnp.where(pr.keep[k], pr.candidate_id[k], jnp.int32(I))       # 안 실린 열은 범위 밖으로
    return jnp.full((I,), EMPTY_ID, jnp.int32).at[cid].set(
        jnp.arange(C, dtype=jnp.int32), mode="drop")


def n_items(pr) -> jnp.ndarray:
    """크레인별 items 길이 (K,) int32 — 실린 후보 + WAIT (= WAIT 의 candidate_id + 1)."""
    return jnp.sum(pr.keep, axis=1).astype(jnp.int32)


def sel_tokens(fl, sel) -> jnp.ndarray:
    """선택 열 (K,) → token (K,) int32 — SERVE·PRE 는 오더 번호, REPO·WAIT·없음은 −1.

    v5 token 규약: SERVE `JobRef.token = job_id`(engine.py:546) · PRE 도 job_id · REPO 는 None
    (candidates.py:394, 413) · WAIT 는 job_ref 자체가 None.
    """
    K, C = fl.kind.shape
    kk = jnp.arange(K, dtype=jnp.int32)
    c = jnp.asarray(sel, jnp.int32)
    cc = jnp.clip(c, 0, C - 1)
    has = (c >= 0) & (c < C)
    kind = fl.kind[kk, cc]
    is_tok = has & ((kind == PK_SERVE) | (kind == PK_PRE_REHANDLE))
    return jnp.where(is_tok, fl.job[kk, cc], EMPTY_ID).astype(jnp.int32)


def taken_tokens(toks, n: int) -> jnp.ndarray:
    """token 열 (K,) → 오더별 '이미 잡혔나' (N,) bool — v5 `tokens` 집합 (ppo/crane.py:23)."""
    return jnp.any(jnp.asarray(toks, jnp.int32)[:, None] == jnp.arange(int(n), dtype=jnp.int32)[None, :],
                   axis=0)


# ───────────────────────────────────────────────── 선호 정렬키 통일 (baselines.py:28-60)
def pref_cols(pref: str, params: ResolverParams, world: BlockWorld, fl, g: Geom) -> tuple:
    """선호 `rank(sim, crane_id, gc)` 의 정렬 열 — **사전식 앞→뒤**, 각 열 (K,C).

    세 선호가 모두 `(tier int32, value float64, name int32)` **삼조**로 정리된다 — 문자열(job_id ·
    REPO 이름 "REPO:<크레인>:<int(bay)>")은 호스트가 구운 정수 순위표(`dispatch.resolver_params`)로,
    파이썬 `float` 은 float64 열로 옮긴다. `ServiceFirstSPT` 만 그 앞에 (SERVE 우선 tier, 소요 value)
    두 열이 더 붙는다 (baselines.py:37 `+ super().rank(...)`).

        baseline  (resolver.py:27-33)  (본선?0:1 [WAIT 2],  −누적대기,           job_id 순위)
        sf_spt    (baselines.py:34-37) (SERVE?0:1, 소요[없으면 ∞]) + baseline 삼조
        fifo      (baselines.py:51-60) (SERVE?0:1 [WAIT 2], 실현 블록도착[없으면 ∞], job_id 순위)

    ⚠️ "fifo" 는 `actual_block_arrival`(트럭의 **진짜** 도착시각)을 `now` 게이트 없이 읽는다 —
       배포 불가·상한 진단 전용 (`FIFOPreference.USES_FUTURE_INFORMATION = True`).
    """
    K, C = fl.kind.shape
    N = world.n
    B = int(g.bay_count)
    o, clock = world.orders, world.clock
    kind = fl.kind
    is_wait = kind == PK_WAIT
    is_serve = kind == PK_SERVE
    is_pre = kind == PK_PRE_REHANDLE
    is_repo = kind == PK_REPOSITION
    jc = jnp.clip(fl.job, 0, N - 1)
    # 이름 열 — rank 의 마지막 항 (job_id 또는 REPO 이름). WAIT 는 "" 라 어떤 이름보다 앞 → −1
    nrj = _pad_orders(params.name_rank_job, N, jnp.int32(_NAME_PAD))
    bay_i = jnp.clip(jnp.floor(jnp.where(jnp.isnan(fl.bay), 0.0, fl.bay)).astype(jnp.int32), 0, B)
    k_idx = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32)[:, None], (K, C))
    name = jnp.where(is_wait, EMPTY_ID,
                     jnp.where(is_repo, params.name_rank_repo[k_idx, bay_i], nrj[jc])).astype(jnp.int32)

    if pref in ("baseline", "sf_spt"):
        # JobRef 의 is_vessel·is_external (engine.py:547-549) — PRE 는 vessel False · external True
        ref_vessel = is_serve & o.is_vessel[jc]
        ref_ext = (is_serve & o.is_external[jc]) | is_pre
        arrived = o.is_external & (o.block_in_s < EMPTY_TIME) & (o.block_in_s <= clock)
        cum = jnp.where(arrived, clock - o.block_in_s, 0.0)             # engine.py:258-265 cum_wait
        cum_key = jnp.where(is_wait, 0.0, jnp.where(ref_ext, -cum[jc], 0.0))
        cum_key = jnp.where(cum_key == 0.0, 0.0, cum_key)               # −0.0 → +0.0 (파이썬 정렬은 같게 본다)
        tier = jnp.where(is_wait, 2, jnp.where(ref_vessel, 0, 1)).astype(jnp.int32)
        base = (tier, cum_key, name)
        if pref == "baseline":
            return base
        dur = jnp.where(fl.plan_ok, fl.dur, jnp.inf)                    # plan None → inf (baselines.py:36)
        dur = jnp.where(dur == 0.0, 0.0, dur)
        return (jnp.where(is_serve, 0, 1).astype(jnp.int32), dur) + base

    if pref == "fifo":
        arr = o.actual_arrival_s                                        # ⚠️ 미래정보 (YR-107)
        seen = o.is_external & (arr < EMPTY_TIME)
        val = jnp.where(is_wait, 0.0,
                        jnp.where(is_repo, jnp.inf,                     # "REPO:…" 는 sim.jobs 에 없다 → inf
                                  jnp.where(seen[jc], arr[jc], jnp.inf)))
        tier = jnp.where(is_wait, 2, jnp.where(is_serve, 0, 1)).astype(jnp.int32)
        return (tier, val, name)

    raise ValueError(f"모르는 선호 {pref!r} — {PREF_NAMES}")


def pair_key_cols(pref: str, params: ResolverParams, world: BlockWorld, fl, pr, g: Geom) -> tuple:
    """v5 `CentralResolver._pair_key` (resolver.py:79-82) 전체 — 사전식 앞→뒤.

        (mandatory?0:1) + 선호.rank + (kind 순위, 크레인 번호, token 순위, candidate_id)

    꼬리의 문자열 둘(crane_id · token)은 번호로 옮긴다 — 크레인 번호는 `sorted(crane_id)` 순위이고
    token 순위는 오더 번호(= `sorted(job_id)` 순위)다. REPO·WAIT 의 token `""` 은 어떤 token 보다
    앞이므로 −1.
    """
    K, C = fl.kind.shape
    N = world.n
    kind = fl.kind
    is_serve = kind == PK_SERVE
    is_pre = kind == PK_PRE_REHANDLE
    is_repo = kind == PK_REPOSITION
    jc = jnp.clip(fl.job, 0, N - 1)
    tkr = _pad_orders(params.tok_rank, N, jnp.int32(_NAME_PAD))
    mand = jnp.where(fl.mandatory, 0, 1).astype(jnp.int32)
    kind_rank = jnp.where(is_serve, 0, jnp.where(is_pre, 1, jnp.where(is_repo, 2, 3))).astype(jnp.int32)
    k_idx = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32)[:, None], (K, C))
    tok = jnp.where(is_serve | is_pre, tkr[jc], EMPTY_ID).astype(jnp.int32)
    return (mand,) + pref_cols(pref, params, world, fl, g) + (kind_rank, k_idx, tok, pr.candidate_id)


def pair_order(pref: str, params: ResolverParams, world: BlockWorld, fl, pr, valid, g: Geom) -> jnp.ndarray:
    """쌍 (크레인 k, 열 c) 의 정렬 순 (K·C,) — 유효 쌍이 앞, 그 안은 `_pair_key` 순 (resolver.py:53-54).

    `valid` (K,C) bool = `prune.keep & flat.feasible & open[:, None]` (53행 `gc.feasible`).
    돌려주는 번호는 `p = k·C + c` 다. `jnp.lexsort` 는 **마지막 키가 최우선**이라 열을 거꾸로 넣는다.
    """
    cols = pair_key_cols(pref, params, world, fl, pr, g)
    keys = tuple(c.reshape(-1) for c in reversed(cols))
    keys += ((~jnp.asarray(valid, bool)).reshape(-1).astype(jnp.int32),)
    return jnp.lexsort(keys)


# ───────────────────────────────────────────────── 순차 조건부 (ppo/crane.py:20-36, 52-56, 72-78)
def prior_open(open_) -> jnp.ndarray:
    """(K,) int32 — 크레인 k 의 **직전 물은 크레인** 번호 (없으면 −1).

    v5 `next(reversed(selected.values()), None)` (ppo/crane.py:53): `selected` 는 `sorted(dp.crane_ids)`
    순으로 쌓이므로 '직전 값' = 번호가 k 보다 작은 **열린** 크레인 중 가장 큰 번호다.
    """
    open_ = jnp.asarray(open_, bool)
    K = int(open_.shape[0])
    idx = jnp.where(open_, jnp.arange(K, dtype=jnp.int32), jnp.int32(EMPTY_ID))
    run = lax.associative_scan(jnp.maximum, idx)
    return jnp.concatenate([jnp.full((1,), EMPTY_ID, jnp.int32), run[:-1]])


def joint_mask_items(world: BlockWorld, fl, pr, sel, k, g: Geom, *, k_max: int = K_MAX):
    """v5 `ppo/crane.py:20-36` `joint_mask` — 크레인 k 의 items 마스크 (I,) bool 과 열표 (I,) int32.

    `sel` (K,) int32 = 지금까지의 조건부 약속 (열 번호 · −1 = 아직/안 물음). 자기 행은 무시한다
    (v5 는 `selected` 에 자기가 아직 없다).

        ok = gc.feasible                                              (WAIT 은 여기서 끝 — True)
        ok &= token 이 앞 크레인에게 잡히지 않았다                      (REPO 는 token None → 통과)
        ok &= dry_run_commit({앞 크레인들} ∪ {나}) 가 **전원** 성립     (`dispatch.dry_run_joint`)

    v5 는 마스크가 전부 거짓이면 RuntimeError 를 낸다(35행). LEGACY_DEFAULT 후보 설정에서는 WAIT
    후보의 `feasible` 이 항상 True 라(candidates.py:444-448) **구조상 도달 불가**하고, 배열판도
    WAIT 열의 `flat.feasible` 이 항상 True 다 — 시험이 이 불가능성을 상시 단언한다.
    """
    C = int(fl.raw.shape[1])
    N = world.n
    icol = item_cols(pr, k, k_max=k_max)
    sel_c = jnp.asarray(sel, jnp.int32).at[k].set(jnp.int32(EMPTY_ID))       # committed (자기 제외)
    taken = taken_tokens(sel_tokens(fl, sel_c), N)
    have = icol >= 0
    cc = jnp.clip(icol, 0, C - 1)
    kind = fl.kind[k, cc]
    is_wait = kind == PK_WAIT
    feas = have & fl.feasible[k, cc]
    tok = jnp.where((kind == PK_SERVE) | (kind == PK_PRE_REHANDLE), fl.job[k, cc], EMPTY_ID).astype(jnp.int32)
    tok_free = (tok < 0) | ~taken[jnp.clip(tok, 0, N - 1)]
    joint = jax.vmap(lambda c: dry_run_joint(world, fl, sel_c.at[k].set(c), g))(cc)
    return feas & (is_wait | (tok_free & joint)), icol


def joint_mask_flat(world: BlockWorld, fl, pr, sel, k, g: Geom, *, k_max: int = K_MAX) -> jnp.ndarray:
    """`joint_mask_items` 를 평면 열 공간 (C,) bool 로 — 안 실린 열은 거짓."""
    C = int(fl.raw.shape[1])
    m, icol = joint_mask_items(world, fl, pr, sel, k, g, k_max=k_max)
    return jnp.zeros((C,), bool).at[jnp.where(icol >= 0, icol, C)].set(m, mode="drop")


class SeqOut(NamedTuple):
    """순차 조건부 한 결정 — `apply_choices` 가 그대로 받는 `choice` 와 그 흔적."""

    choice: jnp.ndarray     # (K,)  int32  평면 열 번호 (−1 = 안 물음)
    item: jnp.ndarray       # (K,)  int32  items 색인 (v5 `candidates[idx]` 의 idx; −1 = 안 물음)
    mask: jnp.ndarray       # (K,I) bool   단계 k 에서 크레인 k 가 본 joint_mask
    item_col: jnp.ndarray   # (K,I) int32  items 색인 → 열 번호
    prior: jnp.ndarray      # (K,)  int32  직전 물은 크레인 (ppo/crane.py:53)
    flags: jnp.ndarray      # ()    int32  V_RESOLVER_TRUNC (items 칸 넘침) | pick_fn 이 올린 비트(V_NET_NONFINITE)


def sequential_conditional(world: BlockWorld, fl, pr, open_, pick_fn: Callable, g: Geom, *,
                           k_max: int = K_MAX) -> SeqOut:
    """v5 `ppo/crane.py:72-78` 의 순차 조건부 — 크레인 번호 순으로 마스크를 만들어 정책에게 묻는다.

        for cid in sorted(dp.crane_ids):
            mask = joint_mask(sim, items, selected);  selected[cid] = candidates[정책(mask)]

    `pick_fn(k, prior, sel, mask, item_col) -> () int32` (items 색인) 는 jit 안에서 불린다 —
    37특징 tanh 망(key=v5_policy)이 들어올 자리이고, 시험은 `pick_first_masked` 같은 결정적 대역을 쓴다.
    안 물은 크레인(open 거짓)은 묻지 않고 −1 로 둔다 — v5 는 `dp.crane_ids` 만 순회한다.

    ★`pick_fn` 은 `(items 색인, 위반 비트 () int32)` **두 값**을 돌려줄 수도 있다 (2026-09-26 통합).
      학습 정책망이 v5 `encode` 의 유한성 거부(`ppo/model.py:17-18`)를 `V_NET_NONFINITE` 로 알리는 길이다 —
      jit 안에서는 예외를 던질 수 없으므로 조용히 이상한 결정을 내지 않고 비트로 크게 알린다.
      돌려주는 `SeqOut.flags` 가 전 단계의 비트를 OR 해 담는다.
    """
    K = world.k
    I = item_max(k_max)
    open_ = jnp.asarray(open_, bool)
    prior = prior_open(open_)
    flags0 = jnp.where(jnp.any(n_items(pr) > I), V_RESOLVER_TRUNC, 0).astype(jnp.int32)

    def body(carry, k):
        sel, viol = carry
        m, icol = joint_mask_items(world, fl, pr, sel, k, g, k_max=k_max)
        out = pick_fn(k, prior[k], sel, m, icol)
        i_raw, vb = out if isinstance(out, tuple) else (out, jnp.zeros((), jnp.int32))
        i = jnp.asarray(i_raw, jnp.int32)
        ok = open_[k] & (i >= 0) & (i < I)
        c = jnp.where(ok, icol[jnp.clip(i, 0, I - 1)], jnp.int32(EMPTY_ID)).astype(jnp.int32)
        #: 안 물은 크레인의 위반 비트는 세지 않는다 — v5 는 그 크레인에게 묻지도 않는다
        viol = viol | jnp.where(open_[k], jnp.asarray(vb, jnp.int32), 0).astype(jnp.int32)
        return (sel.at[k].set(c), viol), (jnp.where(ok, i, jnp.int32(EMPTY_ID)).astype(jnp.int32), m, icol)

    sel0 = jnp.full((K,), EMPTY_ID, jnp.int32)
    (sel, viol), (item, mask, icols) = lax.scan(body, (sel0, jnp.zeros((), jnp.int32)),
                                                jnp.arange(K, dtype=jnp.int32))
    return SeqOut(choice=sel, item=item, mask=mask, item_col=icols, prior=prior,
                  flags=(flags0 | viol).astype(jnp.int32))


def pick_first_masked(k, prior, sel, mask, item_col) -> jnp.ndarray:
    """마스크가 산 **가장 앞** items 색인 — 정책 자리의 결정적 대역 (시험용)."""
    return jnp.where(jnp.any(mask), jnp.argmax(mask), jnp.int32(EMPTY_ID)).astype(jnp.int32)


def pick_last_masked(k, prior, sel, mask, item_col) -> jnp.ndarray:
    """마스크가 산 **가장 뒤** items 색인 (= 보통 WAIT) — 순차 의존을 다르게 밟는 둘째 대역."""
    I = int(mask.shape[0])
    rev = jnp.argmax(mask[::-1])
    return jnp.where(jnp.any(mask), I - 1 - rev, jnp.int32(EMPTY_ID)).astype(jnp.int32)


# ───────────────────────────────────────────────── 공동 실행가능 (baselines.py:134-149)
def feasible_joint(world: BlockWorld, fl, trial, g: Geom) -> jnp.ndarray:
    """v5 `baselines._feasible_joint` — token 중복 없음 **그리고** dry_run 오라클 전원 성립. () bool.

    `trial` (K,) int32 = 크레인별 평면 열 번호 (−1 · WAIT 열은 건너뛴다).
    v5 의 결속 PREPO 중복 검사(141-146행)는 `bound_repo` 설정에서만 나는 후보라 LEGACY_DEFAULT 경로
    에는 그런 후보가 없다 — 배열 후보 생성기(`cands3`)도 PREPO 를 만들지 않으므로 항등이다.
    ★token 중복은 `reject_code` 의 DUP_JOB 이 어차피 잡지만(reservation.py:91-92), v5 가 **먼저**
      검사하는 자리라 뜻을 남겨 둔다 — 답은 같다 (시험이 둘의 일치를 못박는다).
    """
    K, C = fl.kind.shape
    t = jnp.asarray(trial, jnp.int32)
    tok = sel_tokens(fl, t)
    kk = jnp.arange(K, dtype=jnp.int32)
    dup = jnp.any((tok[:, None] == tok[None, :]) & (tok[:, None] >= 0) & (kk[:, None] < kk[None, :]))
    return (~dup) & dry_run_joint(world, fl, t, g)


# ───────────────────────────────────────────────── 전원 WAIT 예외 대체 (stage/episode.py:208-216)
def normalize_choice(fl, pr, choice, open_) -> jnp.ndarray:
    """정책 답 → `dry_run_joint` 가 읽는 열 번호 (K,) — 안 물음·WAIT·목록 밖은 −1.

    ★`flat.feasible` 은 **보지 않는다**. v5 `_apply` 는 정책이 고른 것을 곧바로 `sim.assign` 에 넘기고
      (baselines.py:168) 거기서 계획·예약을 다시 세우므로, 실린 후보이기만 하면 feasible=False 여도
      실제로 시도되고 **그때 터진다** (mandatory PLAN_FAILED 후보가 그 경우다).
      `engine_step.apply_choices` 는 목록 밖 열만 V_DECISION_COVERAGE 로 실격시키므로 같은 규약이다.
    """
    K, C = fl.kind.shape
    kk = jnp.arange(K, dtype=jnp.int32)
    c = jnp.asarray(choice, jnp.int32)
    cc = jnp.clip(c, 0, C - 1)
    in_list = (c >= 0) & (c < C) & pr.keep[kk, cc]
    work = jnp.asarray(open_, bool) & in_list & (fl.kind[kk, cc] != PK_WAIT)
    return jnp.where(work, cc, jnp.int32(EMPTY_ID)).astype(jnp.int32)


def decision_would_raise(world: BlockWorld, fl, pr, choice, open_, g: Geom) -> jnp.ndarray:
    """이 결정을 적용하면 v5 `_apply`→`assign` 이 예외를 내는가 () bool.

    v5 가 터지는 자리 둘을 그대로 본다 (engine.py:694-697): `_plan` 이 None(NOT_DISPATCHABLE) ·
    `reservations.reserve` 의 5-lock 거절. 둘 다 `dry_run_joint` 가 **크레인 순으로** 판정하는 것과
    같은 계산이다 (D-ORACLE, engine.py:744-748).
    """
    return ~dry_run_joint(world, fl, normalize_choice(fl, pr, choice, open_), g)


def first_failing_crane(world: BlockWorld, fl, pr, choice, open_, g: Geom) -> jnp.ndarray:
    """**처음 갈리는 크레인** 번호 () int32 (−1 = 전원 성립) — 진단·보고용.

    앞에서부터 하나씩 더해 가며 `dry_run_joint` 을 다시 묻는다 (단조라 처음 거짓이 되는 자리가 답).
    새 물리 계산 없음 — 오라클을 K 번 부를 뿐이다.
    """
    K = world.k
    t = normalize_choice(fl, pr, choice, open_)
    kk = jnp.arange(K, dtype=jnp.int32)
    ok = jax.vmap(lambda j: dry_run_joint(world, fl, jnp.where(kk <= j, t, jnp.int32(EMPTY_ID)), g))(kk)
    bad = ~ok
    return jnp.where(jnp.any(bad), jnp.argmax(bad), jnp.int32(EMPTY_ID)).astype(jnp.int32)


def all_wait_choice(fl, open_) -> jnp.ndarray:
    """전 크레인 WAIT (K,) int32 — v5 `_apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})`."""
    return jnp.where(jnp.asarray(open_, bool), jnp.int32(wait_col(fl)), jnp.int32(EMPTY_ID)).astype(jnp.int32)


def substitute_all_wait(fl, choice, lost, open_, failed):
    """실패하면 답 한 벌을 **전원 WAIT** 로 바꾼다 → (choice', lost').

    ★`lost` 도 함께 끈다. v5 대체 경로는 `baselines._apply`(166행)라 `yield_reason` 을 넘기지 않고,
      그래서 `recent_yield_count` 가 오르지 않는다 (resolver.apply 122행과 다르다).
    """
    K = int(jnp.asarray(open_, bool).shape[0])
    ch = jnp.where(failed, all_wait_choice(fl, open_), jnp.asarray(choice, jnp.int32))
    ls = jnp.where(failed, jnp.zeros((K,), bool), jnp.asarray(lost, bool))
    return ch.astype(jnp.int32), ls


@lru_cache(maxsize=None)
def guard_all_wait(policy_fn: Callable, g: Geom):
    """공동 규약 정책을 "실패하면 전원 WAIT" 로 감싼다 — `stage/episode.py:209-216` 의 배열판.

    서명은 그대로 `policy_fn(params, world, c3, fl, pr, open_) → (choice, lost, flags)` 라
    `engine_step.decide_joint`·`multiblock.Engine` 에 그대로 끼운다.
    ★예외 계수(`policy_exceptions`)는 여기서 세지 않는다 — `BlockWorld` 에 그 칸이 없다
      (state.py 는 다른 담당 · open_issues). 관측 거동(전 크레인 WAIT)은 완전히 재현한다.
    ★`lru_cache` — `make_resolver` 와 같은 이유로 같은 (policy_fn, g) 면 **같은 함수 객체**를 돌려준다
      (jit static 키에 id 가 들어가 매번 새 클로저면 전체 재추적이 난다).
    """
    def guarded(params, world, c3, fl, pr, open_):
        choice, lost, flags = policy_fn(params, world, c3, fl, pr, open_)
        failed = decision_would_raise(world, fl, pr, choice, open_, g)
        ch, ls = substitute_all_wait(fl, choice, lost, open_, failed)
        return ch, ls, jnp.asarray(flags, jnp.int32)
    guarded.__name__ = f"guard_all_wait({getattr(policy_fn, '__name__', 'policy')})"
    return guarded


@lru_cache(maxsize=None)
def make_seq_policy(pick_fn: Callable, g: Geom, *, k_max: int = K_MAX, guard: bool = True):
    """순차 조건부를 공동 규약 `policy_fn` 으로 — `engine_step.decide_joint` 가 부를 수 있는 형.

    `policy_fn(params, world, c3, fl, pr, open_) → (choice (K,) 열 번호, lost (K,) bool, flags ())`.
    `guard=True` 면 `guard_all_wait` 을 겹쳐 v5 `_rule_policy` 와 같은 예외 대체까지 재현한다.
    `params` 는 순차 조건부에 필요 없지만(선호 순위표를 안 쓴다) 규약을 지켜 받는다.
    """
    def policy_fn(params, world, c3, fl, pr, open_):
        out = sequential_conditional(world, fl, pr, open_, pick_fn, g, k_max=k_max)
        return out.choice, jnp.zeros((world.k,), bool), out.flags
    policy_fn.__name__ = f"seq_{getattr(pick_fn, '__name__', 'pick')}"
    return guard_all_wait(policy_fn, g) if guard else policy_fn
