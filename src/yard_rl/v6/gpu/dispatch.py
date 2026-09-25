"""순차 배정 scan — 정책을 **크레인마다 scan 안에서** 부른다 ([[YR-327]] 조각 2 · key=dispatch).

v5 정본 의미 = `integrated/dispatcher.py:19-32` `ReferenceDispatcher.run` +
`engine.py:733-736` `commit_decisions`(crane_id 정렬 순) + `679-722` `assign`:

    for cid in dp.crane_ids:                       # 정렬됨 — 앞 크레인 예약 순차 전파
        cands = sim.candidates_for(cid)            # ★live (앞 배정이 반영된 예약표로 다시 계산)
        WAIT if not cands else SERVE(select(cands))

조각 1 의 `engine_step.decide` 는 정책을 scan **앞에서 한 번** 불러 두 크레인이 같은 오더를
고르면 DUP_JOB(16) 실격이었다. 조각 2 통합으로 엔진 `assign_scan` 이 단계 k 의 carry(앞 크레인의
예약이 든 세계)로 **크레인 k 의 후보를 다시 뽑고 그때 정책을 부른다** — v5 와 같은 의미. 이 파일의
`dispatch` 는 그 scan 을 그대로 부르고 단계별 흔적(live 후보·코드·수용 코드)을 겉옷에 담아 돌려준다.

■ 단계 k (크레인 번호 순 = v5 crane_id 정렬 순 = `host_convert` 번호 규칙)
    ① live 후보  cand_k[n] = eligible[k] & dispatchable[k,n] & ~taken_live[n]
                            & plan_live(k,n).ok & reject_code(carry.res, k, n, plan_live)==0
       · dispatchable·eligible 은 결정 시작 시점 것을 그대로 쓴다 — 결정 중 바뀌는 것은
         '방금 잡힌 오더'(status RUNNING·assigned) 뿐이고 그건 taken_live 가 똑같이 가린다.
         스택·크레인 위치는 결정 중 불변(deferred commit)이라 find_slot 도 불변.
       · plan_live 는 carry.res 의 예약 칸을 제외한 계획 — v5 `_plan` 이 live `reserved_slots`
         를 읽는 것(588행)과 같다. 앞에서 수용된 배정이 하나도 없으면(any_acc 거짓) 결정 시작
         시점의 계획표 P0[k] 와 같으므로 `lax.cond` 로 재계산을 건너뛴다 (K=1 은 항상 건너뜀).
    ② 정책 호출  pick = policy_fn(params, x_k, mask_k)[k]
       · 서명은 조각 1 과 같이 (params, x (K,N,F), mask (K,N)) → (K,). **mask 는 k 행만 살아
         있고 나머지 행은 전부 거짓**, x 는 k 행만 live 계획으로 갱신한 것. 정책은 행마다 독립
         이어야 한다(행 k 의 답이 다른 행 마스크에 의존하지 않음 — first_by_id·policy.choose_batch
         모두 그렇다). 후보가 하나도 없으면 정책을 묻지 않고 WAIT (dispatcher.py:27-28).
    ③ WAIT(-1)  → yielded[k]=True (686행)
    ④ SERVE     → 계획 = plan_live(k, pick) (assign 의 `_plan` 재호출은 같은 상태의 같은 계획) →
                  reserve (697행) → 수용이면 예약·계획·크레인·오더·비용·JOB_COMPLETED push·DISPATCH 로그.
       거절은 구조상 나올 수 없지만(후보가 곧 예약 가능 집합) 방어로 V_RESERVE_REJECT(16),
       정책이 후보 밖을 고르면 V_DECISION_COVERAGE(512) 를 켜고 그 배정은 건너뛴다(조각 1 규약).

■ 결정 개방(open)·교착 술어 — `candidates`
    open[k] = any_n cand0[k,n]   (v5 `_decision_cranes` 453-464행, 결정 시작 시점)
    deadlock = ~any(assigned) & ~any(cand0) & any(blocked)      (engine.py:430-450)
    blocked[k,n] = dispatchable & ~taken & plan.ok & reject_code==CRANE_INTERFERENCE   (eligible 무관)
  탈출 결정(`_try_escape`)은 open 을 '유휴 크레인 전원'으로 **바꿔** 연다 — `open_override`.

■ 같은 답을 내기 위한 규칙
  · 실수는 전부 float64. 계획·예약 판정은 plan.py·reserve.py 가 담당(FMA 방어 포함), 여기의
    새 실수 연산은 `clock + dur` (701행 start_s + duration_s) 하나뿐이다.
  · 크레인 순서는 번호 순 scan — v5 `sorted(assignments, key=crane_id)` 와 같다.
  · 고정 크기: scan 길이 K, 단계마다 오더 N 전부를 vmap 으로 계산(마스크). 예외는 위반 비트.

■ 통합 뒤 (반박 검증 2026-09-25 완전성 렌즈: "같은 (K,N) 계산의 사본이 셋") — **사본 없음**
    `candidates` → `escape.candidate_matrices` + `deadlock_predicate`,  `dispatch` → `engine_step.assign_scan(with_trace)`,
    `decide_seq` → `engine_step.close_decision`. 이 파일은 시험·진단·조각 7 이 쓰는 **얇은 겉옷**(CandOut·DispatchOut 형)
    이고 계산은 엔진 것 하나뿐이다. 엔진 `decide` 와 `decide_seq` 의 잎 전부 동일은 그래서 항등이며, 시험이 그래도
    대조하는 것은 두 겉옷이 같은 인자를 넘기는지(open_override·check) 를 지키기 위해서다.

■ ★공동 결정 계층 (조각 3 통합 · 조각 7 의 앞부분) — `resolve_central` = v5 `resolver.CentralResolver.resolve`(48-77행)
    입력은 엔진의 공동 규약 (engine_step 머리말 ■ 두 결정 규약): 결정 시작 시점의 cands3 후보(flat_view·prune) · 물은 크레인.
    ① 쌍 = (크레인 k, 실린 후보 열 c) 중 feasible (53행 `gc.feasible`, WAIT 포함)
    ② 완전순서 키 `_pair_key` (79-82행) = (mandatory?0:1,) + 선호.rank + (kind_rank, crane_id, token, candidate_id)
         BaselinePreference.rank (resolver.py:27-33)  WAIT → (2, 0.0, "") · 그 밖 (본선?0:1, −cum_wait(외부트럭만), job_id)
         ServiceFirstSPTPreference.rank (baselines.py:34-37) = (SERVE?0:1, 소요[계획 없음 inf]) + Baseline.rank
       문자열 키(job_id · REPO 이름 "REPO:<cid>:<int(bay)>" · token) 는 호스트가 **정수 순위표**로 굽는다 (`resolver_params`).
    ③ 정렬 순으로 그리디 (59-76행): 이미 답한 크레인 건너뜀 · 토큰이 잡혔으면 DUP_JOB 거절 · 아니면 지금까지 고른 것 +
       이 쌍을 `dry_run_joint`(= engine.dry_run_commit 738-763행: 크레인 순으로 scratch 예약표에 그 kind 의 계획을 투영)
       으로 **공동** 검사 → 전원 실행가능이면 수용 (단조), 아니면 JOINT_CONFLICT 거절.
    ④ 마무리 (84-114행): 안 고른/WAIT 고른 크레인은 WAIT, 사유 NO_FEASIBLE(거절 없음)/LOST_CONTENTION(거절 있음) —
       `count_lost` 면 lost[k]=거절 있음 (resolver.apply 122행이 yield_reason 을 넘겨 yield_count 가 오른다; baselines._apply
       166행은 안 넘긴다 → 정답 궤적 Y01 구동은 count_lost=False).
    고정 길이: 쌍은 K·(k_max+1) 개까지 scan (실린 후보 ≤ k_max−1 + WAIT; mandatory 초과로 더 실리면 V_RESOLVER_TRUNC).
    dry_run 은 `lax.cond` 로 고려 대상 쌍에서만 돈다 (vmap 아래서는 전부 계산).

■ `dry_run` — v5 `dry_run_commit` (engine.py:738-763, 조각 7 resolver 의 joint-feasibility 오라클, 불변식 D-ORACLE)
    choices (K,) int32 (-1 = 그 크레인 선택 없음) 를 **크레인 번호 순**으로 scratch 예약표(carry)에 투영한다:
      plan = _plan(k, n, extra_exclude=scratch.reserved_slots)  (752행)  → 불성립이면 NO_PLAN(6)
      reason = scratch.reject_reason(plan)                      (757행)  → ≠0 이면 그 5-lock 코드
      수용이면 scratch.reserve                                    (760행)
    반환 (plans PlanOut 각 열 앞에 (K,), reasons (K,) int32: -1 없음 · 0 수용 · 1..5 거절 · 6 NO_PLAN). 세계는 안 바꾼다.
    정책을 부르지 않으므로 후보 밖 선택도 사유 코드로 나온다 (엔진 배정은 512 비트로 실격시키는 자리).
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from . import engine_step as ES
from .cands3 import K_MAX, plan_pre, plan_repo
from .engine_step import DecideOut, plan_row, tree_where
from .escape import candidate_matrices, deadlock_predicate
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .geom import Geom
from .plan import PlanOut, plan_serve
from .reserve import OK, reject_code, reserve
from .state import (PK_PRE_REHANDLE, PK_REPOSITION, PK_SERVE, PK_WAIT, PK_WAIT as _PK_WAIT, V_RESOLVER_TRUNC,
                    BlockWorld)

__all__ = ["CandOut", "DispatchOut", "NO_PLAN", "candidates", "dispatch", "decide_seq", "dry_run", "plan_row",
           "ResolverParams", "resolver_params", "repo_names", "dry_run_joint", "resolve_central", "make_resolver",
           "policy_reference"]

F = TIME_DTYPE
#: `dry_run` 사유 코드 — reserve.py 의 5-lock 코드(0..5) 뒤에 v5 'NO_PLAN' (engine.py:754) 을 잇는다
NO_PLAN = 6


# ───────────────────────────────────────────────── 후보 (결정 시작 시점)
class CandOut(NamedTuple):
    """결정 시작 시점의 (K,N) 후보 행렬과 그 재료 — v5 `_decision_cranes`·`candidates_for`·교착 술어."""

    eligible: jnp.ndarray   # (K,)  bool   idle & ~yielded (cranes.py:33 · engine.py:456, 473)
    disp: jnp.ndarray       # (K,N) bool   `_dispatchable` (494-513행)
    taken: jnp.ndarray      # (N,)  bool   `job_taken` (481행) — token_owner ≥ 0
    P: PlanOut              # (K,N) 계획 (`_jobref`+`_plan`, 483-488행) — 예약 칸 제외 = 현재 예약표
    code: jnp.ndarray       # (K,N) int32  `reject_reason` 5-lock 코드 (489행 can_reserve 의 재료)
    cand: jnp.ndarray       # (K,N) bool   eligible & disp & ~taken & P.ok & code==0
    open: jnp.ndarray       # (K,)  bool   any_n cand — `_decision_cranes`
    blocked: jnp.ndarray    # (K,N) bool   disp & ~taken & P.ok & code==CRANE_INTERFERENCE (437-450행; eligible 무관)
    deadlock: jnp.ndarray   # ()    bool   ~any(assigned) & ~any(cand) & any(blocked)  (430-436행)


def candidates(world: BlockWorld, g: Geom) -> CandOut:
    """결정 시작 시점의 후보·개방·교착 술어를 한 번에 — `escape.candidate_matrices` 의 겉옷 (계산 사본 없음)."""
    m = candidate_matrices(world, g)
    deadlock, blocked = deadlock_predicate(world, m)
    return CandOut(eligible=m.eligible, disp=m.disp, taken=m.taken, P=m.P, code=m.code, cand=m.cand,
                   open=jnp.any(m.cand, axis=1), blocked=blocked, deadlock=deadlock)


# ───────────────────────────────────────────────── 배정 scan (겉옷)
class DispatchOut(NamedTuple):
    """배정 scan 의 결과 — 세계 + 결정 요약 + 단계별 live 후보 (시험·진단·조각 7 resolver 용)."""

    world: BlockWorld
    decided: jnp.ndarray       # ()    bool   물은 크레인이 하나라도 있었나
    open: jnp.ndarray          # (K,)  bool   물은 크레인 (v5 TerminalDecision.crane_ids)
    pick: jnp.ndarray          # (K,)  int32  답 (-1 = WAIT; 안 물은 크레인도 -1)
    cand0: jnp.ndarray         # (K,N) bool   결정 시작 시점 후보 (`candidates`.cand)
    cand_live: jnp.ndarray     # (K,N) bool   단계 k 에서 크레인 k 가 본 live 후보 (안 물은 행은 False)
    taken_live: jnp.ndarray    # (K,N) bool   단계 k 시작 시 잡힌 오더
    plan_ok_live: jnp.ndarray  # (K,N) bool   단계 k 의 live 계획 성립
    code_live: jnp.ndarray     # (K,N) int32  단계 k 의 live 거절 코드 (계획 불성립 칸은 무의미)
    commit_code: jnp.ndarray   # (K,)  int32  SERVE 시 reserve 코드 (0 = 수용), WAIT·미개방·후보 밖 = -1
    deadlock: jnp.ndarray      # ()    bool   결정 시작 시점 교착 술어 (`candidates`.deadlock)
    P_live: PlanOut | None     # (K,N) 단계 k 의 live 계획 전체 (with_trace 일 때만; 아니면 None)


def dispatch(world: BlockWorld, params, g: Geom, policy_fn: Callable, *,
             open_override=None, with_trace: bool = False, lost=None) -> DispatchOut:
    """배정 scan 한 번 — `engine_step.assign_scan` 을 부른다 (머리말). 세계는 배정만 반영한 것.

    open_override (K,) bool 을 주면 그것이 물을 크레인이다 (탈출 결정: 유휴 크레인 전원, engine.py:410).
    결정 장부·last_decision_at·rate 갱신은 `decide_seq` 가 한다. g·policy_fn·with_trace 는 jit 의 static 인자.
    """
    c = candidates(world, g)
    open_ = c.open if open_override is None else jnp.asarray(open_override, bool)
    w2, pick, tr = ES.assign_scan(world, params, g, policy_fn, P0=c.P, disp=c.disp, open_=open_,
                                  lost=lost, with_trace=with_trace)
    return DispatchOut(world=w2, decided=jnp.any(open_), open=open_, pick=pick, cand0=c.cand,
                       cand_live=tr.cand_live, taken_live=tr.taken_live, plan_ok_live=tr.plan_ok_live,
                       code_live=tr.code_live, commit_code=tr.commit_code, deadlock=c.deadlock,
                       P_live=tr.P_live)


def decide_seq(world: BlockWorld, params, g: Geom, policy_fn: Callable, *,
               open_override=None, check: bool = True) -> DecideOut:
    """결정 국면 한 번 — 엔진 `decide` 와 같은 DecideOut (open 을 밖에서 주는 판).

    `dispatch` 뒤에 `engine_step.close_decision`(295-299행 결정 장부·last_decision_at·eta_armed 소진, 724-729행
    rate 갱신, 730행 불변식). 열린 크레인이 없으면 세계를 **그대로** 돌려준다.
    open_override(탈출 결정)일 때는 eta_armed 를 건드리지 않는다 — `_try_escape`(384-411행)는
    `_eta_armed` 를 소진하지 않는다(297행은 일반 결정 경로에만 있다).
    """
    d = dispatch(world, params, g, policy_fn, open_override=open_override, with_trace=False)
    w2 = ES.close_decision(d.world, d.open, d.pick, g, consume_armed=(open_override is None), check=check)
    K = world.k
    kind = jnp.where(d.open, jnp.where(d.pick >= 0, PK_SERVE, PK_WAIT), EMPTY_ID).astype(jnp.int32)
    return DecideOut(tree_where(d.decided, w2, world), d.decided, d.open, d.pick, kind, jnp.full((K,), jnp.nan, F))


# ───────────────────────────────────────────────── dry-run 오라클 (738-763행)
def dry_run(world: BlockWorld, choices, g: Geom):
    """v5 `dry_run_commit` — 임의 joint 선택을 크레인 순으로 투영해 계획·사유를 돌려준다 (머리말). 세계 불변.

    choices (K,) int32: 크레인 k 가 고른 오더 (-1 = 선택 없음 → 사유 -1, 건너뜀).
    반환 (plans: PlanOut 각 열 앞에 (K,) — 수용/거절 무관하게 그 단계의 계획, reasons (K,) int32).
    """
    K, N = world.k, world.n
    B, R, _ = world.stacks.shape
    choices = jnp.asarray(choices, jnp.int32)
    gap = jnp.asarray(g.gap, F)
    zero_ex = jnp.zeros((B, R), bool)

    def body(res, k):
        n = choices[k]
        has = n >= 0                                                     # 749-750행 None → continue
        nc = jnp.clip(n, 0, N - 1)
        P = plan_serve(world._replace(res=res), k, nc, zero_ex, g)       # 752행 _plan(extra_exclude=scratch.reserved_slots)
        code = reject_code(res, k, nc, P.lo, P.hi, P.lane, P.slots, gap)  # 757행 scratch.reject_reason
        reason = jnp.where(~has, EMPTY_ID, jnp.where(~P.ok, NO_PLAN, code)).astype(jnp.int32)
        accept = has & P.ok & (code == OK)
        res2, _ = reserve(res, k, nc, P.lo, P.hi, P.lane, P.slots, world.clock + P.dur, gap)   # 760행
        return tree_where(accept, res2, res), (P, reason)

    _, (plans, reasons) = lax.scan(body, world.res, jnp.arange(K, dtype=jnp.int32))   # 748행 sorted(choices)
    return plans, reasons


# ───────────────────────────────────────────────── 참조 정책 (ReferenceDispatcher.select) — 순차 규약용
def policy_reference(params, x, mask) -> jnp.ndarray:
    """v5 `ReferenceDispatcher.select` (dispatcher.py:14-17) 의 배열판 — min by (0 if 본선 else 1, −누적대기, 오더 번호).

    행마다 독립 (K,N) → (K,). params = (is_vessel (N,) bool, cum (N,) f64) 를 호스트가 결정 시점마다 줄 수 없으므로
    누적대기는 정책 특징 f0 = cum/3600 (**float32**) 로 읽는다 — 도착이 정수 초이거나 외부트럭이 없는 무대(정답 궤적 Y01)
    에서는 v5 와 같은 순서다. params = is_vessel (N,) bool.
    """
    ves = jnp.asarray(params, bool)
    k1 = jnp.where(ves, 0, 1).astype(jnp.int32)[None, :]
    cum = x[..., 0]
    m1 = mask & (k1 == jnp.min(jnp.where(mask, k1, 9), axis=1, keepdims=True))
    m2 = m1 & (cum == jnp.max(jnp.where(m1, cum, -jnp.inf), axis=1, keepdims=True))
    return jnp.where(jnp.any(mask, axis=1), jnp.argmax(m2, axis=1), EMPTY_ID).astype(jnp.int32)


# ───────────────────────────────────────────────── 공동 결정 계층 (머리말 ■ ★공동 결정 계층)
class ResolverParams(NamedTuple):
    """문자열 키의 정수 순위표 — 호스트 `resolver_params` 가 굽는다 (jit 안에서는 gather 만)."""

    name_rank_job: jnp.ndarray    # (N,)     int32  오더 id 의 순위 — 전체 이름 집합(오더 id ∪ 모든 REPO 이름) 안에서
    name_rank_repo: jnp.ndarray   # (K,B+1)  int32  "REPO:<cid>:<b>" 의 순위 (같은 집합) — b = int(bay) ∈ [0, B]
    tok_rank: jnp.ndarray         # (N,)     int32  token(=오더 id) 의 순위 — 오더 id 끼리 ("" 는 -1)


def repo_names(crane_ids, B: int) -> list[str]:
    """v5 REPO 후보 이름 전부 — candidates.py:413 `f"REPO:{cid}:{int(tb)}"`, bay 절사값 0..B."""
    return [f"REPO:{cid}:{b}" for cid in crane_ids for b in range(B + 1)]


def resolver_params(tables, g: Geom) -> ResolverParams:
    """host_convert.IdTables + Geom → 순위표. 오더 번호 n 은 sorted(job_id) 순위라 tok_rank[n] = n (빈 칸은 그 뒤)."""
    import numpy as np
    K, B = len(tables.crane_ids), int(g.bay_count)
    N = len(tables.job_ids)
    names = sorted(set(tables.job_ids) | set(repo_names(tables.crane_ids, B)))
    rank = {nm: i for i, nm in enumerate(names)}
    nj = np.asarray([rank[j] for j in tables.job_ids], np.int32)
    nr = np.asarray([[rank[f"REPO:{cid}:{b}"] for b in range(B + 1)] for cid in tables.crane_ids], np.int32)
    tk = np.arange(N, dtype=np.int32)
    return ResolverParams(name_rank_job=jnp.asarray(nj), name_rank_repo=jnp.asarray(nr), tok_rank=jnp.asarray(tk))


def _pad_orders(arr, n_max: int, fill):
    """(N0,) 순위표를 오더 칸 N 에 맞춘다 (빈 칸은 fill)."""
    n0 = int(arr.shape[0])
    if n0 >= n_max:
        return arr[:n_max]
    return jnp.concatenate([arr, jnp.full((n_max - n0,), fill, arr.dtype)])


def _plan_of_kind(world: BlockWorld, k, kind, job, bay, g: Geom):
    """열 하나의 kind 로 계획 — SERVE plan_serve · PRE plan_pre · REPO plan_repo (모두 계산하고 고른다)."""
    B, R_, _ = world.stacks.shape
    N = world.n
    nc = jnp.clip(job, 0, N - 1)
    is_repo = kind == PK_REPOSITION
    is_pre = kind == PK_PRE_REHANDLE
    P_s = plan_serve(world, k, nc, jnp.zeros((B, R_), bool), g)
    P_p = plan_pre(world, k, nc, g)
    P_r = plan_repo(world.cranes, k, jnp.where(jnp.isnan(bay), world.cranes.bay[k], bay), g)
    P = tree_where(is_repo, P_r, tree_where(is_pre, P_p, P_s))
    token = jnp.where(is_repo, EMPTY_ID, nc).astype(jnp.int32)
    return P, token


def dry_run_joint(world: BlockWorld, fl, trial, g: Geom) -> jnp.ndarray:
    """v5 `dry_run_commit` (738-763행) 을 kind 있는 열 선택 trial (K,) 에 — 전원(job_ref 있는 크레인) 실행가능하면 True.

    크레인 순으로: 그 kind 의 계획을 scratch 예약표(=live 예약 ∪ 앞서 수용된 것) 로 세우고 (752행 extra_exclude) →
    scratch.reject_reason (757행) → 수용이면 scratch.reserve (760행). WAIT/-1 은 건너뛴다 (750행). () bool.
    """
    K = world.k
    C = fl.raw.shape[1]
    gap = jnp.asarray(g.gap, F)

    def body(res, k):
        c = trial[k]
        cc = jnp.clip(c, 0, C - 1)
        kind = fl.kind[k, cc]
        has = (c >= 0) & (c < C) & (kind != PK_WAIT)                     # job_ref is not None
        P, token = _plan_of_kind(world._replace(res=res), k, kind, fl.job[k, cc], fl.bay[k, cc], g)
        code = reject_code(res, k, token, P.lo, P.hi, P.lane, P.slots, gap)
        acc = has & P.ok & (code == OK)
        res2, _ = reserve(res, k, token, P.lo, P.hi, P.lane, P.slots, world.clock + P.dur, gap)
        return tree_where(acc, res2, res), (~has) | acc

    _, oks = lax.scan(body, world.res, jnp.arange(K, dtype=jnp.int32))
    return jnp.all(oks)


def resolve_central(params: ResolverParams, world: BlockWorld, c3, fl, pr, open_, g: Geom, *,
                    pref: str = "baseline", count_lost: bool = True, k_max: int = K_MAX):
    """v5 `CentralResolver.resolve` (머리말 ■ ★공동 결정 계층) → (choice (K,) 열 번호 [-1 WAIT], lost (K,), flags () int32).

    pref: "baseline" (BaselinePreference) · "sf_spt" (ServiceFirstSPTPreference — 정답 궤적 Y01 의 규칙).
    """
    K, C = fl.raw.shape
    N = world.n
    B = int(g.bay_count)
    o = world.orders
    clock = world.clock
    kind = fl.kind
    is_wait = kind == PK_WAIT
    is_serve = kind == PK_SERVE
    is_pre = kind == PK_PRE_REHANDLE
    is_repo = kind == PK_REPOSITION
    jc = jnp.clip(fl.job, 0, N - 1)
    open_ = jnp.asarray(open_, bool)
    valid = pr.keep & fl.feasible & open_[:, None]                       # 53행 (결정 대상 크레인의 feasible 후보)
    # BaselinePreference.rank (resolver.py:27-33)
    ref_vessel = is_serve & o.is_vessel[jc]                              # PRE/REPO 의 JobRef.is_vessel=False
    ref_ext = (is_serve & o.is_external[jc]) | is_pre                    # PRE 의 JobRef.is_external=True
    arrived = o.is_external & (o.block_in_s < EMPTY_TIME) & (o.block_in_s <= clock)
    cum = jnp.where(arrived, clock - o.block_in_s, 0.0)                  # engine.py:258-265 cum_wait
    cum_key = jnp.where(is_wait, 0.0, jnp.where(ref_ext, -cum[jc], 0.0))
    cum_key = jnp.where(cum_key == 0.0, 0.0, cum_key)                    # −0.0 → +0.0 (파이썬 정렬은 둘을 같게 본다)
    ves_key = jnp.where(is_wait, 2, jnp.where(ref_vessel, 0, 1)).astype(jnp.int32)
    nrj = _pad_orders(params.name_rank_job, N, jnp.int32(1 << 30))
    tkr = _pad_orders(params.tok_rank, N, jnp.int32(1 << 30))
    bay_i = jnp.clip(jnp.floor(jnp.where(jnp.isnan(fl.bay), 0.0, fl.bay)).astype(jnp.int32), 0, B)
    k_idx = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32)[:, None], (K, C))
    name_key = jnp.where(is_wait, -1, jnp.where(is_repo, params.name_rank_repo[k_idx, bay_i], nrj[jc])).astype(jnp.int32)
    # _pair_key 의 꼬리 (resolver.py:79-82)
    mand_key = jnp.where(fl.mandatory, 0, 1).astype(jnp.int32)
    kind_rank = jnp.where(is_serve, 0, jnp.where(is_pre, 1, jnp.where(is_repo, 2, 3))).astype(jnp.int32)
    tok_key = jnp.where(is_serve | is_pre, tkr[jc], -1).astype(jnp.int32)   # REPO/WAIT token "" → 가장 앞
    cid_key = pr.candidate_id
    keys = [cid_key, tok_key, k_idx, kind_rank, name_key, cum_key, ves_key]
    if pref == "sf_spt":                                                 # baselines.py:34-37 (앞에 (SERVE?0:1, 소요))
        dur_key = jnp.where(fl.plan_ok, fl.dur, jnp.inf)
        dur_key = jnp.where(dur_key == 0.0, 0.0, dur_key)
        keys += [dur_key, jnp.where(is_serve, 0, 1).astype(jnp.int32)]
    elif pref != "baseline":
        raise ValueError(f"모르는 선호 {pref!r}")
    keys += [mand_key, (~valid).astype(jnp.int32)]                       # 가장 앞: mandatory · 그보다 앞: 유효 쌍 먼저
    order = jnp.lexsort(tuple(kk.reshape(-1) for kk in keys))           # (K·C,) 정렬 순 (마지막 키가 최우선)
    L = int(K) * (int(k_max) + 1)
    L = min(L, int(K) * int(C))
    n_valid = jnp.sum(valid).astype(jnp.int32)
    flags = jnp.where(n_valid > L, V_RESOLVER_TRUNC, 0).astype(jnp.int32)
    valid_f = valid.reshape(-1)
    tok_f = jnp.where((is_serve | is_pre), jc, EMPTY_ID).astype(jnp.int32).reshape(-1)   # 토큰 = 오더 번호

    def body(carry, p):
        choice, chosen, taken, rejects = carry
        p = jnp.asarray(p, jnp.int32)                                    # lexsort 는 x64 에서 int64 — scatter dtype 맞춤
        k = p // C
        c = p % C
        v = valid_f[p]
        tok = tok_f[p]
        tk = jnp.clip(tok, 0, N - 1)
        dup = v & ~chosen[k] & (tok >= 0) & taken[tk]                    # 65-67행 DUP_JOB
        consider = v & ~chosen[k] & ~dup
        trial = choice.at[k].set(c)
        ok = lax.cond(consider, lambda: dry_run_joint(world, fl, trial, g), lambda: jnp.zeros((), bool))   # 68-71행
        accept = consider & ok
        choice = jnp.where(accept, trial, choice)
        chosen = chosen.at[k].set(chosen[k] | accept)
        taken = taken.at[tk].set(taken[tk] | (accept & (tok >= 0)))
        rejects = rejects.at[k].add(jnp.where(dup | (consider & ~ok), 1, 0).astype(jnp.int32))   # 66·76행
        return (choice, chosen, taken, rejects), None

    carry0 = (jnp.full((K,), EMPTY_ID, jnp.int32), jnp.zeros((K,), bool), jnp.zeros((N,), bool),
              jnp.zeros((K,), jnp.int32))
    (choice, chosen, _, rejects), _ = lax.scan(body, carry0, order[:L])
    cc = jnp.clip(choice, 0, C - 1)
    final_wait = ~chosen | is_wait[jnp.arange(K), cc]                    # 93행 gc None 또는 WAIT
    lost = final_wait & (rejects > 0) if count_lost else jnp.zeros((K,), bool)   # 96행 LOST_CONTENTION
    return choice, lost, flags


def make_resolver(pref: str, g: Geom, *, count_lost: bool = True, k_max: int = K_MAX):
    """엔진 공동 규약의 policy_fn — `resolve_central` 을 static 인자로 묶는다 (jit/scan 안에서 호출됨).

    policy_fn(params: ResolverParams, world, c3, fl, pr, open_) → (choice, lost, flags).
    """
    def policy_fn(params, world, c3, fl, pr, open_):
        return resolve_central(params, world, c3, fl, pr, open_, g, pref=pref, count_lost=count_lost, k_max=k_max)
    policy_fn.__name__ = f"resolver_{pref}{'_lost' if count_lost else ''}"
    return policy_fn
