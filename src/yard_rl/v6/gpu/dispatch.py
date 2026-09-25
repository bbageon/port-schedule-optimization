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
from .engine_step import DecideOut, plan_row, tree_where
from .escape import candidate_matrices, deadlock_predicate
from .events import EMPTY_ID, TIME_DTYPE
from .geom import Geom
from .plan import PlanOut, plan_serve
from .reserve import OK, reject_code, reserve
from .state import BlockWorld

__all__ = ["CandOut", "DispatchOut", "NO_PLAN", "candidates", "dispatch", "decide_seq", "dry_run", "plan_row"]

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
    return DecideOut(tree_where(d.decided, w2, world), d.decided, d.open, d.pick)


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
