"""단일 블록 엔진의 한 스텝을 **순수 함수로** ([[YR-327]] 조각 1 · 명세 §7·§8·§9 — 조각 2 통합).

v5 `world/integrated/engine.py` 의 `run_until_decision`(276-336행) 한 순회가 여기서
**한 스텝**이다. v5 는 파이썬 while 루프·dict·예외로 굴러가고, 여기서는 고정 크기 배열을
받아 고정 크기 배열을 돌려주는 함수 하나(`step`)가 `lax.scan` 으로 S_max 번 돈다.

■ 한 스텝의 여섯 국면 (명세 step_structure — 서로 배타) — v5 순서: 동시각 사건 소진 → wake → 결정 → 탈출 → 종료/전진/사건
    nt = min(queue.time); alive = nt<inf; inwin = nt ≤ end+EPS      (282-287행)
    due_now = alive & inwin & (nt ≤ clock+EPS)                          (288행)
    [E 사건]  due_now                                          → 사건 하나 처리 (289행)
    [W 깨우기] ~due_now & wake 도래 (wake.consume_due_wakes)   → 로그·armed·yield 해제, **그 스텝 끝** (291-292행 continue)
    [D 결정]  ~due_now & ~W & 열린 크레인 있음 & clock < end−EPS  → 결정 (293-299행; 개방 = wake.open_with_armed)
    [X 탈출]  ~due_now & ~W & ~D & 간섭 교착 (escape.try_escape) → 유휴 전원에게 결정 (302-305행, immediate)
    wt = wake.next_wake_in_window (306-308행)
    [F 종료]  ~D & ~X & ~(alive & inwin) & wt 없음              → _finalize (323-332행)
    [A 전진]  ~D & ~X & wt 있음 & (nt 없음 | wt < nt−EPS)       → advance(wt) 만 (333-335행; 로그·rate 갱신 없음)
    [E 사건]  ~D & ~X & 그 밖                                   → 사건 하나 처리 (336행)
    [R 검토]  ~D & ~X & ~W & ~due_now & 검토 시각 ep 가 (nt 없음 | ep ≤ nt+EPS) & (wt 없음 | ep ≤ wt+EPS) (314-322행)
              → review_idx+1 · ep > clock+EPS 면 advance(ep) 만 · **그 스텝 끝** (ReviewEpoch 반환) — static `review=True`
              일 때만 산다 (조각 6 다중블록 조정자 `gpu/multiblock.py` 가 켠다; 단일 블록 시험은 v5 review_epochs=[] 와 같다).
              A·E·F 보다 **앞**이고 D·X 보다 뒤 — v5 순서 그대로 (적대검증 critical-1: 결정보다 앞이면 크레인을 놀린다).
  v5 가 "동시각 사건을 다 소진한 뒤에만" 결정을 여는 규칙이 `~due_now` 한 항으로 재현된다. W 가 발화한 스텝에서는
  결정을 열지 않는다 (v5 는 continue 뒤 소비를 다시 시도(False)한 뒤에야 결정을 본다 — 상태는 같아 답도 같지만 스텝
  경계를 v5 순회와 맞춘다). D 와 X 는 `decide` 하나가 담당한다 — 같은 (K,N) 후보 행렬
  (`escape.candidate_matrices`)을 한 번 만들어 D 의 개방과 X 의 술어가 나눠 쓴다 (명세 hard_parts ①).
  ★조각 3·4 통합 (2026-09-26): `step` 의 static 인자 `pre_advice`(정보수준 PRE_ADVICE)·`horizon_s`(결정 지평)·
    `joint`(결정 계층 — 아래 ■ 두 결정 규약). 조각 1·2 무대(W=0·D=0·pre_advice=False·joint=False)에서는 W·A 국면이
    정적으로 빠지고 개방 조건의 armed 항이 0 이라 답이 그대로다 (회귀 시험).
  ★D·X 를 한 번에 건너뛰는 지름길: 유휴·비양보 크레인이 없고 **작업 중 크레인이 있으면** v5 도 후보를 안 세고
    (`_decision_cranes` 빈 튜플) 탈출 술어 ①(430행)에서 즉시 () 다 — 그때만 행렬을 안 만든다. 전원 유휴인데
    모두 양보/고장이면 탈출이 발화할 수 있으므로(down-both·regain 무대) 행렬을 만든다.

■ ★두 결정 규약 (조각 3 통합) — static `joint`
    joint=False  **순차** (`assign_scan`, 아래) — v5 `ReferenceDispatcher.run`: 크레인 순서대로 live SERVE 후보를 다시 뽑아
                 정책을 부른다. 후보는 SERVE 뿐 (candidates_for). policy_fn(params, x, mask) → (K,) 오더 번호 (-1 WAIT).
    joint=True   **공동** (`decide_joint`·`apply_choices`) — v5 `CandidateGenerator.generate`(SERVE·PRE_REHANDLE·REPOSITION·
                 WAIT, cands3.candidates3 + flat_view + prune) 를 결정 시작 시점에 **한 번** 만들고 정책이 크레인 전원의 답을
                 한꺼번에 낸 뒤(v5 CentralResolver.resolve → dispatch.resolve_central), `baselines._apply`/`resolver.apply`
                 (크레인 순 assign 679-722행) 로 적용한다. policy_fn(params, world, c3, flat, prune, open) → (choice (K,) 열 번호
                 [-1 = WAIT], lost (K,) bool [WAIT 사유 LOST_CONTENTION → yield_count], flags () int32 [violation 에 OR]).
                 배정 시 계획은 **그 kind 로 live 재계획** (694행 `_plan(cid, ref)`): SERVE plan_serve · PRE plan_pre ·
                 REPO plan_repo → reserve(2차 방어선) → 크레인·계획·비용·큐·로그. REPO 는 오더가 없어 `cranes.assigned =
                 cands3.BUSY_NO_ORDER` (유휴 판정은 전부 `assigned<0` 이라 충분), is_loaded False, 오더 상태 불변 (699-713행).
                 PRE 는 오더 PLANNED 잔존·token = 오더 번호. 완료(`h_completed`)는 kind 로 갈린다 (915행 가드).
    X(탈출) 결정은 두 규약 모두 yielded 해제 **뒤** 후보를 본다 — joint 는 `_recandidates` 가 eligible 의존 마스크·교착
    술어·REPO 목표(`_escape_bays` 는 generate 시점의 술어를 읽는다, candidates.py:347)를 해제 뒤 상태로 다시 센다.

■ ★결정 국면 (§8 · 조각 2) — v5 `_decision_cranes`→`ReferenceDispatcher.run`(dispatcher.py:19-32)→`assign`
    결정 시작 시점 (K,N):  cand0[k,n] = eligible[k] & dispatchable[k,n] & ~taken[n] & plan_ok[k,n] & (reject_code==0)
    open[k] = any_n cand0[k,n]  (D, 460행)   또는   open = 유휴 전원 (X, 399행 esc)
    배정은 **크레인 번호 순 lax.scan** `assign_scan` (733-736행 commit_decisions 의 crane_id 정렬 순):
      단계 k ① live 후보 — 앞 크레인이 하나라도 예약했으면 carry 예약표로 **다시 계획**(plan_row) 하고
                 (694행 _plan 은 live reserved_slots 을 읽는다) reject 5-lock 을 다시 판정, 잡힌 오더는 token_owner 로 가림
              ② **정책을 그때 부른다** — mask 는 k 행만 산 (K,N), x 는 k 행만 live 계획. 후보가 없으면 묻지 않고 WAIT
                 (dispatcher.py:27-28). 정책은 행마다 독립이어야 한다 (first_by_id·policy.choose_batch 모두 그렇다).
              ③ WAIT → yielded (686행)  ④ SERVE → reserve (697행) → 크레인·오더·계획·비용·큐·로그 갱신.
      거절은 구조상 못 나오지만 방어로 violation |= 16, 후보 밖 답은 |= 512 + 건너뜀 (조각 1 규약).
    K=1 이면 조각 1 과 같은 계산(정책 1회·재계획 없음)이라 답이 그대로다. `gpu/dispatch.py` 의 `dispatch` 는
    이 함수에 **위임**한다(사본 없음) — `with_trace` 로 단계별 live 후보·코드·수용 코드를 함께 돌려준다.
    ★단계 0 은 scan **밖**에서 돈다 (반박 검증 2026-09-25 벡터화 렌즈): any_acc 가 scan carry 라 XLA 가 정적으로
      없애지 못해 vmap 아래서는 k=0 에서도 plan_row 를 계산했다(결정당 2·K·N 계획). 단계 0 은 앞 크레인이 없어
      재계획이 구조상 불가능하므로 P0 행을 그대로 쓰는 본문을 따로 추적한다 → 결정당 계획 수 K·N + (K−1)·N.
    ★`lost` (K,) bool 을 주면 WAIT 이면서 lost[k] 인 크레인의 yield_count 를 올린다 (687-688행 LOST_CONTENTION —
      조각 7 resolver 의 몫; 정본 ReferenceDispatcher 구동에서는 None 이라 항상 0).

■ ★탈출 결정 (X) 의 마무리는 D 와 다르다 — v5 297행 `_eta_armed -= idle` 은 정상 결정 경로에만 있다.
    X 는 try_escape 가 이미 escape_at=last_decision_at=clock·escape_count+1·로그를 적었고, 정책은 yielded
    해제 **뒤** 후보(EscapeOut.cand)를 본다. SERVE/WAIT 뿐인 조각 2 에서는 전원 WAIT 로 답하는 것이 보통이며,
    다음 사건이 yielded 를 지우면 같은 교착이 다시 발화한다 — v5 와 같은 동작이고 REPOSITION(조각 3)이 풀 몫.

■ ★시계 전진 advance (784-813행) — v5 와 **같은 순서·같은 반올림 횟수**로
    lo=min(clock,end), hi=min(t,end), dt=hi−lo, dt>0 일 때만:
      queue_area += dt·Σwaiting                                  (kpis.py:75 — 곱은 mul_exact)
      tail       += ov_n  (waiting 트럭을 **도착 순서**로 하나씩; ov = hi−max(lo, B+sla) > 0 만)   (77-80행)
      block_area += dt·Σin_block (n_blk>0 일 때만)                (time_contract.py:82-83)
      block_tail += ov_n  (in_block 트럭을 도착 순서로 하나씩)     (84-87행)
      terminal_area: A(진입)·O(출문) 경계로 [lo,hi] 를 조각내 **차례로** (nxt−t)·n_inside 를 더한다
                     (89-105행의 while 루프를 정렬 + 고정 길이 scan 으로 — 경계 순서·누적 순서 동일)
      truck_wait/long_wait 비용 += 장부 모드면 block_area/block_tail 증분, 아니면 queue/tail 증분 (797-805행)
      cong_area += lane_mean·dt (806-808행; lane.py:52-56);  rate 5항 × dt (812행; cost.py:55-61)
    ★증분은 v5 처럼 "적분한 뒤 − 적분 전" 으로 계산한다 — 같은 반올림.
    ★레인 혼잡은 조각 1 무대(레인 1개)에서도 0 이 아니다 — 크레인이 일하면 점유율 1.0 (반박 검증 finding 1).
    ★누적합은 `jnp.sum` 으로 모아 더하지 않는다 — v5 는 `+=` 로 하나씩 더하므로 (acc+a)+b 와
      acc+(a+b) 는 마지막 비트가 다를 수 있다. 도착 순서 = argsort(block_in_s, stable) (동시각은 오더
      번호 순 = 큐 seq 순). 시험 ⑨ 는 이 적분들을 **비트 동일(==)** 로 단언한다.
    ★★N 단·2N 단 scan 은 `unroll=ADVANCE_UNROLL`(16) 로 편다 (반박 검증 2026-09-25 벡터화 렌즈 finding 1):
      GPU 에서 스텝 시간의 ≈90% 가 이 두 직렬 scan 의 커널 왕복(3N 회 × ≈12µs)이라 스텝 시간이 N 에 비례했고
      vmap 으로 세계를 쌓아도 줄지 않았다(N=64 7–11ms · N=256 32ms · N=1024 ≈102ms, B 무관). unroll 은
      **같은 순서·같은 반올림**으로 커널 수만 N/16 로 내린다 — 실측 GPU N=64 K=2: step 6.04→2.51ms, vmap(run)
      B=8 6.5s→0.9s, 잎 전부 비트 동일(CPU·GPU). 시험 `test_advance_unroll_is_bit_identical` 이 unroll=1 판과
      == 로 고정한다. 학습 모드(float32)에서 닫힌 식으로 바꾸는 것은 별도 결정(README).

■ ★FMA(곱셈-덧셈 융합) — 플래그는 방어가 아니다 (exact.py 머리말, 반박 검증 2026-09-25)
    `XLA_FLAGS=--xla_allow_excess_precision=false` 를 켜도 XLA CPU 는 `a + dt*n` 을 한 번에 반올림해
    v5(두 번 반올림)와 block_area·truck_wait 의 마지막 비트가 갈렸다(crowded·censored 무대 3.64e-12 —
    이전 보고가 'terminal_area 합산 순서' 로 귀속한 것은 틀렸다). 유일한 방어는 optimization_barrier 로
    곱을 실체화하는 것 → 이 파일의 모든 `a*b + c` 는 `mul_exact` 를 거친다. 조각 2 이후도 같은 규약.
    GPU 는 (이 기계의 CUDA 1종에서) 융합하지 않았지만 장치별로 다를 수 있어 규약은 백엔드 무관.

■ 사건 처리기 (§7) — `lax.switch(kind, …)` 로 하나만 실행 (vmap 아래서는 전부 계산 후 select)
    0 JOB_COMPLETED  이동 M 단계 scan(remove/place) → 예약 해제·idle 장벽 → 크레인 유휴 → KPI → 오더 DONE
                     → 장부 O=C+max(0,exit) → down_pending→down → yielded 전부 해제 (887-945행)
    1 EQUIPMENT_DOWN 작업 중이면 down_pending, 아니면 down (1041-1046행; 없는 크레인 -1 은 무시 866행)
    2 EQUIPMENT_UP   down·down_pending 해제 + yielded 해제 (868-874행)
    5 BLOCK_ARRIVAL  WAITING·waiting·B=clock·(장부) in_block + yielded 해제 (847-853행)
    6/7 *_RELEASED   RELEASED + yielded 해제 (854-856, 877-881행)
    10/11            로그만 (882-883행)
    3 TRANSFER_ARRIVE · 4 STS_MOVE · 8 VESSEL_START · 9 PLAN_CHANGE — 조각 4 `gpu/vessel.py` 처리기 (948-1065행).
                     7 VESSEL_RELEASED 도 vessel.h_vessel_released (h_released 와 같은 일). 완료의 본선 훅(929-935행)은
                     h_completed 가 `vessel.on_load_completed` 로, STS 대기 적분·요율(806-811·816-817행)은 advance·
                     refresh_rates 가, clearout(1072-1080행)은 _fin 이 부른다. V_UNSUPPORTED_EVENT(2048) 는 더 켜지지 않는다.

■ 빈칸·위반 규약
    없는 번호 -1 은 `.at[-1]` 이 마지막 칸을 고치므로 **clip + where** 로 막는다 (`_set1`).
    v5 가 예외를 던지는 자리는 전부 violation 비트다 (state.py VIOLATION_NAMES):
      16 예약 거절 · 32 음수 비용 · 64 시간 역행 · 128 계획 없는 완료 · 256 스텝 소진 ·
      512 정책이 후보(cand) 밖 오더를 골랐다(v5 는 kpis.service_started KeyError — 배정을 건너뛴다) ·
      1024 큐 비었는데 작업 중(325-326행) · 2048 미지원 사건 · 4096 큐/로그 칸 부족(run 끝) ·
      8192 장부 모드에서 등록 안 된 트럭 도착(time_contract.py:61 KeyError — in_block 을 안 켠다) ·
      16384 레일 순서 뒤집힘 · 32768 레일 이웃 간격 < gap · 65536 활성 예약 쌍 자원 공유
        (`check_invariants` = v5 1107-1137행 check_invariants 의 K≥2 부분; v5 처럼 close_decision(730행)·
         사건 처리(781행) 뒤에 `check` static 플래그가 참일 때 검사한다 — 기본 참, 학습 경로는 꺼도 된다).

■ 실행 두 경로 — `run`(lax.scan · S_max 고정 · StepTrace 반환 · 동등성 시험용) 과 `run_while`
    (lax.while_loop · 배치 전부 terminal 이면 멈춤 · 흔적 없음 · 학습 경로용). vmap 아래 scan 은 끝난 세계도
    S_max 까지 매 스텝 결정·사건·종료 국면을 전부 계산한다(실사용 스텝은 S_max 의 8–30%) — while_loop 은
    배치 안에 살아 있는 세계가 없어지면 멈춘다. 두 경로의 최종 세계는 잎 전부 비트 같다(시험).

■ 나눗셈 주의 (travel.py 머리말)
    상수로 나누는 식(`Σvals/L`, `I/shift_len`)은 XLA 가 역수 곱으로 바꿀 수 있어 `exact.div_const`
    (optimization_barrier 뒤 나눗셈) 로 둔다. 조각 1 무대(L=1·K=1)에서는 값이 0 또는 x/1 이라
    영향이 없지만, 조각 2 부터 마지막 비트를 지킨다.

■ ★파이썬 3.12 `sum()` (exact.py 머리말 실측 2) — v5 가 `sum(실수 목록)` 인 자리는 `exact.sum_python`(Neumaier 보정합)
    lane.py:50 `sum(vals)/len(vals)` (lane_mean) · engine.py:835 `sum(loads)` (imbalance) — 둘 다 항이 파이썬 float.
    2항까지는 순차합과 같고(K=2·L≤2) 3항부터 갈릴 수 있다(L=3·K=3 무대가 이 경로를 밟는다).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from . import vessel as VS
from .cands3 import (BUSY_NO_ORDER, REPO_MIN_MOVE, REPO_SCORE_BASE, CandOut3, candidates3, flat_view,
                     plan_pre, plan_repo, prune, repo_targets)
from .escape import candidate_matrices, deadlock_predicate, try_escape
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE, next_event, push_event
from .exact import div_const, mul_exact, sum_python
from .geom import Geom
from .plan import PlanOut, plan_serve
from .reserve import (OK, corridor_overlaps, lane_occupancy, reject_code, reject_code_over_orders, release,
                      reserve, set_idle_position)
from .stack_ops import blockers_above, find_slot, place, rehandle_capacity_ok, remove
from .state import (C_CRANE_TRAVEL, C_EMPTY_TRAVEL, C_LONG_WAIT, C_REHANDLE, C_TRUCK_WAIT,
                    CR_HANDLING, CR_IDLE, EV_JOB_COMPLETED, JS_DONE, JS_RELEASED, JS_RUNNING,
                    JS_WAITING, LOG_DISPATCH, MV_RETRIEVE, MV_STORE, N_COST, PK_PRE_REHANDLE, PK_REPOSITION,
                    PK_SERVE, PK_WAIT,
                    RATE_IDX, V_BUSY_NO_EVENT, V_COMPLETE_NO_PLAN, V_CRANE_MIN_GAP,
                    V_CRANE_ORDER_SWAP, V_DECISION_COVERAGE, V_LEDGER_UNREGISTERED, V_NEG_COST,
                    V_OVERFLOW, V_PAIRWISE_LOCK, V_RESERVE_REJECT, V_STEPS_EXHAUSTED,
                    V_TIME_BACKWARD, V_UNSUPPORTED_EVENT, BlockWorld)
from .wake import consume_due_wakes, eta_opportunity_mask, next_wake_in_window, open_with_armed

__all__ = ["EPS", "ADVANCE_UNROLL", "V_UNSUPPORTED_EVENT", "StepTrace", "DecideOut", "AssignTrace",
           "tree_where", "ledger_mode", "lane_mean", "refresh_rates", "advance", "log_event", "h_arrival",
           "h_released", "h_completed", "h_down", "h_up", "h_noop", "dispatchable", "features",
           "first_by_id", "plan_row", "assign_scan", "check_invariants", "close_decision", "decide",
           "recandidates", "apply_choices", "decide_joint", "repo_aux",
           "step", "run", "run_jit", "run_while", "run_while_jit", "run_python", "cut", "finish",
           "ACTIVE_FEATURES"]

F = TIME_DTYPE
#: v5 `_EPS` (engine.py:35) — 시각 비교 여유
EPS = 1e-9
#: 연구 설계 원칙 2(핵심 정보 우선): 정책 특징은 f0~f3 만 살리고 나머지는 0 (명세 decision_interface ★)
ACTIVE_FEATURES = 4
#: ★advance 의 N 단·2N 단 직렬 scan 을 몇 단씩 펼치나 (머리말 ★★). 의미·반올림은 unroll 과 무관하다 —
#: 1 이면 조각 1 원판(커널 3N 회), 16 이면 커널 ≈3N/16. 컴파일 시간이 문제면 줄인다 (전부 펼치면 N=256 에서 24초).
ADVANCE_UNROLL = 16
#: h_completed 의 M 단 이동 scan 을 전부 펼치나 (M ≤ 6). 의미 무관이지만 **기본 꺼 둔다** — jax 0.11.2 XLA CPU 에서
#: ADVANCE_UNROLL=16 과 함께 켜면 fusion_compiler.cc:614 `llvm_module != nullptr` RET_CHECK 로 컴파일이 죽는다
#: (N=16·K=2 crowded 무대에서 재현; 12 이하 또는 M scan 미펼침이면 정상 — XLA 버그, 2026-09-25). GPU 이득도 M 단(≤6)이라 작다.
COMPLETE_UNROLL = False
N_FEATURES = 9


# ───────────────────────────────────────────────── 작은 도구
def tree_where(c, a, b):
    """c 면 a, 아니면 b — pytree 잎마다 where. 두 pytree 는 같은 구조여야 한다."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


def _set1(arr, i, v, valid):
    """arr[i] = v (valid 일 때만). i 가 -1 이면 `.at[-1]` 이 마지막 칸을 고치므로 clip + where."""
    ic = jnp.clip(jnp.asarray(i, jnp.int32), 0, arr.shape[0] - 1)
    return arr.at[ic].set(jnp.where(valid, jnp.asarray(v, arr.dtype), arr[ic]))


def _const_div(a, c: float):
    """`a / c` (c 는 파이썬 상수) — exact.div_const (역수 곱 변환 방지)."""
    return div_const(a, c, F)


def _i32(x):
    return jnp.asarray(x, jnp.int32)


def _as_res(template, r):
    """(조각 1 잔재 · 호환용) reserve.py 와 state.py 의 `ReservationArrays` 가 조각 2 에서 하나로 통일돼
    이제 **항등**이다 — 엔진 안에서는 더 쓰지 않는다. 시험(test_gpu_dispatch)이 항등임을 확인한다."""
    return r


def ledger_mode(o) -> jnp.ndarray:
    """시간계약 v2 장부 활성 — v5 engine.py:127-129 `_v2 = [외부트럭 & exit_travel_s is not None]`. () bool."""
    return jnp.any(o.is_external & (o.exit_travel_s >= 0.0))


# ───────────────────────────────────────────────── 레인 혼잡 · rate (815-838행)
def lane_mean(res, adj, g: Geom) -> jnp.ndarray:
    """활성 예약 레인 집합의 평균 혼잡률 — v5 `LaneNetwork.occupancy(occ)[0]` (lane.py:36-50).

    레인마다 (자기 점유 + 인접 점유 수)/(1+차수), 그 평균. L=0 이면 0.0 (42-43행).
    """
    L = int(g.n_lanes)
    if L <= 0:
        return jnp.asarray(0.0, F)
    occ = lane_occupancy(res, L)                                        # (L,) bool — engine.py:819
    deg = jnp.sum(adj, axis=1).astype(F)                                # 46행 len(adj[lid])
    nbr = jnp.sum(adj & occ[None, :], axis=1).astype(F)                 # 47-48행 인접 점유 수 (정수 합 → 정확)
    load = occ.astype(F) + nbr
    vals = load / (1.0 + deg)                                           # 49행
    return _const_div(sum_python(vals), float(L))                       # 50행 sum(vals)/len(vals) — 파이썬 3.12 보정합


def refresh_rates(world: BlockWorld, g: Geom) -> BlockWorld:
    """v5 `_refresh_rates` (815-822행) — rate 5항 [sts_wait, transfer_wait, lane_cong, interference, imbalance].

    sts_wait·transfer_wait 는 `vessel.vessel_rates` (816-818행 — 막힌 배 수 · 이송 대기 요청 수). set_rate 의
    `max(0.0, ·)` 를 그대로 둔다. imbalance = load_imbalance()/shift_len (824-838행): K<2 또는 Σload≤0 이면 0.
    """
    cr = world.cranes
    K = cr.k
    z = jnp.asarray(0.0, F)
    sts_r, tw_r = VS.vessel_rates(world.vessels, world.transfer)         # 816-818행
    lane_r = lane_mean(world.res, world.lane.adj, g)                    # 819-820행
    interf = jnp.sum(cr.yielded).astype(F)                              # 821행
    if K < 2:
        imb = z                                                         # 836행 len(loads) < 2
    else:
        loads = jnp.where(cr.assigned >= 0, jnp.maximum(0.0, cr.available_at - world.clock), 0.0)   # 833행
        total = sum_python(loads)                                       # 835행 sum(loads) — 파이썬 3.12 보정합 (항은 float)
        imb = jnp.where(total <= 0.0, 0.0, (jnp.max(loads) - jnp.min(loads)) / total)             # 836-838행
    imb_rate = _const_div(imb, g.shift_len_s)                           # 822행
    rate = jnp.stack([sts_r, tw_r, lane_r, interf, imb_rate]).astype(F)
    rate = jnp.maximum(0.0, rate)                                       # cost.py:53 set_rate
    return world._replace(cost=world.cost._replace(rate=rate))


# ───────────────────────────────────────────────── 불변식 (1107-1137행, K≥2 부분)
def check_invariants(world: BlockWorld, g: Geom) -> jnp.ndarray:
    """v5 `check_invariants` 의 크레인·예약 부분 → 위반 비트 () int32 (0 = 통과).

    ① 레일 순서·최소 간격 (1107-1121행): `rail_order[i]=k` 순열 **그대로** i 번째·i+1 번째 자리의 크레인 bay 를
       읽는다 (역순열로 읽지 말 것 — 조각 1 미결: K=2 에서 우연히 같아 못 잡는다).
         pa > pb + EPS         → CRANE_ORDER_SWAP (16384)   관통이 있었던 것
         pb − pa < gap − EPS   → CRANE_MIN_GAP (32768)
       이동 중 크레인의 bay 는 출발값(완료 시 갱신)이라 사건 경계에서 항상 유효 — v5 주석 그대로.
    ② 활성 예약 쌍별 (1123-1137행 `_assert_pairwise_resources`, i<j): 토큰 같음(≥0) · 레인 같음(≥0) ·
       통로 겹침(gap, `corridor_overlaps`) · 칸 겹침 중 하나라도 → PAIRWISE_LOCK (65536).
    K=1 이면 쌍이 없어 항상 0. 고정 길이 (K−1)·(K,K) 벡터 연산이라 vmap·jit 에 그대로 든다.
    """
    cr, res = world.cranes, world.res
    K = cr.k
    gap = jnp.asarray(g.gap, F)
    bays = cr.bay[cr.rail_order]                                        # 자리 i 의 크레인 bay
    pa, pb = bays[:-1], bays[1:]                                        # K−1 이웃 쌍 (K=1 → 빈 배열 → any=False)
    swap = jnp.any(pa > pb + EPS)                                       # 1114-1115행
    min_gap = jnp.any((pb - pa) < gap - EPS)                            # 1116-1117행
    ks = jnp.arange(K, dtype=jnp.int32)
    pair = res.active[:, None] & res.active[None, :] & (ks[:, None] < ks[None, :])
    tok = (res.token[:, None] >= 0) & (res.token[:, None] == res.token[None, :])      # TOKEN_DOUBLE
    lane = (res.lane[:, None] >= 0) & (res.lane[:, None] == res.lane[None, :])        # LANE_DOUBLE
    cor = corridor_overlaps(res.lo[:, None], res.hi[:, None], res.lo[None, :], res.hi[None, :], gap)   # CORRIDOR_OVERLAP
    slot = jnp.any(res.slots[:, None] & res.slots[None, :], axis=(2, 3))              # SLOT_DOUBLE
    pairwise = jnp.any(pair & (tok | lane | cor | slot))
    return (jnp.where(swap, V_CRANE_ORDER_SWAP, 0).astype(jnp.int32)
            | jnp.where(min_gap, V_CRANE_MIN_GAP, 0).astype(jnp.int32)
            | jnp.where(pairwise, V_PAIRWISE_LOCK, 0).astype(jnp.int32))


def _checked(world: BlockWorld, g: Geom, check: bool) -> BlockWorld:
    """check 가 참이면 violation 에 불변식 비트를 OR (v5 `if self._check: self.check_invariants()`)."""
    if not check:
        return world
    return world._replace(violation=world.violation | check_invariants(world, g))


# ───────────────────────────────────────────────── 시계 전진 (784-813행)
def _tail_accumulate(tail0, block_tail0, o, ov, unroll: int | None = None):
    """SLA 꼬리 적분 두 개를 v5 와 **같은 순서**로 하나씩 더한다 — kpis.py:76-80 · time_contract.py:84-87.

    v5 는 `_waiting`/`_in_block` dict 를 삽입 순서(= BLOCK_ARRIVAL 처리 순서)로 돌며 `overlap > 0`
    인 항만 `+=` 한다. 삽입 순서 = (block_in_s, 오더 번호) 오름차순 — 같은 시각 도착은 큐 seq
    (= 시드 순서 = 오더 번호) 순으로 꺼내지므로 stable argsort 가 그 순서다.
    고정 길이(N) scan — 마스크 밖 항은 carry 를 그대로 둔다 (`+0.0` 도 쓰지 않는다: v5 는 더하지 않는다).
    unroll (머리말 ★★): 같은 순서로 같은 덧셈을 하되 커널 수만 줄인다. None 이면 ADVANCE_UNROLL.
    """
    order = jnp.argsort(o.block_in_s, stable=True)
    add_t = o.waiting & (ov > 0)
    add_b = o.in_block & (ov > 0)

    def body(carry, idx):
        ta, tb = carry
        v = ov[idx]
        return (jnp.where(add_t[idx], ta + v, ta), jnp.where(add_b[idx], tb + v, tb)), None

    u = ADVANCE_UNROLL if unroll is None else int(unroll)
    (tail, btail), _ = lax.scan(body, (tail0, block_tail0), order, unroll=max(1, min(u, int(order.shape[0]))))
    return tail, btail


def _terminal_walk(area0, lo, hi, gate_in, gate_out, unroll: int | None = None):
    """터미널 점유 조각 적분 — v5 `TimeLedger.integrate` 의 while 루프 (time_contract.py:89-105) 그대로.

    v5 상태(A 포인터 `_a_idx`·O 힙·`_n_inside`)는 "이 구간 앞에서 소비된 경계 = 값 < lo 인 것"
    이라는 불변식으로 대신한다 (연속 호출에서 lo = 직전 hi 이고, O 는 완료 시각 이후에만 생기므로
    값 ≥ lo). 그래서
        n0      = #(A < lo) − #(O < lo)                        (구간 시작 시 안에 있는 트럭 수)
        경계    = lo ≤ 값 < hi 인 A(+1)·O(−1) 를 값 순으로   (동시각은 A 먼저 — 길이 0 조각이라 합엔 무관)
        조각    = (nxt − t)·n 을 **차례로** 더한다 (nxt > t 일 때만; 곱은 mul_exact)
        마지막  = (hi − t)·n
    2N 경계를 정렬한 뒤 고정 길이(2N) scan — 마스크 밖(값 +inf) 단계는 carry 를 그대로 둔다.
    등록 안 된 오더(A=+inf)·O 미확정(+inf) 은 경계가 아니다. 장부 없는 세계는 A 가 전부 +inf 라 0.
    unroll (머리말 ★★): 같은 순서·같은 곱셈·덧셈, 커널 수만 2N/unroll. None 이면 ADVANCE_UNROLL.
    """
    N = gate_in.shape[0]
    in_a = (gate_in >= lo) & (gate_in < hi)
    in_o = (gate_out >= lo) & (gate_out < hi)
    n0 = jnp.sum(gate_in < lo).astype(jnp.int32) - jnp.sum(gate_out < lo).astype(jnp.int32)
    vals = jnp.concatenate([jnp.where(in_a, gate_in, EMPTY_TIME), jnp.where(in_o, gate_out, EMPTY_TIME)])
    delta = jnp.concatenate([jnp.ones((N,), jnp.int32), -jnp.ones((N,), jnp.int32)])
    order = jnp.argsort(vals, stable=True)              # 값 오름차순; 동률이면 A(앞 절반) 먼저 (103행 `<=`)

    def body(carry, idx):
        area, t, n = carry
        v = vals[idx]
        valid = v < EMPTY_TIME
        piece = mul_exact(v - t, n.astype(F))           # 98행 (nxt − t) * _n_inside
        area = jnp.where(valid & (v > t), area + piece, area)
        t = jnp.where(valid & (v > t), v, t)
        n = n + jnp.where(valid, delta[idx], 0)
        return (area, t, n), None

    u = ADVANCE_UNROLL if unroll is None else int(unroll)
    (area, t, n), _ = lax.scan(body, (jnp.asarray(area0, F), jnp.asarray(lo, F), n0), order,
                               unroll=max(1, min(u, int(order.shape[0]))))
    last = mul_exact(hi - t, n.astype(F))               # nxt = hi 인 마지막 조각
    return jnp.where(hi > t, area + last, area)


def advance(world: BlockWorld, t, g: Geom) -> BlockWorld:
    """`_advance(t)` — 구간 [clock, t] 의 적분을 한 번에. 머리말 식 참조."""
    t = jnp.asarray(t, F)
    o, kp, ld = world.orders, world.kpi, world.ledger
    clock, end = world.clock, world.end_s
    backward = t < clock - EPS                                          # 785행 → V_TIME_BACKWARD
    lo = jnp.minimum(clock, end)                                        # 788행
    hi = jnp.minimum(t, end)
    go = (t > clock) & (hi > lo)                                        # 787, 789행
    dt = hi - lo
    sla = jnp.asarray(g.sla_s, F)

    # kpis.integrate (kpis.py:71-80) — S−B 대기 적분. ★곱은 mul_exact (FMA 방어), 꼬리는 도착 순서로 하나씩
    n_wait = jnp.sum(o.waiting).astype(F)
    queue_area = kp.queue_area + mul_exact(dt, n_wait)                  # 75행 dt * len(_waiting)
    ov = hi - jnp.maximum(lo, o.block_in_s + sla)                       # 77-78행 overlap
    tail_area, block_tail = _tail_accumulate(kp.tail_area, ld.block_tail, o, ov)   # 79-80행 · 84-87행

    # time_ledger.integrate (time_contract.py:77-105) — 블록 점유·꼬리·터미널 점유
    lm = ledger_mode(o)
    n_blk = jnp.sum(o.in_block).astype(F)
    block_area = jnp.where(n_blk > 0, ld.block_area + mul_exact(dt, n_blk), ld.block_area)   # 82-83행
    terminal_area = _terminal_walk(ld.terminal_area, lo, hi, o.gate_in_s, o.gate_out_s)     # 89-105행

    # 비용 truck_wait/long_wait (797-805행): 증분 = 적분 뒤 − 적분 전 (v5 와 같은 반올림)
    d_wait = jnp.where(lm, block_area - ld.block_area, queue_area - kp.queue_area)
    d_long = jnp.where(lm, block_tail - ld.block_tail, tail_area - kp.tail_area)
    # 이송 대기 적분 (806행 transfer.integrate) · STS 대기 적분 (809-811행) — vessel.integrate_vessel_wait
    ves2, tr2 = VS.integrate_vessel_wait(world.vessels, world.transfer, dt)
    # 레인 혼잡 적분 (806-808행; lane.py:52-56)
    mean = lane_mean(world.res, world.lane.adj, g)
    cong = world.lane.cong_area_s + mul_exact(mean, dt)                 # 56행 mean * dt
    # cost.advance (812행; cost.py:55-61): rate × dt 를 5항에 (COST_TERMS 순서 — 항별 독립)
    rate_add = mul_exact(world.cost.rate, dt)                           # 61행 rate * dt
    pending = world.cost.pending.at[C_TRUCK_WAIT].add(d_wait).at[C_LONG_WAIT].add(d_long)
    pending = pending.at[RATE_IDX].add(rate_add)
    episode = world.cost.episode.at[C_TRUCK_WAIT].add(d_wait).at[C_LONG_WAIT].add(d_long)
    episode = episode.at[RATE_IDX].add(rate_add)
    neg = (d_wait < 0) | (d_long < 0) | jnp.any(rate_add < 0)          # cost.py:70-71 → V_NEG_COST

    new = world._replace(
        kpi=kp._replace(queue_area=queue_area, tail_area=tail_area),
        ledger=ld._replace(block_area=block_area, block_tail=block_tail, terminal_area=terminal_area),
        lane=world.lane._replace(cong_area_s=cong),
        cost=world.cost._replace(pending=pending, episode=episode),
        vessels=ves2, transfer=tr2)
    w2 = tree_where(go, new, world)
    viol = (world.violation
            | jnp.where(backward, V_TIME_BACKWARD, 0).astype(jnp.int32)
            | jnp.where(go & neg, V_NEG_COST, 0).astype(jnp.int32))
    return w2._replace(clock=t, violation=viol)                         # 813행


# ───────────────────────────────────────────────── 로그 (engine.py:157, 845, 721)
def log_event(world: BlockWorld, t, kind, target, aux=EMPTY_ID) -> BlockWorld:
    """event_log.append((t, kind, target[, aux])). 칸이 없으면 overflow++ (조용히 버리지 않는다).

    aux (state.LogArrays.aux): DISPATCH 의 배정 내용 — 오더 번호(SERVE·PRE) 또는 `repo_aux(bay)` (REPOSITION)."""
    lg = world.log
    E = lg.capacity
    has = lg.n < E
    i = jnp.clip(lg.n, 0, E - 1)
    put = lambda arr, v: jnp.where(has, arr.at[i].set(jnp.asarray(v, arr.dtype)), arr)
    lg2 = lg._replace(t=put(lg.t, t), kind=put(lg.kind, kind), target=put(lg.target, target),
                      aux=put(lg.aux, aux), n=lg.n + jnp.where(has, 1, 0).astype(jnp.int32))
    return world._replace(log=lg2, overflow=world.overflow + jnp.where(has, 0, 1).astype(jnp.int32))


def repo_aux(bay) -> jnp.ndarray:
    """REPOSITION DISPATCH 로그의 aux = −(2 + int(bay)) — v5 payload "REPO:<cid>:<int(tb)>" (candidates.py:413) 의 bay.
    bay ≥ 1 이라 int() 절사 = floor."""
    return (-(2 + jnp.floor(jnp.asarray(bay, F)).astype(jnp.int32))).astype(jnp.int32)


# ───────────────────────────────────────────────── 처리기 (§7)
def _clear_yields(cr):
    """`_clear_yields` (840-842행)."""
    return cr._replace(yielded=jnp.zeros_like(cr.yielded))


def h_arrival(world: BlockWorld, n) -> BlockWorld:
    """BLOCK_ARRIVAL (847-853행): WAITING · kpis.truck_arrived · (장부) block_arrival · yielded 해제."""
    o = world.orders
    n = _i32(n)
    valid = (n >= 0) & (n < o.n)
    lm = ledger_mode(o)
    nc = jnp.clip(n, 0, o.n - 1)
    registered = o.gate_in_s[nc] < EMPTY_TIME                           # 장부 등록 (engine.py:131-132)
    unreg = valid & lm & ~registered                                    # time_contract.py:61 KeyError 자리
    o2 = o._replace(
        status=_set1(o.status, n, JS_WAITING, valid),                   # 849행
        waiting=_set1(o.waiting, n, True, valid),                       # 850행 kpis._waiting[job] = t
        block_in_s=_set1(o.block_in_s, n, world.clock, valid),          # 850·852행 B = ev.time (= clock)
        in_block=_set1(o.in_block, n, True, valid & lm & registered))   # 851-852행 time_contract.py:60-62
    viol = world.violation | jnp.where(unreg, V_LEDGER_UNREGISTERED, 0).astype(jnp.int32)
    return world._replace(orders=o2, cranes=_clear_yields(world.cranes), violation=viol)   # 853행


def h_released(world: BlockWorld, n) -> BlockWorld:
    """JOB_RELEASED (854-856행) · VESSEL_RELEASED (877-881행): RELEASED + yielded 해제."""
    o = world.orders
    n = _i32(n)
    valid = (n >= 0) & (n < o.n)
    return world._replace(orders=o._replace(status=_set1(o.status, n, JS_RELEASED, valid)),
                          cranes=_clear_yields(world.cranes))


def h_completed(world: BlockWorld, k) -> BlockWorld:
    """JOB_COMPLETED (887-945행) — 계획의 이동을 **여기서만** 실현한다 (deferred commit)."""
    o, cr, pl, kp = world.orders, world.cranes, world.plan, world.kpi
    K, N, M = cr.k, o.n, pl.m
    C = world.conts.c
    k = _i32(k)
    valid = (k >= 0) & (k < K)
    kc = jnp.clip(k, 0, K - 1)
    has_plan = valid & (pl.kind[kc] >= 0)                                # 888-891행 (없으면 RuntimeError)
    clock = world.clock

    # 892-900행 물리 실현 — M 단계 고정 scan. inbound=place · depart=remove · blocker=remove→place
    n_moves = jnp.where(has_plan, pl.n_moves[kc], 0)
    mv_cont, mv_kind, mv_dst = pl.mv_cont[kc], pl.mv_kind[kc], pl.mv_dst[kc]

    def mv_step(carry, i):
        stacks, conts, v = carry
        live = i < n_moves
        c, kind, dst = mv_cont[i], mv_kind[i], mv_dst[i]
        s1, c1, v1 = remove(stacks, conts, c)
        do_rm = live & (kind != MV_STORE)                                # 반출·재배치는 먼저 뺀다
        stacks, conts = tree_where(do_rm, (s1, c1), (stacks, conts))
        v = v | jnp.where(do_rm, v1, 0).astype(jnp.int32)
        s2, c2, v2 = place(stacks, conts, c, dst[0], dst[1])
        do_pl = live & (kind != MV_RETRIEVE)                             # 반입·재배치는 쌓는다
        stacks, conts = tree_where(do_pl, (s2, c2), (stacks, conts))
        v = v | jnp.where(do_pl, v2, 0).astype(jnp.int32)
        return (stacks, conts, v), None

    (stacks, conts, mviol), _ = lax.scan(mv_step, (world.stacks, world.conts, jnp.int32(0)),
                                        jnp.arange(M, dtype=jnp.int32), unroll=bool(COMPLETE_UNROLL))   # M 단 (COMPLETE_UNROLL 주석)

    # 901-903행 예약 해제 · 위치 · idle 장벽
    res = release(world.res, kc)
    res = set_idle_position(res, kc, pl.end_bay[kc])
    is_serve = pl.kind[kc] == PK_SERVE                                   # 915행 kind 가드 — PRE_REHANDLE(대상 잔존·오더 미완료)
    n = pl.job[kc]                                                       #   · REPOSITION(오더 없음, job=-1) 은 오더를 건드리지 않는다
    nc = jnp.clip(n, 0, N - 1)
    nvalid = is_serve & (n >= 0) & (n < N)
    is_ext = o.is_external[nc]
    # 904-910행 크레인
    cr2 = cr._replace(
        bay=cr.bay.at[kc].set(pl.end_bay[kc]), row=cr.row.at[kc].set(pl.end_row[kc]),   # 902행
        assigned=cr.assigned.at[kc].set(EMPTY_ID), status=cr.status.at[kc].set(CR_IDLE),
        available_at=cr.available_at.at[kc].set(clock), is_loaded=cr.is_loaded.at[kc].set(False),
        completions=cr.completions.at[kc].add(1),
        loaded_m=cr.loaded_m.at[kc].add(pl.loaded_m[kc]), empty_m=cr.empty_m.at[kc].add(pl.empty_m[kc]),
        served=cr.served.at[kc].add(jnp.where(is_serve, 1, 0).astype(jnp.int32)),      # 916행
        down=cr.down.at[kc].set(cr.down[kc] | cr.down_pending[kc]),                     # 943-944행
        down_pending=cr.down_pending.at[kc].set(False))
    cr2 = _clear_yields(cr2)                                             # 945행
    # 911-912행 KPI · 922-923행 job_completed (kpis.py:90-96)
    deadline = o.deadline_s[nc]
    late = nvalid & ~is_ext & (deadline < EMPTY_TIME) & (clock > deadline)
    kp2 = kp._replace(
        loaded_m=kp.loaded_m + pl.loaded_m[kc], empty_m=kp.empty_m + pl.empty_m[kc],
        rehandles=kp.rehandles + pl.rehandles[kc],
        completed_ext=kp.completed_ext + jnp.where(nvalid & is_ext, 1, 0).astype(jnp.int32),
        completed_ves=kp.completed_ves + jnp.where(nvalid & ~is_ext, 1, 0).astype(jnp.int32),
        vessel_delay_s=kp.vessel_delay_s + jnp.where(late, clock - deadline, 0.0))
    # 917-928행 오더 DONE · C · 재조작 · 장부 O = C + max(0, exit) (time_contract.py:67-74)
    lm = ledger_mode(o)
    exit_s = o.exit_travel_s[nc]
    gate = nvalid & lm & is_ext & (exit_s >= 0.0)
    o2 = o._replace(
        status=_set1(o.status, n, JS_DONE, nvalid),
        done_s=_set1(o.done_s, n, clock, nvalid),
        rehandles=_set1(o.rehandles, n, pl.rehandles[kc], nvalid),
        gate_out_s=_set1(o.gate_out_s, n, clock + jnp.maximum(0.0, exit_s), gate),
        in_block=_set1(o.in_block, n, False, gate))
    # 888행 pop — 계획 칸 비움
    pl2 = pl._replace(kind=pl.kind.at[kc].set(EMPTY_ID), job=pl.job.at[kc].set(EMPTY_ID),
                      n_moves=pl.n_moves.at[kc].set(0))
    new = world._replace(stacks=stacks, conts=conts, res=res, cranes=cr2, kpi=kp2, orders=o2, plan=pl2)
    new = VS.on_load_completed(new, n, nvalid)                           # 929-932행 적하 반출 완료 → 이송 요청 (조각 4)
    w2 = tree_where(has_plan, new, world)
    viol = (world.violation
            | jnp.where(valid & ~has_plan, V_COMPLETE_NO_PLAN, 0).astype(jnp.int32)
            | jnp.where(has_plan, mviol, 0).astype(jnp.int32))
    return w2._replace(violation=viol)


def h_down(world: BlockWorld, k) -> BlockWorld:
    """EQUIPMENT_DOWN (865-867, 1041-1046행): 작업 중이면 down_pending(비선점), 아니면 down."""
    cr = world.cranes
    k = _i32(k)
    valid = (k >= 0) & (k < cr.k)                                        # 866행 없는 id 는 무시
    kc = jnp.clip(k, 0, cr.k - 1)
    busy = cr.assigned[kc] >= 0
    cr2 = cr._replace(down_pending=cr.down_pending.at[kc].set(cr.down_pending[kc] | (valid & busy)),
                      down=cr.down.at[kc].set(cr.down[kc] | (valid & ~busy)))
    return world._replace(cranes=cr2)


def h_up(world: BlockWorld, k) -> BlockWorld:
    """EQUIPMENT_UP (868-874행): down·down_pending 해제 + yielded 해제 (아는 크레인일 때만)."""
    cr = world.cranes
    k = _i32(k)
    valid = (k >= 0) & (k < cr.k)
    kc = jnp.clip(k, 0, cr.k - 1)
    cr2 = cr._replace(down=cr.down.at[kc].set(cr.down[kc] & ~valid),
                      down_pending=cr.down_pending.at[kc].set(cr.down_pending[kc] & ~valid),
                      yielded=jnp.where(valid, jnp.zeros_like(cr.yielded), cr.yielded))
    return world._replace(cranes=cr2)


def h_noop(world: BlockWorld, _) -> BlockWorld:
    """HORIZON · ETA_UPDATED (882-883행) — 로그만."""
    return world


def _handlers(g: Geom):
    """kind 0..11 → 처리기 (w, target) → w. lax.switch 의 branch 목록. 3/4/7/8/9 는 조각 4 `gpu/vessel.py`."""
    return [
        lambda w, t: h_completed(w, t),      # 0 JOB_COMPLETED
        h_down,                              # 1 EQUIPMENT_DOWN
        h_up,                                # 2 EQUIPMENT_UP
        VS.h_transfer_arrive,                # 3 TRANSFER_ARRIVE (1008-1022행)
        VS.h_sts_move,                       # 4 STS_MOVE (966-989행)
        h_arrival,                           # 5 BLOCK_ARRIVAL
        h_released,                          # 6 JOB_RELEASED
        VS.h_vessel_released,                # 7 VESSEL_RELEASED (877-881행)
        VS.h_vessel_start,                   # 8 VESSEL_START (948-957행)
        VS.h_plan_change,                    # 9 PLAN_CHANGE (1048-1065행; 모르는 선박 -1 은 무동작)
        h_noop,                              # 10 ETA_UPDATED
        h_noop,                              # 11 HORIZON
    ]


# ───────────────────────────────────────────────── 결정 (§8)
def dispatchable(world: BlockWorld, g: Geom) -> jnp.ndarray:
    """v5 `_dispatchable(j, crane_id)` (494-513행) 를 (K,N) 마스크로.

    status ∈ {RELEASED, WAITING} & 미배정 & (대상 있음 → 야드에 있고 가용 · 담당 구간 안 ·
    재조작 칸 충분) & (STORE → 제외 없는 find_slot 성공, 기준점 크레인 현위치 510-511행).
    """
    o, cr, st, ct = world.orders, world.cranes, world.stacks, world.conts
    B, R, T = st.shape
    C = ct.c
    base = (((o.status == JS_WAITING) | (o.status == JS_RELEASED))      # 495행
            & (o.assigned_crane < 0) & (o.block >= 0))                   # 497행 · 빈 칸
    has_t = o.target_cont >= 0
    tcc = jnp.clip(o.target_cont, 0, C - 1)
    t_ok = ct.c_alive[tcc] & ct.c_avail[tcc]                             # 501-503행
    tb, trow = ct.c_bay[tcc], ct.c_row[tcc]
    blk_fn = jax.vmap(partial(blockers_above, g=g), in_axes=(None, None, None, None, None, 0))
    _, n_block = blk_fn(st.grid, st.height, ct.c_bay, ct.c_row, ct.c_tier, o.target_cont)   # (N,)
    zero_ex = jnp.zeros((B, R), bool)

    def per_kn(bmin, bmax, kbay, krow, tb_, tr_, nb_, has_t_, t_ok_, is_store_, size_):
        in_range = (bmin <= tb_) & (tb_ <= bmax)                         # 504행
        cap = rehandle_capacity_ok(st.height, st.top_size, tb_, tr_, nb_, bmin, bmax, g)   # 506행
        tgt_ok = jnp.where(has_t_, t_ok_ & in_range & cap, True)
        found, _, _ = find_slot(st.height, st.top_size, zero_ex, size_, bmin, bmax, kbay, krow, g)   # 508-512행
        return tgt_ok & jnp.where(is_store_, found, True)

    over_n = jax.vmap(per_kn, in_axes=(None, None, None, None, 0, 0, 0, 0, 0, 0, 0))
    over_kn = jax.vmap(over_n, in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None))
    m = over_kn(cr.bay_min, cr.bay_max, cr.bay, cr.row, tb, trow, n_block, has_t, t_ok,
                o.is_store, o.inbound_size)
    return base[None, :] & m


def features(world: BlockWorld, P: PlanOut, g: Geom, *, n_active: int = ACTIVE_FEATURES) -> jnp.ndarray:
    """정책 입력 x (K,N,9) float32 — 명세 decision_interface 의 특징 9칸. 실현 미래(actual_*)는 안 읽는다.

    n_active 뒤의 칸은 0 으로 둔다(연구 설계 원칙 2 — 핵심 정보 f0~f3 먼저). 동등성 시험은 특징을 안 쓴다.
    """
    o = world.orders
    K, N = P.dur.shape
    clock = world.clock
    arrived = o.is_external & (o.block_in_s < EMPTY_TIME)
    cum = jnp.where(arrived, jnp.maximum(0.0, clock - o.block_in_s), 0.0)      # engine.py:258-265 cum_wait
    h = 3600.0
    f0 = jnp.broadcast_to((cum / h)[None, :], (K, N))
    f1 = jnp.where(P.ok, P.dur / h, 0.0)
    f2 = jnp.where(P.ok, P.rehandles.astype(F), 0.0)
    f3 = jnp.where(P.ok, P.empty_m / (float(g.bay_count) * float(g.bay_len)), 0.0)
    f4 = jnp.broadcast_to(o.is_store.astype(F)[None, :], (K, N))
    f5 = jnp.broadcast_to(o.is_vessel.astype(F)[None, :], (K, N))
    dl = jnp.where(o.deadline_s < EMPTY_TIME, jnp.clip((o.deadline_s - clock) / h, -2.0, 24.0), 0.0)
    f6 = jnp.broadcast_to(dl[None, :], (K, N))
    f7 = jnp.broadcast_to((o.is_external & (cum >= 0.8 * g.sla_s)).astype(F)[None, :], (K, N))
    eta = jnp.where(o.provided_eta_s < EMPTY_TIME, jnp.clip((o.provided_eta_s - clock) / h, -2.0, 24.0), 0.0)
    f8 = jnp.broadcast_to(eta[None, :], (K, N))
    x = jnp.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8], axis=-1)
    keep = (jnp.arange(N_FEATURES) < int(n_active)).astype(F)
    return (x * keep).astype(jnp.float32)


def first_by_id(params, x, mask) -> jnp.ndarray:
    """시험 정책 — 후보 중 **번호가 가장 작은** 오더 (v5 candidates_for 첫 후보), 없으면 -1(WAIT). (K,) int32."""
    return jnp.where(jnp.any(mask, axis=1), jnp.argmax(mask, axis=1), EMPTY_ID).astype(jnp.int32)


class DecideOut(NamedTuple):
    world: BlockWorld
    decided: jnp.ndarray   # () bool  열린 크레인이 하나라도 있었나
    open: jnp.ndarray      # (K,) bool 이번 결정에서 물은 크레인 (v5 TerminalDecision.crane_ids)
    pick: jnp.ndarray      # (K,) int32 답 — 오더 번호 (SERVE·PRE_REHANDLE); -1 = WAIT/REPOSITION/안 물음
    kind: jnp.ndarray      # (K,) int32 답의 종류 PK_* (안 물은 크레인 -1) — 조각 3
    bay: jnp.ndarray       # (K,) f64   REPOSITION 목표 bay (그 밖 NaN) — 조각 3


def plan_row(world: BlockWorld, k, g: Geom) -> PlanOut:
    """크레인 k 의 오더 N 전부 계획 — 각 열 앞에 (N,). 예약 칸 제외 = `world.res` 의 **현재** 예약표
    (694행 `_plan` 이 live `reserved_slots` 를 읽는 것). 배정 scan 단계 k 의 live 재계획에 쓴다."""
    N = world.n
    B, R, _ = world.stacks.shape
    f = jax.vmap(partial(plan_serve, g=g), in_axes=(None, None, 0, None))
    return f(world, jnp.asarray(k, jnp.int32), jnp.arange(N, dtype=jnp.int32), jnp.zeros((B, R), bool))


def _row(P: PlanOut, i) -> PlanOut:
    """(K,…) 계획 묶음의 i 번째 행 (또는 (N,…) 묶음의 i 번째 오더)."""
    return jax.tree_util.tree_map(lambda a: a[i], P)


def _expand(P: PlanOut) -> PlanOut:
    return jax.tree_util.tree_map(lambda a: a[None], P)


class AssignTrace(NamedTuple):
    """배정 scan 의 단계별 흔적 (시험·진단·조각 7 resolver 용) — `assign_scan(with_trace=True)`."""

    cand_live: jnp.ndarray     # (K,N) bool   단계 k 에서 크레인 k 가 본 live 후보 (안 물은 행은 False)
    taken_live: jnp.ndarray    # (K,N) bool   단계 k 시작 시 잡힌 오더
    plan_ok_live: jnp.ndarray  # (K,N) bool   단계 k 의 live 계획 성립
    code_live: jnp.ndarray     # (K,N) int32  단계 k 의 live 거절 코드 (계획 불성립 칸은 무의미)
    commit_code: jnp.ndarray   # (K,)  int32  SERVE 시 reserve 코드 (0 = 수용), WAIT·미개방·후보 밖 = -1
    P_live: PlanOut | None     # (K,N) 단계 k 의 live 계획 전체 (with_trace 일 때만; 아니면 None)


def assign_scan(world: BlockWorld, params, g: Geom, policy_fn: Callable, *, P0: PlanOut, disp, open_,
                lost=None, with_trace: bool = False):
    """배정 scan — v5 `ReferenceDispatcher.run`(dispatcher.py:19-32) + `commit_decisions`(733-736) + `assign`(679-722)
    의 한 결정. 정책은 **크레인마다 scan 안에서** 부른다 (머리말 ★결정 국면).

    P0 (K,N) 결정 시작 시점 계획표 · disp (K,N) `_dispatchable` · open_ (K,) 물을 크레인 (D 는 any(cand0), X 는 유휴 전원).
    lost (K,) bool | None — WAIT 이면서 lost[k] 면 yield_count+1 (687-688행). with_trace 면 P_live 까지 돌려준다.
    eligible 은 이 세계(탈출이면 yielded 해제 **뒤**)에서 다시 읽는다.
    반환 (세계', pick (K,) int32 · -1 = WAIT/안 물음, AssignTrace). 결정 장부·rate 는 `close_decision` 이 한다.
    단계 0 은 scan 밖(재계획 없음 — 머리말 ★), 단계 1..K−1 은 scan (K=1 이면 길이 0 scan).
    """
    K, N = world.k, world.n
    cr = world.cranes
    eligible = (cr.assigned < 0) & ~cr.down & ~cr.yielded                # 473행 (탈출이면 해제 뒤 값)
    x0 = features(world, P0, g)
    n_idx = jnp.arange(N, dtype=jnp.int32)
    gap = jnp.asarray(g.gap, F)

    def body(carry, k, *, replan: bool):
        w, any_acc = carry
        a = open_[k]
        # ① live 후보 — 앞 크레인이 하나라도 예약했으면 다시 계획 (예약 칸 제외가 자랐다), 아니면 P0 행 그대로.
        #   단계 0 (replan=False) 은 앞 크레인이 없어 P0 행이 곧 live 계획이다 — cond 자체를 두지 않는다.
        if replan:
            P_k = lax.cond(any_acc, lambda: plan_row(w, k, g), lambda: _row(P0, k))
        else:
            P_k = _row(P0, k)
        taken = w.res.token_owner >= 0                                   # 481행 job_taken (앞 크레인이 잡은 것 포함)
        code_k = reject_code_over_orders(w.res, k, n_idx, P_k.lo, P_k.hi, P_k.lane, P_k.slots, gap)   # 489행
        cand_k = a & eligible[k] & disp[k] & ~taken & P_k.ok & (code_k == OK)
        has = jnp.any(cand_k)
        # ② 정책 — k 행만 산 (K,N) 마스크, x 는 k 행만 live 계획으로. 후보 없으면 묻지 않고 WAIT (dispatcher.py:27-28)
        mask = jnp.zeros((K, N), bool).at[k].set(cand_k)
        x_k = x0.at[k].set(features(w, _expand(P_k), g)[0])
        pick_all = jnp.asarray(policy_fn(params, x_k, mask), jnp.int32)
        n = jnp.where(a & has, pick_all[k], EMPTY_ID)
        # ③ WAIT / ④ SERVE — `assign` (679-722행). carry 는 세계 전체 (예약표가 앞 크레인을 반영)
        wait = a & (n < 0)
        nc = jnp.clip(n, 0, N - 1)
        # ★정책이 후보 밖 오더를 고르면 실격 비트 512 를 켜고 그 배정은 **건너뛴다** — v5 는 같은 배정에서
        #   kpis.service_started 의 `_waiting.pop` KeyError 로 죽는다 (아직 안 온 트럭을 DISPATCH).
        off = a & (n >= 0) & ~cand_k[nc]
        serve = a & (n >= 0) & cand_k[nc]
        cr_, o_, pl_ = w.cranes, w.orders, w.plan
        lost_k = jnp.zeros((), bool) if lost is None else jnp.asarray(lost, bool)[k]
        cr_y = cr_._replace(yielded=cr_.yielded.at[k].set(cr_.yielded[k] | wait),          # 686행
                            yield_count=cr_.yield_count.at[k].add(jnp.where(wait & lost_k, 1, 0).astype(jnp.int32)))   # 687-688행
        P2 = _row(P_k, nc)                                               # 694행 재계획 = 같은 상태의 같은 계획 (deferred commit)
        rel = w.clock + P2.dur                                           # 701행 start_s + duration_s
        res2, code2 = reserve(w.res, k, nc, P2.lo, P2.hi, P2.lane, P2.slots, rel, gap)   # 697행 2차 방어선
        ok = serve & P2.ok & (code2 == OK)
        viol = (w.violation
                | jnp.where(serve & ~ok, V_RESERVE_REJECT, 0).astype(jnp.int32)      # 695-697행 예외
                | jnp.where(off, V_DECISION_COVERAGE, 0).astype(jnp.int32))
        pl2 = pl_._replace(                                              # 698행 _active_plans[crane] = plan
            kind=pl_.kind.at[k].set(PK_SERVE), job=pl_.job.at[k].set(nc),
            lo=pl_.lo.at[k].set(P2.lo), hi=pl_.hi.at[k].set(P2.hi), dur=pl_.dur.at[k].set(P2.dur),
            end_bay=pl_.end_bay.at[k].set(P2.end_bay), end_row=pl_.end_row.at[k].set(P2.end_row),
            rehandles=pl_.rehandles.at[k].set(P2.rehandles),
            loaded_m=pl_.loaded_m.at[k].set(P2.loaded_m), empty_m=pl_.empty_m.at[k].set(P2.empty_m),
            n_moves=pl_.n_moves.at[k].set(P2.n_moves), start_s=pl_.start_s.at[k].set(w.clock),
            mv_cont=pl_.mv_cont.at[k].set(P2.mv_cont), mv_src=pl_.mv_src.at[k].set(P2.mv_src),
            mv_dst=pl_.mv_dst.at[k].set(P2.mv_dst), mv_kind=pl_.mv_kind.at[k].set(P2.mv_kind))
        cr2 = cr_y._replace(                                             # 699-702행
            assigned=cr_.assigned.at[k].set(nc), status=cr_.status.at[k].set(CR_HANDLING),
            available_at=cr_.available_at.at[k].set(rel), is_loaded=cr_.is_loaded.at[k].set(True))
        is_ext = o_.is_external[nc]
        o2 = o_._replace(                                                # 706-713행 (SERVE)
            status=o_.status.at[nc].set(JS_RUNNING), assigned_crane=o_.assigned_crane.at[nc].set(k),
            service_s=o_.service_s.at[nc].set(w.clock),
            waiting=o_.waiting.at[nc].set(jnp.where(is_ext, False, o_.waiting[nc])),        # kpis.py:51-53
            wait_sample_s=o_.wait_sample_s.at[nc].set(
                jnp.where(is_ext, w.clock - o_.block_in_s[nc], o_.wait_sample_s[nc])))
        reh = P2.rehandles.astype(F)                                     # 718행 float(plan.rehandles)
        acc = lambda v: (v.at[C_CRANE_TRAVEL].add(P2.loaded_m).at[C_EMPTY_TRAVEL].add(P2.empty_m)
                         .at[C_REHANDLE].add(reh))                       # 714-719행
        cost2 = w.cost._replace(pending=acc(w.cost.pending), episode=acc(w.cost.episode))
        q2 = push_event(w.queue, rel, EV_JOB_COMPLETED, k)               # 720행
        w_ok = w._replace(res=res2, plan=pl2, cranes=cr2, orders=o2, cost=cost2, queue=q2)
        w_ok = log_event(w_ok, w.clock, LOG_DISPATCH, k, nc)             # 721행 (aux = 오더 번호)
        w_no = w._replace(cranes=cr_y)
        w2 = tree_where(ok, w_ok, w_no)._replace(violation=viol)
        ys = (n, cand_k, taken, P_k.ok, code_k, jnp.where(serve, code2, EMPTY_ID).astype(jnp.int32),
              P_k if with_trace else None)
        return (w2, any_acc | ok), ys

    # 734행 crane_id 정렬 순 — 단계 0 은 밖에서(재계획 불가), 1..K−1 은 scan (K=1 이면 길이 0)
    carry0, ys0 = body((world, jnp.zeros((), bool)), jnp.int32(0), replan=False)
    (w2, _), ys_rest = lax.scan(partial(body, replan=True), carry0, jnp.arange(1, K, dtype=jnp.int32))
    ys = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b], axis=0), ys0, ys_rest)
    pick, cand_live, taken_live, ok_live, code_live, commit_code, P_live = ys
    trace = AssignTrace(cand_live=cand_live, taken_live=taken_live, plan_ok_live=ok_live, code_live=code_live,
                        commit_code=commit_code, P_live=P_live)
    return w2, pick, trace


def close_decision(world: BlockWorld, open_, pick, g: Geom, *, consume_armed, check: bool = True,
                   kind=None, bay=None) -> BlockWorld:
    """결정 마무리 — 결정 장부(295-296행 `_pending`·`_assigned`)·`last_decision_at`(298행)·
    eta_armed 소진(297행, **정상 결정만** — consume_armed)·`close_decision` 의 `_refresh_rates`(729행)·
    check 면 불변식 검사(730-731행 → violation 비트).
    kind/bay (K,) 를 주면(공동 규약) 장부의 act_kind/act_bay 에 그대로, 없으면 SERVE/WAIT (순차 규약)."""
    K = world.k
    if kind is None:
        kind = jnp.where(open_, jnp.where(pick >= 0, PK_SERVE, PK_WAIT), EMPTY_ID).astype(jnp.int32)
    if bay is None:
        bay = jnp.full((K,), jnp.nan, F)
    dec = world.decision._replace(
        pending=open_, answered=open_, act_kind=jnp.asarray(kind, jnp.int32),
        act_job=pick, act_bay=jnp.asarray(bay, F))
    armed = jnp.where(consume_armed, world.wake.eta_armed & ~open_, world.wake.eta_armed)   # 297행 (탈출은 안 건드림)
    w2 = world._replace(last_decision_at=world.clock,                    # 298행 (탈출은 404행이 이미 같은 값)
                        decision=dec, wake=world.wake._replace(eta_armed=armed))
    return _checked(refresh_rates(w2, g), g, check)                      # 729행 · 730-731행


def decide(world: BlockWorld, params, g: Geom, policy_fn: Callable, *, check: bool = True,
           pre_advice: bool = False, horizon_s: float = 0.0, joint: bool = False) -> DecideOut:
    """결정 국면 한 번 = D(정상, 293-299행) 또는 X(탈출, 302-305행) — 둘은 배타이고 같은 (K,N) 행렬을 쓴다.

    D: open = `_decision_cranes` (453-464행) = eligible & (SERVE 후보 있음 | (armed & eta_opportunity)) —
       `wake.open_with_armed`; armed 항은 pre_advice 일 때만 산다 (eta_opportunity_mask).
    X: D 가 안 열렸을 때 `escape.try_escape` 가 발화하면 open = 유휴 전원 (yielded 해제 뒤).
    둘 다 `assign_scan` → `close_decision`. 아무것도 안 열리면 세계를 그대로 돌려준다 (decided=False) — 단, 술어까지
    참인데 유휴가 없으면 v5 398행처럼 yielded 만 해제된 세계. joint=True 면 `decide_joint` (머리말 ■ 두 결정 규약).
    """
    if joint:
        return decide_joint(world, params, g, policy_fn, check=check, pre_advice=pre_advice, horizon_s=horizon_s)
    K = world.k
    m = candidate_matrices(world, g)                                     # (K,N) 한 번 — D 개방 + X 술어 (hard_parts ①)
    eta_opp = eta_opportunity_mask(world, g, horizon_s=horizon_s, pre_advice=pre_advice)   # candidates.py:154-170
    open_D = open_with_armed(m.cand, m.eligible, world.wake.eta_armed, eta_opp)           # 453-464행
    is_D = jnp.any(open_D) & (world.clock < world.end_s - EPS)           # 294행
    esc = try_escape(world, m)                                           # 302-305행 — deadlock 은 any(cand) 와 배타
    fired = ~is_D & esc.fired
    w_in = tree_where(is_D, world, esc.world)                            # X: yielded 해제·표식·로그가 반영된 세계
    open_ = jnp.where(is_D, open_D, esc.open)
    decided = is_D | fired
    w2, pick, _ = assign_scan(w_in, params, g, policy_fn, P0=m.P, disp=m.disp, open_=open_)
    w2 = close_decision(w2, open_, pick, g, consume_armed=is_D, check=check)
    kind = jnp.where(open_, jnp.where(pick >= 0, PK_SERVE, PK_WAIT), EMPTY_ID).astype(jnp.int32)
    return DecideOut(tree_where(decided, w2, w_in), decided, open_, pick, kind, jnp.full((K,), jnp.nan, F))


# ───────────────────────────────────────────────── 공동 결정 규약 (조각 3 · cands3) — 머리말 ■ 두 결정 규약
def recandidates(c3: CandOut3, world: BlockWorld, elig, g: Geom, *, horizon_s, pre_advice: bool) -> CandOut3:
    """`candidates3` 결과에서 **eligible 에 의존하는 것만** 다른 eligible 로 다시 센다 — 탈출(X) 결정용.

    v5 X 경로: `_try_escape` 가 yielded 를 전부 지운 **뒤** 정책이 generate() 를 부른다 (candidates.py:263 eligible ·
    347행 `_escape_bays` 의 술어 `interference_deadlock_corridors` 도 그 시점). 계획·거절 코드·게이트는 yielded 와 무관하므로
    그대로 두고 원시 마스크(serve/pre/repo raw)·교착 술어·REPO 목표 집합(escape 목표가 술어에 달렸다)·개방만 다시 계산한다.
    elig == c3.eligible 이면 결과는 c3 와 같다 (같은 함수·같은 입력).
    """
    m = c3.m
    o, cr = world.orders, world.cranes
    K, N = world.k, world.n
    B, R_, _ = world.stacks.shape
    gap = jnp.asarray(g.gap, F)
    e = jnp.asarray(elig, bool)
    has_ref = (o.target_cont >= 0) | o.is_store
    serve_raw = e[:, None] & m.disp & has_ref[None, :] & (m.P.ok | c3.mandatory[None, :])
    serve = e[:, None] & m.feasible & (m.code == OK)
    pre_raw = e[:, None] & c3.pre_gate & c3.P_pre.ok
    pre = pre_raw & (c3.pre_code == OK)
    busy = jnp.any(cr.assigned >= 0)
    deadlock = ~busy & ~jnp.any(serve) & jnp.any(c3.blocked)             # 430-436행 (해제 뒤 후보로)
    rb, rvalid, resc, ovf = repo_targets(world, g, m, deadlock, c3.blocked, horizon_s, pre_advice)
    f_repo = jax.vmap(partial(plan_repo, g=g), in_axes=(None, None, 0))
    P_repo = jax.vmap(f_repo, in_axes=(None, 0, 0))(cr, jnp.arange(K, dtype=jnp.int32), rb)
    zero_slots = jnp.zeros((B, R_), bool)
    rcode_fn = jax.vmap(reject_code, in_axes=(None, None, None, 0, 0, None, None, None))
    repo_code = jax.vmap(rcode_fn, in_axes=(None, 0, None, 0, 0, None, None, None))(
        world.res, jnp.arange(K, dtype=jnp.int32), jnp.int32(EMPTY_ID), P_repo.lo, P_repo.hi,
        jnp.int32(EMPTY_ID), zero_slots, gap)
    far = jnp.abs(rb - cr.bay[:, None]) > REPO_MIN_MOVE
    repo_raw = e[:, None] & rvalid & (resc | far) & P_repo.ok
    repo = repo_raw & (repo_code == OK)
    open_ = e & (jnp.any(serve, axis=1) | (world.wake.eta_armed & c3.eta_opp))
    return c3._replace(eligible=e, serve_raw=serve_raw, serve=serve, pre_raw=pre_raw, pre=pre,
                       repo_bay=rb, repo_valid=rvalid, repo_escape=resc, P_repo=P_repo, repo_code=repo_code,
                       repo_raw=repo_raw, repo=repo, repo_score=jnp.asarray(REPO_SCORE_BASE, F) + rb,
                       esc_overflow=ovf, open=open_, deadlock=deadlock)


def apply_choices(world: BlockWorld, fl, pr, choice, open_, lost, g: Geom):
    """공동 결정의 적용 — v5 `baselines._apply`/`resolver.apply` (크레인 순 sorted) → `assign` (679-722행).

    choice (K,) int32 = 평면 후보 열 번호 (cands3.flat_view; -1 = WAIT). 열마다 kind·오더·bay 를 읽어 그 kind 로
    **live 재계획** (694행 — 앞 크레인의 예약이 반영된다) → reserve (697행 2차 방어선) → 크레인·계획·오더(SERVE 만)·비용·
    JOB_COMPLETED push·DISPATCH 로그(aux). WAIT → yielded (686행), lost[k] 면 yield_count+1 (687-688행).
    정책이 실린 후보(prune.keep & feasible) 밖 열을 고르면 V_DECISION_COVERAGE 를 켜고 건너뛴다 (v5 는 NOT_DISPATCHABLE
    /예약 예외). 반환 (세계', pick (K,) 오더 번호 [-1], kind (K,) PK_* [-1], bay (K,) REPO 목표 [NaN]).
    """
    K, N = world.k, world.n
    B, R_, _ = world.stacks.shape
    C = fl.raw.shape[1]
    gap = jnp.asarray(g.gap, F)
    zero_ex = jnp.zeros((B, R_), bool)
    choice = jnp.asarray(choice, jnp.int32)
    lost = jnp.zeros((K,), bool) if lost is None else jnp.asarray(lost, bool)

    def body(w, k):
        a = open_[k]
        c = choice[k]
        cc = jnp.clip(c, 0, C - 1)
        in_set = (c >= 0) & (c < C) & pr.keep[k, cc] & fl.feasible[k, cc]
        kind = jnp.where(in_set, fl.kind[k, cc], PK_WAIT).astype(jnp.int32)
        is_wait = a & ((c < 0) | (kind == PK_WAIT))
        off = a & (c >= 0) & ~in_set                                     # 목록 밖 → 512
        work = a & ~is_wait & in_set
        job = fl.job[k, cc]
        bay = fl.bay[k, cc]
        nc = jnp.clip(job, 0, N - 1)
        is_repo = kind == PK_REPOSITION
        is_pre = kind == PK_PRE_REHANDLE
        # 694행 `_plan(cid, ref)` — 그 kind 로 live 재계획 (carry 의 예약표)
        P_s = plan_serve(w, k, nc, zero_ex, g)
        P_p = plan_pre(w, k, nc, g)
        P_r = plan_repo(w.cranes, k, jnp.where(jnp.isnan(bay), w.cranes.bay[k], bay), g)
        P = tree_where(is_repo, P_r, tree_where(is_pre, P_p, P_s))
        token = jnp.where(is_repo, EMPTY_ID, nc).astype(jnp.int32)
        rel = w.clock + P.dur                                            # 701행 start_s + duration_s
        res2, code2 = reserve(w.res, k, token, P.lo, P.hi, P.lane, P.slots, rel, gap)   # 697행
        ok = work & P.ok & (code2 == OK)
        viol = (w.violation
                | jnp.where(work & ~ok, V_RESERVE_REJECT, 0).astype(jnp.int32)
                | jnp.where(off, V_DECISION_COVERAGE, 0).astype(jnp.int32))
        cr_, o_, pl_ = w.cranes, w.orders, w.plan
        cr_y = cr_._replace(yielded=cr_.yielded.at[k].set(cr_.yielded[k] | is_wait),          # 686행
                            yield_count=cr_.yield_count.at[k].add(jnp.where(is_wait & lost[k], 1, 0).astype(jnp.int32)))
        pl2 = pl_._replace(                                              # 698행 _active_plans[crane] = plan
            kind=pl_.kind.at[k].set(kind), job=pl_.job.at[k].set(jnp.where(is_repo, EMPTY_ID, nc)),
            lo=pl_.lo.at[k].set(P.lo), hi=pl_.hi.at[k].set(P.hi), dur=pl_.dur.at[k].set(P.dur),
            end_bay=pl_.end_bay.at[k].set(P.end_bay), end_row=pl_.end_row.at[k].set(P.end_row),
            rehandles=pl_.rehandles.at[k].set(P.rehandles),
            loaded_m=pl_.loaded_m.at[k].set(P.loaded_m), empty_m=pl_.empty_m.at[k].set(P.empty_m),
            n_moves=pl_.n_moves.at[k].set(P.n_moves), start_s=pl_.start_s.at[k].set(w.clock),
            mv_cont=pl_.mv_cont.at[k].set(P.mv_cont), mv_src=pl_.mv_src.at[k].set(P.mv_src),
            mv_dst=pl_.mv_dst.at[k].set(P.mv_dst), mv_kind=pl_.mv_kind.at[k].set(P.mv_kind))
        cr2 = cr_y._replace(                                             # 699-702행
            assigned=cr_.assigned.at[k].set(jnp.where(is_repo, BUSY_NO_ORDER, nc).astype(jnp.int32)),
            status=cr_.status.at[k].set(CR_HANDLING),
            available_at=cr_.available_at.at[k].set(rel),
            is_loaded=cr_.is_loaded.at[k].set(~is_repo))
        is_serve = kind == PK_SERVE
        is_ext = o_.is_external[nc]
        o2 = o_._replace(                                                # 705-713행 (SERVE 만)
            status=o_.status.at[nc].set(jnp.where(is_serve, JS_RUNNING, o_.status[nc])),
            assigned_crane=o_.assigned_crane.at[nc].set(jnp.where(is_serve, k, o_.assigned_crane[nc])),
            service_s=o_.service_s.at[nc].set(jnp.where(is_serve, w.clock, o_.service_s[nc])),
            waiting=o_.waiting.at[nc].set(jnp.where(is_serve & is_ext, False, o_.waiting[nc])),
            wait_sample_s=o_.wait_sample_s.at[nc].set(
                jnp.where(is_serve & is_ext, w.clock - o_.block_in_s[nc], o_.wait_sample_s[nc])))
        reh = P.rehandles.astype(F)                                      # 718행 float(plan.rehandles)
        acc = lambda v: (v.at[C_CRANE_TRAVEL].add(P.loaded_m).at[C_EMPTY_TRAVEL].add(P.empty_m)
                         .at[C_REHANDLE].add(reh))                       # 714-719행
        cost2 = w.cost._replace(pending=acc(w.cost.pending), episode=acc(w.cost.episode))
        q2 = push_event(w.queue, rel, EV_JOB_COMPLETED, k)               # 720행
        w_ok = w._replace(res=res2, plan=pl2, cranes=cr2, orders=o2, cost=cost2, queue=q2)
        w_ok = log_event(w_ok, w.clock, LOG_DISPATCH, k, jnp.where(is_repo, repo_aux(bay), nc))   # 721행
        w_no = w._replace(cranes=cr_y)
        w2 = tree_where(ok, w_ok, w_no)._replace(violation=viol)
        pick = jnp.where(ok & ~is_repo, nc, EMPTY_ID).astype(jnp.int32)
        kind_out = jnp.where(a, jnp.where(ok, kind, PK_WAIT), EMPTY_ID).astype(jnp.int32)
        bay_out = jnp.where(ok & is_repo, P.end_bay, jnp.nan)
        return w2, (pick, kind_out, bay_out)

    w2, (pick, kind, bay) = lax.scan(body, world, jnp.arange(K, dtype=jnp.int32))   # 크레인 순 = sorted(crane_id)
    return w2, pick, kind, bay


def decide_joint(world: BlockWorld, params, g: Geom, policy_fn: Callable, *, check: bool = True,
                 pre_advice: bool = False, horizon_s: float = 0.0) -> DecideOut:
    """공동 규약의 결정 국면 — `candidates3` 한 번 (D 개방 + X 술어) → 정책(전 크레인 한 번) → `apply_choices` → 마무리."""
    K = world.k
    c3 = candidates3(world, g, horizon_s=horizon_s, pre_advice=pre_advice)
    open_D = c3.open                                                     # 453-464행 (_decision_cranes)
    is_D = jnp.any(open_D) & (world.clock < world.end_s - EPS)           # 294행
    esc = try_escape(world, c3.m)                                        # 302-305행
    fired = ~is_D & esc.fired
    w_in = tree_where(is_D, world, esc.world)
    open_ = jnp.where(is_D, open_D, esc.open)
    decided = is_D | fired
    # X: yielded 해제 뒤의 eligible 로 원시 마스크·술어·REPO 목표를 다시 센다 (v5 는 generate 가 해제 뒤 상태를 읽는다)
    elig_x = c3.m.idle & ~w_in.cranes.yielded
    c3x = lax.cond(fired, lambda: recandidates(c3, w_in, elig_x, g, horizon_s=horizon_s, pre_advice=pre_advice),
                   lambda: c3)
    fl = flat_view(c3x)
    pr = prune(fl, g)
    choice, lost, flags = policy_fn(params, w_in, c3x, fl, pr, open_)
    w2, pick, kind, bay = apply_choices(w_in, fl, pr, choice, open_, lost, g)
    w2 = w2._replace(violation=w2.violation | jnp.asarray(flags, jnp.int32))
    w2 = close_decision(w2, open_, pick, g, consume_armed=is_D, check=check, kind=kind, bay=bay)
    return DecideOut(tree_where(decided, w2, w_in), decided, open_, pick, kind, bay)


# ───────────────────────────────────────────────── 스텝 · 실행 (§9)
class StepTrace(NamedTuple):
    """스텝마다 남기는 흔적 — 시험이 v5 결정열·계획 moves 와 대조한다."""

    clock: jnp.ndarray        # () f64   스텝 뒤 시각
    decided: jnp.ndarray      # () bool
    escaped: jnp.ndarray      # () bool  이 결정이 탈출(X) 로 열렸나 (escape_count 가 올랐다)
    kind: jnp.ndarray         # () int32 꺼낸 사건 종류 (-1 = 없음)
    target: jnp.ndarray       # () int32
    open: jnp.ndarray         # (K,) bool
    pick: jnp.ndarray         # (K,) int32   오더 번호 (SERVE·PRE) · -1 (WAIT·REPO·안 물음)
    pick_kind: jnp.ndarray    # (K,) int32   답의 종류 PK_* (-1 = 안 물음) — 조각 3
    pick_bay: jnp.ndarray     # (K,) f64     REPOSITION 목표 bay (NaN) — 조각 3
    woke: jnp.ndarray         # () bool      이 스텝이 W(깨우기) 로 끝났나 — 조각 3
    advanced: jnp.ndarray     # () bool      이 스텝이 A(wake 시각 전진) 로 끝났나 — 조각 3
    reviewed: jnp.ndarray     # () bool      이 스텝이 R(검토 시각 park) 로 끝났나 — 조각 6 (review=False 면 항상 False)
    plan_job: jnp.ndarray     # (K,) int32   결정 뒤 활성 계획 (kind<0 이면 없음)
    plan_kind: jnp.ndarray    # (K,) int32
    plan_n_moves: jnp.ndarray # (K,) int32
    plan_dur: jnp.ndarray     # (K,) f64
    plan_rehandles: jnp.ndarray  # (K,) int32
    plan_mv_cont: jnp.ndarray    # (K,M) int32
    plan_mv_src: jnp.ndarray     # (K,M,3) int32
    plan_mv_dst: jnp.ndarray     # (K,M,3) int32
    plan_mv_kind: jnp.ndarray    # (K,M) int32


def step(world: BlockWorld, _, *, params, g: Geom, policy_fn: Callable, check: bool = True,
         pre_advice: bool = False, horizon_s: float = 0.0, joint: bool = False, review: bool = False):
    """한 스텝 = `run_until_decision` 한 순회 (머리말). 반환 (세계', StepTrace). terminal 이면 항등.
    check (static): 결정 마무리·사건 처리 뒤 불변식 검사 (v5 `check_invariants` 플래그).
    pre_advice · horizon_s · joint (static): 정보수준 PRE_ADVICE · 결정 지평(profile.decision_horizon_s) · 결정 규약 (머리말).
    review (static, 조각 6): `wake.review_s`/`review_idx` 의 검토 시각을 소비한다 — [R] 국면 (머리말). False 면 v5 의
    `review_epochs=[]` 와 같이 그 국면이 정적으로 빠진다 (조각 1~5 시험·정답 궤적 Y01 은 그대로)."""
    K = world.k
    raw_nt = jnp.min(world.queue.time)                                   # 282행 peek_time
    alive = raw_nt < EMPTY_TIME
    inwin = raw_nt <= world.end_s + EPS                                  # 287행 평가창 밖 사건은 없는 것
    nt_ok = alive & inwin
    due_now = nt_ok & (raw_nt <= world.clock + EPS)                      # 288행
    # [W] 깨우기 (291-292행) — 동시각 사건을 다 소진한 뒤 · 결정 전. 발화하면 이 스텝은 여기서 끝난다.
    W_, D_ = world.wake.eta_wake_s.shape[0], world.wake.defer_wake_s.shape[0]
    if (pre_advice and W_ > 0) or D_ > 0:
        w_c, fired_c = consume_due_wakes(world, pre_advice=pre_advice)
        woke = fired_c & ~due_now & ~world.terminal
        w_w = tree_where(woke, w_c, world)
    else:
        w_w, woke = world, jnp.zeros((), bool)
    cr = w_w.cranes
    any_eligible = jnp.any((cr.assigned < 0) & ~cr.down & ~cr.yielded)   # 453-456행 idle & ~yielded
    any_busy = jnp.any(cr.assigned >= 0)                                 # 430행 탈출 술어 ①
    # ★지름길 (머리말): 유휴·비양보 크레인이 없고 작업 중이 있으면 v5 도 후보를 안 세고 탈출 술어 ①에서 () 다.
    #   전원 유휴(모두 양보/고장)면 탈출이 발화할 수 있으므로 행렬을 만든다. 시각 조건은 D(294행)·X(390행) 공통.
    try_decide = (~world.terminal & ~due_now & ~woke & (world.clock < world.end_s - EPS)
                  & (any_eligible | ~any_busy))

    def _decide(w):
        d = decide(w, params, g, policy_fn, check=check, pre_advice=pre_advice, horizon_s=horizon_s, joint=joint)
        return d.world, d.decided, d.open, d.pick, d.kind, d.bay

    def _skip(w):
        return (w, jnp.zeros((), bool), jnp.zeros((K,), bool), jnp.full((K,), EMPTY_ID, jnp.int32),
                jnp.full((K,), EMPTY_ID, jnp.int32), jnp.full((K,), jnp.nan, F))

    w_d, decided, open_, pick, pkind, pbay = lax.cond(try_decide, _decide, _skip, w_w)
    escaped = decided & (w_d.escape_count > world.escape_count)
    handlers = _handlers(g)
    # 306-308행 다음 wake 시각 (평가창 안) — A 국면의 전진 목표
    wt = next_wake_in_window(w_d, pre_advice=pre_advice)
    # [R] 검토 시각 (314-322행) — 결정·탈출이 안 열렸을 때, 다음 사건·wake 보다 이르거나 같으면(EPS) 그 시각으로 park.
    #   v5 는 review_epochs[0] > end+EPS 면 목록을 통째로 비운다 — 정렬 목록이라 '머리가 유효한가' 한 조건과 같다.
    Rv = int(world.wake.review_s.shape[0])
    if review and Rv > 0:
        wk = w_d.wake
        ri = wk.review_idx
        ep = jnp.where(ri < Rv, wk.review_s[jnp.clip(ri, 0, Rv - 1)], EMPTY_TIME)
        ep_ok = ep <= w_d.end_s + EPS                                    # 314-315행
        is_R = (~world.terminal & ~due_now & ~woke & ~decided & ep_ok
                & (~nt_ok | (ep <= raw_nt + EPS)) & (ep <= wt + EPS))    # 318행 (wt=+inf 면 참 = v5 None)
    else:
        ep = jnp.asarray(EMPTY_TIME, F)
        is_R = jnp.zeros((), bool)
    is_A = (wt < EMPTY_TIME) & (~nt_ok | (wt < raw_nt - EPS))            # 333행

    def _event(w):
        """국면 E — `_process_next_event` (776-782행)."""
        q2, t, kind, tgt, _ = next_event(w.queue)
        w1 = w._replace(queue=q2)
        w1 = advance(w1, t, g)                                           # 778행
        w1 = log_event(w1, t, kind, tgt)                                 # 845행
        w1 = lax.switch(kind, handlers, w1, tgt)                         # 846-885행
        w1 = _checked(refresh_rates(w1, g), g, check)                    # 780행 · 781-782행
        return w1, kind, tgt

    def _fin(w):
        """국면 F — 323-332행 검사 + `_finalize` (1068-1084행)."""
        w1 = advance(w, jnp.maximum(w.clock, w.end_s), g)                # 1071행
        w1 = VS.clearout_vessels(w1)                                     # 1072-1080행 미완 배 선석 초과 정산 (조각 4)
        busy = jnp.any(w1.cranes.assigned >= 0) & ~alive                 # 324-325행 RuntimeError
        o = w1.orders
        ws = jnp.where(o.waiting & jnp.isnan(o.wait_sample_s),
                       jnp.maximum(0.0, w1.end_s - o.block_in_s), o.wait_sample_s)   # 1081행 kpis.py:58-63
        lm = ledger_mode(o)
        ld = w1.ledger._replace(closed_end=jnp.where(lm, w1.end_s, w1.ledger.closed_end))   # 1082-1083행
        w1 = w1._replace(orders=o._replace(wait_sample_s=ws), ledger=ld,
                         terminal=jnp.ones((), bool),                    # 1084행
                         violation=w1.violation | jnp.where(busy, V_BUSY_NO_EVENT, 0).astype(jnp.int32))
        return w1, jnp.int32(EMPTY_ID), jnp.int32(EMPTY_ID)

    def _adv(w):
        """국면 A — 334행 `_advance(wt)` 만 (로그·rate 갱신 없음; 다음 스텝의 W 가 소비한다)."""
        return advance(w, wt, g), jnp.int32(EMPTY_ID), jnp.int32(EMPTY_ID)

    def _identity(w):
        return w, jnp.int32(EMPTY_ID), jnp.int32(EMPTY_ID)

    def _review(w):
        """국면 R — 319-322행: pop(0) · ep > clock+EPS 면 `_advance(ep)` 만 (로그·rate 갱신 없음) · ReviewEpoch."""
        w1 = w._replace(wake=w.wake._replace(review_idx=w.wake.review_idx + 1))
        w2 = tree_where(ep > w.clock + EPS, advance(w1, ep, g), w1)      # 320-321행
        return w2, jnp.int32(EMPTY_ID), jnp.int32(EMPTY_ID)

    # ★[R] 갈래는 **static 으로** 붙인다 — review=False 인 경로(조각 1~5·학습용 run_while)에서 갈래를 5개로 두면
    #   vmap 이 switch 를 select 로 접으면서 쓰지도 않는 `_review` 의 advance(world, +inf) 를 매 스텝 계산한다
    #   (ep = EMPTY_TIME 이라 술어가 상수 True → (N,) 오더 배열 전체를 end_s 까지 적분). 답은 같고 계산만 버린다.
    _has_R = bool(review and Rv > 0)
    branches = [_identity, _adv, _event, _fin] + ([_review] if _has_R else [])
    base = jnp.where(is_A, 1, jnp.where(nt_ok, 2, 3))                 # A / E / F
    phase = jnp.where(is_R, 4, base) if _has_R else base              # R 은 A·E·F 보다 앞
    idx = jnp.where(decided | world.terminal | woke, 0, phase)        # 0 = 항등 (D·X·W·terminal)
    w_out, kind, tgt = lax.switch(idx, branches, w_d)
    advanced = (idx == 1)
    reviewed = (idx == 4) if _has_R else jnp.zeros((), bool)
    w_out = w_out._replace(steps=world.steps + jnp.where(world.terminal, 0, 1).astype(jnp.int32))
    pl = w_out.plan
    trace = StepTrace(clock=w_out.clock, decided=decided, escaped=escaped, kind=kind, target=tgt, open=open_, pick=pick,
                      pick_kind=pkind, pick_bay=pbay, woke=woke, advanced=advanced, reviewed=reviewed,
                      plan_job=pl.job, plan_kind=pl.kind, plan_n_moves=pl.n_moves, plan_dur=pl.dur,
                      plan_rehandles=pl.rehandles, plan_mv_cont=pl.mv_cont, plan_mv_src=pl.mv_src,
                      plan_mv_dst=pl.mv_dst, plan_mv_kind=pl.mv_kind)
    return w_out, trace


def finish(w: BlockWorld) -> BlockWorld:
    """실행 끝 마무리 — 스텝 소진(256)·칸 부족(4096) 을 violation 한 열로 모은다.

    큐 넘침은 `queue.overflow` 에, 로그·wake 넘침은 `world.overflow` 에, 이송 대기 링버퍼 넘침은 `transfer.overflow`
    에 따로 쌓이므로 학습 루프가 `violation` 만 봐도 실격을 놓치지 않게 여기서 합친다 (명세 '0 이 아니면 실격').
    """
    total_overflow = w.overflow + w.queue.overflow + w.transfer.overflow   # 이송 대기 링버퍼 넘침도 실격 (조각 4)
    viol = (w.violation
            | jnp.where(w.terminal, 0, V_STEPS_EXHAUSTED).astype(jnp.int32)
            | jnp.where(total_overflow > 0, V_OVERFLOW, 0).astype(jnp.int32))
    return w._replace(violation=viol)


def run(world0: BlockWorld, params, g: Geom, policy_fn: Callable, S_max: int, check: bool = True,
        pre_advice: bool = False, horizon_s: float = 0.0, joint: bool = False, review: bool = False):
    """끝까지 굴린다 — `lax.scan(step, w0, None, length=S_max)`. 끝나기 전에 스텝이 소진되면 violation |= 256,
    큐/로그 칸이 모자랐으면 |= 4096 (`finish`).

    반환 (세계, StepTrace 각 열 앞에 (S_max,)). jit 은 `run_jit` (g·policy_fn·S_max·check·pre_advice·horizon_s·joint·review 가 static).
    review=True 면 검토 시각마다 [R] 스텝이 하나씩 낀다 (park 는 하지 않는다 — 다음 스텝이 이어 간다; 조정자는 `gpu/multiblock.py`).
    """
    f = partial(step, params=params, g=g, policy_fn=policy_fn, check=check,
                pre_advice=pre_advice, horizon_s=horizon_s, joint=joint, review=review)
    w, trace = lax.scan(lambda w, x: f(w, x), world0, None, length=int(S_max))
    return finish(w), trace


run_jit = jax.jit(run, static_argnames=("g", "policy_fn", "S_max", "check", "pre_advice", "horizon_s", "joint", "review"))


def run_while(world0: BlockWorld, params, g: Geom, policy_fn: Callable, S_max: int, check: bool = True,
              pre_advice: bool = False, horizon_s: float = 0.0, joint: bool = False, review: bool = False):
    """학습 경로 — `lax.while_loop` 로 terminal 까지만 돈다 (흔적 없음; 머리말 ■ 실행 두 경로).

    vmap 아래서는 술어가 배치 any 로 바뀌어 **살아 있는 세계가 하나라도 있으면** 계속, 끝난 세계는 `step` 이
    항등이라 그대로다. S_max 를 넘기면 `finish` 가 256 을 켠다. 반환 세계 = `run` 의 세계와 잎 전부 비트 동일.
    """
    f = partial(step, params=params, g=g, policy_fn=policy_fn, check=check,
                pre_advice=pre_advice, horizon_s=horizon_s, joint=joint, review=review)

    def cond(s):
        i, w = s
        return (i < int(S_max)) & ~w.terminal

    def body(s):
        i, w = s
        w2, _ = f(w, None)
        return i + 1, w2

    _, w = lax.while_loop(cond, body, (jnp.int32(0), world0))
    return finish(w)


run_while_jit = jax.jit(run_while, static_argnames=("g", "policy_fn", "S_max", "check", "pre_advice", "horizon_s", "joint", "review"))


def run_python(world0: BlockWorld, params, g: Geom, policy_fn: Callable, S_max: int, check: bool = True,
               pre_advice: bool = False, horizon_s: float = 0.0, joint: bool = False, review: bool = False):
    """jit **없이** 파이썬 루프로 `step` 을 반복 — 시험 4) jit 유무가 답을 바꾸지 않는지.

    terminal 이 되면 멈춘다. 반환 (세계, StepTrace 각 열 앞에 (사용한 스텝 수,)).
    """
    w = world0
    traces = []
    for _ in range(int(S_max)):
        w, tr = step(w, None, params=params, g=g, policy_fn=policy_fn, check=check,
                     pre_advice=pre_advice, horizon_s=horizon_s, joint=joint, review=review)
        traces.append(tr)
        if bool(w.terminal):
            break
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *traces)
    return finish(w), stacked


def cut(world: BlockWorld):
    """v5 `CostAccumulator.cut` (cost.py:82-88) — 현재 결정구간 raw 13항을 돌려주고 pending 을 비운다
    (episode 는 보존). 조각 5/7 의 보상 차분용. 반환 (세계', pending (13,))."""
    pend = world.cost.pending
    return world._replace(cost=world.cost._replace(pending=jnp.zeros((N_COST,), F))), pend
