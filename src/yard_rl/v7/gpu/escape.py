"""간섭 교착 탈출을 **배열로** ([[YR-327]] 조각 2 · v5 `world/integrated/engine.py:384-450`
`_try_escape` · `interference_deadlock_corridors` 의 배열판).

■ 왜 있나 (YR-112)
  두 크레인이 서로의 **유휴 위치 장벽**(YR-091 점 통로, reservation.py:70-76) 때문에 어느 쪽도
  어떤 작업도 못 잡는 상태가 생긴다 — 모두 놀고 있고 후보는 하나도 없는데 거절 사유는 오직
  크레인 간섭. 그동안 대기비용은 계속 쌓이는데 정책은 **질문조차 받지 못한다**. 그래서 v5 는 그
  순간 **결정 기회만** 연다(물리는 그대로다 — 관통 금지는 옳다). 정책이 REPOSITION(조각 3)으로
  비켜 주면 풀리고, SERVE/WAIT 뿐이면 전원 WAIT 로 답한다.

■ 술어 (engine.py:430-450) — 셋 다 참일 때만 교착
    ① 작업 중 없음   ~any(assigned ≥ 0)                                     (430행)
    ② 후보 없음      ~any(cand)     cand = eligible & feasible & code==OK       (431행 candidates_for 전 크레인)
    ③ 간섭 거절 존재 any(blocked)   blocked = feasible & code==CRANE_INTERFERENCE (437-450행)
      feasible = dispatchable & ~taken & plan_ok  — 예약 판정 **직전까지** 통과한 (크레인, 오더) 쌍
  ★③ 은 **전 크레인**(고장·양보 포함)을 돈다 — 437행 `for cid in self.fleet.ids()` 에 크레인 상태
    검사가 없고 `_dispatchable`·`_jobref`·`_plan` 도 크레인 상태를 안 본다. ② 의 candidates_for 는
    유휴·비양보 크레인만 후보를 낸다(473행). 두 조건의 크레인 마스크가 **다르다**.
  ★거절 사유는 reject_code **전체**(5-lock 사다리)를 쓰고 ==4 로 비교한다 — 조각 1 이 남긴 주의
    (reserve.py 의 lane 검사는 자기 예약을 안 빼므로 c3 만 따로 쓰면 v5 와 갈린다).
  ★①이 참이면 활성 예약이 없다(예약은 배정과 함께 잡히고 완료와 함께 풀린다) — 그래서 교착에서
    간섭은 오직 **유휴 위치 점 장벽**에서 온다. 그래도 술어는 reject_code 를 그대로 쓴다(v5 와 같은 소스).

■ 발화 규칙 (engine.py:384-410 `_try_escape`) — 순서 그대로
    시각   clock < end−EPS  &  escape_at ≠ clock (정확 비교, 390행)  &  ~(clock ≤ last_decision_at + EPS) (394행)
    술어   위 셋 (396행)
    → yielded 전부 해제 (398행 — 술어까지 통과하면 **esc 가 비어도** 해제된다)
    → esc = 유휴(assigned<0 & ~down) 크레인 (399행); 비어 있으면 없던 일 (400-401행)
    → escape_at = last_decision_at = clock · escape_count+1 · pending = esc · _assigned = {} ·
      로그 (clock, DEADLOCK_ESCAPE, esc)  (403-408행)
  escape_mode 는 'immediate' 만 옮긴다. 325-329행의 지연 호출은 immediate 에서 항상 무동작이다
  (같은 시각·같은 상태를 다시 묻는 것이라 escape_at 표식 또는 같은 술어값으로 막힌다).
  None 표식(escape_at·last_decision_at)은 -inf 다 — 유한 clock 과 `==`·`≤+EPS` 어느 쪽도 참이 안 된다.

■ 호출 자리 (engine.py:302-305) — 결정 국면 D 가 **열리지 않은** 뒤 · 사건 E / 종료 F 국면 **앞**
    ~terminal & ~due_now & D 미개방  →  m = candidate_matrices(w, g)  →  esc = try_escape(w, m)
      esc.fired  →  결정: pick = policy(features(w, m.P), esc.cand) → 배정 scan(open=esc.open) → close(refresh_rates)
      아니면    →  E/F
  D 의 `any_eligible` 지름길(유휴·비양보 크레인이 없으면 후보를 안 센다)은 X 에 쓸 수 없다 — ③ 은 고장·양보
  크레인 쌍도 세므로 m 은 D 가 안 열려도 만들어야 한다. 배정 scan 은 open=esc.open(후보 없는 크레인 포함)으로
  돌아야 그 크레인이 WAIT → yielded 가 된다 (686행) — engine_step.decide 는 open 을 any(cand) 로 정하므로
  그대로 재사용할 수 없고, open·cand 를 인자로 받게 나누어야 한다.
  결정 대상은 **유휴 전원** — 후보 없는 크레인도 묻고, WAIT 면 yielded 가 된다(interference rate 에 잡힌다).
  정책이 볼 후보는 yielded 해제 **뒤** 다시 센 `EscapeOut.cand` (= open & feasible & code==OK) —
  양보 중이던 크레인이 후보를 되찾을 수 있다(② 는 해제 전 값이라). 위치·예약은 안 변하므로 P·code 는 그대로다.
  v5 탈출 결정은 `_eta_armed` 를 **건드리지 않는다** (297행은 정상 결정 경로에만 있다).
  D 와 X 가 같은 (K,N) 행렬을 쓰도록 `candidate_matrices` 를 D 앞에서 한 번 만들어 넘긴다
  (hard_parts 1 — 술어가 후보 생성 전체를 재계산하던 것을 없앤다).

■ 로그 target 인코딩
  DEADLOCK_ESCAPE 의 target 은 유휴 크레인 **비트마스크**(bit k) — host_convert 가 ",".join(정렬 id) 로
  되돌린다. K > 31 이면 int32 가 모자란다(단일 블록 K ≤ 2).

■ 통로 (interference_deadlock_corridors 의 반환값)
  v5 는 막힌 오더마다 **크레인 번호 순 첫** 간섭 크레인의 (lo, hi) 를 모아 sorted(set(…)) 로 돌려준다
  (449행 break). 배열판 `blocked_corridors` 는 오더 축 (N,) lo·hi + has 마스크다 — 정렬·중복제거는
  호스트/조각 3 몫. 조각 2 엔진은 술어값만 쓴다.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .events import TIME_DTYPE
from .plan import PlanOut, plan_serve_all
from .reserve import CRANE_INTERFERENCE, OK, reject_code_over_orders
from .state import EMPTY_ID, EMPTY_TIME, LOG_DEADLOCK_ESCAPE, BlockWorld, CraneArrays

__all__ = ["EPS", "CandMats", "EscapeOut", "idle_mask", "candidate_matrices",
           "deadlock_predicate", "blocked_corridors", "crane_bits", "try_escape"]

F = TIME_DTYPE
#: v5 `_EPS` (engine.py:35) — engine_step.EPS 와 같은 값 (순환 import 를 피해 여기 다시 적는다)
EPS = 1e-9


def _es():
    """`engine_step` 을 **호출 시점에** 불러온다.

    통합 뒤 engine_step 이 이 모듈을 import 하므로 여기서 위에서 import 하면 순환이 된다.
    필요한 것은 `dispatchable`·`log_event`·`tree_where` 셋뿐이고, 복사해 두면 두 벌이 된다.
    """
    from . import engine_step
    return engine_step


# ───────────────────────────────────────────────── 후보 행렬 (K,N) — D 와 X 가 공유
class CandMats(NamedTuple):
    """결정 국면이 쓰는 (크레인 K × 오더 N) 행렬 묶음 — v5 candidates_for 의 단계별 결과."""

    P: PlanOut              # 각 열 앞에 (K,N) — `_jobref` + `_plan` (483-488행)
    disp: jnp.ndarray       # (K,N) bool  `_dispatchable` (479행) — 크레인 상태와 무관
    taken: jnp.ndarray      # (N,)  bool  `job_taken` (481행)
    code: jnp.ndarray       # (K,N) int32 `reject_reason` 5-lock 코드 (489행 can_reserve 의 근거)
    feasible: jnp.ndarray   # (K,N) bool  disp & ~taken & P.ok — 예약 판정 직전까지 통과 (전 크레인)
    idle: jnp.ndarray       # (K,)  bool  assigned<0 & ~down (cranes.py:33)
    eligible: jnp.ndarray   # (K,)  bool  idle & ~yielded (473행)
    cand: jnp.ndarray       # (K,N) bool  eligible & feasible & code==OK — candidates_for 의 후보


def idle_mask(cr: CraneArrays) -> jnp.ndarray:
    """v5 `YcRuntime.idle` (cranes.py:31-33) — 작업 없음 & 고장 아님. (K,) bool."""
    return (cr.assigned < 0) & ~cr.down


def candidate_matrices(world: BlockWorld, g) -> CandMats:
    """(K,N) 후보 행렬을 **한 번** 만든다 — engine_step.decide 550-560행과 같은 계산.

    D(정상 결정)는 `cand` 로 열린 크레인을 정하고, X(탈출)는 `feasible`·`code`·`idle` 을 더 쓴다.
    g 는 static (jit 하려면 `static_argnames='g'`).
    """
    ES = _es()
    K, N = world.k, world.n
    B, R, _ = world.stacks.shape
    cr = world.cranes
    idle = idle_mask(cr)
    eligible = idle & ~cr.yielded                                          # 456, 473행
    disp = ES.dispatchable(world, g)                                       # 479행
    taken = world.res.token_owner >= 0                                     # 481행
    P = plan_serve_all(world, jnp.zeros((B, R), bool), g)                  # 483-488행
    code_fn = jax.vmap(reject_code_over_orders, in_axes=(None, 0, None, 0, 0, 0, 0, None))
    code = code_fn(world.res, jnp.arange(K, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32),
                   P.lo, P.hi, P.lane, P.slots, jnp.asarray(g.gap, F))    # 489행 reject_reason
    feasible = disp & ~taken[None, :] & P.ok
    cand = eligible[:, None] & feasible & (code == OK)
    return CandMats(P=P, disp=disp, taken=taken, code=code, feasible=feasible,
                    idle=idle, eligible=eligible, cand=cand)


# ───────────────────────────────────────────────── 술어 (430-450행)
def deadlock_predicate(world: BlockWorld, m: CandMats):
    """`interference_deadlock_corridors()` 가 비어 있지 않은가 → (deadlock () bool, blocked (K,N) bool).

    blocked 는 술어값과 무관하게 계산된다(③ 의 재료). ①·② 가 거짓이면 v5 는 () 를 돌려주고
    blocked 를 보지 않는다 — 호출자는 deadlock 이 참일 때만 blocked/통로를 읽는다.
    """
    busy = jnp.any(world.cranes.assigned >= 0)                             # 430행 ①
    blocked = m.feasible & (m.code == CRANE_INTERFERENCE)                  # 437-450행 ③ (전 크레인)
    deadlock = ~busy & ~jnp.any(m.cand) & jnp.any(blocked)                 # 431행 ②
    return deadlock, blocked


def blocked_corridors(m: CandMats, blocked: jnp.ndarray):
    """막힌 오더의 통로 — 오더마다 **크레인 번호 순 첫** blocked 크레인의 (lo, hi) (449행 break).

    반환 (lo (N,) f64, hi (N,) f64, has (N,) bool). 막히지 않은 오더는 +inf. 정렬·중복제거 없음.
    """
    N = blocked.shape[1]
    has = jnp.any(blocked, axis=0)
    k1 = jnp.argmax(blocked, axis=0)                                       # 첫 True (없으면 0 — has 로 가림)
    n_idx = jnp.arange(N, dtype=jnp.int32)
    lo = jnp.where(has, m.P.lo[k1, n_idx], EMPTY_TIME)
    hi = jnp.where(has, m.P.hi[k1, n_idx], EMPTY_TIME)
    return lo, hi, has


def crane_bits(mask: jnp.ndarray) -> jnp.ndarray:
    """(K,) bool → 비트마스크 int32 (bit k = 크레인 k). 로그 target 인코딩 (host_convert 머리말)."""
    K = mask.shape[0]
    bits = jnp.left_shift(jnp.int32(1), jnp.arange(K, dtype=jnp.int32))
    return jnp.sum(jnp.where(mask, bits, jnp.int32(0))).astype(jnp.int32)


# ───────────────────────────────────────────────── 탈출 (384-410행)
class EscapeOut(NamedTuple):
    world: BlockWorld       # fired 면 갱신된 세계, 아니면 (go 일 때 yielded 만 해제된) 세계
    fired: jnp.ndarray      # () bool   v5 가 TerminalDecision 을 돌려줬나
    open: jnp.ndarray       # (K,) bool 결정 대상 = 유휴 전원 (fired 거짓이면 전부 False)
    cand: jnp.ndarray       # (K,N) bool 탈출 결정에서 정책이 볼 후보 — yielded 해제 뒤 (open & feasible & code==OK)
    deadlock: jnp.ndarray   # () bool   술어 자체 (fired 와 다를 수 있다 — 시각·표식 조건)
    blocked: jnp.ndarray    # (K,N) bool 간섭으로 막힌 쌍


def try_escape(world: BlockWorld, m: CandMats) -> EscapeOut:
    """v5 `_try_escape` (384-410행) 한 번 — 순수 함수. 호출 자리는 머리말.

    발화하지 않으면 세계는 그대로다 — 단, 시각·표식·술어를 다 통과하고 esc 만 비었을 때는
    v5 처럼 yielded 가 해제된 세계를 돌려준다(398행이 400행 앞에 있다).
    """
    ES = _es()
    K = world.k
    clock = world.clock
    cr = world.cranes

    time_ok = (clock < world.end_s - EPS) & (world.escape_at != clock)     # 390행
    mark_ok = ~(clock <= world.last_decision_at + EPS)                     # 394행 (None → -inf 라 항상 통과)
    deadlock, blocked = deadlock_predicate(world, m)                       # 396행
    go = time_ok & mark_ok & deadlock
    open_ = m.idle & go                                                    # 399행 esc
    fired = jnp.any(open_)                                                 # 400-401행

    cr2 = cr._replace(yielded=jnp.where(go, False, cr.yielded))            # 398행 _clear_yields (esc 비어도)
    dec = world.decision._replace(                                         # 406-407행 _pending = esc · _assigned = {}
        pending=open_, answered=jnp.zeros((K,), bool),
        act_kind=jnp.full((K,), EMPTY_ID, jnp.int32), act_job=jnp.full((K,), EMPTY_ID, jnp.int32),
        act_bay=jnp.full((K,), jnp.nan, F))
    w_fire = world._replace(cranes=cr2, escape_at=clock, last_decision_at=clock,   # 403-404행
                            escape_count=world.escape_count + jnp.int32(1),        # 405행
                            decision=dec)
    w_fire = ES.log_event(w_fire, clock, LOG_DEADLOCK_ESCAPE, crane_bits(open_))   # 408행
    w2 = ES.tree_where(fired, w_fire, world._replace(cranes=cr2))
    cand = open_[:, None] & m.feasible & (m.code == OK)                    # 해제 뒤 정책이 볼 후보
    return EscapeOut(world=w2, fired=fired, open=open_, cand=cand, deadlock=deadlock, blocked=blocked)
