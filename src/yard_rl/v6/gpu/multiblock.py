"""다중블록 조정자 — v5 `MultiBlockTerminal.run` 의 공용 시계 루프를 **배열로** ([[YR-327]] 조각 6 · key=integrate).

v5 `world/integrated/multiblock.py` 169-211행 `run` 은 이렇게 돈다:
    시계가 가장 이른 블록 하나를 골라 `run_until_decision` (동률이면 dict 삽입순 = layout.ids 순, 192행)
    → None 이면 terminal 제외 · ReviewEpoch 면 `_sync_locks` 뒤 park · TerminalDecision 이면 정책
    → 살아 있는 블록이 **전부 park** 하면 t = min(parked) 로 `review_fn(self, t)` (투입 = ScheduledAnnouncer.review)
    → 끝에 `ledger.harvest`.
여기서는 그것을  **바깥 lax.scan(검토 에폭 e) × 안쪽 vmap(블록 b) × lax.while_loop('다음 에폭까지')**  로 편다:
    에폭 e 마다  ① 전 블록을 vmap 으로 `engine_step.step(review=True)` 반복 — 블록마다 [R] 국면(ReviewEpoch)이 나오면 park
                 (parked 마스크; terminal 은 제외) — 살아 있는 블록이 전부 park/terminal 이면 멈춤
                 ② `_sync_locks` (전 블록 · 212-227행)  ③ 투입 `admission.admit_epoch` (전 블록 vmap × 명단 순 scan)
                 ④ 전역 원장 등록 (290-291행)  ⑤ epoch_idx = e+1
    마지막 에폭 뒤  전 블록을 terminal 까지 (`finish_run`) → `engine_step.finish` → `harvest` (66-83행).

■ ★블록 순서가 답을 바꾸지 않는가 — v5 코드로 확인한 것 (piece6_spec ■ 블록 전진 순서)
    검토 시각 사이 블록끼리 쓰는 경로는 없다: `run_until_decision` 은 자기 sim 만 만지고, `policy_fn(sim, dp)` 도
    그 sim 만 읽는다 (CandidateGenerator 캐시는 id(sim) 별). 블록 간 쓰기는 **review_fn 안** 에만 있다 —
    `admit_external_job` (자기 블록 + 전역 원장의 자기 레코드) · `commit`/`defer` (src·dst 두 블록 + 원장) —
    그리고 그 시각에는 전 블록이 같은 t 에 park 해 있다 (계약 ②, 152-153행). 그래서 (i) 에폭 사이는 블록 병렬(vmap)이
    v5 의 어떤 직렬 순서와도 같은 답이고, (ii) 에폭 안 투입은 블록별로 명단 순서(=v5 순서)만 지키면 된다 —
    다른 블록의 투입이 이 블록의 `free_slots` 에 영향을 주지 않는다 (230-240행은 자기 블록만 센다).
    `min(movable, key=clock)` 의 동률 순서가 영향을 주는 곳은 `totals` 의 `sum(totals.values())` 덧셈 순서뿐이고
    (host_terminal.from_terminal_world 가 같은 순서로 더한다) cost_fn 이 없으면 전부 0 이다.
    ★예외 — 이송(commit)·이연은 두 블록의 상태를 바꾸므로 review 콜백에서만, 순차로 (`run_python` 의 review_fn).

■ ★park 시각 — `_sync_locks` 는 ReviewEpoch 를 돌려준 블록만 (203행) 이지만 살아 있는 블록은 전부 park 하므로
    '전 블록 sync' 와 같다. terminal 인 블록은 상태가 더 안 바뀌어 빼도 같다. `harvest` (207행) 는 런 끝 1회.
    park 시각 = ep (엔진 320-321행이 ep > clock+EPS 면 advance(ep)); 투입의 clock 비교(267행)는 그 블록 시계로 한다.

■ 상태 — `TerminalRun` (pytree · jit/scan 의 carry · numpy 로 저장/복원)
    tw       host_terminal.TerminalWorld — 블록 세계 (B,…) · 명단 · 검토 시각 · 원장(TerminalLedgerArrays) · epoch_idx
    adm      admission.ScheduleArrays — 투입 버킷 (E_grid, B, M) (명단 색인 = tw.sched 의 s)
    params   정책 파라미터 (블록별로 쌓은 것 · 없으면 None)
    locked   (S,) bool  v5 JobRecord.locked (sync_locks · harvest)
    open_txn / open_overflow / hist_overflow  이송 트랜잭션 상태 (transfer_txn.TxnLedger 의 것 — 브리지 `txn_ledger`)
    n_admitted () int32 · exhausted (B,) int32 (에폭 안 스텝 상한에 걸린 횟수 — 0 이어야 정상 · V_STEPS_EXHAUSTED 도 켠다)
    ★원장은 세 표현(TerminalLedgerArrays · TxnLedger · TruckLedger) 가운데 **TerminalLedgerArrays 를 정본**으로 두고
      이송·이연 때만 `txn_ledger`/`with_txn_ledger` 로 TxnLedger 를 잠깐 만든다 (열 이름이 같아 기계적).
    ★★**원장의 (owner, row) 를 바꾸는 문은 `with_txn_ledger` 하나뿐이다** (계약). 그 함수가 되돌릴 때 `a_gate_in` 을
      새 소유 행 `orders.gate_in_s[owner,row]` 에서 다시 읽기 때문에, 이 문을 우회해 owner/row 를 고치면 턴타임 표본이
      **옛 행을 읽는다** (`host_terminal.from_terminal_world` 가 그 owner/row 로 gate_out 을 찾는다). 새 경로를 만들 때는
      반드시 이 함수를 지나가게 하고, 의심되면 `check_invariants(run, tt)` 로 (owner,row) 가 실제 그 트럭 행인지 확인한다.

■ 세션 이어 돌리기 (README ⚠️ 환경 — WSL 창 80초)
    `run_epochs_jit(run, e0, n, eng)` 가 에폭 n 개를 scan 으로 돌리고, `save_run`/`load_run` 이 numpy 로 저장·복원한다.
    시험 드라이버(tests/v6/test_gpu_terminal_equiv.py)는 부하 300 을 여러 세션으로 나눠 잇는다. GPU 도 같은 함수.

■ ★적용 범위 — **`MultiBlockTerminal` 전용**이다 (2026-09-26 검증). 위 '블록 순서가 답을 바꾸지 않는가' 논증은
    MultiBlockTerminal 에서만 성립한다: 30일 경로의 `stage/cargo_runtime.CargoTerminal` 은 전역 큐 4단 키
    `(우선순위, 블록 이름, …)` · affected 블록 마스크 · 원격 본선 인계(블록 간 쓰기) · 끝의 물리 재고 등식을 두어
    블록 간 결합이 있다 — 그 이식은 조각 6 범위 밖이며 Dashboard backlog 로 넘긴다.

■ 동등성 (규칙 1) — 이 파일에는 실수 연산이 없다 (덧셈도 없음). 적분·비용은 engine_step 이, 투입 시각은 admission 이,
    이송·이연은 transfer_txn 이 v5 식 그대로 한다. 여기서는 순서·마스크·색인만 다룬다.
"""
from __future__ import annotations

import dataclasses
import pickle
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import admission as AD
from . import dispatch as DP
from . import engine_step as ES
from . import host_terminal as HT
from . import transfer_txn as TX
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .geom import Geom
from .state import (JS_ASSIGNED, JS_DONE, JS_PLANNED, JS_RUNNING, JS_WAITING, V_STEPS_EXHAUSTED,
                    BlockWorld)

__all__ = [
    "Engine", "TerminalRun", "make_run", "terminal_resolver_params", "run_to_epoch", "review_epoch",
    "epoch_step", "run_epochs", "run_epochs_jit", "finish_run", "finish_run_jit", "run_all", "run_python",
    "txn_ledger", "with_txn_ledger", "try_pre_gate_transfer", "try_transfer", "try_defer_admitted_entry",
    "check_invariants", "from_run", "epoch_time", "save_run", "load_run", "tree_to_numpy", "tree_to_jax",
    "steps_for",
]

F = TIME_DTYPE
#: 이송 트랜잭션 슬롯 기본값 (transfer_txn.empty_txn_ledger max_open)
MAX_OPEN = 4


# ───────────────────────────────────────────────── 정적 설정 (jit static)
@dataclasses.dataclass(frozen=True)
class Engine:
    """조정자의 정적 설정 — jit 의 static 인자 (hashable). 값은 test_gpu_y01 의 `run_jit` 인자와 같은 뜻.

    g · policy_fn · check · pre_advice · horizon_s · joint  → engine_step.step
    margin(CAPACITY_MARGIN) · end_ann(ScheduledAnnouncer.end_s; +inf = None) → admission.admit_epoch
    steps_per_epoch · steps_final  while_loop 상한 (v5 guard 2,000,000 의 자리 — 걸리면 exhausted+1 · V_STEPS_EXHAUSTED)
    k0  v5 `profile.cranes[0]` 의 정렬 크레인 번호 (validate 의 find_slot 기준, transfer_txn.validate)
    """

    g: Geom
    policy_fn: Callable
    check: bool = True
    pre_advice: bool = True
    horizon_s: float = 0.0
    joint: bool = True
    margin: int = AD.CAPACITY_MARGIN
    end_ann: float = float("inf")
    #: ⚠️ while_loop 상한은 **무대 크기에서** 정해야 한다 — `steps_for(tt)`(= 8·n_max + 256) 를 쓰라.
    #   아래 리터럴은 옛 기본값이라 큰 무대에서 모자랄 수 있고, 상한에 걸리면 그 블록을 에폭 중간에 놔둔 채
    #   투입으로 넘어가 **블록 시계가 검토 시각에 닿지 않은 상태에서 투입 검사(clock 비교)** 가 돈다.
    #   그런 일이 있으면 exhausted 가 0 이 아니고 V_STEPS_EXHAUSTED 가 켜진다 — 시험은 그것을 0 으로 단언한다.
    steps_per_epoch: int = 2304
    steps_final: int = 4096
    k0: int = 0
    max_transfers: int = 1


class TerminalRun(NamedTuple):
    """조정자 상태 전체 (머리말 ■ 상태)."""

    tw: HT.TerminalWorld
    adm: AD.ScheduleArrays
    params: Any
    locked: jnp.ndarray          # (S,) bool
    open_txn: jnp.ndarray        # (MAX_OPEN,) int32
    open_overflow: jnp.ndarray   # () int32
    hist_overflow: jnp.ndarray   # () int32
    n_admitted: jnp.ndarray      # () int32
    exhausted: jnp.ndarray       # (B,) int32

    @property
    def b(self) -> int:
        return self.tw.b

    @property
    def n_epochs(self) -> int:
        return int(self.tw.epochs.t.shape[0])


# ───────────────────────────────────────────────── 호스트 구성
def _pad_i32(arr, n: int, fill: int):
    a = jnp.asarray(arr, jnp.int32)
    n0 = int(a.shape[0])
    if n0 >= n:
        return a[:n]
    return jnp.concatenate([a, jnp.full((n - n0,), fill, jnp.int32)])


def terminal_resolver_params(tt: HT.TerminalTables, g: Geom, tw: HT.TerminalWorld | None = None) -> DP.ResolverParams:
    """블록별 `dispatch.resolver_params(tables_b, g)` 를 (B,…) 로 쌓는다 — 순위표는 **namespaced** 번호표로
    (host_terminal 머리말 ★ — REPO 이름과 오더 이름의 순서가 블록 단독과 터미널에서 다르다).

    `tw` 를 주면 `block_tables(tw, tt, b)` (원장 기준 — 이송으로 여분 행에 들어온 트럭 이름 반영) 로 굽는다.
    (N0_b,) 열은 n_max 로 채운다 (빈 칸 1<<30 — dispatch._pad_orders 와 같은 값).
    """
    N = tt.n_max
    rows = []
    for b in range(len(tt.block_ids)):
        tb = tt.tables[b] if tw is None else HT.block_tables(tw, tt, b)
        p = DP.resolver_params(tb, g)
        rows.append(p._replace(name_rank_job=_pad_i32(p.name_rank_job, N, 1 << 30),
                               tok_rank=_pad_i32(p.tok_rank, N, 1 << 30)))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def _admission_schedule(tw: HT.TerminalWorld, tt: HT.TerminalTables) -> AD.ScheduleArrays:
    """host_terminal.ScheduleArrays (S,) → admission.ScheduleArrays (버킷 (E_grid,B,M)) — 같은 명단, 같은 순서 s."""
    S = tw.sched.s
    arr = np.asarray(tw.sched.arrival_s); tr = np.asarray(tw.sched.travel_s); tb = np.asarray(tw.sched.travel_base_s)
    blk = np.asarray(tw.sched.block); row = np.asarray(tw.sched.row)
    schedule = [{"job_id": tt.truck_ids[s], "block": tt.block_ids[int(blk[s])], "arrival_s": float(arr[s]),
                 "travel_s": float(tr[s]), "travel_base_s": float(tb[s])} for s in range(S)]
    row_of_job = {tt.truck_ids[s]: int(row[s]) for s in range(S)}
    period = float(tt.period_s)
    t_max = float(np.max(np.asarray(tw.epochs.t))) if tw.epochs.t.shape[0] else 0.0
    n_epochs = int(np.floor(t_max / period)) + 1                        # 격자 슬롯 수 (admission_epochs 와 같다)
    # ★통지 시각은 호스트가 트럭별 리드로 이미 계산해 뒀다 (host_terminal.to_terminal_world) — 다시 계산하지 않는다.
    #   그래야 V3Announcer 처럼 리드가 트럭마다 다른 명단에서도 버킷이 v5 와 같다.
    adm, info = AD.schedule_arrays(schedule, block_index=tt.block_index, row_of_job=row_of_job,
                                   lead_s=float(tt.lead_s), n_epochs=n_epochs, period_s=period,
                                   notify_s=np.asarray(tw.sched.notice_s))
    if info["dropped"] != 0:
        raise ValueError(f"명단 {info['dropped']} 건의 통지 슬롯이 검토 격자 밖이다 (E={n_epochs}) — 조용히 버리지 않는다")
    return adm


def steps_for(tt: HT.TerminalTables) -> int:
    """무대 크기에서 나오는 while_loop 상한 = 8·n_max + 256 (test_gpu_y01 의 상한과 같은 식).

    `Engine(steps_per_epoch=steps_for(tt), steps_final=steps_for(tt))` 로 쓴다 — 리터럴 기본값(2304/4096)은
    무대와 무관해 큰 무대에서 조용히 모자랄 수 있다 (머리말 ⚠️)."""
    return 8 * int(tt.n_max) + 256


def make_run(tw: HT.TerminalWorld, tt: HT.TerminalTables, *, params=None, max_open: int = MAX_OPEN) -> TerminalRun:
    """변환 직후의 `TerminalWorld` → 조정자 상태. `params` 는 정책 파라미터 (블록별 (B,…) 로 쌓은 것; None 가능)."""
    S = tw.sched.s
    B = tw.b
    return TerminalRun(tw=tw, adm=_admission_schedule(tw, tt), params=params,
                       locked=jnp.zeros((S,), bool), open_txn=jnp.full((int(max_open),), EMPTY_ID, jnp.int32),
                       open_overflow=jnp.zeros((), jnp.int32), hist_overflow=jnp.zeros((), jnp.int32),
                       n_admitted=jnp.zeros((), jnp.int32), exhausted=jnp.zeros((B,), jnp.int32))


# ───────────────────────────────────────────────── 블록 전진 (vmap × while_loop)
def _where_b(active, new, old):
    """블록 마스크 (B,) 로 잎마다 고른다 — 앞축이 B 인 pytree."""
    def pick(x, y):
        m = jnp.reshape(active, (active.shape[0],) + (1,) * (x.ndim - 1))
        return jnp.where(m, x, y)
    return jax.tree_util.tree_map(pick, new, old)


def _vstep(eng: Engine, params):
    """블록 묶음 한 스텝 (review=True). params 가 None 이면 in_axes None."""
    f = partial(ES.step, params=None, g=eng.g, policy_fn=eng.policy_fn, check=eng.check,
                pre_advice=eng.pre_advice, horizon_s=eng.horizon_s, joint=eng.joint, review=True)
    if params is None:
        return jax.vmap(lambda w: f(w, None))
    g = lambda w, p: ES.step(w, None, params=p, g=eng.g, policy_fn=eng.policy_fn, check=eng.check,
                             pre_advice=eng.pre_advice, horizon_s=eng.horizon_s, joint=eng.joint, review=True)
    return jax.vmap(g, in_axes=(0, 0))


def run_to_epoch(W: BlockWorld, params, eng: Engine, *, max_steps: int):
    """전 블록을 **다음 검토 시각까지** (또는 terminal 까지) 굴린다 — v5 179-206행에서 review_fn 이 열리기 직전 상태.

    반환 (W', parked (B,) bool, n_iter () int32, stuck (B,) bool [상한에 걸려 park 도 terminal 도 못 한 블록]).
    step 은 park 한 블록·terminal 블록에는 적용하지 않는다 (마스크) — 그 블록의 상태는 그대로다.
    """
    B = int(W.clock.shape[0])
    vstep = _vstep(eng, params)

    def cond(c):
        i, W_, parked = c
        return (i < int(max_steps)) & jnp.any(~parked & ~W_.terminal)

    def body(c):
        i, W_, parked = c
        active = ~parked & ~W_.terminal
        W2, tr = vstep(W_) if params is None else vstep(W_, params)
        W3 = _where_b(active, W2, W_)
        parked2 = parked | (active & tr.reviewed)
        return i + 1, W3, parked2

    i, W_out, parked = lax.while_loop(cond, body, (jnp.int32(0), W, jnp.zeros((B,), bool)))
    stuck = ~parked & ~W_out.terminal
    return W_out, parked, i, stuck


# ───────────────────────────────────────────────── 검토 시각 처리 (sync_locks · 투입 · 원장)
def epoch_time(run: TerminalRun, e) -> jnp.ndarray:
    """검토 시각 e 의 t (전 블록 공통 목록 · round(·,6) 값)."""
    E = int(run.tw.epochs.t.shape[0])
    return run.tw.epochs.t[jnp.clip(jnp.asarray(e, jnp.int32), 0, E - 1)]


def _grid_slot(run: TerminalRun, e):
    """검토 시각 → 투입 격자 슬롯 (-1 = 격자 밖 → announcer 804행 return). epochs.on_grid 가 v5 on_grid 와 같다."""
    E = int(run.tw.epochs.t.shape[0])
    ec = jnp.clip(jnp.asarray(e, jnp.int32), 0, E - 1)
    t = run.tw.epochs.t[ec]
    on = run.tw.epochs.on_grid[ec] & (jnp.asarray(e, jnp.int32) < E)
    return jnp.where(on, jnp.round(t / run.tw.epochs.period_s).astype(jnp.int32), EMPTY_ID)


def _owned_status(W: BlockWorld, L: HT.TerminalLedgerArrays):
    """등록 트럭의 소유 (owner,row) 행 status (S,)."""
    B, R = W.orders.block.shape
    oc = jnp.clip(L.owner, 0, B - 1)
    rc = jnp.clip(L.row, 0, R - 1)
    return W.orders.status[oc, rc]


def review_epoch(run: TerminalRun, e, eng: Engine):
    """검토 시각 e 의 review — v5 203행 `_sync_locks` + review_fn(= `ScheduledAnnouncer.review`) + 원장 등록.

    전 블록이 t_e 에 park 한 상태여야 한다 (`run_to_epoch` 뒤). 반환 (run', codes (B,M) int32 — admission 코드).
    """
    tw = run.tw
    W, L = tw.blocks, tw.ledger
    # ① _sync_locks (212-227행) — 등록 트럭이 PLANNED 가 아니면 lock.
    #    ★v5 는 **ReviewEpoch 를 돌려준 블록만** 동기화한다 (203행) — 이미 terminal 인 블록은 건드리지 않는다.
    #    블록마다 end 가 다른 무대에서 갈리므로 owner 기준 alive 마스크를 씌운다 (전 블록 end 가 같으면 이전과 동일).
    alive_b = ~W.terminal                                              # (B,) bool
    Bn = int(alive_b.shape[0])
    alive_s = alive_b[jnp.clip(L.owner, 0, Bn - 1)]                    # (S,) 소유 블록이 아직 살아 있나
    locked = run.locked | (L.registered & alive_s & (_owned_status(W, L) != JS_PLANNED))
    # ② 투입 (announcer 802-825행 + admit_external_job 243-291행) — 격자 밖이면 항등
    slot = _grid_slot(run, e)
    W2, codes = AD.admit_epoch(W, run.adm, slot, eng.g, margin=int(eng.margin), reserved=L.reserved_inbound,
                               end_ann=jnp.asarray(eng.end_ann, F), registered=L.registered)
    # ③ 전역 원장 등록 (290-291행): registered · a_gate_in = arrival (owner/row/origin 은 변환 때 채워져 있다)
    Eg, _, _ = run.adm.bucket.shape
    idx = jnp.where((slot >= 0) & (slot < Eg), run.adm.bucket[jnp.clip(slot, 0, Eg - 1)], EMPTY_ID)   # (B,M) 명단 색인
    hit = (codes == AD.ADMIT) & (idx >= 0)
    S = int(L.registered.shape[0])
    hit_s = jnp.zeros((S,), jnp.int32).at[jnp.clip(idx, 0, S - 1).reshape(-1)].max(hit.reshape(-1).astype(jnp.int32)) > 0
    L2 = L._replace(registered=L.registered | hit_s,
                    a_gate_in=jnp.where(hit_s, run.adm.arrival_s, L.a_gate_in))
    tw2 = tw._replace(blocks=W2, ledger=L2, epoch_idx=jnp.asarray(e, jnp.int32) + 1)
    return run._replace(tw=tw2, locked=locked, n_admitted=run.n_admitted + jnp.sum(hit).astype(jnp.int32)), codes


def epoch_step(run: TerminalRun, e, eng: Engine):
    """에폭 하나 = 전 블록을 t_e 까지 → review. 반환 (run', codes (B,M))."""
    W, parked, _, stuck = run_to_epoch(run.tw.blocks, run.params, eng, max_steps=eng.steps_per_epoch)
    W = W._replace(violation=W.violation | jnp.where(stuck, V_STEPS_EXHAUSTED, 0).astype(jnp.int32))
    run = run._replace(tw=run.tw._replace(blocks=W), exhausted=run.exhausted + stuck.astype(jnp.int32))
    return review_epoch(run, e, eng)


def run_epochs(run: TerminalRun, e0, n: int, eng: Engine):
    """에폭 e0 .. e0+n−1 을 `lax.scan` 으로. 반환 (run', codes (n,B,M)). n 은 static (jit 재컴파일 단위), e0 는 traced."""
    def body(r, e):
        return epoch_step(r, e, eng)
    return lax.scan(body, run, jnp.asarray(e0, jnp.int32) + jnp.arange(int(n), dtype=jnp.int32))


run_epochs_jit = jax.jit(run_epochs, static_argnames=("n", "eng"))


def _harvest_locked(run: TerminalRun) -> jnp.ndarray:
    """v5 `harvest` 80-82행: status ∈ {WAITING, ASSIGNED, RUNNING, DONE} 이면 lock (RELEASED 는 아님)."""
    st = _owned_status(run.tw.blocks, run.tw.ledger)
    lockable = (st == JS_WAITING) | (st == JS_ASSIGNED) | (st == JS_RUNNING) | (st == JS_DONE)
    return run.locked | (run.tw.ledger.registered & lockable)


def finish_run(run: TerminalRun, eng: Engine) -> TerminalRun:
    """마지막 검토 시각 뒤 — 전 블록을 terminal 까지 (v5 199-201행) → `engine_step.finish` → `harvest` (207행)."""
    W, parked, _, stuck = run_to_epoch(run.tw.blocks, run.params, eng, max_steps=eng.steps_final)
    # 검토 시각이 남아 있었다면(parked) 호출 순서 오류 — 그래도 조용히 넘기지 않고 exhausted 로 표시
    W = jax.vmap(ES.finish)(W)
    W = W._replace(violation=W.violation | jnp.where(parked, V_STEPS_EXHAUSTED, 0).astype(jnp.int32))
    run = run._replace(tw=run.tw._replace(blocks=W), exhausted=run.exhausted + (stuck | parked).astype(jnp.int32))
    return run._replace(locked=_harvest_locked(run))


finish_run_jit = jax.jit(finish_run, static_argnames=("eng",))


def run_all(run: TerminalRun, eng: Engine, *, chunk: int | None = None):
    """전 에폭 + 마무리를 한 번에 (호스트 루프 없음 · 세션 하나). 반환 (run', codes (E,B,M)).
    chunk 를 주면 scan 을 그 크기로 나눠 부른다 (컴파일 단위가 작아진다; 답은 같다)."""
    E = run.n_epochs
    if chunk is None or chunk >= E:
        run, codes = run_epochs_jit(run, 0, E, eng)
    else:
        parts = []
        e = 0
        while e < E:
            n = min(chunk, E - e)
            run, c = run_epochs_jit(run, e, n, eng)
            parts.append(c)
            e += n
        codes = jnp.concatenate(parts, axis=0)
    return finish_run_jit(run, eng), codes


epoch_step_jit = jax.jit(epoch_step, static_argnames=("eng",))


def run_python(run: TerminalRun, eng: Engine, *, review_fn=None, e0: int = 0, e1: int | None = None,
               finish: bool = True):
    """에폭마다 호스트로 돌아오는 루프 — review_fn(run, e, t) → run' 을 투입 **뒤** 에 부른다 (이송·이연 강제용).

    v5 대응: `mbt.run(policy, review_fn=hook)` 에서 hook 이 `ann.review(mbt, t)` 뒤에 try_* 를 부르는 골격
    (tests/v6/test_gpu_transfer_txn.py). 반환 (run', codes 목록[(B,M)…]).
    """
    E = run.n_epochs
    e1 = E if e1 is None else int(e1)
    codes = []
    for e in range(int(e0), e1):
        run, c = epoch_step_jit(run, e, eng)
        codes.append(c)
        if review_fn is not None:
            run = review_fn(run, e, float(epoch_time(run, e)))
    if finish and e1 >= E:
        run = finish_run_jit(run, eng)
    return run, codes


# ───────────────────────────────────────────────── 이송·이연 브리지 (transfer_txn.TxnLedger ↔ TerminalLedgerArrays)
def txn_ledger(run: TerminalRun, tt: HT.TerminalTables) -> TX.TxnLedger:
    """정본 원장 + locked/open_txn → `transfer_txn.TxnLedger` (열 이름 대응: origin↔origin · hist_*↔transfer_* ·
    hist_overflow↔hist_overflow). 미등록 트럭은 owner −1 (v5 records 에 없음)."""
    L = run.tw.ledger
    reg = L.registered
    cont_base = jnp.asarray([int(tb.c0) for tb in tt.tables], jnp.int32)
    return TX.TxnLedger(
        owner=jnp.where(reg, L.owner, EMPTY_ID).astype(jnp.int32), row=L.row, origin=L.origin, version=L.version,
        transfer_count=L.transfer_count, hist_src=L.transfer_src, hist_dst=L.transfer_dst, hist_t=L.transfer_t,
        locked=run.locked, entry_deferrals=L.entry_deferrals, entry_deferred_s=L.entry_deferred_s,
        reserved_inbound=L.reserved_inbound, cont_base=cont_base, route_cost_s=L.route_cost_s, txn_seq=L.txn_seq,
        open_txn=run.open_txn, open_overflow=run.open_overflow, hist_overflow=run.hist_overflow)


def with_txn_ledger(run: TerminalRun, X: TX.TxnLedger, worlds: BlockWorld | None = None) -> TerminalRun:
    """`txn_ledger` 의 역 — 이송·이연 결과를 정본에 되돌린다 (등록 여부는 그대로; owner 는 등록분만 갱신).

    ★`a_gate_in` 은 TxnLedger 가 열로 들지 않고 `orders.gate_in_s[owner,row]` 에서 파생한다 — 이연이 A 를 바꾸므로
    (v5 363행 `rec.a_gate_in = new_a`) 등록 트럭의 A 를 그 (새) 소유 행에서 다시 읽는다."""
    L = run.tw.ledger
    W = run.tw.blocks if worlds is None else worlds
    owner = jnp.where(L.registered, X.owner, L.owner).astype(jnp.int32)
    B, R = W.orders.block.shape
    a_now = W.orders.gate_in_s[jnp.clip(owner, 0, B - 1), jnp.clip(X.row, 0, R - 1)]
    L2 = L._replace(owner=owner, row=X.row, version=X.version, transfer_count=X.transfer_count,
                    transfer_src=X.hist_src, transfer_dst=X.hist_dst, transfer_t=X.hist_t,
                    a_gate_in=jnp.where(L.registered, a_now, L.a_gate_in),
                    entry_deferrals=X.entry_deferrals, entry_deferred_s=X.entry_deferred_s,
                    reserved_inbound=X.reserved_inbound, route_cost_s=X.route_cost_s, txn_seq=X.txn_seq)
    tw = run.tw._replace(ledger=L2, blocks=W)
    return run._replace(tw=tw, locked=X.locked, open_txn=X.open_txn, open_overflow=X.open_overflow,
                        hist_overflow=X.hist_overflow)


_J_PRE = jax.jit(TX.try_pre_gate_transfer, static_argnames=("g", "k0"))
_J_POST = jax.jit(TX.try_transfer, static_argnames=("g", "k0"))
_J_DEFER = jax.jit(TX.try_defer_admitted_entry)
_J_FREE_ROW = jax.jit(TX.first_free_row)


def _refresh_params(run: TerminalRun, tt: HT.TerminalTables, eng: Engine, blocks) -> TerminalRun:
    """이송 뒤 두 블록의 resolver 순위표를 원장 기준 번호표(block_tables)로 다시 굽는다 (여분 행의 트럭 이름)."""
    if run.params is None or not isinstance(run.params, DP.ResolverParams):
        return run
    p = run.params
    for b in blocks:
        q = DP.resolver_params(HT.block_tables(run.tw, tt, b), eng.g)
        p = p._replace(name_rank_job=p.name_rank_job.at[b].set(_pad_i32(q.name_rank_job, tt.n_max, 1 << 30)),
                       tok_rank=p.tok_rank.at[b].set(_pad_i32(q.tok_rank, tt.n_max, 1 << 30)),
                       name_rank_repo=p.name_rank_repo.at[b].set(q.name_rank_repo))
    return run._replace(params=p)


def try_pre_gate_transfer(run: TerminalRun, tt: HT.TerminalTables, eng: Engine, now, s: int, dst: int, *,
                          travel_s: float, route_delta_s: float = 0.0):
    """v5 `try_pre_gate_transfer(job_id, dst, travel_s, route_delta_s)` — 트럭 s 를 블록 dst 로 (게이트 진입 전).
    수신 행 = dst 사본의 첫 여분 행 (host_terminal 규약 n_spare). 반환 (run', ok, code)."""
    X = txn_ledger(run, tt)
    W = run.tw.blocks
    dst_row = _J_FREE_ROW(W, int(dst), int(tt.n_used[int(dst)]))
    W2, X2, ok, code = _J_PRE(W, X, float(now), int(s), int(dst), travel_s=float(travel_s),
                              route_delta_s=float(route_delta_s), g=eng.g, k0=int(eng.k0),
                              capacity_margin=int(eng.margin), max_transfers=int(eng.max_transfers), dst_row=dst_row)
    run2 = with_txn_ledger(run, X2, W2)
    if bool(ok):
        run2 = _refresh_params(run2, tt, eng, (int(np.asarray(X.owner)[s]), int(dst)))
    return run2, bool(ok), int(code)


def try_transfer(run: TerminalRun, tt: HT.TerminalTables, eng: Engine, now, s: int, dst: int, *,
                 route_s: float, travel_s: float):
    """v5 `try_transfer` (게이트 진입 후). 반환 (run', ok, code)."""
    X = txn_ledger(run, tt)
    W = run.tw.blocks
    dst_row = _J_FREE_ROW(W, int(dst), int(tt.n_used[int(dst)]))
    W2, X2, ok, code = _J_POST(W, X, float(now), int(s), int(dst), route_s=float(route_s), travel_s=float(travel_s),
                               g=eng.g, k0=int(eng.k0), capacity_margin=int(eng.margin), dst_row=dst_row)
    run2 = with_txn_ledger(run, X2, W2)
    if bool(ok):
        run2 = _refresh_params(run2, tt, eng, (int(np.asarray(X.owner)[s]), int(dst)))
    return run2, bool(ok), int(code)


def try_defer_admitted_entry(run: TerminalRun, tt: HT.TerminalTables, now, s: int, delta_s: float, *,
                             max_deferrals: int = 1):
    """v5 `try_defer_admitted_entry(job_id, delta_s)`. 반환 (run', ok, code)."""
    X = txn_ledger(run, tt)
    W2, X2, ok, code = _J_DEFER(run.tw.blocks, X, float(now), int(s), float(delta_s), max_deferrals=int(max_deferrals))
    return with_txn_ledger(run, X2, W2), bool(ok), int(code)


def check_invariants(run: TerminalRun, tt: HT.TerminalTables, *, conserve: bool = True) -> bool:
    """v5 `check_invariants` (653-666행) 의 배열판 (transfer_txn.check_invariants) — True 면 통과.

    ★`conserve=True` (기본): '살아 있는 외부트럭 행 수 == 등록 수' 보존식까지 본다 — 조정자 원장은 명단 트럭만
      담으므로 등식이 정확하다. 시나리오가 처음부터 외부트럭을 들고 있는 무대에서만 False 로 내린다."""
    return bool(TX.check_invariants(run.tw.blocks, txn_ledger(run, tt), conserve=conserve))


# ───────────────────────────────────────────────── 결과 (v5 run_terminal 모양) · 저장
def from_run(run: TerminalRun, tt: HT.TerminalTables) -> dict:
    """`host_terminal.from_terminal_world` + 조정자 계수 (n_admitted · exhausted · locked · 트랜잭션 넘침).

    ★잎을 먼저 numpy 로 내린다 — from_block_world 는 원소마다 색인하므로 jax 배열 그대로면 21블록·수천 원소에
    장치 왕복이 그만큼 생겨 분 단위로 느려진다 (numpy 면 초 단위)."""
    d = HT.from_terminal_world(tree_to_numpy(run.tw), tt)
    d["admitted"] = int(run.n_admitted)
    d["exhausted"] = [int(x) for x in np.asarray(run.exhausted)]
    d["open_overflow"] = int(run.open_overflow)
    d["hist_overflow"] = int(run.hist_overflow)
    d["locked"] = {tt.truck_ids[s]: bool(v) for s, v in enumerate(np.asarray(run.locked))
                   if bool(np.asarray(run.tw.ledger.registered)[s])}
    return d


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def save_run(path, run: TerminalRun) -> None:
    """numpy 잎으로 pickle (NamedTuple 구조 포함 — 템플릿 없이 복원). 세션 이어 돌리기용."""
    with open(path, "wb") as f:
        pickle.dump(tree_to_numpy(run), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_run(path) -> TerminalRun:
    """★x64 를 먼저 확인한다 — 꺼진 세션에서 복원하면 float64 잎이 조용히 float32 가 되고 값 비교는 못 잡는다."""
    from . import state as ST
    ST.check_x64()
    with open(path, "rb") as f:
        saved = pickle.load(f)
    run = tree_to_jax(saved)
    paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_leaves_with_path(run)]
    bad = [f"{paths[i]}: 저장 {np.asarray(y).dtype} → 복원 {np.asarray(x).dtype}"
           for i, (x, y) in enumerate(zip(jax.tree_util.tree_leaves(run), jax.tree_util.tree_leaves(saved)))
           if np.asarray(x).dtype != np.asarray(y).dtype]
    if bad:
        raise RuntimeError("복원 dtype 이 저장과 다르다 (x64 가 꺼졌나?) " + "; ".join(bad[:5]))
    return run
