"""이송/이연 원자 확정 — 배열판(gpu/transfer_txn.py)이 v5 multiblock 과 **같은 답**을 내는가 ([[YR-327]] 조각 6 · key=txn).

■ 무대 (명세 equivalence_test 2단계)
  layout = terminal_layout().subset(("Y01","Y21")) · obs = ObservationContract(0, 7200, 300) · build_diurnal(seed 9_900_777,
  load_4h=40, day_total=40, n_streams=0, drain 1200) → 트럭 40 · 본선 0 · 크레인 2. v5 = test_world_equivalence.py:38-56 골격
  (MultiBlockTerminal + ensure_time_ledger + admission_epochs + ScheduledAnnouncer(lead 1800) + ResolverPolicy(SF_SPT)
  + CandidateGenerator(LEGACY_DEFAULT)). 본선 0 이라 vessels.py 의 set max 비결정론(PYTHONHASHSEED)에 걸리지 않는다.

■ 방법 — v5 를 실제로 굴리며 검토 시각(review_fn)마다 **v5 상태를 배열로 옮겨** 같은 질의를 양쪽에 넣는다
  변환기 `Harness.convert(mbt)` 는 이송·이연이 읽고 쓰는 부분(오더 행 전열·큐·격자/컨테이너·시계·원장)을 v5 객체에서
  그대로 읽는다 — 크레인·계획·KPI 등은 reset 값(`to_block_world`)이라 앞뒤가 같다. 질의 뒤 **v5 를 다시 변환해** 배열
  결과와 잎 전부 `==` (큐는 (시각·종류·대상·순번) 정렬열 + 카운터). 거절 사유는 v5 예외 문구 → 코드(`reason_code`) 로
  사상해 배열 코드와 `==`. 기대값을 손으로 적지 않는다.

■ 시험
  ① forced   t=1800 부근 try_pre_gate_transfer 1건 (Y01→Y21, route 220s) · t=2400 부근 try_defer 1건 (Δ600) — 성공 뒤
             owner/version/transfer_history/entry_deferrals/entry_deferred_s/route_cost_s/큐/오더 행 == (명세 (6))
  ② random   검토 시각 12곳 × 의도 목록(성공·미등록·부적격·자격·상한·창·용량·도착·음수Δ·창밖…) ≥ 30 질의 — 코드·상태 == ,
             거절 사유 전 종류(구조상 불가한 셋 제외) 가 실제로 나왔는지 집계
  ③ staged   deepcopy 무대에서 prepare→(version 증가|lock|상태 변경|rollback|이중 commit)→commit — validate/commit 단계 코드,
             lock 해제 조작으로 '소스 상태 위반'·'상태 위반', 야드 조작(빈 칸에 FT20·FT40 칸 만재)으로 '규격 적합 슬롯 없음'
  ④ ledger   300초마다 free_slots (B,) · sync_locks · check_invariants == v5
실행: WSL venv · x64 CPU (`JAX_PLATFORMS=cpu`). 전체 ≈ 40초 (v5 4회 구동 + jit 컴파일).
"""
from __future__ import annotations

import copy
import os
import random
from functools import lru_cache

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import transfer_txn as TX                                       # noqa: E402
from yard_rl.v6.gpu.events import EMPTY_ID, EMPTY_TIME, empty_queue                # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import EV_NAMES, FLOW_NAMES, STATUS_NAMES, to_block_world   # noqa: E402
from yard_rl.v6.gpu.stack_ops import SIZE_INDEX, from_v5_stacks                    # noqa: E402
from yard_rl.v6.gpu.state import JS_RELEASED, SZ_FT20, SZ_FT40                      # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, InformationLevel, JobStatus, LoadStatus   # noqa: E402
from yard_rl.v6.world.domain.models import Container                                # noqa: E402
from yard_rl.v6.world.integrated import (baselines as bl, candidates as cd, engine as eng,   # noqa: E402
                                          multiblock as mb, policy_config as pc, profiles as pr,
                                          terminal_stream as ts, yard_layout as yl)

SEED = 9_900_777
BLOCKS = ("Y01", "Y21")
DAY_TOTAL = 40
LEAD_S = 1800.0
EXTRA_CONT = 1500          # 야드 조작(③) 컨테이너 여분 칸
MAX_T = 8                  # transfer_history 칸 — 진입 후 이송(prepare_transfer)은 상한이 없어 한 트럭이 여러 번 옮겨질 수 있다
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── v5 무대·구동
@lru_cache(maxsize=None)
def harness():
    return Harness()


class Harness:
    """v5 무대 1회 빌드 + 블록별 배열 세계 바탕(reset) + 번호표. v5 구동은 `make_mbt` 로 매번 새로."""

    def __init__(self):
        self.layout = yl.terminal_layout().subset(BLOCKS)
        self.obs = ts.ObservationContract(warmup_s=0.0, measure_s=7200.0, snapshot_s=300.0)
        self.prof = pr.build_h21_profile()
        self.built = ts.build_diurnal(self.prof, SEED, obs=self.obs, layout=self.layout,
                                      params=ts.TerminalStreamParams(load_4h=DAY_TOTAL),
                                      day_total=DAY_TOTAL, n_streams=0, drain_s=1200.0, background_seed=SEED)
        self.g = Geom.from_profile(self.prof)
        self.block_ids = tuple(self.layout.ids)
        self.bidx = {b: i for i, b in enumerate(self.block_ids)}
        scn_jobs = [f"{b}:{j.job_id}" for b in self.block_ids for j in self.built["scenarios"][b].jobs]
        self.job_ids = tuple(sorted([e["job_id"] for e in self.built["schedule"]] + scn_jobs))
        self.jidx = {j: i for i, j in enumerate(self.job_ids)}
        self.N = len(self.job_ids)
        self.size_of = {e["job_id"]: (SZ_FT40 if e["size_ft40"] else SZ_FT20)
                        for e in self.built["schedule"] if e["flow"] == "GATE_IN"}
        self.init_ids = [sorted(self.built["scenarios"][b].containers) for b in self.block_ids]
        self.c0 = [len(x) for x in self.init_ids]
        self.c_max = max(self.c0) + self.N + EXTRA_CONT
        self.q_cap = 4 * self.N + 64
        self.base = []
        for b in self.block_ids:
            w0, tb = to_block_world(self.prof, self.built["scenarios"][b], n_max=self.N, q_cap=self.q_cap,
                                    log_cap=8 * self.N + 256, c_max=self.c_max)
            self.base.append((w0, tb))
        self.crane_index = self.base[0][1].crane_index
        self.k0 = self.crane_index[self.prof.cranes[0].crane_id]      # multiblock.py:485 profile.cranes[0]

    def make_mbt(self):
        lvl = InformationLevel.PRE_ADVICE

        def sim_from(scn):
            s = eng.TerminalSimulator(self.prof, scn, check_invariants=True)
            s.info_level = lvl
            return s

        mbt = mb.MultiBlockTerminal(
            {b: ts.ensure_time_ledger(sim_from(s)) for b, s in self.built["scenarios"].items()},
            extra_review_epochs=ts.admission_epochs(self.obs))
        ann = ts.ScheduledAnnouncer(self.built["schedule"], lead_s=LEAD_S, end_s=self.built["sim_end_s"])
        gens: dict[int, object] = {}
        pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
            gb = {c: g.generate(sim, c, lvl) for c in dp.crane_ids}
            bl._apply(sim, pol.decide(sim, dp, gb))
        return mbt, ann, exec_policy

    # ── v5 → 배열 (이송·이연이 읽고 쓰는 부분만 런 중 상태에서; 나머지는 reset 값) ──
    def cont_ids(self, b: int, extra=None) -> list[str]:
        ids = self.init_ids[b] + [f"IN_{j}" for j in self.job_ids] + list(extra or [])
        if len(ids) > self.c_max:
            raise ValueError(f"컨테이너 {len(ids)} > c_max {self.c_max}")
        return ids + [f"PAD_#{i}" for i in range(len(ids), self.c_max)]

    def convert(self, mbt, extra_cont: dict[int, list[str]] | None = None):
        ws = []
        for b, bid in enumerate(self.block_ids):
            sim = mbt.blocks[bid]
            w0, tb = self.base[b]
            cids = self.cont_ids(b, (extra_cont or {}).get(b))
            cidx = {c: i for i, c in enumerate(cids)}
            o = {f: np.asarray(getattr(w0.orders, f)).copy() for f in w0.orders._fields}
            tl = sim.time_ledger
            for jid, j in sim.jobs.items():
                n = self.jidx[jid]
                is_store = j.inbound_size is not None
                rec = tl.records.get(jid)
                o["block"][n] = b
                o["flow"][n] = FLOW_NAMES.index(j.flow.value)
                o["status"][n] = STATUS_NAMES.index(j.status.name)
                o["is_external"][n] = j.is_external_truck
                o["is_vessel"][n] = j.is_vessel_linked
                o["is_store"][n] = is_store
                o["target_cont"][n] = cidx[j.target_container] if j.target_container is not None else EMPTY_ID
                o["inbound_cont"][n] = self.c0[b] + n if is_store else EMPTY_ID
                o["inbound_size"][n] = SIZE_INDEX[j.inbound_size.value] if is_store else EMPTY_ID
                o["vessel"][n] = EMPTY_ID
                o["assigned_crane"][n] = self.crane_index.get(j.assigned_crane, EMPTY_ID) if j.assigned_crane else EMPTY_ID
                o["rehandles"][n] = j.rehandle_count
                o["release_s"][n] = float(j.release_time)
                o["provided_eta_s"][n] = EMPTY_TIME if j.provided_eta is None else float(j.provided_eta)
                o["deadline_s"][n] = EMPTY_TIME if j.deadline is None else float(j.deadline)
                o["exit_travel_s"][n] = -1.0 if j.exit_travel_s is None else float(j.exit_travel_s)
                o["actual_arrival_s"][n] = EMPTY_TIME if j.actual_block_arrival is None else float(j.actual_block_arrival)
                o["gate_in_s"][n] = rec.gate_in if rec is not None else EMPTY_TIME
                o["block_in_s"][n] = (rec.block_arrival if rec is not None and rec.block_arrival is not None else EMPTY_TIME)
                o["service_s"][n] = EMPTY_TIME if j.service_start is None else float(j.service_start)
                o["done_s"][n] = EMPTY_TIME if j.service_end is None else float(j.service_end)
                o["gate_out_s"][n] = (rec.gate_out if rec is not None and rec.gate_out is not None else EMPTY_TIME)
                o["waiting"][n] = jid in sim.kpis._waiting
                o["in_block"][n] = jid in tl._in_block
            orders = w0.orders._replace(**{f: jnp.asarray(v) for f, v in o.items()})
            stacks, conts, _ = from_v5_stacks(sim.stacks, self.g, cont_ids=cids, n_cont=self.c_max)
            c_size = np.asarray(conts.c_size).copy()
            for jid, sz in self.size_of.items():                    # 예비칸 규격 — 전 행·전 블록 미리
                c_size[self.c0[b] + self.jidx[jid]] = sz
            conts = conts._replace(c_size=jnp.asarray(c_size))
            queue = self._queue_from_v5(sim, tb)
            ws.append(w0._replace(clock=jnp.asarray(sim.clock, jnp.float64), end_s=jnp.asarray(sim.end, jnp.float64),
                                  terminal=jnp.asarray(bool(sim._terminal)), orders=orders, stacks=stacks,
                                  conts=conts, queue=queue))
        worlds = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ws)
        ledger = TX.txn_ledger_from_v5(mbt, self.jidx, self.bidx, self.c0, n=self.N, max_t=MAX_T)
        return worlds, ledger

    def _queue_from_v5(self, sim, tb):
        """v5 힙 → 큐 배열. 배열 순번 = v5 seq − 1 (v5 는 push 전에 +1, 배열은 push 뒤 +1 — 시드 규칙과 같다)."""
        q = empty_queue(self.q_cap)
        time = np.full((self.q_cap,), EMPTY_TIME, np.float64); kind = np.full((self.q_cap,), EMPTY_ID, np.int32)
        target = np.full((self.q_cap,), EMPTY_ID, np.int32); seq = np.full((self.q_cap,), EMPTY_ID, np.int32)
        ents = sorted(sim.queue._heap, key=lambda e: (e.time, e.priority, e.seq))
        assert len(ents) <= self.q_cap
        for i, e in enumerate(ents):
            k = EV_NAMES.index(e.kind_name)
            if k in (5, 6, 7, 10):
                tg = self.jidx[e.payload]
            elif k in (0, 1, 2):
                tg = self.crane_index.get(e.payload, EMPTY_ID)
            else:
                tg = EMPTY_ID
            time[i], kind[i], target[i], seq[i] = e.time, k, tg, e.seq - 1
        return q._replace(time=jnp.asarray(time), kind=jnp.asarray(kind), target=jnp.asarray(target),
                          seq=jnp.asarray(seq), counter=jnp.asarray(int(sim.queue._seq), jnp.int32))

    # ── 후보 고르기 (v5 상태에서) ──
    def candidates(self, mbt, t: float, *, pre_gate: bool, flow: str | None = "GATE_IN", owner: str | None = None):
        """등록·PLANNED·미lock 이고 (pre_gate: A > t+1e-6 · else A ≤ t+1e-6) 인 작업 id 정렬열."""
        out = []
        for jid in self.job_ids:
            rec = mbt.ledger.records.get(jid)
            if rec is None or rec.locked or (owner is not None and rec.owner != owner):
                continue
            if flow is not None and rec.flow != flow:
                continue
            j = mbt.blocks[rec.owner].jobs.get(jid)
            if j is None or j.status != JobStatus.PLANNED or rec.a_gate_in is None:
                continue
            if pre_gate != (rec.a_gate_in > t + 1e-6):
                continue
            out.append(jid)
        return out


# ───────────────────────────────────────────────── v5 호출 (try_* 와 같은 순서 · 사유 포착)
def v5_try_pre_gate(mbt, jid, dst, *, travel_s, route_delta_s, margin=2, max_t=1):
    mbt.capacity_margin = margin
    txn = None
    try:
        txn = mbt.prepare_pre_gate_transfer(jid, dst, travel_s=travel_s, route_delta_s=route_delta_s,
                                            max_transfers=max_t)
        mbt.commit(txn)
        return True, TX.OK
    except mb.TransferError as ex:
        if txn is not None:
            mbt.rollback(txn)
        return False, TX.reason_code(str(ex))
    finally:
        mbt.capacity_margin = mb.CAPACITY_MARGIN


def v5_try_post(mbt, jid, dst, *, route_s, travel_s, margin=2):
    mbt.capacity_margin = margin
    try:
        try:
            txn = mbt.prepare_transfer(jid, dst, route_s=route_s, travel_s=travel_s)
        except mb.TransferError as ex:
            return False, TX.reason_code(str(ex))
        try:
            mbt.commit(txn)
            return True, TX.OK
        except mb.TransferError as ex:
            mbt.rollback(txn)
            return False, TX.reason_code(str(ex))
    finally:
        mbt.capacity_margin = mb.CAPACITY_MARGIN


def v5_try_defer(mbt, jid, delta, *, max_d=1):
    try:
        mbt.defer_admitted_entry(jid, delta, max_deferrals=max_d)
        return True, TX.OK
    except mb.TransferError as ex:
        return False, TX.reason_code(str(ex))


# ───────────────────────────────────────────────── 배열 호출 (jit)
J_PRE = jax.jit(TX.try_pre_gate_transfer, static_argnames=("g", "k0"))
J_POST = jax.jit(TX.try_transfer, static_argnames=("g", "k0"))
J_DEFER = jax.jit(TX.try_defer_admitted_entry)
J_PREP_PRE = jax.jit(TX.prepare_pre_gate_transfer)
J_PREP_POST = jax.jit(TX.prepare_transfer)
J_COMMIT = jax.jit(TX.commit, static_argnames=("g", "k0"))
J_ROLLBACK = jax.jit(TX.rollback)
J_FREE = jax.jit(TX.free_slots)
J_SYNC = jax.jit(TX.sync_locks)
J_INV = jax.jit(TX.check_invariants)


# ───────────────────────────────────────────────── 대조
def _queue_view(worlds, b: int):
    q = worlds.queue
    t = np.asarray(q.time[b]); k = np.asarray(q.kind[b]); tg = np.asarray(q.target[b]); sq = np.asarray(q.seq[b])
    ents = sorted((float(t[i]), int(k[i]), int(tg[i]), int(sq[i])) for i in range(t.shape[0]) if np.isfinite(t[i]))
    return ents, int(q.counter[b]), int(q.overflow[b])


def _diff(a, b, *, skip=("queue",)) -> list[str]:
    """두 pytree 의 갈리는 잎 이름 (skip 경로 제외)."""
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(a)]
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    out = []
    for nm, x, y in zip(names, la, lb):
        if any(s in nm for s in skip):
            continue
        if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True):
            out.append(nm)
    return out


def _assert_same(H, label, worlds, ledger, mbt, extra_cont=None):
    """배열 상태 == v5 를 다시 변환한 상태 (오더·격자·컨테이너·시계 잎 전부 · 큐 정렬열 · 원장 잎 전부)."""
    w5, l5 = H.convert(mbt, extra_cont)
    bad = _diff(worlds, w5)
    if bad:
        detail = []
        for nm in bad[:6]:
            xa = [x for p, x in jax.tree_util.tree_leaves_with_path(worlds) if jax.tree_util.keystr(p) == nm][0]
            xb = [x for p, x in jax.tree_util.tree_leaves_with_path(w5) if jax.tree_util.keystr(p) == nm][0]
            idx = np.argwhere(~np.isclose(np.asarray(xa, float), np.asarray(xb, float), equal_nan=True))[:3]
            detail.append(f"{nm} at {idx.tolist()}: arr={np.asarray(xa)[tuple(idx.T)] if idx.size else '?'} "
                          f"v5={np.asarray(xb)[tuple(idx.T)] if idx.size else '?'}")
        pytest.fail(f"[{label}] 세계 잎이 갈린다 ({len(bad)}개): {bad[:12]}\n  " + "\n  ".join(detail))
    for b in range(len(H.block_ids)):
        qa, qb = _queue_view(worlds, b), _queue_view(w5, b)
        assert qa == qb, f"[{label}] 큐 {H.block_ids[b]} 갈림\n  arr={qa}\n  v5 ={qb}"
    badl = _diff(ledger, l5, skip=())
    assert not badl, (f"[{label}] 원장 잎이 갈린다 {badl}\n  arr={TX.txn_ledger_to_v5(ledger, H.job_ids, H.block_ids)}"
                      f"\n  v5 ={TX.txn_ledger_to_v5(l5, H.job_ids, H.block_ids)}")


def _same_code(label, ok, code, ok5, code5):
    assert (bool(ok), int(code)) == (ok5, code5), \
        f"[{label}] 판정 갈림: arr=({bool(ok)}, {TX.REASON_NAMES.get(int(code), int(code))}) " \
        f"v5=({ok5}, {TX.REASON_NAMES.get(code5, code5)})"


# ───────────────────────────────────────────────── ① 강제 이송 1건 · 이연 1건 (명세 2단계)
def test_forced_pre_gate_transfer_and_defer_match_v5():
    H = harness()
    mbt, ann, pol = H.make_mbt()
    done = {"xfer": None, "defer": None}
    lay = H.layout

    def hook(m, t):
        ann.review(m, t)
        if done["xfer"] is None and t >= 1800.0:
            for src in H.block_ids:
                cands = H.candidates(m, t, pre_gate=True, owner=src)
                if cands:
                    break
            else:
                return
            jid, dst = cands[0], [b for b in H.block_ids if b != src][0]
            worlds, ledger = H.convert(m)
            assert TX.terminal_now(worlds) == t
            travel, route = lay.gate_to_block_s(dst), lay.pre_gate_route_delta_s(src, dst)
            ok5, code5 = v5_try_pre_gate(m, jid, dst, travel_s=travel, route_delta_s=route)
            assert ok5, f"v5 강제 이송 실패 {TX.REASON_NAMES.get(code5)} — 무대가 명세와 다르다"
            w2, l2, ok, code = J_PRE(worlds, ledger, t, H.jidx[jid], H.bidx[dst], travel_s=travel, route_delta_s=route,
                                     g=H.g, k0=H.k0, capacity_margin=2, max_transfers=1)
            _same_code("forced-xfer", ok, code, ok5, code5)
            _assert_same(H, "forced-xfer", w2, l2, m)
            rec = m.ledger.records[jid]
            j = m.blocks[dst].jobs[jid]
            assert rec.owner == dst and rec.version == 1 and rec.transfer_count == 1 \
                and rec.transfer_history == ((src, dst, t),) and m.route_cost_s == route
            assert j.estimated_block_arrival == j.provided_eta          # 배열은 provided_eta_s 한 열로 둘을 대표
            L = TX.txn_ledger_to_v5(l2, H.job_ids, H.block_ids)
            assert L["records"][jid]["transfer_history"] == ((src, dst, t),) and L["route_cost_s"] == route
            assert bool(J_INV(w2, l2))
            done["xfer"] = dict(t=t, job=jid, src=src, dst=dst, route_s=route, arrival=j.actual_block_arrival)
        elif done["xfer"] is not None and done["defer"] is None and t >= 2400.0:
            cands = H.candidates(m, t, pre_gate=True, flow=None)
            if not cands:
                return
            jid = cands[0]
            worlds, ledger = H.convert(m)
            ok5, code5 = v5_try_defer(m, jid, 600.0)
            assert ok5, f"v5 강제 이연 실패 {TX.REASON_NAMES.get(code5)}"
            w2, l2, ok, code = J_DEFER(worlds, ledger, t, H.jidx[jid], 600.0, max_deferrals=1)
            _same_code("forced-defer", ok, code, ok5, code5)
            _assert_same(H, "forced-defer", w2, l2, m)
            rec = m.ledger.records[jid]
            assert rec.entry_deferrals == 1 and rec.entry_deferred_s == 600.0
            assert bool(J_INV(w2, l2))
            done["defer"] = dict(t=t, job=jid, a=rec.a_gate_in)

    out = mbt.run(pol, review_fn=hook)
    assert done["xfer"] is not None, "이송 후보(진입 전 반입)가 한 번도 없었다"
    assert done["defer"] is not None, "이연 후보가 없었다"
    assert ann.n_admitted == DAY_TOTAL
    assert done["xfer"]["route_s"] == H.layout.pre_gate_route_delta_s("Y01", "Y21") or \
        done["xfer"]["route_s"] == H.layout.pre_gate_route_delta_s("Y21", "Y01")
    REPORT["forced"] = dict(**{f"xfer_{k}": v for k, v in done["xfer"].items()},
                            **{f"defer_{k}": v for k, v in done["defer"].items()},
                            terminal_total=out["terminal_total"], route_cost_s=out["route_cost_s"])


# ───────────────────────────────────────────────── ② 무작위 질의 — 거절 사유 전 종류
INTENTS = ("pre_ok", "unreg", "bad_dst", "not_reass", "already_in", "capacity", "bad_arrival",
           "before_in", "post_ok", "defer_ok", "defer_neg", "defer_window", "defer_unreg", "defer_already",
           "random", "random")
QUERY_EPOCHS = tuple(float(t) for t in range(600, 7201, int(os.environ.get("TXN_EPOCH_STEP", "600"))))   # GPU 등 느린 환경: 큰 걸음
REQUIRED = {("pre", TX.OK), ("defer", TX.OK), ("pre", TX.R_UNREGISTERED), ("pre", TX.R_BAD_DST),
            ("pre", TX.R_NOT_REASSIGNABLE), ("pre", TX.R_MAX_TRANSFERS), ("pre", TX.R_ALREADY_GATE_IN),
            ("post", TX.R_BEFORE_GATE_IN), ("pre", TX.R_CAPACITY), ("pre", TX.R_BAD_ARRIVAL),
            ("defer", TX.R_NONPOSITIVE_DELTA), ("defer", TX.R_MAX_DEFERRALS), ("defer", TX.R_OUT_OF_WINDOW),
            ("defer", TX.R_ALREADY_GATE_IN), ("defer", TX.R_UNREGISTERED)}


def _make_query(H, mbt, t, intent, rng):
    """의도 → 질의 dict (없으면 None). 후보는 v5 상태에서 고른다."""
    lay, ids = H.layout, H.block_ids
    other = lambda b: [x for x in ids if x != b][0]
    pre = H.candidates(mbt, t, pre_gate=True)
    post = H.candidates(mbt, t, pre_gate=False)
    reg_out = [j for j in H.job_ids if j in mbt.ledger.records and mbt.ledger.records[j].flow == "GATE_OUT"]
    unreg = [j for j in H.job_ids if j not in mbt.ledger.records]
    defer_c = H.candidates(mbt, t, pre_gate=True, flow=None)
    owner = lambda j: mbt.ledger.records[j].owner
    if intent == "random":
        kind = rng.choice(("pre", "post", "defer"))
        j = rng.choice(H.job_ids)
        d = rng.choice(ids)
        travel = rng.choice((lay.gate_to_block_s(d), lay.gate_to_block_s(d), 1e6))
        margin = rng.choice((2, 2, 2, 10 ** 6))
        if kind == "defer":
            return dict(kind=kind, job=j, delta=rng.choice((600.0, -5.0, 1e6)), max_d=rng.choice((1, 1, 0)))
        if kind == "post":
            src = owner(j) if j in mbt.ledger.records else ids[0]
            return dict(kind=kind, job=j, dst=d, travel=travel, route=lay.post_gate_route_s(src, d), margin=margin)
        src = owner(j) if j in mbt.ledger.records else ids[0]
        return dict(kind=kind, job=j, dst=d, travel=travel, route=lay.pre_gate_route_delta_s(src, d),
                    margin=margin, max_t=rng.choice((1, 1, 0)))
    if intent in ("pre_ok", "bad_dst", "capacity", "bad_arrival"):
        if not pre:
            return None
        j = rng.choice(pre); s = owner(j)
        d = s if intent == "bad_dst" else other(s)
        return dict(kind="pre", job=j, dst=d, travel=(1e6 if intent == "bad_arrival" else lay.gate_to_block_s(d)),
                    route=lay.pre_gate_route_delta_s(s, d), margin=(10 ** 6 if intent == "capacity" else 2), max_t=1)
    if intent == "unreg":
        if not unreg:
            return None
        j = rng.choice(unreg)
        return dict(kind="pre", job=j, dst=ids[1], travel=lay.gate_to_block_s(ids[1]), route=0.0, margin=2, max_t=1)
    if intent == "not_reass":
        if not reg_out:
            return None
        j = rng.choice(reg_out); s = owner(j); d = other(s)
        return dict(kind="pre", job=j, dst=d, travel=lay.gate_to_block_s(d), route=lay.pre_gate_route_delta_s(s, d),
                    margin=2, max_t=1)
    if intent in ("already_in", "post_ok"):
        if not post:
            return None
        j = rng.choice(post); s = owner(j); d = other(s)
        if intent == "already_in":
            return dict(kind="pre", job=j, dst=d, travel=lay.gate_to_block_s(d),
                        route=lay.pre_gate_route_delta_s(s, d), margin=2, max_t=1)
        return dict(kind="post", job=j, dst=d, travel=lay.gate_to_block_s(d), route=lay.post_gate_route_s(s, d), margin=2)
    if intent == "before_in":
        if not pre:
            return None
        j = rng.choice(pre); s = owner(j); d = other(s)
        return dict(kind="post", job=j, dst=d, travel=lay.gate_to_block_s(d), route=lay.post_gate_route_s(s, d), margin=2)
    if intent == "defer_unreg":
        return dict(kind="defer", job=rng.choice(unreg), delta=600.0, max_d=1) if unreg else None
    if intent == "defer_already":
        past = [j for j in H.job_ids if j in mbt.ledger.records and mbt.ledger.records[j].a_gate_in <= t + 1e-6]
        return dict(kind="defer", job=rng.choice(past), delta=600.0, max_d=1) if past else None
    if intent.startswith("defer"):
        if not defer_c:
            return None
        j = rng.choice(defer_c)
        delta = {"defer_ok": 600.0, "defer_neg": -5.0, "defer_window": 1e6}[intent]
        return dict(kind="defer", job=j, delta=delta, max_d=1)
    raise KeyError(intent)


def _apply_query(H, mbt, worlds, ledger, t, q):
    """같은 질의를 v5 와 배열에 — (worlds', ledger', ok, code, ok5, code5)."""
    n = H.jidx[q["job"]]
    if q["kind"] == "pre":
        ok5, c5 = v5_try_pre_gate(mbt, q["job"], q["dst"], travel_s=q["travel"], route_delta_s=q["route"],
                                  margin=q["margin"], max_t=q["max_t"])
        w2, l2, ok, c = J_PRE(worlds, ledger, t, n, H.bidx[q["dst"]], travel_s=q["travel"], route_delta_s=q["route"],
                              g=H.g, k0=H.k0, capacity_margin=q["margin"], max_transfers=q["max_t"])
    elif q["kind"] == "post":
        ok5, c5 = v5_try_post(mbt, q["job"], q["dst"], route_s=q["route"], travel_s=q["travel"], margin=q["margin"])
        w2, l2, ok, c = J_POST(worlds, ledger, t, n, H.bidx[q["dst"]], route_s=q["route"], travel_s=q["travel"],
                               g=H.g, k0=H.k0, capacity_margin=q["margin"])
    else:
        ok5, c5 = v5_try_defer(mbt, q["job"], q["delta"], max_d=q["max_d"])
        w2, l2, ok, c = J_DEFER(worlds, ledger, t, n, q["delta"], max_deferrals=q["max_d"])
    return w2, l2, ok, c, ok5, c5


def test_random_queries_match_v5():
    """검토 시각 12곳에서 의도 목록 + 무작위 질의를 v5·배열 양쪽에 lockstep 으로 — 판정·상태 == , 사유 전 종류 집계."""
    H = harness()
    mbt, ann, pol = H.make_mbt()
    rng = random.Random(SEED)
    seen: dict[tuple[str, int], int] = {}
    n_q = [0]
    first_bad = []

    def hook(m, t):
        ann.review(m, t)
        if t not in QUERY_EPOCHS:
            return
        worlds, ledger = H.convert(m)
        for intent in INTENTS:
            q = _make_query(H, m, t, intent, rng)
            if q is None:
                continue
            for rep in range(2 if intent in ("pre_ok", "defer_ok") else 1):       # 성공 뒤 되돌리는 질의 → 상한 초과
                if rep == 1 and q["kind"] == "pre":                                 # 옮긴 트럭을 원래 블록으로 (dst≠owner)
                    new_owner = m.ledger.records[q["job"]].owner
                    back = [x for x in H.block_ids if x != new_owner][0]
                    q = dict(q, dst=back, travel=H.layout.gate_to_block_s(back),
                             route=H.layout.pre_gate_route_delta_s(new_owner, back))
                worlds, ledger, ok, c, ok5, c5 = _apply_query(H, m, worlds, ledger, t, q)
                n_q[0] += 1
                label = f"t={t:.0f} {intent}#{rep} {q}"
                _same_code(label, ok, c, ok5, c5)
                _assert_same(H, label, worlds, ledger, m)
                seen[(q["kind"], int(c))] = seen.get((q["kind"], int(c)), 0) + 1
        assert bool(J_INV(worlds, ledger)), f"t={t} 불변식 위반"

    mbt.run(pol, review_fn=hook)
    assert n_q[0] >= 30, f"질의 {n_q[0]} < 30"
    missing = REQUIRED - set(seen)
    assert not missing, f"안 나온 거절 사유 {[(k, TX.REASON_NAMES[c]) for k, c in sorted(missing)]} — 본 것 {seen}"
    REPORT["random"] = dict(n_queries=n_q[0], admitted=ann.n_admitted,
                            codes={f"{k}:{TX.REASON_NAMES[c]}": v for (k, c), v in sorted(seen.items())})


# ───────────────────────────────────────────────── ③ 단계별 거절 (validate·commit) — deepcopy 무대
T_STAGE = 2400.0


def _snapshot_at(H, t_stage: float):
    """v5 를 t_stage 검토 시각까지 굴려 deepcopy 를 남긴다 (그 뒤는 안 쓴다)."""
    mbt, ann, pol = H.make_mbt()
    snap = {}

    def hook(m, t):
        ann.review(m, t)
        if t == t_stage and "mbt" not in snap:
            snap["mbt"] = copy.deepcopy(m)
    mbt.run(pol, review_fn=hook)
    return snap["mbt"]


def test_staged_rejections_match_v5():
    H = harness()
    base = _snapshot_at(H, T_STAGE)
    t = T_STAGE
    lay = H.layout
    other = lambda b: [x for x in H.block_ids if x != b][0]
    seen = {}

    def fresh():
        return copy.deepcopy(base)

    def pre_cand(m):
        c = H.candidates(m, t, pre_gate=True)
        assert c, "진입 전 반입 후보 없음"
        return c[0]

    def prep_both(m, jid, dst):
        worlds, ledger = H.convert(m)
        travel, route = lay.gate_to_block_s(dst), lay.pre_gate_route_delta_s(m.ledger.records[jid].owner, dst)
        txn5 = m.prepare_pre_gate_transfer(jid, dst, travel_s=travel, route_delta_s=route)
        l1, txn, code = J_PREP_PRE(worlds, ledger, t, H.jidx[jid], H.bidx[dst], travel_s=travel, route_delta_s=route,
                                   capacity_margin=2, max_transfers=1)
        assert int(code) == TX.OK and int(txn.txn_id) == txn5.txn_id and float(txn.new_arrival_s) == txn5.new_arrival_s
        _assert_same(H, "prepare", worlds, l1, m)
        return worlds, l1, txn, txn5

    def commit_both(m, worlds, ledger, txn, txn5, label):
        try:
            m.commit(txn5)
            ok5, c5 = True, TX.OK
        except mb.TransferError as ex:
            m.rollback(txn5)
            ok5, c5 = False, TX.reason_code(str(ex))
        w2, l2, c = J_COMMIT(worlds, ledger, txn, t, g=H.g, k0=H.k0)
        l2 = J_ROLLBACK(l2, txn)
        _same_code(label, c == TX.OK, c, ok5, c5)
        _assert_same(H, label, w2, l2, m)
        seen[label] = TX.REASON_NAMES[int(c)]
        return w2, l2

    # (a) version 증가 → stale
    m = fresh(); jid = pre_cand(m); dst = other(m.ledger.records[jid].owner)
    w, l, txn, txn5 = prep_both(m, jid, dst)
    m.ledger.records[jid].version += 1
    l = l._replace(version=l.version.at[H.jidx[jid]].add(1))
    commit_both(m, w, l, txn, txn5, "stale")
    # (b) 준비 후 lock
    m = fresh(); jid = pre_cand(m); dst = other(m.ledger.records[jid].owner)
    w, l, txn, txn5 = prep_both(m, jid, dst)
    m.ledger.records[jid].locked = True
    l = l._replace(locked=l.locked.at[H.jidx[jid]].set(True))
    commit_both(m, w, l, txn, txn5, "locked-after")
    # (c) 준비 후 상태 변경 (PLANNED → RELEASED)
    m = fresh(); jid = pre_cand(m); src = m.ledger.records[jid].owner; dst = other(src)
    w, l, txn, txn5 = prep_both(m, jid, dst)
    m.blocks[src].jobs[jid].status = JobStatus.RELEASED
    w = w._replace(orders=w.orders._replace(status=w.orders.status.at[H.bidx[src], H.jidx[jid]].set(JS_RELEASED)))
    commit_both(m, w, l, txn, txn5, "status-after")
    # (d) rollback 뒤 commit → 닫힌 트랜잭션 · (e) commit 두 번 → 닫힌 트랜잭션
    m = fresh(); jid = pre_cand(m); dst = other(m.ledger.records[jid].owner)
    w, l, txn, txn5 = prep_both(m, jid, dst)
    m.rollback(txn5); l = J_ROLLBACK(l, txn)
    _assert_same(H, "rolled-back", w, l, m)
    commit_both(m, w, l, txn, txn5, "closed-after-rollback")
    m = fresh(); jid = pre_cand(m); dst = other(m.ledger.records[jid].owner)
    w, l, txn, txn5 = prep_both(m, jid, dst)
    w, l = commit_both(m, w, l, txn, txn5, "commit-ok")
    assert seen["commit-ok"] == "OK"
    commit_both(m, w, l, txn, txn5, "closed-after-commit")
    # (f) lock 해제 조작 — 도착한 작업을 '미lock' 으로 두면 v5 는 소스 상태 위반 / 이연은 상태 위반
    m = fresh()
    arrived = [j for j in H.job_ids if j in m.ledger.records and m.ledger.records[j].locked
               and m.ledger.records[j].flow == "GATE_IN"]
    assert arrived, "도착한 반입 작업이 없다"
    jid = arrived[0]; src = m.ledger.records[jid].owner; dst = other(src)
    m.ledger.records[jid].locked = False
    m.ledger.records[jid].b_block_arrival = None          # v5 reassignable 의 b 항 — 배열은 lock 에 포함 (머리말)
    worlds, ledger = H.convert(m)
    ok5, c5 = v5_try_pre_gate(m, jid, dst, travel_s=lay.gate_to_block_s(dst), route_delta_s=lay.pre_gate_route_delta_s(src, dst))
    w2, l2, ok, c = J_PRE(worlds, ledger, t, H.jidx[jid], H.bidx[dst], travel_s=lay.gate_to_block_s(dst),
                          route_delta_s=lay.pre_gate_route_delta_s(src, dst), g=H.g, k0=H.k0, capacity_margin=2, max_transfers=1)
    _same_code("src-status", ok, c, ok5, c5); _assert_same(H, "src-status", w2, l2, m)
    assert int(c) == TX.R_SRC_STATUS; seen["src-status"] = TX.REASON_NAMES[int(c)]
    # 이연 쪽 상태 위반: A ≤ now 인 도착 작업은 '이미 gate-in' 이 먼저라, A 를 미래로 조작해 status 검사까지 간다
    m.ledger.records[jid].a_gate_in = t + 100.0
    m.blocks[src].time_ledger.records[jid].gate_in = t + 100.0
    worlds, ledger = H.convert(m)
    ok5, c5 = v5_try_defer(m, jid, 600.0)
    w2, l2, ok, c = J_DEFER(worlds, ledger, t, H.jidx[jid], 600.0, max_deferrals=1)
    _same_code("defer-status", ok, c, ok5, c5); _assert_same(H, "defer-status", w2, l2, m)
    assert int(c) == TX.R_STATUS; seen["defer-status"] = TX.REASON_NAMES[int(c)]
    # (g) 규격 적합 슬롯 없음 — dst 야드 조작: 빈 칸에 FT20 하나 · FT40 칸은 만재 → FT40 반입이 갈 곳이 없다
    m = fresh()
    ft40 = [j for j in H.candidates(m, t, pre_gate=True)
            if m.blocks[m.ledger.records[j].owner].jobs[j].inbound_size == ContainerSize.FT40]
    assert ft40, "FT40 진입 전 반입 후보 없음"
    jid = ft40[0]; src = m.ledger.records[jid].owner; dst = other(src)
    dsim = m.blocks[dst]; geom = H.prof.block
    hack = []
    for bay in range(1, geom.bay_count + 1):
        for row in range(1, geom.row_count + 1):
            pile = dsim.stacks.stack(bay, row)
            if not pile:
                cid = f"HACK_{bay}_{row}_0"
                dsim.stacks.place(Container(cid, ContainerSize.FT20, LoadStatus.FULL, dst, bay, row, 0), bay, row)
                hack.append(cid)
            elif dsim.stacks.containers[pile[-1]].size == ContainerSize.FT40:
                for k in range(len(pile), geom.tier_max):
                    cid = f"HACK_{bay}_{row}_{k}"
                    dsim.stacks.place(Container(cid, ContainerSize.FT40, LoadStatus.FULL, dst, bay, row, 0), bay, row)
                    hack.append(cid)
    worlds, ledger = H.convert(m, extra_cont={H.bidx[dst]: hack})
    free5 = m.free_slots(dst)
    assert list(np.asarray(J_FREE(worlds, ledger))) == [m.free_slots(b) for b in H.block_ids]
    ok5, c5 = v5_try_pre_gate(m, jid, dst, travel_s=lay.gate_to_block_s(dst), route_delta_s=lay.pre_gate_route_delta_s(src, dst),
                              margin=-10 ** 6)                                       # 용량 검사를 지나 validate 까지
    w2, l2, ok, c = J_PRE(worlds, ledger, t, H.jidx[jid], H.bidx[dst], travel_s=lay.gate_to_block_s(dst),
                          route_delta_s=lay.pre_gate_route_delta_s(src, dst), g=H.g, k0=H.k0, capacity_margin=-10 ** 6,
                          max_transfers=1)
    _same_code("no-slot", ok, c, ok5, c5); _assert_same(H, "no-slot", w2, l2, m, {H.bidx[dst]: hack})
    assert int(c) == TX.R_NO_SLOT, f"야드 조작 뒤 코드 {TX.REASON_NAMES[int(c)]} (free={free5}, hack={len(hack)})"
    seen["no-slot"] = TX.REASON_NAMES[int(c)]
    REPORT["staged"] = dict(seen=seen, hack_containers=len(hack), free_after_hack=free5)


# ───────────────────────────────────────────────── ③b 블록별 행 규약 — dst 사본의 **다른 행**으로 이송 (dst_row)
def test_transfer_into_spare_row_matches_v5():
    """host_terminal.py 의 블록별 행 규약: 이송 트럭은 dst 사본의 빈 행(여분 행)에 앉는다. 배열은 `dst_row` 로 그 행을 받고,
    v5(행 개념 없음)와는 **행 번호를 되돌려 붙인 뒤** 잎 전부 == 이어야 한다."""
    H = harness()
    m = _snapshot_at(H, T_STAGE)
    t = T_STAGE
    lay = H.layout
    cands = H.candidates(m, t, pre_gate=True)
    assert cands
    jid = cands[0]; n = H.jidx[jid]
    src = m.ledger.records[jid].owner; dst = [b for b in H.block_ids if b != src][0]
    s_, d_ = H.bidx[src], H.bidx[dst]
    worlds, ledger = H.convert(m)
    free_rows = [r for r in range(H.N) if int(worlds.orders.block[d_, r]) < 0 and r != n]
    r2 = free_rows[-1]                                        # n 이 아닌 빈 행 — 여분 행 흉내
    assert int(TX.first_free_row(worlds, d_, r2)) == r2
    travel, route = lay.gate_to_block_s(dst), lay.pre_gate_route_delta_s(src, dst)
    ok5, c5 = v5_try_pre_gate(m, jid, dst, travel_s=travel, route_delta_s=route)
    assert ok5
    c_before = np.asarray(worlds.conts.c_size).copy()
    w2, l2, ok, c = J_PRE(worlds, ledger, t, n, d_, travel_s=travel, route_delta_s=route, g=H.g, k0=H.k0,
                          capacity_margin=2, max_transfers=1, dst_row=r2)
    assert bool(ok) and int(c) == TX.OK
    assert int(l2.row[n]) == r2 and int(l2.owner[n]) == d_ and bool(J_INV(w2, l2))
    assert int(w2.orders.block[d_, r2]) == d_ and int(w2.orders.block[d_, n]) < 0 and int(w2.orders.block[s_, n]) < 0
    assert int(w2.orders.inbound_cont[d_, r2]) == H.c0[d_] + r2
    # 행 번호 되돌리기 (r2 → n) 뒤 v5 와 대조
    o = w2.orders
    row = jax.tree_util.tree_map(lambda x: x[d_, r2], o)
    blank = jax.tree_util.tree_map(lambda x: x[d_, n], o)                                   # 빈 행 (원래 n 자리)
    row = row._replace(inbound_cont=jnp.where(row.is_store, H.c0[d_] + n, -1).astype(jnp.int32))
    o3 = jax.tree_util.tree_map(lambda x, b, v: x.at[d_, r2].set(b).at[d_, n].set(v), o, blank, row)
    q = w2.queue
    hit = (q.kind[d_] == 5) & (q.target[d_] == r2)
    q3 = q._replace(target=q.target.at[d_].set(jnp.where(hit, n, q.target[d_])))
    cs = np.asarray(w2.conts.c_size).copy()
    cs[d_, H.c0[d_] + n] = cs[d_, H.c0[d_] + r2]; cs[d_, H.c0[d_] + r2] = c_before[d_, H.c0[d_] + r2]
    w3 = w2._replace(orders=o3, queue=q3, conts=w2.conts._replace(c_size=jnp.asarray(cs)))
    l3 = l2._replace(row=l2.row.at[n].set(n))
    _assert_same(H, "spare-row", w3, l3, m)
    # 빈 행이 없으면 R_NO_SPARE_ROW (배열판 전용) — 아무것도 바꾸지 않는다
    m2 = _snapshot_at(H, T_STAGE)
    worlds, ledger = H.convert(m2)
    w4, l4, ok4, c4 = J_PRE(worlds, ledger, t, n, d_, travel_s=travel, route_delta_s=route, g=H.g, k0=H.k0,
                            capacity_margin=2, max_transfers=1, dst_row=-1)
    assert not bool(ok4) and int(c4) == TX.R_NO_SPARE_ROW
    assert not _diff(w4, worlds) and _queue_view(w4, d_) == _queue_view(worlds, d_)
    assert int(l4.txn_seq) == int(ledger.txn_seq) + 1                      # prepare 는 성공 → v5 처럼 txn id 는 소모된다
    assert not _diff(l4._replace(txn_seq=ledger.txn_seq), ledger, skip=())   # 나머지 원장은 그대로 (예약도 풀렸다)
    REPORT["spare_row"] = dict(job=jid, src=src, dst=dst, row_from=n, row_to=r2)


# ───────────────────────────────────────────────── ④ free_slots · sync_locks · 불변식 — 300초마다
def test_free_slots_sync_locks_invariants_match_v5():
    H = harness()
    mbt, ann, pol = H.make_mbt()
    n_ep = [0]

    def hook(m, t):
        ann.review(m, t)
        if t % 300.0 != 0.0:
            return
        worlds, ledger = H.convert(m)
        n_ep[0] += 1
        got = [int(x) for x in np.asarray(J_FREE(worlds, ledger))]
        exp = [m.free_slots(b) for b in H.block_ids]
        assert got == exp, f"t={t} free_slots arr={got} v5={exp}"
        cleared = ledger._replace(locked=jnp.zeros_like(ledger.locked))
        re = J_SYNC(worlds, cleared)
        assert np.array_equal(np.asarray(re.locked), np.asarray(ledger.locked)), f"t={t} sync_locks 갈림"
        assert bool(J_INV(worlds, ledger)), f"t={t} 불변식"
        m.check_invariants()
        assert float(TX.terminal_now(worlds)) == m.now
        for jid in H.job_ids:
            rec = m.ledger.records.get(jid)
            if rec is None:
                continue
            n = H.jidx[jid]
            assert bool(TX.reassignable(worlds, ledger, n)) == rec.reassignable, f"t={t} {jid} reassignable"

    mbt.run(pol, review_fn=hook)
    assert n_ep[0] >= 20
    REPORT["ledger"] = dict(epochs=n_ep[0], admitted=ann.n_admitted)


def test_zz_report(capsys):
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    with capsys.disabled():
        print("\n[transfer_txn report]")
        for k, v in REPORT.items():
            print(f"  {k}: {v}")
