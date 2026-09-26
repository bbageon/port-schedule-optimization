"""전역 트럭 원장(gpu/ledger.py) 이 v5 `MultiBlockTerminal.ledger` 와 **같은 답**을 내는가 ([[YR-327]] 조각 6 · key=ledger).

■ 무엇을 지키나 (허용오차 없음 · 기대값 손기입 없음 — v5 를 실제로 굴려 받은 답과 `==`)
  v5 MultiBlockTerminal 을 test_world_equivalence 골격(21블록 · SF_SPT · PRE_ADVICE · ScheduledAnnouncer lead 1800 ·
  admission_epochs(OBS_24H) 1,441 에폭)으로 굴리며, 조정자가 원장을 만지는 **자리마다** 배열 원장에 같은 수술을 비춘다
  (조정자가 배열 세계에서 gather_owned 로 만들 입력을 v5 상태에서 읽는다):
    등록        admit_external_job 성공 직후 → register           (can_register 가 먼저 참이어야 한다)
    런 중 lock  _sync_locks(sim) 을 ReviewEpoch 마다 가로채 → sync_locks (그 블록 행만) → ① locked · b_block_arrival ==
    이송/이연   review 에서 try_pre_gate_transfer · try_defer_admitted_entry 를 강제 → commit_transfer · defer_entry →
                ② owner·version·transfer_count·transfer_history·a_gate_in·entry_deferrals·entry_deferred_s · route_cost_s ==
                + 술어 pre_gate_ok · defer_ok 가 v5 try_* 의 참/거짓과 같다 (같은 트럭 재시도 = 상한 초과 거절까지)
    종료        run 끝 harvest → ③ records 전 필드 (dict 순서 포함) · a_to_o 표본 (순서·정렬열) · route_cost_s ·
                reassignable · 등록 순서 · 불변식 0
    + 에폭마다 '전 블록을 한 번에 훑는' sync_locks == '블록별로 21번 훑은' 원장 (조정자가 쓸 호출 방식)
  부하 20 (이송 0 · 원장만) · 부하 60 (pre-gate 이송 1건 + 이연 1건 강제).
  단위: 빈 원장·등록 거절·이력 칸 넘침·numpy 왕복·jit==eager·reassignable(v5 JobRecord 64조합)·a_to_o(v5 TerminalLedger).

실행: WSL venv · x64 CPU. 한 부하 ≈ 30초 (v5 11초 + 가로채기 30k회) — 88초 창에는 부하 하나씩.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import ledger as LG                                        # noqa: E402
from yard_rl.v6.gpu.host_convert import FLOW_NAMES                              # noqa: E402
from yard_rl.v6.gpu.state import EMPTY_TIME, FL_GATE_IN, FL_GATE_OUT, JS_PLANNED, JS_WAITING   # noqa: E402
from yard_rl.v6.world.domain.enums import InformationLevel, JobFlow, JobStatus  # noqa: E402
from yard_rl.v6.world.integrated import (baselines as bl, candidates as cd,      # noqa: E402
                                          engine as eng, multiblock as mb,
                                          policy_config as pc, profiles as pr,
                                          terminal_stream as ts, yard_layout as yl)

SEED = 9_900_777
LVL = InformationLevel.PRE_ADVICE
JS = {s: i for i, s in enumerate(JobStatus)}          # v5 선언 순 = state.JS_*
FL = {f.value: i for i, f in enumerate(JobFlow)}      # v5 선언 순 = state.FL_*
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── v5 구동 골격 (test_world_equivalence.py 38-56)
def _build(load: int):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, SEED, obs=ts.OBS_24H, layout=layout,
                             params=ts.TerminalStreamParams(load_4h=load),
                             day_total=load, background_seed=SEED)
    return prof, layout, built


def _make_terminal(prof, built):
    def sim_from(scn):
        s = eng.TerminalSimulator(prof, scn, check_invariants=True)
        s.info_level = LVL
        return s
    mbt = mb.MultiBlockTerminal(
        {b: ts.ensure_time_ledger(sim_from(s)) for b, s in built["scenarios"].items()},
        extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=1800.0, end_s=built["sim_end_s"])
    gens: dict[int, object] = {}
    pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LVL) for c in dp.crane_ids}
        bl._apply(sim, pol.decide(sim, dp, gb))
    return mbt, ann, exec_policy


def _rec_dict(rec) -> dict:
    return dataclasses.asdict(rec)


# ───────────────────────────────────────────────── 그림자 원장 — v5 조정자의 수술을 배열 원장에 비춘다
class Shadow:
    """오더 번호 n = v5 records 삽입 순서(초기 본선 작업 블록순×sim.jobs 순) 뒤에 명단 순 트럭 — 그래서 reg_seq == n 이 기대된다."""

    def __init__(self, mbt, layout, schedule, *, h_max: int = 1, force_ops: bool = False):
        self.mbt, self.layout = mbt, layout
        self.blocks = list(mbt.blocks)                       # layout 순 = dict 순
        self.bidx = {b: i for i, b in enumerate(self.blocks)}
        self.sim_block = {id(s): b for b, s in mbt.blocks.items()}
        ids = list(mbt.ledger.records)
        for e in schedule:
            if e["job_id"] not in mbt.ledger.records:
                ids.append(e["job_id"])
        self.ids, self.idx = ids, {j: i for i, j in enumerate(ids)}
        N = len(ids)
        mask = np.zeros(N, bool)
        origin = np.full(N, -1, np.int32); flow = np.full(N, -1, np.int32); a = np.full(N, np.inf)
        for jid, rec in mbt.ledger.records.items():           # 137-140행 초기 등록
            n = self.idx[jid]
            mask[n] = True; origin[n] = self.bidx[rec.origin_block]; flow[n] = FL[rec.flow]
            a[n] = np.inf if rec.a_gate_in is None else rec.a_gate_in
        self.L = LG.register_rows(LG.empty_truck_ledger(N, h_max), mask, origin, origin, flow, a)
        # 소유 블록에서 읽는 (N,) 열 — 조정자가 gather_owned 로 만들 것
        self.status = np.full(N, JS_PLANNED, np.int32)
        self.block_in = np.full(N, np.inf); self.done = np.full(N, np.inf); self.gate_out = np.full(N, np.inf)
        self.tl = np.zeros(N, bool)
        self.n_sync = self.n_admit = self.n_epoch = 0
        self.mismatch: list[tuple] = []
        self.n_mismatch = 0
        self.force_ops = force_ops
        self.transfer_done = self.defer_done = None
        self.pred_checks = 0
        self.L_epoch_start = self.L
        self._sync_jit = jax.jit(LG.sync_locks)
        # 가로채기 방식 — block: _sync_locks 마다 그 블록 행만 배열 sync (30k 회 · CPU 기본) /
        #                 epoch: 에폭마다 전 행을 한 번에 (1,441 회 · GPU 처럼 호출당 왕복이 비싼 장치용 — 조정자의 호출 방식)
        #   기본: CPU 는 block, GPU 는 epoch (호출당 왕복 ~1ms × 30k 는 GPU 에서 세션 창을 넘긴다)
        default_mode = "block" if jax.devices()[0].platform == "cpu" else "epoch"
        self.sync_mode = os.environ.get("LEDGER_SYNC_MODE", default_mode)
        assert self.sync_mode in ("block", "epoch"), self.sync_mode
        self._progress = os.environ.get("LEDGER_PROGRESS")          # 진행 로그 파일 (세션이 죽어도 어디까지 갔는지)
        self._t0 = __import__("time").perf_counter()
        self._install()

    def _log(self, msg: str):
        if self._progress:
            with open(self._progress, "a", encoding="utf-8") as f:
                f.write(f"[{__import__('time').perf_counter() - self._t0:6.1f}s] {msg}\n")

    # ── v5 상태 → 조정자 입력 열 ──
    def _refresh(self, sim, b: str) -> np.ndarray:
        jobs = sim.jobs
        tl = sim.time_ledger.records
        rows = np.fromiter((self.idx[j] for j in jobs), np.int32, len(jobs))
        st = np.empty(len(jobs), np.int32)
        bi = np.full(len(jobs), np.inf); dn = np.full(len(jobs), np.inf); go = np.full(len(jobs), np.inf)
        tlm = np.zeros(len(jobs), bool)
        for i, (jid, j) in enumerate(jobs.items()):
            st[i] = JS[j.status]
            r = tl.get(jid)
            if r is not None:
                tlm[i] = True
                if r.block_arrival is not None:
                    bi[i] = r.block_arrival
                if r.job_done is not None:
                    dn[i] = r.job_done
                if r.gate_out is not None:
                    go[i] = r.gate_out
        self.status[rows] = st; self.block_in[rows] = bi; self.done[rows] = dn; self.gate_out[rows] = go
        self.tl[rows] = tlm
        return rows

    def _refresh_all(self):
        for b, sim in self.mbt.blocks.items():
            self._refresh(sim, b)

    def _note(self, what, *info):
        self.n_mismatch += 1
        if len(self.mismatch) < 20:
            self.mismatch.append((what, *info))

    # ── 가로채기 ──
    def _install(self):
        mbt = self.mbt
        orig_sync, orig_admit = mbt._sync_locks, mbt.admit_external_job

        def sync(sim):
            b = self.sim_block[id(sim)]
            rows = self._refresh(sim, b)
            orig_sync(sim)                                          # v5 먼저 (원장 갱신)
            self.n_sync += 1
            if self.sync_mode == "epoch":
                return                                              # 배열 sync·비교는 after_review 에서 전 행 한 번에
            mask = np.zeros(len(self.ids), bool); mask[rows] = True
            self.L = self._sync_jit(self.L, mask, self.status, self.block_in, self.tl)
            # ① 그 블록 행 — locked · b_block_arrival
            recs = mbt.ledger.records
            jl = list(sim.jobs)
            v5l = np.fromiter((recs[j].locked for j in jl), bool, len(jl))
            v5b = np.fromiter(((np.inf if recs[j].b_block_arrival is None else recs[j].b_block_arrival) for j in jl),
                              np.float64, len(jl))
            al = np.asarray(self.L.locked)[rows]; ab = np.asarray(self.L.b_block_arrival)[rows]
            if not (np.array_equal(al, v5l) and np.array_equal(ab, v5b)):
                bad = np.nonzero((al != v5l) | (ab != v5b))[0]
                for i in bad[:3]:
                    self._note("sync", b, float(sim.clock), jl[i], ("v5", bool(v5l[i]), float(v5b[i])),
                               ("arr", bool(al[i]), float(ab[i])))
                self.n_mismatch += max(0, len(bad) - 3)

        def admit(bid, job, *, gate_in_s, travel_s):
            n = self.idx[job.job_id]
            ok_arr = bool(LG.can_register(self.L, n)[0])
            orig_admit(bid, job, gate_in_s=gate_in_s, travel_s=travel_s)   # TransferError 면 여기서 끝 — 등록 없음
            if not ok_arr:
                self._note("can_register", bid, gate_in_s, job.job_id)
            self.L = LG.register(self.L, n, self.bidx[bid], self.bidx[bid], FL[job.flow.value], gate_in_s)
            self.n_admit += 1
            rec = mbt.ledger.records[job.job_id]
            got = self._row(n)
            exp = _rec_dict(rec)
            if got != exp:
                self._note("register", bid, gate_in_s, job.job_id, exp, got)

        mbt._sync_locks, mbt.admit_external_job = sync, admit

    def _row(self, n: int) -> dict:
        return LG.to_v5_records(self.L, self.ids, self.blocks, FLOW_NAMES, rows=[n])[self.ids[n]]

    # ── review 마다 (ann.review 뒤) ──
    def _v5_lock_cols(self):
        """v5 records 의 locked · b_block_arrival 을 n 순 (N,) 열로 (미등록 행 False · +inf)."""
        N = len(self.ids)
        v5l = np.zeros(N, bool); v5b = np.full(N, np.inf)
        for jid, rec in self.mbt.ledger.records.items():
            n = self.idx[jid]
            v5l[n] = rec.locked
            if rec.b_block_arrival is not None:
                v5b[n] = rec.b_block_arrival
        return v5l, v5b

    def after_review(self, t: float):
        self.n_epoch += 1
        if self.n_epoch % 100 == 1:
            self._log(f"epoch {self.n_epoch} t={t} syncs={self.n_sync} admits={self.n_admit} mismatch={self.n_mismatch}")
        allm = np.ones(len(self.ids), bool)
        if self.sync_mode == "epoch":
            # 조정자 호출 방식 — 전 블록이 park 한 뒤 전 행을 한 번에 → ① 전 행 locked · b_block_arrival
            self.L = self._sync_jit(self.L, allm, self.status, self.block_in, self.tl)
            v5l, v5b = self._v5_lock_cols()
            al, ab = np.asarray(self.L.locked), np.asarray(self.L.b_block_arrival)
            if not (np.array_equal(al, v5l) and np.array_equal(ab, v5b)):
                bad = np.nonzero((al != v5l) | (ab != v5b))[0]
                for n in bad[:3]:
                    self._note("sync-epoch", t, self.ids[n], ("v5", bool(v5l[n]), float(v5b[n])), ("arr", bool(al[n]), float(ab[n])))
                self.n_mismatch += max(0, len(bad) - 3)
        else:
            # 전 블록을 한 번에 훑는 호출 == 블록별 21번 (조정자 호출 방식) — 에폭 시작 원장에서 다시 계산
            L_all = self._sync_jit(self.L_epoch_start, allm, self.status, self.block_in, self.tl)
            if not (np.array_equal(np.asarray(L_all.locked), np.asarray(self.L.locked))
                    and np.array_equal(np.asarray(L_all.b_block_arrival), np.asarray(self.L.b_block_arrival))):
                self._note("sync-all-vs-per-block", t)
        if self.force_ops:
            self._force_ops(t)
        self.L_epoch_start = self.L

    def _pick(self, t: float, *, flow_in_only: bool):
        """아직 게이트 전(A > t+1e-6)·PLANNED·미잠금·이송/이연 0 인 트럭 (records 순 첫 것)."""
        for jid, rec in self.mbt.ledger.records.items():
            if rec.a_gate_in is None or rec.a_gate_in <= t + 1e-6 or rec.locked:
                continue
            if flow_in_only and rec.flow != JobFlow.GATE_IN.value:
                continue
            if rec.transfer_count or rec.entry_deferrals:
                continue
            j = self.mbt.blocks[rec.owner].jobs.get(jid)
            if j is not None and j.status == JobStatus.PLANNED:
                return jid, rec
        return None

    def _force_ops(self, t: float):
        mbt, B = self.mbt, len(self.blocks)
        if self.transfer_done is None:
            pick = self._pick(t, flow_in_only=True)
            if pick is not None:
                jid, rec = pick
                n, src = self.idx[jid], rec.owner
                dst = max((b for b in self.blocks if b != src), key=lambda b: (mbt.free_slots(b), b))
                route = self.layout.pre_gate_route_delta_s(src, dst)
                ok_arr, code = LG.pre_gate_ok(self.L, n, self.bidx[dst], t, n_blocks=B)
                ok = mbt.try_pre_gate_transfer(jid, dst, travel_s=self.layout.gate_to_block_s(dst), route_delta_s=route)
                self.pred_checks += 1
                if not ok_arr and ok:
                    self._note("pre_gate_ok says no but v5 transferred", t, jid, int(code))
                if ok:
                    self.L = LG.commit_transfer(self.L, n, self.bidx[dst], t, route, True)
                    self.transfer_done = (jid, src, dst, t, route)
                    got, exp = self._row(n), _rec_dict(mbt.ledger.records[jid])
                    if got != exp or float(self.L.route_cost_s) != mbt.route_cost_s:
                        self._note("commit", t, jid, exp, got, mbt.route_cost_s, float(self.L.route_cost_s))
                    # 같은 트럭 재이송 → 상한 초과 거절 (v5 431행 · 술어 LR_TRANSFER_CAP)
                    ok2 = mbt.try_pre_gate_transfer(jid, src, travel_s=self.layout.gate_to_block_s(src),
                                                    route_delta_s=self.layout.pre_gate_route_delta_s(dst, src))
                    ok_arr2, code2 = LG.pre_gate_ok(self.L, n, self.bidx[src], t, n_blocks=B)
                    self.pred_checks += 1
                    if ok2 or bool(ok_arr2) or int(code2) != LG.LR_TRANSFER_CAP:
                        self._note("retransfer", t, jid, ok2, bool(ok_arr2), int(code2))
        elif self.defer_done is None and t > self.transfer_done[3]:
            pick = self._pick(t, flow_in_only=False)
            if pick is not None:
                jid, rec = pick
                n = self.idx[jid]
                ok_arr, code = LG.defer_ok(self.L, n, 600.0, t)
                ok = mbt.try_defer_admitted_entry(jid, 600.0)
                self.pred_checks += 1
                if not ok_arr and ok:
                    self._note("defer_ok says no but v5 deferred", t, jid, int(code))
                if ok:
                    self.L = LG.defer_entry(self.L, n, 600.0, True)
                    self.defer_done = (jid, rec.owner, t)
                    got, exp = self._row(n), _rec_dict(mbt.ledger.records[jid])
                    if got != exp:
                        self._note("defer", t, jid, exp, got)
                    ok2 = mbt.try_defer_admitted_entry(jid, 600.0)            # 상한 1 → 거절 (314행)
                    ok_arr2, code2 = LG.defer_ok(self.L, n, 600.0, t)
                    self.pred_checks += 1
                    if ok2 or bool(ok_arr2) or int(code2) != LG.LR_DEFER_CAP:
                        self._note("redefer", t, jid, ok2, bool(ok_arr2), int(code2))

    # ── 종료 ──
    def finish(self, out: dict, end: float) -> dict:
        mbt = self.mbt
        self._refresh_all()
        self.L = LG.harvest(self.L, self.status, self.block_in, self.done, self.gate_out, self.tl)
        got = LG.to_v5_records(self.L, self.ids, self.blocks, FLOW_NAMES)
        exp = {jid: _rec_dict(r) for jid, r in mbt.ledger.records.items()}
        diffs = []
        if list(got) != list(exp):
            diffs.append(("order", len(got), len(exp), next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), None)))
        for jid in exp:
            g = got.get(jid)
            if g != exp[jid]:
                diffs.append((jid, {k: (exp[jid][k], None if g is None else g.get(k))
                                    for k in exp[jid] if g is None or g.get(k) != exp[jid][k]}))
        v5_samples = mbt.ledger.a_to_o_samples_s(end)
        arr_samples = LG.sample_list(self.L, end)
        v5_reas = [r.reassignable for r in mbt.ledger.records.values()]
        order = np.asarray(LG.registration_order(self.L))[: int(self.L.n_registered)]
        arr_reas = [bool(x) for x in np.asarray(LG.reassignable(self.L))[order]]
        world_owner = np.full(len(self.ids), -1, np.int32)
        for b, sim in mbt.blocks.items():
            for jid in sim.jobs:
                world_owner[self.idx[jid]] = self.bidx[b]
        return {
            "diffs": diffs, "n_diff": len(diffs),
            "samples_equal": arr_samples == v5_samples, "sorted_equal": sorted(arr_samples) == sorted(v5_samples),
            "n_samples": (len(arr_samples), len(v5_samples)),
            "route": (float(self.L.route_cost_s), out["route_cost_s"]),
            "reassignable_equal": arr_reas == v5_reas,
            "order_is_arange": bool(np.array_equal(order, np.arange(len(order)))),
            "invariant_errors": int(LG.invariant_errors(self.L, world_owner)),
            "n_registered": (int(self.L.n_registered), len(mbt.ledger.records)),
            "overflow": int(self.L.overflow),
        }


# ───────────────────────────────────────────────── 통합 — v5 를 굴리며 비춘다
@pytest.mark.parametrize("load,force_ops", [(20, False), (60, True)], ids=["load20", "load60-transfer-defer"])
def test_ledger_shadows_v5_terminal(load, force_ops):
    prof, layout, built = _build(load)
    mbt, ann, exec_policy = _make_terminal(prof, built)
    sh = Shadow(mbt, layout, built["schedule"], force_ops=force_ops)

    def review(m, t):
        ann.review(m, t)
        sh.after_review(t)

    out = mbt.run(exec_policy, review_fn=review)
    mbt.check_invariants()
    end = ts.OBS_24H.observe_s
    res = sh.finish(out, end)
    label = f"load{load}{'-ops' if force_ops else ''}"
    REPORT[label] = dict(mode=sh.sync_mode, device=jax.devices()[0].platform,
                         syncs=sh.n_sync, epochs=sh.n_epoch, admits=sh.n_admit, records=res["n_registered"][1],
                         n_samples=res["n_samples"][1], mismatch=sh.n_mismatch, diffs=res["n_diff"],
                         transfer=sh.transfer_done, defer=sh.defer_done, route=res["route"], pred_checks=sh.pred_checks)
    n_ep = len(ts.admission_epochs(ts.OBS_24H))
    assert sh.n_epoch == n_ep and sh.n_sync == n_ep * len(mbt.blocks), (sh.n_epoch, sh.n_sync, n_ep)
    assert sh.n_admit == ann.n_admitted == load, (sh.n_admit, ann.n_admitted)
    assert sh.n_mismatch == 0, (f"[{label}] 런 중 원장 불일치 {sh.n_mismatch}건 — 처음 갈리는 곳 (종류·블록·시각·오더·v5·arr):\n"
                                + "\n".join(f"  {m}" for m in sh.mismatch))
    assert res["n_diff"] == 0, f"[{label}] harvest 뒤 records 가 갈린다 {res['n_diff']}건: {res['diffs'][:5]}"
    assert res["n_registered"][0] == res["n_registered"][1]
    assert res["samples_equal"] and res["sorted_equal"], f"[{label}] a_to_o 표본 {res['n_samples']}"
    assert res["n_samples"][1] == load
    assert res["route"][0] == res["route"][1], f"[{label}] route_cost_s arr={res['route'][0]!r} v5={res['route'][1]!r}"
    assert res["reassignable_equal"] and res["order_is_arange"]
    assert res["invariant_errors"] == 0 and res["overflow"] == 0
    if force_ops:
        assert sh.transfer_done is not None and sh.defer_done is not None, "강제 이송/이연이 한 건도 성사되지 않았다 — 무대를 바꿔야 한다"
        assert res["route"][1] != 0.0 and sh.pred_checks == 4
        jid = sh.transfer_done[0]
        rec = mbt.ledger.records[jid]
        assert rec.transfer_count == 1 and rec.owner == sh.transfer_done[2] and rec.version >= 1
        assert mbt.ledger.records[sh.defer_done[0]].entry_deferrals == 1
    else:
        assert res["route"][1] == 0.0


# ───────────────────────────────────────────────── 단위
def test_empty_register_and_duplicate():
    L = LG.empty_truck_ledger(6, 2)
    assert L.n == 6 and L.h == 2 and int(L.n_registered) == 0
    assert not bool(LG.can_register(L, -1)[0]) and int(LG.can_register(L, 6)[1]) == LG.LR_UNREGISTERED
    L = LG.register(L, 3, 2, 2, FL_GATE_IN, 100.0)
    assert bool(L.registered[3]) and int(L.reg_seq[3]) == 0 and int(L.n_registered) == 1
    ok, code = LG.can_register(L, 3)
    assert not bool(ok) and int(code) == LG.LR_DUPLICATE
    L2 = LG.register(L, 3, 4, 4, FL_GATE_OUT, 5.0)                       # 중복 — 아무것도 안 바뀐다
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(L, L2))
    L3 = LG.register(L, 3, 4, 4, FL_GATE_OUT, 5.0, ok=False)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(L, L3))
    L = LG.register_rows(L, np.array([1, 0, 1, 1, 0, 1], bool), np.arange(6), np.arange(6), FL_GATE_IN, np.full(6, np.inf))
    assert [int(x) for x in L.reg_seq] == [1, -1, 2, 0, -1, 3] and int(L.n_registered) == 4
    assert [int(x) for x in LG.registration_order(L)[:4]] == [3, 0, 2, 5]
    assert int(L.flow[3]) == FL_GATE_IN and float(L.a_gate_in[3]) == 100.0     # 등록된 행은 register_rows 가 건너뛴다
    assert bool(np.all(np.isnan(np.asarray(LG.a_to_o_samples_s(L, 1000.0))[[0, 2, 5]])))   # A 없음 → 표본 아님
    assert float(LG.a_to_o_samples_s(L, 1000.0)[3]) == 900.0


def test_transfer_history_overflow_and_defer():
    L = LG.empty_truck_ledger(4, 1)
    L = LG.register(L, 1, 0, 0, FL_GATE_IN, 3600.0)
    L = LG.commit_transfer(L, 1, 5, 1800.0, 220.0, True)
    L = LG.commit_transfer(L, 1, 2, 1860.0, -30.0, False)                # ok=False → 무변경
    assert int(L.owner[1]) == 5 and int(L.version[1]) == 1 and int(L.transfer_count[1]) == 1
    assert (int(L.transfer_src[1, 0]), int(L.transfer_dst[1, 0]), float(L.transfer_t[1, 0])) == (0, 5, 1800.0)
    assert float(L.route_cost_s) == 220.0 and int(L.overflow) == 0
    L = LG.commit_transfer(L, 1, 2, 1860.0, -30.0, True)                 # 이력 칸 1 → 넘침 표시, 나머지는 진행
    assert int(L.owner[1]) == 2 and int(L.transfer_count[1]) == 2 and int(L.overflow) == 1
    assert float(L.route_cost_s) == 220.0 + -30.0 and int(L.transfer_src[1, 0]) == 0
    L = LG.defer_entry(L, 1, 600.0, True)
    assert float(L.a_gate_in[1]) == 3600.0 + 600.0 and int(L.version[1]) == 3
    assert int(L.entry_deferrals[1]) == 1 and float(L.entry_deferred_s[1]) == 600.0
    rec = LG.to_v5_records(L, ["a", "b", "c", "d"], [f"B{i}" for i in range(6)])["b"]
    assert rec["transfer_history"] == (("B0", "B5", 1800.0),) and rec["owner"] == "B2" and rec["a_gate_in"] == 4200.0
    assert rec["b_block_arrival"] is None and rec["flow"] == "GATE_IN"


def test_reassignable_matches_v5_jobrecord():
    """v5 JobRecord.reassignable 를 (locked × flow 6 × B 유무) 24 조합 전부 실제로 불러 대조."""
    combos = [(lk, fl, hb) for lk in (False, True) for fl in JobFlow for hb in (False, True)]
    N = len(combos)
    L = LG.empty_truck_ledger(N + 1)
    exp = []
    for n, (lk, fl, hb) in enumerate(combos):
        rec = mb.JobRecord(job_id=f"J{n}", origin_block="Y01", owner="Y01", flow=fl.value,
                           b_block_arrival=(123.0 if hb else None), locked=lk)
        exp.append(rec.reassignable)
        L = LG.register(L, n, 0, 0, FL[fl.value], 10.0)
        L = L._replace(locked=L.locked.at[n].set(lk),
                       b_block_arrival=L.b_block_arrival.at[n].set(123.0 if hb else EMPTY_TIME))
    got = [bool(x) for x in np.asarray(LG.reassignable(L))]
    assert got[:N] == exp and got[N] is False           # 미등록 행은 False
    assert sum(exp) == 1                                 # not locked · GATE_IN · B 없음 하나뿐


def test_a_to_o_matches_v5_terminal_ledger():
    """v5 TerminalLedger 에 레코드를 넣고 a_to_o_samples_s(end) 를 실제로 부른 답과 == (O 있음 · O 없음 검열 · A 없음 제외)."""
    tl = mb.TerminalLedger()
    rows = [("v", None, None), ("t1", 100.0, 900.5), ("t2", 200.25, None), ("t3", 3000.0, 3500.0),
            ("t4", 86400.0, None), ("t5", 86500.0, None)]
    L = LG.empty_truck_ledger(len(rows))
    for n, (jid, a, o) in enumerate(rows):
        rec = mb.JobRecord(job_id=jid, origin_block="Y01", owner="Y01", flow="GATE_IN", a_gate_in=a, o_gate_out=o)
        tl.register(rec)
        L = LG.register(L, n, 0, 0, FL_GATE_IN, np.inf if a is None else a)
        L = L._replace(o_gate_out=L.o_gate_out.at[n].set(EMPTY_TIME if o is None else o))
    for end in (86400.0, 1000.0, 0.0):
        assert LG.sample_list(L, end) == tl.a_to_o_samples_s(end)
    assert list(LG.to_v5_records(L, [r[0] for r in rows], ["Y01"])) == list(tl.records)


def test_sync_and_harvest_rules_against_v5():
    """`_sync_locks`·`harvest` 규칙을 v5 함수 그 자체로 대조 — 상태 7종 × 장부 항목 유무 × B 유무 를 가짜 sim 에 넣어 부른다."""
    from yard_rl.v6.world.integrated.time_contract import TruckTimes

    class _Sim:
        def __init__(self):
            self.jobs, self.time_ledger = {}, type("TL", (), {"records": {}})()
    combos = [(st, tl, hb) for st in JobStatus for tl in (False, True) for hb in (False, True)]
    N = len(combos)
    mbt = mb.MultiBlockTerminal.__new__(mb.MultiBlockTerminal)
    mbt.ledger = mb.TerminalLedger()
    sim = _Sim()
    L = LG.empty_truck_ledger(N)
    status = np.zeros(N, np.int32); b_in = np.full(N, np.inf); dn = np.full(N, np.inf); go = np.full(N, np.inf)
    tlm = np.zeros(N, bool)
    for n, (st, tl, hb) in enumerate(combos):
        jid = f"J{n}"
        job = type("J", (), {})(); job.status = st
        sim.jobs[jid] = job
        mbt.ledger.register(mb.JobRecord(job_id=jid, origin_block="Y01", owner="Y01", flow="GATE_IN", a_gate_in=1.0))
        L = LG.register(L, n, 0, 0, FL_GATE_IN, 1.0)
        status[n] = JS[st]
        if tl:
            r = TruckTimes(gate_in=1.0, block_arrival=(50.0 + n if hb else None))
            if st == JobStatus.DONE:
                r.job_done, r.gate_out = 500.0 + n, 900.0 + n
            sim.time_ledger.records[jid] = r
            tlm[n] = True
            if hb:
                b_in[n] = 50.0 + n
            if st == JobStatus.DONE:
                dn[n], go[n] = 500.0 + n, 900.0 + n
    mb.MultiBlockTerminal._sync_locks(mbt, sim)
    L = jax.jit(LG.sync_locks)(L, np.ones(N, bool), status, b_in, tlm)
    ids, blocks = [f"J{n}" for n in range(N)], ["Y01"]
    assert LG.to_v5_records(L, ids, blocks) == {j: _rec_dict(r) for j, r in mbt.ledger.records.items()}
    L2 = jax.jit(LG.sync_locks)(L, np.ones(N, bool), status, b_in, tlm)        # 멱등
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(L, L2))
    mbt.ledger.harvest({"Y01": sim})
    L = jax.jit(LG.harvest)(L, status, b_in, dn, go, tlm)
    assert LG.to_v5_records(L, ids, blocks) == {j: _rec_dict(r) for j, r in mbt.ledger.records.items()}
    # RELEASED 만 있는 신선한 원장에 harvest 만 → v5 처럼 unlocked
    L0 = LG.register(LG.empty_truck_ledger(1), 0, 0, 0, FL_GATE_IN, 1.0)
    L0 = LG.harvest(L0, np.array([JS[JobStatus.RELEASED]], np.int32), np.array([np.inf]), np.array([np.inf]),
                    np.array([np.inf]), np.array([False]))
    assert not bool(L0.locked[0])


def test_numpy_roundtrip_jit_and_gather():
    L = LG.empty_truck_ledger(5, 2)
    L = LG.register_rows(L, np.array([1, 1, 0, 1, 1], bool), np.array([0, 1, 0, 1, 0]), np.array([0, 1, 0, 1, 0]),
                         FL_GATE_IN, np.array([10.0, 20.0, np.inf, 40.0, np.inf]))
    L = LG.commit_transfer(L, 0, 1, 5.0, 2.5, True)
    import io
    buf = io.BytesIO()
    np.savez(buf, **LG.to_numpy(L))                                             # npz 로 저장했다가
    buf.seek(0)
    L2 = LG.from_numpy(np.load(buf))                                            # 되살린다 (세션 이어 돌리기)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) and a.dtype == b.dtype for a, b in zip(L, L2))
    # jit 판 == eager 판 (모든 함수)
    st = np.array([JS_WAITING, JS_PLANNED, JS_PLANNED, JS_WAITING, JS_PLANNED], np.int32)
    b_in = np.array([15.0, np.inf, np.inf, 45.0, np.inf]); tl = np.array([1, 1, 0, 1, 0], bool)
    for f, args in ((LG.sync_locks, (L, np.ones(5, bool), st, b_in, tl)),
                    (LG.harvest, (L, st, b_in, b_in + 1, b_in + 2, tl)),
                    (LG.defer_entry, (L, 3, 600.0, True)),
                    (LG.commit_transfer, (L, 3, 0, 7.0, -1.0, True))):
        a, b = f(*args), jax.jit(f)(*args)
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)), f.__name__
    for f, args in ((LG.a_to_o_samples_s, (L, 100.0)), (LG.reassignable, (L,)), (LG.registration_order, (L,)),
                    (LG.pre_gate_ok, (L, 3, 0, 30.0)), (LG.defer_ok, (L, 3, 1.0, 30.0)),
                    (LG.validate_ok, (L, 0, 1, 1)), (LG.can_register, (L, 2))):
        kw = {"n_blocks": 2} if f is LG.pre_gate_ok else {}
        a, b = f(*args, **kw), jax.jit(f, static_argnames=tuple(kw))(*args, **kw)
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), f.__name__
    # gather_owned — (B,N) 에서 소유 행
    x = jnp.arange(10, dtype=jnp.float64).reshape(2, 5) * 1.0
    got = LG.gather_owned(x, L.owner)
    assert [float(v) for v in got] == [5.0, 6.0, np.inf, 8.0, 4.0]          # owner 0:[1,4]→행0 · 1:[0,1,3]→행1 · 미등록 → +inf
    assert [int(v) for v in LG.gather_owned(jnp.arange(10, dtype=jnp.int32).reshape(2, 5), L.owner)] == [5, 6, -1, 8, 4]
    # 술어 — 이송 뒤 validate 는 stale
    ok, code = LG.validate_ok(L, 0, 0, 0)
    assert not bool(ok) and int(code) == LG.LR_STALE
    ok, code = LG.validate_ok(L, 0, 1, 1)
    assert bool(ok) and int(code) == LG.LR_OK
    assert int(LG.pre_gate_ok(L, 0, 1, 5.0, n_blocks=2)[1]) == LG.LR_BAD_DST        # dst == owner
    assert int(LG.pre_gate_ok(L, 0, 0, 5.0, n_blocks=2, max_transfers=1)[1]) == LG.LR_TRANSFER_CAP
    assert int(LG.pre_gate_ok(L, 3, 0, 40.0, n_blocks=2)[1]) == LG.LR_ALREADY_GATE_IN   # A=40 ≤ now+1e-6
    assert bool(LG.pre_gate_ok(L, 3, 0, 39.999998, n_blocks=2)[0])
    assert int(LG.post_gate_ok(L, 3, 0, 39.0, n_blocks=2)[1]) == LG.LR_BEFORE_GATE_IN
    assert bool(LG.post_gate_ok(L, 3, 0, 40.0, n_blocks=2)[0])
    assert int(LG.defer_ok(L, 3, 0.0, 1.0)[1]) == LG.LR_DELTA_NONPOS
    assert int(LG.defer_ok(L, 2, 1.0, 1.0)[1]) == LG.LR_UNREGISTERED
    assert int(LG.invariant_errors(L, np.array([1, 1, -1, 1, 0]))) == 0
    assert int(LG.invariant_errors(L, np.array([0, 1, 0, -1, 0]))) == 3


def test_zz_report(capsys):
    """마지막 — 통합 시험의 가로채기 횟수·불일치 수를 보고한다 (조각 실행 뒤 REPORT 병합)."""
    if not REPORT:
        pytest.skip("통합 시험이 이 조각에서 돌지 않았다")
    with capsys.disabled():
        print("\n[ledger shadow report]")
        for k, r in REPORT.items():
            print(f"  {k:14s} {r['device']}/{r['mode']} syncs={r['syncs']} epochs={r['epochs']} admits={r['admits']} records={r['records']} "
                  f"samples={r['n_samples']} mismatch={r['mismatch']} diffs={r['diffs']} route={r['route']} "
                  f"transfer={r['transfer']} defer={r['defer']} pred_checks={r['pred_checks']}")
    assert all(r["mismatch"] == 0 and r["diffs"] == 0 for r in REPORT.values())
