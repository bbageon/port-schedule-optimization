"""크레인 결정 하나에서 **두 세계**를 굴린다 ([[YR-308]] · [[YR-248]] 2단계).

■ 재배정층과 무엇이 같고 무엇이 다른가
  같다 — 같은 순간에서 **행동만 바꿔** H(3시간)를 다시 굴리고 Φ 차이를 정답으로 쓴다.
  다르다 — **분기점이 review epoch 이 아니라 크레인 결정 시각**이다.

      재배정층   epoch(60초 격자)에서 시장이 열린다 → 전 블록이 같은 t 에 park
      크레인층   크레인이 놀게 된 그 순간 → 한 블록만 결정 중이고 나머지는 제각각

  그래서 분기 절차가 하나 다르다. 스냅샷을 뜰 때 그 블록은 이미 `_pending`
  (결정을 열어 둔 상태)이라 복제본을 그냥 `run()` 하면
  *"직전 결정 미해소"* 로 엔진이 거절한다. 대신 **그 결정을 손으로 한 번 돌리고**
  (여기서 강제를 먹인다) 그 뒤부터 평소대로 굴린다.

■ 왜 세계가 둘인가 (재배정층은 셋)
  재배정층은 판매 하나에 구매 응답이 따로 붙어 (사실 · 판매대안 · 구매대안) 셋이다.
  크레인 결정은 응답이 없다 — **고른 것**과 **차점자** 둘뿐이다.

■ ★사실 세계도 굴린다
  실제 궤적에서 읽지 않는다. 분기 세계는 `sim.end = t+H` 라 창끝 직전에 새 작업을
  시작하지 않는데 실제 궤적은 계속 구르기 때문이다 — 한쪽만 절단하면 그 편향이
  라벨에 그대로 들어간다 (`stage/rollout.py` 머리말의 실측 참조).

■ ⚠️ 강제가 안 먹을 수 있다
  resolver 는 `mandatory` 를 어떤 선호보다 앞에 둔다(엔진 계약). 필수 작업이 걸린
  결정에서는 차점자를 강제해도 필수가 이긴다. 그런 세계는 **버린다** — 안 버리면
  Φ_alt 가 Φ_factual 과 같아져 *"이 선택은 무의미"* 를 배운다.
"""
from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass

from ..reward.counterfactual import _count_rollout
from ..reward.phi import terminal_cost_krw


@dataclass
class CraneBranchJob:
    """크레인 결정 하나 = 세계 둘. 작업자에게 이 단위로 보낸다."""

    key: tuple                  # (t, block, crane) — 결과를 짝짓는 열쇠
    t: float
    block: str
    crane: str
    mbt: object                 # ★결정 **직전** 스냅샷 (`_pending` 이 열린 채)
    decision: object            # TerminalDecision — 시각과 크레인 목록뿐
    orders: dict
    records: dict
    decided: set
    picked_job: str             # 실제로 고른 일감
    alt_job: str                # 차점자


class CraneRollout:
    """분기점 하나에서 세계를 굴린다. `stage/rollout.SnapshotRollout` 의 크레인 판."""

    def __init__(self, ctx, *, horizon_s: float):
        self.ctx = ctx
        self.horizon_s = float(horizon_s)
        self.n_worlds = 0

    def branch(self, job: CraneBranchJob, *, force_job: str | None) -> dict:
        """`force_job` 을 그 크레인에 먹이고 H 만큼 굴린다. `None` 이면 사실 세계.

        돌려주는 것: `{"phi": Φ(원), "picked": 그 세계가 실제로 고른 일감}`.
        `picked` 를 함께 내는 이유는 **강제가 먹었는지 확인**하기 위해서다.
        """
        _count_rollout()
        self.n_worlds += 1
        end = job.t + self.horizon_s

        snap = copy.deepcopy(job.mbt)
        for sim in snap.blocks.values():
            sim.end = min(sim.end, end)

        o2 = dict(job.orders)
        r2 = copy.deepcopy(job.records)
        market2 = self.ctx.make_market(snap, decided=set(job.decided))
        bridge2 = self.ctx.make_bridge(market2, orders=o2, records=r2,
                                       on_decision=None)      # ★교사 재귀 금지
        ann2 = self.ctx.announcer.clone_fresh()

        seen: dict = {}

        def watch(sim, dp, gb, assign):
            """첫 결정에서 그 크레인이 무엇을 잡았는지만 적는다."""
            if "picked" not in seen:
                gc = assign.get(job.crane)
                ref = getattr(gc, "job_ref", None)
                seen["picked"] = getattr(ref, "job_id", None)

        exec_policy = self.ctx.make_exec_policy(on_crane=watch)
        pref = getattr(exec_policy, "pref", None)

        # ── ① 열려 있는 결정을 손으로 한 번 돌린다 (여기서만 강제가 먹는다)
        if force_job is not None and hasattr(pref, "force_once"):
            pref.force_once[job.crane] = force_job
        exec_policy(snap.blocks[job.block], job.decision)
        if hasattr(pref, "force_once"):
            pref.force_once.clear()          # ★한 번만 — 이후는 정책이 평소대로

        # ── ② 그 뒤로는 평소대로 H 까지
        def review(m, tt):
            ann2.review(m, tt)
            bridge2.review(m, tt)

        snap.run(exec_policy, review_fn=review)
        bridge2._sync(snap, end)             # 마지막 epoch 뒤 완료분까지 흡수

        phi = terminal_cost_krw(r2, end_s=end,
                                vessel_idle=self.ctx.vessel_idle(snap, end),
                                yc_extra_move_s=self.ctx.yc_extra_move_s(snap),
                                rehandles=self.ctx.rehandles(snap))
        return {"phi": phi.total, "picked": seen.get("picked")}


# ──────────────────────────────────────────────────────────── 작업자 프로세스
_CTX = None
_HORIZON = 10_800.0


def _init_worker(ctx, horizon_s: float) -> None:
    global _CTX, _HORIZON
    _CTX, _HORIZON = ctx, float(horizon_s)
    try:                       # 작업자마다 1스레드 — 서로 방해만 한다
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_crane_job(job: CraneBranchJob) -> dict:
    """세계 둘을 굴린다. **작업자에서 도는 함수**(전역이어야 pickle 된다)."""
    return run_crane_job_with(_CTX, _HORIZON, job)


def run_crane_job_with(ctx, horizon_s: float, job: CraneBranchJob) -> dict:
    roll = CraneRollout(ctx, horizon_s=horizon_s)
    fact = roll.branch(job, force_job=None)
    alt = roll.branch(job, force_job=job.alt_job)
    return {"key": job.key, "t": job.t, "crane": job.crane,
            "picked_job": job.picked_job, "alt_job": job.alt_job,
            "phi_factual": fact["phi"], "phi_alt": alt["phi"],
            #: ★두 진단 — 없으면 거짓 라벨이 조용히 섞인다
            "factual_ok": fact["picked"] == job.picked_job,
            "forced_ok": alt["picked"] == job.alt_job,
            "worlds": 2}


def default_workers() -> int:
    env = os.environ.get("V3_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 2) - 2)


class CranePool:
    """작업을 받아 두었다가 굴린다. `workers <= 1` 이면 같은 프로세스에서 순차로.

    `stage/branchpool.BranchPool` 과 같은 골격이다 — 작업·결과 모양만 다르다.
    """

    def __init__(self, ctx, *, horizon_s: float, workers: int = 1,
                 max_inflight: int | None = None):
        self.ctx = ctx
        self.horizon_s = float(horizon_s)
        self.workers = max(1, int(workers))
        self.max_inflight = max_inflight or (self.workers * 2)
        self._ex = None
        self._futs: list = []
        self._done: list[dict] = []
        self.n_worlds = 0

    def __enter__(self):
        if self.workers > 1:
            self._ex = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_init_worker,
                initargs=(self.ctx, self.horizon_s))
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def submit(self, job: CraneBranchJob) -> None:
        if self._ex is None:
            self._done.append(run_crane_job_with(self.ctx, self.horizon_s, job))
            return
        while len(self._futs) >= self.max_inflight:   # ★스냅샷 무한 적재 방지
            done, pend = wait(self._futs, return_when="FIRST_COMPLETED")
            self._done.extend(f.result() for f in done)
            self._futs = list(pend)
        self._futs.append(self._ex.submit(run_crane_job, job))

    def results(self) -> list[dict]:
        """전부 끝날 때까지 기다렸다가 **결정 시각 순서로** — 재현을 위해서다."""
        if self._futs:
            done, _ = wait(self._futs)
            self._done.extend(f.result() for f in done)
            self._futs = []
        self._done.sort(key=lambda r: (r["t"], r["crane"]))
        self.n_worlds = sum(r["worlds"] for r in self._done)
        return self._done

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None
