"""반사실 세계를 **여러 프로세스에 나눠** 굴린다 ([[YR-219]]).

■ 왜 GPU 가 아니라 프로세스인가
  병목이 신경망이 아니라 **시뮬레이터**다. 망은 5,761 + 5,505 파라미터뿐이라
  GPU 로 보내는 전송 비용이 계산보다 크다. 반면 반사실 세계 하나는 파이썬
  이산사건 시뮬레이션 **15초 안팎**이고, 이게 회차 시간의 대부분이다.

■ 왜 나눌 수 있나 — 세계들이 **완전히 독립**이다
  각 세계는 자기 스냅샷을 복제해 따로 굴리고, 서로의 결과를 안 본다. 공유 상태가
  없으므로 프로세스로 쪼개도 결과가 안 바뀐다. 탐색도 좌표 기반이라(`explore.py`)
  **순서에 의존하지 않는다** — 순차 난수였다면 이 병렬화가 결과를 바꿨을 것이다.

■ 무엇이 안 나뉘나 (Amdahl)
  에피소드 자체는 **순차**다 — 하루를 시간순으로 굴려야 하고 결정이 그 안에서
  일어난다. 회차 시간 = `에피소드(순차) + 세계들(병렬) + 학습(짧음)` 이라
  가속 상한이 에피소드 몫에 걸린다.

■ 메모리
  스냅샷 하나가 터미널 전체(컨테이너 2만·작업 4천)라 가볍지 않다. 무한정 쌓지
  않도록 **동시 진행 작업 수에 뚜껑**을 씌운다(`max_inflight`).

■ 작업자에는 무엇이 가나
  `ctx`(망·레이아웃·공고기)는 **작업자마다 한 번만** 보낸다(initializer). 작업마다
  가는 것은 스냅샷과 강제 행동뿐이다.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass

from .rollout import SnapshotRollout

#: 작업자 안에서 쓰는 전역 — initializer 가 한 번 채운다.
_CTX = None
_HORIZON = 3600.0


@dataclass
class BranchJob:
    """결정 하나 = 세계 셋. 작업자에게 이 단위로 보낸다."""

    doc_key: str
    t: float
    mbt: object                 # ★결정 **전** 스냅샷
    orders: dict
    records: dict
    decided: set
    seller_alt: str
    buyer_alt: str | None


def _init_worker(ctx, horizon_s: float) -> None:
    global _CTX, _HORIZON
    _CTX, _HORIZON = ctx, float(horizon_s)
    try:                       # 작업자마다 1스레드 — 20프로세스 × N스레드는 서로 방해만 한다
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_job(job: BranchJob) -> dict:
    """세계 셋을 굴린다. **작업자 프로세스에서 도는 함수**(전역이어야 pickle 된다)."""
    return run_job_with(_CTX, _HORIZON, job)


def run_job_with(ctx, horizon_s: float, job: BranchJob) -> dict:
    """`ctx` 를 명시로 받는 판 — 단일 프로세스 경로와 시험이 같은 코드를 쓴다."""
    roll = SnapshotRollout(ctx, horizon_s=horizon_s)
    kw = dict(orders=job.orders, records=job.records, decided=job.decided,
              doc_key=job.doc_key)
    fact = roll.branch(job.mbt, job.t, **kw)
    alt = roll.branch(job.mbt, job.t, **kw,
                      force_seller=(job.doc_key, job.seller_alt))
    out = {"doc_key": job.doc_key, "t": job.t,
           "phi_factual": fact.phi_krw,
           "phi_seller_alt": alt.phi_krw,
           "seller_alt_coord": alt.seller_coord,
           "factual": fact, "worlds": 2}
    if job.buyer_alt is not None:
        b = roll.branch(job.mbt, job.t, **kw,
                        force_buyer=(job.doc_key, job.buyer_alt))
        out["phi_buyer_alt"] = b.phi_krw
        out["worlds"] = 3
    return out


def default_workers() -> int:
    """코어 수 − 여유 2. 환경변수 `V3_WORKERS` 로 덮어쓴다."""
    env = os.environ.get("V3_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 2) - 2)


class BranchPool:
    """작업을 받아 두었다가 굴린다. `workers <= 1` 이면 **같은 프로세스에서** 돈다.

    단일 프로세스 경로를 남겨 두는 이유는 재현 대조 때문이다 — 병렬 결과가 순차
    결과와 같은지 검사할 수 있어야 한다(`tests/v3/test_branch_pool.py`).
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

    def submit(self, job: BranchJob) -> None:
        """작업 하나를 맡긴다. 에피소드가 도는 **중에** 불려도 된다."""
        if self._ex is None:
            self._done.append(run_job_with(self.ctx, self.horizon_s, job))
            return
        # ★뚜껑 — 스냅샷이 무한정 쌓이지 않게 한다.
        while len(self._futs) >= self.max_inflight:
            done, pend = wait(self._futs, return_when="FIRST_COMPLETED")
            self._collect(done)
            self._futs = list(pend)
        self._futs.append(self._ex.submit(run_job, job))

    def _collect(self, futs) -> None:
        for f in futs:
            self._done.append(f.result())

    def results(self) -> list[dict]:
        """전부 끝날 때까지 기다렸다가 **결정 시각 순서로** 돌려준다.

        순서를 못 박는 이유는 재현이다 — 작업자가 끝나는 순서는 매번 다르다.
        """
        if self._futs:
            done, _ = wait(self._futs)
            self._collect(done)
            self._futs = []
        self._done.sort(key=lambda r: (r["t"], r["doc_key"]))
        self.n_worlds = sum(r["worlds"] for r in self._done)
        return self._done

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None
