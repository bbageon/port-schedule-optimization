"""YR-167 자격용 **실측 관측기** (39차 감사 보정).

기존 자격 검사 W5(스냅샷)·W8′(재고 소진)·W4(본선)는 이름만 있고 실질을 보증하지
못했다 — 블록 수 21 확인, 생성단계 fallback 확인, 배치 폭 확인이 전부였다.
여기서는 **런 중에 실제로 수집**한다:

  · `SnapshotCollector` — 5분 격자에서 블록별 재공·대기열·backlog·재고·여유칸
  · `vessel_concurrency` — 시간대별 동시 활성 본선(설계값 c=6 은 **평균**이다)
  · `vessel_realized` — 본선 계획 물량 대비 실제 완료(작업률 앵커 대조용)
  · `run_digest` — 실현 시각 전체의 해시 (시뮬레이션 결정론 검사)

수집은 review epoch(전 블록 동일 시각 park)에서만 읽으므로 블록 시계가 어긋난
상태를 보지 않는다. 런에 개입하지 않는다(읽기 전용).
"""
from __future__ import annotations

import hashlib
from statistics import fmean


def _wip_queue(sim, t: float) -> tuple[int, int]:
    """(재공, 대기열) — 재공 = A ≤ t < O, 대기열 = B ≤ t < S (미시작)."""
    tl = getattr(sim, "time_ledger", None)
    if tl is None:
        return 0, 0
    wip = q = 0
    for r in tl.records.values():
        if r.gate_in is not None and r.gate_in <= t and (
                r.gate_out is None or r.gate_out > t):
            wip += 1
        if r.block_arrival is not None and r.block_arrival <= t and (
                r.service_start is None or r.service_start > t):
            q += 1
    return wip, q


def _backlog(sim, t: float) -> int:
    """t 시점에 **이미 존재하고** 아직 완료되지 않은 작업 수.

    yr149.backlog_at 은 사후 재구성이라 t 이후 투입된 트럭까지 세어 부풀려진다
    (5차 계약은 런 중 투입이 있다). 여기서는 release/도착 시각으로 존재 여부를
    먼저 거른다.
    """
    n = 0
    for j in sim.jobs.values():
        if j.status.name == "CANCELLED":
            continue
        born = (j.actual_gate_in if j.is_external_truck else j.release_time)
        if born is None or born > t + 1e-9:
            continue
        done = (j.status.name == "DONE" and j.service_end is not None
                and j.service_end <= t + 1e-9)
        if not done:
            n += 1
    return n


class SnapshotCollector:
    """5분 격자 스냅샷 — review_fn 안에서 호출한다(런 개입 없음)."""

    def __init__(self, times):
        self.times = [round(t, 6) for t in times]
        self._want = set(self.times)
        self._seen: set[float] = set()
        self.rows: list[dict] = []       # {t, block, wip, queue, backlog, stock, free}

    def observe(self, mbt, t: float) -> None:
        k = round(t, 6)
        if k not in self._want or k in self._seen:
            return
        self._seen.add(k)
        for bid, sim in mbt.blocks.items():
            wip, q = _wip_queue(sim, t)
            self.rows.append({
                "t": k, "block": bid, "wip": wip, "queue": q,
                "backlog": _backlog(sim, t),
                "stock": len(sim.stacks.containers),
                "free": mbt.free_slots(bid)})

    # --- 요약 ---
    def complete(self) -> bool:
        return self._seen == self._want

    def summary(self) -> dict:
        if not self.rows:
            return {"n_rows": 0}
        stock = [r["stock"] for r in self.rows]
        free = [r["free"] for r in self.rows]
        wip = [r["wip"] for r in self.rows]
        q = [r["queue"] for r in self.rows]
        bl = [r["backlog"] for r in self.rows]
        by_t: dict[float, int] = {}
        for r in self.rows:
            by_t[r["t"]] = by_t.get(r["t"], 0) + r["wip"]
        return {
            "n_rows": len(self.rows), "n_times": len(self._seen),
            "complete": self.complete(),
            "block_stock_min": min(stock), "block_stock_max": max(stock),
            "block_free_min": min(free),
            "block_wip_max": max(wip), "block_wip_mean": round(fmean(wip), 2),
            "block_queue_max": max(q), "block_queue_mean": round(fmean(q), 2),
            "block_backlog_max": max(bl), "block_backlog_mean": round(fmean(bl), 2),
            "terminal_wip_max": max(by_t.values()),
            "terminal_wip_mean": round(fmean(by_t.values()), 2),
            "degenerate": len(set(wip)) <= 1 or len(set(q)) <= 1,
        }

    def hourly_terminal_wip(self) -> dict[str, float]:
        by_h: dict[int, dict[float, int]] = {}
        for r in self.rows:
            by_h.setdefault(int(r["t"] // 3600), {}).setdefault(r["t"], 0)
            by_h[int(r["t"] // 3600)][r["t"]] += r["wip"]
        return {str(h): round(fmean(v.values()), 1) for h, v in sorted(by_h.items())}


def vessel_concurrency(schedule: list[dict], dur_s: float, end_s: float,
                       step_s: float = 300.0) -> dict:
    """동시 활성 본선 — 설계값 c 는 **평균**이며 순간값은 등간격 배치상 오르내린다."""
    n = int(end_s // step_s)
    series = [sum(1 for v in schedule
                  if v["start_s"] <= i * step_s < v["start_s"] + dur_s)
              for i in range(n)]
    hourly: dict[int, list[int]] = {}
    for i, c in enumerate(series):
        hourly.setdefault(int(i * step_s // 3600), []).append(c)
    return {"min": min(series), "max": max(series),
            "mean": round(fmean(series), 3),
            "hourly": {str(h): round(fmean(v), 2) for h, v in sorted(hourly.items())},
            "vessel_hours": round(len(schedule) * dur_s / 3600.0, 2),
            "planned_moves_per_h": None}


def vessel_realized(mbt, moves_per_vessel: int, end_s: float) -> dict:
    """본선 연계 작업의 실제 완료 — 계획 대비 완료율과 실현 작업률(moves/h)."""
    planned = done = done_in_window = 0
    for sim in mbt.blocks.values():
        for j in sim.jobs.values():
            if not j.is_vessel_linked:
                continue
            planned += 1
            if j.status.name == "DONE":
                done += 1
                if j.service_end is not None and j.service_end < end_s:
                    done_in_window += 1
    return {"planned_moves": planned, "done_moves": done,
            "done_in_window": done_in_window,
            "completion_ratio": round(done / planned, 4) if planned else None,
            "realized_moves_per_h": round(done_in_window / (end_s / 3600.0), 1),
            "moves_per_vessel": moves_per_vessel}


def cancelled_external(mbt) -> int:
    """런타임에 취소된 외부트럭 작업 수 (반출 대상 소실 등)."""
    return sum(1 for sim in mbt.blocks.values() for j in sim.jobs.values()
               if j.is_external_truck and j.status.name == "CANCELLED")


def stock_conservation(mbt, snap: SnapshotCollector) -> dict:
    """재고 보존 항등식 — Δ재고 == (완료 반입 + 완료 양하) − (완료 반출 + 완료 적하).

    임계값이 아니라 **항등식**이라 사후 조정 여지가 없다. 아무 일도 안 한 런이
    검사를 통과하는 것을 막는 실측 근거이기도 하다(양변이 실제 처리량이다).
    """
    if not snap.rows:
        return {"ok": False, "reason": "스냅샷 없음"}
    t0, t1 = min(r["t"] for r in snap.rows), max(r["t"] for r in snap.rows)
    first = {r["block"]: r["stock"] for r in snap.rows if r["t"] == t0}
    last = {r["block"]: r["stock"] for r in snap.rows if r["t"] == t1}
    worst = 0.0
    detail = {"in": 0, "out": 0, "disc": 0, "load": 0, "other": 0}
    for bid, sim in mbt.blocks.items():
        d = {"in": 0, "out": 0, "disc": 0, "load": 0, "other": 0}
        for j in sim.jobs.values():
            if j.status.name != "DONE" or j.service_end is None:
                continue
            if not (t0 - 1e-9 < j.service_end <= t1 + 1e-9):
                continue
            f = j.flow.name
            key = {"GATE_IN": "in", "GATE_OUT": "out",
                   "VESSEL_DISCHARGE": "disc", "VESSEL_LOAD": "load"}.get(f, "other")
            d[key] += 1
        for k in detail:
            detail[k] += d[k]
        expect = d["in"] + d["disc"] - d["out"] - d["load"]
        worst = max(worst, abs((last[bid] - first[bid]) - expect))
    return {"ok": worst < 1e-9, "max_abs_error": worst,
            "window": [t0, t1], "done_by_flow": detail,
            "terminal_stock_delta": sum(last.values()) - sum(first.values())}


def run_digest(mbt) -> str:
    """실현 시각 + **엔진 사건열** 해시 — 같은 시드 재실행이 같은 런인지 검사(W3).

    장부 시각만 보면 사건 순서가 달라져도 같은 값이 나올 수 있으므로
    engine.event_stream_hash() 를 함께 넣는다.
    """
    h = hashlib.sha256()
    for bid in sorted(mbt.blocks):
        sim = mbt.blocks[bid]
        try:
            h.update(f"{bid}|EV|{sim.event_stream_hash()}\n".encode())
        except Exception:
            pass
        tl = getattr(sim, "time_ledger", None)
        if tl is None:
            continue
        for jid in sorted(tl.records):
            r = tl.records[jid]
            h.update(f"{bid}|{jid}|{r.gate_in}|{r.block_arrival}|"
                     f"{r.service_start}|{r.job_done}|{r.gate_out}\n".encode())
    return h.hexdigest()
