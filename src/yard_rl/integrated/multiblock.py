"""YR-099-b — 다중블록 재배정 브리지 계약 (공용 시계·전역 장부·2단계 commit).

MVP(yr099_midrun_review)가 근사로 둔 6가지를 계약으로 고친다 (사용자 순서 결정 2026-07-27
— "기반이 틀리면 YR-105 임계를 잘 찾아도 실제 중앙 resolver 에서 재현되지 않는다"):

| # | MVP 근사 | 본 모듈의 계약 |
|---|---|---|
| ① | 결정시점 = 다음 결정경계(드리프트 ≤1결정·missed 창 발생) | **정확한 gate-in epoch** (engine `review_epochs` 훅) |
| ② | 두 sim 각자 시계 (lockstep 근사) | **공용 시계** — 전 블록이 같은 epoch 에 park 한 뒤에야 review |
| ③ | 블록별 TimeLedger 를 이관 (A 가 블록을 따라다님) | **전역 A→O 장부** — A 는 터미널이 보유, 이관해도 연속 |
| ④ | job_id 를 `@M` 로 개명 (정체성 파괴) | **canonical id 불변** + `owner`/`version`/`transfer_history` |
| ⑤ | 단일 함수 이식 (부분 실패 시 상태 불명) | **prepare → validate → commit/rollback** 2단계 |
| ⑥ | 수신 블록 무한수용 가정 | **수신 용량 검사** (물리 슬롯 + 미도착 예약분 차감) |

블록 간 id 네임스페이스는 **구성 시 1회** `{block}:{job_id}` 로 통일해 터미널 전역 유일성을
만든다 — 이후 이송해도 id 는 불변이라 장부·감사가 끊기지 않는다(MVP 의 개명 해킹 제거).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

from ..domain.enums import JobFlow, JobStatus
from .engine import ReviewEpoch, TerminalDecision

CAPACITY_MARGIN = 2          # 수신 블록에 남겨둘 여유 슬롯 (assumed — 만재 직전 이송 금지)


# ---------------------------------------------------------------- 전역 장부 (계약 ③④)
@dataclass
class JobRecord:
    """터미널 전역 작업 원장 — 블록을 옮겨도 이 레코드 하나가 A→O 를 잇는다."""

    job_id: str                       # canonical (불변)
    origin_block: str                 # TOS 최초 배정 (감사용 불변)
    owner: str                        # 현재 실행 블록
    flow: str
    version: int = 0                  # 낙관적 동시성 (prepare 시점 대비 변경 감지)
    transfer_count: int = 0
    transfer_history: tuple[tuple[str, str, float], ...] = ()   # (src, dst, t)
    a_gate_in: float | None = None    # A — 터미널 보유 (이송 무관)
    b_block_arrival: float | None = None
    c_job_done: float | None = None
    o_gate_out: float | None = None
    locked: bool = False              # block-in/배정 이후 = 재배정 금지

    @property
    def reassignable(self) -> bool:
        return (not self.locked and self.flow == JobFlow.GATE_IN.value
                and self.b_block_arrival is None)


class TerminalLedger:
    """전역 A→O 장부 — 블록별 TimeLedger(B→C 비용)와 분리된 터미널 단일 원장."""

    def __init__(self) -> None:
        self.records: dict[str, JobRecord] = {}

    def register(self, rec: JobRecord) -> None:
        self.records[rec.job_id] = rec

    def harvest(self, blocks: dict[str, object]) -> None:
        """각 블록의 실현 시각(B·C·O)을 canonical id 로 흡수 — 소유 블록 무관."""
        for bid, sim in blocks.items():
            for jid, j in sim.jobs.items():
                rec = self.records.get(jid)
                if rec is None:
                    continue
                tl = getattr(sim, "time_ledger", None)
                r = tl.records.get(jid) if tl is not None else None
                if r is not None:
                    rec.b_block_arrival = r.block_arrival if r.block_arrival is not None \
                        else rec.b_block_arrival
                    rec.c_job_done = r.job_done if r.job_done is not None else rec.c_job_done
                    rec.o_gate_out = r.gate_out if r.gate_out is not None else rec.o_gate_out
                if j.status in (JobStatus.WAITING, JobStatus.ASSIGNED, JobStatus.RUNNING,
                                JobStatus.DONE):
                    rec.locked = True

    def a_to_o_samples_s(self, end: float) -> list[float]:
        """터미널 턴타임 A→O (미완료는 end−A 검열 — 미완료가 이득 보지 않게)."""
        out = []
        for r in self.records.values():
            if r.a_gate_in is None:
                continue
            if r.o_gate_out is not None:
                out.append(r.o_gate_out - r.a_gate_in)
            else:
                out.append(max(0.0, end - r.a_gate_in))
        return out


# ---------------------------------------------------------------- 2단계 transaction (계약 ⑤)
@dataclass(frozen=True)
class TransferTxn:
    job_id: str
    src: str
    dst: str
    seen_version: int             # prepare 시점 version (validate 에서 재확인)
    new_arrival_s: float
    prepared_at_s: float
    route_s: float
    # 게이트 D: 트랜잭션 **고유 식별자**. 구판은 예약 키가 (job, dst, prepared_at) 이라
    # ①같은 시각 재-prepare 시 키가 겹쳐 예약이 영구 누수되고 ②rollback 한 txn 을 다시
    # commit 할 수 있었다(멱등 해제가 키를 지워버려 재사용 가능). 단조증가 id 로 분리한다.
    txn_id: int = -1


class TransferError(RuntimeError):
    pass


# ---------------------------------------------------------------- 조정자
class MultiBlockTerminal:
    """공용 시계로 N개 블록 sim 을 구동하고, 반입 재배정을 원자적으로 확정한다."""

    def __init__(self, blocks: dict[str, object], *,
                 capacity_margin: int = CAPACITY_MARGIN) -> None:
        self.blocks = dict(blocks)
        self.ledger = TerminalLedger()
        self.capacity_margin = capacity_margin
        self._reserved_inbound: dict[str, int] = {b: 0 for b in self.blocks}
        self._open_txn: set[int] = set()        # 살아있는 예약의 txn_id (rollback 멱등)
        self._txn_seq: int = 0                  # 게이트 D: 트랜잭션 고유 id 발급기
        self.route_cost_s: float = 0.0          # 이송 추가주행 누적 (critical-2)
        self._parked: dict[str, float] = {}
        self._terminal: set[str] = set()
        for bid, sim in self.blocks.items():
            _namespace_jobs(sim, bid)
            for jid, j in sim.jobs.items():
                self.ledger.register(JobRecord(
                    job_id=jid, origin_block=bid, owner=bid, flow=j.flow.value,
                    a_gate_in=getattr(j, "actual_gate_in", None)))
        self._schedule_review_epochs()

    # -------------------------------------------------- 공용 시계 (계약 ①②)
    @property
    def now(self) -> float:
        live = [s.clock for b, s in self.blocks.items() if b not in self._terminal]
        return min(live) if live else max(s.clock for s in self.blocks.values())

    def _schedule_review_epochs(self) -> None:
        """재배정 가능한 반입의 **gate-in 시각**을 전 블록 sim 에 동일하게 예약.

        전 블록이 같은 epoch 에 park 해야 review 가 열리므로(=공용 시계), 관측되는
        혼잡은 정확히 그 시각의 상태다(이산사건 시뮬의 상태는 이벤트 사이 불변).
        """
        ts = set()
        for sim in self.blocks.values():
            end = sim.end
            for j in sim.jobs.values():
                a = getattr(j, "actual_gate_in", None)
                if a is not None and j.flow == JobFlow.GATE_IN and 0.0 < a <= end:
                    ts.add(round(a, 6))
        eps = sorted(ts)
        for sim in self.blocks.values():
            sim.review_epochs = [t for t in eps if t <= sim.end]

    def run(self, policy_fn, review_fn=None, cost_fn=None) -> dict:
        """공용 시계 루프. policy_fn(sim, dp) → assignment · review_fn(self, t) → None.

        cost_fn(sim, t0, t1, raw) → float 이면 블록별 구간비용을 누적한다.
        """
        totals = {b: 0.0 for b in self.blocks}
        last = {b: s.now for b, s in self.blocks.items()}
        for s in self.blocks.values():
            s.cost.cut()
        guard = 0
        while len(self._terminal) < len(self.blocks):
            guard += 1
            if guard > 2_000_000:
                raise RuntimeError("multiblock 루프 상한 — 엔진 계약 위반 의심")
            live = [b for b in self.blocks if b not in self._terminal]
            # park 된 블록은 전진 대상에서 제외 (전원 park → review 발화)
            movable = [b for b in live if b not in self._parked]
            if not movable:
                t = min(self._parked.values())
                self._parked.clear()
                if review_fn is not None:
                    review_fn(self, t)
                continue
            bid = min(movable, key=lambda b: self.blocks[b].clock)
            sim = self.blocks[bid]
            out = sim.run_until_decision()
            if cost_fn is not None:
                raw = sim.cost.cut()
                totals[bid] += cost_fn(sim, last[bid], sim.now, raw)
                last[bid] = sim.now
            if out is None:
                self._terminal.add(bid)
                self._parked.pop(bid, None)
            elif isinstance(out, ReviewEpoch):
                self._sync_locks(sim)          # major-6: 런 중 원장 lock/B 갱신
                self._parked[bid] = out.time
            elif isinstance(out, TerminalDecision):
                policy_fn(sim, out)
        self.ledger.harvest(self.blocks)
        return {"totals": totals, "route_cost_s": self.route_cost_s,
                "terminal_total": round(sum(totals.values()), 6),
                "end": max(s.end for s in self.blocks.values())}

    def _sync_locks(self, sim) -> None:
        """검증 major-6: 원장 `locked`/`b_block_arrival` 을 **런 중에** 갱신.

        (기존엔 harvest 가 종료 시 1회라 `reassignable` 이 항상 True — 실제 창 방어는
        `status != PLANNED` 검사가 하고 있었다. 원장 수준 lock 계약을 실제로 살린다.)
        """
        tl = getattr(sim, "time_ledger", None)
        for jid, j in sim.jobs.items():
            rec = self.ledger.records.get(jid)
            if rec is None or rec.locked:
                continue
            if j.status != JobStatus.PLANNED:
                rec.locked = True
                r = tl.records.get(jid) if tl is not None else None
                if r is not None and r.block_arrival is not None:
                    rec.b_block_arrival = r.block_arrival

    # -------------------------------------------------- 용량 (계약 ⑥)
    def free_slots(self, bid: str) -> int:
        """수신 가능 슬롯 = 물리 여유 − 미도착 예약분. 서비스 구간·규격 제약은 commit 시 재검."""
        sim = self.blocks[bid]
        g = sim.profile.block
        stk = sim.stacks
        used = len(stk.containers)
        phys = g.bay_count * g.row_count * g.tier_max
        pending = sum(1 for j in sim.jobs.values()
                      if j.status == JobStatus.PLANNED and j.flow in
                      (JobFlow.GATE_IN, JobFlow.VESSEL_DISCHARGE))
        return phys - used - pending - self._reserved_inbound[bid]

    # -------------------------------------------------- 2단계 transaction (계약 ⑤)
    def prepare_transfer(self, job_id: str, dst: str, *, route_s: float,
                         travel_s: float) -> TransferTxn:
        rec = self.ledger.records.get(job_id)
        if rec is None:
            raise TransferError(f"{job_id}: 미등록")
        if dst not in self.blocks or dst == rec.owner:
            raise TransferError(f"{job_id}: 수신 블록 부적격 {dst}")
        if not rec.reassignable:
            raise TransferError(f"{job_id}: lock/자격 위반")
        src_sim = self.blocks[rec.owner]
        j = src_sim.jobs.get(job_id)
        if j is None or j.status != JobStatus.PLANNED:
            raise TransferError(f"{job_id}: 소스 상태 위반")
        if rec.a_gate_in is None or rec.a_gate_in > self.now + 1e-6:
            raise TransferError(f"{job_id}: gate-in 전 (창 밖)")
        # 검증 major-4: 장부 유무 비대칭이면 이송 시 트럭 시간이 통째로 증발 — fail-closed
        if (src_sim.time_ledger is None) != (self.blocks[dst].time_ledger is None):
            raise TransferError(f"{job_id}: 블록 간 time_ledger 비대칭 (장부 유실 위험)")
        if self.free_slots(dst) <= self.capacity_margin:
            raise TransferError(f"{dst}: 용량 부족 (free={self.free_slots(dst)})")
        arr = rec.a_gate_in + travel_s + route_s
        if arr <= self.blocks[dst].clock + 1e-9 or arr > self.blocks[dst].end:
            raise TransferError(f"{job_id}: 도착시각 무효 {arr:.1f}")
        self._reserved_inbound[dst] += 1                     # 예약 (rollback 대상)
        self._txn_seq += 1
        txn = TransferTxn(job_id=job_id, src=rec.owner, dst=dst, seen_version=rec.version,
                          new_arrival_s=arr, prepared_at_s=self.now, route_s=route_s,
                          txn_id=self._txn_seq)
        self._open_txn.add(txn.txn_id)
        return txn

    def validate(self, txn: TransferTxn) -> None:
        rec = self.ledger.records[txn.job_id]
        if rec.version != txn.seen_version or rec.owner != txn.src:
            raise TransferError(f"{txn.job_id}: version/owner 변경 (stale quote)")
        if not rec.reassignable:
            raise TransferError(f"{txn.job_id}: 준비 후 lock")
        j = self.blocks[txn.src].jobs.get(txn.job_id)
        if j is None or j.status != JobStatus.PLANNED:
            raise TransferError(f"{txn.job_id}: 준비 후 상태 변경")
        dst_sim = self.blocks[txn.dst]
        spec = dst_sim.fleet.spec(dst_sim.profile.cranes[0].crane_id)
        if j.inbound_size is not None and dst_sim.stacks.find_slot(
                j.inbound_size, spec, spec.service_bay_min, 1.0) is None:
            raise TransferError(f"{txn.dst}: 규격 적합 슬롯 없음")

    def _precommit(self, txn: TransferTxn) -> int | None:
        """게이트 D — **변경 전에** 실패할 수 있는 검사를 전부 소진한다.

        (구판은 소스 장부 정합 검사가 `src.jobs.pop()` **뒤에** 있어, 거기서 예외가 나면
        작업이 어느 블록에도 없는 상태로 남았다 — `rollback` 은 예약만 풀 뿐 소유권을
        되돌리지 않으므로 `check_invariants` 의 "소유자 없음"으로만 뒤늦게 드러난다.)

        반환: 소스 장부에서 지울 `_a_sorted` 인덱스 (장부 없으면 None).
        """
        src, dst = self.blocks[txn.src], self.blocks[txn.dst]
        j = src.jobs[txn.job_id]
        if (src.time_ledger is not None or dst.time_ledger is not None) \
                and j.actual_gate_in is None:
            raise TransferError(f"{txn.job_id}: gate-in 결측 — 장부 이관 불가")
        if src.time_ledger is None:
            return None
        import bisect as _bs
        tl = src.time_ledger
        i = _bs.bisect_left(tl._a_sorted, j.actual_gate_in)
        if i >= len(tl._a_sorted) or tl._a_sorted[i] != j.actual_gate_in:
            raise TransferError(f"{txn.job_id}: 소스 장부에 A 부재 (정합 위반)")
        return i

    def _snapshot(self, txn: TransferTxn) -> dict:
        """게이트 D — 변경 구간에서 예상 못 한 예외가 나도 **원상복구**하기 위한 최소 스냅샷."""
        src, dst = self.blocks[txn.src], self.blocks[txn.dst]
        j = src.jobs[txn.job_id]
        rec = self.ledger.records[txn.job_id]

        def ledger_state(sim):
            tl = getattr(sim, "time_ledger", None)
            if tl is None:
                return None
            return (dict(tl.records), list(tl._a_sorted), tl._a_idx, tl._n_inside)

        return {"job": j, "src_heap": list(src.queue._heap), "dst_heap": list(dst.queue._heap),
                "arrival": j.actual_block_arrival,
                "est": getattr(j, "estimated_block_arrival", None),
                "eta": getattr(j, "provided_eta", None),
                "src_ledger": ledger_state(src), "dst_ledger": ledger_state(dst),
                "rec": (rec.owner, rec.version, rec.transfer_count, rec.transfer_history),
                "route_cost_s": self.route_cost_s}

    def _restore(self, txn: TransferTxn, snap: dict) -> None:
        src, dst = self.blocks[txn.src], self.blocks[txn.dst]
        jid = txn.job_id
        dst.jobs.pop(jid, None)
        src.jobs[jid] = snap["job"]
        src.queue._heap = snap["src_heap"]
        heapq.heapify(src.queue._heap)
        dst.queue._heap = snap["dst_heap"]
        heapq.heapify(dst.queue._heap)
        snap["job"].actual_block_arrival = snap["arrival"]
        if snap["est"] is not None:
            snap["job"].estimated_block_arrival = snap["est"]
        if snap["eta"] is not None:
            snap["job"].provided_eta = snap["eta"]
        for sim, st in ((src, snap["src_ledger"]), (dst, snap["dst_ledger"])):
            if st is None:
                continue
            tl = sim.time_ledger
            tl.records, tl._a_sorted, tl._a_idx, tl._n_inside = st[0], st[1], st[2], st[3]
        rec = self.ledger.records[jid]
        rec.owner, rec.version, rec.transfer_count, rec.transfer_history = snap["rec"]
        self.route_cost_s = snap["route_cost_s"]

    def commit(self, txn: TransferTxn) -> None:
        """검증 통과분만 **원자적으로** 이관 — 전부 반영되거나 전혀 반영되지 않는다.

        게이트 D 구조: ①닫힌 txn 거절 ②validate(무변경) ③_precommit(무변경·실패 가능 검사
        소진) ④변경 구간(예외 시 _restore 로 원상복구 후 재발생).
        """
        if txn.txn_id not in self._open_txn:
            # 게이트 D: rollback 한 txn 을 다시 commit 하면 **예약 없이** 이송이 성사됐다.
            raise TransferError(f"{txn.job_id}: 닫힌 트랜잭션 (이미 commit/rollback 됨)")
        self.validate(txn)
        src_i = self._precommit(txn)
        snap = self._snapshot(txn)
        try:
            src, dst = self.blocks[txn.src], self.blocks[txn.dst]
            jid = txn.job_id
            j = src.jobs.pop(jid)
            src.queue._heap = [e for e in src.queue._heap
                               if not (e.kind_name == "BLOCK_ARRIVAL" and e.payload == jid)]
            heapq.heapify(src.queue._heap)
            if src.time_ledger is not None:                   # 블록 장부에서만 해제
                # 검증 major-3: _a_sorted 만 고치면 그것을 가리키는 _a_idx·_n_inside 가 어긋나
                # terminal_area 가 조용히 틀어진다(실측 226.9s 오차). 포인터도 함께 보정.
                tl = src.time_ledger
                tl.records = dict(tl.records)                 # 스냅샷과 별개 객체로
                tl.records.pop(jid, None)
                tl._a_sorted = list(tl._a_sorted)
                del tl._a_sorted[src_i]
                if src_i < tl._a_idx:
                    tl._a_idx -= 1
                    tl._n_inside -= 1
            j.actual_block_arrival = txn.new_arrival_s
            est = getattr(j, "estimated_block_arrival", None)
            if est is not None:
                j.estimated_block_arrival = est + txn.route_s
            # 게이트 D: **정책이 실제로 읽는 예측 도착은 provided_eta** 다. 구판은
            # estimated_block_arrival 만 밀어 두 필드가 이송 후 서로 어긋났다(생성기가
            # 별칭으로 만든 값이므로 함께 밀어야 정책이 보는 세계가 일관된다).
            eta = getattr(j, "provided_eta", None)
            if eta is not None:
                j.provided_eta = eta + txn.route_s
            dst.jobs[jid] = j
            from .events import EventKind
            dst.queue.push(txn.new_arrival_s, EventKind.BLOCK_ARRIVAL, jid)
            if dst.time_ledger is not None:
                import bisect
                from .time_contract import TruckTimes
                tl = dst.time_ledger
                tl.records = dict(tl.records)
                tl.records[jid] = TruckTimes(gate_in=j.actual_gate_in)
                tl._a_sorted = list(tl._a_sorted)
                i = bisect.bisect_left(tl._a_sorted, j.actual_gate_in)
                tl._a_sorted.insert(i, j.actual_gate_in)
                if i < tl._a_idx:                # 검증 major-3: 이미 소비된 구간에 삽입되면
                    tl._a_idx += 1               # 포인터·카운터를 함께 밀어 이중소비 방지
                    tl._n_inside += 1
            rec = self.ledger.records[jid]                    # 전역 장부: owner/version 만 갱신
            rec.owner = txn.dst
            rec.version += 1
            rec.transfer_count += 1
            rec.transfer_history = rec.transfer_history + ((txn.src, txn.dst, self.now),)
            self.route_cost_s += txn.route_s                  # 검증 critical-2: 추가주행 계상
        except BaseException:
            self._restore(txn, snap)
            raise
        self._release(txn)                                    # 예약 → 실도착 대기로 승격

    def rollback(self, txn: TransferTxn) -> None:
        self._release(txn)

    def _release(self, txn: TransferTxn) -> None:
        """예약 해제 — **멱등**(이중 호출이 남의 예약을 훔치지 않게 txn_id 추적)."""
        if txn.txn_id in self._open_txn:
            self._open_txn.discard(txn.txn_id)
            self._reserved_inbound[txn.dst] = max(0, self._reserved_inbound[txn.dst] - 1)

    def try_transfer(self, job_id: str, dst: str, *, route_s: float,
                     travel_s: float) -> bool:
        """prepare→validate→commit, 어느 단계 실패든 rollback 후 KEEP (원자성).

        검증 major-5: TransferError 뿐 아니라 **모든 예외**에서 예약을 해제한 뒤 재발생
        시킨다(비TransferError 가 새어나가 예약이 영구 누수되던 경로 차단).
        """
        try:
            txn = self.prepare_transfer(job_id, dst, route_s=route_s, travel_s=travel_s)
        except TransferError:
            return False
        try:
            self.commit(txn)
            return True
        except TransferError:
            self.rollback(txn)
            return False
        except Exception:
            self.rollback(txn)
            raise

    # -------------------------------------------------- 불변식 (G0)
    def check_invariants(self) -> None:
        seen: dict[str, str] = {}
        for bid, sim in self.blocks.items():
            for jid in sim.jobs:
                if jid in seen:
                    raise AssertionError(f"이중 소유 {jid}: {seen[jid]}·{bid}")
                seen[jid] = bid
        for jid, rec in self.ledger.records.items():
            if jid not in seen:
                raise AssertionError(f"소유자 없음 {jid}")
            if seen[jid] != rec.owner:
                raise AssertionError(f"owner 불일치 {jid}: 장부 {rec.owner}·실제 {seen[jid]}")
        if len(seen) != len(self.ledger.records):
            raise AssertionError("보존 위반 — 작업 수 불일치")


def _namespace_jobs(sim, prefix: str) -> None:
    """구성 시 1회 — job_id 를 `{block}:{id}` 로 통일 (터미널 전역 유일·이후 불변).

    sim.jobs·큐 payload·블록 장부·ETA wake 를 함께 개명한다. 시계 0·미실행 시점 전제.
    """
    if getattr(sim, "_mb_namespaced", False):
        return
    old = dict(sim.jobs)
    sim.jobs.clear()
    for jid, j in old.items():
        nid = f"{prefix}:{jid}"
        j.job_id = nid
        sim.jobs[nid] = j
    sim.queue._heap = [replace(e, payload=f"{prefix}:{e.payload}")
                       if e.payload in old else e for e in sim.queue._heap]
    heapq.heapify(sim.queue._heap)
    if getattr(sim, "time_ledger", None) is not None:
        sim.time_ledger.records = {f"{prefix}:{k}": v
                                   for k, v in sim.time_ledger.records.items()}
    sim._eta_wakes = [(t, f"{prefix}:{jid}") for t, jid in sim._eta_wakes]
    sim._mb_namespaced = True
