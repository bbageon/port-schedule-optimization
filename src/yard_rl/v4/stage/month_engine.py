"""30일 무대 전용 **엔진 확장** — 사본을 안 고치고 계약 둘을 더한다.

사용자 지시 2026-08-26: *"엔진 계약은 v3 폴더에 새로운 엔진으로 추가하고 그대로 간다."*
그래서 `v3/world/integrated/` 사본은 **한 줄도 안 고치고**, 필요한 것만 여기서 잇는다.

■ 왜 두 가지가 더 필요한가

  ① **루프 상한** — 사본은 `run()` 안에 `guard > 2_000_000` 을 박아 두었다.
     하루짜리 무대에서는 여유가 크지만 30일은 30배를 돈다. 상한만 올린
     `MonthTerminal.run()` 을 둔다 (**나머지 줄은 사본과 같다**).

  ② ★**본선 실시간 투입** — 이게 진짜 이유다.
     사본의 `_validate` 는 *"모든 job 의 대상 컨테이너가 t=0 야드에 있어야 한다"*
     를 강제한다. 하루면 맞지만 30일은 못 맞춘다:

         초기 적재  21블록 약 19,700상자
         하루 소모  본선 적하 ~3,200 + 반출 트럭 ~2,500 = **~5,700/일**

     ⇒ 초기 적재는 **사흘 반**이면 바닥난다. 5일차 본선이 실어 갈 상자는
       **3일차에 들어온 상자**여야 하는데, 그 상자는 t=0 에 존재하지 않으므로
       무대를 세우는 시점에는 대상으로 지정할 수 없다.

     그래서 배를 **그날 아침에 붙인다** — `admit_external_job` 이 트럭에 대해
     이미 하는 일과 같은 계약이다(검사 먼저·통과하면 원자적으로 수술).
     적하 대상은 **그 순간 야드에 실제로 있는 상자** 중에서 고른다.

■ 계약 (사본 `admit_external_job` 과 같게 맞춘다)
  · 검사 단계에서 실패하면 **아무것도 안 바꾼다** (fail-closed)
  · 투입은 **review epoch** 에서만 — 블록 시계가 서로 앞서 있는 동안은 금지
  · 조용히 건너뛰지 않는다 — 못 넣은 배는 호출부가 기록으로 남긴다
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..world.domain.enums import ContainerSize, JobFlow, LoadStatus, ServiceMode
from ..world.domain.models import Job
from ..world.integrated.events import EventKind
from ..world.integrated.multiblock import (JobRecord, MultiBlockTerminal,
                                           ReviewEpoch, TerminalDecision,
                                           TransferError)
from ..world.integrated.scenario_gen import phys_min_completion_s
from ..world.integrated.vessel import (VesselPlan, VesselProcess,
                                       VesselWorkType)

#: 계획 완료시각 여유 배수 — 사본 생성기(`scenario_gen`)의 기본값과 같게 둔다.
#:
#: ★이 값이 본선 신호를 죽인다 ([[YR-248]] 0단계 · 2026-09-06)
#:   `slack = 계획완료 − 지금 − 남은작업` 에 `계획완료 = 시작 + M·c·d` 를 넣으면
#:   진척이 **소거되어 `slack = M·c·(d − 1)`** — 진척·시각과 무관한 상수가 된다.
#:   `d = 2.0` 이면 slack 이 그 배의 **총 작업시간 전체**(4.4~26.2h)라 특징의
#:   `±2h` 클램프에 전부 붙는다. 그래서 정책은 어느 배가 급한지 못 본다.
#:
#:   `d = 1.15` 로 내리면 `slack = M·c·0.15` 라 계획보다 15%만 밀려도 신호가 산다.
#:   `d ≥ 1.0` 이어야 한다 — 그 아래는 야드가 무한히 빨라도 달성 불가
#:   (`scenario_gen.py` 머리말 · `phys_min_completion_s`).
VESSEL_DEADLINE_MULT = 2.0
#: 40ft 비율 — 사본 `TerminalStreamParams.size_mix_ft40` 기본값과 같다.
SIZE_MIX_FT40 = 0.6


class MonthTerminal(MultiBlockTerminal):
    """30일을 도는 터미널 — 사본과 **루프 상한만** 다르다.

    아래 `run()` 은 사본 `MultiBlockTerminal.run()` 을 **그대로 옮긴 것**이고
    바뀐 곳은 `guard` 상한 한 줄뿐이다. 사본을 고칠 수 없으니 여기서 덮는다
    (`tests/v4/test_month_engine.py` 가 사본과 대조해 표류를 막는다).
    """

    #: 30일 x 21블록. 하루짜리 상한(200만)의 30배에 여유를 더 얹는다.
    LOOP_GUARD = 90_000_000

    def run(self, policy_fn, review_fn=None, cost_fn=None) -> dict:
        totals = {b: 0.0 for b in self.blocks}
        last = {b: s.now for b, s in self.blocks.items()}
        for s in self.blocks.values():
            s.cost.cut()
        guard = 0
        while len(self._terminal) < len(self.blocks):
            guard += 1
            if guard > self.LOOP_GUARD:          # 여기 한 줄만 다르다
                raise RuntimeError("multiblock 루프 상한 — 엔진 계약 위반 의심")
            live = [b for b in self.blocks if b not in self._terminal]
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
                self._sync_locks(sim)
                self._parked[bid] = out.time
            elif isinstance(out, TerminalDecision):
                policy_fn(sim, out)
        self.ledger.harvest(self.blocks)
        return {"totals": totals, "route_cost_s": self.route_cost_s,
                "terminal_total": round(sum(totals.values()), 6),
                "end": max(s.end for s in self.blocks.values())}


# ---------------------------------------------------- 본선 실시간 투입
@dataclass(frozen=True)
class VesselAdmission:
    """투입 결과 한 줄 — 무엇을 몇 개 넣었고 왜 줄었는지."""

    vessel_key: str
    block: str
    work: str
    asked_moves: int
    moves: int                     # 실제로 실은 물량 (적하는 재고 한도로 깎일 수 있다)
    start_s: float
    planned_completion_s: float
    reason: str = ""

    @property
    def clipped(self) -> bool:
        return self.moves < self.asked_moves


def free_targets(sim, *, limit: int, seed: str) -> list[str]:
    """적하가 실어 갈 수 있는 상자 — **지금 야드에 있고 아무도 안 찍은 것**.

    이미 다른 job 의 `target_container` 인 상자는 뺀다. 두 job 이 한 상자를 두고
    다투면 뒤에 오는 쪽이 사라진 상자를 찾는다.

    고르는 순서는 **시드 고정 섞기**다 — 위에서부터 고르면 접근 쉬운 상자만 나가고
    (재조작 0) 야드 상단이 계속 비는 인공적인 패턴이 생긴다.
    """
    taken = {j.target_container for j in sim.jobs.values()
             if j.target_container is not None}
    cand = [c for c in sorted(sim.stacks.containers) if c not in taken]
    random.Random(seed).shuffle(cand)
    return cand[:limit]


def inject_vessel(mbt: MultiBlockTerminal, bid: str, row: dict, *,
                  key: str, size_seed: str,
                  deadline_mult: float = VESSEL_DEADLINE_MULT) -> VesselAdmission:
    """배 한 척(정확히는 STS 스트림 하나)을 **런 중에** 블록에 붙인다.

    사본 `_seed_events` 가 t=0 에 하는 일과 같은 것을 시각 `start_s` 에 한다:
    `VesselProcess` 를 등록하고 `VESSEL_START` 를 예약하고, 연계 야드 job 을 넣는다.

    ■ 양하와 적하가 다르다 (사본 `scenario_gen` 본선 절과 같은 규칙)
      · 양하(DISCHARGE) — 배가 상자를 **내린다**. 야드 재고를 안 쓴다. job 해제는
        시각이 아니라 **박스의 물리 도착**이라 `JOB_RELEASED` 를 안 건다.
      · 적하(LOAD) — 배가 상자를 **싣는다**. 야드에 있는 상자를 찍어야 하고,
        재고가 모자라면 물량이 깎인다(`clipped`).

    실패하면 `TransferError` 를 던지고 **아무것도 안 바꾼다.**
    """
    if bid not in mbt.blocks:
        raise TransferError(f"{key}: 블록 없음 {bid}")
    sim = mbt.blocks[bid]
    if key in sim.vessels:
        raise TransferError(f"{key}: 이미 붙어 있는 배")
    start = float(row["start_s"])
    if start < sim.clock - 1e-9:
        raise TransferError(f"{key}: 과거에 붙일 수 없다 start={start:.1f} "
                            f"clock={sim.clock:.1f}")
    if start > sim.end:
        raise TransferError(f"{key}: 창 밖 start={start:.1f} end={sim.end:.1f}")

    work = (VesselWorkType.DISCHARGE if row["work"] == "DISCHARGE"
            else VesselWorkType.LOAD)
    cadence = float(row["cadence_s"])
    asked = int(row["moves"])
    targets: list[str] = []
    reason = ""
    if work == VesselWorkType.LOAD:
        targets = free_targets(sim, limit=asked, seed=f"{size_seed}:tgt")
        if len(targets) < asked:
            reason = f"재고 부족 {len(targets)}/{asked}"
    moves = asked if work == VesselWorkType.DISCHARGE else len(targets)
    if moves <= 0:
        raise TransferError(f"{key}: 실을 물량이 0 — {reason or '물량 0'}")

    # -- 계획 시각 — 사본 생성기와 같은 식 (본선 절)
    pc = start + moves * cadence * deadline_mult
    etd = start + moves * cadence * (deadline_mult + 1.0)
    tgt_c = ([sim.stacks.containers[t] for t in targets]
             if work == VesselWorkType.LOAD else None)
    phys_min = phys_min_completion_s(sim.profile, work=work, start_s=start,
                                     moves=moves, cadence_s=cadence,
                                     load_targets=tgt_c)
    if phys_min > pc:
        pc = phys_min
        etd = pc + moves * cadence
    plan = VesselPlan(planned_start_s=start, planned_completion_s=pc,
                      completion_basis=None, etd_s=etd, total_moves=moves,
                      sts_move_interval_s=cadence,
                      phys_min_completion_s=phys_min)

    # -- 여기부터 실패하지 않는 연산만 (원자성 — 사본 admit_external_job 과 같은 계약)
    sim.vessels[key] = VesselProcess(key, work, plan)
    sim.queue.push(start, EventKind.VESSEL_START, key)
    flow = (JobFlow.VESSEL_DISCHARGE if work == VesselWorkType.DISCHARGE
            else JobFlow.VESSEL_LOAD)
    rng = random.Random(f"{size_seed}:size")
    for m in range(moves):
        jid = f"{bid}:J-{key}-{m:04d}"       # 블록 접두 — `_namespace_jobs` 와 같은 꼴
        if work == VesselWorkType.DISCHARGE:
            j = Job(job_id=jid, flow=flow, release_time=start + m * cadence,
                    actual_gate_in=None, actual_block_arrival=None,
                    target_container=None,
                    inbound_size=(ContainerSize.FT40 if rng.random() < SIZE_MIX_FT40
                                  else ContainerSize.FT20),
                    inbound_load=LoadStatus.FULL,
                    deadline=pc + 1800.0, priority_class=1, vessel_id=key)
        else:
            j = Job(job_id=jid, flow=flow, release_time=start + m * cadence,
                    actual_gate_in=None, actual_block_arrival=None,
                    target_container=targets[m],
                    deadline=pc + 1800.0, priority_class=1, vessel_id=key)
        sim.jobs[jid] = j
        # 양하는 **박스 물리 도착**이 해제한다 — 시각으로 안 푼다 (사본 `_seed_events`)
        if not (j.is_vessel_linked and j.service_mode == ServiceMode.STORE):
            sim.queue.push(j.release_time, EventKind.JOB_RELEASED, jid)
        mbt.ledger.register(JobRecord(job_id=jid, origin_block=bid, owner=bid,
                                      flow=j.flow.value, a_gate_in=None))
    sim._refresh_rates()                     # STS 대기 요율에 새 배가 잡히게
    return VesselAdmission(vessel_key=key, block=bid, work=work.value,
                           asked_moves=asked, moves=moves, start_s=start,
                           planned_completion_s=pc, reason=reason)
