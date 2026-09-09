"""하루 공개 예약 장부 (YR-171-A 정보 계약) — **명단은 공개, 실현은 비공개**.

■ 왜 필요한가 (실측 근거: yr171_horizon_probe)
현 계약은 트럭을 **도착 30분 전**에만 통지한다(`ANNOUNCE_LEAD_S` = `SLOT_S` = 1,800초).
그래서 48칸 계획표에서 미래 칸 중 트럭이 보이는 칸이 **0.4~5.6%** 뿐이고, 어느 시각·
어느 블록에서도 **1칸을 넘지 않는다**. 그런데 실제로는 하루 48칸 전부에 트럭이 온다.

이 상태로 시간 좌표를 48칸으로 열면 정책은 **"먼 슬롯일수록 한가하다"** 를 배운다 —
정말 한가한 것이 아니라 **아직 통지되지 않은 것**인데도. 최적화가 아니라 계측 결함이다.

■ 무엇을 공개하는가 (경계 — 이 모듈의 존재 조건)
공개하는 것은 **예약 장부**다. 트럭이 미리 오는 것이 아니다.

  공개 O  예약 gate-in 시각(재예약 반영) · 블록 · 반입/반출 · 규격 · 계약 주행 기대값
  공개 X  실현 진입(`actual_gate_in`) · 실현 블록도착(`actual_block_arrival`) ·
          서비스 시작/완료 · 재취급 수 · 그 밖의 엔진 실현값

**이 모듈은 엔진(sim·job·ledger)을 한 번도 읽지 않는다.** 자기 안의 예약 사본만 읽는다.
그래서 나중에 "기사가 예약보다 늦게 온다"(예약 준수오차)를 도입해도 **구조적으로**
누출이 생기지 않는다 — 실현값에 접근할 경로 자체가 없다. 현 개방 루프에서는 예약과
실현이 우연히 같지만, 그 우연에 기대지 않는다(`time_sell.notified_gate_in` 의 교훈).

■ 버전과 만료
재예약이 확정될 때마다 `plan_version` 이 오른다. 견적은 자기가 본 버전을 들고 다니고,
버전이 다르면 **만료**로 처리한다. Resolver 만 24시간을 보고 블록은 30분을 보는
비대칭, 또는 서로 다른 버전 견적을 한 matching 에 섞는 것은 하드 실패다(YR-171 명세).

■ 하지 않는 것
· 날짜를 넘기는 재예약(하루 격자 밖) — 거부한다.
· 예약을 엔진 상태로부터 역산하는 것 — 그 순간 이 모듈의 보증이 깨진다.
"""
from __future__ import annotations

from .cost_curve_v2 import GATE_BLOCK_MEAN_S
from .slot_plan import DAY_S, N_SLOTS, SLOT_S

_TRUCK_FLOWS = ("GATE_IN", "GATE_OUT")


class DayPlanError(RuntimeError):
    """예약 장부 계약 위반 — fail-closed(조용히 넘어가지 않는다)."""


class DayPlan:
    """같은 작업일 24시간 공개 예약 장부. 엔진을 읽지 않는다.

    `from_schedule` 로 하루 명단에서 만들고, 재예약이 확정될 때만 `reschedule` 로
    갱신한다. 계획표(`slot_plan`)와 견적은 이 장부만 본다.
    """

    __slots__ = ("_appt", "_gate_in", "_block", "_flow", "_version", "_reschedules")

    def __init__(self):
        self._appt: dict[str, float] = {}        # 최초 예약 (불변 — 기사 외부대기 원점)
        self._gate_in: dict[str, float] = {}     # 현재 예약 (재예약 반영)
        self._block: dict[str, str] = {}
        self._flow: dict[str, str] = {}
        self._version: int = 0
        self._reschedules: list[dict] = []

    # ---------------------------------------------------------------- 생성
    @classmethod
    def from_schedule(cls, schedule: list[dict]) -> "DayPlan":
        """하루 도착 명단(공개 예약)에서 장부를 만든다.

        `schedule` 은 `build_diurnal` 이 사전 확정한 명단이며 **예약 정보**다
        (`arrival_s` = 예약 gate-in). 실현값 필드는 애초에 들어 있지 않다.
        """
        p = cls()
        for e in schedule:
            jid = e["job_id"]
            if jid in p._appt:
                raise DayPlanError(f"중복 예약 job_id: {jid}")
            t = float(e["arrival_s"])
            p._appt[jid] = t
            p._gate_in[jid] = t
            p._block[jid] = e["block"]
            p._flow[jid] = e["flow"]
        return p

    # ---------------------------------------------------------------- 조회
    @property
    def plan_version(self) -> int:
        return self._version

    @property
    def n_jobs(self) -> int:
        return len(self._gate_in)

    def gate_in(self, job_id: str) -> float | None:
        """현재 예약 gate-in 시각 (재예약 반영). 없는 작업이면 None."""
        v = self._gate_in.get(job_id)
        return None if v is None else float(v)

    def appointment(self, job_id: str) -> float | None:
        """**최초** 예약 — 기사 외부 대기의 원점. 재예약해도 바뀌지 않는다."""
        v = self._appt.get(job_id)
        return None if v is None else float(v)

    def block_eta(self, job_id: str) -> float | None:
        """공개 예측 블록도착 = 현재 예약 + 계약 기대 주행(중심값).

        실현 주행이 아니라 **계약 중심값**을 쓴다 — `slot_plan.public_block_eta` 의
        마지막 분기와 같은 규칙이라 통지 전/후에 값이 튀지 않는다.
        """
        gi = self.gate_in(job_id)
        return None if gi is None else gi + GATE_BLOCK_MEAN_S

    # ---------------------------------------------------------------- 재예약
    def reschedule(self, job_id: str, new_gate_in_s: float, *,
                   t: float | None = None) -> int:
        """재예약 확정 1건 반영 → 새 `plan_version` 반환.

        하루 격자를 벗어나는 재예약은 거부한다(YR-171 명세: 날짜 변경 금지).
        """
        if job_id not in self._gate_in:
            raise DayPlanError(f"장부에 없는 작업의 재예약: {job_id}")
        new = float(new_gate_in_s)
        if not (0.0 <= new < DAY_S):
            raise DayPlanError(
                f"하루 격자 밖 재예약 거부 — {job_id} → {new}s (0~{DAY_S}s)")
        old = self._gate_in[job_id]
        self._gate_in[job_id] = new
        self._version += 1
        self._reschedules.append({"job_id": job_id, "t": t, "from_s": old,
                                  "to_s": new, "version": self._version})
        return self._version

    @property
    def reschedules(self) -> list[dict]:
        """재예약 이력 사본 — 감사용."""
        return list(self._reschedules)

    # ---------------------------------------------------------------- 계획표 원료
    def slot_hist(self) -> dict[str, dict[str, list[float]]]:
        """블록별 48칸 반입/반출 예약 히스토그램 — `slot_plan` 특징 1·2의 원료.

        칸은 **공개 예측 블록도착** 기준이다(그 시각에 블록이 일을 받는다). 하루 격자
        밖으로 나가는 예약(배수 구간)은 어느 칸에도 얹지 않는다 — `slot_plan._day_slot`
        과 같은 규칙이다.
        """
        acc: dict[str, dict[str, list[float]]] = {}
        for jid, bid in self._block.items():
            flow = self._flow[jid]
            if flow not in _TRUCK_FLOWS:
                continue
            eta = self.block_eta(jid)
            if eta is None or eta < 0.0 or eta >= DAY_S:
                continue
            d = acc.get(bid)
            if d is None:
                d = acc[bid] = {"in": [0.0] * N_SLOTS, "out": [0.0] * N_SLOTS}
            d["in" if flow == "GATE_IN" else "out"][int(eta // SLOT_S)] += 1.0
        return acc

    # ---------------------------------------------------------------- 견적 만료
    def stamp(self) -> int:
        """지금 버전을 찍는다 — 견적이 들고 다닐 표식."""
        return self._version

    def check_fresh(self, stamped: int, *, where: str = "quote") -> None:
        """찍어둔 버전이 아직 유효한지 검사 — 다르면 **하드 실패**(만료).

        서로 다른 버전의 견적을 한 matching 에 섞으면 "받아준 줄 알았는데 그 사이
        자리가 없어진" 배정이 나온다. 조용히 넘어가면 원인 추적이 불가능하다.
        """
        if stamped != self._version:
            raise DayPlanError(
                f"{where} 견적 만료 — 찍은 버전 {stamped} ≠ 현재 {self._version}")


def attach(mbt, schedule: list[dict]) -> DayPlan:
    """터미널에 공개 장부를 붙인다. 이미 있으면 하드 실패(이중 부착 금지)."""
    if getattr(mbt, "day_plan", None) is not None:
        raise DayPlanError("day_plan 이 이미 붙어 있다 — 이중 부착")
    plan = DayPlan.from_schedule(schedule)
    mbt.day_plan = plan
    return plan


def get(mbt) -> DayPlan | None:
    """붙어 있으면 장부, 아니면 None (구 계약 = 30분 통지만)."""
    return getattr(mbt, "day_plan", None)
