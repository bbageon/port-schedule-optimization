"""비용 인과 ledger — 각 raw 증분의 귀속 (YR-038, 최종전략 §10).

13항 raw 는 예외 없이 CostAccumulator.accrue() 단일 경로를 통과(rate 항은 advance→accrue)한다.
ledger 를 accrue 안에서만 기록하면 **Σledger == episode_raw 가 구성상 성립**하고 항목 중복계상 0 이
자동 보장된다. scale/weight/λ 는 담지 않는다 — raw 물리단위 전용 side-channel(민감도는 재시뮬 0 후처리).
guardrail 분리: accrue 되는 13 cost 항만 담고 안전/mandatory(mask, YR-037)는 원천 부재.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..contract.schema import COST_TERMS


class CostCause(str, Enum):
    WAIT_INTEGRAL = "WAIT_INTEGRAL"     # 트럭/장기대기 시간적분
    DISPATCH = "DISPATCH"               # 배정시 이동·재조작 임펄스
    STS_BLOCK = "STS_BLOCK"             # 본선 STS 대기
    TRANSFER_QUEUE = "TRANSFER_QUEUE"   # 이송장비 대기
    LANE = "LANE"                       # 레인 혼잡
    INTERFERENCE = "INTERFERENCE"       # 크레인 상호대기
    IMBALANCE = "IMBALANCE"             # 부하 불균형
    VESSEL_FINISH = "VESSEL_FINISH"     # 본선 완료시 지연
    CLEAROUT = "CLEAROUT"               # 종료시 미완 본선 지연
    RESEQUENCE = "RESEQUENCE"           # 순번 변경 (v1 생산자 부재 — inactive)


# rate 항 → cause (advance 가 부여)
RATE_CAUSE: dict[str, CostCause] = {
    "sts_wait": CostCause.STS_BLOCK, "transfer_wait": CostCause.TRANSFER_QUEUE,
    "lane_cong": CostCause.LANE, "interference": CostCause.INTERFERENCE,
    "imbalance": CostCause.IMBALANCE}

# 항별 허용 cause 화이트리스트 — accrue 가 허용밖 cause 를 거부
TERM_CAUSES: dict[str, frozenset] = {
    "truck_wait": frozenset({CostCause.WAIT_INTEGRAL}),
    "long_wait": frozenset({CostCause.WAIT_INTEGRAL}),
    "crane_travel": frozenset({CostCause.DISPATCH}),
    "empty_travel": frozenset({CostCause.DISPATCH}),
    "rehandle": frozenset({CostCause.DISPATCH}),
    "sts_wait": frozenset({CostCause.STS_BLOCK}),
    "transfer_wait": frozenset({CostCause.TRANSFER_QUEUE}),
    "vessel_delay": frozenset({CostCause.VESSEL_FINISH, CostCause.CLEAROUT}),  # 유일한 2-cause
    "depart_delay": frozenset({CostCause.VESSEL_FINISH}),
    "lane_cong": frozenset({CostCause.LANE}),
    "interference": frozenset({CostCause.INTERFERENCE}),
    "resequence": frozenset({CostCause.RESEQUENCE}),
    "imbalance": frozenset({CostCause.IMBALANCE}),
}


@dataclass(frozen=True)
class LedgerEntry:
    interval: int
    term: str
    cause: CostCause
    subject: str | None
    amount: float


@dataclass
class CostLedger:
    sealed: list = field(default_factory=list)      # list[tuple[LedgerEntry,...]] — cut 경계별
    _pending: list = field(default_factory=list)

    def record(self, term: str, cause: CostCause, subject, amount: float) -> None:
        if amount == 0.0:
            return                                   # 항등식 불변·size 절약
        self._pending.append(LedgerEntry(len(self.sealed), term, cause, subject, float(amount)))

    def seal(self) -> None:
        self.sealed.append(tuple(self._pending))     # raw 파티션(cut)과 동일 경계
        self._pending = []

    def all_entries(self) -> list:
        out = list(self._pending)
        for seg in self.sealed:
            out.extend(seg)
        return out

    def term_totals(self) -> dict[str, float]:
        tot = {t: 0.0 for t in COST_TERMS}
        for e in self.all_entries():
            tot[e.term] += e.amount
        return tot

    def interval_term_totals(self, k: int) -> dict[str, float]:
        seg = self.sealed[k] if k < len(self.sealed) else tuple(self._pending)
        tot = {t: 0.0 for t in COST_TERMS}
        for e in seg:
            tot[e.term] += e.amount
        return tot

    def by_term_cause(self) -> dict:
        out: dict = {}
        for e in self.all_entries():
            out.setdefault((e.term, e.cause.value), 0.0)
            out[(e.term, e.cause.value)] += e.amount
        return out

    def by_subject(self, term: str) -> dict:
        out: dict = {}
        for e in self.all_entries():
            if e.term == term and e.subject is not None:
                out.setdefault(e.subject, 0.0)
                out[e.subject] += e.amount
        return out

    def inactive_terms(self) -> tuple:
        t = self.term_totals()
        return tuple(k for k in COST_TERMS if t[k] == 0.0)

    def digest(self) -> str:
        import hashlib
        rows = sorted((f"{t}:{c}:{round(v, 6)}") for (t, c), v in self.by_term_cause().items())
        return hashlib.sha1("|".join(rows).encode("utf-8")).hexdigest()[:16]


def assert_ledger_identity(acc, *, tol: float = 1e-6) -> None:
    """(1) Σledger[t] == episode_raw[t] ∀13 (2) 항폐쇄(term∈COST_TERMS) (3) cause 화이트리스트."""
    led = acc.ledger
    if led is None:
        raise ValueError("ledger 비활성 — enable_cost_ledger=True 로 시뮬 필요")
    ep = acc.episode_raw()
    lt = led.term_totals()
    for t in COST_TERMS:
        if abs(lt[t] - ep[t]) > tol:
            raise AssertionError(f"{t}: Σledger {lt[t]} != episode_raw {ep[t]}")
    for e in led.all_entries():
        if e.term not in COST_TERMS:
            raise AssertionError(f"비-COST_TERMS 항 혼입 {e.term} (guardrail 위반)")
        if e.cause not in TERM_CAUSES[e.term]:
            raise AssertionError(f"{e.term}: 허용밖 cause {e.cause}")


def build_ledger_report(acc, *, breakdowns=None) -> dict:
    """cost-ledger-report-v1 (결정론 JSON) — identity·term_cause 행렬·inactive·subjects."""
    led = acc.ledger
    ep = acc.episode_raw()
    lt = led.term_totals()
    per_term = {t: {"raw": round(ep[t], 6), "ledger": round(lt[t], 6),
                    "residual": round(lt[t] - ep[t], 9)} for t in COST_TERMS}
    tc = {f"{t}|{c}": round(v, 6) for (t, c), v in sorted(led.by_term_cause().items())}
    subjects = {t: {s: round(v, 6) for s, v in sorted(led.by_subject(t).items())}
                for t in ("crane_travel", "empty_travel", "rehandle", "vessel_delay", "depart_delay")}
    return {"report_id": "cost-ledger-report-v1", "identity_per_term": per_term,
            "term_cause": tc, "inactive_terms": list(led.inactive_terms()),
            "subjects": subjects, "digest": led.digest()}
