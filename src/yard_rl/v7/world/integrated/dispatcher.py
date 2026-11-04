"""참조 디스패처 — 결정적 규칙 하네스 (YR-036).

RL 정책이 아니다. 시뮬레이터를 완주시키고 결정론을 검증하기 위한 테스트용 규칙이다.
진짜 공동 최적화(중앙 resolver)는 YR-037. 선택은 완전순서 key 로 결정론.
"""
from __future__ import annotations

from ..contract.schema import CandidateKind
from .engine import CraneAssignment
from .jobplan import JobRef


class ReferenceDispatcher:
    def select(self, sim, crane_id: str, cands: list[JobRef]) -> JobRef:
        """본선 우선 → 최장 트럭대기 → job_id 오름차순 (완전순서 tie-break)."""
        return min(cands, key=lambda j: (
            0 if j.is_vessel else 1, -sim.cum_wait(j.job_id), j.job_id))

    def run(self, sim) -> None:
        """계약 record 없이 순수 시뮬레이터 완주 (invariant·golden 용)."""
        while True:
            dp = sim.run_until_decision()
            if dp is None:
                return
            for cid in dp.crane_ids:                     # 정렬됨 — 앞 크레인 예약 순차 전파
                cands = sim.candidates_for(cid)          # live (앞 배정 반영)
                if not cands:
                    sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))
                else:
                    sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE,
                                                    job_ref=self.select(sim, cid, cands)))
            sim.close_decision()
