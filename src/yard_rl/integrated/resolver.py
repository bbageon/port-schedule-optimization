"""중앙 joint resolver — 결정적 baseline (YR-037, 최종전략 §8.6·§11.3).

순차 greedy 를 원칙적 결정적 baseline 으로 교체. 목적은 비용최소가 아니라 **결정적 feasibility +
mandatory 보호**. 알고리즘: mandatory-우선 완전순서 제약 그리디 + engine.dry_run_commit 오라클.
- 0-위반: 수용된 joint 는 dry_run 이 전원 동시 feasible 로 판정 → apply 의 실제 commit 도 무예외
  (D-ORACLE: dry_run.plans == commit plans). 그 위에 assign→reserve() 2차 방어선.
- 결정성: pair 완전순서(말미 (crane_id, token, candidate_id) 유일키) + dry_run 순수함수 → 동일 입력 동일 배정.
- 비용 가중치(YR-038)·Q값(YR-039)은 Preference 로 주입 — resolver 골격은 불변(joint action masking 재사용).
"""
from __future__ import annotations

from collections import defaultdict

from ..contract.schema import CandidateKind
from .audit import (CandidateVerdict, CraneResolution, JointResolution)
from .candidates import _KIND_RANK
from .engine import CraneAssignment


class BaselinePreference:
    """본선 우선 → 최장 트럭대기 → job_id. ReferenceDispatcher.select 와 동일 key."""

    def rank(self, sim, crane_id, gc) -> tuple:
        ref = gc.job_ref
        is_v = bool(ref and ref.is_vessel)
        cum = sim.cum_wait(ref.job_id) if (ref and ref.is_external) else 0.0
        return (0 if is_v else 1, -cum, ref.job_id if ref else "")


class DispatcherPreference(BaselinePreference):
    """레거시 ReferenceDispatcher 규칙 재사용 (record_episode 하위호환)."""

    def __init__(self, dispatcher=None):
        self.dispatcher = dispatcher


class CentralResolver:
    def __init__(self, preference=None):
        self.preference = preference or BaselinePreference()

    # ------------------------------------------------------------ resolve
    def resolve(self, sim, decision, gen_by_crane) -> JointResolution:
        cranes = decision.crane_ids                       # crane_id 정렬 (engine)
        pairs = [(c, gc) for c in cranes
                 for gc in gen_by_crane[c].items
                 if gc.feasible and gc.kind != CandidateKind.WAIT]
        pairs.sort(key=lambda cg: self._pair_key(sim, cg[0], cg[1]))
        chosen: dict = {}
        taken_token: set = set()
        rejects: dict = defaultdict(list)
        contested: dict = defaultdict(set)
        for (c, gc) in pairs:
            tok = gc.job_ref.token
            if tok is not None:
                contested[tok].add(c)
            if c in chosen:
                continue
            if tok is not None and tok in taken_token:
                rejects[c].append((gc, "DUP_JOB"))
                continue
            trial = {**chosen, c: gc}
            proj = sim.dry_run_commit({x: g.job_ref for x, g in trial.items()})
            if set(proj.plans) == set(trial):             # 전원 동시 feasible → 수용 (단조)
                chosen = trial
                if tok is not None:
                    taken_token.add(tok)
            else:
                rejects[c].append((gc, "JOINT_CONFLICT"))
        return self._finalize(sim, decision, gen_by_crane, chosen, rejects, contested)

    def _pair_key(self, sim, c, gc) -> tuple:
        return ((0 if gc.mandatory else 1,) + tuple(self.preference.rank(sim, c, gc))
                + (_KIND_RANK[gc.kind], c, gc.job_ref.token or "", gc.candidate_id))

    def _finalize(self, sim, decision, gen_by_crane, chosen, rejects, contested) -> JointResolution:
        resolutions = []
        for c in decision.crane_ids:
            gc = chosen.get(c)
            verdicts = tuple(sorted(
                (CandidateVerdict(c, g.candidate_id, g.job_ref.token if g.job_ref else None,
                                  g.kind, False, reason, g.mandatory)
                 for (g, reason) in rejects.get(c, [])),
                key=lambda v: (v.candidate_id, v.reason or "")))
            if gc is None:
                yield_reason = "NO_FEASIBLE" if not rejects.get(c) else "LOST_CONTENTION"
                resolutions.append(CraneResolution(c, CandidateKind.WAIT, None, None,
                                                   yield_reason, (), verdicts))
            else:
                resolutions.append(CraneResolution(
                    c, gc.kind, gc.candidate_id, gc.job_ref.token,
                    None, self._pair_key(sim, c, gc), verdicts))
        # 미수용 mandatory token (드롭 아님 — 다음 결정 재생성·최우선)
        chosen_tokens = {g.job_ref.token for g in chosen.values() if g.job_ref and g.job_ref.token}
        deferred = sorted({g.job_ref.token
                           for c in decision.crane_ids
                           for g in gen_by_crane[c].items
                           if g.mandatory and g.job_ref and g.job_ref.token
                           and g.job_ref.token not in chosen_tokens})
        contest = tuple(sorted((tok, tuple(sorted(cs))) for tok, cs in contested.items()
                               if len(cs) > 1))
        return JointResolution(decision.time, tuple(decision.crane_ids),
                               tuple(resolutions), tuple(deferred), contest)

    # ------------------------------------------------------------- apply
    def apply(self, sim, resolution: JointResolution, gen_by_crane) -> None:
        by_crane = {r.crane_id: r for r in resolution.resolutions}
        for c in sorted(resolution.crane_ids):        # crane_id 순 == commit·dry_run 순
            r = by_crane[c]
            if r.action == CandidateKind.WAIT:
                sim.assign(c, CraneAssignment(c, CandidateKind.WAIT))
            else:
                gc = gen_by_crane[c].items[r.chosen_candidate_id]
                sim.assign(c, CraneAssignment(c, gc.kind, gc.job_ref))   # 2차 reserve() backstop
        sim.close_decision()
        sim.resolution_log.append(resolution)
