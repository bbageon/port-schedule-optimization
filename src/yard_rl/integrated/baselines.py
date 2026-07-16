"""총비용용 baseline 군 + 행동분포 건전성 (YR-044).

YR-039 무효 사유 2 — baseline 이 "최단 서비스"가 아니라 "최단 **행동**"을 골라 결정의 54%(val 5-seed)~81%(test seed) 가
REPOSITION(순수 크레인 이동)인 퇴화 정책이었다. 비교 기준이 무너지면 어떤 승리 주장도 성립하지 않는다.

여기서 제공:
- `JointRolloutGreedy` — **1차 baseline**. 두 YC 의 공동 feasible 행동 조합을 열거해 **고정 시간창
  (horizon_s) 누적비용** argmin — 행동 후 base_policy 로 시간창 끝까지 진행 = 1-step 정책개선.
- `JointImmediateCostGreedy` — **진단 전용** (무효판정 §6.2 문자 그대로의 "즉시비용(다음 결정까지)
  argmin"). 짧은 행동일수록 rate 비용이 덜 쌓여 구성상 퇴화 — baseline 으로 쓰지 않는다 (매핑 §3).
- `BeamLookahead` — 강 baseline. 1차를 width W 로 가지치기해 시간창 2개까지 확장한 rolling-horizon.
- `ServiceFirstSPTPreference`·`FIFOPreference` — 보조 진단군 (동일 정보·후보·제약·비용 config).
- `ActionMix`·`assert_healthy_action_mix` — 퇴화 사전 검출 계약 (REPOSITION 지배 등).
전 정책은 동일 드라이버(`run_joint_episode`)·동일 후보 생성기·동일 resolver 제약을 쓴다.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

from ..contract.schema import CandidateKind
from .engine import CraneAssignment
from .resolver import BaselinePreference, CentralResolver


# ------------------------------------------------------------ 보조 진단군
class ServiceFirstSPTPreference(BaselinePreference):
    """실작업(SERVE) 우선 → 그 안에서 최단 소요. YR-039 진단 정책 (무효 판정 §3).

    퇴화 SPT(전 kind 에 duration 적용)와 달리 SERVE 를 먼저 소진한다.
    """

    def rank(self, sim, crane_id, gc) -> tuple:
        is_serve = gc.kind == CandidateKind.SERVE
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        return (0 if is_serve else 1, dur) + super().rank(sim, crane_id, gc)


class FIFOPreference(BaselinePreference):
    """실작업 우선 → 먼저 도착한 트럭부터 (선착순). 보조 진단군."""

    def rank(self, sim, crane_id, gc) -> tuple:
        ref = gc.job_ref
        if ref is None:
            return (2, 0.0, "")
        is_serve = gc.kind == CandidateKind.SERVE
        j = sim.jobs.get(ref.job_id)
        arrival = (j.actual_block_arrival if (j is not None and j.is_external_truck
                                              and j.actual_block_arrival is not None)
                   else float("inf"))
        return (0 if is_serve else 1, arrival, ref.job_id)


# ------------------------------------------------------------ 행동분포 건전성
@dataclass
class ActionMix:
    """결정별 선택 행동 분포 + '실작업이 가능했는데 안 골랐는가' 진단."""

    counts: dict[str, int] = field(default_factory=dict)
    serve_available: int = 0        # SERVE 가 feasible 했던 크레인-결정 수
    serve_taken: int = 0            # 그 중 실제 SERVE 를 고른 수

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def share(self, kind: str) -> float:
        return self.counts.get(kind, 0) / self.total if self.total else 0.0

    def serve_when_available(self) -> float:
        """SERVE 가능했을 때 실제로 SERVE 를 고른 비율 — 퇴화 판별의 핵심 지표."""
        return self.serve_taken / self.serve_available if self.serve_available else 1.0

    def record(self, kind: CandidateKind, serve_was_feasible: bool) -> None:
        self.counts[kind.value] = self.counts.get(kind.value, 0) + 1
        if serve_was_feasible:
            self.serve_available += 1
            if kind == CandidateKind.SERVE:
                self.serve_taken += 1

    def as_dict(self) -> dict:
        return {"counts": dict(sorted(self.counts.items())), "total": self.total,
                "shares": {k: round(self.share(k), 4) for k in sorted(self.counts)},
                "serve_available": self.serve_available, "serve_taken": self.serve_taken,
                "serve_when_available": round(self.serve_when_available(), 4)}


class ActionMixError(RuntimeError):
    """행동분포 퇴화 — baseline/정책으로 사용 금지."""


def assert_healthy_action_mix(mix: ActionMix, *, min_serve_when_available: float = 0.25,
                              max_nonserve_share: float = 0.60, label: str = "") -> None:
    """퇴화 **검출기** (성능 게이트 아님) — YR-044 계약.

    - `serve_when_available`: 실작업이 가능한데도 거의 안 고르면 퇴화.
    - `max_nonserve_share`: 단일 비-SERVE 행동이 전체를 장악하면 퇴화.

    문턱 보정 근거 (YR-044 실측, seed 310000): **퇴화 0.03~0.08** (YR-039 SPT·즉시비용 greedy —
    REPOSITION 50~59%·완료율 41%) vs **건전 0.46~0.57** (ServiceFirstSPT·FIFO·VesselWait·
    JointRollout). 두 무리가 뚜렷이 갈리므로 그 사이(0.25)를 문턱으로 둔다 — 좋은 정책이
    선제 위치조정을 섞는 것을 벌하지 않으면서 퇴화만 잡는다.
    """
    if mix.total == 0:
        return
    swa = mix.serve_when_available()
    if swa < min_serve_when_available:
        raise ActionMixError(
            f"{label} 퇴화: 실작업 가능 시 SERVE 선택 {swa:.1%} < {min_serve_when_available:.0%} "
            f"— {mix.as_dict()}")
    for kind, n in mix.counts.items():
        if kind == CandidateKind.SERVE.value:
            continue
        if n / mix.total > max_nonserve_share:
            raise ActionMixError(
                f"{label} 퇴화: {kind} 가 전체 결정의 {n / mix.total:.1%} 장악 "
                f"(> {max_nonserve_share:.0%}) — {mix.as_dict()}")


# ------------------------------------------------------------ joint 정책
def _wait_of(gen) -> object:
    return next(g for g in gen.items if g.kind == CandidateKind.WAIT)


def _feasible_joint(sim, assign) -> bool:
    """동일 오라클(dry_run_commit) + token 중복 검사로 공동 실행가능 판정."""
    toks = [g.job_ref.token for g in assign.values() if g.job_ref and g.job_ref.token]
    if len(toks) != len(set(toks)):
        return False
    proj = sim.dry_run_commit({c: g.job_ref for c, g in assign.items()})
    work = {c for c, g in assign.items() if g.job_ref is not None}
    return set(proj.plans) == work


def _apply(sim, assign) -> None:
    for c in sorted(assign):
        g = assign[c]
        if g.kind == CandidateKind.WAIT:
            sim.assign(c, CraneAssignment(c, CandidateKind.WAIT))
        else:
            sim.assign(c, CraneAssignment(c, g.kind, g.job_ref))
    sim.close_decision()


def _rollout_cost(sim, assign, rc, *, horizon_s: float = 0.0, base_policy=None,
                  generator=None) -> tuple[float, object]:
    """joint 행동의 평가비용 — deepcopy 후 **실제 엔진**으로 진행 (정확).

    horizon_s=0 : 다음 결정까지의 구간비용 = 문자 그대로의 "즉시비용".
      ⚠ 이 기준은 **짧은 행동을 체계적으로 우대**한다 — 구간이 짧으면 대기·혼잡 rate 비용이
      덜 쌓이기 때문. YR-039 의 퇴화 SPT("최단 행동")와 같은 함정에 비용으로 도달한다
      (YR-044 실측: REPOSITION 59%·SERVE 8%). 진단용으로만 남긴다.
    horizon_s>0: **고정 시간창** 누적비용 — 행동 후 base_policy 로 t0+horizon 까지 진행.
      모든 분기를 같은 시간축에서 비교하므로 길이 편향이 사라진다 (rolling-horizon rollout =
      base_policy 위의 1-step 정책개선).
    """
    from .adapter import _max_vessel_risk
    scratch = copy.deepcopy(sim)
    scratch.cost.cut()                       # 구간 리셋 → 이 행동 이후 증분만 계측
    t0 = scratch.now
    risk = _max_vessel_risk(scratch, t0)
    _apply(scratch, assign)
    total = 0.0
    tk = t0
    while True:
        dp = scratch.run_until_decision()
        raw = scratch.cost.cut()
        total += rc.cost_for(interval_start_s=tk, interval_end_s=scratch.now,
                             raw=raw, risk_max=risk).total_normalized
        tk = scratch.now
        if dp is None or horizon_s <= 0.0 or scratch.now - t0 >= horizon_s:
            break
        gen = generator or _default_gen()
        gb = {c: gen.generate(scratch, c, scratch.info_level) for c in dp.crane_ids}
        _apply(scratch, base_policy.decide(scratch, dp, gb))
    return total, scratch


def _default_gen():
    from .candidates import CandidateGenerator
    return CandidateGenerator()


class JointRolloutGreedy:
    """YR-044 1차 baseline — 공동 feasible 행동 조합 열거 → 동일 13항 비용 argmin.

    평가 기준은 **고정 시간창(horizon_s) 누적비용**이다. 무효 판정 §6.2 의 문자 그대로의
    "즉시비용(다음 결정까지) argmin"(horizon_s=0)은 **짧은 행동을 우대해 퇴화**함을 실측했으므로
    (REPOSITION 59%·SERVE 8%·대기 4.67분 — YR-039 SPT 와 동일 함정), 기본값을 고정 시간창으로 둔다.
    행동 후 base_policy 로 시간창 끝까지 진행 → base_policy 위의 1-step 정책개선(rollout algorithm).
    """

    name = "JOINT_ROLLOUT_GREEDY"

    def __init__(self, reward_calc, *, horizon_s: float = 600.0, base_policy=None,
                 max_combos: int = 64, generator=None):
        self.rc = reward_calc
        self.horizon_s = horizon_s
        self.base_policy = base_policy or ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
        self.max_combos = max_combos
        self.generator = generator

    def decide(self, sim, dp, gen_by) -> dict:
        best, best_key = None, None
        for combo in self._combos(dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            cost, _ = _rollout_cost(sim, assign, self.rc, horizon_s=self.horizon_s,
                                    base_policy=self.base_policy, generator=self.generator)
            # 완전순서 tie-break (결정론): 비용 → 후보 id 조합
            key = (round(cost, 9), tuple((c, assign[c].candidate_id) for c in dp.crane_ids))
            if best_key is None or key < best_key:
                best, best_key = assign, key
        return best or {c: _wait_of(gen_by[c]) for c in dp.crane_ids}

    def _combos(self, dp, gen_by):
        opts = [[g for g in gen_by[c].items if g.feasible] for c in dp.crane_ids]
        n = 1
        for o in opts:
            n *= max(1, len(o))
        if n > self.max_combos:              # 조합 폭발 방지 (절단은 보고 대상 — 조용한 축소 금지)
            per = max(1, int(self.max_combos ** (1 / max(1, len(opts)))))
            opts = [o[:per] for o in opts]
        return itertools.product(*opts)


class JointImmediateCostGreedy(JointRolloutGreedy):
    """§6.2 문자 그대로의 즉시비용(다음 결정까지) argmin — **진단 전용**.

    ⚠ 짧은 행동 우대 편향으로 퇴화한다 (YR-044 실측). baseline 으로 쓰지 말 것 —
    `assert_healthy_action_mix` 가 걸러낸다. 편향의 존재를 문서화·회귀로 고정하기 위해 보존.
    """

    name = "JOINT_IMMEDIATE_GREEDY_DIAG"

    def __init__(self, reward_calc, *, max_combos: int = 64):
        super().__init__(reward_calc, horizon_s=0.0, max_combos=max_combos)


class BeamLookahead(JointRolloutGreedy):
    """강 baseline — 고정 시간창 rollout 을 width W 로 가지치기해 2단 확장 (rolling-horizon).

    1단에서 상위 W 조합을 고른 뒤 각 분기를 base_policy 로 한 시간창 더 진행해 누적비교.
    deepcopy rollout 이라 W·horizon 에 비례해 비싸다 — 평가 전용.
    """

    name = "BEAM_LOOKAHEAD"

    def __init__(self, reward_calc, *, horizon_s: float = 600.0, width: int = 3,
                 base_policy=None, max_combos: int = 64, generator=None):
        super().__init__(reward_calc, horizon_s=horizon_s, base_policy=base_policy,
                         max_combos=max_combos, generator=generator)
        self.width = width

    def decide(self, sim, dp, gen_by) -> dict:
        scored = []
        for combo in self._combos(dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            cost, scratch = _rollout_cost(sim, assign, self.rc, horizon_s=self.horizon_s,
                                          base_policy=self.base_policy, generator=self.generator)
            scored.append((round(cost, 9),
                           tuple((c, assign[c].candidate_id) for c in dp.crane_ids),
                           assign, scratch))
        if not scored:
            return {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
        scored.sort(key=lambda x: (x[0], x[1]))
        best, best_key = None, None
        for cost, tie, assign, scratch in scored[:self.width]:
            total = cost + self._tail(scratch)          # 2번째 시간창
            key = (round(total, 9), tie)
            if best_key is None or key < best_key:
                best, best_key = assign, key
        return best

    def _tail(self, scratch) -> float:
        """분기 이후 한 시간창을 base_policy 로 더 진행한 누적비용."""
        from .adapter import _max_vessel_risk
        if scratch.terminal:
            return 0.0
        gen = self.generator or _default_gen()
        total, t0, tk = 0.0, scratch.now, scratch.now
        risk = _max_vessel_risk(scratch, t0)
        while scratch.now - t0 < self.horizon_s:
            dp = scratch.run_until_decision() if not scratch._pending else None
            raw = scratch.cost.cut()
            total += self.rc.cost_for(interval_start_s=tk, interval_end_s=scratch.now,
                                      raw=raw, risk_max=risk).total_normalized
            tk = scratch.now
            if dp is None:
                break
            gb = {c: gen.generate(scratch, c, scratch.info_level) for c in dp.crane_ids}
            _apply(scratch, self.base_policy.decide(scratch, dp, gb))
        return total


class ResolverPolicy:
    """Preference + CentralResolver 를 joint 정책 인터페이스로 — 동일 드라이버로 공정 비교."""

    def __init__(self, preference, name: str = "RESOLVER"):
        self.resolver = CentralResolver(preference)
        self.name = name

    def decide(self, sim, dp, gen_by) -> dict:
        resn = self.resolver.resolve(sim, dp, gen_by)
        out = {}
        for r in resn.resolutions:
            out[r.crane_id] = (_wait_of(gen_by[r.crane_id]) if r.chosen_candidate_id is None
                               else gen_by[r.crane_id].items[r.chosen_candidate_id])
        return out


# ------------------------------------------------------------ 공통 드라이버
def run_joint_episode(sim, policy, reward_calc, *, level=None, generator=None) -> dict:
    """전 정책 공통 드라이버 — 동일 정보·후보·제약·비용 config (YR-044 공정 비교 계약)."""
    from .candidates import CandidateGenerator
    gen = generator or CandidateGenerator()
    level = level or sim.info_level
    sim.info_level = level
    mix = ActionMix()
    total, n_dec = 0.0, 0
    dp = sim.run_until_decision()
    sim.cost.cut()                              # 첫 결정 이전 구간은 선행 행동 없음 — 폐기
    from .adapter import _max_vessel_risk
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, level) for c in dp.crane_ids}
        assign = policy.decide(sim, dp, gen_by)
        for c in dp.crane_ids:
            serve_ok = any(g.feasible and g.kind == CandidateKind.SERVE
                           for g in gen_by[c].items)
            mix.record(assign[c].kind, serve_ok)
        t_k, risk = dp.time, _max_vessel_risk(sim, dp.time)
        _apply(sim, assign)
        n_dec += 1
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        total += reward_calc.cost_for(interval_start_s=t_k, interval_end_s=sim.now,
                                      raw=raw, risk_max=risk).total_normalized
    jobs = list(sim.jobs.values())
    done = sum(1 for j in jobs if j.status.name == "DONE")
    waits = [w / 60.0 for w in sim.kpis.wait_samples_s]
    ws = sorted(waits)
    return {"policy": getattr(policy, "name", type(policy).__name__),
            "total_cost": total, "n_decisions": n_dec,
            "completion_rate": done / max(1, len(jobs)), "backlog": len(jobs) - done,
            "mean_wait_min": (sum(waits) / len(waits)) if waits else 0.0,
            "p95_wait_min": (ws[min(len(ws) - 1, int(0.95 * len(ws)))] if ws else 0.0),
            "vessel_delay_min": sim.kpis.vessel_delay_s / 60.0,
            "action_mix": mix.as_dict(), "_mix": mix}
