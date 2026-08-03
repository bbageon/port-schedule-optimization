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
    """실작업 우선 → 먼저 도착한 트럭부터 (선착순). 보조 진단군.

    ⚠ **약한 미래정보 누출** (YR-107 발견, 2026-07-28): 정렬키가 `actual_block_arrival` 을
    `now` 게이트 없이 읽는다. PRE_ADVICE 에서는 **미도착** 트럭에 대한 후보(PRE_REHANDLE·
    REPOSITION)가 존재하므로 이 진단군도 진짜 도착순을 안다. 배포 후보 아님(진단 전용).
    거동을 고치면 과거 골든이 전부 바뀌므로 **표기만 하고 수정은 별건**으로 둔다.
    """

    USES_FUTURE_INFORMATION = True

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
    REPOSITION 50~59%·완료율 41%) vs **건전 0.46~0.57**. YR-048 ETA 주입 후에도 건전군은
    0.46~0.58로 유지돼 0.25 문턱은 유효하다. PRE_REHANDLE도 비-SERVE로 세므로 YR-045에서는
    해당 선택 횟수를 별도 보고한다 — 문턱은 성능 판정이 아니라 명백한 퇴화 검출용이다.
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
    # YR-142(18차 감사): 결속 PREPO 는 token=None 이라 위 검사를 건너뛴다 — 같은 결속
    # 작업을 두 크레인이 한 공동결정에서 동시에 고르는 조합은 의미상 중복이므로 제외.
    from .candidates import prepo_bound_jid
    bjids = [b for b in (prepo_bound_jid(g.job_ref.job_id) for g in assign.values()
                         if g.job_ref) if b is not None]
    if len(bjids) != len(set(bjids)):
        sim._prepo_dup_removed = getattr(sim, "_prepo_dup_removed", 0) + 1
        return False
    proj = sim.dry_run_commit({c: g.job_ref for c, g in assign.items()})
    work = {c for c, g in assign.items() if g.job_ref is not None}
    return set(proj.plans) == work


def _untriggered_defer(g) -> bool:
    """YR-147 C: 관측 trigger 없는 유한 DEFER — 전략 선택지에서 제외 대상 (구조 fallback 전용)."""
    return (g.kind == CandidateKind.WAIT
            and getattr(g, "defer_until", None) is not None
            and getattr(g, "defer_trigger", None) is None)


def _apply(sim, assign) -> None:
    for c in sorted(assign):
        g = assign[c]
        if g.kind == CandidateKind.WAIT:
            du = getattr(g, "defer_until", None)
            if du is not None:
                sim.schedule_defer_wake(du)      # YR-147: 유한 DEFER 재개방 예약
            sim.assign(c, CraneAssignment(c, CandidateKind.WAIT))
        else:
            sim.assign(c, CraneAssignment(c, g.kind, g.job_ref))
    sim.close_decision()


def _rollout_cost(sim, assign, rc, *, horizon_s: float = 0.0, base_policy=None,
                  generator=None, term_sink: dict | None = None) -> tuple[float, object]:
    """joint 행동의 평가비용 — deepcopy 후 **실제 엔진**으로 진행 (정확).

    ⚠ **오라클(미래정보 사용) — 배포 금지** (YR-107, 2026-07-28).
    deepcopy 한 사본의 이벤트 큐에는 엔진이 `actual_block_arrival`(트럭의 **진짜** 블록
    도착시각)로 밀어 넣은 BLOCK_ARRIVAL 이 그대로 남아 있다. 따라서 이 함수로 평가하는
    정책은 시간창 안의 트럭 도착을 **정확히 알고** 결정한다. 실측(high-tight/830700,
    PRE_ADVICE): 1800초 창의 트럭 9대를 전부 진짜 시각으로 실현, 같은 순간 정책이 볼 수
    있는 공개 ETA 는 평균 447초·최대 1234초 어긋나 있었다.
    공개정보만 쓰는 대응물은 `predictive_rollout.PredictiveRollout` 이다.
    이 함수를 쓰는 정책은 반드시 `USES_FUTURE_INFORMATION = True` 를 선언해야 한다.

    term_sink 가 주어지면 창 누적 **항별 기여**(CostBreakdown.contributions)를 그
    dict 에 합산한다 — 계층목적(YR-071 objectives.hierarchy_key) 평가용. None(기본)
    이면 기존 경로와 바이트 동일.

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
        cb = rc.cost_for(interval_start_s=tk, interval_end_s=scratch.now,
                         raw=raw, risk_max=risk)
        total += cb.total_normalized
        if term_sink is not None:
            for k, v in cb.contributions().items():
                term_sink[k] = term_sink.get(k, 0.0) + v
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


# ------------------------------------------------------------ 배포 자격 (YR-107)
# 이름만으로도 판정 가능해야 한다 — 실험이 arm 을 **문자열**로 다루기 때문(arm="JR1800").
ORACLE_ARM_PREFIXES = ("JR", "JOINT_ROLLOUT", "BEAM", "KROLL", "FIFO")


def uses_future_information(policy_or_name) -> bool:
    """이 정책이 **진짜 미래**를 소비하는가 (= 배포 불가·상한 benchmark 전용)."""
    if isinstance(policy_or_name, str):
        n = policy_or_name.upper()
        return any(n.startswith(p) for p in ORACLE_ARM_PREFIXES)
    if getattr(policy_or_name, "USES_FUTURE_INFORMATION", False):
        return True
    if uses_future_information(getattr(policy_or_name, "name", "")):
        return True
    # 선호규칙을 감싼 정책(ResolverPolicy → CentralResolver → preference)까지 따라간다
    for attr in ("pref", "preference", "resolver"):
        inner = getattr(policy_or_name, attr, None)
        if inner is not None and not isinstance(inner, str):
            if uses_future_information(inner):
                return True
    return False


def is_deployable(policy_or_name) -> bool:
    """**배포 후보 자격**. YR-107 규칙 결손 대응.

    사고 경위: YR-100-[3] 에서 학습 arm 6개가 전부 healthy guard 에 걸려 탈락하자,
    사전등록 규칙("guard 통과 arm 중 Δtotal 최소")이 **다른 선택지가 없어** 오라클
    JR1800 을 배포 후보로 자동 지명했다. 라벨을 고치는 것만으로는 같은 사고가 재발한다 —
    후보 선정에서 **구조적으로** 배제해야 한다.
    """
    return not uses_future_information(policy_or_name)


class JointRolloutGreedy:
    """YR-044 1차 baseline — 공동 feasible 행동 조합 열거 → 동일 13항 비용 argmin.

    평가 기준은 **고정 시간창(horizon_s) 누적비용**이다. 무효 판정 §6.2 의 문자 그대로의
    "즉시비용(다음 결정까지) argmin"(horizon_s=0)은 **짧은 행동을 우대해 퇴화**함을 실측했으므로
    (REPOSITION 59%·SERVE 8%·대기 4.67분 — YR-039 SPT 와 동일 함정), 기본값을 고정 시간창으로 둔다.
    행동 후 base_policy 로 시간창 끝까지 진행 → base_policy 위의 1-step 정책개선(rollout algorithm).

    ⚠ **오라클 — 배포 후보가 될 수 없다** (YR-107). 평가에 `_rollout_cost` 를 쓰므로
    시간창 안의 트럭 도착을 **진짜 시각으로** 알고 결정한다. 성능 **상한(benchmark)** 으로만
    읽어야 하며, "공개정보 rollout" 이 아니다(그건 `PredictiveRollout`).
    """

    name = "JOINT_ROLLOUT_GREEDY"
    USES_FUTURE_INFORMATION = True          # 배포 후보 자동 지명 차단 (is_deployable)

    def __init__(self, reward_calc, *, horizon_s: float = 600.0, base_policy=None,
                 max_combos: int = 64, generator=None,
                 forbid_strategic_wait: bool = False, objective=None):
        self.rc = reward_calc
        self.horizon_s = horizon_s
        self.base_policy = base_policy or ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
        self.max_combos = max_combos
        self.generator = generator
        # YR-071: 조합 비교 key 교체 훅 — None(기본)이면 기존 scalar 누적비용 argmin
        # 바이트 동일. callable 이면 창 누적 항별 기여 dict → comparable (예:
        # objectives.hierarchy_key 사전식 tuple). 창·base_policy·후보·resolver 는 불변.
        self.objective = objective
        # YR-045/052 ablation arm: 실작업(전원 비-WAIT) 조합이 공동 실행가능하면 WAIT 포함
        # 조합을 argmin 후보에서 제외. 구조적 WAIT(경합 양보·NO_FEASIBLE)은 아래 fallback 과
        # "실작업 조합 부재 시 전체 유지"로 보존 — 강제 WAIT 는 계약상 어느 arm 에서도 필수.
        self.forbid_strategic_wait = forbid_strategic_wait
        self.n_truncated = 0     # 조합 절단 발생 횟수 — "조용한 축소 금지": 평가 리포트가 보고

    def _admissible_combos(self, sim, dp, gen_by) -> list:
        combos = list(self._combos(dp, gen_by))
        from . import candidates as _cand
        if _cand.WAIT_MODE == "DEFER_TRIGGER":
            # YR-147 C: trigger 없는 DEFER 포함 조합은 전략 선택지에서 제외 — 단 그런
            # 조합밖에 실행가능한 것이 없으면 전체 유지 (구조적 fallback 보존,
            # forbid_strategic_wait 와 동일 패턴).
            trig = [c for c in combos if not any(_untriggered_defer(g) for g in c)]
            if any(_feasible_joint(sim, dict(zip(dp.crane_ids, c))) for c in trig):
                combos = trig
        if not self.forbid_strategic_wait:
            return combos
        full = [c for c in combos
                if all(g.kind != CandidateKind.WAIT for g in c)]
        if any(_feasible_joint(sim, dict(zip(dp.crane_ids, c))) for c in full):
            return full
        return combos            # 실작업 조합이 전부 공동 실행불가 → 구조적 WAIT 경로 보존

    def decide(self, sim, dp, gen_by) -> dict:
        best, best_key = None, None
        for combo in self._admissible_combos(sim, dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            sink = None if self.objective is None else {}
            cost, _ = _rollout_cost(sim, assign, self.rc, horizon_s=self.horizon_s,
                                    base_policy=self.base_policy, generator=self.generator,
                                    term_sink=sink)
            score = round(cost, 9) if self.objective is None else self.objective(sink)
            # 완전순서 tie-break (결정론): 점수(scalar 또는 계층 tuple) → 후보 id 조합
            key = (score, tuple((c, assign[c].candidate_id) for c in dp.crane_ids))
            if best_key is None or key < best_key:
                best, best_key = assign, key
        return best or {c: _wait_of(gen_by[c]) for c in dp.crane_ids}

    def _combos(self, dp, gen_by):
        opts = [[g for g in gen_by[c].items if g.feasible] for c in dp.crane_ids]
        n = 1
        for o in opts:
            n *= max(1, len(o))
        if n > self.max_combos:              # 조합 폭발 방지 (절단은 보고 대상 — 조용한 축소 금지)
            self.n_truncated += 1            # YR-048 리뷰: ETA 로 후보 밀도↑ → 발동 빈도↑, 계수 의무화
            per = max(1, int(self.max_combos ** (1 / max(1, len(opts)))))
            # YR-051: 절단이 WAIT(no-op)을 떨구면, 남은 실작업 조합이 전부 공동 실행불가일 때
            # decide 가 진행 가능한 조합을 못 찾아 라이브락(WAIT 무한반복·완료율 0%)에 빠진다.
            # WAIT 는 정렬 말미라 [:per] 에서 우선 잘려나간다 → 항상 재보존해 SERVE/WAIT 등
            # 부분작업 조합을 보장한다 (WAIT/WAIT 최소한의 feasible 조합도 항상 확보).
            trunc = []
            for o in opts:
                head = list(o[:per])
                w = next((g for g in o if getattr(g, "kind", None) == CandidateKind.WAIT), None)
                if w is not None and w not in head:
                    head.append(w)
                trunc.append(head)
            opts = trunc
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
                 base_policy=None, max_combos: int = 64, generator=None,
                 forbid_strategic_wait: bool = False):
        super().__init__(reward_calc, horizon_s=horizon_s, base_policy=base_policy,
                         max_combos=max_combos, generator=generator,
                         forbid_strategic_wait=forbid_strategic_wait)
        self.width = width

    def decide(self, sim, dp, gen_by) -> dict:
        scored = []
        for combo in self._admissible_combos(sim, dp, gen_by):
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
        """분기 이후 한 시간창을 base_policy 로 더 진행한 누적비용.

        YR-055 정정: 첫 window(_rollout_cost)는 지평 도달 시 **미해소 결정(pending)** 상태로
        scratch 를 반환한다. 구버전은 pending 이면 dp=None 으로 간주해 즉시 반환 — tail 이
        전 분기에서 ≈0 이 되어 2단 lookahead 가 순위를 한 번도 못 바꿨다 (YR-045 에서
        BEAM==JointRollout 60/60 seed 완전 동일로 발현). pending 결정을 base_policy 로
        해소한 뒤 진행한다.
        """
        from .adapter import _max_vessel_risk
        from .engine import TerminalDecision
        if scratch.terminal:
            return 0.0
        gen = self.generator or _default_gen()
        total, t0, tk = 0.0, scratch.now, scratch.now
        risk = _max_vessel_risk(scratch, t0)
        while scratch.now - t0 < self.horizon_s:
            if scratch._pending:
                dp = TerminalDecision(scratch.now, scratch._pending)
            else:
                dp = scratch.run_until_decision()
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
    truncated_before = int(getattr(policy, "n_truncated", 0))
    total, n_dec = 0.0, 0
    term_contrib: dict[str, float] = {}
    listed: dict[str, int] = {}
    listed_feasible: dict[str, int] = {}
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
            for g in gen_by[c].items:           # ETA 경로별 후보 발생 보고 의무 (YR-045)
                listed[g.kind.value] = listed.get(g.kind.value, 0) + 1
                if g.feasible:
                    listed_feasible[g.kind.value] = listed_feasible.get(g.kind.value, 0) + 1
        t_k, risk = dp.time, _max_vessel_risk(sim, dp.time)
        _apply(sim, assign)
        n_dec += 1
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        cb = reward_calc.cost_for(interval_start_s=t_k, interval_end_s=sim.now,
                                  raw=raw, risk_max=risk)
        total += cb.total_normalized
        for k, v in cb.contributions().items():
            term_contrib[k] = term_contrib.get(k, 0.0) + v
    jobs = list(sim.jobs.values())
    done = sum(1 for j in jobs if j.status.name == "DONE")
    waits = [w / 60.0 for w in sim.kpis.wait_samples_s]
    ws = sorted(waits)
    res = {"policy": getattr(policy, "name", type(policy).__name__),
           # YR-107: **정보등급을 원자료마다 박제**한다. 사람 눈에 의존한 라벨은 이미 한 번
           # 실패했다 — YR-087 이 정확히 라벨했는데 이틀 뒤 YR-100/041 에서 유실됐다.
           "uses_future_truth": uses_future_information(policy),
           "deployable": is_deployable(policy),
           "total_cost": total, "n_decisions": n_dec,
           "combo_truncations": (int(getattr(policy, "n_truncated", 0))
                                   - truncated_before),
           "completion_rate": done / max(1, len(jobs)), "backlog": len(jobs) - done,
           "mean_wait_min": (sum(waits) / len(waits)) if waits else 0.0,
           "p95_wait_min": (ws[min(len(ws) - 1, int(0.95 * len(ws)))] if ws else 0.0),
           "vessel_delay_min": sim.kpis.vessel_delay_s / 60.0,   # 야드 job 지각 (구 정의 병기)
           "berth_overrun_min": sim.kpis.berth_overrun_s / 60.0,  # 선석 초과 (YR-080 — 비용과 동일 정의)
           "action_mix": mix.as_dict(), "_mix": mix,
           # YR-045 보고 의무 확장 — 비용 기여·후보 발생·물리 지표
           "term_contrib": dict(sorted(term_contrib.items())),
           "cand_listed": dict(sorted(listed.items())),
           "cand_feasible": dict(sorted(listed_feasible.items())),
           "sts_wait_s": sum(v.sts_wait_accum_s for v in sim.vessels.values()),
           "transfer_wait_s": sim.transfer.transfer_wait_accum_s,
           "loaded_travel_m": sim.kpis.loaded_gantry_m,
           "empty_travel_m": sim.kpis.empty_gantry_m,
           "rehandles": sim.kpis.rehandle_count}
    # YR-089 시간계약 v2 지표 — ledger 활성(v2 시나리오)일 때만 추가 (v1 결과 dict 불변).
    # 이름 규약: mean_wait/p95_wait(=S−B 대기)와 절대 혼용 금지 (spec 수용기준).
    tl = getattr(sim, "time_ledger", None)
    if tl is not None:
        def _stats(sec):
            m = sorted(x / 60.0 for x in sec)
            return ((sum(m) / len(m)) if m else 0.0,
                    (m[min(len(m) - 1, int(0.95 * len(m)))] if m else 0.0))
        bt_mean, bt_p95 = _stats(tl.block_turntime_samples_s())
        tt_mean, tt_p95 = _stats(tl.terminal_turntime_samples_s())
        res.update(block_turntime_mean_min=bt_mean, block_turntime_p95_min=bt_p95,
                   terminal_turntime_mean_min=tt_mean, terminal_turntime_p95_min=tt_p95,
                   terminal_truck_area_h=tl.terminal_area_s / 3600.0,
                   censored_exposure_h=tl.censored_exposure_s() / 3600.0)
    return res
