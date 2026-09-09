"""정책용 **최소** 본선 신호 (스케줄중심 — YR-088 파생, 사용자 최소화 반영).

원칙: 야드크레인 정책이 보는 본선 상태는 STS/YT 의 작업·유휴 raw 가 아니라 **결정에 필요한
요약** 둘 뿐이다.
  ① schedule_slack (feedforward) — 알려진 스케줄로 계산한 마감 대비 급함 (트럭 ETA 의 본선판).
  ② flow_margin  (feedback)     — STS 가 굶기(적하)/막히기(양하)까지 여유 한 숫자. YT/버퍼를
                                   raw 로 노출하지 않고 **하나로 요약**. 스케줄만으론 못 잡는
                                   경합·교란을 잡는다.
정확한 미래는 rollout 몫 — 이 둘은 **싼 근사**다(계약: feature 는 rollout 인 척 금지, YR-088 §5).

시뮬레이터는 STS/YT/버퍼를 전부 모델하지만, 정책엔 이 요약만 반영한다(상태공간 최소·일반화 유리).
"""
from __future__ import annotations

from .vessel import VesselWorkType


def schedule_slack_s(v, now: float) -> float | None:
    """스케줄 급함(feedforward) = 계획 완료 − 지금 − 남은 계획 작업시간. 음수=계획상 이미 초과 위험.

    완료계획 결측(SYMPTOM) 이면 None. = vessel.slack_s (알려진 스케줄만 사용, 싸다).
    """
    return v.slack_s(now)


def flow_margin_s(sim, v, now: float) -> float | None:
    """STS 흐름 여유(feedback) 한 숫자 — 음수면 아직 안 멈췄어도 곧 STS 정지 위험.

    적하(LOAD): STS 가 버퍼 소비 → `버퍼 소진까지 시간 − 다음 보충 예상`.
    양하(DISCHARGE): STS 가 버퍼 채움 → `버퍼 만재까지 시간 − 다음 배수 예상`.
    다음 보충/배수 예상 = YT 한 트립 × (1 + 대기열/유닛수) — 싼 근사(정확 = rollout).
    """
    if v.done or not v.started:
        return None
    cadence = v.plan.sts_move_interval_s
    n = max(1, sim.transfer.n_units)
    supply_eta = sim.transfer.move_time_s * (1.0 + sim.transfer.waiting_count() / n)
    if v.work_type == VesselWorkType.LOAD:
        endurance = v.buffer_level * cadence            # 버퍼 소진까지 (0 이면 즉시 굶주림 위험)
        return endurance - supply_eta
    headroom = (v.plan.quay_buffer_cap - v.buffer_level) * cadence   # 만재까지 (0 이면 즉시 막힘)
    return headroom - supply_eta


def minimal_vessel_state(sim, now: float) -> dict[str, dict]:
    """정책용 최소 본선 상태 — 미완 선박별 (schedule_slack, flow_margin) 요약만.

    STS/YT 작업·유휴·버퍼 raw 를 **노출하지 않는다**(사용자 최소화). 가장 위험한 선박 보호는
    호출부가 min(schedule_slack) 등으로 집계.
    """
    return {vid: {"schedule_slack_s": schedule_slack_s(v, now),
                  "flow_margin_s": flow_margin_s(sim, v, now)}
            for vid, v in sim.vessels.items() if not v.done}


def most_urgent(sim, now: float) -> tuple[str, dict] | None:
    """가장 급한 선박(schedule_slack 최소) — 평균 아닌 최악 보호 (설계 §2)."""
    st = {k: s for k, s in minimal_vessel_state(sim, now).items()
          if s["schedule_slack_s"] is not None}
    if not st:
        return None
    vid = min(st, key=lambda k: st[k]["schedule_slack_s"])
    return vid, st[vid]
