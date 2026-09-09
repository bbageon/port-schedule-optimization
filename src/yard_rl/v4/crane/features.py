"""크레인 정책이 보는 것 — **핵심만** ([[YR-248]] 2단계).

■ 왜 이렇게 적은가 (연구 설계 원칙 2 · 사용자 지시 2026-07-26)
  *"강화학습 정책 최적화는 핵심 정보만으로 먼저 구성하고, 효과가 실증된 뒤에만 부가
  정보를 추가한다"* — 한꺼번에 많이 넣으면 어디서 문제가 생겼는지 디버깅할 수 없다.

  그래서 **크레인이 지금 이 선택을 할 때 반드시 알아야 하는 것**만 넣는다.
  `adapter.py` 는 20칸 넘게 만들지만 여기서는 **8칸**만 쓴다. 뺀 칸들
  (`eta_confidence`·`interference_penalty_s`·`resequence_count` 등)은 현재 구현에서
  상수 0 이거나([[YR-235]] 가 재배치층에서 같은 이유로 6→3 칸으로 줄였다)
  이 결정과 인과가 멀다. 효과가 실증되면 한 축씩 추가한다.

■ 여덟 칸이 답하는 물음
  ① 이건 무슨 일인가        본선인가 · 실작업인가
  ② 얼마나 급한가           트럭이 얼마나 기다렸나 · 배 여유가 얼마나 남았나
  ③ 얼마나 비싼가           가는 데 · 하는 데 · 파내는 데 얼마나 걸리나
  ④ 지금 얼마나 붐비나       내 블록이 밀렸나

■ 정보 경계 — 실현값은 한 줄도 안 읽는다
  `USES_FUTURE_INFORMATION = False` 계약을 지킨다. 공개된 것(통지·계획·현재 상태)만
  본다. 이 계약이 깨지면 현장에서 재현되지 않는 성능이 나온다([[YR-107]] 의 누출 사고).
"""
from __future__ import annotations

from ..world.contract.schema import CandidateKind

#: 특징 이름 — 순서가 곧 입력 벡터의 자리다. 바꾸면 학습된 망과 어긋난다.
CRANE_FEATURES: tuple[str, ...] = (
    "is_vessel",              # ① 본선 일감인가 (0/1)
    "is_serve",               # ① 실작업인가 (파내기·자리이동·대기와 구분)
    "cum_wait_h",             # ② 이 트럭이 기다린 시간 (시간 · 외부 트럭만)
    "vessel_slack_h",         # ② 배 마감까지 여유 (시간 · ±2h 로 자름)
    "reach_h",                # ③ 크레인이 거기까지 가는 시간
    "service_h",              # ③ 그 일을 하는 데 걸리는 시간
    "rehandle_h",             # ③ 위에 쌓인 걸 파내는 시간
    "n_cands",                # ④ 지금 고를 수 있는 후보 수 (혼잡 대리지표 · 18로 나눔)
)
CRANE_DIM = len(CRANE_FEATURES)

_SLACK_CLIP_H = 2.0           # 배 여유 상한 — 원값이 배 크기 상수라 자른다([[YR-248]] 0단계)
_WAIT_REF_H = 2.0             # 트럭 대기 눈금 — 1시간 초과분에 할증이 붙는 지점의 2배


def crane_features(sim, crane_id: str, gc) -> list[float]:
    """후보 하나를 **8칸**으로 만든다. 값은 대략 0~2 범위로 맞춘다.

    ⚠️ `rank()` 안에서 후보마다 불린다 — 여기가 느리면 시뮬레이션 전체가 느려진다.
       그래서 새 계산을 안 하고 **이미 있는 값만** 꺼내 쓴다.
    """
    ref = gc.job_ref
    if ref is None:                      # WAIT — 물리 특징이 없다
        return [0.0] * CRANE_DIM

    plan = gc.plan
    is_v = 1.0 if ref.is_vessel else 0.0
    is_s = 1.0 if gc.kind == CandidateKind.SERVE else 0.0

    # ② 급한가
    cum = sim.cum_wait(ref.job_id) if ref.is_external else None
    wait_h = 0.0 if cum is None else min(float(cum) / 3600.0, _WAIT_REF_H)

    slack_h = 0.0
    if ref.is_vessel:
        j = sim.jobs.get(ref.job_id)
        vid = getattr(j, "vessel_id", None) if j is not None else None
        v = sim.vessels.get(vid) if vid else None
        if v is not None:
            s = v.slack_s(sim.clock)
            if s is not None:
                slack_h = max(-_SLACK_CLIP_H, min(_SLACK_CLIP_H, s / 3600.0))

    # ③ 비싼가 — 계획이 없으면(실행 불가) 0 으로 둔다
    #   ⚠️ `reach` 는 `JobPlan` 에 없다. `adapter.py:196` 처럼 따로 계산해야 하는데
    #      그건 `rank()` 안에서 후보마다 부르기엔 비싸다. 대신 **빈 주행거리**를 쓴다 —
    #      `empty_gantry_m` 이 이미 계획에 들어 있고 "얼마나 멀리 가나" 를 같은 뜻으로 담는다.
    if plan is None:
        reach_h = svc_h = reh_h = 0.0
    else:
        reach_h = min(float(getattr(plan, "empty_gantry_m", 0.0) or 0.0) / 200.0, 1.0)
        svc_h = min(float(getattr(plan, "duration_s", 0.0) or 0.0) / 3600.0, 1.0)
        reh_h = min(float(getattr(plan, "rehandles", 0) or 0) * 120.0 / 3600.0, 1.0)

    # ④ 붐비나 — 지금 이 크레인이 고를 수 있는 후보가 몇 개인가
    #
    #   ★처음엔 `available_at - clock`(이미 잡힌 일감)을 쓰려 했는데 **죽은 칸**이었다.
    #     결정은 크레인이 **놀 때만** 일어나므로 `available_at ≤ clock` 이 항상 참이다
    #     (실측: 10,651회 호출 중 예외 0회 · 2026-09-10). 구조적으로 0 이다.
    #     [[YR-235]] 가 재배치층에서 같은 이유로 죽은 칸 셋을 걷어낸 적이 있고,
    #     `tests/v4/test_crane_policy.py::test_features_are_not_all_dead` 가 이를 잡았다.
    #
    #   대신 **후보 수**를 쓴다. 고를 게 많다 = 밀린 일이 많다 는 뜻이라 혼잡의 대리지표다.
    #   실측 중앙 3개 · 최대 18개라 18 로 나눠 0~1 로 맞춘다.
    n_cands = float(len(getattr(gc, "_siblings", ()) or ())) or _live_cands(sim, crane_id)

    return [is_v, is_s, wait_h, slack_h, reach_h, svc_h, reh_h,
            min(n_cands / 18.0, 1.0)]


def _live_cands(sim, crane_id: str) -> float:
    """이 크레인이 지금 손댈 수 있는 일감 수 — 후보 목록을 못 받을 때의 대안.

    `rank()` 는 후보를 **하나씩** 받으므로 형제 목록을 모른다. 그래서 엔진의
    대기 일감에서 센다. 새 계산을 안 하려고 상태 문자열만 본다.
    """
    n = 0
    for j in sim.jobs.values():
        st = getattr(j, "status", None)
        if st is not None and "WAIT" in str(st):
            n += 1
    return float(n)
