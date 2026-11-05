"""크레인 정책 입력 — 기본 9칸, 명시적 단일 입력 추가 실험은 10칸.

`feature_mode='legacy'`는 아래의 기존 의미를 보존한다. `'feasible'`은 8번째 칸을
그 크레인이 고를 수 있는 SERVE 후보 수로 바꾼다(개별 실행 가능성 기준, 공동 충돌 전).
`include_flow=True`만 STORE 여부 1칸을 더한다. 두 설정은 체크포인트에 기록한다.

■ 왜 이렇게 적은가 (연구 설계 원칙 2 · 사용자 지시 2026-07-26)
  *"강화학습 정책 최적화는 핵심 정보만으로 먼저 구성하고, 효과가 실증된 뒤에만 부가
  정보를 추가한다"* — 한꺼번에 많이 넣으면 어디서 문제가 생겼는지 디버깅할 수 없다.

  ①~⑧ 은 v4 `crane/features.py` 를 **그대로** 옮겼다(계산식·눈금 동일).
  ⑨ 하나만 더한다 — [[YR-315]] 처방 *"줄을 줄이는 일을 하면서 줄을 안 보고 있다"*.

■ 아홉 칸이 답하는 물음
  ① 이건 무슨 일인가        본선인가 · 실작업인가
  ② 얼마나 급한가           트럭이 얼마나 기다렸나 · 배 여유가 얼마나 남았나
  ③ 얼마나 비싼가           가는 데 · 하는 데 · 파내는 데 얼마나 걸리나
  ④ 지금 얼마나 붐비나       ⑧ `n_cands` (v4 이름 그대로) · ⑨ 줄 선 트럭 수

■ ⚠️ ⑧ `n_cands` 는 이름과 다른 것을 잰다 (v4 에서부터 · 2026-11-04 이식 중 확인)
  v4 는 `gc._siblings`(형제 후보 목록)를 읽으려 했는데 그 속성은 **어디서도 만들지 않는다**
  (저장소 전체 검색 0건). 그래서 늘 대안 `_live_cands` 로 떨어지고, 그것은 블록의 일감 중
  상태 문자열에 "WAIT" 가 든 것 = **`JobStatus.WAITING` = 블록에 도착해 작업을 기다리는
  외부 트럭**만 센다(본선 일감은 WAITING 이 되지 않는다 · `engine.py` BLOCK_ARRIVAL).
  즉 ⑧ ≈ min(줄 선 트럭 수 / 18, 1) 이다. ⑨ 와 거의 같은 정보이고, ⑧ 이 18대에서 포화하는
  것만 다르다. **legacy 설정은 v4와의 연속성을 위해 이 의미를 보존한다**.
  feasible 설정에서는 후보 생성기가 확정한 수를 받아 중복 계산과 대기열 대체를 막는다.

■ ⑨ 줄 선 트럭 수 — `sim.kpis.waiting_count()` 를 그대로 쓴다
  정의 `blockIn ≤ t < serviceStart` — 채점·재배정층 `features/block.waiting_count` 와 같은 뜻.
  재배정층 함수는 기록(`records`)에서 세는데 기록은 60초 격자에서만 동기화된다. 크레인
  결정은 격자 사이에서 나므로, 엔진이 **지금** 들고 있는 같은 집합(`KpiTracker._waiting`:
  도착 시 넣고 작업 시작 시 뺀다)을 읽는다. 눈금은 재배정층 ⑨ 와 같은 `/10`.

■ 정보 경계 — 실현값은 한 줄도 안 읽는다
  `USES_FUTURE_INFORMATION = False` 계약을 지킨다. 공개된 것(통지·계획·현재 상태)만
  본다. 줄 선 트럭은 **이미 도착한** 트럭이라 현재 상태다.
"""
from __future__ import annotations

from ..world.contract.schema import CandidateKind
from ..world.domain.enums import JobStatus, ServiceMode
from ..job_index import IndexedJobs

#: 특징 이름 — 순서가 곧 입력 벡터의 자리다. 바꾸면 학습된 망과 어긋난다.
CRANE_FEATURES: tuple[str, ...] = (
    "is_vessel",              # ① 본선 일감인가 (0/1)
    "is_serve",               # ① 실작업인가 (파내기·자리이동·대기와 구분)
    "cum_wait_h",             # ② 이 트럭이 기다린 시간 (시간 · 외부 트럭만)
    "vessel_slack_h",         # ② 배 마감까지 여유 (시간 · ±2h 로 자름)
    "reach_h",                # ③ 크레인이 거기까지 가는 시간
    "service_h",              # ③ 그 일을 하는 데 걸리는 시간
    "rehandle_h",             # ③ 위에 쌓인 걸 파내는 시간
    "n_cands",                # ④ legacy=WAITING 수, feasible=해당 크레인의 실행 가능 SERVE 수
    "queue_trucks",           # ④ ★v7 — 블록에 줄 선 트럭 수/10 ([[YR-315]])
)
CRANE_DIM = len(CRANE_FEATURES)
#: v4 와 같은 앞 8칸 — 시험이 계산식이 안 바뀌었는지 이 길이로 대조한다.
V4_CORE_DIM = 8

_SLACK_CLIP_H = 2.0           # 배 여유 상한 — 원값이 배 크기 상수라 자른다([[YR-248]] 0단계)
_WAIT_REF_H = 2.0             # 트럭 대기 눈금 — 1시간 초과분에 할증이 붙는 지점의 2배
_QUEUE_REF = 10.0             # 줄 길이 눈금 — 재배정층 블록 요약 ⑨ 와 같다(`features/block.py`)


def queue_trucks(sim) -> int:
    """블록에 **지금** 줄 서 있는 외부 트럭 수 — 도착했고 아직 작업 시작 전."""
    return int(sim.kpis.waiting_count())


def crane_features(sim, crane_id: str, gc, *, common=None, feature_mode='legacy',
                   include_flow=False) -> list[float]:
    """후보 하나를 9칸, STORE 입력을 켜면 10칸으로 만든다.

    ⚠️ `rank()` 안에서 후보마다 불린다 — 여기가 느리면 시뮬레이션 전체가 느려진다.
       그래서 새 계산을 안 하고 **이미 있는 값만** 꺼내 쓴다.
    """
    ref = gc.job_ref
    if ref is None:                      # WAIT — 물리 특징이 없다
        return [0.0] * (CRANE_DIM + int(include_flow))

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
    #   `reach` 는 `JobPlan` 에 없어 **빈 주행거리**로 대신한다(v4 그대로).
    if plan is None:
        reach_h = svc_h = reh_h = 0.0
    else:
        reach_h = min(float(getattr(plan, "empty_gantry_m", 0.0) or 0.0) / 200.0, 1.0)
        svc_h = min(float(getattr(plan, "duration_s", 0.0) or 0.0) / 3600.0, 1.0)
        reh_h = min(float(getattr(plan, "rehandles", 0) or 0) * 120.0 / 3600.0, 1.0)

    # ④ 붐비나 — ⑧ 은 v4 식 그대로(머리말 ⚠️: `_siblings` 는 없어 늘 `_live_cands`)
    live, queue = common if common is not None else decision_common(sim, crane_id)
    n_cands = float(len(getattr(gc, "_siblings", ()) or ())) or live
    if feature_mode == 'feasible':
        n_cands = getattr(gc, 'feasible_serve_count', None)
        if n_cands is None:
            raise ValueError('feasible feature mode requires generator-provided SERVE count')
    elif feature_mode != 'legacy':
        raise ValueError(f'unknown feature_mode: {feature_mode}')

    row = [is_v, is_s, wait_h, slack_h, reach_h, svc_h, reh_h,
           min(n_cands / 18.0, 1.0), queue]
    if include_flow:
        job = sim.jobs.get(ref.job_id)
        row.append(float(job is not None and job.service_mode == ServiceMode.STORE))
    return row


def decision_common(sim, crane_id=''):
    return _live_cands(sim, crane_id), float(queue_trucks(sim)) / _QUEUE_REF


def _live_cands(sim, crane_id: str) -> float:
    """v4 의 대안 계산 — 상태 문자열에 "WAIT" 가 든 일감 수.

    `JobStatus` 는 `(str, Enum)` 이라 `str()` 이 `"JobStatus.WAITING"` 꼴이다. 그래서
    실제로는 **WAITING(블록 도착 · 작업 대기) 일감**만 걸린다. v4 와 같은 값을 내려고
    고치지 않는다(머리말 ⚠️).
    """
    if isinstance(sim.jobs, IndexedJobs):
        return float(sim.jobs.count(JobStatus.WAITING))
    n = 0
    for j in sim.jobs.values():
        st = getattr(j, "status", None)
        if st is not None and "WAIT" in str(st):
            n += 1
    return float(n)
