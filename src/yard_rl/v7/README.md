# v7 — 수평형 터미널 · 결정 주체는 크레인 하나 · 공정한 장부

v6 를 독립 복사했다(기준 커밋 `fec9a151` · 2026-11-04 · `V6_BASE_COMMIT`). 원본 v6 는 수정하지 않는다.
상위 설계: `.claude/docs/dashboard-task-specs/YR-332-v7-crane-only-horizontal.md`.
v6 시절 설명은 `README-v6.md`·`README-piece8.md`(복사본 · 역사 문서)에 그대로 있다.

## 한 줄로

판매·구매 재배정을 끄고 **크레인 결정만** 배운다. 트럭은 원래 블록·시각으로 오고,
크레인이 "다음에 어느 일감을 할까" 를 고를 때마다 세계 둘을 굴려 정답을 만든다(v4 반사실 교사).

## v6 와 무엇이 다른가

| 축 | v6 | **v7** | 어디 |
|---|---|---|---|
| 기본 재배치 팔 | `RL`(판매·구매 시장) | **`NO_REALLOC`** — 시장을 안 연다 | `stage/episode.py` `CRANE_ONLY_ARM` · `month_run.run_month(arm=None)` |
| 본선 유휴 단가 | 2.99 원/GT·시간 | **35.88** (용선료 근거 · v4 [[YR-307]]) | `reward/krw.py` |
| 통지 리드타임 | 반입·반출 한 분포(`DIST`) | **`DGT`** — 반입·반출 따로, 통지 전 도착은 리드 0 | `stage/dgt_lead.py` · `stage/orders.py` |
| 크레인 행동 | 미리 자리잡는 빈 이동 포함 | **끔**(`safety_only`) — 교착 탈출·파내기·쉬기는 남김 | `stage/exec_config.py` |
| 크레인 학습 | (없음 · PPO 는 시장 쪽) | **v4 반사실 교사** + 입력 9칸 | `crane/` |
| 물리 세계 `world/` | — | v6 복사본 + **양하 인계 용량 보정·계산 최적화**(YR-332-h/j) | `tests/v7/test_world_clone.py`의 명시적 예외 |

옛 동작은 지우지 않았다 — 비교·회귀용 선택지로 남는다:
`arm="RL"`, `lead_mode="DIST"|"FIXED"`, `exec_config=exec_config.V6_LEGACY`.
PPO 유산 경로(`ppo/`)는 `run_month(ppo=…)` 일 때 v6 계약(`arm="RL"`)을 그대로 쓴다.

## 양하 지연 연결 (2026-11-05)

CPU 하루·연속 크레인 경로는 양하 인출부터 야드 적재 완료까지의 미완 상자 수를 제한한다.
같은 블록의 모든 선박이 실행 프로파일의 운반차 대수만큼 용량을 공유한다(현재 H21 3대).
야드 처리가 밀리면 본선도 대기하고 그 시간이 기존 본선 비용에 들어간다.

`discharge_backpressure=False`로 이전 환경을 비교할 수 있다. 결과의 `environment`에
설정이 기록된다. 구 PPO는 기본 OFF이며 GPU 배열 경로는 새 연결을 구현하지 않았다.
차량별 왕복 궤적 대신 집계 근사이므로 실제 용량·마감 보정은 별도 과제다.
[모형·문헌](../../../docs/research/v7-discharge/model.md) ·
[진단 결과](../../../outputs/reports/yr332_discharge_latency/report.md).

## 상태 목록·계획 재사용 (2026-11-05)

오더 상태 목록을 갱신하고 후보 검색에 사용한다. 공통 정보와 같은 입력의 신경망 계산을 공유한다.
후보 계획·적재 위치 탐색은 물리 조건이 같을 때 재사용한다. 야드·크레인·예약이 바뀌면
재계산하며 최종 공동 계획·예약 검사는 기존 경로로 수행한다. 복제 세계끼리 캐시를 공유하지 않는다.
기존 9입력 의미·보상·후보 기본 상한은 그대로다. `CandidateGenerator(reuse_plans=False)`와
`CranePolicy(batched=False)`로 계산 재사용을 끈 대조를 구성할 수 있다.
[구현 계약](../../../docs/research/v7-discharge/dispatch-reuse.md) ·
[진단 결과](../../../outputs/reports/yr332_dispatch_reuse/report.md).

## DGT 리드타임 — 무엇을 빌렸나

원 표본(1시간·100건 중 통지 있는 71건)에서 **분포 모양만** 빌렸다. 상수표와 출처(sha256)는
`stage/dgt_lead.py` 머리말. 뽑는 분포의 분위(시간):

| | p25 | 중앙 | p75 | 통지 전 도착(리드 0) |
|---|---|---|---|---|
| 반입 | 0.35 | 0.95 | 2.02 | 11.1% (5/45) |
| 반출 | 0.19 | 2.40 | 91.2 | 19.2% (5/26) |

⚠️ 하루 무대에서는 통지가 그날 0시보다 앞설 수 없어, 반출의 긴 꼬리(91시간)는
"0시부터 알던 트럭" 으로 눌린다 — 무대에 실제로 담긴 반출 p75 는 약 12시간이다
(`outputs/reports/yr332_v7_scaffold/scaffold_evidence.json`).

## 크레인 교사 (`crane/`) — v4 에서 옮긴 것

| 파일 | 무엇 |
|---|---|
| `features.py` | 입력 9칸 = v4 핵심 8칸 + **줄 선 트럭 수**(`sim.kpis.waiting_count()/10`) |
| `policy.py` | 2층 64 망 · 실작업 우선 울타리 · 강제 손잡이 `force_once` |
| `rollout.py` | 결정 하나 → 세계 둘(사실·차점자 강제) · 진단 `TRACE_DECISIONS` |
| `collect.py` | 쓸 수 있는 결정만 예산을 쓴다 · 버린 것까지 센다 |
| `teacher.py`·`fit.py` | 짝 중심 정답 · 그날 격차 눈금(하한 1,000원) · Huber |
| `train.py` | 하루→라벨→갱신 · **고정 평가일 짝비교**(`evaluate`) |
| `judge.py` | v4 하루 무대 28일 판정기(연속 30일 판정은 [[YR-332-d]] 에서 따로) |

    PYTHONPATH=src python -m yard_rl.v7.crane --iters 20 --workers 6

⚠️ **⑧ `n_cands` 는 이름과 다르다** — v4 에서부터 `gc._siblings` 가 없어 늘 대안식으로 떨어지고,
그 값은 블록의 WAITING 일감 수(=줄 선 외부 트럭) / 18 이다. 새 ⑨ 와 18대 아래에서 **같은 정보**다
(하루 전 행에서 100% 일치 · `tests/v7/test_v7_crane_teacher.py::test_n_cands_is_really_the_waiting_queue`).
v4 연속성을 위해 고치지 않았다 — 고칠지는 별도 결정.

## 돌려 보기

    # WSL · venv ~/.venvs/yard-rl
    PYTHONPATH=src:tests/v7 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m pytest tests/v7 -q
    PYTHONPATH=src python scripts/v7/scaffold_evidence.py outputs/reports/yr332_v7_scaffold/scaffold_evidence.json
    PYTHONPATH=src python scripts/v7/probe_crane_scale.py --loads 3500,7500,15000 --labels 12 --workers 2
    PYTHONPATH=src python scripts/v7/probe_branch_trace.py 1500 10800 8 outputs/reports/yr332_v7_scaffold/branch_trace_1500.json

## 한계

- 합성 터미널이다. v4·v5·v6 와 비용표·무대·행동 범위가 달라 **이전 세대 수치와 직접 비교하지 않는다.**
- 생성 입력 그대로 연속 무대를 굴리면 반출 대상 상자가 모자라 못 들어오는 트럭이 생긴다
  (3일·부하 300 에서 900대 중 32대 NO_TARGET) — [[YR-331-c]] 의 몫이다.
- `gpu/`(배열 세계)는 v6 사본 그대로이고 v7 의 새 기본값(크레인 행동·리드타임)을 반영하지 않았다.
