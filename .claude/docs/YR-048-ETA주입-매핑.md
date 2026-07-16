# YR-048 — 통합 시나리오 제공 ETA 주입

> YR-047 적대 리뷰 파생 발견의 해소. 코드 원본은
> `src/yard_rl/integrated/scenario_gen.py`, 검증은
> `tests/integrated/test_yr048_eta_injection.py`다.

## 1. 문제와 결론

선제 재조작(PRE_REHANDLE)은 트럭이 오기 전에 반출 컨테이너 위의 방해물을 치우는 행동이다.
통합 시나리오가 트럭 도착예정시각(ETA)인 `provided_eta`를 만들지 않아 이 후보가 항상 0건이었다.

외부트럭 반입·반출에 ETA를 주입한 결과, PRE_REHANDLE 후보는 전체 후보의 **14~15%**로
발생하고 기본 정책도 에피소드당 3~7회 선택했다. 다만 리뷰에서 ETA가 위치 이동(REPOSITION)도
함께 활성화한다는 사실을 확인했으므로 두 효과를 하나로 해석하면 안 된다.

## 2. 구현 계약

| 항목 | 동결 내용 |
|---|---|
| ETA 모델 | `max(0, 실제도착 + uniform(-eta_error_s, +eta_error_s))` |
| 기본 오차 | `eta_error_s=300초`; 0이면 완전한 ETA |
| 적용 대상 | 외부트럭 반입·반출; 본선 연계 작업은 `None` |
| 난수 격리 | `random.Random(f"eta:{seed}")`; 기존 도착·대상·본선 난수열 불변 |
| 시나리오 식별 | `scenario.meta["eta_error_s"]`에 박제; YR-019 품질 실험의 arm 식별값 |
| 정보 경계 | PRE_ADVICE에서만 ETA 가시; GATE_IN·BLOCK_ARRIVAL에서는 후보 0건 |

±300초 균등오차는 단일야드 생성기의 기존 EMPIRICAL 관행을 승계한 가정값이다. 실제 분포와
미등록·편향·오래된 ETA는 YR-019가 다룬다.

## 3. 발생 실측

`BaselinePreference`, PRE_ADVICE, 미처리 작업 0건 완주 기준이다.

| seed | 전체 후보 | PRE_REHANDLE | 선택 | 완료 |
|---|---:|---:|---:|---|
| 310000 | 364 | 51 (14.0%) | 3 | backlog 0 |
| 310007 | 714 | 110 (15.4%) | 5 | backlog 0 |
| 700123 | 755 | 115 (15.2%) | 7 | backlog 0 |

## 4. ETA의 두 행동 경로

같은 seed 310000에서 ETA만 제거한 사본과 비교했다. 전용 난수 흐름 덕분에 컨테이너·도착·본선
구조는 같고 `provided_eta`만 다르다.

| 조건 | SERVE | PRE_REHANDLE | REPOSITION | WAIT |
|---|---:|---:|---:|---:|
| ETA 없음 | 85 | 0 | 48 | 75 |
| ETA 있음 | 81 | 51 | 159 | 73 |

REPOSITION은 48→159건으로 **3.3배** 늘었다. 미래 트럭의 대상 bay를 미리 알아 위치를 잡는
두 번째 경로다. 따라서 YR-045는 다음 3개 제거 실험으로 기여를 분리한다.

1. `NO_ETA`: ETA 없음 — 두 경로 모두 꺼짐.
2. `ETA_NO_PRE`: ETA는 보이되 PRE_REHANDLE만 차단 — 위치선점 효과.
3. `FULL`: ETA와 두 행동 모두 사용 — 선제 재조작의 추가효과.

사전등록 원본은
[YR-045 정정판 locked 재실험](strategy-history/2026-07-16-YR-045-corrected-locked-rerun-prereg.md)이다.

## 5. 기존 baseline 수치 정정

YR-044의 `93.4 < 96.6`은 ETA가 없던 시점의 보정값이다. ETA 주입 후 같은 seed에서
ServiceFirstSPT는 **79.643**, JointRolloutGreedy는 **62.927**이고 두 정책 모두 완료율 100%다.
실작업 가능 시 SERVE 선택률은 0.578과 0.456으로 기존 건전성 문턱 0.25를 계속 통과한다.
JointRollout은 후보 조합 64개 초과 축소가 4회 발생했으며 이제 실행 결과에 횟수를 보고한다.

## 6. 검증과 리뷰 후속

전용 테스트 7건은 ETA 범위·본선 제외·난수 격리·고정 ETA 값·결정론·정보 경계·두 행동 경로·
계약 기록을 고정한다. baseline 테스트는 조합 축소 계수와 결과 노출을 고정한다.

- 관련 묶음: **16 passed** (`test_yr048_eta_injection.py` + `test_baselines.py`).
- Windows 순수 파이썬 전체: **287 passed**.
- WSL torch 포함 전체: **328 passed, 3 skipped, 1 UI test deselected**.

리뷰가 확인한 다음 두 항목은 이벤트·feature 계약 변경이라 YR-048 범위 밖이다.

- 작업 가능한 SERVE가 없으면 결정 시점이 열리지 않아 첫 트럭 전 선제 재조작은 불가능하다.
- ETA가 지났지만 트럭이 아직 안 온 최대 300초 동안 음수 도착차가 feature에서 0으로 잘린다.

두 항목은 [YR-050 명세](dashboard-task-specs/YR-050-eta-decision-epoch.md)에 등록했고,
YR-045의 선결조건으로 두었다. 그래서 YR-048의 결론은 **ETA 데이터 경로와 기회주의적 선제행동
활성화 완료**이며, “한산기 조기 선제작업까지 완전 활성화”는 YR-050 완료 전 주장하지 않는다.
