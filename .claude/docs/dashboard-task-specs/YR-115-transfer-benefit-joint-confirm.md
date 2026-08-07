# YR-115 — 현행 0.10 이송의 공동 순효과 확증
- **상태**: **ready** (2026-08-07 중간점검 — board ready.md 와 정합)

## 질문

문헌 보정 비대칭 2블록에서 현행 창중 이송 규칙 `gap>=0.10`은 이송을 전혀 하지 않는 것보다
터미널 총비용과 평균 게이트 진입→진출 시간(A→O)을 **함께** 줄이는가?

YR-105-b가 `0.05/0.10/0.20` 중 대안 후보를 찾지 못했으므로 임계는 결과 전에 0.10으로
고정됐다. 이 과제에서는 임계를 다시 선택하지 않는다.

## 비교와 부호

| arm | 정책 |
|---|---|
| ADOPTED | `gap>=0.10`이면 창중 이송 |
| NOTRANSFER | 임계 무한대, 이송 0 |

```text
B(metric) = Metric(NOTRANSFER) - Metric(ADOPTED)
```

양수면 현행 이송이 이롭다는 뜻이다.

## 공동 주지표

- total: 터미널 총비용, 실질기준 `δ=10`.
- A→O: 평균 게이트 진입→진출 분, 실질기준 `δ=1분`.
- 두 지표의 95% 신뢰구간 하한이 **모두 0보다 커야** 연구적 개선이다.
- 두 하한이 각각 δ까지 넘어야 운영적으로 의미 있는 개선이다.
- δ는 실 SLA가 아닌 assumed 연구 기준이다.

truck·vessel·move·other·B→C·berth는 진단이다. truck은 블록 도착부터 시계가 켜져 다른
블록으로 가는 추가 주행을 빠뜨릴 수 있으므로 채택 판정에 쓰지 않는다.

## 공통 실행 계약

- 동일 실현에서 두 arm은 같은 블록 A/B 시나리오·review epoch·SF 정책·resolver를 쓴다.
- 정보등급 `PRE_ADVICE`; 미래 실제값을 읽지 않는다.
- 시간계약 v2·gate-block 계약·달성 가능한 본선마감·즉시 교착탈출을 공통 적용한다.
- 추가 이송주행은 total의 move 항과 A→O에 반영한다.
- ADOPTED의 이송 trace 수와 `n_moved`, NOTRANSFER의 `n_moved=0`을 강제한다.
- 완주율 1.0·backlog 0·정책 예외 0·채널합 일치를 강제한다.
- A→O는 실제 gate-in 원장 전부에 실제 gate-out이 있어야 하며 검열 표본은 0이어야 한다.
- arm별 전체 작업 수와 A→O 완료 수가 같아야 하며 ADOPTED 전체 이송은 1건 이상이어야 한다.

## 파일럿과 검정력

신규 pilot 16쌍은 두 주지표의 평균·CI·raw row를 저장·출력하지 않고 짝차이 표준편차만
연다. 결과를 보기 전에 YR-113의 더 약한 대역을 바탕으로 다음 계획효과를 고정한다.

- total `3.70`.
- A→O `0.82분`.

이는 채택 기준 δ가 아니라 **잡고 싶은 최소 연구효과**다. pilot SD의 80% 상측한계와 정확한
t 분포, endpoint power 0.90으로 지표별 필요 n을 계산한다. 두 endpoint가 각각 0.90이므로
공동 AND 성공확률의 보장 하한은 0.80이며, 공동 power 0.90으로 표현하지 않는다.

```text
N_confirm = max(24, n_total(effect=3.70), n_A2O(effect=0.82))
```

pilot 지문은 확증에서 제외하고 top-up은 금지한다. 확증의 실현 MDE90도 각 계획효과 이하여야
한다. pilot 평균은 어떤 형태로도 arm·표본 선택에 사용하지 않는다.

## 단계 동결

```text
manifest commit
  → blind pilot → power_note+confirm-band commit
  → confirm
```

manifest는 절대 seed·실현지문·기열람 비중복, 지표·효과·δ·판정식과 `src/yard_rl`,
`configs` 전체 추적 tree의 blob digest를 고정한다. pilot guard 실패 시 power note 자체를
만들지 않는다. confirm은 power note의
스키마·guard·상수·필요 n·독립 대역을 다시 계산하며, 직전 산출물이 HEAD와 같을 때만 열린다.

## 무효 v1 파일럿

첫 파일럿 16쌍은 평균을 열지 않았으나, transitive import 일부가 동결 밖에 있었고 미출문
차량을 `평가종료-A`로 검열해도 A→O guard가 통과하는 결함이 뒤늦게 발견됐다. v1의 n=238은
확증에 쓰지 않으며 해당 32개 실현도 기열람 집합에 넣는다. v2는 전체 source/config tree와
실제 gate-out 완료 100%를 강제하고 새로운 파일럿 대역에서 다시 시작한다.

## 판정

| 라벨 | 조건 |
|---|---|
| `INVALID` | guard·지문·소스·원자료 계약 실패 |
| `POWER_FAIL` | 어느 주지표든 실현 MDE90이 계획효과 초과 |
| `JOINT_PRACTICAL_IMPROVEMENT` | 두 95% CI 하한이 각각 `>10`, `>1분` |
| `JOINT_CONFIRMED_SMALL` | 두 하한 `>0`, 실질기준은 모두 확정 못 함 |
| `TRADEOFF_FAIL` | 한쪽 개선·다른 쪽 유의 악화 |
| `HARMFUL` | 어느 주지표든 유의 악화 |
| `EQUIVALENT` | 두 90% CI가 각 `±δ` 안 |
| `INCONCLUSIVE` | 위 어느 조건에도 해당하지 않음 |

두 지표를 모두 통과해야 하는 교집합 검정이므로 추가 다중비교 보정은 하지 않는다.
`JOINT_CONFIRMED_SMALL`은 연구결과로 인정하지만 운영 채택은 실자료 δ 확정 전 보류한다.

## 주장 범위

통과해도 문헌 보정 비대칭 2블록·반입 재배정·현행 0.10 규칙에 한정한다. 다른 터미널,
양하 재배정, 연속 임계 전역최적, 실제 운영 배포는 주장하지 않는다.
