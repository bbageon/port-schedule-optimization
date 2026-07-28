# YR-105-b — 상대 혼잡격차 임계 정책 최적화

## 상태

- **결과 미열람 실행본**.
- 선결 YR-116 완료: 1 bay 탈출 후보와 gate-in 0초 review 계약을 닫았고, 같은 YR-113
  106쌍 민감도에서 이송 이득 방향·유의성이 유지됐다(`dad5df8`, `942c413`).
- YR-117은 과거 결과를 새 지표로 읽은 **사후 재해석**이다. 본 작업이 총비용+A→O
  공동 주지표를 결과 전에 고정하는 첫 임계 최적화 실험이다.

## 질문과 단일 변경축

현행 창중 재배정 임계 `0.10`보다 터미널 총비용과 평균 게이트 진입→진출 시간(A→O)을
함께 낮추는 임계가 `0.05/0.20` 중에 있는가?

```text
gap = C_zero(source) - C_zero(destination)
TRANSFER iff gap >= τ_gap
τ_gap ∈ {0.05, 0.10, 0.20}
```

| arm | 임계 | 뜻 |
|---|---:|---|
| AGGRESSIVE | 0.05 | 작은 상대 혼잡격차에도 이송 |
| BASE | 0.10 | 현행 YR-113 규칙 |
| CONSERVATIVE | 0.20 | 큰 상대 혼잡격차에서만 이송 |

절대혼잡, 지속시간, 본선 필터, quote, 학습망은 넣지 않는다. 임계는 이송량·시점·대상을
함께 바꾸므로 결과를 “순수 볼륨 효과”가 아니라 **상대 혼잡격차 임계 정책 효과**로 부른다.

## 공통 실행 계약

- 동일 실현의 세 arm은 같은 블록 A/B 시나리오·review epoch·정책·resolver를 쓴다.
- 정보등급 `PRE_ADVICE`; 미래 실제값을 읽지 않는다.
- `time_contract_v2=True`, `gate_block_contract=True`,
  `vessel_deadline_achievable=True`.
- YR-112-b 즉시 탈출 계약을 공통 적용하고 arm별 발화 수를 저장한다.
- 추가 이송주행은 `move` 채널과 A→O에 모두 반영한다.
- pilot·선택·확증·기열람 대역의 실현지문 교집합은 0이어야 한다.

## 추정량과 공동 주지표

기준은 항상 `τ=0.10`이다.

```text
B(τ, metric) = Metric(0.10) - Metric(τ)
```

양수면 후보 임계가 현행보다 개선됐다는 뜻이다.

- **공동 주지표**: `total`(터미널 총비용), `a2o_min`(평균 A→O 분).
- 관심효과: total `δ=10`, A→O `δ=1분`; 둘 다 실 SLA가 아닌 assumed 기준.
- 진단: truck, vessel, move, other, B→C, berth minutes.
- truck은 블록 도착부터 시계가 켜져 이송 주행을 빠뜨릴 수 있으므로 진단으로만 쓴다.
- 조작 확인: 이송 수·방향·시점·reject 수. 성공 지표로 쓰지 않는다.

## 결과 전 동결

`prereg_manifest.json`을 pilot 전에 별도 커밋한다. 이 파일은 다음을 고정한다.

- 격자·기준·공동 주지표·δ·판정식.
- pilot의 절대 seed·실현지문·digest와 기열람 지문 비중복.
- 정책·물리·통계·비용설정·본 문서의 HEAD blob digest.

각 단계는 다음 산출물이 현재 HEAD에 추적되고 변경이 없을 때만 열린다.

```text
manifest commit
  → pilot → power_note commit
  → select → results_select commit
  → freeze → winner_freeze commit
  → confirm
```

소스 계약 파일이 한 바이트라도 바뀌면 새 사전등록부터 다시 시작한다.

## 파일럿과 검정력

1. 신규 pilot `n=16`에서 세 arm을 실행한다.
2. 평균·신뢰구간·raw row는 저장하거나 출력하지 않고 아래 **6개 표준편차**만 연다.
   - 비교 3개: `0.05-0.10`, `0.20-0.10`, `0.05-0.20`
   - 지표 2개: total, A→O
3. 정확한 t 분포와 pilot 표준편차의 80% 상측한계를 쓴다.
4. 공동 성공 누락확률을 보수적으로 제한하려고 지표별 power를 0.90으로 둔다.
5. `N_select=max(24, 6개 필요 표본수)`.
6. `N_confirm=2×N_select`.
7. 계획·실현 MDE90이 total `10`, A→O `1분` 이하인지 모두 검사한다.

pilot 지문은 이후 대역에서 제외한다. 결과를 본 뒤 표본을 보충하는 top-up은 금지한다.

## 선택 대역

선택 대역은 승자만 정하고 유의성 주장을 하지 않는다.

```text
score(τ) = min(B_total(τ)/10, B_A2O(τ)/1)
```

- 두 편익 평균이 모두 양수인 `0.05/0.20`만 후보가 된다.
- 후보 중 score 최대를 `τ*`로 고른다.
- 동률이면 이송을 덜 여는 `0.20`을 우선한다.
- 적격 후보가 없으면 `NO_CANDIDATE`로 종료하고 현행 `0.10`을 유지한다.
- 승자·확증 표본수·확증 seed·실현지문·판정식을 별도 커밋으로 동결한다.

## 독립 확증과 라벨

확증에서는 동결한 `τ*`와 `0.10`만 실행한다. 후보 교체, 선택·확증 pooling, 실패 뒤 임계
재조정은 금지한다. 두 공동 지표가 모두 통과해야 하므로 각 5% 교집합 검정에 추가
다중비교 보정은 하지 않는다.

| 라벨 | 조건 |
|---|---|
| `INVALID` | guard·지문·소스·원자료 계약 실패 |
| `POWER_FAIL` | 어느 주지표든 실현 MDE90이 δ를 초과 |
| `JOINT_PRACTICAL_IMPROVEMENT` | 두 95% CI 하한이 각각 `>10`, `>1분` |
| `JOINT_CONFIRMED_SMALL` | 두 하한 `>0`, 단 위 실질 기준은 모두 확정 못 함 |
| `TRADEOFF_FAIL` | 한쪽 개선, 다른 쪽 유의 악화 |
| `HARMFUL` | 어느 주지표든 유의 악화 |
| `EQUIVALENT` | 두 90% CI가 각각 `±δ` 안 |
| `INCONCLUSIVE` | 위 어느 조건에도 해당하지 않음 |

연구적 개선은 `JOINT_CONFIRMED_SMALL`부터 인정하되, 운영 채택은
`JOINT_PRACTICAL_IMPROVEMENT`에서만 허용한다.

## 하드 guard와 주장 범위

- 전 arm 완주율 1.0, backlog 0, 정책 예외 0, 물리 불변식 통과.
- A→O 표본 누락 0·arm별 표본 수 동일.
- 채널 합과 terminal total 일치.
- 임계 숫자를 제외한 실제 행동 trace가 arm 사이에서 달라야 한다.
- 이송 trace 수와 `n_moved`, reject trace와 `n_rejected`가 일치해야 한다.

통과해도 문헌 보정 비대칭 2블록·반입 재배정·검사한 세 임계에 한정한다. 연속 임계
전역최적, 다른 터미널 일반화, 이송정책 자체의 무이송 대비 최종 채택은 주장하지 않는다.
마지막 질문은 YR-115가 동결 승자와 `NOTRANSFER`를 독립 비교해 답한다.
