# 독립 Seller·Buyer 행동과 반사실 비용 계약 (사용자 결정 2026-08-20)

> 상태: YR-203~205 설계 보완. 구현·사전등록·성능 판정 전이다. 현재 performance
> 게이트가 `INCONCLUSIVE`이므로 기존 YR-200→201→202 뒤의 backlog로만 유지한다.

## 1. 보완이 필요했던 이유

2026-08-19 등록안은 판매 세계와 비판매 세계의 학습용 장기 rollout은 명시했지만,
Seller·Buyer를 독립 actor가 아니라 relief·burden 견적모델처럼 읽을 수 있었다.
그대로 구현하면 Resolver가 사실상 SELL/BUY까지 대신 결정해 다음 사용자 의도와 어긋난다.

1. Seller는 `KEEP/SELL`을 직접 선택한다.
2. Buyer는 `REJECT/BUY`를 직접 선택한다.
3. Resolver는 양쪽이 동의한 요청의 가능한 batch 조합만 고른다.
4. 실제 terminal objective는 하나이고, 각 actor에는 자기 행동만 바꾼 반사실 비용을 준다.

따라서 새 ID를 만들지 않고 정확히 이 범위를 가진 YR-203~205를 보완한다. YR-206은
Buyer 귀속의 동기 기록으로 남기고 독립 실행축으로 쓰지 않는다.

## 2. 운영 행동 계약

작업 `j`가 공개 ETA 창에 최초 진입한 epoch에서 한 번만 판단한다.

```text
Seller A: a^S_A(j) ∈ {KEEP, SELL}
Buyer  B: a^B_B(A,j) ∈ {REJECT, BUY}

후보 edge (A,B,j)
⇔ SELL_A(j) ∧ BUY_B(A,j)
```

Seller는 `SELL(j)`와 작업의 공개 정보·offer score를 방송한다. 허용 Buyer들은 자기
정책으로 각각 응답한다. Resolver는 상호 동의 edge 밖의 거래를 만들 수 없다.
동의가 여러 개면 용량·중복소유·동시오더 상호작용을 지키는 batch만 원자 확정한다.
확정·KEEP 뒤에는 `allocation_decided=True`로 잠가 다시 판매하지 않는다.

## 3. 비용 정본 — 하나의 terminal objective

새 SELL 상금, BUY 벌점, 가상 가격을 만들지 않는다. 잠금평가가 쓰는 동일한
`TerminalCostConfig`와 같은 구간 비용을 모든 세계에 적용한다.

```text
J_H(W) = Σ_[t,t+H] C_terminal(interval | W)
```

구현 전 다음을 설정 hash와 함께 동결한다.

- 활성 `cost_id`와 13개 비용항의 scale·weight.
- 트럭대기·장기대기, 크레인 주행·공차, 재취급, 본선지연 등 활성 terminal 항.
- 외생 도착·서비스시간·고장·ExecutionHead 난수와 평가 구간.
- 이송이 기존 terminal 장부에 기록되면 actor 식에 별도로 다시 더하지 않는 규칙.

따라서 직접 행동비가 없는 행동도 시간이 흐르며 대기·지연이 늘면 비용이 생긴다.
Seller와 Buyer의 지역 비용을 따로 최적화하지 않고 같은 terminal 목적을 사용한다.

## 4. actor별 반사실 세계

눈가림 파일럿에서 정한 decision epoch의 actor 행동좌표를 짝으로 비교한다.

```text
W_S^1: A가 SELL(j), Buyer 응답 후 Resolver 실행
W_S^0: A가 KEEP(j), j의 판매 edge 없이 Resolver 실행

W_B^1: Seller offer 고정, B가 BUY(A,j), Resolver 재매칭
W_B^0: Seller offer 고정, B가 REJECT(A,j), Resolver 재매칭
```

작은 값을 선호하는 비용학습 정답은 다음이다.

```text
c_S^H(A,j)   = J_H(W_S^1) − J_H(W_S^0)   # SELL 대 KEEP
c_B^H(A,B,j) = J_H(W_B^1) − J_H(W_B^0)   # BUY 대 REJECT
```

- `c<0`: 실제 SELL 또는 BUY가 장기 terminal 비용을 낮췄다.
- `c>0`: 기본행동 KEEP 또는 REJECT가 더 좋았다.
- 보상 최대화 코드만 `r=−c`로 바꾸며 한 실험에서 두 부호를 섞지 않는다.

Seller가 KEEP하면 작업의 모든 수신 edge가 사라지지만, Buyer가 REJECT하면 다른 Buyer가
대신 받을 수 있다. 따라서 두 비용은 일반적으로 다르다. Seller 짝은 `j` 외 요청을,
Buyer 짝은 Seller offer와 다른 Buyer 응답을 고정하고 Resolver만 재매칭한다. 실제 행동과
반대 행동을 모두 굴리는 강제 토글은 교사 생성에서만 허용한다.

## 5. 이중계상 방지

실제 성능은 `J_H(W)` 한 번으로만 평가한다.

```text
금지: terminal improvement = c_S + c_B
허용: Seller 학습표적 = c_S, Buyer 학습표적 = c_B
```

한 거래에서 두 actor 표본이 생겨도 학습배치 기여는 `(L_S+L_B)/2`로 평균한다.
Resolver도 `c_S+c_B`를 거래비용으로 쓰지 않는다. Resolver의 중앙 critic은 같은
terminal objective로 batch 전체를 한 번 평가한다.

## 6. 학습과 운영의 정보 경계

학습은 centralized training으로 전체 공개 snapshot과 joint request를 critic이 볼 수 있다.
운영 actor 입력은 최소 공개정보로 제한한다.

- Seller: 자기 블록 핵심 부하 + 작업 `j`의 공개 ETA·유형·마감.
- Buyer: 자기 블록 핵심 부하 + `j` 정보 + Seller offer message.
- 금지: actual future arrival, 실현 서비스시간, 향후 고장 등 미래진실.

Buyer가 Seller message를 못 보면 같은 Buyer 상태·같은 작업이라도 어느 소스가 내놓았는지에
따른 terminal 이익 차이를 구분할 수 없다. message는 금전 가격이 아니라 정책 정보다.

운영에서는 반사실 rollout을 하지 않는다.

```text
Seller actor → KEEP/SELL
Buyer actor  → REJECT/BUY
상호 동의 edge → 중앙 critic Resolver → batch 확정
```

## 7. 지평 H와 밀어내기 방지

짧은 `H`는 비용을 창 밖으로 미루는 행동을 좋게 보일 수 있다. 새 tail model을 붙여
이를 우회하지 않는다. 결과 열람 전에 `H`, 영향 작업 완료조건, 마지막 구간 비용차
안정 허용치를 고정한다. 영향이 `H` 밖에 남으면 해당 라벨은 실격한다. 필요한 표본이나
벽시계가 예산을 넘으면 YR-204를 `POWER_FAIL`로 종료한다.

## 8. 검토한 대안과 판정

| 대안 | 판정 | 이유 |
|---|---|---|
| Resolver가 relief−burden으로 SELL/BUY까지 결정 | 기각 | 독립 actor가 아니며 행동 귀속 가설을 검증하지 못함 |
| Seller·Buyer 지역비용을 각각 최적화 | 기각 | 떠넘기기와 무조건 REJECT를 유발 |
| `c_S+c_B`를 terminal 비용으로 사용 | 기각 | 한 거래 효과 이중계상 |
| 한 pair label을 relief·burden으로 임의 분해 | 기각 | 분해가 유일하지 않음 |
| 운영 때 H-rollout | 기각 | 학습정책이 아니라 온라인 시뮬레이션 최적화가 됨 |
| 기존 YR-203~205와 별도 새 row | 기각 | 같은 축을 중복 등록해 순서·증거가 갈라짐 |

## 9. 판정과 되돌릴 조건

YR-205는 독립 actor를 `CentralJoint` 계산 상한, 현행 quote-only, greedy와 같은 신규
시드에서 비교한다. 중앙 직접배정이 더 좋다는 결과를 독립 actor 성공으로 바꿔 말하지
않는다. 독립 actor가 사전 허용손실을 넘거나 unilateral 요청이 이송되면 가설을 기각한다.

현재 순서는 바꾸지 않는다.

```text
YR-200 → YR-201 → YR-202
                         ↓ performance 게이트·별도 허가
YR-203 → YR-204 → YR-205
```
