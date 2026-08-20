# YR-205 — Seller·Buyer 독립 actor와 운영 추론

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-08-19 / **상태**: backlog
- **아키텍처 분류**: ★v3 정책 4/4 — 독립 Seller·Buyer 운영 학생
- **3대 게이트 보정 대상**: `performance`
- **선행**: [[YR-204]]가 `POWER_FAIL` 없이 장기 반사실 교사 라벨 자격 통과
- **1줄**: Seller actor는 `KEEP/SELL`, Buyer actor는 `REJECT/BUY`를 각각 선택하고,
  Resolver는 상호 동의한 요청만 배치로 확정한다. 운영 중 미래 rollout은 하지 않는다.

## 이론적 위치

표준 `KEEP/SELL·BUY/REJECT` 항만 알고리즘을 주장하지 않는다. 기존의 difference
reward·COMA형 반사실 행동 비교·장기 credit assignment를 **yard-block task
reallocation**에 적용한 구조로 한정한다.

## 중앙학습·독립실행

```text
학습: 전체 공개 snapshot + joint request
      → YR-204 actor별 반사실 비용
      → Seller actor·Buyer actor·중앙 critic 학습

운영: 공개 현재상태
      → Seller actor: KEEP / SELL 요청
      → Buyer actor: REJECT / BUY 요청
      → Resolver: 상호 동의 edge만 용량·배치 조합으로 확정
```

Seller actor의 최소 입력은 자기 블록 핵심 부하·작업 `j`의 공개 정보다. Buyer actor의
최소 입력은 자기 블록 핵심 부하·`j` 정보·Seller offer message다. 같은 Buyer 상태라도
어느 소스가 무엇을 내놓았는지에 따라 터미널 이익이 달라지므로 source message 없는
Buyer는 정보부족으로 실격한다. message는 가격이 아니라 공개 상태·Seller 점수다.

## 행동과 비용

각 actor는 YR-204의 비용 차이를 근사한다.

```text
ĝ_S(s_A,j) ≈ c_S^H         # SELL 대 KEEP 비용차
ĝ_B(s_B,j,m_S) ≈ c_B^H     # BUY 대 REJECT 비용차

SELL if ĝ_S < 0
BUY  if ĝ_B < 0
```

탐험 중에는 사전 고정 확률로 반대 행동을 표본화하지만 평가·운영은 결정론이다. 실제
문턱은 라벨 잡음 하한과 calibration으로 사전등록하며 결과를 보고 0 주변을 넓히지 않는다.
`ĝ_S`, `ĝ_B`는 서로 다른 반사실의 한계비용이라 더해서 거래비용으로 쓰지 않는다.

## Resolver 계약

Resolver 입력은 다음 상호 동의 edge뿐이다.

```text
E_t = {(A,B,j) | SELL_A(j)=1 ∧ BUY_B(A,j)=1}
```

Resolver는 `E_t` 밖 거래를 만들지 못한다. `E_t` 안에서는 중앙 critic이 같은
terminal cost로 예측한 **배치 전체 비용**과 용량·소유권으로 조합을 정한다. actor 비용
두 개를 합산하지 않는다. 중앙 critic은 조합 선택기일 뿐 actor의 SELL/BUY를 대신하지
않고, Resolver가 거절한 요청도 request 원장에 그대로 남긴다.

## 비교팔 — 독립 actor 가설을 판정한다

같은 YR-204 교사 데이터·핵심 특징·파라미터 예산으로 비교한다.

1. `IndependentActors`: 독립 SELL/BUY 요청 + 상호동의 batch Resolver.
2. `CentralJoint`: 중앙 모델이 목적지를 직접 고르는 계산 상한 대조.
3. 현행 `quote-only` Resolver와 greedy 기준선.

CentralJoint가 더 좋으면 독립 actor 성공으로 바꿔 말하지 않는다. 독립 구조가 사전
허용손실 안에 들면서 요청 귀속·운영지연 이점을 보여야 가설을 지지한다.

## 수용 기준

- 교사 train/validation/test 시드와 날짜 대역 완전 분리.
- 선택 전 전체 후보에서 actor별 RMSE·부호 정확도·calibration을 보고.
- 실제 이송 전 `SELL∧BUY∧Resolver_selected` 3조건 충족률 100%.
- unilateral SELL, unilateral BUY, Resolver가 만든 무동의 거래 각각 0건.
- 운영 평가에서 counterfactual rollout 호출 0회, 모델+Resolver 지연 예산 준수.
- `IndependentActors`·`CentralJoint`·quote-only·greedy를 같은 신규 시드에서 짝비교.
- 주판정은 terminal total cost 후보−greedy 신뢰구간과 완주·본선·물리 가드.
- 중앙 critic을 제거해도 actor가 binary request를 직접 냈다는 원장 증거가 남아야 한다.

## 하지 않는 것

- actor 출력을 실제 가격이나 Seller relief·Buyer burden 회계로 해석하는 것.
- 별도 지역 목적함수로 Seller와 Buyer를 경쟁시키는 것.
- Resolver가 동의 없는 거래를 만들거나 actor 대신 SELL/BUY를 결정하는 것.
- `c_S+c_B`를 거래비용으로 사용해 터미널 효과를 두 번 세는 것.
- PREMOVE·WAIT 실행정책 동시 재학습, 온라인 rollout, 실제 TOS 제어.
