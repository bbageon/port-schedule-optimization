# YR-205 — Seller·Buyer 장기비용 학생모델과 운영 추론

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-08-19 / **상태**: backlog
- **3대 게이트 보정 대상**: `performance`
- **선행**: [[YR-204]]가 `POWER_FAIL` 없이 장기 반사실 교사 라벨 자격 통과
- **1줄**: 학습 때 만든 장기 반사실 비용을 Seller·Buyer 모델이 근사하고, 운영 때는
  미래 시뮬레이션 없이 모델 예측과 중앙 Resolver만으로 즉시 배정한다.

## 이론적 위치

표준 `KEEP/SELL·BUY/REJECT` 항만 알고리즘을 주장하지 않는다. 기존의 difference
reward·COMA형 반사실 행동 비교·장기 credit assignment를 **yard-block task
reallocation**에 적용한 구조로 한정한다.

## 모델과 실행

```text
학습: snapshot → 실제/반사실 H-rollout → D_k^H → 모델 회귀

운영: 공개 현재상태
      → Seller: 소스 relief/offer 점수
      → Buyer: 수신 burden/accept 점수
      → 중앙 Resolver: 용량·배치 상호작용 포함 최종 목적지
```

Buyer가 터미널 손익을 판단하려면 소스 relief가 필요하다. `R_src` 견적을 거래
metadata로 전달하거나 Resolver가 두 출력을 중앙에서 합친다. 이는 내부 가격이 아니라
정보이며 실제 금전·제로섬 회계는 만들지 않는다.

## 식과 식별성 가드

주 학습 정답은 거래 전체의 한 값이다.

```text
Ĉ_txn(A,B,j) = B̂_B(s_B,j) + C_transfer(A,B,j) − R̂_A(s_A,j)
target        = ΔC_k^H
```

`ΔC_k^H` 하나만으로 `R̂_A`와 `B̂_B`의 내부 분해는 유일하지 않다. 따라서 블록별
원장으로 만든 component label의 합이 pair target과 사전 허용오차 안에서 맞을 때만
각 출력을 “소스 절감”·“수신 부담”으로 해석한다. 실패하면 두 값을 설명용으로 부르지
않고 직접 joint-score 모델만 사용한다.

## 비교팔 — 한 축을 가른다

같은 교사 데이터·특징·파라미터 예산으로 다음을 비교한다.

1. `Joint`: `(s_A,s_B,j) → ΔC_k^H` 직접 예측.
2. `Factorized`: Seller relief + Buyer burden + 고정 이송비.

Factorized가 Joint보다 나아야 한다고 가정하지 않는다. 분산 실행·설명 가능성 이득이
예측력 또는 최종 비용 손실을 만들면 Joint를 유지한다.

## 수용 기준

- 교사 train/validation/test 시드와 날짜 대역 완전 분리.
- 선택 전 전체 후보에서 RMSE·부호 정확도·순위상관·calibration을 보고.
- 운영 평가에서 counterfactual rollout 호출 0회, 모델+Resolver 지연 예산 준수.
- `Joint`·`Factorized`·greedy·현 채택 Q를 같은 신규 시드에서 짝비교.
- 주판정은 terminal total cost 후보−greedy 신뢰구간과 완주·본선·물리 가드.
- component identity 실패 시 Seller/Buyer 의미 주장을 자동 차단.

## 하지 않는 것

- 실험 결과가 좋다는 이유로 모델 출력을 실제 가격으로 해석하는 것.
- 별도 지역 목적함수로 Seller와 Buyer를 경쟁시키는 것.
- PREMOVE·WAIT 실행정책 동시 재학습, 온라인 rollout, 실제 TOS 제어.
