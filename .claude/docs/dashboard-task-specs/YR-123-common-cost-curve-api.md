# YR-123 — 공통 작업비용 계산기 통일 + 지연 한계비용 곡선 API

> 상태: **done (2026-07-30 — 곡선 API·테스트 11 완료. quote 배선은 YR-133 이월)** · 2세대
> 세대 기준: [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md)

## 현재 사실

- 본선 완료비용 공식 `vessel_cost.py`는 구현돼 있고 YR-100의 CALC 실험이 직접 사용한다.
- 현재 다중블록 이송은 중앙 혼잡도 규칙이 결정하며, 이 공식은 Transfer quote 경로에
  **아직 연결되지 않았다**.
- 트럭 비용은 엔진 시간적분에는 있으나, 작업 하나의 지연 한계비용을 반환하는 공용 API가 없다.

따라서 “Block Q와 TransferResolver가 이미 같은 계산기를 공유한다”는 설명은 목표 구조이지
현행 코드가 아니다.

## 목적

본선과 트럭 작업을 같은 numeraire(공통 비용 단위)로 비교하는 단일 API를 만든다.

```text
marginal_delay_cost(job, Δt ∈ {1, 5, 10분}, observation)
→ {cost, uncertainty, escalation_start}
```

YR-133에서 같은 API·시간창·종결 규칙으로 다음 값을 계산할 수 있어야 한다.

```text
OutRelief = source KEEP 비용 - source REMOVE 비용
InBurden  = receiver ADD 비용 - receiver NO-ADD 비용
```

## 계약

1. source와 receiver가 같은 기준시각·비용단위·평가창·미완 잔여비용을 사용한다.
2. 예약·공개 ETA·현재 상태만 사용하고 실제 미래 실현은 읽지 않는다.
3. 유형별 내부 계산은 허용하되 출력 단위는 하나다. 원시 작업종류·마감값을 Block Q 상태에서
   제거할지는 이 작업이 아니라 YR-124의 별도 비교로 결정한다.
4. 값뿐 아니라 불확실성과 급증 시작시점을 반환해 판매 안전여유를 사전등록할 수 있게 한다.
5. 분해 항등·단조성·단위변환·결측·결정론을 테스트로 고정한다.

## 산출물

- 공통 비용곡선 모듈과 API
- YR-100 CALC와 YR-133 source/receiver quote의 실제 연결
- 대표 작업별 지연 한계비용 곡선과 단위·가정 evidence
- YR-124 상태표현 C안의 입력 재료


## 구현 결과 (2026-07-30)

- [cost_curve.py](../../../src/yard_rl/integrated/cost_curve.py):
  `delay_cost_curve(sim, job_id, Δt)` → cost·불확실성 대역(lo/hi)·급증 시작 지연.
  유형별 정의는 전부 **현행 동결 비용계약에서 유도** (새 가격 발명 없음): 트럭 = 대기
  1.0/h + SLA 초과 2.0/h(급증 시작 = 남은 여유), 미도착 = 하한 0/ETA 결측 fail-closed,
  본선 LOAD = vessel_cost paired 반사실(κ 대역·ρ 민감도 의무 승계), 양하·기타 = 0 명시.
- 계약 ⑤ 고정: [tests 11](../../../tests/integrated/test_yr123_cost_curve.py).
- evidence: [대표 곡선](../../../outputs/reports/yr123_cost_curve/report.md).
- **산출물 2(quote 실제 연결)는 YR-133 의 몫으로 이월** (7차 피드백 역할 갱신과 정합 —
  이 모듈은 그 견적의 공용 원료·YR-132 식 개수 보정 용도 금지).
