# YR-123 — 공통 작업비용 계산기 통일 + 지연 한계비용 곡선 API

> 상태: ready · 2세대 · 설계 정정 2026-07-30
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
