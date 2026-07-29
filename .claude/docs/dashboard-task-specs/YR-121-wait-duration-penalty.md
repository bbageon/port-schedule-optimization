# YR-121 — 2차 단일축: WAIT 지속시간 벌점 1.0/h
> 상태: done 2026-07-29 · verdict: 기각 · 1세대식 학습 진단
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 설계
유일 차이 = WAIT_TIME_PENALTY 1.0(의도적 유휴 1h = 트럭 대기 1h 앵커). 대조군 = YR-119
WAITON 동결 산물. 경계: 열거 조합의 WAIT(구조적 포함)가 대상, assigns 빈 강제 WAIT 제외.
## 판정
전략적 WAIT 사실상 불변(0.43~0.48)·총비용 동등·A→O 악화 → 기각. 사후 추정: 1h 유휴의
할인 이득 ≈3~7 — 벌점이 3~7배 부족(사전등록 아님·튜닝 금지 준수).
## Evidence
[report](../../../outputs/reports/yr121_wait_duration_penalty/report.md) · `5cb48cf`+`c30a390`
