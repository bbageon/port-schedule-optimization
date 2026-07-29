# YR-109 — 본선 마감 물리 정합
> 상태: done 2026-07-28 (+106-b 게이트A 확장) · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 문제
dmult<1 은 야드가 무한히 빨라도 못 지키는 상수 선석초과(33~38분)를 만든다 —
slack<0 이 초기조건이 되어 본선 신호 퇴화.
## 결과
opt-in `vessel_deadline_achievable`. **착수 근거("분산↓→필요 n↓")는 자체 반증** —
상수는 짝지은 차이에서 상쇄. 실제 성과 = 타당성(slack 창발화·본선 채널 부호 반전).
게이트A 에서 STS 단독→YC→YT→STS 전체 사슬로 확장.
## Evidence
[report](../../../outputs/reports/yr109_deadline_physics/report.md) · [재현 하네스](../../../src/yard_rl/experiments/yr109_deadline_physics.py) · `1952a6a`
