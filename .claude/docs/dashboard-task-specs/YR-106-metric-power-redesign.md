# YR-106 — 판정 지표·검정력 재설계
> 상태: done 2026-07-28 (+106-b·117 정정) · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 목적
총비용 분산의 70%가 본선이라 n=8 판정이 전부 "검출 실패"였음을 교정 — 채널 분해·MDE·라벨.
## 내용
truck/vessel/move/other 완전분할 · MDE 동반 · 유의/효과없음/검출실패/미검출 분리 ·
guard 기계화 · 확증 n≥선택×2. **주의**: 트럭 채널 1차 승격은 YR-117 에서 계약 이탈로
정정(주판정 = 총비용+A→O 복귀).
## Evidence
[evalkit](../../../src/yard_rl/integrated/evalkit.py) · [report](../../../outputs/reports/yr106_metric_power/report.md) · `7372892`
