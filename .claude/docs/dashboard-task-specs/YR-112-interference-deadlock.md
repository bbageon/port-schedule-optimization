# YR-112(+b) — 크레인 간섭 교착: 활성 결함 해소 + 탈출 계약
> 상태: done 2026-07-28 · 측정계약 축(엔진 활성)
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 결함
두 유휴 크레인이 대상 통로를 안전간격 안에서 가리면 비켜설 결정 기회가 안 열려 작업이
미완으로 끝남(표본 6% 오염). 드레인 3배도 동일 = 활성 결함.
## 조치
술어 `interference_deadlock_corridors`(엔진·생성기 공유) · 탈출 결정(통로 기준 최소거리
재배치·1 bay 면제) · escape_mode=immediate 확정(기존 대역 32런 발화 0=골든 안전) ·
YR-050 결정시각 엄격증가 준수 · arm 별 deadlock_escapes 박제.
**+b**: "미완 비용 미계상" 서술 철회(방향 반대 — 더 비싸짐)·계약민감도 4조합 결론 동일.
## Evidence
[report](../../../outputs/reports/yr112_interference_deadlock/report.md) · [tests 18](../../../tests/integrated/test_yr112_interference_deadlock.py)
