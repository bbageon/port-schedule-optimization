# YR-116 — 후보·리뷰창 계약 완결 (1 bay 탈출·gate-in 0초)
> 상태: done 2026-07-28 (병렬 세션) · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 발견 2
①YR-112 의 1 bay 탈출 목표가 일반 미세이동 필터(≤1 bay)에서 재삭제 — "종단 검증" 과장.
②`0 < actual_gate_in` 이 t=0 반입을 review 에서 영구 누락(YR-113 원자료 선택 40·확증 77건).
## 조치
탈출 목표만 하한 면제 · gate-in 범위 `0 ≤ a ≤ end` · 표적 회귀 35 · YR-113 민감도 재검
(결론 유지 확인은 YR-117 정본 재실행에서).
## Evidence
[report](../../../outputs/reports/yr116_candidate_review_contract/report.md) · `942c413`
