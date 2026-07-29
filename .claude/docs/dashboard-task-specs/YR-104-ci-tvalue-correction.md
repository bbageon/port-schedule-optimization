# YR-104 — 공용 신뢰구간 t계수 정정·소급 재계산
> 상태: done 2026-07-27 · verdict: 결론 변경 0건 · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 목적
`_ci` 가 df=19 외 fallback 2.1 을 쓰던 결함(df≤18 반보수) 정정 + 전 판정 소급 재계산.
## 결과
df 1~29 t-table + 보수 fallback. 소급: YR-099-G1·101·041-a·098 전부 결론 불변.
## Evidence
[report](../../../outputs/reports/yr104_ci_correction/report.md) · `4888540`
