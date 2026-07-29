# YR-106-b — YR-105-b 착수 4게이트 (A 물리하한·B 재현성·C 통계·D 원자성)
> 상태: done 2026-07-28 · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 내용
A: 마감 물리하한을 YC→YT→STS 전체 사슬로(적하 145.1s 잔존 제거, max() 클램프).
B: 재현 스탬프(repro.py)·CLI 화·시드별 원자료(YR-109 원자료 생성 스크립트 부재 적발).
C: statfuncs 정확 t/χ² · 필요 n 반복수렴+sd 상측한계("트럭 17"은 df 고정 오류→24) · TOST.
D: 브리지 commit 원자성(고아 job)·txn id·provided_eta 동반 이동.
## Evidence
[report](../../../outputs/reports/yr106b_gates/report.md) · [tests](../../../tests/integrated/test_yr106b_gates.py) · `cad55c1`
