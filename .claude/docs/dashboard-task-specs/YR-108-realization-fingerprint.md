# YR-108 — 실현 지문 기반 시드 대역 배정
> 상태: done 2026-07-28 · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 원칙
실현 정체성은 시드·설정이 아니라 **생성물 내용**으로만 결정된다(마감 배율은 난수 미소비 —
배율×정합 10조합 = 지문 1개). 같은 부하수준 셀 BASE 200 간격 공명이 사고 기전.
## 조치
`seedbank.realization_hash/assign_band/independence_report` — 대역 간 교집합 금지·열람
대역 재사용 금지. 자체 정정: 셀 간 시드 커서 공유(블록 독립성) 07-28.
## Evidence
[seedbank](../../../src/yard_rl/integrated/seedbank.py) · [tests](../../../tests/integrated/test_yr106b_gates.py)
