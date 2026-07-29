# YR-099-b — 다중블록 재배정 브리지 계약
> 상태: done 2026-07-27(+정정 07-28) · 2세대 기반
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 목적
MVP 근사 6종을 계약으로 교체 — 판매 창 재배정이 실제 resolver 구조에서 재현되게 하는 기반.
## 계약 6
①정확 gate-in epoch(엔진 `review_epochs` opt-in) ②공용 시계(전 블록 동일 epoch park)
③전역 A→O 장부(TerminalLedger) ④canonical id 불변+owner/version/transfer_history
⑤prepare→validate→commit/rollback 원자성 ⑥수신 용량검사.
## 결과·정정
적대검증 critical 2(epoch 선점→기준선 24~32% 팽창 / route 비용 미계상)·major 4 정정.
게이트 D(YR-106-b)에서 고아 job·txn 재사용 추가 봉합. 골든: epoch 깔고 이송 0 → 바이트 동일.
## Evidence
[구현](../../../src/yard_rl/integrated/multiblock.py) · [tests](../../../tests/integrated/test_yr099b_bridge_contract.py) · done row `a71b7fe`·`e4b303a`
