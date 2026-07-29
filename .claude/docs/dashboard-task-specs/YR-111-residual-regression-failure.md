# YR-111 — 선존재 회귀 실패 판별
> 상태: ready · 1세대 코드 잔재 점검
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 목적
`test_update_regresses_toward_residual_target` 이 400 step 후 −2.22(목표 −2.5±0.15)로
실패 — **수렴 미달 vs 잔차 회귀식 결함 판별**. 잔차 Δ 학습(YR-012·YR-102) 사용 전 필수.
## 계획
step 사다리(400/2000/10000)·lr 민감도만으로 수렴 미달 여부 판정 → 미달이면 테스트 상수
정정, 아니면 YR-012 학습식 재검. 판정 기준을 결과 열람 전 동결.
## Evidence
ready row · tests/unit/test_residual_delta_net.py
