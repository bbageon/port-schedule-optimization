# YR-141 — v4-B: 구속적 PREPOSITION 단일축 (15차 확장 판정 7항)
> 상태: **in-progress (2026-08-02 — 등록된 조건부 분기 발동·사용자 승인 순서)**
> 근거: [YR-140](../../../outputs/reports/yr140_ppo_unitfix/report.md) — 장악 1/36 잔존 +
> 재배치 비중 0.25~0.40 + 본선 초과 악화(트럭↔본선 교환). 목표 = **트럭 이득 보존 +
> 본선·이동 손실 방지 + 재배치 구속** (재배치 감소만이 아님 — 15차).

## 계약 ([하네스](../../../src/yard_rl/experiments/yr141_bound_prepo.py) docstring 이 동결 정본)

- 유일 변경 = candidates.BOUND_REPO=True (opt-in — 기본 False 는 회귀망 39 테스트로
  바이트 안전 확인): REPO:<crane>:<bay> → **PREPO:<jid>:<bay>** 결속·근접(≤1bay) 소멸·
  ETA/도착 만료 내재·교착 탈출 REPO 안전기능 분리. 반복 잔여 가능성은 판정 ⑦이 0 요구.
- 비교군 3 = SF / v4-A(YR-140 체크포인트 재사용·flag off) / v4-B(신규 학습·flag on).
  평가 = 미열람 BASE+3200..3202 (12ep) + **시나리오 실현 지문 동봉**.
- 판정 7항 (전부 = 성공): J1 완주 100%∧backlog 0 · J2 장악 0 · J3 vs SF v2 방향 ≥2/3 ·
  J4 (B−A) v2 ≤0 ≥2/3 · J5 (B−A) 본선 초과 ≤0 ≥2/3 · J6 (B−A) v1 ≤0 ≥2/3 ·
  J7 같은 작업 반복 이동 0 ∧ 만료(도착) 후 이동 0 (실행 계수).
- 성공 → 잠금평가(신규 시드·신뢰구간·독립성 계정) 사전등록. 실패 → 실패 항목별 보고
  (트랙 중단 논의는 J3 상실 시).

## Evidence
사전동결: (커밋 예정) · 결과: outputs/reports/yr141_bound_prepo/
