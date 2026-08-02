# YR-140 — v4-A-fix: PPO 단위 계약 정정 + 신규 시드 재실행
> 상태: **in-progress (2026-08-02 — 14차 피드백·사용자 지시 착수)**
> 근거: [YR-139 정정](../../../outputs/reports/yr139_blockq_v4_ppo/report.md) — GAE 단위
> 혼합(원 단위 r + 1/20 단위 V) 확정. 실험 성립 조건 수리 — 튜닝 아님.

## 계약

- **수정 = 단위 통일 하나**: 보상·가치·GAE·returns 전부 /SCALE 단위 (_gae 내부 변환,
  가치 손실 이중 스케일 제거). 그 외 v4-A 와 동일(행동·상태·하이퍼·60 iter × 8 ep).
- **단위 테스트 2건 고정**: ①가치가 scaled 미래수익을 정확히 맞히면 advantage ≈ 0
  (구판은 실패) ②비용 낮은 행동 1회 학습 → 그 행동 확률 증가. + 기존 등식 테스트 3.
- 평가 = **신규 미열람 대역 BASE+2900..2902** (2600 열람됨). 판정 = v4-A 와 동일:
  완주 100%∧backlog 0 ∧ ≥2/3 초기화 비용 감소 방향 ∧ WAIT·REPO 장악 0.
- 분기(동결): REPO 장악 재현 시에만 v4-B(PREPOSITION(job_id, target_bay, expires_at,
  eta_version) — 같은 작업 반복 이동 금지·도착/취소/ETA 변경 시 만료·목표 도착 후 소멸·
  교착 REPO 는 안전기능으로 분리) 1회 → 그래도 실패면 **트랙 중단 확정**.

## Evidence
사전동결: (커밋 예정) · 결과: outputs/reports/yr140_ppo_unitfix/
