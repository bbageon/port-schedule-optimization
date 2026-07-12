# YR-008 — Phase 3: Baseline 정책 + KPI·paired runner

- **Epic**: Baseline / **Priority**: 🟠 / **등록일**: 2026-07-12
- **배경**: [03 §1.1](../../../docs/구현계획/03_정책_실험_평가.md) 필수 Baseline — CURRENT_RULE·FIFO·LONGEST_WAIT·NEAREST_JOB·MIN_REHANDLE·FIXED_WEIGHT_RULE(+소규모 oracle). **CURRENT_RULE 미확보 시 다른 Baseline 을 "실제 운영 대비 개선율"로 표현 금지.**
- **목표(수용 기준)**: [05 §4 Phase 3](../../../docs/구현계획/05_테스트_로드맵_산출물.md) — 모든 Baseline 이 같은 사건열(공통난수·동일 초기상태)에서 실행되고 결과가 설명 가능. weighted heuristic 튜닝 포함.
- **범위 밖(non-goal)**: 학습 정책(→ YR-010).
- **계획**: policies/baselines.py → 공통 KPI·Reward 집계 → paired runner (03 §2.3 기록 항목 저장).
- **산출물**: `src/yard_rl/policies/baselines.py`, `experiments/runner·metrics`, 정책별 비교표.
- **의존**: YR-006·007.
