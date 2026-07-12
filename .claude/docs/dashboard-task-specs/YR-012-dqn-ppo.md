# YR-012 — Phase 7: Masked DQN/PPO 함수근사

- **Epic**: RL / **Priority**: ⚪ / **등록일**: 2026-07-12
- **배경**: [02 §5.2](../../../docs/구현계획/02_시뮬레이터_RL환경.md) 함수근사 관측값(global·crane·candidate features + mask). 착수는 필요성 판단 후에만 — Tabular 의 방문율 부족·bucket 정보손실·다중 후보 처리 한계가 확인될 때.
- **목표(수용 기준)**: [05 §4 Phase 7](../../../docs/구현계획/05_테스트_로드맵_산출물.md) — **복잡도 증가를 정당화할 일관된 out-of-sample 개선**. 동일 reward·constraint·실험조건으로 Tabular·강한 heuristic 대비 평가. 복수 seed 안정성 + 추론시간 기준 충족.
- **범위 밖(non-goal)**: 다중 agent (→ YR-013 에서 필요 시), reward 구조 변경.
- **계획**: 후보집합 observation encoding → action mask 지원 학습 루프 → validation 기반 선택 → paired 비교.
- **산출물**: `src/yard_rl/policies/dqn_policy.py`(또는 PPO), 비교 리포트.
- **착수조건**: YR-010 결과에서 전환조건 확인 후 ready 승격.
