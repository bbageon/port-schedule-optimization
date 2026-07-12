# YR-010 — Phase 4: Tabular Q-learning PoC

- **Epic**: RL / **Priority**: 🟡 / **등록일**: 2026-07-12
- **배경**: [02 §4~7](../../../docs/구현계획/02_시뮬레이터_RL환경.md) — SMDP 할인(`γ_τ`), 집계 bucket state(경계는 train 데이터로 결정 후 고정), Top-K 후보(mandatory 후보 포함), priority rule 9종(FIFO~WAIT_YIELD, `EARLIEST_PROVIDED_ARRIVAL` 포함) + Action Masking, 정규화 Core Cost. [03 §1.2](../../../docs/구현계획/03_정책_실험_평가.md) 사용 목적: 시뮬레이터·reward 검증 + rule 전환 가치 확인.
- **목표(수용 기준)**: [05 §4 Phase 4](../../../docs/구현계획/05_테스트_로드맵_산출물.md) — 학습 재현성, 미방문 상태 fallback 동작, 제약 위반 0. greedy(ε=0) 평가 재현 + 최소 1개 휴리스틱 대비 개선 또는 실패원인 분석.
- **범위 밖(non-goal)**: 함수근사(→ YR-012). ETA 정확도 평가.
- **계획**: 집계 state encoder → CandidateGenerator·ActionMask → SMDP Q update → validation checkpoint 선택.
- **산출물**: `src/yard_rl/policies/q_learning.py`, 학습 로그(03 §2.3 항목), 학습곡선.
- **의존**: YR-008 (Baseline 이 비교 기준).
