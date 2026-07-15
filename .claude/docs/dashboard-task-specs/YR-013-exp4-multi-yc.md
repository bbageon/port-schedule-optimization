# YR-013 — Phase 8: Exp-4 다중 YC 협조

- **Epic**: Exp / **Priority**: ⚪ / **등록일**: 2026-07-12
- **배경**: [03 §1.3](../../../docs/구현계획/03_정책_실험_평가.md)·[06](../../../docs/구현계획/06_동적후보_Deep_Q_다중YC.md) 구현 계약 — 동적 후보 Candidate Double DQN → 중앙 공동 matching → QMIX CTDE 추가효과 순으로 분리한다. 처음부터 검증되지 않은 로컬 평가망과 mixer를 함께 도입하지 않는다. [02 §8](../../../docs/구현계획/02_시뮬레이터_RL환경.md) 비통과·안전거리·인계지점 점유.
- **목표(수용 기준)**: [05 §4 Phase 8](../../../docs/구현계획/05_테스트_로드맵_산출물.md) — 실제 공통 서비스영역 구조 반영 + **간섭 제약 위반 0**. 간섭·양보·교착 테스트 및 UI 표시(YR-015 UI-5) 포함. 단일 정책 대비 공동 KPI 비교 (H4 판정).
- **범위 밖(non-goal)**: 수직형·수평형 터미널 혼합, 물리적 레일 재배치.
- **계획**: YC별 동적 후보 tensor·mask → Candidate DQN/Double/Dueling 비교 → 중앙 joint-action resolver → QMIX paired 비교. 학습은 GPU 권장·CPU fallback, 운영 추론은 CPU 계약.
- **산출물**: `candidate_q_network.py`, `candidate_double_dqn.py`, `joint_assignment.py`, `qmix.py`, Exp-4 리포트.
- **의존**: YR-011, 실제 서비스영역 자료(YR-002).
