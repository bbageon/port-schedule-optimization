# YR-007 — Phase 2: SafetyConstraintEngine + invariant 테스트

- **Epic**: Sim / **Priority**: 🟠 / **등록일**: 2026-07-12
- **배경**: [02 §8](../../../docs/구현계획/02_시뮬레이터_RL환경.md) YC·컨테이너/슬롯·작업 제약을 중앙 관리. 제약위반 행동은 후보 생성 단계와 실행 직전 **2중 차단**. 안전은 학습 대상이 아니라 항상 지키는 규칙.
- **목표(수용 기준)**: 매 이벤트 후 상태 불변조건 자동검사 (02 §1.3) 전부 통과, 제약 위반 0. [05 §1.2](../../../docs/구현계획/05_테스트_로드맵_산출물.md) invariant 목록 (컨테이너 위치 유일성·tier 연속성·단일 할당·서비스영역·Hold·비통과·안전거리·**미공개 작업 유출 방지**) 테스트 구현.
- **범위 밖(non-goal)**: soft penalty 방식의 안전 처리 (금지 원칙).
- **계획**: constraints.py (제약 판정) → 시뮬레이터 이벤트 루프에 검사 훅 → tests/invariants/.
- **산출물**: `src/yard_rl/sim/constraints.py`, `tests/invariants/`.
- **의존**: YR-006 과 동반 개발.
