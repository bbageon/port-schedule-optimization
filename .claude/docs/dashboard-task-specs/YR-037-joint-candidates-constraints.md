# YR-037 — 동적 후보·공동 Action·Hard Constraint

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-15
- **배경**: 최종 Action은 큐 등록이나 규칙 번호가 아니라 각 YC의 실행 가능한 작업이며, 동일 Job·레인·비통과 제약은 공동으로 검사해야 한다.
- **목표(수용 기준)**: Top-K가 아닌 mandatory 보존형 가변 후보, padding mask, `SERVE/PRE_REHANDLE/REPOSITION/WAIT` 계약과 중앙 joint resolver를 구현한다. 불가능 행동 선택·중복 Job·레인 충돌·비통과 위반 0건을 테스트로 보장한다.
- **범위 밖**: Q-network 학습, 비용 가중치 선정, 물리적 레일 변경.
- **계획**: 후보 생성 → 이중 constraint 검사 → deterministic pruning/tie-break → feasible matching → deadlock yield.
- **산출물**: candidate generator, constraint extensions, joint assignment baseline, 감사 로그.
- **의존**: YR-035·YR-036. SLA 임박 보호는 YR-029 계약을 흡수한다.
