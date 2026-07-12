# YR-006 — Phase 2: 단일 YC 이벤트 시뮬레이터

- **Epic**: Sim / **Priority**: 🟠 / **등록일**: 2026-07-12
- **배경**: [02 §1](../../../docs/구현계획/02_시뮬레이터_RL환경.md) 이벤트 종류·동시 이벤트 우선순위·non-preemptive 원칙, 02 §2 이동·서비스시간 모델 (축별 합/phase 모델, 적재·무부하 구분).
- **목표(수용 기준)**: FIFO 정책으로 한 운영일 종단실행 성공 + 고정 시나리오·고정 seed 재현성 ([05 §4 Phase 2](../../../docs/구현계획/05_테스트_로드맵_산출물.md)). 이벤트 시각 단조 증가.
- **범위 밖(non-goal)**: 다중 YC 간섭(→ YR-013), 실측 분포 보정(→ YR-009).
- **계획**: EventQueue → YardState·Stack → CraneResource·travel_time → 작업 transition → 외부트럭 큐·KPI 집계.
- **산출물**: `src/yard_rl/sim/` (engine·events·yard_state·crane·stack·travel_time).
- **의존**: YR-004 도메인 모델.
