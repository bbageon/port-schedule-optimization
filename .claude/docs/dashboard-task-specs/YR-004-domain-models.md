# YR-004 — Phase 1 전반: 도메인 모델

- **Epic**: Data / **Priority**: 🟠 / **등록일**: 2026-07-12
- **배경**: [01 §5](../../../docs/구현계획/01_범위_아키텍처_데이터.md) — JobFlow·LoadStatus 등 Enum 과 ContainerState·Job·CraneState·TruckState 객체. 반입/반출과 공/적을 하나의 Enum 으로 합치지 않는 원칙.
- **목표(수용 기준)**: 01 §5 의 Enum·객체·validator 전부 구현 + schema validation unit test 통과 ([05 §1.1](../../../docs/구현계획/05_테스트_로드맵_산출물.md)).
- **범위 밖(non-goal)**: 원천자료 전처리 파이프라인(→ YR-005).
- **계획**: enums.py → models.py (dataclass/Pydantic) → validators.py → tests/unit.
- **산출물**: `src/yard_rl/domain/`, 단위 테스트.
