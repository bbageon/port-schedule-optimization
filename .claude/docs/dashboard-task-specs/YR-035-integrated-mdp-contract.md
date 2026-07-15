# YR-035 — 최종 통합 MDP·데이터 계약과 수직절편

- **Epic**: Data / **Priority**: 🔴 / **등록일**: 2026-07-15
- **배경**: [최종전략 전환](../strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md)은 별도 Exp 정책 대신 차량·본선·이송장비·레인·다중 YC를 처음부터 같은 계약으로 다루도록 결정했다.
- **목표(수용 기준)**: Global State·Local Observation·가변 Candidate·Mask·Joint Action·구간 Cost schema를 버전 고정한다. 모든 최종 도메인이 synthetic fixture와 missing/assumed 표기를 가지며, 한 결정 transition을 직렬화·복원하는 테스트가 통과한다.
- **범위 밖**: 네트워크 성능, 실자료 수치 확정, ETA 예측, 자동 장비제어.
- **계획**: 데이터 source/time-of-knowledge 표 → tensor·ID 감사필드 schema → 최소 통합 fixture → 정보누출·단위·결측 validation.
- **산출물**: 통합 MDP/데이터 매핑 문서, schema·fixture·contract tests.
- **의존**: 합성 계약은 즉시 가능. 실제 범위·분포 보정과 운영 주장은 YR-002·YR-009 이후.
