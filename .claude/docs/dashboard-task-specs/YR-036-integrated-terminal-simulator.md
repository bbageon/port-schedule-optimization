# YR-036 — 통합 터미널 이벤트 시뮬레이터

- **Epic**: Sim / **Priority**: 🔴 / **등록일**: 2026-07-15
- **배경**: 현재 단일 YC·외부트럭 환경으로는 본선 우선, 이송장비 대기, 레인 충돌과 공동 작업배정을 평가할 수 없다.
- **목표(수용 기준)**: 외부트럭·본선·STS·YT/AGV/SC·레인·다중 YC 이벤트를 같은 시계에서 처리하고, 동일 seed 결정론·구간 비용 적분·clear-out·정보시점 검사를 통과한다.
- **범위 밖**: 실제 TOS 연동, 미확보 확률분포의 임의 확정, 정책 학습.
- **계획**: YR-035 fixture 구동 → 본선/이송/레인 이벤트 → 다중 YC 자원예약 → 장애·계획변경 → golden scenario.
- **산출물**: 통합 simulator 모듈, synthetic terminal fixtures, invariant·golden tests.
- **의존**: YR-035. 실측 validation은 YR-002·YR-009.
