# YR-038 — 정규화 터미널 Total Cost·Reward

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-15
- **배경**: 트럭 대기만 줄이면 본선·이송장비 작업을 미룰 수 있다. 최종 Q 목표는 여러 손실을 정규화한 장기 누적 터미널 비용이다.
- **목표(수용 기준)**: 트럭/장기대기·YC 이동/빈이동·재조작·STS/이송대기·본선/출항지연·레인혼잡·간섭·순번변경·부하불균형의 raw delta, scale, weight, 합계를 한 구간에서 산출한다. 항목 중복계상 0, 구간합=에피소드 비용 항등식을 검증한다.
- **범위 밖**: 안전위반 벌점화, 합성자료만으로 가중치 확정, 회계원가라고 주장.
- **계획**: 비용 인과 ledger → train baseline scale 동결 → 정적/동적 본선계수 비교 → YR-026 민감도 흡수 → guardrail 분리.
- **산출물**: cost config/schema, reward calculator, cost identity·sensitivity report.
- **의존**: YR-035·YR-036. 실제 scale은 YR-002 이후 확정.

