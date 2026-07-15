# YR-039 — 동적 후보 Candidate Double DQN

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-15
- **배경**: Q-table은 최종 연속 State·가변 후보를 저장하기 어렵다. YR-031-b는 대기열 요약이 필요하지만 후보별 독립 점수 구조는 충분하다고 판정했다.
- **목표(수용 기준)**: `[Global, YC, Candidate, 고정 Queue Summary] → Q_cost` 공유망을 구현하고 Candidate DQN→Double DQN→Dueling을 같은 조건에서 비교한다. 기본 후보는 Double DQN이며 locked test에서 강한 동정보 휴리스틱 대비 총비용 CI와 모든 guardrail을 보고한다.
- **범위 밖**: QMIX mixer, test 기반 튜닝, Job ID 입력, 안전조건의 보상학습.
- **계획**: permutation-invariant 고정요약 → replay/target DDQN → masked batch → GPU/CPU parity → checkpoint protocol YR-033 적용.
- **산출물**: candidate Q-network/learner, device-independent checkpoint, paired report.
- **의존**: YR-037·YR-038. YR-012-c 결과는 feature 선택 근거로 사용하되 최종 test와 분리한다.
