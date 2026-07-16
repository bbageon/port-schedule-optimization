# YR-045 — 정정판 locked 재실험

- **목표**: YR-043 비용과 YR-044 baseline 위에서 Candidate DQN/DDQN/Dueling을 신규 seed로 다시
  평가하고, ETA 위치선점과 선제 재조작의 기여를 분리한다.
- **선결조건**: YR-050 완료, clean commit, Windows·WSL 검증 통과.
- **동결 원본**:
  [사전등록](../strategy-history/2026-07-16-YR-045-corrected-locked-rerun-prereg.md).
- **핵심 비교**: `NO_ETA / ETA_NO_PRE / FULL` 3-arm + JointRolloutGreedy·BeamLookahead·
  ServiceFirstSPT·FIFO.
- **수용 기준**: 평균대기 개선과 P95·본선·STS·이송 비악화, 이동/재조작 1개 이상 개선,
  완료 100%, backlog·위반 0, 비용 지배도·행동분포 계약 통과.
- **보고 의무**: 비용 기여율, 행동 4종, ETA 경로별 후보·선택, 후보 조합 축소 횟수,
  seed별 paired 원자료와 95% 신뢰구간.
