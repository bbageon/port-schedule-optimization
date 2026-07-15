# YR-012-c — 고정 대기열 요약 feature 검증

- **Epic**: RL / **Priority**: 🟡 / **등록일**: 2026-07-15
- **배경**: [YR-031-b 결과](../strategy-history/2026-07-15-YR-031-b-oracle-pattern-prereg.md)는 이탈 시점 예측 AUC 0.852를 확인했고, 상위 신호가 현 Δ-net에 없는 대기열 요약이었다. 후보쌍만으로 선택 AUC 0.993이므로 학습형 집합 인코더 필요성은 기각됐다.
- **목표(수용 기준)**: 기존 14개 입력에 사전등록한 고정 요약 8개를 추가하고 동일 online-TD·seed·비용으로 locked test. greedy 대비 평균대기 CI, P95·완료·backlog·불변조건과 이탈구간 회수율을 보고한다.
- **범위 밖**: QMIX, 본선·다중 YC, attention/set encoder, replay 재도입, test 기반 feature 선택.
- **계획**: feature 공식·정보시점 동결 → zero/scale·permutation 테스트 → 동일 예산 재학습 → YR-012와 paired 비교.
- **산출물**: context-feature agent 설정·리포트·사전등록. 최종 통합망 YR-039의 feature 근거로만 사용한다.

