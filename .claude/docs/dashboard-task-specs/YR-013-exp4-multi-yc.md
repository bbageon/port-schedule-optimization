# YR-013 — 중앙 공동배정·QMIX 다중 YC 협조

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-12 / **재기준화**: 2026-07-15
- **배경**: [최종전략 전환](../strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md)·[06](../../../docs/구현계획/06_동적후보_Deep_Q_다중YC.md) — 검증된 Candidate Double DQN utility와 중앙 공동 matching을 기준선으로 두고 QMIX의 추가효과를 분리한다.
- **목표(수용 기준)**: 실제 공통 서비스영역과 joint mask를 반영하고 동일 Job·레인충돌·비통과·안전거리 위반 0을 보장한다. 중앙 matching 대비 QMIX의 locked 총비용·P95·본선·이송·교착 KPI를 paired 비교한다.
- **범위 밖(non-goal)**: 수직형·수평형 터미널 혼합, 물리적 레일 재배치.
- **계획**: YR-037 resolver 기준선 → YR-039 local utility 공유 → monotonic mixer → Double joint target → central-vs-QMIX ablation.
- **산출물**: `joint_assignment.py`, `qmix.py`, 협조 학습 runner·리포트·UI replay.
- **의존**: YR-036~039, 실제 서비스영역 자료 YR-002.
- **⚠ 착수 조건 미충족 회귀 (2026-07-15)**: 계획의 전제 "검증된 YR-039 local
  utility" 가 [무효 판정](../strategy-history/2026-07-15-YR-039-무효판정-imbalance-지배.md)
  (imbalance 지배 reward·퇴화 baseline). YR-043(목적함수 정정)·YR-044(baseline)·
  YR-045(정정 재실험) 통과 후의 utility 로 재판정한다.
