# YR-056 — 협조 신호 경량 실험 (COORD feature, itc-v3)

- **Epic**: RL / **Priority**: 🟡 / **등록일**: 2026-07-18 (YR-054 권고 파생) / **착수**: 2026-07-18
- **배경**: [YR-054 재감사](../YR-054-RL-재감사.md) — RL 게이트 전패의 85%가 interference(경합 양보).
  원인은 "상대 크레인이 다음에 무엇을 할지"가 어떤 feature 로도 주어지지 않는 구조 공백.
  QMIX(YR-013) 전에 **feature 만으로** 격차가 줄어드는지 확인하는 경량책.
- **목표(수용 기준)**: 동일 학습예산 dueling 2-arm (COORD on / ablation off=itc-v2 동일 정보)
  paired 비교에서 ① total_cost·interference 차이의 CI 를 보고하고 ② 방향을 판정한다.
  줄면 QMIX 없이 부분해소 경로 확보, 안 줄면 "feature 로는 부족 — 구조(QMIX) 필요" 근거.
- **범위 밖(non-goal)**: QMIX·mixer 구조 변경, 상대 행동 예측 모델, baseline 정의 변경.
- **계획**: itc-v3 스키마(COORD 5종) → adapter 산출(관측 사실만) → run_episode ablation arm
  → dueling 500ep × 2 arm + JointRollout(forbid) 60 test seeds (신규 대역 500k/510k/520k).
- **산출물**: [매핑](../YR-056-협조feature-매핑.md) · `experiments/yr056_coord_experiment.py`
  · `outputs/reports/yr056_coord/` · `tests/integrated/test_yr056_coord.py`
- **의존**: YR-052(전략 WAIT 제외 — 완료), itc-v3 golden 재생성.
