# v6 조각 3 구현 명세 — PRE_ADVICE 정보수준 — ETA wake·armed·eta_opportunity·PRE_REHANDLE/REPOSITION 후보·DEFER wake

원본: outputs/v6/port_plan_wf_9e7366e6.json (plan.pieces[2])

files_to_port:
- engine.py:159-179, 339-382 (wake 스케줄·소비)
- engine.py:453-464 (armed 게이트), 563-576 (_plan REPOSITION)
- integrated/candidates.py:50-203 (iter_pre_rehandle_jobs·iter_eta_reposition_*·eta_opportunity 마스크판), 334-363 (_escape_bays), 371-456 (_reposition·_wait DEFER)
- integrated/policy_config.py (wait_mode·bool 4개 → static 인자)

hard_parts:
- wake 는 큐 밖 데이터라 스텝 사슬에 국면 W 추가 + `advance(wt)` 로 빈 구간 시계 전진(engine.py:333-335)
- 후보 종류가 SERVE 외 PRE/REPO/WAIT 로 늘어 후보 행렬이 (K, N + R_max + 1) 이 됨 — REPO 목표 bay 집합(set→sorted, 407행)은 정렬+인접중복 제거 고정 길이
- yielded/armed 의 소진 규칙(297, 380-381행)이 결정 수를 정한다 — 한 스텝에 wake 소비와 결정을 같이 하면 v5 와 결정 횟수가 달라짐(별 스텝으로)

array_technique:
eta_wake_s (W,) 정렬 고정 + idx; armed (K,) bool; 후보 행렬을 종류 열·목표 bay 열로 확장; PRE 게이트는 (N,) 마스크 AND (flow==GATE_OUT & PLANNED & 대상 실재·가용·구간 내 & n_block>0 & cap_ok & 0<eta−clock≤horizon).

equivalence_test:
조각 2 무대에 provided_eta=actual 도착 부여, info_level=PRE_ADVICE. v5 정책 = CentralResolver(BaselinePreference) (test_yr050:50-61 _drive). 비교 = 사건열에 ETA_WAKE 포함 정확, 결정 시각열(wake 시각 700=2500−1800 등) 정확, PRE_REHANDLE 발생 시각·kpis.rehandle_count vs job.rehandle_count 분리(test_yr050:69-76 규약), WAIT 만 하는 정책으로 결정 수 유한·엄격 증가(test_yr050:147-161).

예상 규모: ~250줄

의존: ['2']

## 관련 event_kinds (설계)