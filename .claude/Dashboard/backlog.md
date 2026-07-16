# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).
> 순서는 [최종전략 전환 이력](../docs/strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md)과 각 task spec의 의존관계가 기준이다. 과거 Phase/Exp 명칭은 완료 이력에만 남긴다.

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| YR-048 | Sim | PRE_REHANDLE(ETA 선제 재조작) 후보가 통합 실험에서 전혀 발생하지 않음 — `integrated/scenario_gen.py` 가 `provided_eta` 미설정 | 🟠 | **YR-047 적대 리뷰 파생 발견 (2026-07-16)**: 후보 생성기는 PRE_ADVICE + `job.provided_eta` 를 요구하는데(candidates.py:172) 통합 시나리오 생성기·fixture 는 provided_eta 를 설정하지 않는다 (설정처는 단일야드 `io/scenario_gen.py` 뿐). 실측: 3개 시나리오 후보 385건 중 PRE_REHANDLE **0건**. ETA 기반 선제정리는 연구 핵심 축(가설 H2·최종전략 §8.2)이므로 이대로 YR-045 를 돌리면 해당 축이 통째로 비활성인 채 판정하게 된다. 할 일: 통합 생성기에 ETA 분포(기존 gaussian 파라미터와 정합) 주입 + PRE_REHANDLE 발생율 검증 테스트 + YR-045 사전등록에 반영. **YR-045 착수 전 처리 권장** |
| YR-013 | RL | 중앙 공동배정·QMIX 다중 YC 협조 | 🟠 | [spec](../docs/dashboard-task-specs/YR-013-exp4-multi-yc.md) · **착수 조건 미충족 회귀 (2026-07-15)** — 전제였던 YR-039 로컬 utility 검증이 [무효](../docs/strategy-history/2026-07-15-YR-039-무효판정-imbalance-지배.md). YR-045 정정 재실험 통과 후 재판정 |
| YR-045 | Exp | YR-039 정정판 locked 재실험 — 신규 seed 대역 + §18 다중 게이트(평균대기↓·P95/본선/STS·이송 비악화·이동/재조작 1+ 개선·위반 0) + 항목별 기여율 보고 의무 | 🟠 | 기존 test 대역(320000~) 진단 사용으로 소각. 의존 YR-043·044. 총비용 CI 단독으로는 불채택 |
| YR-042 | Exp | DGT·HJNC 근사 프로파일 일반화 게이트 (재실행 대기) | 🟡 | **run 중단 (2026-07-15)** — YR-039 무효로 전제 상실 (동일 imbalance reward·동일 baseline 상속). 구현(`0cd547d`·`f51818c`)은 유효 — YR-045 통과 후 정정판 정책으로 재실행 |
| YR-029 | RL | P95 보호 — SLA 임박 후보 필터 | 🟠 | YR-018 negative 파생 · 보상형이 아닌 YR-037 mandatory 후보/명시적 제약으로 흡수 |
| YR-005 | Data | Phase 1 후반: 원천자료 loader·익명화·품질 플래그·날짜 split | 🟡 | [spec](../docs/dashboard-task-specs/YR-005-data-pipeline.md) · 실자료는 YR-002 후 |
| YR-009 | Sim | Phase 2 게이트: 시뮬레이터 실측 validation | 🟡 | [spec](../docs/dashboard-task-specs/YR-009-simulator-validation.md) · 실자료 의존, 미충족 시 RL 평가 금지 |
| YR-019 | Exp | ETA 품질 시나리오 매트릭스 (PERFECT/BIASED/NO_SHOW/STALE) | 🟡 | §18.2 — 현재는 EMPIRICAL(±300s)만 구현. Exp-3 결과의 강건성 확인용 |
| YR-020 | RL | Exp-2/3 열세 원인 분석 — 상태공간 희석 vs 정보 무익 판별 | 🟡 | [수렴진단](../docs/YR-020-수렴진단-2026-07-14.md): 희석 방문통계 증거 확보. **YR-030 전환 결정으로 "함수근사 판단재료" 역할 종결** — 학술적 원인 규명 가치로만 유지 (🟠→🟡, 2026-07-14) |
| YR-021 | Exp | 부하조건별 통합정책 강건성 — peak·고장치율·고재조작 | 🟡 | 혼잡일 상금 편중(YR-031/031-b) 파생 · YR-014 부하 ablation 입력 |
| YR-041 | Exp | 본선 위험도 λ_vessel 설계 실험 — 정적 {1.0, 2.5, 6.0} × 동적 밴드 paired (고부하·타이트 deadline 시나리오 강화 포함) | 🟡 | 사용자 결정 파생 (2026-07-15): 그 전까지 통합 실험은 [정적 중간값 2.5](../docs/strategy-history/2026-07-15-vessel-lambda-static-interim.md) 사용. 현 시나리오는 본선지연 0 이라 판별력 없음 — 시나리오 축 동반 |
| YR-024 | Sim | 취급시간 확률화 — DGT 육측 원격 인계 분산(PEMA) 반영, 결정적 모델 확장 | 🟡 | YR-023 발견: 공개정보만으론 HJNC·DGT 프로파일이 수치 동일 — 케이스 차별화의 유일한 문헌 경로 |
| YR-015-b | UI | 통합정책 설명·동기비교·운영자 승인/반려 UI | 🟡 | [spec](../docs/dashboard-task-specs/YR-015-verification-ui.md) · YR-035 schema부터 recorder 선반영, 정책 연결은 YR-013/039 후 |
| YR-014 | Exp | 통합정책 locked 평가·ablation·운영 적용판정 | 🟡 | [spec](../docs/dashboard-task-specs/YR-014-final-evaluation.md) · YR-002/009/013/035~039 후 |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
