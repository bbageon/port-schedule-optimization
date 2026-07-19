# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).
> 순서는 [최종전략 전환 이력](../docs/strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md)과 각 task spec의 의존관계가 기준이다. 과거 Phase/Exp 명칭은 완료 이력에만 남긴다.

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| YR-066 | RL | 차분 신호 잔여 용의자 — on-policy counterfactual base·창 4800s·amortized-JR 회귀 | 🟡 | **YR-065 파생 (수확 체감 구간 — 사전 기대치 낮음, 사용자 결정)**: 창 40분이 유의 개선(78.91)했으나 천장이 BC(56.25) 미만. base_policy 불일치·WAIT 앵커 변별이 잔여 후보. rollout 비용 8배 주의 |
| YR-060 | RL | QMIX 타깃 PopArt 보완 (return 크기 비정상성) | 🟡 | [적용전략](../docs/상태정규화-보상가중치-적용전략.md) §6-2 · **조건 발동 (2026-07-19, YR-059: 입력 정규화로 미해결)** — 단 신용 축(YR-061~065)이 1차 용의자로 승격돼 순서는 그 종합 뒤. state_norm 결합 필수 |
| YR-057 | RL | checkpoint 선택 안정화 — val seed 확대·이동평균 선택 | ⚪ | **YR-054 부차 발견** — val 20-seed 곡선 변동폭이 평균의 1.5배, val→test 격차 +3~+9. 격차 주인 아님(우선순위 낮음) |
| YR-058 | Infra | 머신 민감 수렴 테스트 안정화 — residual Δ-net 허용오차·반복수 재조정 | ⚪ | **YR-052 검증 중 발견 (2026-07-18)**: geonu 클론 새 WSL 에서 −2.5±0.15 기대에 −2.22 (변경 전 코드도 동일 — 회귀 아님, CPU 부동소수점 경로 차). 닫힌 단일야드 트랙 테스트라 낮은 우선순위 |
| YR-042 | Exp | DGT·HJNC 근사 프로파일 일반화 게이트 (재실행 대기) | 🟡 | **run 중단 (2026-07-15)** — YR-039 무효로 전제 상실. 구현(`0cd547d`·`f51818c`)은 유효. **2026-07-18 갱신**: YR-045에서 RL 미채택 — 일반화할 채택 정책이 없어 보류. 협조 재판정(YR-013/056) 후 재개 |
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
