# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).
> 순서는 [최종전략 전환 이력](../docs/strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md)과 각 task spec의 의존관계가 기준이다. 과거 Phase/Exp 명칭은 완료 이력에만 남긴다.

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| YR-079 | Exp | 적응형 내다보기 창 재도전 — 소진시간 추정식 보정(재조작량·미도착 유입 반영) 후 고정 1800s 와 재대결 | ⚪ | **사용자 결정 (2026-07-20, YR-078 파생)**: 적응 기준이 불명확해 지금은 고정 30분 채택 — 적응형은 추후 과제. v1 패인: 대기 작업 수만 세는 추정이 밀린 일 과소평가(창 14~23분 vs 필요 30분). 채택 조건: 보정판이 고정 1800s 와 전 셀 대등 이상 + 한산 구간 계산비용 우위 유지 |
| YR-077-2 | Sim | 돌발 강건성 2단계 — 장부 불일치(컨테이너 실제≠기록 위치) 모델 + 노쇼 통합 | ⚪ | **PoC 이후로 연기 (2026-07-20, 사용자 결정)**: "휴먼 에러는 채택 검증 축이지 원리 증명 아님" — 신규 엔진 모델(기록≠실제) 필요·정책 구조 확정 후 재야 정확(YR-075 확장 시 재측정 회피). 진짜 겨누는 것 = rollout 교사 세계모델 오류 붕괴. 실배치 고민 단계에서 재개 |
| YR-075-c | RL | 재조작 목적지 **K후보·30분 국소 rollout 헤드룸** — H1 잔여격차 재판정 | 🟠 | [spec](../docs/dashboard-task-specs/YR-075-c-destination-rollout-headroom.md) · 일반화 동결 전 필수. YR-009 환경·YR-080 목적 뒤 실행하고, 목적지 규칙이 바뀌면 정책을 재검증한 뒤 조건·구조 일반화 개방 · [재검토](../docs/strategy-history/2026-07-20-YR-075a-신뢰성재검토-오라클상한성반박.md) |
| YR-041 | Exp | **본선 마감 조건 locked 시험 — 목적함수 자체결함 검증 (필수, YR-080 직후)** | 🟠 | **격상 (2026-07-20, 사용자 결정)**: 019·021 과 성격이 다름 — 외부 돌발 강건성이 아니라 **우리 목적함수가 트럭 이익을 위해 본선을 지연시키는 결함**의 검증축. YR-080 프로토타입이 실측(본선지연 721.92분 방치, 트럭대기 6배 우위 일부가 본선 희생 산물). 항만 KPI 위계상 선박 1차 → **본선 죽이면 "운영 개선" 주장 자체 무너짐** = 원리 증명 필수. 순서: YR-080 으로 본선 병기·목적 고친 **뒤**, 타이트 마감·신규 seed 에서 재선택 없이 본선·트럭 guard 검사(그 전엔 틀린 목적이라 무의미). [spec](../docs/dashboard-task-specs/YR-041-vessel-deadline-locked.md) · [프로토타입 근거](../docs/strategy-history/2026-07-20-YR-080-문헌정합목적-프로토타입.md) |
| YR-019 | Exp | ETA 품질 조건 일반화 locked 시험 — BIASED/NO_SHOW/STALE | ⚪ | **실증 단계로 보류 (2026-07-20, 사용자 결정)**: ETA 편향·노쇼·낡음은 **운영에서도 예측 불가한 외부 돌발** — "원리 증명(PoC)"이 아니라 "실배치 강건성(실증)" 축. YR-077 폐기와 동일 기준(휴먼에러=채택 검증축). PoC 단계 비대상, 실증 과정에서 재개. [spec](../docs/dashboard-task-specs/YR-019-eta-quality-locked.md) |
| YR-021 | Exp | 학습 밖 부하·혼잡 조건 일반화 locked 시험 | ⚪ | **실증 단계로 보류 (2026-07-20, 사용자 결정)**: 학습 밖 극단 부하·혼잡은 외부 돌발 강건성(실증) 축 — PoC 원리 증명 비대상(019 와 동일 기준). 실증 단계에서 재개. [spec](../docs/dashboard-task-specs/YR-021-ood-load-locked.md) |
| YR-082 | Data | **부산항 10개 터미널 프로파일 자격심사·PoC 전환 게이트** | 🟠 | [spec](../docs/dashboard-task-specs/YR-082-busan-terminal-poc-profile.md) · A 공개 registry는 YR-009·080과 병행 가능. 공개 Level 0~1은 성능근거로 쓰지 않고, 같은 구조군 Level 2 프로파일 2개 미만이면 YR-042 실제구조 arm은 `DATA-BLOCKED`; 현장 실증 PoC는 Level 3~4 뒤 |
| YR-083 | Sim | **Level 2 터미널 구조계약 런타임화 — 도로·인계점·크레인 역할을 엔진이 실제 소비** | 🟠 | [spec](../docs/dashboard-task-specs/YR-083-level2-runtime-contract.md) · 현재 엔진은 실제 방향·용량·차종·육해측 역할·통과규칙을 버리므로 YR-082 자료만으로 YR-042 실행 불가. 구조 compiler·mask·resolver와 체크포인트 호환성 판정이 선결 |
| YR-075-b | RL | 결정권 확장 2: 반입 장치위치 (생애주기 연결 선행) | ⚪ | 현 시뮬은 반입 컨테이너가 같은 에피소드에서 재반출되지 않아 보관품질 검증 불가 — **반입→보관→미래 반출 생애주기 시나리오 연결 후에만** 개방. YR-075-a 후 |
| YR-066 | RL | 차분 신호 개량 — on-policy counterfactual base·창 확대·amortized-JR 회귀 | 🟠 | **우선순위 상향 (2026-07-19, YR-068 기각 근거)**: 본 시나리오에서 D_mean 급감(−5~−7→−1.6~−2.5) — 결정 밀도 상승이 창 내 반사실 차이를 희석. **규모에서 살아남는 차분 신호 설계가 협조 트랙의 선결 병목**. rollout 비용 주의 · **개정 전략 재정의 (2026-07-19)**: OLD 비용 기준 차분 개량은 철회 — 대체 경로 = YR-073(JR_NEW 교사·NEW 목적), 반사실 기법은 YR-074 미세조정 표적으로 흡수 |
| YR-069 | RL | 차분 표적 QMIX 민감도 — λ_mix·창 그리드 (별도 사전등록) | ⚪ | YR-013c prereg 가 금지한 knob 탐색의 정식 경로 — **개정 전략 (2026-07-19): 2크레인 1블록에선 중앙 공동가치망(YR-073)이 우선 — QMIX 재개 조건은 다중 블록/크레인 확장(공동조합 폭발)으로 변경** |
| YR-060 | RL | QMIX 타깃 PopArt 보완 (return 크기 비정상성) | 🟡 | [적용전략](../docs/상태정규화-보상가중치-적용전략.md) §6-2 · **조건 발동 (2026-07-19, YR-059: 입력 정규화로 미해결)** — 단 신용 축(YR-061~065)이 1차 용의자로 승격돼 순서는 그 종합 뒤. state_norm 결합 필수 |
| YR-057 | RL | checkpoint 선택 안정화 — val seed 확대·이동평균 선택 | ⚪ | **YR-054 부차 발견** — val 20-seed 곡선 변동폭이 평균의 1.5배, val→test 격차 +3~+9. 격차 주인 아님(우선순위 낮음) |
| YR-058 | Infra | 머신 민감 수렴 테스트 안정화 — residual Δ-net 허용오차·반복수 재조정 | ⚪ | **YR-052 검증 중 발견 (2026-07-18)**: geonu 클론 새 WSL 에서 −2.5±0.15 기대에 −2.22 (변경 전 코드도 동일 — 회귀 아님, CPU 부동소수점 경로 차). 닫힌 단일야드 트랙 테스트라 낮은 우선순위 |
| YR-042 | Exp | **다른 단일 블록·2크레인 profile-shift 게이트 — 동결 정책 무재학습 → 조건부 재적응** | 🟠 | [spec](../docs/dashboard-task-specs/YR-042-cross-terminal-generalization.md) · 정책 동결 + YR-082 Level 2 프로파일 2개 + YR-083 런타임 지원이 선결. 미충족이면 `DATA/STRUCTURE-BLOCKED`; 공개 Level 0~1·임의 수치는 stress test로만 표기 |
| YR-029 | RL | P95 보호 — SLA 임박 후보 필터 | 🟠 | YR-018 negative 파생 · 보상형이 아닌 YR-037 mandatory 후보/명시적 제약으로 흡수 |
| YR-005 | Data | Phase 1 후반: 원천자료 loader·익명화·품질 플래그·날짜 split | ⚪ | [spec](../docs/dashboard-task-specs/YR-005-data-pipeline.md) · 현재는 D5 협약 폐기로 휴면. 향후 YR-082 Level 3 운영로그 접근권한이 실제 확정될 때만 재개하고, 그 전에는 공개자료를 실측로그로 대체하지 않음 |
| YR-020 | RL | Exp-2/3 열세 원인 분석 — 상태공간 희석 vs 정보 무익 판별 | 🟡 | [수렴진단](../docs/YR-020-수렴진단-2026-07-14.md): 희석 방문통계 증거 확보. **YR-030 전환 결정으로 "함수근사 판단재료" 역할 종결** — 학술적 원인 규명 가치로만 유지 (🟠→🟡, 2026-07-14) |
| YR-024 | Sim | 취급시간 확률화 — DGT 육측 원격 인계 분산(PEMA) 반영, 결정적 모델 확장 | 🟡 | YR-023 발견: 공개정보만으론 HJNC·DGT 프로파일이 수치 동일 — 케이스 차별화의 유일한 문헌 경로 |
| YR-015-b | UI | 통합정책 설명·동기비교·운영자 승인/반려 UI | 🟡 | [spec](../docs/dashboard-task-specs/YR-015-verification-ui.md) · 현재는 읽기전용 replay. YR-014 채택 bundle과 YR-082 Level 4 범위가 확정되면 실제 제어 없는 Shadow Mode의 승인·반려 기록으로 확장 |
| YR-014 | Exp | **채택 정책 locked 종합평가·주장 확정** | 🟡 | [spec](../docs/dashboard-task-specs/YR-014-final-evaluation.md) · 정책 bundle 동결, 조건검증, YR-082 자료등급·YR-083 런타임 호환·YR-042 구조 분류 후. 기준·stress·무재학습·재적응 결과를 분리하고 완주 guard 우선 판정 |
| YR-081 | RL | **구조 확장 게이트 — 가변 크레인 수·다중 블록 계층관제** | ⚪ | [spec](../docs/dashboard-task-specs/YR-081-variable-crane-yard-scaling.md) · YR-082가 나눈 RMG/ATC-YT·수직 AGV·수직 S/C·북항 혼합형을 별도 환경군으로 확장. YR-042·014 뒤 독립 블록 기준선부터 측정하고 공유 병목 상금이 있을 때만 상위 관제·QMIX 검토 |
| YR-084 | Infra | **장기 운영화 북극성 — LLM 감독형 터미널 오케스트레이션·관제 Shadow PoC** | ⚪ | [spec](../docs/dashboard-task-specs/YR-084-llm-terminal-orchestration.md) · **지금 착수하지 않음(2026-07-22 사용자 결정)**. 블록→야드(YR-081)→결정론적 터미널 계획기가 먼저이며, LLM은 장비제어가 아니라 예외 해석·검증된 재계획 도구 호출·설명만 담당. Level 3 운영로그·Level 4 Shadow 승인과 권한/감사/복귀 계약 뒤 개방 |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
