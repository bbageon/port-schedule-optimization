# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).
> Phase 는 [구현계획서](../../부산항_야드크레인_강화학습_구현계획서.md) 분할문서 [05 §4](../../docs/구현계획/05_테스트_로드맵_산출물.md) 기준 (0~9).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| YR-005 | Data | Phase 1 후반: 원천자료 loader·익명화·품질 플래그·날짜 split | 🟡 | [spec](../docs/dashboard-task-specs/YR-005-data-pipeline.md) · 실자료는 YR-002 후 |
| YR-009 | Sim | Phase 2 게이트: 시뮬레이터 실측 validation | 🟡 | [spec](../docs/dashboard-task-specs/YR-009-simulator-validation.md) · 실자료 의존, 미충족 시 RL 평가 금지 |
| YR-019 | Exp | ETA 품질 시나리오 매트릭스 (PERFECT/BIASED/NO_SHOW/STALE) | 🟡 | §18.2 — 현재는 EMPIRICAL(±300s)만 구현. Exp-3 결과의 강건성 확인용 |
| YR-020 | RL | Exp-2/3 열세 원인 분석 — 상태공간 희석 vs 정보 무익 판별 | 🟡 | [수렴진단](../docs/YR-020-수렴진단-2026-07-14.md): 희석 방문통계 증거 확보. **YR-030 전환 결정으로 "함수근사 판단재료" 역할 종결** — 학술적 원인 규명 가치로만 유지 (🟠→🟡, 2026-07-14) |
| YR-021 | Exp | 부하조건별 정보효과 — peak·고장치율·고재조작 시나리오 재실험 | 🟡 | 정보 선행 편익은 혼잡 조건 의존 가설 — §18.3 운영부하 축 |
| YR-032 | RL | 계열 2 미래정보 Δ-net — ETA 정보효과→사전포지셔닝→선재조작 단계 검증 | 🟠 | [spec](../docs/dashboard-task-specs/YR-032-future-info-residual-rl.md) · YR-031 종료 후 승인 대기; 본선·다중 YC는 YR-013으로 분리 |
| YR-032 | RL | checkpoint 선택 프로토콜 보완 — validation 확대·이중 검증 (winner's curse 대응) | 🟡 | YR-012-b 부수 발견: val 최저와 test 최저 역전 (60~120 ckpt × val 30일) — arm 간 ~0.1분 분별에 표본 부족. 저비용 보완 |
| YR-029 | RL | P95 보호 — SLA 임박 시 후보 필터 (계열 2, YR-030 라인) | 🟠 | YR-018 negative 파생. **보상형 후보(분위수/비선형 페널티) 폐기 — 사용자 결정**: 상태별 트레이드오프 신호 부재·기준 모호 — 학습이 아닌 명시적 제약(후보 제한)으로 강제 |
| YR-026 | RL | 비용계수 민감도 + tail 60분 임계 KPI 확장 (안전운임 제도 정합) | 🟡 | YR-025 후속: 계수 4/5가 assumed·tail 은 30분 proxy — 본선·tail 계수 grid 로 negative 결과의 강건성 판별. tail 지표 재정의는 YR-029 와 연계 |
| YR-024 | Sim | 취급시간 확률화 — DGT 육측 원격 인계 분산(PEMA) 반영, 결정적 모델 확장 | 🟡 | YR-023 발견: 공개정보만으론 HJNC·DGT 프로파일이 수치 동일 — 케이스 차별화의 유일한 문헌 경로 |
| YR-015-b | UI | Phase 6 후반: UI-3 정책설명 패널 고도화·UI-4 동기비교·검증 테스트 확충 | 🟡 | [spec](../docs/dashboard-task-specs/YR-015-verification-ui.md) · UI-1/2 는 YR-015-a 로 분할 착수 (2026-07-13) |
| YR-013 | Exp | Phase 8: Exp-4 다중 YC 협조 | ⚪ | [spec](../docs/dashboard-task-specs/YR-013-exp4-multi-yc.md) |
| YR-014 | Exp | Phase 9: 최종평가·ablation·탄소 사후평가 | ⚪ | [spec](../docs/dashboard-task-specs/YR-014-final-evaluation.md) |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
