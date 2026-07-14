# ✅ Done

> 완료 작업 (evidence 박제). 20+ 누적 시 하단 `## Archive` 로 압축 이동. [index](README.md).

| ID | Epic | Title | 완료 | Evidence |
|---|---|---|---|---|
| YR-001 | Infra | **Dashboard 하네스 구축 완료** — board 6파일·훅 2종·AGENTS.md 전역규칙·설계문서 요약 2건·skills 체계(index+운영절차) | 2026-07-12 | [템플릿](../../dashboard-board.md) §8 체크리스트 적용 · 산출물: [Dashboard/](README.md)·[settings.json](../settings.json)·[docs 요약](../docs/)·[skills](../skills.md) · git 미초기화로 commit 없음 (2026-07-12 대화) |
| YR-003 | Infra | **스캐폴드·git init 완료** — pyproject(src 레이아웃)·패키지 골격·tests·configs·evidence 체계 가동 | 2026-07-12 | `3d9398a` |
| YR-004 | Data | **도메인 모델 완료** — Enum·dataclass·validator·프로파일 로더, 가정 프로파일(assumed:true), 테스트 6건 통과 | 2026-07-12 | `083c676` |
| YR-006 | Sim | **단일 YC 이벤트 시뮬레이터 완료** — 이벤트 우선순위·순차합 이동모델·정확적분 KPI·clear-out·결정론 재현성 검증 | 2026-07-12 | `62c1ce5` |
| YR-007 | Sim | **SafetyConstraintEngine 완료** — 2중 차단(후보+실행직전)·매 이벤트 불변조건 검사·invariant 테스트 6건 | 2026-07-12 | `62c1ce5` (YR-006 과 동일 커밋 — 동반 개발) |
| YR-017 | Data | **합성 시나리오 생성기 완료** — 시드 결정론·피크 도착·재조작위험 파라미터. 부수: EventKind alias 버그(본선 release→트럭도착 오처리, KPI 과대계상) 발견·수정 + 정합성 회귀 가드 | 2026-07-12 | `794ede5` |
| YR-008 | Baseline | **Baseline 4종 + 공통 RL 환경 완료** — 정보필터(누출 자동검사)·rule 실행기(결정론 동점체인)·mask·bucket 인코더·Core Cost 보상·paired 통계·golden 회귀 | 2026-07-12 | `cd5fbd6` |
| YR-010 | RL | **Tabular Q-learning(SMDP) 완료** — elapsed-time 할인·masked ε-greedy·미방문 fallback·CLI 파이프라인. quick 검증에서 val 4/4 seed 휴리스틱 우위 | 2026-07-12 | `8df03b4` |
| YR-011-a | Exp | **Exp-1 예비 지지 (합성·가정 조건)** — QL 이 FIFO 대비 평균대기 -15.2% (12/12 seed 유의)·이동 -16%·본선지연 0. 단 P95 +10.8% trade-off (→YR-018) | 2026-07-12 | `24b095a`·`bb1e8e5` · [report](../../outputs/reports/exp_matrix/exp_matrix_report.md) (`a8d9039` 재실행 복원 — .gitignore 재포함 무효로 원커밋 누락, 결정론 재현 일치 확인) |
| YR-011-b | Exp | **Exp-2 예비결과: 정보선행 이득 미확인** — -5.7% vs FIFO 로 Exp-1 대비 유의 열세. 상태공간 희석 의심 (→YR-020) | 2026-07-12 | 〃 (동일 matrix, paired) |
| YR-011-c | Exp | **Exp-3A/B/C 예비결과: H2 이 조건 미지지** — 3A 는 학습예산 2.5×에서 격차 축소(16.56→15.14분, tabular 한계 시사), 3B 포지셔닝 빈이동 +60% 무익, 3C 선재조작 총재조작 +3.3% 역효과 | 2026-07-12 | 〃 + [민감도](../../outputs/reports/exp_matrix_e10/exp_matrix_report.md) (`a8d9039` 복원 포함) |
| YR-015-a | UI | **검증 replay UI MVP 가동** — recorder(04 §3 계약, visible-only 누출 방지) + Streamlit 단일 replay(야드 평면도·크레인 이동·Q-value/mask 패널·KPI·자동재생). HJNC·DGT × QL/FIFO seed301 replay 4벌 박제, 기록 무간섭 regression·AppTest 렌더 포함 54 tests | 2026-07-13 | `a08294f` · [recorder](../../src/yard_rl/experiments/recorder.py)·[ui](../../src/yard_rl/ui/)·[replays](../../outputs/replays/) · 실행: `streamlit run src/yard_rl/ui/app.py` |
| YR-015-e | UI | **애니메이션 모드 가동 (깜빡임 제거) + 항만 환경 디자인** — plotly frames 클라이언트 재생 (차트 내 ▶/⏸·t 슬라이더, 서버 rerun 왕복 제거). 바다·안벽(해측)·게이트 캐노피(육측)·차선 마킹·운전석/차대 분할 트럭·스프레더 크레인. 장치상태는 시작시점 고정 (payload 경량화, 캡션 명시 — 정확 검증은 스텝 모드). e2e 재생 확인, 105 tests | 2026-07-13 | `d11138f` · [yard3d](../../src/yard_rl/ui/yard3d.py) |
| YR-015-d | UI | **즉석 실행 패널 가동 (실시간 테스트)** — UI 에서 터미널 환경·정책·부하(외부트럭 수·본선 수·피크) 선택 → 즉석 시뮬 → 자동 재생. record→replay 원칙 유지 (04 §2.1)·즉석분 `outputs/replays/live/` 미추적 분리. 컨테이너 slot 해시 배색(선사 톤 10색+tier 밝기)으로 균일색 제거. e2e(playwright) 검증, 104 tests | 2026-07-13 | `56e7ab2` · [live.py](../../src/yard_rl/ui/live.py) · QL 은 기본부하 학습 — 타 부하는 일반화 시험 주석 표기 |
| YR-015-c | UI | **야드 2.5D/3D 입체 뷰 가동** — mesh3d 컨테이너 적층(tier 음영)·ARMG 갠트리+작업 bay 지면 밴드·트럭 대상 bay 정렬·orthographic 2.5D. 32-agent 리뷰로 확정결함 9건 수정 (job 메타 종료후 캡처→마커 전멸·자동재생 widget-key 예외·치수 하드코딩 등), 회귀 3건 추가 (102 tests) | 2026-07-13 | `5ca5299`·`f61e873`(조감 시점 상향) · [yard3d](../../src/yard_rl/ui/yard3d.py) · replay 4벌 재생성 (치수 3필드 포함) |
| YR-025 | RL | **목적함수 원화비용 argmin 교체, 효과 유의차 없음 (negative)** — QL_EXP1_COST 총비용 Δ+3.2만원 [-10.1,+16.5] ns vs QL_EXP1·P95 점추정 악화. 원인 추정: 비용 구성 대기항 지배(56%)로 정규화 가중과 상대구조 유사 + tabular 한계. 비용 관점 자체는 유효 — MIN_REHANDLE(평균대기 -32%)이 본선지연 폭증으로 비용 +3.9% 로 반전 노출. 인프라(CostConfig·run-exp1-cost·total_cost_manwon)는 상시 자산 | 2026-07-13 | `6d38d99` · [hjnc](../../outputs/reports/exp1_cost_hjnc/exp1_report.md)·[dgt](../../outputs/reports/exp1_cost_dgt/exp1_report.md) · [cost config](../../configs/costs/won_cost_v1.yaml) |
| YR-027-a | RL | **Direct-Job Cost-Q 전략 명세·히스토리 동결 완료** — 선박·deadline 제외, `BLOCK_ENTRY` 정보경계, 동적 Job action, 평균대기 cost `argmin` 계약 기록 | 2026-07-13 | `9b46ae2` · [v1 전략](../docs/strategy-history/2026-07-13-YR-027-exp1-direct-job-cost-q.md) · [v2 최소상태](../docs/strategy-history/2026-07-13-YR-027-exp1-direct-job-cost-q-minimal-state.md) |
| YR-027 | RL | **최소상태로 coverage 해결, greedy 열세 확인 (negative)** — fallback 55.04%→0.01%지만 shortest-service 대비 평균 +1.195분(CI +0.963~+1.438)·P95 +47.57%; completion 100%·backlog 0 | 2026-07-13 | `687e5d5`·`325abf2` · [v2 report](../../outputs/reports/exp1_direct_costq_minimal_hjnc/exp1_direct_costq_report.md) · [v1 report](../../outputs/reports/exp1_direct_costq_hjnc/exp1_direct_costq_report.md) · [spec](../docs/dashboard-task-specs/YR-027-direct-job-cost-q.md) |
| YR-015-f | UI | **Three.js 실시간 3D 뷰어 완료** — 이벤트 타임스탬프 연속 재생 (크레인 활주·스프레더 사이클·트럭 진입/대기(SLA 적색)/퇴장), 항만 씨너리(바다·안벽·게이트·차선·본선 비콘), 레이아웃 재배치. 리뷰 후속 2건 수정 (shadow 절두체 미적용·`</script>` 이스케이프) + live replay 오추적 정리 (.gitignore 인라인 주석 무효 발견) | 2026-07-14 | `057eebf`(구현)·`06a0ba9`(리뷰 후속) · [viewer3d](../../src/yard_rl/ui/viewer3d.py) · 검증: 107 tests + playwright e2e (부팅·연속재생 00:00→00:06·WebGL·콘솔 0) |
| YR-022 | Data | **프로파일 v2 초안 2벌 완료 — 공개정보 수준에서 HJNC·DGT 수치 동일 수렴** (10열×6단·Kalmar ASC 문헌속도·SLA/gate_travel 보정, 전 항목 근거 주석, assumed 유지). 케이스 차별화 항목은 전부 🤝 협약 필요 → 확률화는 YR-024 | 2026-07-13 | `7b62738` · [hjnc](../../configs/terminals/hjnc_armg.yaml)·[dgt](../../configs/terminals/dgt_armg.yaml) · [요구정보](../docs/YR-002-HJNC-DGT-요구정보.md) |
| YR-023 | Exp | **ARMG 프로파일에서도 Exp-1 방향 유지, 폭 축소** — QL_EXP1 이 FIFO 대비 평균대기 -10.4% (9/12 유의, POC 프로파일 -15.2% 대비 축소)·이동 -11.3%·본선지연 -55.8%, P95 +17.4% 악화 재현 (YR-018 전제 유지). 두 프로파일 결과 동일 (수치 동일 수렴 탓) | 2026-07-13 | `7b62738` · [hjnc report](../../outputs/reports/exp1_hjnc/exp1_report.md)·[dgt report](../../outputs/reports/exp1_dgt/exp1_report.md) |
| YR-016 | Infra | **구현계획서 분할(index+5문서) 하네스 반영 완료** — Phase 0~9 재편·UI 신규 epic/row(YR-015)·task-spec 14건 생성·요약 재작성 | 2026-07-12 | 원본: [구현계획서](../../부산항_야드크레인_강화학습_구현계획서.md)·`docs/구현계획/01~05` · 산출물: [task-specs](../docs/dashboard-task-specs/)·[구현계획서-요약](../docs/구현계획서-요약.md) · commit 없음 (2026-07-12 대화) |

---

## Archive
