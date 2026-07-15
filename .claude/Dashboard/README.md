# `.claude/Dashboard/` — 작업 board (index)

> Jira/Plane 스타일 작업 board. **상태별로 파일 분리** (각 state = 한 파일). 본 README 가 **index**
> (운영 규약 + state 링크 + epic + 현황 overview). 상세 evidence 는 설계문서/산출물/git commit 이
> single source — board 는 그 위의 index 일 뿐 중복 서술하지 않는다.
> board 규약 원본: [dashboard-board.md](../../dashboard-board.md)

## 상태별 파일

| State | 파일 | 현황 |
|---|---|---|
| 🗂️ Backlog | [backlog.md](backlog.md) | 미착수·미래 |
| 📋 Ready | [ready.md](ready.md) | 착수 준비 |
| 🟢 In Progress | [in-progress.md](in-progress.md) | 진행 중 (한 번에 1개) |
| ✅ Done | [done.md](done.md) | 완료 (evidence 박제) |
| 🚫 Cancelled | [cancelled.md](cancelled.md) | 폐기 (사유 박제) |

흐름: `Backlog → Ready → In Progress → Done / Cancelled`.

## 사용 규약 (요지)

- **Issue ID**: `YR-NNN` (yard_rl 패키지명 기반). 닫힌 ID 재사용 금지. **Priority**: 🔴/🟠/🟡/⚪. **Epic**: 아래 표.
- 상태 이동(pull/done/폐기)·evidence 박제·grooming **절차 상세**: [dashboard-ops skill](../skills/dashboard-ops.md) (index: [skills.md](../skills.md)).
- **순서 제시 원칙·1줄 index 원칙** 등 전역 규칙: [AGENTS.md](../../AGENTS.md). row 상세 명세는 [task-specs](../docs/dashboard-task-specs/) 의 `<ID>-<slug>.md` (spec 이 원본 문서 § 를 링크).

## 🧭 Epics

| Epic | 의미 | 상태 |
|---|---|---|
| Infra | 하네스·환경·도구·프로젝트 스캐폴드 | 상시 |
| Data | TOS·VBS·ETA·본선·장비·레인 통합 schema와 실자료 매핑 | active |
| Sim | 통합 이벤트 시뮬레이터·Hard Constraint·실측 validation | active |
| Baseline | 강한 동정보 휴리스틱·중앙 matching·paired runner | active |
| RL | 동적 후보 Double DQN·Total Cost·QMIX 협조학습 | active |
| Exp | 동일 통합정책의 locked 평가·ablation·민감도 | active |
| UI | 정책 replay·설명·운영자 승인/반려 피드백 | active |

## 📌 현재 상태 overview (한눈에)

- **최종 목표 전환 (2026-07-15, 사용자 결정)**: 별도 Exp 정책이 아니라 차량·본선·이송장비·레인·다중 YC를 처음부터 같은 State·Action·Total Cost 계약으로 다루는 단일 통합정책. [최종전략](../docs/부산항_레인_다중야드크레인_협조최적화_강화학습_최종전략.md) · [결정 이력](../docs/strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md).
- **정책 구조**: 가변 후보 `Q_cost`를 Candidate Double DQN으로 평가하고 중앙 resolver가 공동제약을 보장한 뒤 QMIX 추가효과를 검증한다. 안전·물리·마감 위반은 보상이 아니라 mask다.
- **단일 야드 트랙 종료 (2026-07-15, 사용자 결정)**: "현 환경에서 greedy(SPT)는 near-optimal" 로 결론. 근거 사슬 — 격차 +1.195→+0.083(해상도)→+0.035(feature, 220k) 축소했으나 YR-033 이 정정: 동일 checkpoint 가 fresh 240k 에선 +0.111·최적선택 하한도 +0.111·winner's curse 기각(Spearman 0.96). 커버리지·초기화·구조(H-B)·해상도·학습기법·선택 전 축 소진 → robust 격차 ~+0.1 는 정책이 아니라 문제 성질. [종료 결론서](../docs/strategy-history/2026-07-15-single-yard-track-closure.md). RL 잔존가치(tail·oracle 상금 +0.182)는 통합전략(YR-034)에서 재탐색.
- **통합 파이프라인 토대 완성 (2026-07-15)**: YR-035 계약(itc-v1)·YR-036 이벤트 시뮬레이터(다중 YC·본선·이송·레인)·YR-037 동적 후보+중앙 joint resolver·YR-038 정규화 비용/보상 **모두 done** — 계약→환경→공동배정→비용 토대 완성. 각 태스크 설계→구현→적대리뷰. sim/(단일 YC) 전 구간 동결·golden 불변. 매핑: [YR-035](../docs/YR-035-통합계약-매핑.md)·[YR-036](../docs/YR-036-통합시뮬레이터-매핑.md)·[YR-037](../docs/YR-037-후보-resolver-매핑.md)·[YR-038](../docs/YR-038-비용-reward-매핑.md).
- **통합 RL 첫 형식 승리 (2026-07-15, YR-039)**: Candidate DQN 3-variant 전부 SPT baseline 유의 승리 — **Dueling 총비용 −84% [CI 상한<0 최초]·대기 4.2→1.3분·P95 −54%·guardrail 4/4**. 통합 지형에서 greedy 준최적성 붕괴 (단일야드 종결과 대칭). 기본 채택 후보 = dueling. 한계: baseline 비강자·scale assumed. [매핑+결과](../docs/YR-039-학습기-매핑.md)
- **남은 실행 순서**: YR-013 중앙배정·QMIX 추가효과 판정 (**착수 조건 충족**) → YR-014 locked 평가·ablation (강화 휴리스틱 대조 포함). 실측·validation 은 YR-002/009.
- **주장 게이트**: 실측자료·CURRENT_RULE은 YR-002, 실측 validation은 YR-009가 담당한다. 두 게이트 전 모든 결과는 합성·가정 조건의 구현 증거이며 부산항 실운영 개선 주장이 아니다.
