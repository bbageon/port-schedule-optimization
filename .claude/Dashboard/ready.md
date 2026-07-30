# 📋 Ready (선택됨)

> 착수 준비된 작업 (착수 신호 대기). [index](README.md) · 인접: [backlog](backlog.md) → 여기 → [in-progress](in-progress.md).
> **정비 완료 (2026-07-28)**: YR-106·106-b·107·108·109·110·112·112-b 전부 [done](done.md).
> **판정 2건 확보**: ①본선 악화 방지 필터 = **기각** ②**창중 이송 자체 = 이롭다**(2대역 재현).
> 이제 하네스는 신뢰할 수 있고, **이송이 이롭다**는 전제 위에서 최적화 축을 열 수 있다.

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|
| YR-136 | RL | **점증 지연비용 계약 v2 — 트럭 SLA 접근 1→2·본선 완료접근 최대 10** | 🔴 | **사용자 결정(2026-07-30)**: 완료된 YR-123 v1(계단형·본선 33)은 이력 보존. 새 opt-in 계약에서 smoothstep 적분으로 트럭 마지막 10분 `1→2/h`, 본선 적하 마지막 30분 `0→10/h`; 평가 본선 가중치도 10으로 정렬한다. 곡선만 바꿔 학습·평가가 어긋나는 것 금지. 트럭만→본선만→결합 순으로 단일축 검증하고, v1 라벨의 YR-135는 구조 진단용으로만 한정 · [spec](../docs/dashboard-task-specs/YR-136-smooth-cost-contract-v2.md) · [결정이력](../docs/strategy-history/2026-07-30-YR-136-점증비용곡선-v2-사용자결정.md) |
| YR-124 | RL | **블록 Q 상태표현 3-arm 사전등록 비교 — 라벨 삭제는 실증 후에만** | 🟠 | **사용자 피드백 (2026-07-29) 그대로**: `is_vessel`·마감 원시값을 바로 지우지 말고 **A(현행: 작업종류+마감 원시값 — [adapter](../../src/yard_rl/integrated/adapter.py) `is_vessel`·`deadline_slack_s` 노출) vs B(단일 계산비용 cvec — 현 CALC 방식) vs C(종류 없이 지연 한계비용 곡선 — YR-123 산출)** 를 비교한다. 상태추상화 경고(Li·Walsh·Littman 2006): 비용 하나로 다른 미래를 뭉치면 최적정책을 잃을 수 있다. **선결 = 유휴 쏠림 축 해소(YR-122+)** — 안 풀린 채 돌리면 표현 차이가 쏠림 잡음에 묻힌다. ≥3 초기화 필수(YR-118 s88000 불안정), 판정 = 총비용+A→O · [spec](../docs/dashboard-task-specs/YR-124-state-representation-3arm.md) **보류 강화(7차 피드백)**: "검증되지 않은 추가 상태정보 중단" — v3 최소 정책(YR-135) 성능 확인 후에만 별도 단일축 |
| YR-120 | Exp | **arm 자격 규칙 완화 — 보류 (사용자 결정 2026-07-28)** | ⚪ | **지금 건드리지 않는다.** 미건전 에피소드가 실제로 비쌌으므로(+6.0 [+0.76,+11.23]) 단순 "24개 중 1개 허용" 식 완화는 나쁜 정책을 통과시킨다. YR-119 로 실패율이 충분히 내려가면 바꿀 필요 자체가 없다. **나중에 완화한다면 두 조건 병용**(사용자 지정): ①실패율의 **통계적 상한**이 사전 허용치보다 낮을 것 ②실패 에피소드의 **비용 악화가 안전 한계** 이내일 것 · [spec](../docs/dashboard-task-specs/YR-120-arm-qualification-rule.md) |
| YR-115 | Exp | **이송 순효과 공동지표 확증 — 새 시드 대역으로 재시작 필요** | 🟠 | **상태 정정 (2026-07-28)**: 진행 중 아님. v2 파일럿 16쌍 통과·확증 254쌍 동결까지 갔으나 확증을 잘못 시작해 **156/254 에서 중단** — 로그에 156쌍 수치가 노출됐으므로 그 대역(1000000+)은 **독립 확증용으로 재사용 불가**(열람 대역 재사용 금지 원칙). 재개하려면 **새 시드 대역**으로 확증 254쌍을 처음부터. 동결된 계약(총비용+A→O 공동 AND·δ total 10/A→O 1분)은 유효 · [spec](../docs/dashboard-task-specs/YR-115-transfer-benefit-joint-confirm.md) |
| YR-111 | Infra | **선존재 회귀 실패 1건 — `test_update_regresses_toward_residual_target`** | 🟡 | 400 step 후 −2.22(목표 −2.5, 허용 ±0.15). 수렴 미달 vs 잔차 회귀식 결함 **판별**이 목적 — 잔차 Δ 학습(YR-012·YR-102)을 쓰기 전에 처리 · [spec](../docs/dashboard-task-specs/YR-111-residual-regression-failure.md) |
| YR-114 | Sim | **크레인 3기 이상의 연쇄 간섭 교착 재검** | ⚪ | YR-112 는 2기 프로파일에서만 실증했다. 3기 이상이면 한 크레인이 물러난 자리가 다른 교착을 만드는 연쇄가 가능하다. 크레인 수 확장(YR-081) 시 선결 · [spec](../docs/dashboard-task-specs/YR-114-multi-crane-chain-deadlock.md) |

---

운영: [backlog.md](backlog.md) 에서 승격. 착수 시 [in-progress.md](in-progress.md) 로 이동.
