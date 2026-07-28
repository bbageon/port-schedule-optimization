# 📋 Ready (선택됨)

> 착수 준비된 작업 (착수 신호 대기). [index](README.md) · 인접: [backlog](backlog.md) → 여기 → [in-progress](in-progress.md).
> **정비 완료 (2026-07-28)**: YR-106·106-b·107·108·109·110·112·112-b 전부 [done](done.md).
> **판정 2건 확보**: ①본선 악화 방지 필터 = **기각** ②**창중 이송 자체 = 이롭다**(2대역 재현).
> 이제 하네스는 신뢰할 수 있고, **이송이 이롭다**는 전제 위에서 최적화 축을 열 수 있다.

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|
| YR-118 | RL | **학습 정책이 건전성 검사에 전부 걸린 원인 규명 — 학습 트랙의 진짜 병목** | 🔴 | **사용자 지시 (2026-07-28): YR-105-b 보다 먼저.** YR-100-[3] 에서 학습 arm **6개(CALC×3·CONTROL×3)가 전부** `assert_healthy_action_mix` 에 걸려 탈락했고, 그 결과 남은 유일한 통과 arm 이 오라클(JR1800)이라 [YR-107](../../outputs/reports/yr106b_gates/report.md) 규칙 결손까지 유발했다. **이게 안 풀리면 학습 arm 을 판정에 올릴 수 없다** — 지금까지의 이득이 전부 규칙에서 나온 구조적 이유. 검사 내용: ①`serve_when_available < 0.25`(실작업 가능한데 안 고름) ②단일 비-SERVE 행동이 60% 초과. **문제**: 저장 원자료에 `action_mix` 가 없어(`arm_*.json` 에 `healthy` bool 만) **어느 조건에 걸렸는지조차 모른다** → 진단 재실행부터. 산출물: 실패 조건·행동분포·원인 가설과 그 반증 |
| YR-115 | Exp | **새 공동 주지표로 현행 0.10 이송의 순효과 확증** | 🟠 | YR-105-b가 대안 임계를 찾지 못해 채택 임계는 **0.10으로 고정**됐다. 이제 `0.10 vs NOTRANSFER`를 total+A→O 공동 AND로 결과 전에 사전등록한다. YR-113/117은 사후 재해석이라 확증 아님. fresh blinded pilot으로 분산만 열어 n을 동결하며 δ는 total 10·A→O 1분(assumed) |
| YR-111 | Infra | **선존재 회귀 실패 1건 — `test_update_regresses_toward_residual_target`** | 🟡 | 400 step 후 −2.22(목표 −2.5, 허용 ±0.15). 수렴 미달 vs 잔차 회귀식 결함 **판별**이 목적 — 잔차 Δ 학습(YR-012·YR-102)을 쓰기 전에 처리 |
| YR-114 | Sim | **크레인 3기 이상의 연쇄 간섭 교착 재검** | ⚪ | YR-112 는 2기 프로파일에서만 실증했다. 3기 이상이면 한 크레인이 물러난 자리가 다른 교착을 만드는 연쇄가 가능하다. 크레인 수 확장(YR-081) 시 선결 |

---

운영: [backlog.md](backlog.md) 에서 승격. 착수 시 [in-progress.md](in-progress.md) 로 이동.
