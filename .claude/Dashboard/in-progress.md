# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-136 | RL | **점증 지연비용 계약 v2.1 (softplus 확률 계약) — 0단계 착수** | 🔴 | 2026-07-31 | **v2.0 smoothstep → v2.1 softplus 개정(외부 피드백·사용자 지시)**: r_T = 1+σ((Ô−D_T)/κ_T) = 1+Pr(SLA 초과)·r_V = 10σ((F̂−P)/κ_V). 고정 10/30분 경계 폐지 — 전환 폭 = 예측오차 적합 κ(b=평균·κ=(√3/π)SD, 훈련 시드 적합 후 동결). 의미 변화 명기: 마감에서 최대율의 **절반**(1.5/5), 최대율 점근. 신규 구성요소 = 트럭 Ô 예측기(공개정보만: 크레인 잔여+선행 대기×180s+출문 평균 300s — exit_travel_s 실현값 미열람). L_T = 2,580s 유도 앵커. **0단계 실행 중**: SF·4셀×4시드 오차 측정 → kappa_fit.json 동결. 이후 1 계약·골든 → 2 트럭 축 → 3 본선 축(33→10) → 4 결합 → 5 v2 재라벨로 YR-135 구조 재평가(채택 경로) · [spec v2.1](../docs/dashboard-task-specs/YR-136-smooth-cost-contract-v2.md)·[v2 모듈](../../src/yard_rl/integrated/cost_curve_v2.py)·[0단계 하네스](../../src/yard_rl/experiments/yr136_softplus_contract.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
