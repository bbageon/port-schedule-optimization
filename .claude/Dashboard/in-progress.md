# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-135 | RL | **공동 Advantage-Q — 1단계 완료(J2 통과)·2단계 순위 보조손실 학습 중 (BlockQ-v3)** | 🔴 | 2026-07-30 | **1단계 판정(구조 진단, v1 계약 라벨)**: **J2 절대 적합 최초 3/3 통과** — r(Q̂,C600) 0.690/0.513/0.611 (신규 대역 BASE+1100/1101, 여정: 1~3%→31%→0.42~0.52→**0.51~0.69**) = V(상태 공통비용)가 분산을 흡수하자 비용 크기 능력 확보. **J1 순위 미달**(ρ 0.288/0.136/−0.037 — 회귀만으론 A 에 순위 안 생김, YR-131 재확인)·J3 혼재(2/3 개선). **동결 분기 발동 → 2단계**: 유일 추가 = 결정 내 pairwise 순위 보조손실(α=1.0 동등 앵커·margin 0.01), 선택지표 = select ρ, **판정 대역 BASE+1300/1301 신규**(1100 열람됨). 판정 = J1 ρ≥0.30∧top1≥0.35 ∧ **J2 r≥0.5 유지**(순위손실이 절대 적합을 파괴하지 않아야 — v3 존재 이유) ∧ J3 regret≤131-b. 성공 시에도 채택은 YR-136 v2 재라벨 후 · [1단계 report](../../outputs/reports/yr135_advantage_q/report_stage1.md)·[spec](../docs/dashboard-task-specs/YR-135-advantage-q.md)·[하네스](../../src/yard_rl/experiments/yr135_advantage_q.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
