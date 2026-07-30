# YR-135 — 공동 Advantage-Q 1단계: V/A 분리 구조 (BlockQ-v3, 오프라인)
> 상태: **in-progress (2026-07-30 — 외부 7차 피드백·사용자 지시로 착수)**
> 근거: [BlockQ-v3 전환 기록](../strategy-history/2026-07-30-BlockQ-v3-AdvantageQ-전환.md)
> — v2 마감 근거 5건 + "결함은 절대 Q 가 아니라 단일 회귀값에 크기·순위를 동시에 맡긴 계약".

## 계약 ([하네스](../../../src/yard_rl/experiments/yr135_advantage_q.py) docstring 이 동결 정본)

- 유일 변경 = 점수망 구조: 단일 점수 → `Q = V(상태 문맥) + A(행 전체) − mean A(결정 내)`.
  V 입력 = 행의 상태 구획(ctx_a 0:116 ⊕ ctx_b 154:212 — 검증 통과), A 입력 = 행 전체.
- 표적 = 후보별 **절대 C600** (차분·잔여항 없음 — 중단 목록 준수). 순위 진실 = 결정 내
  C600 순서(이전 D 순서와 동일 — 비교 가능성 유지).
- SF-SPT = 후보집합 내 동등 경쟁자 + rollout 후속 정책 (학습표적 기준 아님).
- 학습: 결정 그룹 배치 Huber(Q, y)·fresh init·선택지표 = select 대역 r(Q̂,y)·patience 100.
  순위 보조손실 없음 (2단계 몫 — J1 미달 시에만).
- 라벨 3대역 신규 생성(절대비용 저장): train BASE+0/1 · select BASE+700/701 ·
  **judge BASE+1100/1101 (미열람)**. 궤적 = DIFF1 argmin (프로토콜 연속).
- 판정: J1 순위 3/3 ρ≥0.30∧top1≥0.35 · J2 절대 적합 3/3 r(Q̂,C600)≥0.5 ·
  J3 3/3 regret ≤ 같은 대역 YR-131-b 참조. 성공 → 4단계(에피소드 판정) 제안 /
  J2 통과·J1 미달 → 2단계(A 순위 보조손실) / J2 미달 → 구조 재검.

## Evidence
사전동결: (커밋 예정) · 결과: outputs/reports/yr135_advantage_q/
