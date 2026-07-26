# YR-041 잠금 평가 — 사전등록 (2026-07-26, 결과 미열람 동결)

## 후보 결정 규칙 (재선택 금지)

- 후보 = [3] `yr100_candidate_eval/verdict.json` 에서 **guard(완주 1.0·healthy 전부) 통과
  arm 중 pooled Δtotal(vs SF) 최소** arm. 학습 arm 이면 3시드 중 개별 재선택 없이
  **pooled 최소 시드가 아니라 사전규칙: 시드 88000 checkpoint** 를 대표로 동결
  (시드 선택 자유도 제거 — 88000 은 등록순 첫 시드).
- 이 평가에서 어떤 하이퍼파라미터·정책·시드 재선택도 하지 않는다.

## 평가 격자 (전부 학습·[3] 대역 밖)

- 셀 3: `high-0.40` · `high-0.75` · `mid-0.40` (vessel_deadline_mult — 학습은 0.5/2.0,
  0.40=더 타이트·0.75=중간 타이트 신규 강도), time_contract_v2=True.
- seed: 셀별 8 — `BASE[level]+900..907` (train +0..15 · val +50 · [2] eval +500 ·
  [3] +700 대역과 불겹침).
- 비교군: 후보 vs SF-SPT (paired, 같은 seed).

## 판정 게이트 (guard 우선)

1. **guard**: 전 seed 완주율 1.0 · healthy action-mix · 본선 job 미완료 0.
2. **본선**: Δberth(후보−SF) pooled CI **상한 < 0** 이면 "본선 보호 우월" /
   CI 가 0 포함하며 상한 < +5.0분이면 "비열등" / 그 외 실패.
3. **트럭 비열등**: ΔP95(후보−SF) pooled CI **상한 < +3.0분** (δ=3.0, assumed SLA —
   실측 SLA 확보 전 관행값. CI 0 포함 여부가 아니라 상한<δ 증명).
4. **총비용**: Δtotal pooled CI 보고 (판정 보조 — 상한<0 이면 우월 병기).

전부 통과 = "잠금 채택 가능(assumed δ 하)". guard 실패 = 즉시 기각.
학습 밖 강도(0.40·0.75)는 stress 표시 — 문헌 보정 시뮬레이션 한정 주장.
