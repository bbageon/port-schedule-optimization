# YR-030-b — v1 최종안 상태 + greedy-prior Q0 + γ grid 사전등록

> 기록일: 2026-07-14 · 상태: **사전등록 — 본 실행 전 동결**
> 상위: [YR-030 계열 2 baseline 승격](2026-07-14-YR-030-series2-baseline-pivot.md) · 선행: [YR-028 결과](2026-07-14-YR-028-coverage-ablation-prereg.md)

## 1. 결정 경위

1. YR-028 판정(CHECKPOINT_RULE)과 부가 발견 — **fallback↓=성능↓ 단조** (순수 학습 Q 의
   순서가 shortest-service greedy 열세: v1 순수 +0.580분, v2 순수 +1.283분) — 로
   YR-030 1차 과제가 "순서품질"로 확정됨.
2. **사용자 v1 상태 전략 최종안 제출 (2026-07-14, 원문 요지)**:
   - `YardState = (work_phase, crane_area, waiting_truck_level, longest_wait_level,
     over_30min_truck_count)` — 운영단계 5 / 크레인구역 4 / 대기규모 4 /
     최장대기 5 / 30분 초과 수 4.
   - `JobState = (job_type, truck_wait, crane_travel_time, total_work_time,
     containers_to_move_first)` — 반입·반출 2 / **트럭대기 4단계(짧음·보통·김·30분
     이상)** / **크레인이동 3단계(가까움·보통·멂)** / 총작업시간 4 / 선행이동 4.
     반입은 선행이동 해당 없음 (job_type 이 키에 있어 반출 '없음'과 비충돌).
   - **상태 일관성 규칙 4건**: ① 결정 시 대기 ≥1 ② 30분 초과 수 ≤ 대기 수
     ③ 후보 대기 ≤ 최장 대기 ④ 반입에 선행이동 미적용.
   - 명시된 한계: 대기열 맥락 부재·작업 후 크레인 위치 누락은 순서품질 문제로 잔존.
3. **사용자 학습설정 지시 (2026-07-14 원문)**: "처음에는 greedy 하게 해서 보상율
   0.95 로 해서 하고 주변값들도 적용해서 비교해보자" — greedy 초기화 + γ=0.95
   중심 grid 로 해석 (아래 §3).

## 2. 기존 대비 델타 (v1_rich → v1_final)

| 항목 | v1_rich (YR-027 복원) | v1_final (사용자 최종안) |
|---|---|---|
| YardState | 동일 5필드 | 동일 (YardState NamedTuple 공유) |
| truck_wait | 4분위+SLA edge → 5단계 | **3분위(<SLA)+SLA edge → 4단계** |
| crane_travel | 4분위 → 4단계 | **3분위 → 3단계** |
| 일관성 규칙 | 없음 | **env 불변조건 4건 — 위반 즉시 중단** |
| Q0 | 0 (낙관) | **greedy 즉시비용 ĉ(j)=(대기-1)×service/(60N)** |
| 탐험 | 미방문 우선 → ε | **ε-random / argmin(prior 포함)** — 미방문 우선 제거 |
| 평가 | 미방문 시 결정 전체 greedy fallback | **fallback 없음** — argmin(학습값∪prior), coverage 는 진단만 |
| bootstrap | 미방문 다음 키 = 0 | 미방문 다음 키 = **그 후보의 prior** |

α₁=1 (n^-p, n=1) 이므로 실방문 1회로 prior 가 실측 target 으로 대체된다 — prior 는
"안 가본 곳의 안내값"으로만 남는다.

## 3. 동결 설정

| 항목 | 값 |
|---|---|
| 프로파일 / arm | HJNC-ARMG (assumed) / `SLA_OFF` |
| seed band | train 70000+3000 / val 80000+30 / test 90000+100 — 기존 band(10k~/40k~) 와 분리, 코드가 재사용 거부 |
| γ grid | **{0.90, 0.95, 0.99, 1.0}** — 사용자 지정 0.95 + 주변값. γ=1.0 은 "prior 단독 효과" ablation 겸용 |
| p | **1.0 고정** — YR-028 R2 선택값. grid 폭 통제 (γ 4 × p 3 = 12 arm 은 과대) |
| 학습 | 3,000 ep/γ, ckpt 50 ep, ε=1/√ep |
| 선택 | γ 별 validation mean_wait 최소 checkpoint (coverage gate 는 prior 로 자동 충족 — signature coverage 진단 보고) |
| 비교군 | direct baseline 6종 (alias assert 유지) — validation 최저 mean 이 paired 기준. **YR-028 v1_rich R2@3000 agent (Q0=0, γ=1) 를 같은 test band 에 재평가한 reference 포함** |
| 통계 | paired bootstrap 10,000, mean CI + p95 변화율 CI |
| 판정 | **개선 = 어떤 γ 의 mean CI 상한 < 0 (vs 선택 baseline)**. P95 CI 상한 +5% guardrail·completion 100%·backlog 0·invariant 함께 보고 |

## 4. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| p grid 유지 (γ×p 12 arm) | YR-028 곡선에서 p 3종이 사실상 동일 궤적 — 예산 대비 정보 없음 |
| 잔차학습 (Q = ĉ + Δ 분해) | 구현 침습 큼. prior 초기화가 최소변경 근사 (α₁=1 로 첫 방문 시 대체) — 이번 결과가 나쁘면 후속 옵션 |
| v1_rich 상태 그대로 + prior 만 | 사용자 최종안 상태(저입도·일관성 규칙) 검증 자체가 본 실험 목적의 절반 |
| γ<0.9 포함 | 유효 horizon 이 결정 수(~100) 대비 과단축 — 근시안 정책은 greedy 와 구분 불가 예상 |

## 5. 한계 (사전 명시)

- prior 는 **즉시비용**, 학습 Q 는 **잔여 누적비용** — 스케일 불일치로 미방문 키가
  bootstrap min 에서 과소평가될 수 있음. γ<1 이 누적 스케일을 줄여 이 불일치를
  완화하는지가 grid 의 관찰 포인트 중 하나.
- 사용자 최종안이 명시한 대기열 맥락·종료 위치 부재는 이번 범위 밖 (다음 v3 축).

## 6. 비목표

- YR-027/028 결론 재해석 금지. 선박·미래정보·다중 YC 금지.
- 상태 v3(맥락 feature) 설계는 본 결과 해석 후 별도 사전등록.

## 7. 산출물

`outputs/reports/costq_v1final_hjnc/` — checkpoint_curve.json(γ별)·selections.json·
test_results.json·v1final_results.json·v1final_report.md. CLI: `run-costq-v1final`.
