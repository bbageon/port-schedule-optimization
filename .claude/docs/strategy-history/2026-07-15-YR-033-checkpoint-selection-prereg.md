# YR-033 — checkpoint 선택 프로토콜 보완 사전등록

> 기록일: 2026-07-15 · 상태: **사전등록 — 본 실행 전 동결** · 사용자 승인: "해봐"
> 트랙: 단일 야드 greedy 격차 추격 (YR-034 통합전략과 별개) · spec: [YR-033](../dashboard-task-specs/YR-033-checkpoint-selection.md)
> 상위: [YR-012-c 결과](2026-07-15-YR-012-c-setfeat-prereg.md) · 근거: [YR-012-b winner's curse](2026-07-15-YR-012-b-delta-stable-prereg.md)

## 1. 결정 경위

1. YR-012-c: SetFeat[22] +0.035 [−0.034, +0.105] — greedy 통계적 동률(CI 0 포함)
   첫 달성이나 형식 승리 미달(CI 상한 +0.105>0). 선택은 60 checkpoint × val 30일
   argmin.
2. YR-012-b: val 최저 순위와 test 순위 **역전** 실측 (winner's curse). YR-012-c
   격차(+0.035)가 이제 checkpoint 간 노이즈 규모(±0.07)와 같아 **선택 규칙이
   동률→승리 전환에 직접 영향** 가능.
3. 사용자 지시 "해봐" — 선택 프로토콜이 범인인지 검증.

## 2. 가설과 판정

- **H (winner's curse)**: val 30일 argmin 은 val 노이즈에 과적합해 test 최적
  checkpoint 를 놓친다. val 표본 확대·평활화로 재선택하면 fresh test 에서 격차가
  줄고, 형식 승리(CI 상한<0) 가능성이 생긴다.
- **판정**: (a) 어떤 강건 프로토콜이 fresh test 에서 CI 상한<0 달성? (b) val-test
  Spearman 이 낮은가(선택 신뢰 부족 정량)? (c) P1 optimism = P1선택 test − 최적선택
  test > 0 인가(노이즈 손실 크기)?

## 3. 방법 — 학습기·feature·비용 불변 (spec 범위 준수)

| 요소 | 값 |
|---|---|
| 재실행 | YR-012-c setfeat 학습 **결정론 재현** (동일 train band 200000+3000·seed·scaler → **동일 60 checkpoint**). val eval 은 RNG 미사용이라 표본 확대해도 trajectory 불변 |
| checkpoint 저장 | 60개 전부 스냅샷 (in-memory deepcopy) |
| 선택 프로토콜 (val 만) | **P1_val30** = val 30일 argmin (YR-012-c 재현, 사니티: ep550 재현 기대) / **P2_val90** = val 90일(30일 superset) argmin / **P3_val90_smooth3** = 90일 val 3-ckpt 이동평균 argmin |
| val band | 210000+90 (YR-012-c 30일의 superset — 앞 30일이 P1 재현) |
| test band | **240000+100 (fresh)** — 220k(YR-012-c test) 재사용 금지 (재선택 오염 방지). 코드가 220k 겹침 거부 |
| test 사용 | **선택에 절대 미사용** (spec 범위 밖). test-per-checkpoint 는 진단(Spearman·optimism·최적선택 하한)에만 |
| 통계·판정 | greedy vs 각 프로토콜 선택 checkpoint, fresh test paired bootstrap 10,000. 형식 승리 = mean Δ CI 상한<0. guardrail 동반 |

## 4. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| 저장된 YR-012-c 단일 모델만 재평가 | 스냅샷 1개뿐 — 다른 checkpoint test 불가. 결정론 재실행이 유일한 복원 경로 |
| test 로 checkpoint 재선택 | **spec 범위 밖·통계 부정** (test 누출). test 는 진단·최종평가만 |
| 같은 test band(220k) 재사용 | YR-012-c 가 이미 관측 — 다중비교 오염. fresh 240k |
| val 무한 확대(수백 일) | 계산예산 과다. 90일이 30→3× 로 노이즈 절반↓ 기대, 비용 수용 범위 |
| 학습 재튜닝(lr·구조) | spec 범위 밖 — 이번은 순수 선택 방법론 |

## 5. 한계 (사전 명시)

- 결정론 재현 전제: val eval 이 학습 RNG 를 건드리지 않음(무작위 행동은 train ε
  뿐, val 은 argmin). 재현 실패 시 P1≠ep550 로 드러남 — 사니티로 보고.
- "최적선택 하한"(test argmin)은 **도달 불가 상한 참조** — 실제 선택은 val 만.
  이 하한이 greedy 미달이면 어떤 선택도 못 이김(방법론 무익)이란 진단.
- 90일도 유한 표본 — winner's curse 완화이지 제거 아님. 합성+assumed(YR-009 전).

## 6. 비목표

test 재선택 · 학습기/feature/비용 변경 · 집합 8 ablation · 다른 profile ·
YR-012-c 결론 재해석 · YR-034 통합전략 요소.

## 7. 산출물

`outputs/reports/setfeat_selection_hjnc/` — feature_scaler·seed_manifest·
checkpoint_records(ep별 val30/val90/test)·selection_results(프로토콜·paired·진단)·
selection_report.md. 구현: `experiments/setfeat_selection.py`·CLI
`run-setfeat-select [--quick]`. 예상 소요 ~50분 (60 ckpt × (90 val + 100 test) 평가).
