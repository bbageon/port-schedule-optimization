# YR-033 — checkpoint 선택 프로토콜 보완 사전등록

> 기록일: 2026-07-15 · 상태: **완료 — winner's curse 기각, 격차는 실재(+0.111 robust)** · 사용자 승인: "해봐" · 사전등록 원문(§1~§7) 불변, 결과는 하단 append
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

---

## 실행 결과 (2026-07-15 append — 사전등록 원문 불변)

- 실행: clean source `ed13d2f`, 소요 45.4분. 결과 커밋 `dc2fd00` ·
  [리포트](../../../outputs/reports/setfeat_selection_hjnc/selection_report.md)

### 판정: winner's curse **기각** (H 반증) — 선택은 이미 최적이었다

| 프로토콜 | 선택 ep | fresh test Δ vs greedy [95% CI] | 형식승리 |
|---|---|---|---|
| greedy (240k) | — | 6.948분 (기준) | — |
| P1_val30 (YR-012-c 재현) | **550** | +0.111 [+0.045, +0.182] | 미달 |
| P2_val90 (확대) | 550 | +0.111 [+0.046, +0.182] | 미달 |
| P3_val90_smooth3 | 600 | +0.341 [+0.262, +0.422] | 미달 (오히려 열세) |
| **최적선택 하한** (test argmin, 도달불가) | 550 | **+0.111** | 미달 |

- **val-test Spearman +0.935(val30)·+0.958(val90)** — val 이 test 를 거의 완전
  예측. 선택 노이즈 문제 없음.
- **P1 optimism +0.000** — val30 이 고른 ep550 이 곧 test 최적 checkpoint.
  개선할 winner's curse 가 애초에 없었다 (P1=P2=최적선택 전부 ep550 일치).
- P3(평활화)는 ep600 을 골라 **오히려 열세** — 근시안 처방이 무익.

### 핵심 발견 — YR-012-c 의 "동률(+0.035)"은 test-band draw 였다

- **동일 checkpoint ep550** 이 220k(YR-012-c test)에선 +0.035(CI 0 포함, 동률)인데
  **fresh 240k 에선 +0.111** [+0.045,+0.182] (유의 열세). greedy 절대값도
  7.472(220k)→6.948(240k) 로 band 마다 다름.
- 즉 SetFeat[22] 의 **robust 격차는 ~+0.08~0.11**, YR-012-c 의 +0.035 는 유리한
  test 추첨. **단일 band 평가가 정책을 과대평가**할 수 있다는 방법론 교훈 —
  향후 다중 band(또는 더 큰 test) 필수.
- p95 guardrail 도 240k 에선 실패(+5.7~9.9%>5%) — 220k(+2.6%)와 상반. tail 개선도
  band 종속.

### 파생 결정

1. **선택 프로토콜 트랙 종료** — Spearman 0.96·optimism 0 으로 선택은 병목이
   아님이 확정. YR-032/033 계열 닫음.
2. **최적선택 하한(+0.111)이 greedy 미달** — 현 checkpoint 집합 안에서는 어떤
   선택으로도 이 문제에서 greedy 를 못 이김. 남은 것은 (a) 정책 자체의 표현/학습
   개선(구조는 H-B 기각이라 학습 알고리즘 축) 또는 (b) 근본적으로 greedy 가
   near-optimal 인 문제라는 수용.
3. **평가 방법론 시정**: 이후 단일 야드 실험은 다중 test band 또는 test≥300일로
   band-draw 방어 (backlog 등록).
4. **가치 재정리**: RL 의 순가치는 "greedy 대비 평균 승리"가 아니라 (i) 220k 에서
   관측된 tail(p95) 개선 (band 종속이나 방향 유효) (ii) oracle 상금 +0.182 중
   회수 잠재. 통합전략(YR-034)의 다요소 비용에서 RL headroom 재탐색이 본류.
