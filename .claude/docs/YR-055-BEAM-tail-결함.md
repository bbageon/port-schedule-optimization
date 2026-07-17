# YR-055 — BEAM 2단 lookahead 무효 원인: tail 의 pending 조기 반환

> YR-045 파생 (BEAM==JointRollout 60/60 seed 완전 동일). 코드 원본은
> `src/yard_rl/integrated/baselines.py` `_tail`, 검증은
> `tests/integrated/test_yr055_beam_tail.py`.

## 1. 결론 (쉬운 말)

BEAM(두 시간창을 내다보는 강화 비교 정책)이 1단 rollout 과 완전히 같았던 것은 **결함**이었다.
두 번째 시간창 계산(`_tail`)이 시작하자마자 0 을 돌려줘, 모든 후보 분기가 같은 추가 점수(0)를
받아 순위가 한 번도 바뀌지 않았다. 수정 후에는 8/8 표본 seed 에서 실제로 다른 결정을 낸다.

## 2. 기전

1. 첫 시간창 rollout(`_rollout_cost`)은 시간 지평에 도달하면 **미해소 결정(pending)** 상태의
   시뮬레이터 사본을 반환한다 (마지막 `run_until_decision()` 이 연 결정을 적용하지 않은 채 중단).
2. 구 `_tail` 은 `scratch._pending` 이면 `dp = None` 으로 간주 → 비용 구간 0 을 더하고 즉시
   `break` → **tail ≈ 0**.
3. 전 분기 tail 이 0 → `cost + tail` 순위 == 1단 순위 → BEAM 의 선택 == JointRollout 의 argmin
   (동일 tie-break) → 결정·이벤트·비용까지 완전 동일.

수정: pending 결정을 `TerminalDecision(now, _pending)` 으로 복원해 base_policy 로 해소한 뒤
두 번째 창을 진행한다.

## 3. 수정 후 표본 측정 (진단 seed 310000~310007, locked 아님)

| seed | JR | BEAM(수정) | Δ |
|---|---:|---:|---:|
| 310000 | 62.558 | 60.059 | −2.499 |
| 310001 | 76.588 | 78.157 | +1.569 |
| 310002 | 65.805 | 59.121 | −6.685 |
| 310003 | 72.385 | 66.708 | −5.676 |
| 310004 | 78.067 | 73.872 | −4.196 |
| 310005 | 74.437 | 78.711 | +4.274 |
| 310006 | 60.306 | 63.758 | +3.452 |
| 310007 | 80.453 | 83.789 | +3.335 |

- 동일 0/8 (동일성 해소 확인), 완주 8/8. 평균 Δ = **−0.80** — 개선 4, 악화 4.
- 해석: 2단 lookahead 는 이제 실제로 개입하지만, SF-SPT base 정책 위의 rolling-horizon 특성상
  이득이 **소폭·불안정**하다. "강 baseline" 이라 부를 일관 우위는 이 표본에서 확인되지 않았다
  (YR-044 의 지위 문구는 재평가 대상 — 판정은 필요해질 때 전용 실험으로).

## 4. YR-045 결과에 대한 함의

- locked run 의 BEAM 열은 **JointRollout 의 복제**였다 — BEAM 고유 정보가 없다. YR-045 의
  판정(게이트 baseline = JointRollout)에는 영향 없음: BEAM 은 게이트 기준이 아니었다.
- 60 seed × 6조건 BEAM 재측정(~3시간)은 수행하지 않았다 — 참고용 baseline 재측정의 가치가
  비용 대비 낮고, 표본 8 seed 로 "결함·수정·불안정 이득" 판별이라는 YR-055 의 목적은 달성됐다.
  RL 재감사(YR-054)·후속 실험에서 BEAM 을 쓰게 되면 그때 전수 측정한다.

## 5. 검증

- `test_tail_advances_past_pending_decision` — pending scratch 에서 tail > 0 (구현 회귀 고정).
- `test_beam_can_diverge_from_joint_rollout` — 5 seed 내 최소 1개 분기 발생.
- baselines·YR-045 하네스 스위트 통과 (JointRollout·건전성 계약 등 기존 계약 불변 —
  `_tail` 은 BEAM 전용 경로라 JR·SF·FIFO 결과에 영향 없음).
