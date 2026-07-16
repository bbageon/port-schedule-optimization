# YR-044 — 총비용용 baseline·행동분포 건전성 (single source: 코드)

> [YR-039 무효 판정 §6.2](strategy-history/2026-07-15-YR-039-무효판정-imbalance-지배.md) 의 baseline 재설계.
> **비교 기준이 퇴화하면 어떤 승리 주장도 성립하지 않는다** — 무효 사유 2 를 코드로 제거.
> single source: `src/yard_rl/integrated/baselines.py`. 후속: YR-045(정정판 locked 재실험).

## 1. 무엇이 문제였나 (쉬운 말)

YR-039 가 RL 과 비교한 상대(baseline)는 "가장 짧은 **행동**"을 고르는 규칙이었다. 그런데 크레인을
그냥 옮기는 REPOSITION 은 실제 작업(SERVE)보다 늘 짧다 → 상대는 결정의 54%(val 5-seed)~81%(test seed) 를 "일 안 하고
이동"에 썼다. 그런 상대를 이긴 건 의미가 없다.

## 2. 산출물

| 구성 | 역할 |
|---|---|
| `JointRolloutGreedy` | **1차 baseline**. 두 YC 의 공동 feasible 행동 조합을 열거하고, 각 조합을 **고정 시간창(horizon_s) 누적비용**으로 평가해 argmin. 행동 후 base_policy(ServiceFirstSPT)로 시간창 끝까지 진행 → base 위의 1-step 정책개선(rollout algorithm) |
| `BeamLookahead` | **강 baseline**. 위를 width W 로 가지치기해 시간창 2개까지 확장 (rolling-horizon) |
| `JointImmediateCostGreedy` | **진단 전용** — §6.2 문자 그대로의 "즉시비용(다음 결정까지) argmin". 아래 §3 이유로 퇴화 |
| `ServiceFirstSPTPreference`·`FIFOPreference` | 보조 진단군 (실작업 우선 → 최단 / 선착순) |
| `ActionMix`·`assert_healthy_action_mix` | **행동분포 건전성 계약** — 퇴화 사전 검출 |
| `run_joint_episode` | 전 정책 **공통 드라이버** — 동일 정보·후보·제약·비용 config (공정 비교) |

`ResolverPolicy` 가 Preference 계열을 joint 정책 인터페이스로 감싸 같은 드라이버를 태운다.

## 3. 발견 — 명세된 "즉시비용 argmin" 자체가 퇴화한다

§6.2 는 1차 baseline 으로 "즉시비용 argmin" 을 지정했으나, **문자 그대로 구현하면 YR-039 SPT 와
같은 함정에 빠진다**. 즉시비용을 *다음 결정까지의 구간*으로 재면 짧은 REPOSITION(수십 초)은 대기·
혼잡 rate 비용이 덜 쌓이고 긴 SERVE(수백 초)는 더 쌓인다 → **짧은 행동이 체계적으로 이긴다**.
"최단 행동" 편향에 시간 대신 비용으로 도달한 것.

**실측 (seed 310000, λ=1.0 중립, YR-043 정정 비용)**:

| 정책 | 총비용 | 평균대기 | 완료율 | 실작업 가능시 SERVE 선택 | 판정 |
|---|---:|---:|---:|---:|---|
| ServiceFirstSPT (base) | 96.6 | 0.35분 | 100% | 0.57 | 건전 |
| FIFO | 97.0 | 0.43분 | 100% | 0.56 | 건전 |
| VesselWait | 104.0 | 0.93분 | 100% | 0.55 | 건전 |
| YR-039 SPT (퇴화 재현) | 132.2 | 3.23분 | 100% | **0.08** | 🚫 퇴화 |
| 즉시비용 argmin (§6.2 문자대로) | **863.2** | **119.8분** | **41%** | **0.03** | 🚫 퇴화 |
| **JointRolloutGreedy (h=600)** | **93.4** | 0.89분 | 100% | 0.46 | 건전 ✅ |

→ **고정 시간창**으로 재면 모든 분기가 같은 시간축에서 비교돼 길이 편향이 사라진다. 그 결과
1차 baseline 이 base 정책보다 총비용이 낮다(93.4 < 96.6) — 정책개선이 실제로 작동.

## 4. 행동분포 건전성 계약

```python
ActionMix.serve_when_available()   # 실작업이 가능했을 때 실제로 SERVE 를 고른 비율
assert_healthy_action_mix(mix, min_serve_when_available=0.25, max_nonserve_share=0.60)
```
- **문턱 보정 근거**: 퇴화 **0.03~0.08** vs 건전 **0.46~0.57** — 두 무리가 뚜렷이 갈리므로 그 사이
  0.25 를 문턱으로. (초안 0.50 은 건전한 JointRollout(0.46)까지 잘못 잡아 재보정.)
- **성능 게이트가 아니라 퇴화 검출기**다 — 좋은 정책이 선제 위치조정을 섞는 것을 벌하지 않는다.
- 총비용 표에는 안 보이던 퇴화(REPOSITION 81%)가 이 지표에는 즉시 드러난다 (무효 판정 §8 교훈).

## 5. 부수 수정 — mandatory 초과 시 크래시 (rollout 이 드러냄)

혼잡 상태에서 SLA 임박(mandatory) 트럭이 12대가 되자 `_prune` 이 `K_TOO_SMALL` 로 **에피소드를
크래시**시켰다. "조용한 유실 금지" 의도는 유실 0 이지 크래시가 아니다 → **후보칸을 늘려 mandatory
전량 보존**(`budget` 초과 허용, padding 은 `max(k_max, n_real)`). 후보 수는 가변이고 Q망은 후보별
공유 점수 구조(YR-031-b)라 K 확장이 안전하다. 이 버그는 baseline rollout 이 혼잡 상태로 깊이
들어가면서 처음 드러났다 — YR-045 재실험 전에 반드시 필요한 수정.

## 6. 검증

`tests/integrated/test_baselines.py`: 건전성 계약이 YR-039 퇴화 SPT 를 잡음 · ServiceFirstSPT/FIFO 통과 ·
즉시비용 greedy 퇴화 고정(문서화) · JointRollout 건전+base 이하 비용 · 결정론 · beam 완주 ·
전 정책 동일 제약·완주. `tests/unit/test_candidate_generator.py`: mandatory 전량 보존(K 확장)·
non-mandatory budget 준수.

## 7. 범위 밖

- **YR-045**: 정정판 locked 재실험 (신규 seed·§18 다중 게이트). 본 트랙은 baseline 과 계약만 제공.
- `candidate_dqn_experiment.BASELINES` 의 퇴화 `SPTPreference` 교체·RL 재실행은 YR-045.
- 실측 scale·λ 밴드: YR-002/009·YR-041. 전 항목 assumed 유지.

> ⚠ 환경: torch DLL(`_ctypes`)이 애플리케이션 제어 정책에 차단돼 torch 의존 테스트는 미실행
> (본 트랙 코드는 torch 비의존). YR-045 전 복구 필요.
