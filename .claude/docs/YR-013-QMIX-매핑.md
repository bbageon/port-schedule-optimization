# YR-013 — QMIX 협조학습 매핑 (설계, 2026-07-18)

> 원칙 원본: [06 §6](../../docs/구현계획/06_동적후보_Deep_Q_다중YC.md) (resolver 하 CTDE·
> 사후 교체 기록 금지·Double joint target). 착수 근거: YR-054(조정 실패 85%) +
> YR-056(관측 경량책 무효 — 구조적 해법만 잔존, 사용자 승인 2026-07-18).

## 1. 무엇이 바뀌나 (쉬운 말)

지금 RL 은 크레인마다 "내 점수표"를 따로 학습한다 — 팀 성적 개념이 없어 서로 부딪힌다
(interference 가 격차의 85%). QMIX 는 **실행은 지금 그대로** 두고 **학습만 바꾼다**:
두 크레인의 점수를 혼합기(mixer)로 합쳐 "팀 총비용"과 맞추도록 함께 역전파한다.
혼합기는 단조(한 크레인 점수가 나빠지면 팀 점수도 나빠짐) — 그래야 각자 argmin 해도
팀 최적과 일치(IGM). 상황별 혼합 비중은 전역 상태가 결정(hypernetwork).

## 2. 우리 환경 적응 3가지 (설계의 실질)

1. **resolver 하 QMIX (06 §6 준수)**: 개별 utility 는 후보 점수 계산에만 쓰고, 실제 배정은
   중앙 resolver 가 feasibility(동일 Job·corridor·비통과) 최종 승인 — 기존 QPreference 경로
   그대로. replay 에는 **resolver 가 실제 실행한 joint action** 을 저장 (사후 교체 기록 금지).
2. **SMDP 비동기 joint 전이**: 결정 시점 참여 크레인이 1~2대로 가변. joint 표본은
   결정 k → 다음 결정 k+1 구간 — 팀 구간비용 c_k, γ_dt=γ^{Δt/ref}. 참여 1대면
   mixer 가 1-입력으로 자연 퇴화 (presence mask — 부재 슬롯은 값·가중 모두 0).
3. **Double joint target**: 다음 결정에서 각 참여 크레인 online argmin(mask 내) →
   target agent 망 평가 → target mixer 합성. `y = c + γ_dt · Q_tot_target`, terminal 은 y=c.

## 3. 구조 (integrated/qmix.py)

```text
agent 망: CandidateQNet(dueling) 1개를 전 크레인 공유 (기존과 동일 원칙)
MonotonicMixer: Q_tot = |w2(g)|ᵀ·ELU(Σ_i |W1(g)|_i·q_i + b1(g)) + V(g)
  g = 전역 feature(itc-v3 — COORD 포함) + presence 플래그. |·| 로 단조 보장.
JointSample(encs, action_pos, c_disc, gamma_dt, next_encs) — 결정 단위
QmixLearner: run_episode 와 duck-type 호환(scores_for/learn_step/cfg) —
  실행 경로 코드 재사용, run_episode 에 joint_sink 인자만 추가(기존 거동 불변)
```

## 4. 판정 실험 (yr013_qmix_experiment.py)

- **arms**: QMIX vs INDEPENDENT(동일 dueling·mixer 없음) — **둘 다 COORD on** (정보 동일,
  차이는 구조만 — YR-056 과 직교 분리) + JointRollout(forbid) 기준선.
- 동일 예산 500ep·val 20 checkpoint·test 60, 신규 대역 530k/540k/550k (소각대역 가드).
- 판정: QMIX vs INDEP paired 로 **interference·총비용 감소** 여부 (협조 학습 효과 분리).
  JR 대비는 별도 보고 (넘으면 게이트 재도전 근거). 위반 0·완주 100% 는 상시 게이트.
- YR-052 반영: 전략 WAIT 는 양 arm 제외(기본). 구조 양보 WAIT 은 표본 포함(YR-043 계약).

## 5. 검증 (tests/integrated/test_yr013_qmix.py — 06 §11 대응)

mixer 단조성(수치)·부재 슬롯 불변성·terminal 표적 y=c·반복 학습 loss 감소·
joint 스티칭 창 수식·quick e2e(완주·replay 충전·loss 유한)·결정론.

## 6. 결과 (2026-07-18, phase-1 500ep + 예산사다리 500/1000/2000)

**phase-1 (lr 1e-3, 500ep)**: QMIX 유의 열세 — total 110.03 vs INDEP 85.25
(Δ+24.77 [+21.30,+28.28]) · interference 40.48 vs 24.87 · **val 곡선이 ep150 최저(115)
후 발산(147~155)** — "수렴 느림"이 아니라 학습 불안정.

**예산사다리 (사용자 지시 — lr 3e-4 안정화 양 arm 동일, 2000ep 단일 궤적 tier 판정)**:

| tier | QMIX total | INDEP total | Δ (QMIX−INDEP) | QMIX interference |
|---|---:|---:|---:|---:|
| @500 | 116.80 | 86.12 | +30.68 [+27.15,+34.13] | 41.29 |
| @1000 | 116.80 | 86.12 | +30.68 (checkpoint 동일 — 500~1000 개선 0) | 41.29 |
| @2000 | 104.80 | 85.57 | **+19.23 [+15.85,+22.67]** | 37.08 (WAIT 146 — 폭증) |

- **판정**: ① 안정화(3e-4)로 발산은 소멸, ② 예산 반응 있음 (1000→2000 구간 −12) —
  구조 무가치 단정은 이르나, ③ **4배 예산에도 INDEP 를 넘기는커녕 유의 열세 +19.2**,
  INDEP 는 500 에서 이미 포화(86) — 예산만으로 역전 불가 추세. ④ @2000 은 WAIT 146
  (INDEP 38·JR 11) — 비용은 줄었지만 협조가 아니라 회피성 양보 패턴 (학습신호 비정상 시사).
- **원인 1차 용의자 = 입력 표현**: 미정규화 feature(초·미터 등 스케일 수백~수만 혼재)가
  mixer hypernetwork 입력으로 직행 — 발산·저학습의 표준 원인. 대응책이 board 에 등록됨:
  **YR-059**(상태 feature scale-only 정규화·클리핑 → QMIX 재실행, [적용전략](상태정규화-보상가중치-적용전략.md))
  · YR-060(PopArt, 조건부).
- **처분**: 현 입력 표현에서 QMIX **미채택**. YR-013 은 YR-059 선결 후 재판정 —
  구현물(mixer·joint 스티칭·사다리 하네스)은 그대로 재사용. 원자료:
  `outputs/reports/yr013_qmix/`(phase-1)·`yr013_qmix_ladder/`(사다리, JR 재사용).

## 7. 3차 (2026-07-19) — 차분 표적 QMIX: G1 통과, QMIX 계열 첫 유의 기여

2차 기각(입력 표현) 후 신용 축 종합(YR-061~067)을 승계해 **명시 신용 앵커 + mixer
팀 보정** 결합 ([사전등록](strategy-history/2026-07-19-YR-013c-차분표적QMIX-prereg.md)):
`L = Σ huber(Q_i, D_i) + 1.0·huber(Mixer(Q, g_norm), C_W_team)`, 1-step·target 망 없음.

- **G1 통과**: 65.97 vs DIFF2400_NORM 75.38 — Δ −9.41 [−12.81, −5.86]. 차이는 mixer
  항 하나 → 순수 mixer 효과. 계열 궤적: 1차 +24.8 악화 → 2차 +21~29 악화 → **3차 −9.4
  유의 개선**. 비-BC 학습 신기록·FIFO(65.43) 동급.
- **G2 미달**: vs SF_SPT +12.85 [+10.06, +15.62]. guard 통과 (swa 0.424·완주 100%).
- **교훈**: mixer 는 "신용을 처음부터 배우는 도구"가 아니라 "명시 신용이 놓친
  교차항을 보정하는 도구"일 때 작동한다. 원자료: `outputs/reports/yr013c_diff_qmix/`.
