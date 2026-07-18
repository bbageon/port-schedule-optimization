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

## 6. 결과 (실행 후 기입)

(예정)
