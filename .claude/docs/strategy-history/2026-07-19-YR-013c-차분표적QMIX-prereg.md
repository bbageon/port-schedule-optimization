# YR-013c 사전등록 — 차분 표적 QMIX: 명시 신용 앵커 + mixer 팀 보정 (2026-07-19)

> 실행 전 동결. 결과·해석은 `## 실행 결과` 절로 append (원문 수정 금지 — 규약 준수).
> 배경: QMIX 1·2차 — 암묵 분해(팀비용 표적 mixer)는 예산 4배·정규화에도 실패
> ([YR-013 매핑 §6](../YR-013-QMIX-매핑.md)·[YR-059 §4](../YR-059-상태정규화-매핑.md)).
> 차분 명시 신용은 성공 (YR-063 탈퇴화 → YR-065 창 2400s → YR-067 ×정규화 75.38 신기록).
> 본 실험 = 두 패턴의 교차점 (README "차분 표적 QMIX", 사용자 승인 2026-07-19).

## 1. 가설 H-013c

차분 D_i 는 "상대 행동 고정" 반사실이라 **크레인 간 상호작용 교차항을 놓친다**
(둘 다 바꿨을 때의 효과는 D_1+D_2 로 표현 안 됨). 단조 mixer 가 팀 창비용과의
정합을 **보정항**으로 학습하면 — utilities 는 D_i 앵커로 고정된 채 — 상호작용
정보가 통제된 용량으로 utilities 에 유입되어 DIFF×NORM(75.38)을 넘는다.

## 2. 처방 (동결)

- **손실**: `L = Σ_i huber(Q_i(s_i,a_i), D_i) + λ_mix · huber(Mixer(Q_1..Q_n, g), C_W_team)`
  - D_i = YR-063 정의 그대로 (창 C_W: 실제 − counterfactual WAIT_i, 상대 고정,
    base_policy=SF_SPT resolver, WAIT 선택 시 D=0 앵커).
  - C_W_team = 같은 결정의 실제 창 팀비용 (`_rollout_cost` actual — D 계산의 부산물,
    **추가 rollout 0회**).
  - **λ_mix = 1.0** (primary 단일 arm — knob 탐색은 하지 않는다).
  - 기울기는 두 항 모두 agent 망까지 흐른다 (mixer 항이 상호작용 신호 전달 통로).
- **1-step·부트스트랩 없음** (γ_dt=0): YR-061~063 판정 승계 — TD 부트스트랩이 희석
  통로였고 D·C_W 는 자기완결 표적. **target 망 자체가 불필요** (1·2차 QMIX 와 차별점).
- **mixer**: 기존 MonotonicMixer 재사용 (hypernet |W|·presence mask·V(g)) —
  g = state_norm 적용된 전역 feature + presence (YR-059 교훈: mixer 입력도 O(1) 스케일).
- **실행 경로 불변**: 평가·운영은 agent Q 의 per-crane argmin + 중앙 resolver
  (run_episode, state_norm) — mixer 는 학습 전용. YR-052(전략 WAIT 제외) 기본 적용.

## 3. 동결 설정

- YR-067 DIFF2400_NORM arm 승계 (차이는 mixer 항 **하나**): Yr061Config 동결값
  (train 600000~600149 150ep·val 610k 8·test 620k 20·외부 16 진단 시나리오·ddqn·
  lr 1e-3·ε=1/√ep·checkpoint 15ep·val 실비용 최소 선택)·창 2400s·
  state_norm P90 재적합(fit 5-seed, val/test 미접촉)·cost_scale=1.0(D 는 O(1)).
- 학습기 seed 63,000·탐험 rng 63,100 (DIFF arm 과 동일 — 초기화 교란 최소화).
  mixer embed 32. 하네스: `experiments/yr013_diff_qmix.py` (`run_yr013c`).
- 비교군 (동일 test 대역 620000~620019 기존 판정 행 재사용):
  **DIFF2400_NORM(75.38 — 직접 ablation 대조군)**·SF_SPT(53.12)·CONTROL_TD(70.11)·FIFO(65.43).
  paired bootstrap 10k (seed 75,113).

## 4. 사전 판정 기준

- **G1 (primary — mixer 보정 이득)**: vs DIFF2400_NORM paired Δtotal CI 상한 < 0.
- **G2 (트랙 목표)**: vs SF_SPT CI 상한 < 0 — 학습이 규칙을 넘는 순간.
- **guard**: test swa ≥ 0.25·완료율 ≥ 0.95 (탈퇴화 유지 — 미달 시 성능 무관 기각).
- **실패 조건**: G1 미달 → "mixer 경유 상호작용 보정" 가설 기각. QMIX 계열은
  1차(팀표적)·2차(정규화)·3차(차분 앵커) 소진 — YR-013 종결 판정 재료로 박제하고
  협조 개선의 잔여 경로(YR-066 차분 개량·중앙 공동가치 등)는 별도 등록.
- 노이즈 주의: 단일 학습 궤적·소형 시나리오 — 판정은 CI·swa 축.

## 실행 결과 (2026-07-19 기록 — 원자료 `outputs/reports/yr013c_diff_qmix/`)

- 실행: commit `8761f6c` clean tree, WSL. 선택 = ep120 (val 64.11). val 곡선은 요동
  (64~77)하나 1·2차 같은 발산 서명 없음 — D 앵커가 표류를 막는다는 설계 의도대로.
- **G1 통과 — mixer 상호작용 보정 이득 입증**: test total **65.97** vs DIFF2400_NORM
  75.38, **Δ −9.41 [−12.81, −5.86]** (CI 상한 < 0). 차이는 mixer 항 하나였으므로
  개선은 순수 mixer 효과. **QMIX 계열(1차 +24.8 악화 → 2차 +21~29 악화 → 3차 −9.4
  개선) 첫 유의 기여 — H-013c 지지.** 비-BC 학습 신기록 75.38→65.97, FIFO(65.43)
  동급 도달, vs CONTROL_TD −4.14 [−9.09, +0.91] (점추정 우세·유의 미달).
- **G2 미달**: vs SF_SPT +12.85 [+10.06, +15.62] — 규칙 초과는 실패. guard 통과
  (swa 0.424·완료율 1.000·backlog 0).
- **해석**: "명시 신용(앵커) + 암묵 보정(mixer)" 분업이 유효 — 암묵 단독은 신호가
  희석되어 실패했고, 명시 단독은 교차항을 놓쳤으며, 결합이 두 실패를 상쇄했다.
  잔여 격차 +12.85 의 다음 레버: 본 시나리오(외부 40) 확전 검증·λ_mix/창 민감도
  (knob 탐색은 prereg 금지 — 별도 row)·YR-066 차분 개량.
- **판정: H-013c 지지 (G1) / 트랙 목표(G2) 미달** — YR-013 은 "QMIX 기여 입증"으로
  종결, 확전·개량은 후속 row 로.
