# YR-097 — 본선-임계 우선순위 경험재생 (PER)

> 성형(YR-090)의 **짝**. 보상 밀도(성형)가 아니라 **표집 밀도**를 고친다. 근거·설계·수정 지점.
> 상위 index: [backlog.md](../../Dashboard/backlog.md). 관련: [[YR-090]] 성형 · [[YR-098]] 여지감사.

## 왜 (근거)

5각도 진단 + 적대검증 워크플로(2026-07-26)의 **탐색 커버리지** 렌즈가 경고했다:

> 보상 성형(YR-090)이 완료 임펄스를 촘촘히 되돌려도, **좋은 본선 행동 자체가 균일표집에서 묻히면 학습 안 된다.** 한 에피소드 ~170 결정 중 본선 결과 transition 은 배 2척(~1.2%)뿐. `rng.sample`([yr088_joint_rl.py:257](../../../src/yard_rl/experiments/yr088_joint_rl.py#L257))이 이 드문 신호를 촘촘한 트럭 transition 에 파묻는다. → **성형+표집을 함께 가야 "그래디언트도 맞고 표집도 되는" 상태.**

지금 YR-090 이 2/3 시드 무의로 기각 추세인 유력한 이유가 이 **표집 절반의 부재**다. 성형 단독의 한계를 표집으로 보완한다.

## 무엇을 고치나 (수정 지점 3곳)

베이스라인은 [yr088_joint_rl.py](../../../src/yard_rl/experiments/yr088_joint_rl.py). 새 파일 `yr097_prioritized_replay.py`로 **포크**(yr088 은 baseline 으로 보존). 재사용: `_sim`·`build_rows`·`RLPolicy`·`_eval`·`RC`·`CELLS`·`BASE`·`GAMMA`·`REF_S`·`N_STEP=1`.

### ① 버퍼 교체 — 균일 deque → 우선순위
[yr088:241](../../../src/yard_rl/experiments/yr088_joint_rl.py#L241) `replay = deque(maxlen=20_000)` →

```
class PrioritizedReplay:            # 병렬리스트 data/tag/prio
    # p_i(표집가중) = (|δ_i| + ε)^α · (1 + κ·tag_i)
    # α=0.6, κ=4.0, ε=1e-2, capacity=20_000
    add(trans, tag):  prio[pos] = max(prio) or 1.0   # 신규는 최대우선
    sample(batch, β): P = p/Σp; idx = multinomial(P, batch, replace=False)
                      is_w = (N·P[idx])^(−β);  is_w /= is_w.max()
                      return idx, data[idx], is_w
    update(idx, td_abs): prio[idx] = td_abs + ε      # 학습 후 |TD| 재기입
```

### ② 태그 계산 — 본선-임계 transition 표시
[collect_episode](../../../src/yard_rl/experiments/yr088_joint_rl.py#L99-L166)에서 각 transition 에 `tag∈{0,1}`:
- **(a)** 그 transition 구간에 `raw["vessel_delay"] > 0` (배 berth 비용 흡수 — [yr088:118·145](../../../src/yard_rl/experiments/yr088_joint_rl.py#L118)의 `raw`에서). **양하는 `cap=None`이라 vessel_delay≈0 → 이 태그는 사실상 LOAD 전용**(별도 스코핑 불필요).
- **(b)** 결정 시 LOAD 배들의 `flow_margin_s` 최소값 < 임계(예 150s) — STS 굶기 임박 하 트럭 vs 본선을 고른 **인과 결정**. `sim.vessels` 중 `work_type==LOAD`만 필터(양하 제외).
- `tag = (a) or (b)`. transition 을 6-튜플 `[rows,pos,r,gdt,rows_next,tag]`로.

### ③ 손실 IS 가중 + 우선순위 갱신
[train_step:189-207](../../../src/yard_rl/experiments/yr088_joint_rl.py#L189-L207) 포크:
- `loss_i = is_w_i · smooth_l1(q_i, y_i)` (편향보정)
- per-sample `|TD| = |q_i − y_i|` 반환 → `replay.update(idx, td_abs)`
- [run:257](../../../src/yard_rl/experiments/yr088_joint_rl.py#L257): `β = 0.4 + 0.6·ep/episodes` annealing, `rng.sample`→`replay.sample(batch, β)`

**바꾸지 않는 것 (중요):** 보상 바이트 동일(v6 프록시게임 불가)·`N_STEP=1` 타깃 무변경(n-step 분산증폭 회피)·태그는 **입력 아닌 라벨**(itc-v5 aliasing 원천 우회)·인코딩·JointPairNet·argmin·평가 numeraire·체크포인트 전부 불변.

## 조건 (적대검증이 단 필수 게이트)

1. **κ=0 ablation arm 병행** — 순수 TD-오차 PER(태그부스트 0)와 비교해 **태그의 순효과 분리**. 없으면 개선 귀속 불가.
2. **트럭대기 무회귀 하드게이트** — 작은 JointPairNet 용량을 본선 transition 으로 몰면 검증된 트럭 우위가 후퇴할 위험. 트럭 평균/P95 회귀 시 실패.
3. **주판정 = 3시드 훈련곡선 분산(재현성)** — 최종 berth 뿐 아니라 baseline 대비 시드간 분산 축소를 봄(성형과 동일 재현성 기준).
4. **완주·healthy 상시 감시** — 초기 β=0.4 구간에서 큰 태그 transition 이 n-step 붕괴(완주 깨짐)를 약화형으로 재현할 수 있음.

## 의존·순서

- **YR-090 판정 뒤 착수.** 기각이면 **성형 + PER 병행**(둘이 상보)을 새 실험으로. YR-090 이 조건부라도 통과면 PER 는 안정화 보강.
- **선결 스코핑**: [[YR-098]] 통제권한 감사로 "본선 여지가 적하에 실재"를 못박은 뒤 태그 (b)의 LOAD 필터를 확정.

## 기각군 (하지 말 것 — 적대검증 reject, 재탕)

- **값 분해(HRA)**: 전제("본선 신호가 작아 묻힘") 반증 — 본선이 numeraire 55~83%로 최대항. 공유 trunk 는 vessel 헤드가 선형 readout(flow_margin 재현), 별도 trunk 는 레이어(원칙위배).
- **제약 RL(Lagrangian)**: λ=33 이 현 baseline 과 대수적 동일 → v6 가중치 표류 재탕. λ>33 은 n-step 분산증폭과 동형.
- **신용 재타이밍(RUDDER)**: 위장된 프록시 재가중(실효 sts_wait≈69) = v6+n-step 동시 재현.

## 천장 (정직)

반응형 최대 레버(고정규칙 VesselFirst)도 tight 셀 berth −6~19분. **목표 = "고정규칙이 이미 잡는 여지를 학습형 단일 Q 가 재현"** 이지 본선 압도 아님.
