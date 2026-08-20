# v1 — 확률 정책 세대 (PPO)

> 세대 분리 2026-08-20. **설계 정본**: [`.claude/docs/architecture/README.md`](../../../.claude/docs/architecture/README.md)

## 한 줄

"이 트럭을 내놓을 **확률**"을 배운다. **규칙을 못 넘었다 (+97.94).**

## 이 폴더에 있는 것

| 파일 | 무엇 | 어디서 왔나 |
|---|---|---|
| `features.py` | `block_features` · `candidate_features` · `BLOCK_DIM` | `integrated/transfer_head.py` 분리 |
| `ppo_policy.py` | `TransferActor` · `TransferCritic` · `PpoSellPolicy` · `critic_input` · `build_rows` | 〃 |

`features.py` 는 v2 에도 같은 식의 사본이 있다. **의도된 중복**이다 — 이 세대를
얼려 두면 v3 를 아무리 고쳐도 지난 판정이 흔들릴 수 없다.

## 무엇을 세대 폴더에, 무엇을 무대에 두나

가르는 기준은 **"세대가 바뀌면 이게 달라져야 하나"** 하나다.

| | 어디 | 왜 |
|---|---|---|
| **판정에 쓰는 것** — 평가 잣대 Φ · 규칙 팔 · 에피소드 실행 · 시뮬레이터 | `integrated/` | **같아야 비교가 성립한다.** 판정이 짝비교라 세 세대가 같은 무대·같은 자를 받아야 한다 |
| **학습에 쓰는 것** — 특징 · 보상 라벨 · 정책망 · 학습 루프 | `v1/` `v2/` `v3/` | **달라지는 게 세대다.** 세대마다 사본을 갖는다 |

`shared/` 같은 중간 계층은 두지 않는다 — 만들면 결국 거기로 다 모인다
(사용자 지시 2026-08-20). 기계 검사: `tests/v3/test_v3_isolation.py`


## 아직 안 옮긴 것 (2차 대상)

세대는 v1 이지만 **다른 세대가 가져다 쓰는 코드가 섞여 있어** 아직 제자리에 있다.

| 파일 | 쓰는 곳 | v1 부분 | 공용 부분 |
|---|---|---|---|
| 파일 | 쓰는 곳 | v1 으로 | **무대**(`integrated/`)로 |
|---|---|---|---|
| `experiments/yr151_transfer_ppo.py` | 29곳 | `run_episode` · `build_batch` · `ppo_update` · `train_one` | `phi_terminal` · `PhiRecorder` · `load_adopted_execution_head` · `AdoptedExecFleet` · `load_kf` |
| `experiments/yr170_sell_ppo_diurnal.py` | 22곳 | `_episode_worker` · `train_parallel` · `train` | `run_episode_diurnal` · `KeepAllTrail` · `SpaceOnly` · `baseline` |
| `experiments/yr174_txn_reward.py` | 6곳 | **전부**(학습 잣대는 세대 것) — v2 도 사본을 갖는다 | — |
| `experiments/yr179_greedy_baseline.py` | 4곳 | — | **전부**(규칙 팔은 모든 판정의 기준) |
| `experiments/yr174_train.py` · `yr174_eval.py` | 0곳 | 전부 | — |
| `experiments/yr180_locked_scorecard.py` · `yr185_*.py` | 0곳 | 전부 | — |

**Φ·에피소드 실행·규칙 팔을 무대로 먼저 뽑아야** v1 을 옮길 수 있다.

## 판정 이력

| 판정 | 무엇 | 규칙 대비 |
|---|---|---|
| YR-180 | PPO 학습 팔 | **+97.94** |
| YR-185 | 병목 교정 (보상 0 이 91% → critic 붕괴 해소) | +40.90 |

## 왜 접었나

| | |
|---|---|
| 보상의 **91%가 0** | 판매가 드물어 학습 신호가 거의 없다 |
| **critic 붕괴** | 기준선이 무의미 → advantage 가 잡음 |
| 확률의 한계 | "얼마나 나쁜지"의 크기 정보가 버려진다 |

→ v2 는 확률 대신 **비용을 회귀**한다.

## 재현 주의

이 세대의 판정 산출물은 **파일이 옮겨지기 전 경로**로 만들어졌다.
과거 결과를 그대로 재현하려면 `repro_stamp` 에 박힌 `git_head` 로 체크아웃한다.
