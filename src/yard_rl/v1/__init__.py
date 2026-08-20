"""v1 — 확률 정책 세대 (PPO).

■ 한 줄
  "이 트럭을 내놓을 **확률**"을 배우는 정책. **규칙 대비 +97.94 로 졌다.**

■ 무엇이 핵심이었나
  블록마다 후보 K개 + KEEP 1개를 행으로 만들어 actor 가 점수를 내고 softmax 로
  뽑는다. critic 이 기준선(V)을 대고 PPO 로 갱신한다.

■ 무엇이 무너졌나 (v2 로 넘어간 이유)
  · **보상의 91%가 0** — 판매가 드물어 학습 신호가 거의 없다
  · **critic 붕괴** — 기준선이 무의미해져 advantage 가 잡음이 된다
  · 확률로 고르면 "얼마나 나쁜지"의 크기 정보가 버려진다
  → v2 는 확률 대신 **비용**을 회귀한다 (`yard_rl.v2`).

■ 이 패키지의 것 — v1 은 v1 것만 갖는다
  features.py     block_features · candidate_features (**이 세대 전용 사본**)
  ppo_policy.py   TransferActor · TransferCritic · PpoSellPolicy · critic_input · build_rows

  v2 도 같은 특징 식을 쓰지만 **사본을 따로 갖는다**. 세대를 얼리기 위한
  의도된 중복이다 — v3 를 고쳐도 이 세대의 판정은 흔들릴 수 없다.

■ 무대는 공유한다
  `yard_rl.integrated`(엔진·비용·레이아웃·배정기)는 세대별로 복제하지 않는다.
  판정이 **짝비교**라 세 세대가 같은 무대를 받아야 비교가 성립한다.

■ 세대 목록과 판정 이력: `MANIFEST.md`
"""

__all__: list[str] = []

GENERATION = "v1"
