# YR-213 — 크레인 층 비교군을 문헌 표준으로 (오라클은 진단 열로)

- **Epic**: Exp / **Priority**: 🔴 / **등록일**: 2026-08-22
- **3대 게이트 보정 대상**: `performance`
- **사용자 지적 2026-08-22**: *"오라클을 정책이 넘어서 따라잡는 건 물리적으로
  비현실적에 가깝다. 크레인 정책의 비교군을 **기존 최적화 알고리즘**과 그대로
  진행했을 때로 두는 게 논문에서 쓰기 좋다."*
- **아키텍처**: [03 결정층](../architecture/03-결정층.md) §2
- **1줄**: 크레인 층의 비교 상대를 **규칙 하나**에서 **규칙 + 계획법(메타휴리스틱)**
  으로 넓히고, 오라클은 **비교군이 아니라 진단 열**로 위치를 정정한다.

## 우리 오라클이 정확히 무엇인가

`experiments/oracle_gap.py` (YR-031) 정의:

```
정보 반칙:  완벽 ETA (eta_error = 0)
계산 반칙:  lockstep beam search (BEAM wide)
```

**넘어야 할 상대가 아니라 "애초에 상금이 얼마인가"를 잰 진단이다.**
[[YR-070]] 이 이미 답했다 — **완벽정보 이득 0.0**(중간·고부하 둘 다), 헤드룸은
**조율**에서만 6~9%, **SF-SPT 가 천장에 근접**.

지금 문서가 *"오라클 상한 +0.18분"* 을 비교군처럼 읽히게 써놨다. 위치를 고친다.

## 문헌 표준 — 이 분야 논문의 비교군

| 층위 | 무엇 | 역할 |
|---|---|---|
| **규칙** | FCFS/FIFO · SPT · EDD · Nearest | 현장 관행 바닥 |
| **메타휴리스틱** | GA · GA-TS 하이브리드 · Tabu Search | **진짜 경쟁 상대** |
| **정확해** | Branch-and-Bound | 소규모에서 **최적성 격차** |
| 오라클 | 완벽 정보 + 무제한 탐색 | 진단용 상한 (비교군 아님) |

야드 크레인 스케줄링에서 B&B 는 **상·하한을 만들어 최적성 격차를 재는 데** 쓰이고,
대규모 인스턴스는 **빠른 휴리스틱과 개량 GA** 로 근사해를 낸다. DRL 논문들도
**GA 와 규칙 기반**을 상대로 보고한다(예: GA 대비 해 품질 약 16% 개선·계산시간 절반,
FCFS 대비 지연 40% 이상 감소).

**출처**
- [Yard crane scheduling in port container terminals using genetic algorithm](https://www.researchgate.net/publication/200581107_Yard_crane_scheduling_in_port_container_terminals_using_genetic_algorithm)
- [Scheduling of Different Automated Yard Crane Systems at Container Terminals (Transportation Science)](https://pubsonline.informs.org/doi/10.1287/trsc.2016.0687)
- [Scheduling multiple yard cranes in two adjacent container blocks](https://www.sciencedirect.com/science/article/abs/pii/S036083521930405X)
- [Deep RL-based dynamic integrated scheduling of AGVs and yard cranes](https://www.sciencedirect.com/science/article/abs/pii/S0952197625029434)
- [Yard Crane Scheduling Method Based on Deep Reinforcement Learning](https://qikan.cmes.org/jxgcxb/EN/10.3901/JME.2024.06.044)
- [Optimizing quay crane scheduling using DRL with hybrid metaheuristic](https://www.sciencedirect.com/science/article/abs/pii/S0952197625000211)

## ★계획법 원본이 이미 우리 코드에 있다

```python
integrated/baselines.py
    JointRolloutGreedy           # 계획법 — 채택 망이 여기서 증류됐다
    BeamLookahead                # 빔 탐색
    JointImmediateCostGreedy     # 즉시비용 탐욕
```

채택 크레인 망의 계보가 **규칙 → 증류(`JointRolloutGreedy`) → PPO → 채택** 이다.
즉 **증류 전 원본이 그대로 남아 있다.** 새로 구현할 게 없다.

## 지금 왜 문제인가

```
A − R  (얼린 크레인 망 − 규칙 SF-SPT)   Φ +59.97 [7.25, 112.69]  t=+2.42   WORSE
                                        대당 +1.00분
```

[[YR-180]]: *"얼린 채택 크레인 정책이 이 무대에서 현행 규칙보다 유의하게 비싸다.
연구 최초 측정이고 사전등록한 세 시나리오 중 최악이 나왔다."*

**바닥이 새는 상태에서 재배치 이득을 재고 있다.** 오라클 대비 남은 여유는
0.18분/대인데 **실제로 흘리는 것이 1.00분/대** — 5.5배다.

## 무대의 배차를 무엇으로 둘 것인가 — 세 안

| | 배차 | 장점 | 단점 |
|---|---|---|---|
| **A** | 규칙 `SF-SPT` | 단순 · [[YR-180]] 최고 조합이 `R_S` 였다 | 계획법보다 약한 바닥 |
| **B** | **계획법 `JointRolloutGreedy`** | **문헌 표준** · 코드 있음 · 바닥이 튼튼 | 느리다(rollout) — 측정 필요 |
| C | 크레인 재학습 | 최종 목표 | [[YR-184]] 대규모 작업 |

**B 를 권한다.** *"이미 잘 최적화된 배차 위에서도 재배치가 이득이다"* 가
*"규칙 배차 위에서 이득이다"* 보다 훨씬 강한 주장이고, 문헌 비교군과도 맞는다.

**A 도 같이 남긴다** — 규칙은 어느 논문이든 있는 바닥이라 빼면 안 된다.

## 판정 계약

| | |
|---|---|
| 크레인 팔 | `SF-SPT`(규칙) · **`JointRolloutGreedy`(계획법)** · 얼린 망(이력) |
| 재배치 팔 | [06](../architecture/06-학습과-판정.md) §2-1 의 7종 |
| 부하 | 3,500 · 5,000 · 7,500 전부 ([[YR-210]]) |
| 오라클 | **진단 열** — 판정표에 넣지 않고 헤드룸 보고용으로만 |

**핵심 주장은 "배차 방식과 무관하게 재배치가 이득"** 이다. [[YR-180]] 이 이미
규칙 위(−158.13)와 얼린 망 위(−166.69)에서 **거의 같게** 쟀다 — 계획법 위에서도
같은 크기가 나오면 **일반성이 세 배차로 확증**된다.

## 선행·비용

- **선행**: [[YR-212]](무대) → [[YR-210]](부하) — 무대가 바뀌면 다시 재야 한다.
- **비용**: `JointRolloutGreedy` 는 매 결정마다 rollout 한다. 21블록 × 24시간에서
  얼마나 걸리는지 **먼저 1회 측정**하고, 감당 안 되면 `BeamLookahead` 폭을 줄인다.
- **금지**: 결과를 보고 배차 팔을 빼지 않는다. 세 배차 전부 보고한다.
