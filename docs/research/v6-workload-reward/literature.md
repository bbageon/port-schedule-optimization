# 크레인 수용 작업량·보상 설계의 문헌 근거

2026-09-26 · [YR-331 작업 명세](../../../.claude/docs/dashboard-task-specs/YR-331-crane-workload-reward-design.md)
· [수정 설계](../../구현계획/07_v6_적정부하_비용보상.md)

## 1. 조사 결론

**보편적인 최적 가동률이나 크레인당 최적 대수는 확인되지 않았다.** 처리 능력, 도착의 몰림,
작업시간 차이, 본선 업무, 크레인 간 간섭, 허용 대기에 따라 수용량을 정하는 근거는 있다.
가동률을 높이는 것과 트럭·본선의 일을 효율적으로 끝내는 것은 다르다.
아래 수치는 논문 사례·입력값이며 부산항 v6의 실측 최적값이 아니다.

## 2. 원문에서 확인한 연구

### A. Huynh·Walton — 수용량은 서비스 목표와 함께 정한다

- 원저자 기술보고서(2005), *Methodologies for Reducing Truck Turn Time at Marine Container Terminals*,
  SWUTC/05/167830-1. [연구기관 PDF](https://static.tti.tamu.edu/swutc.tamu.edu/publications/technicalreports/167830-1.pdf)
- §5, 인쇄 95·97–100쪽(PDF 111·113–116쪽)을 읽었다. Houston Barbours Cut의 특정 하루 자료를 사용한다.
- 블록별 시간당 상한을 10대보다 낮춘 사례에서는 오히려 체류시간이 증가했다.
  물량을 뒤로 밀어 크레인이 떠났다가 돌아오는 추가 이동이 생겼기 때문이다.
- 전체 평균 체류 45분 이하라는 제약 아래, 특정 두 블록의 예약 상한은
  미방문 비율 5%·15%·25%에서 각각 시간당 7·10·15대였다.
- **크레인 한 대당 보편 권장량이 아니다.** 블록 예약 대수이며 미방문을 포함한다.
  이 보고서의 가동률은 이동 시간만 세고 상하차 시간을 제외하므로 우리 가동률과 직접 비교하지 않는다.
- 관련 후속 논문: Huynh & Walton (2008), *Robust Scheduling of Truck Arrivals at Marine Container Terminals*,
  JTE 134(8), 347–353. [DOI](https://doi.org/10.1061/(ASCE)0733-947X(2008)134:8(347)).
  후속 논문은 서지·초록을 확인했고 위 상세 수치는 읽은 **2005 원보고서**에서 인용한다.

**설계 해석:** 큐를 무조건 비우거나 예약을 무조건 뒤로 보내는 대신, 서비스 수준을 지키는 수용량을 찾는다.

### B. Ma 외(2022) — 처리 능력과 잔여 업무를 함께 고려한다

- Ma, Zhao, Fan & Gong, *Collaborative Optimization of Yard Crane Deployment and Inbound Truck Arrivals
  with Vessel-Dependent Time Windows*, JMSE 10, 1650.
  [논문](https://doi.org/10.3390/jmse10111650) · [발행사 PDF](https://mdpi-res.com/d_attachment/jmse/jmse-10-01650/article_deploy/jmse-10-01650.pdf)
- §2.5 식(14)–(22), §4, 표1(PDF 11–12·19–20쪽)을 확인했다.
  트럭 대기와 다음 교대로 넘어가는 미처리 업무를 줄이며, 야드의 크레인 수와 작업량을 함께 정한다.
- 사례는 19블록·35대의 타이어식 야드크레인, 선박 40척·1주일이다.
- 표1의 한 크레인 서비스율은 **2분당 0.633컨테이너**다.
  환산하면 **시간당 18.99건**, 평균 서비스 **3.1596분/건**이다. 서비스시간 변동계수는 **0.42687**이다.
  변동계수는 작업시간의 표준편차를 평균으로 나눈 값으로, 작업시간이 얼마나 들쭉날쭉한지 나타낸다.
- 18.99는 모델에 넣은 **서비스 능력**이며 적정 도착량이나 실증된 최적 수용량이 아니다.
  본 연구처럼 같은 레일의 크레인 간 간섭까지 똑같이 재현한 근거로 쓰지 않는다.

**설계 해석:** 필요한 작업 분량을 처리 가능한 분량으로 나누고, 본선 업무도 같은 능력을 사용하는 일로 센다.

### C. Riaventin 외(2024) — 높은 가동률이 높은 효율을 뜻하지 않는다

- Riaventin, Cakravastia, Cahyono & Suprayogi, *Sustainable Synchronization of Truck Arrival and Yard Crane
  Scheduling in Container Terminals*, Sustainability 16, 9743.
  [논문](https://doi.org/10.3390/su16229743) · [발행사 PDF](https://mdpi-res.com/d_attachment/sustainability/sustainability-16-09743/article_deploy/sustainability-16-09743.pdf)
- 표5–10(PDF 17–20쪽): 한 블록·크레인 한 대, 도착률 6–9대/시간, 조건별 30회 반복.
  아래는 중앙 예약 방식에서 도착순과 가까운 트럭 우선을 비교한 표7·9의 평균이다.

| 도착량 | 도착순: 가동률 / 체류시간 | 가까운 트럭 우선: 가동률 / 체류시간 |
|---|---|---|
| 8대/시간 | 97% / 1,197초(19.95분) | 85% / 307초(5.12분) |
| 9대/시간 | 99% / 2,380초(39.67분) | 90% / 342초(5.70분) |

같은 도착량에서도 덜 바쁜 크레인이 더 빨리 처리했다. **85%가 최적이라는 실험은 아니다.**
표의 제3정책 일부 행에는 최대가 평균보다 작은 오류도 있어 그 열은 근거로 사용하지 않았다.
위 두 정책 수치는 PDF 그림과 대조했지만, 연구의 실제 터미널 타당성까지 재검증한 것은 아니다.

### D. Kingman(1961) — 혼잡 상한의 분석적 출발점

- *The single server queue in heavy traffic*, 57(4), 902–904.
  [원출판사 서지](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/single-server-queue-in-heavy-traffic/81C55BC00A68FE6D5385638AA0B0AF37)
- 서지 확인과 수식 근거를 구분한다. 원논문의 유료 본문 전체를 읽었다고 주장하지 않는다.
  아래 계산은 B의 대기행렬식에서 **독립 서비스시간·정상상태 포아송 도착·서버 1대**로 제한한
  M/G/1 식을 사용한다. 일반 도착 변동까지 확장하면 Kingman 근사는 출발점이지 항만 보장이 아니다.

```text
평균 큐 대기 Wq = [ρ / (1−ρ)] × [(1+Cs²)/2] × 평균 서비스시간 s
부하 상한 ρ_safe = Wq_limit / [Wq_limit + ((1+Cs²)/2) × s]
시간당 수용률 = ρ_safe × (60 / s)       (s, Wq_limit은 분)
```

여기서 ρ는 도착 작업량/처리 능력이다. 관측창의 실제 가동률이나 적재장의 장치율과 다르다.
정상상태·단일 서버에서는 ρ가 1에 가까워질수록 대기가 크게 늘어난다.
본선 우선순위·고장·두 크레인 간섭이 있으면 이 식만으로 상한을 확정할 수 없다.

**B의 입력값으로 직접 계산한 예시**(논문의 권고가 아니라 이번 설계 검산):

| 가정한 평균 큐 대기 한도 | 계산된 부하 상한 | 계산된 수용률 |
|---|---|---|
| 2분 | 51.71% | 9.82건/시간 |
| 5분 | 72.81% | 13.83건/시간 |
| 10분 | 84.26% | 16.00건/시간 |

같은 크레인도 허용 대기를 바꾸면 답이 달라진다. **5분은 설명용 가정**이다.
저장소의 과거 평균 턴타임 15분 목표는 게이트 주행·서비스를 포함하므로 큐 대기 5분과 동일하지 않다.
위 수용률은 평균값이며 매 시각 큐에 그만큼의 트럭을 유지하라는 뜻도 아니다.

### E. Ng·Harada·Russell(1999) — 적정 상태 보너스를 반복 지급하지 않는다

- *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML.
  [저자 PDF](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf), §2–3.
- 상태의 바람직함을 나타내는 함수 P에 대해 `γP(next)−P(now)`를 더하는 방식은
  원래 목표의 최적 정책을 보존하는 조건을 제공한다. 실제 종료와 중간 수집창 경계를 구분해야 한다.
- 보존 대상은 **같은 할인율의 기대 누적 보상**이다. 할인하지 않은 총비용 순위까지 같아진다는 뜻이 아니다.
  유한 표본·부분 관측·신경망 학습에서 성능 향상이나 수렴을 보장하지도 않는다.

**설계 해석:** 구매 자체에 돈을 주거나 바쁜 상태에 매분 상금을 주지 않고, 적정 배분으로 바뀐 정도를 신호로 준다.

## 3. 현 v6와 대조

- `src/yard_rl/v6/ppo/runtime.py`의 `read_cost/boundary`는 터미널 총비용 증가분의 음수를
  전 블록에 공유한다. 독립적인 크레인별 자기 비용 최소화와 다르다.
- 같은 구조를 가진 기존 v5 저장 실행 `yr306-wait-30d-4549ae9/report.json`에는
  거래 확정 **111,714건 = 공간 17,023 + 시간 94,691**이 있다. 구매 결정 호출 146,032건과 구분한다.
  이는 BUY 불가능설의 반례이며, 학습된 구매가 이익이라는 증거도 **v6 GPU 학습 결과**도 아니다.
  [원기록 발췌·식별값](historical_run_excerpt.json)을 보존했다. 전체 실행 로그를 재검증한 것은 아니다.
- `reward/phi.py`는 아직 게이트 전 지연을 비용에서 제외한다. 적정부하 보조 신호가 이 누락을 고치지는 않는다.
  [YR-328](../../../.claude/docs/dashboard-task-specs/YR-328-deferral-erases-waiting.md)의 장부 문제를 따로 판정해야 한다.

## 4. 출처 확인 범위

원보고서·발행사 PDF 세 건을 직접 읽고 숫자와 단위를 확인했다. PDF는 조사 임시 경로에 두고
재배포하지 않는다. 내려받은 원문의 URL·SHA256(파일 내용 식별값)은 [source_manifest.json](source_manifest.json)에 남긴다.
수식의 재현 값은 [검산 결과](../../../outputs/reports/yr331_workload_reward/design_checks.json)에 저장한다.
새 항만 시뮬레이션이나 강화학습 성능시험은 이번 문헌 조사에서 실행하지 않았다.
