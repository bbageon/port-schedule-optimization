# YR-089 — 트럭 시간계약 정정과 목적함수 재판정

- **Epic**: Sim / **Priority**: 🔴 / **등록일**: 2026-07-23
- **배경**: 현재 `truck_wait`는 트럭이 블록에 도착한 뒤 크레인 작업이 시작될 때까지의 시간만 센다. 터미널 진입부터 진출까지의 턴타임도, 블록 도착부터 작업 완료까지의 처리시간도 아니다.
- **목표**: 예약정보·실제 사건·예측값의 뜻을 분리하고, 블록 정책은 자신이 바꿀 수 있는 처리시간으로 학습하며, 최종 채택은 실제 gate-in→gate-out 턴타임으로 판단하는 이중 계약을 구현한다.
- **선결성**: YR-087의 탐색은 계속할 수 있지만 공식 정책 동결, YR-075-c 목적지 확장, YR-041 locked 시험과 최종 bundle은 이 계약과 재판정 뒤에만 진행한다.

## 한 줄 결정

> 터미널 성과는 `actual_gate_out - actual_gate_in`, 현재 블록 정책의 학습 성과는 `job_done - actual_block_arrival`로 측정한다. 예약시각은 미래 관측일 뿐 비용 시작시각이 아니다.

## 시간축과 이름

```text
게이트 예약 T_appt
       ↓ 예약 준수 오차
실제 게이트 진입 A
       ↓ 게이트 처리·터미널 내부 이동
실제 블록 도착 B
       ↓ 크레인 대기
작업 시작 S
       ↓ 이동·재조작·상하차
작업 완료 C
       ↓ 출구 이동·출문 처리
실제 게이트 진출 O
```

| 필드 | 알게 되는 시점 | 역할 |
|---|---|---|
| `appointment_window_start/end` | 예약 접수 | 정책이 미리 보는 VBS 예약창 |
| `appointment_gate_time` | 예약 접수 | 예약창을 한 시각으로 줄여야 할 때 쓰는 proxy |
| `actual_gate_in` | 실제 입문 | 터미널 비용의 실제 시작 |
| `estimated_block_arrival` | 예약·교통 예측 갱신 | 선제 재조작과 계획용 ETA |
| `actual_block_arrival` | 블록 도착 | 블록 정책 책임구간 시작 |
| `service_start` | YC 배정·작업 시작 | 순수 크레인 대기 종료 |
| `job_done` 또는 `service_end` | 상·하차 완료 | 블록 정책 책임구간 종료 |
| `actual_gate_out` | 실제 출문 | 터미널 턴타임 종료 |

`JobFlow.GATE_IN/GATE_OUT`은 현재 각각 컨테이너 반입·반출 작업 종류다. 실제 차량 입문·출문 사건과 혼동하지 않도록 새 이벤트는 `TRUCK_GATE_IN/TRUCK_GATE_OUT`처럼 별도 이름을 쓴다.

## ETA 의미 정정

현재 `provided_eta`는 실제 블록 도착시각에 ±오차를 더한 값이다. 이는 현실에서 먼저 받는 게이트 예약정보보다 훨씬 직접적이므로 다음 계약으로 바꾼다.

```text
predicted_adherence_delta
= 예상(actual_gate_in - appointment_gate_time)

estimated_block_arrival
= appointment_gate_time
 + predicted_adherence_delta
 + predicted_gate_and_internal_travel
```

- 편차가 양수면 예약보다 늦게, 음수면 일찍 진입한다. 부호가 있는 편차를 뺄셈으로 모호하게 쓰지 않는다.
- 실제 VBS가 예약창을 제공하면 `appointment_window_start/end`를 원본으로 보존하고, 한 점의 예약시각은 명시적인 파생 proxy로만 쓴다.
- 노쇼·취소·조기·지각은 예약정보를 덮어쓰지 않고 별도 실제 상태로 기록한다.
- 실제 `actual_gate_in`과 `actual_block_arrival`은 발생 전 정책 입력에 넣지 않는다.
- `provided_eta`는 즉시 삭제하지 않고 호환기간 동안 `estimated_block_arrival`의 deprecated alias로만 두며, manifest에 의미 버전을 기록한다.
- 예약 준수오차·게이트 처리·내부 이동은 서로 다른 난수 흐름과 provenance(출처·가정)를 가진다.
- 진입 전에는 예약 기반 ETA, 실제 gate-in 뒤에는 진입시각 기반 갱신 ETA, 블록 도착 뒤에는 실제 도착을 공개한다. 기존 ETA wake와 PRE_REHANDLE/REPOSITION 시점도 이 단계별 계약에 맞춰 회귀검사한다.

## 세 지표의 역할

| 지표 | 식 | 사용처 |
|---|---|---|
| YC 순수 대기 | `S - B` | 병목 진단·설명용 |
| 블록 처리시간 | `C - B` | 단일 블록 RL·공동계획기의 1차 학습비용 |
| 터미널 턴타임 | `O - A` | 최종 KPI·정책 채택·현장 PoC 판정 |

블록 처리시간은 기존 대기에 서비스시간을 더하므로 작업순서뿐 아니라 이동, 재조작, 적재위치와 실제 처리완료까지 반영한다. 게이트·내부도로를 아직 완전히 모델링하지 못한 단계에서 블록 정책에 전체 턴타임을 전액 귀속하지 않는다.

전체 턴타임은 다음 등식으로 분해해 함께 보고한다.

```text
O - A
= (B - A)   # 입문→블록
 + (S - B)  # YC 대기
 + (C - S)  # 이동·재조작·상하차
 + (O - C)  # 완료→출문
```

현재 블록 PoC에서는 `B-A`와 `O-C`를 정책 밖 외생구간으로 고정한다. 정책은 `S-B`와 `C-S`를 함께 바꿀 수 있으므로 둘을 합친 `C-B`를 학습해야 “빨리 시작했지만 오래 처리한” 정책을 잘못 채택하지 않는다.

## 비용 적분 계약

터미널 턴타임 비용은 매 순간 터미널 안에 있는 트럭 수를 적분한다.

```text
terminal_truck_area
= ∫ N_inside(t) dt
 = Σ(actual_gate_out - actual_gate_in)
```

- 에피소드 종료 때 미완료 차량도 `end_time - actual_gate_in`만큼 검열 비용을 받고 backlog·완주 guard에 걸린다.
- 실제 gate-out이 구현되기 전의 `service_end + 고정 출문시간`은 `turntime_proxy`로만 이름 붙이며 실제 턴타임이라 주장하지 않는다.
- 전체 턴타임을 경제비용에 넣을 때 그 안에 포함된 YC 대기·블록 처리시간을 기본비용으로 다시 더하지 않는다. 별도 SLA 벌점이면 중복이 아니라는 근거와 단위를 명시한다.
- 평균뿐 아니라 P95 턴타임도 보고한다. P95는 트럭 100대 중 오래 걸린 상위 5대가 겪는 수준이다.

## 학습표적

같은 외생 사건을 쓴 SF-SPT 기준선과 정책을 짝지어 비교한다.

```text
reported_gain
= baseline_block_turntime - policy_block_turntime  # 클수록 좋음

cost_network_target
= policy_block_turntime - baseline_block_turntime  # 작을수록 좋음
```

이렇게 하면 게이트 처리·내부도로처럼 두 정책이 바꾸지 못하는 공통시간은 상쇄되고, 현재 비용 최소화 Q망의 부호도 유지된다. 본선 비용과 결합할 때도 시간 단위 비용률로 환산하고 YR-080의 중복계상 금지 계약을 유지한다.

## 현재 구현과 결과의 재해석

- `engine.py`는 `BLOCK_ARRIVAL`에서 대기 적분을 시작하고 SERVE 배정 순간 끝낸다.
- `test_truck_wait_excludes_service_time`은 서비스시간 제외를 계약으로 고정한다.
- `scenario_gen.py`는 `actual_block_arrival ± eta_error`를 `provided_eta`로 만든다.
- `yr009_turntime.py`는 실제 출문 사건 없이 `service_end - actual_gate_in + 고정 출문시간`을 사용한다.

따라서 기존 YR-071·073·074·080·087의 트럭 성과는 **구 지표 `S-B`를 개선한 증거**로 보존하되 다음을 자동 주장하지 않는다.

- 블록 도착부터 작업완료까지 줄였다는 주장
- 실제 gate-in→gate-out 턴타임을 줄였다는 주장
- 새 비용계약에서도 같은 행동·체크포인트가 최적이라는 주장
- “완벽 ETA 가치 0”이 게이트 예약정보에도 그대로 성립한다는 주장. 기존 결론은 `B`를 직접 예측한 구 ETA에만 한정한다.

엔진 안전성·resolver·본선 인과검사 같은 성과는 그대로 유효하다. 다만 목적이 행동을 바꾸면 교사 표본 재수집, 정책 재학습 또는 rollout 재선택을 한다.

## 구현 순서

1. 스키마에 예약·예측·실제 사건 필드와 시간지식 경계를 추가한다.
2. 게이트 진입, 내부이동, 블록도착, 작업완료, 게이트진출 사건을 연결한다.
3. `yc_queue_wait`, `block_turntime`, `terminal_turntime`, censored exposure를 별도 장부와 KPI로 낸다.
4. SF-SPT와 JointRollout/3600s rollout의 새 목적 헤드룸을 동일 seed로 다시 잰다.
5. 행동이 달라질 때만 교사 재수집·학습 재실행을 한다.
6. YR-009 공개 턴타임 대조를 실제 gate-out 기준으로 다시 실행하고 구 프록시와 차이를 보고한다.
7. YR-041은 동결된 새 계약과 정책 bundle을 보지 않은 seed에서 평가한다.

## 필수 계약 테스트

- 모든 차량에서 `actual_gate_in ≤ block_arrival ≤ service_start ≤ job_done ≤ actual_gate_out`이 보존된다. 예약시각은 조기·지각 때문에 이 사건 순서식 밖에 둔다.
- 실제 사건을 고정한 채 예약시각만 바꾸면 실현 비용은 변하지 않고 정책 관측·선제계획만 바뀐다.
- 정책이 실제 도착·출문 진실값을 발생 전에 읽지 못한다.
- `terminal_truck_area == 완료차량 턴타임 합 + 미완료차량 검열시간 합`이 성립한다.
- `block_turntime == yc_queue_wait + service_start부터 job_done까지의 시간`이 차량별로 성립한다.
- `O-A == (B-A)+(S-B)+(C-S)+(O-C)`가 차량별로 성립한다.
- 차량을 미완료로 남겨 비용을 낮출 수 없고, 완주율 1.0·backlog 0 guard를 유지한다.
- gate-in/out 사건 추가가 호환 arm에서 불필요한 YC 결정시점을 열지 않는다.
- 예약·게이트/내부이동·정책 RNG를 분리해 같은 seed의 결정론을 보존한다.
- 기존 기본 시나리오는 호환 모드에서 golden 결과가 변하지 않으며, 새 의미 버전은 opt-in 후 별도 golden으로 동결한다.

## 수용 기준

- 시간 필드마다 단위, 출처, 알게 되는 시점, 실제값/예측값 구분이 schema에 박제된다.
- 비용 장부와 보고 KPI의 적분 등식이 테스트로 증명된다.
- SF-SPT·현재 rollout 후보를 신규 seed의 반입/반출별 P50·평균·P95 `B-C`와 `A-O`, 서비스시간, 완주·본선 지표로 재평가한다.
- 기존 체크포인트의 유지·재평가 한정·재학습·기각 중 하나를 명시적으로 판정한다.
- schema·cost·manifest 버전을 올리고 기존 golden·checkpoint 호환성을 명시한다.
- 최종 보고서는 `YC 대기 개선`, `블록 처리시간 개선`, `터미널 턴타임 개선`을 섞어 쓰지 않는다.

## 연구·운영 근거

- [Port of Los Angeles Truck Turn-Time Incentive](https://kentico.portoflosangeles.org/getmedia/315f26f4-03cd-4e4a-bee1-db146ed084c4/09_Cargo-Marketing_Temporary-Order-TTT-and-DT-Incentives_Tariff-No-4_Transmittal-2): gate-in과 gate-out을 터미널 턴타임 경계로 사용한다.
- [Kim, Lee & Hwang (2003)](https://doi.org/10.1016/S0925-5273(02)00466-8): 전체 트럭 체류시간 안에서 야드크레인 배차가 바꾸는 대기·처리 구간을 최적화한다.
- [부산항 체인포털 터미널 매뉴얼](https://www.chainportal.co.kr/manual/download?filename=Chainportal_Terminal_Manual.pdf): 터미널·구역별 VBS 준수시간 정의가 달라 단일 시간명을 그대로 쓰면 안 된다는 운영 근거다.

## 범위 밖

- 실제 운영사의 예약 준수분포·게이트 처리분포를 공개자료만으로 확정
- 게이트 예약 슬롯 자체의 최적화
- 다중 블록 내부도로 라우팅과 터미널 전체 교통제어
- 새 계약 구현 전에 기존 결과를 실제 터미널 턴타임 개선으로 소급 해석

## 연결 작업

- **같이 정합**: YR-050의 ETA wake·정보수준, YR-080의 통합 목적 장부
- **이 작업 뒤**: YR-009 공개 턴타임 재검증, YR-075-c, YR-041, 최종 YR-014
- **실증 때 재개**: YR-019의 예약편향·노쇼·낡은 ETA 강건성
- **자료 한계**: 예약 준수·게이트 처리·출문 분포는 YR-082 Level 3 운영로그 전까지 assumed로 표시
