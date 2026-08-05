# YR-151 — Block PPO SELL TransferHead

- **Epic**: RL / **Priority**: 🔴 / **등록일**: 2026-08-05
- **상태**: backlog — YR-149의 5부하 데이터·정보시점 계약 동결 뒤 바로 착수
- **사용자 결정**: 중앙 계산기가 먼저 판매작업을 고르는 구조를 기준선으로 남기되,
  원소유 블록 PPO가 `KEEP/SELL`을 직접 학습하는 정책을 구현한다.

## 연구 질문

“현재 블록의 미래 부하와 작업 ETA를 본 Block PPO가 어떤 반입 작업을 내보낼지 학습하면,
계산식 기반 발의와 이송 없는 정책보다 터미널 v2 총비용을 더 낮출 수 있는가?”

`SELL`은 판매 요청이며 실제 이전 명령이 아니다. PPO가 목적지나 소유권을 직접 바꾸지 않는다.

```text
외부 TOS 최초배정 스냅샷 ──> 각 Block PPO
                                  ├─ ExecutionHead: 블록 안 YC 실행순서
                                  └─ TransferHead: KEEP / OFFER_SELL(job)
                                                       │
                          receiver 부담 계산 <── TransferResolver
                                                       │
                                   KEEP 또는 시뮬레이터 내부 원자 재배정
```

## 권한·단일축 계약

1. YR-148 채택 구성인 `C0+대기허가증 안전장치`의 ExecutionHead·체크포인트를 고정한다.
2. 같은 Block PPO에 **별도 transfer adapter·TransferHead·transfer critic만** 추가·학습한다.
3. `SELL`을 `SERVE/PRE_REHANDLE/DEFER` 크레인 행동 목록에 넣지 않는다.
4. receiver 수용부담식·allowed-block 검사·Resolver의 최종 선택권은 고정한다. 단 현 엔진은
   gate-in 전 이송을 거절하므로 0단계에서 **PRE_GATE 원자 transaction**을 별도 구현한다.
5. `BUY` 학습행동은 만들지 않는다. 첫 실험에서 실행 PPO 동시 미세조정, receiver 학습,
   양하 확대, 다건 matching을 금지한다.
6. 실제 TOS API·승인·현장 명령은 범위 밖이다. TOS 최초배정이 주어진 합성 입력만 쓴다.

## SELL 대상과 30분 ETA 창

- 대상은 아직 실제 gate-in 사건이 발생하지 않은 `PRE_ADVICE/PLANNED` 반입 작업이다.
- 공개된 예상 블록도착시각이 `0 < ETA_block-now <= 1,800초`일 때만 후보가 된다.
- 실제 미래 gate-in/block-in/O, 미통지 고장·계획변경은 actor 입력과 resolver 견적에서 금지한다.
- 검토 사건은 ①ETA가 30분 창에 들어오는 시각 ②공개 ETA 갱신 ③작업 완료·장비상태·소유권·
  용량 변화다. 단순 초단위 반복계산은 하지 않는다.
- offer는 그 review epoch 안에서만 유효하다. Resolver가 수락하면 gate-in 전에
  `execution_block_id`·owner/version·미래 arrival route를 즉시 원자 변경하고,
  `tos_assigned_block_id`는 이력으로 보존한다. 이후 gate-in은 확정 블록으로 들어온다.
- 결정 중 ETA version이 바뀌거나 gate-in이 먼저 발생하면 낡은 offer는 KEEP한다. commit 뒤
  ETA 변경은 owner를 자동 rollback하지 않으며 작업당 1회 이전 상한을 유지한다.
- 한 source는 epoch당 `KEEP` 또는 작업 1건 `OFFER_SELL(job, version, expires_at)`만 낸다.

PRE_GATE 이송비용은 기존 블록 A→B 물리운반 180초가 아니라 `gate→새 블록`과
`gate→기존 블록`의 예측 주행시간 차이다. 실제 실현 주행시간은 평가 장부에만 쓴다.

현재 YR-133/149의 actual gate-in 3~7분 review는 계산 기준선·이력으로 보존한다. YR-151의
30분 PRE_ADVICE 창은 0단계에서 후보 노출·정보누출·결정시점을 먼저 검증한 뒤 학습한다.

## PPO 구성

- **Actor 입력**: source 블록의 현재 queue·평균대기·YC 잔여부하·장치율·본선 여유,
  향후 30분 예정 유입, 후보 작업의 공개 ETA·flow·규격·현재 owner/version.
- **Actor 출력**: 동적 후보 `KEEP + OFFER(job_1..K)`의 masked categorical 확률.
- **Critic 입력**: 결정시점의 양 블록 공개상태·receiver 현재 수용가능 요약을 학습 중에만
  본다. resolver 결과는 다음 transition·보상일 뿐 현재 critic 입력에 미리 넣지 않는다.
  실행 actor는 source의 공개정보만 본다(중앙학습·분산실행).
- **보상**: route 비용과 미완 장부를 포함한 YR-136 v2 실제 증분비용의 음수다. 혼잡점수를
  별도 보상으로 다시 더하지 않는다.
- **학습 분리**: ExecutionHead hash 불변을 매 epoch 검사하고 TransferHead gradient가 실행
  head·encoder 고정영역으로 새지 않게 테스트한다.

## 단계

### ★29·30차 반영 — 0단계는 "측정 먼저", 학습은 신호 확인 후 (2026-08-05)

0단계는 배선 검증만이 아니라 **학습 이전의 신호 존재 확인**이다. 아래 순서를 지킨다.

1. 30분 전 시점에 **실제 SELL 후보가 얼마나 생기는지** 측정(셀별·시점별 분포).
2. **그 순간 수신 블록에 실제 여유가 있는지** 측정 — YR-149 의 "B 평균 비가동 57.9%"는
   6시간 평균이라 **결정 순간의 여유를 뜻하지 않는다**. 순간 여유를 직접 잰다.
3. **PRE_GATE 원자 재배정 구현**(현 엔진은 gate-in 전 이송을 거절).
4. **KEEP 대비 이송의 실제 개선 여지 확인**(반사실).
5. **공개정보로 이득 부호를 구분할 수 있는지 확인**.
6. 신호가 없으면 `NO_LEARNABLE_SIGNAL` 로 표시하고 **PPO 학습을 중단**한다.

**구조 주의**: YR-149 5셀은 4h 유입→2h 비우기의 **유한 물량 처리·회복 시험**이다. 0단계
측정에는 쓸 수 있으나 **지속 운영 성능 확증은 YR-150 의 지속 유입 정상상태 구조**를
따른다(도착률 고정·재공량 자유·본선 전 구간 분산·warm-up/측정창/비우기창 분리).
[결정 문서](../strategy-history/2026-08-05-지속유입-정상상태-시험-전환-사용자결정.md)

### 0. 데이터·결정창 자격

- YR-149의 A/B 5셀 master trace를 사용하되 train/dev/test seedbank를 분리한다.
- 셀별 후보 수, 다후보 epoch, ETA 오차, 양·음 반사실 이득의 수와 분포를 기록한다.
- `prepare_pre_gate_transfer→validate→commit/rollback`을 구현해 owner 정확히 1,
  `execution_block_id`·미래 arrival event·전역 A→O 장부·route 비용의 원자성을 검사한다.
- 현재 5셀 seed는 ETA 갱신 사건이 없으므로 성능 주장은 **정적 ETA 조건**으로 한정한다.
  별도 계약 probe에서 ETA update→version 변경→stale offer KEEP을 반드시 발화시킨다.
- 후보 노출이 부족해 KEEP과 SELL을 구분할 수 없으면 `NO_LEARNABLE_SIGNAL`로 표시한다.
  구현은 남기되 성능 학습을 성공으로 주장하지 않는다.

### 1. shadow 배선·계약 검증

- TransferHead 결정을 원장에만 기록하고 실제 commit은 계산 기준선이 담당한다.
- 미래정보 0, action mask, version/TTL, 실행 PPO hash, 확률·log-prob·GAE 등식과 결정론을
  테스트한다. shadow가 본 실행·난수열을 바꾸면 실패다.
- 실행되지 않은 shadow 행동으로 PPO를 갱신하면 on-policy 계약 위반이므로 이 단계에서는
  정책경사 학습을 금지한다. 후보·상태·행동·후속비용 원장만 검증한다.

### 2. on-policy 학습과 live 단일축 비교

- 별도 train seedbank에서는 TransferHead의 실제 OFFER/KEEP가 같은 resolver를 거쳐 환경상태를
  바꾸며, 그 자기 궤적으로 PPO rollout을 수집한다. Q/K 궤적으로 S를 학습하지 않는다.
- 학습이 끝난 체크포인트를 고정한 뒤 미열람 평가 seedbank에서 아래 세 군을 비교한다.

실행 PPO·receiver·resolver·seed·물리는 모두 같고 source 발의 방식만 다르다.

| 군 | source 발의 | 뜻 |
|---|---|---|
| K | 항상 KEEP | 이송 없는 절대 기준 |
| Q30 | S와 같은 30분 epoch·후보·정보의 계산 견적 | 결정론 기준선 |
| S | PPO TransferHead | 학습 SELL 정책 |

YR-149의 actual gate-in 계산견적은 연결 참고군일 뿐 주 대조군이 아니다. Q30과 S는 결정시점,
후보, 공개정보, PRE_GATE transaction, receiver/Resolver가 같고 source 선택기만 다르다.

### 3. 혼잡 5셀 확증

- YR-149에서 자격을 확인한 `A/B=50/50·75/50·100/50·125/50·150/50`을 그대로 쓴다.
- 이는 source A의 4시간 유입량 변화이며 터미널 합계는 `100·125·150·175·200`이다.
- 셀을 합쳐 하나의 평균으로 판정하지 않는다. 학습 초기화×master scenario를 함께 반영한
  계층형 신뢰구간과 셀 동시검정을 쓴다.
- YR-150 고정 WIP는 선결이 아니다. YR-151 통과 뒤 learned-SELL 군을 별도 추가한다.

## 판정 게이트

1. **G0 계약**: 미래정보 누출·중복 owner·orphan event·stale commit·rollback 실패·route 누락
   전부 0, 실행정책 hash 불변, 미완·backlog 누락 0과 비용 적분·검열 장부 일치. G0은
   용량초과 셀의 완주 100%를 요구하지 않는다.
2. **G1 노출**: S가 Q30과 다른 KEEP/OFFER를 충분히 내고 실제 commit도 발생해야 한다.
   최소 노출수는 YR-149/0단계 자료로 결과 열람 전에 동결한다.
3. **G2 비용**: 독립 seedbank에서 `S-Q30<0`와 `S-K<0`의 v2 총비용 신뢰구간 상한이 모두
   0보다 작아야 PPO SELL의 추가가치로 채택한다.
4. **G3 보호**: 비용판정 가능 셀은 평균 A→O·본선 지연이 사전 허용손실 안이고 완주 100%·
   최종 backlog 0·정책 예외 0이어야 한다. 용량초과 셀은 비용판정 대신 처리량·잔여·
   drain-to-empty만 보고한다.
5. **G4 보고**: offer/accept/commit/rollback, 방향, route 비용, 안전장치 개입률, P95는
   전량 보고하되 P95는 채택 veto로 쓰지 않는다.

전 부하 채택은 **모든 비용판정 가능 셀**에서 G2·G3가 통과할 때만 선언한다. 일부 셀만
통과하면 해당 입력부하 한정 결과이며 글로벌 SELL 채택은 금지한다.

해석 분기는 고정한다. `S<Q30`지만 `S>=K`면 “나쁜 계산 발의를 걸렀으나 이송의 절대 이득은
없음”, `S<K`지만 `S>=Q30`면 “이송은 유효하나 학습 head 추가가치 없음”이다. 두 기준을 모두
이기고 보호게이트를 통과할 때만 “Block PPO SELL 추가가치”라고 쓴다.

## 선결·참조

- [YR-133 원자 이송·견적 기능](YR-133-blockq-sell-quote.md)
- [YR-148 채택 실행 구성](YR-148-guard-on-rejudgment.md)
- [YR-149 5부하 데이터·견적 기준](YR-149-quote-refine-confirm.md)
- [YR-150 고정 WIP 보조실험](YR-150-constant-wip-steady-state.md)
