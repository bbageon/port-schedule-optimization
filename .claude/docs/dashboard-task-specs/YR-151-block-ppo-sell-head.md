# YR-151 — Block PPO SELL TransferHead

- **Epic**: RL / **Priority**: 🔴 / **등록일**: 2026-08-05
- **상태**: **backlog** (2026-08-06 — **0A 종료**, 0B-H 는 YR-150 H-21 현실성 PASS 선결)
  - **0A 완료**: 계약 8종 전 항 통과(20/20 셀·깨끗한 커밋 실행) — 판정 `ae6005b` ·
    [report](../../../outputs/reports/yr151_pre_gate_0a/report.md)
  - **0A 를 하네스에 보고 → 신뢰성 게이트 FAIL → PASS**(YR-153 `current_gate.json`).
    닫힌 3사유 = `runtime_git_dirty` · 빈 `runtime_params` · 사전등록이 파일 경로가 아님
  - 다음: YR-150 **H-21 21블록** 지속 유입 환경 구축 → **0B-H**(신호·학습 GO/STOP)
- **3대 게이트 단계 계약**: `0A=reliability`, `0B=performance` — 각 단계 착수 전 YR-153
  `authorize-next`를 따로 통과하며 두 축 동시 보정 금지
- **사용자 결정**: 중앙 계산기가 먼저 판매작업을 고르는 구조를 기준선으로 남기되,
  원소유 블록 PPO가 `KEEP/SELL`을 직접 학습하는 정책을 구현한다.

## 연구 질문

“H-21과 V-21 각각에서 source Block PPO가 미래 부하와 ETA로 어떤 반입 작업을 내보낼지
학습하고, Resolver가 여러 수신 블록 중 목적지를 고르면 계산식 발의와 이송 없음보다
해당 구조의 터미널 비용이 낮아지는가?”

H-21을 먼저 별도 학습·판정한다. V-21은 YR-083의 육·해측 역할·LSTP/WSTP·AGV 계약과
지속 유입 자격이 끝난 뒤 **같은 알고리즘·별도 가중치**로 판정한다. 비교는 항상 구조별
`정책−동일 구조 SF-SPT`이며 H/V 원비용 평균과 H 체크포인트의 zero-shot 승계를 금지한다.

`SELL`은 판매 요청이며 실제 이전 명령이 아니다. PPO가 목적지나 소유권을 직접 바꾸지 않는다.

```text
TOS 최초배정 ─> Block PPO { ExecutionHead: YC 실행순서 / TransferHead: KEEP·OFFER }
                 └ OFFER ─> TransferResolver(수신 부담 비교) ─> KEEP 또는 원자 재배정
```

## 권한·단일축 계약

1. YR-148 채택 구성인 `C0+대기허가증 안전장치`의 ExecutionHead·체크포인트를 고정한다.
2. 같은 Block PPO에 **별도 transfer adapter·TransferHead·transfer critic만** 추가·학습한다.
3. `SELL`을 `SERVE/PRE_REHANDLE/DEFER` 크레인 행동 목록에 넣지 않는다.
4. receiver 수용부담식·allowed-block 검사·Resolver의 최종 선택권은 고정한다. 단 현 엔진은
   gate-in 전 이송을 거절하므로 0단계에서 **PRE_GATE 원자 transaction**을 별도 구현한다.
5. `BUY` 학습행동은 만들지 않는다. 실행 PPO 동시 미세조정·receiver 학습·양하 확대는 금지한다.
   **★판매 축 확정(사용자 결정 2026-08-08)**: 판매 = **반입(공간·본 spec) + 반출(시간·
   [YR-161](YR-161-time-sell-outbound-deferral.md))** 두 축뿐이다. **양하·적하는 판매에서
   제외 확정** — 적하: 본선 지연 직결·기사 없음·크레인 행동과 중복. 양하 공간축만 반입
   실증·엔진 배선 후의 확장 후보로 보존.
   각 source는 1건만 발의하되 N블록 Resolver는 job 중복·receiver 용량을 지키며 복수 source를
   결정론적으로 matching한다. 2블록의 터미널 전체 1건 상한은 본 성능계약으로 승계하지 않는다.
6. 실제 TOS API·승인·현장 명령은 범위 밖이다. TOS 최초배정이 주어진 합성 입력만 쓴다.

## SELL 대상과 30분 ETA 창

- 대상 = 아직 gate-in 사건이 없는 `PRE_ADVICE/PLANNED` 작업, 공개 예상 블록도착이
  `0 < ETA−now ≤ 1,800초`일 때만. 실현 미래값·미통지 고장·계획변경은 actor 입력과
  resolver 견적에서 금지.
- 검토 사건 = ①창 진입 시각 ②공개 ETA 갱신 ③작업 완료·장비·소유권·용량 변화 —
  단순 초단위 반복계산 금지(현 구현은 60초 격자 근사).
- offer 는 그 epoch 전용(이월 금지). 수락 시 gate-in 전에 owner/version·미래 arrival
  route 를 원자 변경하고 최초배정은 이력 보존. 결정 중 version 변경·선행 gate-in 이면
  KEEP, commit 뒤 ETA 변경은 자동 rollback 없음, 작업당 이전 1회 상한.
- 한 source 는 epoch 당 KEEP 또는 작업 1건 OFFER 만 낸다.

PRE_GATE 이송비용은 `gate→새 블록`과 `gate→기존 블록`의 예측 주행시간 차이다. N블록에서는
반드시 `route(src,dst,job)`로 계산한다. 0A의 전 블록 300초·차이 0은 기술검사 한계이며 성능에 쓰지 않는다.

**★해소(2026-08-06, YR-150 0단계 `5be10ad`)**: `yard_layout.YardLayout` 이 목적지별 주행
matrix 를 제공하고 `travel_fn(src,dst,job)`·`route_fn(src,dst)` 배선이 끝났다. 게이트 진입 전
재배정의 주행 차이는 이제 ±110/±220초(N=3 기준)로 **0 이 아니며 부호도 있다**. 0B 는 이
배치를 쓰되 **합성 배치 가정**임을 주장 범위에 함께 밝힌다.

현재 YR-133/149의 actual gate-in 3~7분 review는 계산 기준선·이력으로 보존한다. YR-151의
30분 PRE_ADVICE 창은 0단계에서 후보 노출·정보누출·결정시점을 먼저 검증한 뒤 학습한다.

## ★TransferHead 설계 확정·선구현 (2026-08-09 대화 확정 — 코드 `transfer_head.py`)

- **의미 규정(사용자 확정)**: PPO 판단 = **"이 작업을 포함한 현 계획이 나쁘다"**.
  K+1 행동 = 계획 변형 K+1개(KEEP = 현 계획 유지 최선 / OFFER(j) = j 를 덜어낸 계획이 우월).
- **B-(b) 구조(사용자 확정)**: PPO 는 축(공간/시간)을 모른다 — "무엇을 덜어낼지"만.
  **축 선택은 resolver** 가 대안 좌표({다른 블록 20곳} ∪ {+15분 이연})를 **비용 통화
  한 저울**에 올려 정한다(`UnifiedSellOrchestrator`). 따라서 **가중치 1벌**(축별 분리
  불요 — 축 귀속은 resolver 원장으로 사후 분해). 반입 재예약 포함 문제도 이로써 해소.
- **matching**: 동결 수집(결정 단계 commit 0) → 전역 한계비용 최소 쌍 반복(외부 순서
  규칙 없음·순열 불변) → 가상 상태 +1/−1 후 **볼록 비용 재계산** → 일괄 원자 확정.
- **보상 확정 = (i) 실현 전역 증분**: r = −(구간 터미널 실현 v2 증분 — 수신 부담·주행·
  기사 외부 대기 포함). 전역이라 떠넘긴 비용 자동 포함(= 이기적 판매 + 가격 청구 동치).
  청구는 실행 중 메시지가 아니라 **학습 때 원장 귀속**. (ii) 거래 귀속형은 반사실 추정이
  필요해 후속 축. Σ 구간보상 = −총비용 검열 등식 승계(학습 잣대 ≡ 평가 잣대).
- **KEEP 근거**: 학습된 가치(이후 결과 전부 반영) + 비가역성 비대칭(SELL 1회 잠금·KEEP
  60초 뒤 재고, 창 닫힐 때까지 ~30회). 신호 존재는 0B 실측(`NO_LEARNABLE_SIGNAL`).
- **lead 가정(0B 사전등록 동결 대상 — 미확정)**: 사전 통지 lead=1800s(판매 창), WIP
  해석은 "확약 총량(내부+통지)=L" 기본 — 대안(L 상향)은 config 교체.
- 60초 검토는 **격자 근사**다(계약의 사건 기반 검토를 최대 60초 지연으로 포착).
- **★감사 반영(2026-08-09 2차)**: 실행 head 는 `adopted`/`sf` 외 값 즉시 거절·채택 모드
  예외는 WAIT 로 숨기지 않고 즉시 실격. 초기화 선정 = **기계 규칙**(CONFIRM_TS 첫 항
  221000·성능 미열람) + **판정런은 3개 초기화(221k/222k/223k) 민감도 보고 필수**.
  실행 **구성 전체 해시** 봉인(가중치+ckpt 파일+플래그+가드 — 매 iter 불변 검사).
  행동 일치 테스트 통과(판정 경로 vs 새 경로 결정 궤적 동일 — 2/2). **Q30 동결본은
  shadow·0B 신호 확인 후에** 구현한다(신호가 없으면 비교 기준만 정교해지고 무익 —
  감사 순서 수용, 구 계획 "Q30 먼저" 철회).

## PPO 구성

- **Actor 입력 (구현 정합 2026-08-09)**: 블록 계획 7(내부 대수·통지 pipeline·크레인
  잔여부하·장치율·본선 여유·향후 30분 통지 유입·후보 수) + 후보 6(공개 ETA 잔여·flow·
  규격·통지 진입까지 잔여·이송/이연 이력) — **공개 통지 시각 접근자만 사용**(실현
  gate-in 미열람). 구 명세의 평균대기·version 특징은 미구현(확장 후보로만 보존).
- **Actor 출력**: 동적 후보 `KEEP + OFFER(job_1..K)`의 masked categorical 확률.
- **Critic 입력**: source 요약 + **수신 시장 요약**(수신별 부하·소스로부터의 주행 차이·
  용량 여유의 평균/최소 pooling — 순열불변, 2026-08-09 강화) + 전역(총 내부·진행률).
  학습 중에만 본다 — 실행 actor 는 source 공개정보만(중앙학습·분산실행). resolver 결과는
  다음 transition·보상일 뿐 현재 critic 입력에 미리 넣지 않는다.
- **보상**: route 비용·미완 장부·기사 외부 대기를 포함한 v2 실제 증분비용의 음수 +
  **관측종료 bootstrap**(V(s_end) — 관측 밖으로 미뤄진 비용 계상, 마감 꼼수 방지).
  혼잡점수를 별도 보상으로 더하지 않는다. 공동 transition(전 블록 결정·OFFER·resolver
  결과·구간 Φ·종료 플래그)을 epoch 단위 한 묶음으로 기록한다.
- **학습 분리**: ExecutionHead hash 불변을 매 epoch 검사하고 TransferHead gradient가 실행
  head·encoder 고정영역으로 새지 않게 테스트한다.

## 단계

### ★30차 반영 — 0A/0B 분리, 학습 GO/STOP 은 0B 에서만 (2026-08-05)

유한 물량(YR-149 5셀)과 지속 유입(YR-150)은 SELL 후보·장기 파급이 다르므로 **5셀 신호
부재로 지속 유입 PPO 를 중단하지 않는다.** 0A(2블록 기술계약 — 완료·성능권한 없음) →
YR-150(21블록 자격) → **0B-H/0B-V**(구조별 신호 재검사 — 실제 SELL 상금·공개정보 부호
구분력·결정 **순간** 수신 여유. 여기서만 GO/STOP, 미달 = `NO_LEARNABLE_SIGNAL` 학습 중단).
0A 중단 사유는 계약 위반뿐("후보 적음"은 STOP 사유 아님). 수신 여유는 순간값(29차 —
6시간 평균 금지). 구조별 0B 통과 시에만 Q30 동결 → shadow → on-policy → K/Q30/S,
H 통과가 V 통과를 대신하지 않는다.
[지속 유입 전환 결정](../strategy-history/2026-08-05-지속유입-정상상태-시험-전환-사용자결정.md)

### 0. 데이터·결정창 자격

- YR-149 A/B trace 는 완료된 0A 기술검사 전용. 0B 는 YR-150 터미널 master stream +
  21블록 최초배정 벡터, train/dev/test seedbank 분리. 셀별 후보 수·다후보 epoch·ETA
  오차·양/음 반사실 이득 분포를 기록한다.
- pre_gate 원자 transaction(owner 정확히 1·미래 arrival event·전역 A→O 장부·route
  원자성)은 0A 에서 검사 완료. 5셀 seed 는 ETA 갱신 사건이 없어 성능 주장은 **정적
  ETA 조건** 한정 — 별도 계약 probe 에서 ETA update→version→stale KEEP 발화 필수.
- **★31차 정정**: "후보 노출 부족 시 `NO_LEARNABLE_SIGNAL`" 은 0A 에서 삭제·0B 로 이관.
  0A 중단 사유는 계약 위반(누출·원자성/rollback 실패·소유권 중복·stale commit)뿐.

### 1. shadow 배선·계약 검증

- **★감사 정정 → 구현 → 실측 검증 통과(2026-08-09, `0808f45`·dirty=false)**: shadow =
  **resolver 관통 dry-run**(수집→저울→matching→용량 관통·원자 확정만 생략·would-commit
  원장·짝 생성자 강제). 실측(w5-L100·채택 PPO): **S1 본 실행 불변**(기준 런 대비 343개
  작업 시간 장부 전수 일치)·S2 흐름(trail 2,090·would-commit 886·critic 입력 비영)·
  S3 결정론·S4 실행 해시 불변·예외 0 — [검증 JSON](../../../outputs/reports/yr151_shadow_verify/shadow_verify.json). 이 단계 정책경사 학습 금지(on-policy 계약).

### 2. on-policy 학습과 live 단일축 비교

- 별도 train seedbank에서는 TransferHead의 실제 OFFER/KEEP가 같은 resolver를 거쳐 환경상태를
  바꾸며, 그 자기 궤적으로 PPO rollout을 수집한다. Q/K 궤적으로 S를 학습하지 않는다.
- 학습 종료 체크포인트를 고정한 뒤 미열람 평가 seedbank 에서 **K(항상 KEEP) / Q30(같은
  30분 epoch·후보·정보의 계산 견적) / S(PPO TransferHead)** 를 비교한다. 실행 PPO·
  receiver·resolver·seed·물리는 모두 같고 **source 발의 방식만** 다르다(결정시점·후보·
  공개정보·PRE_GATE transaction 동일). YR-149 actual gate-in 계산견적은 참고군일 뿐
  주 대조군이 아니다.

### 3. 터미널 전체 부하 5셀 확증 (★사용자 정정 — 21블록에서)

- 부하 셀 `50·75·100·125·150` 을 쓰되, **지속 유입 정상상태 구조(YR-150)** 에서 판정한다.
  이 숫자는 **21블록 전체를 합한 4시간당 외부트럭 도착량**이며 블록별 물량이 아니다.
- YR-149 5셀(유한 물량 회복 구조)은 **0A 계약 검사·참고**용이며 지속 운영 성능의 확증
  환경이 아니다.
- 셀을 합쳐 하나의 평균으로 판정하지 않는다. 학습 초기화×master scenario를 함께 반영한
  계층형 신뢰구간과 셀 동시검정을 쓴다.
- 주지표는 YR-150 평가계약(시간당 누적 v2 비용·시간가중 재공량·backlog 기울기·A→O·
  본선 시간당 지연비용)을 따르고, 처리량은 안정성 조건으로만 쓴다.
- 모든 21블록은 source/receiver가 될 수 있어야 하며 터미널 합계와 블록별 도착−완료·WIP 편차를
  함께 보고한다. N=2 결과로 terminal-wide SELL 성능을 주장하지 않는다.

## 판정 게이트

1. **G0 계약**: 미래정보 누출·중복 owner·orphan event·stale commit·rollback 실패·route 누락
   전부 0, 실행정책 hash 불변, 미완·backlog 누락 0과 비용 적분·검열 장부 일치. G0은
   용량초과 셀의 완주 100%를 요구하지 않는다.
2. **G1 노출**: S가 Q30과 다른 KEEP/OFFER를 충분히 내고 실제 commit도 발생해야 한다.
   최소 노출수는 YR-149/0단계 자료로 결과 열람 전에 동결한다.
3. **G2 비용 (★31차 — 지속 유입 지표로 교체)**: 독립 seedbank에서 `S−Q30<0` 와 `S−K<0`
   의 **시간당 누적 v2 비용** 신뢰구간 상한이 모두 0보다 작아야 채택한다.
   **에피소드 총비용은 쓰지 않는다** — 관측시간 종료 구조에서는 정책별 노출이 다르다.
4. **G3 보호 (★31차 — 지속 유입 지표로 교체)**: **backlog 증가 기울기**·**시간가중
   재공량·대기열**·평균 A→O·본선 시간당 지연비용이 사전 허용손실 안이어야 하고,
   **장부 누락 0**(미완·검열 표본 제외 금지)·정책 예외 0 이어야 한다.
   **"최종 backlog 0·완주 100%"는 요구하지 않는다** — YR-150 은 작업이 남아도 관측시간에
   종료하는 구조이므로 그 요구는 구조와 충돌한다(유한 물량 구조에서만 유효한 가드였다).
   정상상태에 도달하지 못한 셀은 비용 판정 대신 처리량·backlog 기울기만 보고한다.
5. **G4 보고**: offer/accept/commit/rollback, 방향, route 비용, 안전장치 개입률, P95는
   전량 보고하되 P95는 채택 veto로 쓰지 않는다.

전 부하 채택은 **모든 비용판정 가능 셀**에서 G2·G3가 통과할 때만 선언한다. 일부 셀만
통과하면 해당 입력부하 한정 결과이며 글로벌 SELL 채택은 금지한다.

해석 분기는 고정한다. `S<Q30`지만 `S>=K`면 “나쁜 계산 발의를 걸렀으나 이송의 절대 이득은
없음”, `S<K`지만 `S>=Q30`면 “이송은 유효하나 학습 head 추가가치 없음”이다. 두 기준을 모두
이기고 보호게이트를 통과할 때만 “Block PPO SELL 추가가치”라고 쓴다.

## 선결·참조

- [YR-133 원자 이송·견적](YR-133-blockq-sell-quote.md) · [YR-148 채택 실행 구성](YR-148-guard-on-rejudgment.md) · [YR-149 5부하 자격](YR-149-quote-refine-confirm.md) · [YR-150 본 성능환경](YR-150-continuous-inflow-steady-state.md) · [21블록 사용자 결정](../strategy-history/2026-08-06-YR-150-151-터미널전체-21블록-사용자결정.md)
