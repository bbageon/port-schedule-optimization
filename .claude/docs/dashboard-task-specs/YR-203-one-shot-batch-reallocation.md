# YR-203 — 최초 통지 1회·독립 SELL/BUY 요청·배치 Resolver 계약

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-08-19 / **상태**: backlog
- **3대 게이트 보정 대상**: `performance` 후속 구조 — 현재 게이트 해소 전 착수 금지
- **선행**: [[YR-200]]→[[YR-201]]→[[YR-202]] 판정과 별도 착수 허가
- **1줄**: Seller와 Buyer가 각각 `KEEP/SELL`, `REJECT/BUY` 요청을 내고, 양쪽이
  동의한 작업만 Resolver가 같은 시점 신규 작업 전체와 함께 **한 배치로 공동 배정**한다.

## 왜 필요한가

현재도 `MAX_TRANSFERS=1`이라 실제 A→B→C 재이송은 막힌다. 그러나 KEEP한 작업은
30분 창 안에서 60초마다 다시 검토될 수 있고, 한 블록은 같은 epoch에 후보 여러 건 중
1건만 제안한다. 이 계약은 두 모호함을 없앤다.

- 반복 제안·거절이 만든 학습 표본 중복과 공로 왜곡을 제거한다.
- 동시에 들어온 여러 작업을 독립 배정해 한 수신 블록에 몰아넣는 오류를 막는다.
- 배정(어디서 처리)과 실행(언제·어떻게 처리)을 분리해 단일축 실험을 가능하게 한다.

## 결정 사건 — 물리 도착이 아니다

판단 시점은 컨테이너가 블록에 실제 도착한 때가 아니라, **공개 ETA가 재배치 가능
창에 최초 진입한 60초 epoch**다.

```text
이전 epoch: ETA−now > WINDOW_S
현재 epoch: 0 < ETA−now ≤ WINDOW_S
→ 이번 배치 J_t 에 최초 1회 포함
```

같은 epoch에 처음 자격을 얻은 모든 작업만 `J_t`에 넣는다. 아직 통지되지 않았거나
다음 epoch에 자격을 얻을 작업을 미리 넣지 않는다. KEEP 또는 재배정 뒤에는 모두
`allocation_decided=True`로 잠가 다시 거래하지 않는다.

## 독립 행동요청 계약

Seller와 Buyer는 Resolver의 하위 계산식이 아니라 서로 다른 정책 actor다.

```text
Seller A: a^S_A(j) ∈ {KEEP, SELL}
Buyer  B: a^B_B(A,j) ∈ {REJECT, BUY}

거래 후보 (A,B,j)
⇔ SELL_A(j) ∧ BUY_B(A,j)
```

Seller는 작업별 `SELL` 제안을 방송하고, 각 허용 Buyer는 그 제안에 `BUY/REJECT`로
독립 응답한다. Buyer 관측에는 자기 블록·작업 정보와 Seller가 공개한 offer message만
허용한다. Resolver는 양쪽 동의가 없는 edge를 새로 만들거나 강제할 수 없다.
SELL에 상호 동의 edge가 없거나 batch에서 선택되지 않으면 `RESOLVER_KEEP`으로 잠근다.

Resolver는 동의 edge 집합에서 용량·소유권·동시오더 상호작용을 지키는 최종 목적지
벡터 `a_t=[dest(j_1),...,dest(j_m)]`만 확정한다. 여러 Buyer가 같은 작업을 원하면 하나만,
한 Buyer가 여러 작업을 원하면 배치 용량 안에서만 선택한다. 처리 실행은 기존 동결
ExecutionHead의 `SERVE/PRE_REHANDLE/REPOSITION/WAIT`가 맡는다.

## 사건 안의 순서

```text
이번 epoch 신규 작업 J_t
→ Seller actor들의 KEEP/SELL
→ SELL 작업에 대한 Buyer actor들의 REJECT/BUY
→ 상호 동의 edge만 Resolver 입력
→ batch 선택·원자 확정
→ KEEP/판매 모두 allocation_decided=True
```

독립은 각 actor가 자기 행동을 고른다는 뜻이다. Resolver가 두 actor의 행동을 대신
계산한다는 뜻도, Buyer가 자기 지역비용만 이기적으로 최소화한다는 뜻도 아니다.

## 구현 범위

1. 작업 장부에 최초 판단 완료 표식을 추가한다.
2. 후보 생성기를 `현재 창 안 전체`가 아니라 `이번 epoch 신규 자격`으로 바꾼다.
3. Seller request, Buyer response, Resolver selection을 서로 다른 원장 항목으로 남긴다.
4. 블록당 1건 제안 상한을 제거하고 상호 동의 edge 전체를 Resolver에 넘긴다.
5. Resolver는 배정마다 가상 부하·용량을 갱신해 다음 한계비용을 다시 계산한다.
6. KEEP도 결정으로 기록하고 동일 작업의 두 번째 판단을 fail-closed로 막는다.

## 수용 기준

- 작업별 재배정 판단 횟수 정확히 1회; KEEP 포함 재등장 0건.
- 같은 epoch 신규 작업이 2건 이상이면 단일 batch ID로 전부 기록.
- `SELL` 없는 거래, `BUY` 없는 거래, Resolver를 우회한 직접 이송 각각 0건.
- Seller request·Buyer response·Resolver 선택을 `(batch_id, job_id, src, dst)`로 추적 가능.
- 입력 작업·블록 순서를 바꿔도 최종 배정 집합 불변.
- 수신 용량 초과·중복 소유·장부 손실·미래정보 누출 0건.
- 기존 ExecutionHead 행동·가중치·비용식 불변을 hash와 회귀시험으로 확인.
- N=3 계약시험 뒤 N=21에서 물리·완주 가드 통과. 이 단계에서는 성능 주장을 하지 않는다.

## 하지 않는 것

- Seller/Buyer actor 학습과 운영 추론 — [[YR-205]] 몫.
- Seller `KEEP`·Buyer `REJECT` 반사실 비용 라벨 — [[YR-204]] 몫.
- 긴 미래 반사실 rollout — [[YR-204]] 몫.
- PREMOVE·WAIT 실행정책 재학습, 명시적 가격·경매, 실제 TOS API 연동.
