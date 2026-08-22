# YR-214 · v3 패키지 전면 구현 (테스트·판정은 유예)

> Epic Infra · 상태 ✅ **완료(코드 한정)** 2026-08-22
> 사용자 지시 2026-08-22: *"docs 에 있는 v3 문서와 대시보드를 참고해서 v3 를 구현해줘.
> v2 와 혼동해서 v2 버전으로 만들지 않도록 해야해. **테스트는 마지막에 할꺼니까
> 우선 코드 구현부터 다하면 말해줘.**"*

## 1. 무엇을 하는 작업인가

아키텍처 문서(`.claude/docs/architecture/`)가 **to-be 설계**이고 코드가 아직
v2 뿐이었다. 이 row 는 그 설계를 **`src/yard_rl/v3/` 에 전부 코드로 옮기는 것**이다.
성능 판정은 하지 않는다 — 판정은 [[YR-210]]·[[YR-211]]·[[YR-212]]·[[YR-213]] 의 몫이다.

## 2. 왜 지금인가

문서 35개 계약 중 **13개가 `_target`(코드 미도달)** 이라 "설계는 있는데 실행체가
없는" 상태였다. 이 상태에서는 어떤 판정도 못 돌린다.

## 3. 범위 — 네 축 + 진입점 둘

| 축 | 패키지 | 파일 |
|---|---|---|
| ① 데이터 | `v3/schema/` | `lifecycle.py` · `order.py` · `record.py` |
| ② 보상 | `v3/reward/` | `krw.py` · `phi.py` · `counterfactual.py` |
| ③ 구조 | `v3/actors/` | `offer.py` · `nets.py` · `seller.py` · `buyer.py` · `resolver.py` · `market.py` |
| ④ 정보 | `v3/features/` | `block.py` · `candidate.py` |
| 학습 진입점 | `v3/train/` | `labels.py` · `fit.py` |
| 판정 진입점 | `v3/eval/` | `guards.py` · `run.py` |

**무대(`integrated/`)는 복제하지 않고 덧붙이기만** 했다 — 세 세대가 같은 무대를
받아야 짝비교가 성립하기 때문이다. 기존 상수·기본값은 하나도 안 바꿨다.

| 무대 확장 | 내용 |
|---|---|
| `yard_layout.py` | `quay_axis_s()` · `quay_to_block_s()` · `yt_round_trip_s()` — 안벽을 축 반대 끝에 대칭 배치. 축 600초는 **기존 190+410 에서 유도**돼 새 자유변수가 없다 |
| `vessel.py` | `VesselClass` · `VESSEL_CLASSES`(3종) · `PORT_TIME_TABLE` · `port_time_s()` · `sample_vessel_moves()` · `YT_PER_STREAM` |
| `terminal_stream.py` | `DIURNAL_LOAD_LEVELS`(3,500·5,000·7,500) · `LEAD_TIME_DIST` · `sample_lead_s()` |

## 4. 판정 기준 (이 row 한정)

| | 기준 | 결과 |
|---|---|---|
| 임포트 | `yard_rl.v3` 25 모듈 전부 | **실패 0** |
| 문서↔코드 계약 | `verify_architecture.py` | **계약 36 · 일치 44 · 불일치 0 · 미배선 0 · 진행중 0** |
| 무대 회귀 | 기존 `tests/` 전량 | **852 통과 · 9 실패(전부 선존재 — 내 변경 전에도 같음)** |
| 조립 | 오더→Seller→Buyer→Resolver→Φ→교사→학생 1회 관통 | 통과 |

**선존재 실패 9건**은 `git stash` 로 내 변경을 뺀 상태에서 동일하게 재현됨 →
[[YR-111]]·[[YR-191]] 계열이고 이 row 의 산출물이 아니다.

## 5. 이 row 가 **하지 않은** 것

- 성능 판정·학습 실행 (시드 대역 소모 없음)
- 무대 결합 — `Market` 을 실제 `MultiBlockTerminal` 루프에 꽂는 배선
- 반사실 교사의 실제 rollout 구현 (`rollout_fn` 은 무대가 주입할 자리로만 비워둠)
- v1·v2 코드 수정 (세대를 얼린다)

## 6. 부수 정정 (같은 커밋)

| 무엇 | 전 | 후 |
|---|---|---|
| 부분요인 런 수 | 624 | **528** — 주판정 336 + 일반성 288 이 **96 셀을 겹쳐 세고 있었다**(안팔기·RL × 계획법). 겹침은 한 번만 돈다 |
| 계약 검증 배선 | `_target` 을 v2 코드로 채점 | v3 로 옮김. **무대 축**(리드·부하·선급)만 `integrated` 유지 |
| 결함 노트 | "고쳐짐 → 문서 갱신 필요" | **세대별** 표기 — `v2 그대로(정상) · v3 ✅ 고침` |

## 7. 다음

이 row 는 실행체를 만든 것뿐이다. 다음은 **무대 결합**(Market ↔ MultiBlockTerminal)
이고, 그 뒤라야 [[YR-210]] 부하 스윕이 돌아간다.
