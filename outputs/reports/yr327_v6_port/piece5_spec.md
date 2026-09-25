# v6 조각 5 구현 명세 — 비용 Φ 네 항(원화)·검열 수정·계수기 사진 — 보상의 원천

원본: outputs/v6/port_plan_wf_9e7366e6.json (plan.pieces[4])

files_to_port:
- reward/phi.py:62-127 (terminal_cost_krw), krw.py:62-95 (단가 함수)
- schema/lifecycle.py:129-142 (censored_turn_time_s — gpu/state.py:131-141 을 end=min(O,end_s) 절단으로 고침)
- stage/episode.py:225-310 (yc_empty_travel_s·rehandles_of·vessel_idle_of·_CounterTape → 고정 격자 사진 배열)
- ppo/runtime.py:115-121, 152-196 (read_cost·boundary — 플래그 판)

hard_parts:
- ★gpu/state.py:137-141 은 O 가 있으면 end_s 로 안 자른다 — v5 규약(end=min(O,end_s))과 다름; 손계산 사례(A=500,O=9000,end=7200 → tt 6700)가 먼저 깨져야 정상
- 분위수는 정렬 후 index=min(n−1, floor(p·n)) 원소 자체(phi.py:115-118) — jnp.quantile 보간 금지
- 원화 1e7 규모에서 float32 는 1원 단위가 흔들려 delta<−1e-5 오탐 → x64 또는 문턱 완화

array_technique:
(N,) tt 벡터 + is_truck·day 열 마스크 합; 배별 유휴 = segment_sum(sts_wait_accum, ship_idx)/sts; 사진 = (G,) 격자 + searchsorted.

equivalence_test:
A단계: 손입력 6건(A/O 조합)·end_s∈{3600,7200,10800}·vessel_idle 2척·yc 123.4·rehandles 3 — 기대값 wait 280,000.00·move 2056.67·rehandle 6000·vessel 36129.17·total≈324185.83, n_trucks 4·n_censored 2·p50 6700·p90=p99 7000; x64 rtol 1e-9, 정수·분위 비트 일치. B단계: tests/v6/test_cargo_capacity_wait.py:24-73 world() 무대 v5 로 굴려 t 마다 원료(A/O, empty_m, rehandles, sts_accum) JSON 저장 → 배열 phi 만 같은 t 에 계산, |Δ|≤1e-9·max(1,Φ), Φ 비감소.

예상 규모: ~180줄

의존: ['1']

## 관련 event_kinds (설계)