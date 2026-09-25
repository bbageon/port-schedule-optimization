# v6 — 배열 세계 · GPU · 전체 오더를 한 번에

v5 를 독립 복사한 뒤(기준 커밋 `7745dbc2` · 2026-09-25) **가속기에서 도는 배열 세계**를
더한다. 원본 v5 는 수정하지 않는다. 작업 명세: `.claude/docs/dashboard-task-specs/YR-327-*.md`.

## 왜 만드나

v4 의 반사실 교사는 **신호 대 잡음을 2만 배** 올려 주지만 값이 비싸다 —
계산의 **94% 가 세계 복제**이고 결정의 **0.43%** 만 라벨이 된다. v5 는 복제를 없앴지만
보상이 **공유 팀 보상**이라 누구 덕인지 못 가린다.

| | v4 | v5 | **v6** |
|---|---|---|---|
| 세계 복제 | 결정마다 | 없음 | 없음 |
| 쓰는 결정 | 0.43% | 100% | 100% |
| 신용 배분 | 반사실 (정확·비쌈) | 공유 팀 보상 | **반사실 (망 안에서)** |
| 계산 | CPU 프로세스 10~20 | CPU | **GPU 배치 수천** |

## 무엇이 바뀌었나

    파이썬 객체 세계 (v5)          배열 세계 (v6 `gpu/`)
    ────────────────────────────────────────────────────────
    heapq 우선순위 큐          →   (시각·종류·대상·순번) 배열, v5 와 같은 3단 키
    dict[job_id] → Job         →   길이 고정 배열, 빈 칸은 -1 / +inf
    if 조건: A else: B         →   둘 다 계산하고 where 로 고름
    찾을 때까지 훑기(find_slot) →   격자 전수 계산 + 3단 argmin
    blocker 를 하나씩 치우기    →   고정 길이 scan (M−1 단)
    sim.assign(...) 제자리 수정 →   상태를 받아 새 상태를 돌려주는 순수 함수

### 파일

| 파일 | 무엇 | v5 대응 |
|---|---|---|
| `events.py` | 사건 큐 (3단 키 · 넘침 표시) | `integrated/events.py` |
| `geom.py` | 블록 기하·스펙 상수 (jit static) | `BlockGeometry`·`CraneSpec` |
| `state.py` | `BlockWorld` — 오더·크레인·스택·컨테이너·예약·계획·KPI·장부·비용 13항·레인·로그 | `engine.py` 상태 전부 |
| `exact.py` | `mul_exact` — 곱을 실체화해 FMA 융합을 막는다 (아래) | — |
| `travel.py` | `move_container` (10항 좌결합 순서 그대로) | `sim/travel_time.py` |
| `stack_ops.py` | `find_slot`·`blockers_above`·`rehandle_capacity_ok`·`place`·`remove` | `sim/stack.py` |
| `reserve.py` | `reject_code` (5-lock 순서 고정)·`reserve`·`release` | `reservation.py` |
| `plan.py` | `plan_serve` — STORE / RETRIEVE(재조작 scan) | `engine._plan` |
| `host_convert.py` | 시나리오 → 배열 · 배열 → v5 비교용 dict · 사건 로그 복원 | `engine.reset` |
| `engine_step.py` | `advance`·처리기·`decide`·`step`·`run`(lax.scan) | `run_until_decision`·`assign`·`_complete` |
| `policy.py` | 전 오더를 한 번의 순전파로 · `Q = V + A` · 반사실 기준선 | (신규) |

## ★조각 1 — 단일 블록 엔진이 v5 와 **같은 답**을 낸다 (2026-09-25)

v5 `TerminalSimulator` 를 규칙 정책(첫 후보·마지막 후보·둘째는 WAIT) 으로 끝까지 돌린 답과
배열 엔진(`run` jit) 의 답을 **`==` 로** 대조한다 — 사건 로그 전열(시각·종류·대상)+해시,
결정열, 오더 상태, 계획 이동표, 크레인, KPI, 격자, 위반 0, 실수 전 항목(시각·주행거리·
대기 적분·장부 적분·비용 13항). 무대 12종(§10 시나리오·혼잡·검열·fixture 본선 제거판).

    tests/v6/test_gpu_*.py  130 건  (CPU x64 · 220초)
    그중 GPU(RTX 5090)에서  49 건  (core 17 + 동등성 32 · 244초) — **GPU 도 비트 단위 일치**

각 단계가 **v5 를 실제로 불러** 같은 입력의 답을 받는다(기대값 손기입 없음):
find_slot 5야드×300질의 · 이동시간 500건 비트 동일 · 예약 거절 6,000질의 · 계획 여러 시점.

### ⚠️ FMA — 플래그로는 못 막는다

`XLA_FLAGS=--xla_allow_excess_precision=false` 를 켜도 **CPU 는 `x*y+z` 를 한 번에 반올림**
(FMA)한다 — v5(파이썬)는 두 번 반올림하므로 대기 적분·find_slot 비용의 마지막 비트가 갈린다.
`exact.mul_exact` 가 `optimization_barrier` 로 곱을 실체화해 두 번 반올림을 강제한다.
`test_gpu_core` 의 `[fma probe]` 가 백엔드마다 매번 찍는다:

    cpu  plain=FMA        guarded=two-round   ← 보호 필요
    gpu  plain=two-round  guarded=two-round   ← 이 패턴에선 융합 안 함 (실측)

### 부동소수점 규약

동등성은 **float64(x64)** 에서만 — v5 는 `_EPS=1e-9` 비교이고 하루 끝(86,400초)에서 float32
이웃 간격은 7.8ms 다. `TIME_DTYPE` 이 float64 인데 x64 가 꺼져 있으면 `empty_queue` 가
큰 소리로 실패한다(조용히 float32 로 내려앉는 함정). 학습 모드 float32 는 별도 결정.

## 지금 어디까지 왔나

    ✅ 골격 (사건 큐 · 상태 · 정책망 · vmap)
    ✅ 조각 1  단일 블록 엔진 — 크레인 1대 · 트럭 오더 · 스택 · 재조작 · 예약 · 비용 적분
    ⬜ 조각 2  다중 크레인 — 간섭 · 순차 예약 · 교착 탈출 · 장비 고장
    ⬜ 조각 3  PRE_ADVICE — ETA wake · PRE_REHANDLE / REPOSITION / WAIT 후보
    ⬜ 조각 4  본선 · 이송 — STS · 양하 해제 · 이송차
    ⬜ 조각 5  비용 Φ 4항 (원화)
    ⬜ 조각 6  다중블록 조정자 · 조각 7 결정 계층 · 조각 8 학습 루프 (30일 무대 포함)

**⚠️ 지금 v6 는 v5 를 대체하지 않는다.** `gpu/` 밖은 전부 v5 사본이다. 정답 궤적
(`outputs/reports/yr327_v6_port/ground_truth/`)의 블록 Y01 은 본선 240건이라 **조각 2·3·4 가
되어야** 재현할 수 있다. 전부 옮기기 전까지 v5 가 정본이고 v6 로 판정하지 않는다.

### 알려진 한계 (조각 1)

- K≥2: 정책을 배정 scan **앞에서** 한 번 부르므로 두 크레인이 같은 오더를 고르면 위반 16.
  v5 `ReferenceDispatcher`(순차 재선택)와 같으려면 조각 2 가 정책 호출을 scan 안으로.
- 비트 일치를 위해 대기 꼬리·장부 적분을 v5 삽입 순서 그대로 N·2N 단 scan 으로 돈다 —
  CPU 실측 N=256 스텝당 10.8ms. 학습 모드(float32)에서는 닫힌 식으로 바꿔도 된다.
- `reserve.py` 와 `state.py` 가 `ReservationArrays` 를 따로 정의 — 통일 예정.
- Windows 파이썬엔 jax 가 없어 시험이 `importorskip` 으로 조용히 건너뛴다 — **WSL venv 로만**.

## 돌려 보기

    # CPU x64 전체 (WSL venv ~/.venvs/yard-rl)
    PYTHONPATH=src JAX_PLATFORMS=cpu pytest tests/v6/test_gpu_*.py -q -s
    # GPU
    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/v6/test_gpu_engine_equiv.py -q -s
    # 정답 궤적 · 정책망 벤치
    PYTHONPATH=src python scripts/v6/dump_ground_truth.py --load 300
    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/v6/bench_gpu.py
