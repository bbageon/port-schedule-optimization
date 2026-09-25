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
    크레인을 차례로 배정        →   크레인 순 lax.scan (정책 호출이 scan 안)
    본선이 작업을 만들어냄      →   호스트가 생길 job 칸을 미리 잡아 둠
    sim.assign(...) 제자리 수정 →   상태를 받아 새 상태를 돌려주는 순수 함수

### 파일 (`gpu/` 6,380줄 · 시험 8,441줄)

| 파일 | 무엇 | v5 대응 |
|---|---|---|
| `events.py` | 사건 큐 (3단 키 · 넘침 표시) | `integrated/events.py` |
| `geom.py` | 블록 기하·스펙 상수 (jit static) | `BlockGeometry`·`CraneSpec` |
| `state.py` | `BlockWorld` — 오더·크레인·스택·컨테이너·예약·계획·KPI·장부·비용 13항·레인·로그·깨우기·본선·이송·위반 비트 | `engine.py` 상태 전부 |
| `exact.py` | `mul_exact`·`sum_python`(보정합) — FMA·상수 재결합·`sum()` 함정 방어 | — |
| `travel.py` | `move_container` (10항 좌결합 순서 그대로) | `sim/travel_time.py` |
| `stack_ops.py` | `find_slot`·`blockers_above`·`rehandle_capacity_ok`·`place`·`remove` | `sim/stack.py` |
| `reserve.py` | `reject_code` (5-lock 순서 고정)·`reserve`·`release` | `reservation.py` |
| `plan.py` | `plan_serve` — STORE / RETRIEVE(재조작 scan) | `engine._plan` |
| `cands3.py` | 확장 후보 — SERVE + PRE_REHANDLE + REPOSITION + WAIT, 목표 bay 집합 | `candidates_for`·`CandidateGenerator` |
| `wake.py` | ETA wake·armed·DEFER wake·결정 개방(armed 항)·eta_opportunity | `engine.py:339-382, 453-464` |
| `dispatch.py` | 후보 행렬·순차 배정(`ReferenceDispatcher`)·공동 결정(`resolve_central`, SF_SPT 등)·`dry_run` | `dispatcher.py`·`baselines.py`·`commit_decisions` |
| `escape.py` | 교착 술어·탈출 개방 (immediate) | `engine._try_escape` |
| `vessel.py` | 본선(STS·양하 해제 사전식 순위·계획 변경·선석 초과)·이송차 링버퍼 | `engine.py` 본선 처리기·`vessel.py`·`transfer.py` |
| `host_convert.py` | 시나리오 → 배열(본선이 만들 job 칸 예약) · 배열 → v5 dict · 로그 복원 | `engine.reset` |
| `engine_step.py` | `advance`·처리기 12·`decide`·`step`·`run`(scan)·`run_while`(학습용) | `run_until_decision`·`assign`·`_complete` |
| `phi.py` | Φ 원화 4항(대기·이동·재조작·본선)·검열·분위수 | `reward/phi.py` |
| `policy.py` | 전 오더를 한 번의 순전파로 · `Q = V + A` · 반사실 기준선 | (신규) |

## ★블록 하나가 v5 와 **완전히 같은 답** — 조각 1·2·3·4·5 (2026-09-25/26)

v5 를 실제로 돌린 답과 배열판을 **`==`** 로 대조한다(허용 오차 없음, 기대값 손기입 없음).

| 조각 | 무엇 | 시험 | 대조 |
|---|---|---|---|
| 1 | 단일 블록 엔진 | 130 | 사건 로그·결정·오더·계획·크레인·KPI·격자·실수 전 항목, 무대 12종 |
| 2 | 다중 크레인 — 순차 배정·간섭·교착 탈출·장비 고장 | 68+89+44 | K=2·3 lockstep 27무대 · v5 `_try_escape` **1,269회** 가로채 대조 |
| 3 | PRE_ADVICE — ETA/DEFER wake·PRE_REHANDLE/REPOSITION/WAIT | 37+18 | v5 wake 호출 303회 가로채기 · 결정 356건 후보 집합 == |
| 4 | 본선·이송 — STS·양하 해제·이송차·계획 변경·선석 초과 | 16 | 본선 사건마다 상태 가로채기, 무대 4종 |
| 5 | Φ 원화 4항 | 16 | 손입력·v5 무대 41시점·무작위 300건 |

### ★정답 궤적 Y01 재현 (`tests/v6/test_gpu_y01.py`)

`scripts/v6/dump_ground_truth.py` 가 v5 로 굴린 **블록 Y01 단독**(본선 240건·크레인 2·사건 1,318·결정 237·
재조작 259·REPOSITION 3회)을 배열 엔진 `run`(jit) 이 **사건 하나하나·해시·비용 13항·KPI 까지 그대로** 낸다:

    sf_spt / PRE_ADVICE   (원본)   해시 6668fa4902c4efe4   ← SF_SPT 규칙 + LEGACY 후보 + 공동 결정
    sf_spt / BLOCK_ARRIVAL          해시 6668fa4902c4efe4
    reference · first / …          해시 760af9a960d07bf8   ← 순차 규약 (사건 1,321)

    CPU x64: 한 조합 ≈6초 (N=256 · S_max 2,304 · 컴파일 포함)    GPU(RTX 5090): 원본 궤적 통과 (38초)

    tests/v6/test_gpu_*.py  약 450 건 — CPU x64 통과 · GPU 는 조각 1·2·5 와 Y01 원본 궤적 통과

### ⚠️ 부동소수점 함정 셋 — 전부 시험이 상시 탐침한다

| 함정 | 무엇 | 처리 |
|---|---|---|
| **FMA** | `XLA_FLAGS=--xla_allow_excess_precision=false` 를 켜도 CPU 는 `x*y+z` 를 한 번에 반올림 | `exact.mul_exact` (optimization_barrier). GPU 는 이 패턴 융합 안 함 |
| **상수 재결합** | jit 한 `K*v/3600` 을 `v*(K/3600)` 으로 — 4,000점 중 1,251점 갈림 (GPU 도 같음) | 곱 실체화 + 나눗셈 장벽 |
| **Python 3.12 `sum()`** | 보정합(Neumaier)이라 순차 `+=` 와 3항부터 갈림 | v5 가 `sum()` 인 곳(검열 노출·레인 평균·불균형·본선 유휴)은 `exact.sum_python` |

동등성은 **float64(x64)** 에서만 — `TIME_DTYPE` 이 float64 인데 x64 가 꺼져 있으면 `empty_queue`
가 큰 소리로 실패한다. 학습 모드 float32 는 별도 결정.

### GPU 성능 — 비트 일치의 값

GPU 스텝 시간의 **약 90%** 가 대기 꼬리·터미널 점유 적분의 **N·2N 단 직렬 scan**(v5 삽입 순서 그대로
더해 비트를 맞추는 대가)이다 — 커널 왕복이 N 에 비례하고 vmap 으로 세계를 쌓아도 안 줄어든다.
`ADVANCE_UNROLL=16`·`run_while`(while_loop) 학습 경로로 N=64·K=2 스텝 6.0→2.5ms. 공동 결정의
`resolve_central` 쌍 scan(길이 K·(k_max+1))도 vmap 아래서는 전 갈래를 돈다. **학습 모드(float32·
동등성 불필요)에서는 닫힌 식·가지치기로 바꿔야** 한다 — 조각 8 의 몫.

## 지금 어디까지 왔나

    ✅ 골격 · 조각 1 단일 블록 · 2 다중 크레인 · 3 PRE_ADVICE · 4 본선·이송 · 5 Φ  — **블록 하나 완성**
    ⬜ 조각 6  다중블록 조정자 — 21블록 vmap × 검토 에폭 scan · 게이트 투입 · 원장 · 이송 확정
    ⬜ 조각 7  결정 계층 — 37특징 정책·반사실 우위 (SF_SPT 등 규칙 resolver 는 이미 됨)
    ⬜ 조각 8  학습 루프 — 60초 구간 보상·PPO · **30일 무대(run_month)** 포함 · 학습 모드 닫힌 식

**⚠️ 지금 v6 는 v5 를 대체하지 않는다.** 블록 *하나*가 같을 뿐, 터미널 전체(21블록·게이트·이송 확정·
30일 무대)는 아직 v5 파이썬이다. 전부 옮기기 전까지 v5 가 정본이고 v6 로 판정하지 않는다.

### 알려진 한계

- DEFER wake(유한 대기 뒤 재개방)는 단위 시험만 — 정책 반환에 `defer_until` 이 없어 통합 엔진이 못 만든다(조각 7).
- 후보 설정은 `LEGACY_DEFAULT` 한 경로만 (safety_only·bound_repo 등 `policy_config` 플래그 미이식).
- 탈출 `delayed` 모드 · `yard_handover_cap`(opt-in) · `vessel_cost.py`(정책 측 surrogate) · review epochs(조각 6) 미이식.
- 정책망을 scan 안에서 K 번 부르며 (K,N,F) 전체를 넘긴다 — K² 중복 순전파(조각 7 과 함께).
- `test_k2_python_loop_matches_jit`(eager ≈80초)는 아래 환경에서 완주 불가 — 나눠서 통과 확인.
- Windows 파이썬엔 jax 가 없어 시험이 `importorskip` 으로 조용히 건너뛴다 — **WSL venv 로만**.

### ⚠️ 환경 (2026-09-25 22:20 ~)

이 기계의 WSL(Ubuntu)이 **부팅 약 88초 뒤 저절로 종료**된다(배포판 내부·Windows 이벤트·프로세스에
원인 없음, `wsl --terminate` 로는 안 고쳐짐). 정석은 `wsl --shutdown`(VM 재시작)이나 Docker 컨테이너
3개가 멈추므로 사용자 결정 대기. 그동안 `scripts/v6/verify_chunked.sh` 가 85초 조각으로 전부 돌린다.

## 돌려 보기

    # 정상 환경 — CPU x64 전체 (WSL venv ~/.venvs/yard-rl)
    PYTHONPATH=src:tests/v6 JAX_PLATFORMS=cpu pytest tests/v6/test_gpu_*.py -q -s
    # Y01 정답 재현 (GPU)
    PYTHONPATH=src:tests/v6 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/v6/test_gpu_y01.py -q
    # WSL 세션이 짧게 끊기는 환경 — 조각 실행 (Git Bash)
    scripts/v6/verify_chunked.sh --fresh && scripts/v6/verify_chunked.sh --report
    # 정답 궤적 새로 뽑기 (정책·정보수준 선택)
    PYTHONPATH=src python scripts/v6/dump_ground_truth.py --mode block --policy sf_spt --info-level PRE_ADVICE
