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

### 파일 (`gpu/` 9,123줄 · 시험 약 1.2만 줄)

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
| `admission.py` | 게이트 투입 — 명단·통지 리드(트럭별)·검토 에폭 버킷·`free_slots`·SKIP 6종 | `ScheduledAnnouncer`·`admit_external_job` |
| `transfer_txn.py` | 이송·이연의 **원자 확정** (prepare/validate/commit/rollback) | `multiblock.py:294-652` |
| `host_terminal.py` | 터미널 변환기 — 블록 축 `(B,…)` 묶음·전역 원장·명단 배열 | `MultiBlockTerminal.__init__` |
| `multiblock.py` | **조정자** — 21블록 vmap × 검토 에폭 scan × while(다음 에폭까지) | `MultiBlockTerminal.run` |
| `ledger.py` | (참조 구현 — 조정자 경로 밖. 원장 정본은 `host_terminal.TerminalLedgerArrays`) | `ledger` 필드 대조용 |
| `policy.py` | 전 오더를 한 번의 순전파로 · `Q = V + A` · 반사실 기준선 | (신규 · [[YR-326]] 새 축) |
| `v5net.py` | **학습 정책망** — tanh·37칸·행동점수+상태가치, torch `state_dict` 전치 적재 | `ppo/model.BlockPolicy` |
| `v5feat.py` | 37칸 특징 (블록 요약 8 + 후보 16 + 패딩 8 + 역할 5) — 인코딩 **정본** | `features/block.py`·`ppo/crane.candidate_row` |
| `v5cond.py` | 순차 조건부 `joint_mask` · 선호 정렬키 **정본** · "한 크레인 실패 = 전원 WAIT" | `ppo/crane.py`·`baselines.py:28-60` |
| `ppo_buffer.py` | 학습 버퍼 고정 칸 `(R,B,Cmax,Amax,37)` + 역방향 GAE(`lax.scan`) | `ppo/buffer.py` |
| `ppo_update.py` | PPO 갱신 — 미니배치 scan·KL 조기중단·torch 식 클리핑+Adam | `ppo/update.py` |
| `ppo_runtime.py` | 학습 회계 — 60초 경계·구간 보상·결정 기록·보고 (예외는 위반 비트) | `ppo/runtime.py` |
| `month.py` | **30일 무대** — 본선 붙이기·날 경계·계수기 사진첩·날별 Φ (기본 갈래만) | `stage/month_run.py`·`month_engine.py` |
| `train.py` | **학습 드라이버** — 넷을 엮는다 (기록 정책·테이프 합치기·경계·갱신) | `ppo/continuous.py`·`ppo/run.py` |

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
    🟡 조각 6  다중블록 조정자 — 21블록 vmap × 검토 에폭 scan · 게이트 투입 · 원장 · 이송 확정
               (사다리 ①②③ 통과 · 아래 "조각 6 은 어디까지 같은가" 참조 — CargoTerminal(30일) 은 범위 밖)
    🟡 조각 7  학습 정책망 — tanh·37칸 망 + 순차 조건부. **같은 가중치 → 같은 결정** (아래 절 참조)
               (블록 하나 무대에서 확인 · 터미널 21블록 배열 대조는 조각 8)
    🟡 조각 8  학습 루프 — 60초 구간 보상·PPO·30일 무대 · **학습 켠 닫힌 고리까지 v5 와 같은 답**
               (시장 결정을 떼어낸 무대에서 · 아래 절) · 추첨(sample) 수집은 배열판 고유 난수로 이식
               (남은 것: **시장 다리**·**고정 화물 갈래**(연구선의 유일한 경로)·증거 배선·학습 모드 닫힌 식)

**⚠️ 지금 v6 는 v5 를 대체하지 않는다.** 블록 *하나*가 같을 뿐, 터미널 전체(21블록·게이트·이송 확정·
30일 무대)는 아직 v5 파이썬이다. 전부 옮기기 전까지 v5 가 정본이고 v6 로 판정하지 않는다.

### 조각 6 — 어디까지 같은가 · 얼마나 빠른가 (2026-09-26)

같은 것: 블록별 사건 전열·해시(21/21) · 비용 13항 · KPI 10항 · 투입 원장 전행(ADMIT/SKIP/EPOCH) ·
`JobRecord.locked` 전건 · 오더별 (status·크레인·재처리·S·C·A·B·O) · 시간장부 적분 3항 · 턴 표본과 합 ·
이송/이연 원장 8항 · 이연 원장(기사 외부대기). 정답은 `outputs/reports/yr327_v6_port/ground_truth/`.

**★속도 — 세계가 하나면 v5 파이썬이 더 빠르다.** 같은 무대(터미널 300 · 21블록 · 사건 20,720)에서
v5 정본 루프 16~17초 대 배열판 133초(GPU)·576~673초(CPU). lane 이 21개뿐이라 GPU 가 노는 것이고,
이득은 **세계를 쌓을 때** 난다 — 실측은 조각 8 의 손익분기 표(`README-piece8.md`)를 보라.

곁들여 아는 낭비: 래기드 while 1.85배(에폭당 계산 63,042 블록·스텝 vs 필요 34,187 = 유효율 54.2%) ·
오더 칸 패딩 27%(블록별 본선 척수가 1 또는 2라 n_used 134 / 255) · vmap 아래 cond→select 로 결정이
필요 없는 94%(63,042 중 결정 3,905 = 6.2%)도 계산. "배치 cond" 처방의 상한은 1.9배(48%)다.

### 조각 7 — 학습 정책망은 어디까지 같은가 (2026-09-26)

**상세는 `.claude/docs/dashboard-task-specs/YR-327-v6-gpu-array-world.md` 의 "조각 7" 절** (200줄 규약).
요약: (a) 결정마다 블록 요약 8칸·후보 37칸(f32·f64)·마스크·선택 ==(무대 8종 × 망 2벌) · (b) 하루 Y01 3벌 ·
(e) **외부트럭 22대 하루** 3벌(사건·해시·비용·KPI·오더 **원장 세 칸 A·B·O**·**결정마다 블록 요약**) ·
(f) 망 B=3 vmap == 낱개. **⚠️ 이 층만 비트 일치 불가**(torch float32·MKL GEMM·libm tanh 대 XLA float64)이고
기준 "결정(argmax)이 같다" 는 *측정된 확률*이다 — 망 2,600벌 × 85.28만 결정에서 **뒤집힘 4건 = 4.7e-06/결정**,
최악 여유 **0.17배**(`scripts/v6/probe_net_flip_rate.py --nets 2600`) ⇒ 30일 대조가 ≈69% 확률로 해시 불일치다
→ **조각 8 은 '첫 갈린 결정까지의 접두사 일치'** 로 판정한다. 성능: 망 순전파는 하루의 0.3~1.2%뿐이고 정책
비용의 **90%** 가 `joint_mask` 의 `dry_run` · vmap 은 CPU 에서 세계당 1.63배 손해(GPU 재측정 필요).

### 알려진 한계

- DEFER wake 는 단위 시험만 — 정책 반환에 `defer_until` 이 없다. 학습 경로는 `LEGACY_DEFAULT`(wait_mode
  ='WAIT')라 **구조상 안 밟힌다** → 후보 설정을 넓히는 조각의 몫 (조각 7 배정이었던 것을 2026-09-26 정정).
- `CandidateGenerator` 의 `vessel_prep`·`block_pre_rehandle` 미이식 (`policy_config` 플래그가 아니라 생성 인자).
  `vessel_prep=True` 면 특징 칸 12·13 이 갈리므로 `v5feat.cand_rows(vessel_prep=True)` 는 **크게 실패한다**.
- 후보 설정은 `LEGACY_DEFAULT` 한 경로만 (safety_only·bound_repo 등 `policy_config` 플래그 미이식).
- 탈출 `delayed` 모드 · `yard_handover_cap`(opt-in) · `vessel_cost.py`(정책 측 surrogate) 미이식.
- `stage/cargo_runtime.CargoTerminal`(전역 큐 4단 키·원격 본선 인계·끝의 재고 등식)·`cargo_moves` 미이식.
- `V3Announcer.resolve_entry` 훅·SKIP 사유 CONTAINER_ID_CHANGED 미이식 (NO_TARGET 은 조각 8 이 재현).
- `gpu/cands3.py` 의 이름 순위가 행 번호라 **이송으로 여분 행에 앉은 트럭**은 v5 `sorted(job_id)` 와
  어긋날 수 있다(동점이 생겨야 갈린다 — 사다리 ②·③에서는 아직 안 밟혔다).
- `gpu/ledger.py`(TruckLedger)는 조정자 경로 밖의 참조 구현이다 — 원장 정본은 `TerminalLedgerArrays`.
- 정책 비용의 낭비 셋: ① 순차 조건부의 결정당 K²·I 계획(`dry_run`; 90%) ② `features` 가 K 줄을 만들고
  한 줄만 씀(13~31%) ③ 안 물은 크레인까지 K 단계(30%). "(K,M) 점수 미리 계산" 은 **원리상 불가**.
- 시험은 jax 없는 파이썬에서 `importorskip` 으로 **조용히 건너뛴다** — 통과 수를 반드시 확인한다.

### 조각 8 — 학습 루프 · 30일 무대 · 손익분기 (2026-09-27 · 수정 단계 반영)

**상세는 `README-piece8.md`** (층별 대조 수치 · 남은 구멍 · 손익분기 두 경로). 요약: (0) 기록 채널을
켜도 궤적 **잎 175개 비트 일치** · (1) 배치 테이프 == 낱개 `select_record` · (A) **갱신 산술** 정수 `==`·
KL 6.2e-09 · (S) **접합부**(배열이 모은 구간 → 갱신)가 GAE까지 비트 일치 · (B)(C) **갱신 0회**에서
경계 Φ·**경계별** 계수기 `==`(경계 121·814) · (D) ★**갱신 4회 닫힌 고리**에서 Φ 상대오차 0.0·결정
52건 낱개 갈림 0 · (E) 추첨 수집의 분포·재현성 · (F) 배열 단독 2,881 경계 완주(날 경계 넘김).
⚠️ **(D) 는 v5 쪽 시장 결정을 버퍼에서 떼어낸 대조 전용 무대다**(v5 는 시장에도 정책을 물어 학습 표본의
44.9% 를 거기서 얻는다) → 시장 이식 전에는 "30일 학습 재현" 을 주장하지 않는다. **남은 구멍**: 시장 ·
**고정 화물 갈래 = 연구선의 유일한 경로** · 증거 배선(체크포인트·장부·완주 게이트·도장) · 성형 보상.
★**손익분기 답**: **① 굴리기 전용** 표에서 교차점 B* 는 **존재하지 않는다** — 포화 구간의 한계비용이
2.6~3.8 s/세계·하루인데 v5×24 는 1.294 라 **점근선 자체가 v5 위**다(최고 4.338 = 3.4배 · 다른 시드 보정
4.1배). **② ★그 표는 학습 경로가 아니다** — 학습 경로는 경계마다 호스트로 돌아와 세계를 쌓을 수 없고,
같은 무대에서 배열이 v5 단독의 **GPU 21.7배 · CPU 70.6배** 느리다(경계당 GPU 226ms · CPU 1.178s 대
v5 10.4~16.7ms · 21블록·부하 30 · 두 길이의 차 → 30일이면 GPU 2.7시간·CPU 14.1시간 대 v5 7.5분).
시장이 없어 연구 팔과는 비교 불가.
증거·표: `outputs/reports/yr327_v6_port/breakeven.{json,txt}` (`scripts/v6/bench_breakeven.py`).

### 환경 · 돌려 보기 (2026-09-27 — WSL 복구됨)

    # GPU (WSL Ubuntu · RTX 5090 · jax 0.11.2+cuda12 · venv ~/.venvs/yard-rl) — CPU 고정은 JAX_PLATFORMS=cpu
    PYTHONPATH=src:tests/v6 XLA_PYTHON_CLIENT_PREALLOCATE=false ~/.venvs/yard-rl/bin/python -m pytest … -q
    # Windows (CPU 만 · 권장 · WSL 불필요) — PYTHONPATH 구분자 `;` · 한글 경로라 PYTHONIOENCODING=utf-8
    scripts/v6/run_tests_windows.sh [--ladder]        # 사다리까지 약 75분 + 사다리
    scripts/v6/verify_chunked.sh --fresh              # WSL 세션이 짧게 끊기는 환경
    bash scripts/v6/bench_gpu_chain.sh                # 손익분기·학습경로 (한 번에 하나만 — 점유에 흔들린다)
    PYTHONPATH=src python scripts/v6/dump_ground_truth.py --mode block --policy sf_spt --info-level PRE_ADVICE

⚠️ `wsl --terminate` / `wsl --shutdown` 을 부르지 마라 — 여러 담당이 동시에 돌 때 서로의 세션을 죽인다.
⚠️ 시험은 jax 없는 파이썬에서 `importorskip` 으로 **조용히 건너뛴다** — 통과 수를 반드시 확인한다.
