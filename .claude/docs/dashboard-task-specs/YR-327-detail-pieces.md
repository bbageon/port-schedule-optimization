# YR-327 상세 — 조각별 구현·검증 기록

상위: [YR-327](YR-327-v6-gpu-array-world.md). 여기는 조각 1~6 의 **구현 워크플로·반박 검증·수정** 기록이다.
증거 전문은 `outputs/reports/yr327_v6_port/`.

## 3단계 — 조각 1 · 5 · 2 · 3+4 (전부 v5 와 **`==`**, 기대값 손기입 없음)

| 조각 | 워크플로 | 시험 | 핵심 대조 · 발견 |
|---|---|---|---|
| 1 | `wf_bd785f09` · 11 · 113분 | 130 | 무대 12종 사건 로그~실수 전 항목 · **FMA 플래그로 못 막음** |
| 5 | `wf_a4bd510b` · 3 · 29분 | 16 | **Python 3.12 `sum()` 보정합** · **XLA 상수 재결합** |
| 2 | `wf_d88b3e99` · 7 · 4시간 22분 | 68+89+44 | `_try_escape` 1,269회 가로채기 · lockstep 27무대 · **GPU 스텝 90% 가 비트 일치용 직렬 scan** |
| **3+4** | `wf_24a3d958` · 8 · 146분 | 37+18+16+26+**103**+**6** | 아래 |

### ★조각 3+4 — PRE_ADVICE · 본선/이송 · **Y01 재현** (이번)

    모듈 ∥  wake.py(ETA/DEFER wake — v5 호출 303회 가로채기, 무대 18) · cands3.py(SERVE+PRE+REPO+WAIT 후보
            — 결정 356건 후보 집합·계획 ==) · vessel.py(본선 처리기·이송 링버퍼·양하 사전식 순위 — 사건마다 가로채기)
            · exact.sum_python(CPython Neumaier 그대로, 60항×50회 비트 ==)
    통합    step 에 국면 W/A · 확장 후보·공동 결정(resolve_central: SF_SPT 등 규칙 resolver) · 본선 처리기 4종 ·
            호스트가 본선이 만들 job 칸 예약 · sum_python 을 검열 노출·레인 평균·불균형에 적용
    검증    세 렌즈 전부 통과(높음·치명 0, 보통 9) → 수정 생략

**★Y01 정답 재현**: 원본 궤적(sf_spt/PRE_ADVICE) 사건 1,318건 전부 · 해시 `6668fa4902c4efe4` · 결정 237 ·
REPOSITION 3회 · 재조작 259 · 비용 13항 비트(sts_wait 11564.934482758661 …) · KPI 10항 == . 정책·정보수준
5조합 전부. 한 조합 CPU ≈6초. **직접 재확인(2026-09-26)**: GPU 원본 궤적 통과(38초) · CPU Y01 6건 ·
wake+exact+vessel 81건 · cands3 18건. 통합자의 조각 실행: engine_equiv 103 · 회귀 전부 통과.

**보통 발견(후속)**: DEFER wake 가 통합 엔진에 미연결(정책 반환에 `defer_until` 없음 → 조각 7) · 후보 설정
`LEGACY_DEFAULT` 한 경로만 · `resolve_central` 쌍 scan 이 vmap 아래 전 갈래 실행 · 본선 clearout 합 순서
(정렬 vs 삽입) · 조각 3·4 GPU 는 Y01 만 확인.

### ⚠️ 환경 — WSL 이 부팅 88초 뒤 죽는다 (2026-09-25 22:20 ~)

★`wsl --terminate` 를 **부르지 않는다** — 여러 담당이 동시에 돌면 서로의 세션을 죽인다(실제로 4초마다 서로를
끊은 일이 있다). `verify_chunked.sh` 의 `fresh()` 에서 terminate 를 빼고 '같은 명령 재시도 + 백오프' 로 바꿨다.

배포판 내부·Windows 이벤트·호스트 프로세스·예약 작업에 원인 없음. 정석 `wsl --shutdown` 은 **2개월째
가동 중인 Docker 컨테이너 3개**를 멈추므로 **사용자 결정 대기**. `scripts/v6/verify_chunked.sh`(85초 조각·
REPORT 병합 · 공백 든 시험 id 수정) + `tests/v6/report_dump.py` 로 전체 검증.

### 조각 6 (다중블록 조정자) — 검증·수정 (2026-09-26)

**같은 답임을 확인한 것** (사다리 ①②③ 전부 재실행):

| 사다리 | 무대 | 대조한 것 |
|---|---|---|
| ① | Y01 · 20대 · lead 600 · 121 에폭 | 사건 71 전열 · 해시 · 비용 13항 · KPI · 투입 원장 141행 · 턴 20 · locked 20 · 오더 전열 · 장부 적분. **분할점 셋**(55\|66 · 66\|55 · 121)에서 저장·복원 왕복이 **dtype + float 비트** 동일 |
| ② | Y01+Y21 · 40대 · 이송 1 · 이연 1 | 원장 8항 · 재시도 거절 코드 · 사건 144(Y01 70 + Y21 74) · route 220s · **이연 원장(기사 외부대기 600.0s)** == v5 `time_sell.deferral_ledger` |
| ③ | 터미널 30 / 300 · 21블록 · 1,441 에폭 | 21/21 해시 · 사건 19,796 / 20,720 · 비용·KPI · admitted 30/300 · 턴 합 28196.313356 / 334078.47377 · **오더 전열 3,630 / 3,900행** · **투입 원장 1,471 / 1,741행** · **locked 30/300** · 시간장부 적분 3항 · 불변식(보존) |

**검증에서 나온 수정** — ①`OrderArrays.appt_s` 열 신설(v5 `Job.appointment_gate_time` = 이연 비용의 원점. 예전에는
`notice_s` 를 원점이라 잘못 적어 두었고, 그 식대로면 외부대기가 600s 대신 2,400s 가 된다) ②통지 리드를 **트럭별**
(S,) 열로(v3 `V3Announcer` 축) · `retarget`/`resolve_entry` 훅은 fail-loud 거절 ③`_sync_locks` 에 `~terminal` 블록
마스크(v5 는 ReviewEpoch 를 돌려준 블록만) ④SKIP_DUP 을 **전역 원장** 기준으로(v5 `jid in ledger.records`)
⑤`_issue` 의 예약 영구 누수 차단(새 코드 `R_NO_TXN_SLOT`) ⑥`check_invariants` 에 보존식(살아 있는 외부트럭 행 수 ==
등록 수) ⑦`load_run` 이 x64·dtype 을 확인 ⑧`lax.switch` 갈래를 static 으로(review=False 경로가 쓰지도 않는 `[R]`
갈래의 적분을 매 스텝 계산하고 있었다) ⑨`make_resolver` 캐시(환경마다 전체 재추적 방지).

**정답을 두껍게** — `dump_ground_truth.run_terminal` 이 투입 원장·locked·블록별 오더 전열·장부 적분·totals·이연
원장을 함께 덤프한다. 두 정답을 다시 만들어 기존 항목은 재귀 비교로 `wall_s` 만 달랐다(값 자체는 파이썬 3.12↔3.13
재현). ★`terminal_load30_seed9900777.json` 은 **커밋에 반드시 포함** — 없으면 load30 시험이 조용히 skip 된다.

**속도 (정정)** — 보고서에 있던 "GPU 41 · CPU 250 ms/에폭" 은 어떤 로그에도 없다. 재현된 값은 **GPU 70(load30) ·
92(load300) ms/에폭**, 보존된 CPU 최선 412. 그리고 **단일 터미널은 v5 파이썬이 더 빠르다** — 같은 load300 을 v5 가
16.4초, 배열판이 GPU 133~147초(8~9배)·CPU 576~673초(34~39배). 이득은 세계를 쌓을 때 나며, 손익분기 측정을 조각 8 의
첫 항목으로 둔다.

**아직 실행 못 한 단언 둘 (WSL 사용자 세션이 80분 넘게 `E_UNEXPECTED` 로 굳었다 — `--system` 은 응답하는데
사용자 namespace 만 죽는 알려진 증상이고, 복구 수단인 `wsl --terminate` 는 금지)**. 다음 살아 있는 세션에서
이 둘만 돌리면 된다 — 나머지는 전부 통과했다.

    pytest 'tests/v6/test_gpu_terminal_equiv.py::test_ladder2_y01_y21_transfer_and_defer_match_v5'
    pytest 'tests/v6/test_gpu_host_terminal.py::test_per_truck_lead_matches_v3announcer'

둘 다 **기대값은 v5 로 따로 확인해 두었다**: ①이연 원장 = `{'job_id':'Y01:D-00014','block':'Y01','flow':'GATE_IN',
'n_deferrals':1,'deferred_total_s':600.0,'original_appointment_s':2556.179,'actual_gate_in_s':3156.179,
'driver_outside_wait_s':600.0}` (v5 `time_sell.deferral_ledger` 직접 실행) ②트럭별 리드 무대 = 트럭 40·서로 다른
리드 35·버킷 17개가 전부 검토 시각 목록 안·한 버킷 최대 21대. 사다리 ②는 이 단언을 **붙이기 전에** 통과했고,
`appt_s` 열 값 자체는 투입 무대 9개가 v5 `Job.appointment_gate_time` 과 `==` 로 대조해 통과했다. 새로 넣은
`to_terminal_world` 의 fail-loud 검사(est == provided_eta)는 v5 로 5무대 430건을 세어 **어긋남 0** 이라 발화하지 않는다.

### 조각 6 — 독립 재검증 (2026-09-26 · Windows 파이썬)

반박 3 렌즈가 **전부 반박**했고(높음 8) 수정 단계가 **17건**을 고쳤다. 그중 하나는 실제 계산 오류다 —
v5 `Job.appointment_gate_time`(이연 비용의 원점)에 대응하는 배열 열이 **없어서** `notice_s` 를 원점으로
잘못 적어 두었고, 그 식대로면 기사 외부대기가 **600초 대신 2,400초(4배)** 로 계상된다 → `OrderArrays.appt_s` 신설.
나머지: 트럭별 통지 리드 · SKIP 6종 실제 발화 · `sync_locks` 블록 마스크 · SKIP_DUP 전역 원장 기준 ·
`_issue` 예약 영구 누수 · `check_invariants` 보존식 · `load_run` x64 확인 · 이어돌리기 **비트** 비교(분할점 3곳) ·
사다리 ③에 원장·locked·오더 전열·장부 적분 추가 · `lax.switch` 갈래 static · `make_resolver` 캐시 · 성능 주장 정정.

**★내가 직접 재확인** (WSL 이 죽어 **Windows 파이썬 + jax[cpu] 0.11.2** 로 옮겨 — 아래 환경 절):

    사다리 5건   21분 26초 · resumed=False (1,441 에폭을 처음부터)
      ① Y01·20대      사건 71 · 투입 20 · 원장 141행 · locked 20 · 턴 합 16719.491429
      ② 이송·이연     사건 144 · route 220.0s · **외부대기 600.0s** (appt_s 수정 확인)
      ③ 터미널 30     **해시 21/21** · 사건 19,796 · 오더 3,630행 · 원장 1,471행 · locked 30 · 398 ms/에폭
      ③ 터미널 300    **해시 21/21** · 사건 20,720 · 오더 3,900행 · 원장 1,741행 · locked 300 · 405 ms/에폭
    회귀 시험     462 통과 · 1 건너뜀 (28분) — 조각 1~6 전 파일

검증이 *"21블록 규모에서 투입 원장·locked·오더 전열·장부 적분이 한 번도 대조되지 않았다"* 고 지적한 넷이
이제 사다리 ③에 들어가 있고, 그것까지 내 손으로 확인했다.

**남은 위험(미해결)**: `gpu/cands3.py` 의 이름 순위가 **행 번호**라 이송으로 여분 행에 앉은 트럭은
v5 `sorted(job_id)` 와 어긋난다(동점이 생겨야 갈림 · 사다리 ②는 이송 1건, ③은 0건이라 아직 안 밟힘).
정공법은 `OrderArrays` 에 이름 순위 열. `gpu/ledger.py`(445줄)는 조정자가 안 쓰는 참조 구현.
`CargoTerminal`(30일)·`V3Announcer` 훅은 범위 밖.
