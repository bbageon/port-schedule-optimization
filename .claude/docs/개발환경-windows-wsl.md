# 개발환경 — Windows + WSL 2원 구성 (torch 차단 대응)

> 왜 환경이 둘인가에 대한 답. 2026-07-16 확정. 관련: YR-045(정정판 재실험) 사전조건.

## 1. 결론 (쉬운 말)

**Windows에서 torch는 못 돌린다. 그래서 torch만 WSL에서 돌린다.**

Windows 11의 **스마트 앱 컨트롤**(Smart App Control — 서명 없는 프로그램을 차단하는 보안 기능)이
켜져 있고, PyTorch가 배포하는 Windows용 DLL은 서명이 없다. 정책이 로드 자체를 막으므로 Python을
바꾸든 재설치하든 해결되지 않는다. 리눅스 바이너리는 이 정책 대상이 아니라 WSL에서는 정상 동작한다.

| 무엇을 | 어디서 | 왜 |
|---|---|---|
| 순수 파이썬 테스트 (대부분) | Windows `.venv` | 빠르고 IDE와 붙음 |
| torch 의존 8파일 · YR-045 학습 | WSL `~/.venvs/yard-rl` | Windows에서 차단됨 |
| plotly·streamlit UI 테스트 | Windows `.venv` | WSL엔 미설치 |

## 2. 증상 (이게 보이면 여기를 읽어라)

```
ImportError: DLL load failed while importing _ctypes: 애플리케이션 제어 정책에서 이 파일을 차단했습니다.
OSError: [WinError 4551] ... Error loading "...\torch\lib\torch_python.dll"
```

## 3. 진단 근거

정책 상태 — `1` = 강제 모드:
```powershell
Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" | Select VerifiedAndReputablePolicyState
```

서명 여부가 통과/차단을 정확히 가른다:

| 파일 | 서명 | 결과 |
|---|---|---|
| uv 관리 Python 3.12 `_ctypes.pyd` | NotSigned | ❌ 차단 |
| python.org 3.12 `_ctypes.pyd` | Python Software Foundation, Valid | ✅ 통과 |
| torch `torch_cpu.dll`·`torch_python.dll`·`c10.dll` 등 7/9 | NotSigned | ❌ 차단 |

→ **Python 교체는 `ctypes`만 고친다. torch DLL은 여전히 막힌다** (2단 벽).

## 4. 하지 않은 선택 — 스마트 앱 컨트롤 끄기

torch가 네이티브로 돌게 하는 유일한 Windows 방법이지만 **한 번 끄면 Windows 재설치 전까지 다시 켤 수
없다**(Microsoft 설계상 단방향). torch 하나 때문에 서명 없는 모든 프로그램에 대한 보호를 영구히
포기하는 거래 — 채택하지 않았다. 사용자 결정(2026-07-16).

## 5. 현재 구성

**Windows** — python.org 3.12.10 (서명됨, winget `Python.Python.3.12`) 기반 `.venv`.
uv 관리 Python으로 만든 옛 `.venv`는 `ctypes` 자체가 죽어 폐기.
```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --ignore=tests/integrated/test_dgt_generalization.py `
  --ignore=tests/integrated/test_dqn_stage_b.py `
  --ignore=tests/integrated/test_qnet_stage_a.py `
  --ignore=tests/unit/test_oracle_pattern.py `
  --ignore=tests/unit/test_residual_delta_net.py `
  --ignore=tests/unit/test_residual_delta_stable.py `
  --ignore=tests/unit/test_residual_setfeat.py `
  --ignore=tests/unit/test_setfeat_selection.py
```

**WSL** (Ubuntu 24.04 · Python 3.12.3 · 24코어 · 30GB) — `~/.venvs/yard-rl`, torch 2.13.0+cpu.
`sudo` 비밀번호가 필요해 `python3-venv` 대신 **uv**로 구성(사용자 홈에만 설치, 리눅스라 서명 무관).
**소스는 `/mnt/c` 원위치를 `PYTHONPATH`로 붙인다** — 프로젝트 폴더에 쓰지 않아 Windows 환경과 무간섭.
```bash
cd /mnt/c/Users/GeonU/Desktop/side-project/port-schedule-optimization
PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 ~/.venvs/yard-rl/bin/python -m pytest -q \
  --deselect tests/test_recorder.py::test_streamlit_app_renders
```
→ 2026-07-16 기준 **315 passed, 3 skipped** (2분 53초).

## 6. 남은 스킵과 열린 선택

- `test_dqn_stage_b.py:164` — CUDA parity. **RTX A400이 실제로 있으나** CPU 휠을 설치해 건너뛴다.
  **CPU로 충분하다 — 실측으로 확인(2026-07-16)**: 학습 시간의 약 **4%만 torch**이고 나머지는
  순수 파이썬 시뮬레이터라, GPU를 붙여도 상한이 **1.04배**다. 망도 작다(은닉 128·배치 64).
  게다가 `dqn_learner.py:82`가 재현성을 위해 `torch.set_num_threads(1)`로 고정해 둔 상태라
  CUDA 전환은 재현성 질문을 새로 연다. **실제 병목은 GPU가 아니라 `engine.py:287`의
  전체 스택 deepcopy(71.4%)** → backlog **YR-047**(상한 3.57배).
  판단: YR-045는 CPU로 진행. 참고 실측 — 학습 3.26s/에피소드, 3 variant × 500 ep ≈ **81분**.
- `test_recorder.py` plotly 2건 — WSL 미설치. Windows 쪽에서 실행되므로 커버됨.

## 8. 신규 머신 (rjsdn, 2026-07-17 구성 완료 — YR-053)

위 §1~7 은 이전 머신(GeonU) 기준 기록이다. 새 머신(`c:\Users\rjsdn\...`, i7-14700K 20코어)은
**스마트 앱 컨트롤이 꺼져 있어(`VerifiedAndReputablePolicyState=0`) WSL 이 필요 없다** —
서명 없는 torch DLL 차단이 없으므로 **Windows 단일 `.venv` 로 torch 포함 전부** 돌린다.

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe -m pytest -q     # torch·UI 포함 전체, 제외 없음
```

- venv: Anaconda `py -3.12` (3.12.4) 기반. pyyaml·pytest·streamlit·plotly·**torch 2.8.0**.
- **torch 는 2.8.0 고정** — 최신 2.13.0 Windows 휠은 이 머신에서 `c10.dll` 초기화 실패
  (WinError **1114** — 구 머신의 정책 차단 4551 과 다른 원인, base/venv 동일 재현)라 한 단계
  안정 휠로 회피했다. 2.8.0 은 pyproject 요건(`torch>=2.2`)을 충족한다.
- 이전에 임시로 쓰던 Anaconda base 직행(297 passed·streamlit 구버전 1건 실패)은 폐기 —
  venv 의 신버전 streamlit 으로 UI 테스트까지 통과한다.

## 9. 세 번째 환경 (geonu · `Desktop\port_reinforcement` 클론, 2026-07-18 구성)

§5 의 GeonU 머신과 같은 사용자명이나 **별도 클론·별도 환경**이다 (side-project 클론·`.venv` 없음).
- **Windows**: `pythoncore-3.14-64` (`C:\Users\geonu\AppData\Local\Python\`) — PATH 미등록이라 전체
  경로로 호출. 순수 파이썬 스위트(§5 의 8파일 ignore) **297 passed**.
- **WSL**: `~/.venvs/yard-rl` 을 uv 로 재구성 (Python 3.12 standalone + torch 2.13.0+cpu).
  전체 스위트 **348 passed / 1 failed** — 실패 1건은 `test_residual_delta_net.py::
  test_update_regresses_toward_residual_target` (닫힌 단일야드 트랙의 수렴 허용오차 테스트,
  −2.5±0.15 기대에 −2.22). **변경 전 코드에서도 동일 실패 (stash 왕복 확인) — 회귀 아님**,
  CPU 부동소수점 경로 차이로 인한 머신 민감성. 후속: YR-058.

## 7. 이 과정에서 드러난 실제 버그

환경이 막혀 8파일이 몇 달간 미실행이었고, 그 사이 **YR-043이 뒤집은 계약을 옛 테스트가 계속 붙들고
있었다** — `test_actionable_excludes_wait_and_scores_follow`가 "WAIT는 행동집합에서 제외"를 검사.
YR-043이 WAIT를 실제 학습 행동으로 복구(매핑 §4 — 배제 근거 전제 소멸)했으므로 테스트가 낡은 것.
새 규약으로 갱신(`..._includes_wait_...`). **미실행 테스트는 통과가 아니라 미지(unknown)다** —
YR-043·YR-044 evidence의 "8파일 미실행" 표기가 이 위험을 정확히 가리키고 있었다.
