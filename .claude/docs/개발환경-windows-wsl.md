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

- `test_dqn_stage_b.py:164` — CUDA parity. **RTX A400이 실제로 있으나** CPU 휠을 설치해
  건너뛴다. Windows 기존 환경(CPU)과 조건을 맞춰 이전 결과와 비교 가능하게 하려는 의도.
  YR-045에서 학습이 느리면 CUDA 휠로 바꾸는 선택지가 열려 있다 (수치 재현성 영향 검토 필요).
- `test_recorder.py` plotly 2건 — WSL 미설치. Windows 쪽에서 실행되므로 커버됨.

## 7. 이 과정에서 드러난 실제 버그

환경이 막혀 8파일이 몇 달간 미실행이었고, 그 사이 **YR-043이 뒤집은 계약을 옛 테스트가 계속 붙들고
있었다** — `test_actionable_excludes_wait_and_scores_follow`가 "WAIT는 행동집합에서 제외"를 검사.
YR-043이 WAIT를 실제 학습 행동으로 복구(매핑 §4 — 배제 근거 전제 소멸)했으므로 테스트가 낡은 것.
새 규약으로 갱신(`..._includes_wait_...`). **미실행 테스트는 통과가 아니라 미지(unknown)다** —
YR-043·YR-044 evidence의 "8파일 미실행" 표기가 이 위험을 정확히 가리키고 있었다.
