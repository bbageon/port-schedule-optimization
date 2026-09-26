#!/bin/bash
# v6 배열 세계 시험을 **Windows 파이썬**으로 돌린다 ([[YR-327]]).
#
# 왜 이게 있나 — 2026-09-26 이 기계의 WSL 배포판(Ubuntu)이 먼저 부팅 ~88초 뒤 저절로 죽더니
#   (`scripts/v6/verify_chunked.sh` 가 그 대응이다) 결국 **프로세스 기동 자체가 실패**하게 됐다
#   (`Wsl/Service/E_UNEXPECTED` · 배포판은 Running 으로 보이는데 셸이 안 뜬다 · `wsl --shutdown` 만이
#   알려진 복구 수단인데 2개월째 가동 중인 Docker 컨테이너 3개를 멈춘다).
#   그때 발견: **JAX 0.11.2(WSL venv 와 같은 버전)가 Windows 파이썬에 그대로 설치된다.**
#   v5 정본(world/·stage/)은 애초에 jax 를 안 쓰므로 Windows 에서 그냥 돈다 →
#   **CPU x64 동등성 시험 전부를 WSL 없이, 시간 제약 없이** 돌릴 수 있다.
#
# 한계: **GPU 는 안 된다** (JAX 의 CUDA 는 리눅스만). GPU 비트 일치·손익분기 측정은 WSL 이 필요하다.
#
# 준비 (한 번):
#     python -m venv --system-site-packages .venv-jax      # 아나콘다 numpy/scipy/pytest 재사용
#     .venv-jax/Scripts/python.exe -m pip install "jax[cpu]==0.11.2"
#
# 사용 (Git Bash):
#     scripts/v6/run_tests_windows.sh                 # 사다리(터미널 30·300) 빼고 전부
#     scripts/v6/run_tests_windows.sh --ladder        # 사다리까지 (느리다 — 1,441 에폭 × 2)
#     scripts/v6/run_tests_windows.sh tests/v6/test_gpu_y01.py   # 파일 지정
#
# ⚠️ Windows 는 PYTHONPATH 구분자가 `;` 다 (`:` 이 아니다). 경로에 한글이 있어 PYTHONIOENCODING 을 준다.
set -u
cd "$(dirname "$0")/../.."
PY=".venv-jax/Scripts/python.exe"
[ -x "$PY" ] || { echo "★ $PY 가 없다 — 위 '준비' 를 먼저 하라"; exit 1; }
OUT="outputs/v6/verify"; mkdir -p "$OUT"

LADDER=0; FILES=()
for a in "$@"; do case "$a" in --ladder) LADDER=1;; *) FILES+=("$a");; esac; done
if [ ${#FILES[@]} -eq 0 ]; then
  # 사다리(test_gpu_terminal_equiv)는 1,441 에폭 × 2 라 따로 — 아래 --ladder
  FILES=(tests/v6/test_gpu_core.py tests/v6/test_gpu_state.py tests/v6/test_gpu_travel.py
         tests/v6/test_gpu_stack.py tests/v6/test_gpu_reserve.py tests/v6/test_gpu_plan.py
         tests/v6/test_gpu_convert.py tests/v6/test_gpu_phi.py tests/v6/test_gpu_exact.py
         tests/v6/test_gpu_escape.py tests/v6/test_gpu_dispatch.py tests/v6/test_gpu_wake.py
         tests/v6/test_gpu_cands3.py tests/v6/test_gpu_vessel.py tests/v6/test_gpu_engine_equiv.py
         tests/v6/test_gpu_y01.py tests/v6/test_gpu_admission.py tests/v6/test_gpu_ledger.py
         tests/v6/test_gpu_transfer_txn.py tests/v6/test_gpu_host_terminal.py)
fi

echo "■ v6 시험 (Windows · CPU x64) $(date +%T)"
PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 "$PY" -m pytest "${FILES[@]}" \
  -q -p no:cacheprovider -s 2>&1 | tee "$OUT/windows_tests.out" | tail -25

if [ $LADDER -eq 1 ]; then
  echo; echo "■ 사다리 — 터미널 30·300 (21블록 · 1,441 에폭) $(date +%T)"
  #: 세션 예산을 크게 — Windows 는 WSL 의 88초 한도가 없으니 한 번에 완주한다.
  #: 상태 디렉터리를 새로 두면 에폭 0 부터 다시 돈다(독립 재검증). 이어 돌리려면 같은 디렉터리로.
  TERMINAL_STATE_DIR="$OUT/terminal_equiv_win" TERMINAL_SESSION_S=7200 \
  PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 "$PY" -m pytest tests/v6/test_gpu_terminal_equiv.py \
    -q -p no:cacheprovider -s 2>&1 | tee "$OUT/windows_ladder.out" | tail -25
fi
echo "■ 끝 $(date +%T) — 로그: $OUT/windows_tests.out"
