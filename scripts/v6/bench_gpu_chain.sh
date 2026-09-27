#!/bin/bash
# 조각 8 수정 단계 — WSL GPU 에서 잴 것들을 **순서대로** 돌린다 ([[YR-327]]).
#
# 왜 사슬인가: 손익분기·학습경로 측정은 기계 점유에 크게 흔들린다 (같은 값이 다른 작업과 겹치면
#   1.5~1.8배가 된다 — 2026-09-27 실측). 그래서 한 번에 하나만 돌리고, 사람이 다른 무거운 작업을
#   같이 돌리지 않는다. 각 조각은 스크립트가 쓴 JSON 을 그대로 증거로 남긴다 (전사 금지).
#
# 쓰기 (Git Bash 에서):
#     bash scripts/v6/bench_gpu_chain.sh            # 학습 경로 + distinct B=64
#     bash scripts/v6/bench_gpu_chain.sh trainpath  # 학습 경로만
set -u
PROJ="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
PY="\$HOME/.venvs/yard-rl/bin/python"
ENVV="PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false"
OUT="outputs/v6/verify"
WHAT="${1:-all}"

run() {  # run <로그이름> <파이썬 인자…>
  local name="$1"; shift
  echo "■ $name 시작 $(date +%T)"
  wsl.exe -e bash -lc "cd '$PROJ' && $ENVV $PY $*" 2>&1 | tee "$OUT/$name.log" | tail -6
  echo "■ $name 끝 $(date +%T)"
}

if [ "$WHAT" = "all" ] || [ "$WHAT" = "trainpath" ]; then
  # ★학습 경로 (판정 수치) — 경계 100·300 의 차로 정상상태를 뽑는다. v5 도 같은 경로로 나란히.
  run trainpath_gpu "scripts/v6/bench_breakeven.py --which trainpath --boundaries 100,300 \
      --load 30 --ckpt outputs/v5/yr302-final-train/policy.pt --tag gpu"
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "distinct" ]; then
  # 서로 다른 시드 B=64 — 굴리기 표의 `same`(복사본 = 상한) 보정 계수를 큰 B 에서 확인한다
  run breakeven_dist64 "scripts/v6/bench_breakeven.py --which array --worlds 64 --mode distinct \
      --epochs 0 --reps 1 --tag gpu_dist64"
fi
echo "■ 사슬 끝 $(date +%T) — 로그: $OUT/"
