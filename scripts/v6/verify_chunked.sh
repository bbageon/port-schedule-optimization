#!/bin/bash
# tests/v6/test_gpu_*.py 를 **짧은 WSL 세션 조각**으로 나눠 전부 돌린다 ([[YR-327]]).
#
# 왜: 이 기계의 WSL 배포판이 2026-09-25 22:20 부터 부팅 ~88초 뒤 저절로 종료된다. 전체 시험은
#     15분이 넘어 한 세션에 못 담는다. 조각마다 배포판을 새로 띄우고(≤85초) 결과를 모은다.
#     정상 환경이면 그냥 `pytest tests/v6/test_gpu_*.py` 를 쓰면 된다.
#
# 사용 (Windows Git Bash 에서):
#     scripts/v6/verify_chunked.sh [--fresh] [--report] [파일...]
#       --fresh   누적 출력·REPORT 덤프를 지우고 시작 (첫 실행)
#       --report  파일 대신 전체 집계 시험(REPORT 병합)만 실행 — 마지막에 한 번
#       파일 생략 = 전부
#     결과: outputs/v6/verify/verify_chunked.out (조각별 요약 + 총계)
#
# 동작: 파일마다 node id 를 모아 C 개씩 한 세션에 돌린다. 세션이 죽어 요약이 없으면 반으로 쪼개
#       다시(1건까지). 전체 집계 시험(zz_report·ties_were_exercised·escape_paths 등)은 조각별 REPORT 를
#       병합한 뒤 `--report` 로 돌린다 (tests/v6/report_dump.py 플러그인).
set -u
REPO_WIN="$(cd "$(dirname "$0")/../.." && pwd)"                      # /c/Users/.../강화학습-판매
REPO="/mnt/c${REPO_WIN#/c}"                                          # WSL 경로
WORK_WIN="$REPO_WIN/outputs/v6/verify"; WORK="$REPO/outputs/v6/verify"; mkdir -p "$WORK_WIN"
PY="~/.venvs/yard-rl/bin/python"
OUT="$WORK_WIN/verify_chunked.out"
PASS=0; FAIL=0; DIED=0; NCHUNK=0
REPORT_TESTS='zz_report or escape_paths or replan_actually or k3_and_stair or ties_were_exercised'
SKIP_TESTS='python_loop'     # eager 27스텝 ≈80초 — 88초 창에선 완주 불가 (정상 환경에선 돈다)
FRESH=0; REPORT=0; FILES=()
for a in "$@"; do case "$a" in --fresh) FRESH=1;; --report) REPORT=1;; *) FILES+=("$a");; esac; done
[ ${#FILES[@]} -eq 0 ] && FILES=(test_gpu_core.py test_gpu_state.py test_gpu_travel.py test_gpu_stack.py test_gpu_reserve.py test_gpu_plan.py test_gpu_convert.py test_gpu_phi.py test_gpu_exact.py test_gpu_escape.py test_gpu_dispatch.py test_gpu_wake.py test_gpu_cands3.py test_gpu_vessel.py test_gpu_engine_equiv.py test_gpu_y01.py)
if [ $FRESH -eq 1 ]; then : > "$OUT"; rm -f "$WORK_WIN"/report_*.json "$WORK_WIN"/merged.json "$WORK_WIN"/chunk_*.log; fi

fresh() {  # 배포판을 새로 띄우고 응답할 때까지 기다린다
  # ★`wsl --terminate` 를 부르지 않는다 — 여러 담당이 동시에 돌 때 **서로의 세션을 죽인다**
  #   (지난 워크플로에서 4초마다 서로를 끊었다). 이 기계의 Ubuntu 는 저절로 죽고 저절로 다시 뜨므로,
  #   같은 명령을 백오프로 다시 부르면 새 배포판이 열린다.
  local n=0
  until wsl.exe -e bash -lc "echo alive" 2>/dev/null | tr -d ' \0\r' | grep -q alive; do n=$((n+1)); [ $n -gt 60 ] && return 1; sleep $(( n < 10 ? 3 : 7 )); done; return 0; }
collect() {  # $1 = 파일 → node id 목록 (집계·건너뛰기 시험 제외)
  fresh || return 1
  wsl.exe -e bash -lc "cd '$REPO' && PYTHONPATH=src JAX_PLATFORMS=cpu $PY -m pytest --co -q '$1' -p no:cacheprovider -k 'not ($REPORT_TESTS or $SKIP_TESTS)' 2>/dev/null | grep '::'" 2>/dev/null | tr -d '\0\r'; }
run_chunk() {  # $1 label, $2.. node ids → "<초>s <요약>" 또는 "DIED ..."
  local label="$1"; shift; local ids=""; local id
  for id in "$@"; do ids+=" '$id'"; done   # ★node id 에 공백·괄호가 있어도(파라미터 id) 하나로 넘긴다
  fresh || { echo "DIED(boot)"; return; }; local t0=$(date +%s)
  wsl.exe -e bash -lc "cd '$REPO' && PYTHONPATH=src:tests/v6 REPORT_TAG=$label REPORT_DIR='$WORK' JAX_PLATFORMS=cpu $PY -u -m pytest $ids -q -p no:cacheprovider -p report_dump > '$WORK/chunk_$label.log' 2>&1; echo EXIT=\$? >> '$WORK/chunk_$label.log'" > /dev/null 2>&1
  local t1=$(date +%s); local log; log=$(tr -d '\0\r' < "$WORK_WIN/chunk_$label.log" 2>/dev/null)
  if echo "$log" | grep -qE '^EXIT='; then echo "$((t1-t0))s $(echo "$log" | grep -E '[0-9]+ passed|[0-9]+ failed|[0-9]+ error|no tests ran' | tail -1)"; else echo "DIED $((t1-t0))s"; fi; }
tally() {  # 요약 줄의 passed/failed/error 를 더한다 (bc 없이)
  local p f e; p=$(echo "$1" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1); PASS=$((PASS + ${p:-0}))
  f=$(echo "$1" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1); e=$(echo "$1" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' | head -1); FAIL=$((FAIL + ${f:-0} + ${e:-0})); }
run_ids() {  # $1 label, $2 조각 크기, $3.. ids — 죽으면 반으로 쪼갠다
  local label="$1"; local size="$2"; shift 2; local ids=("$@"); local n=${#ids[@]}; local i=0; local c=0
  while [ $i -lt $n ]; do local batch=("${ids[@]:$i:$size}"); c=$((c+1)); NCHUNK=$((NCHUNK+1))
    local res; res=$(run_chunk "${label}_$c" "${batch[@]}")
    if [[ "$res" == DIED* ]]; then
      if [ ${#batch[@]} -gt 1 ]; then local half=$(( (${#batch[@]} + 1) / 2 )); echo "  [$label #$c] $res → ${#batch[@]}건을 $half 씩 쪼갬" >> "$OUT"; run_ids "${label}_${c}r" "$half" "${batch[@]}"
      else DIED=$((DIED+1)); echo "  [$label #$c] $res — 1건도 못 마침: ${batch[0]}" >> "$OUT"; fi
    else tally "$res"; echo "  [$label #$c] ${#batch[@]}건 $res" >> "$OUT"; fi
    i=$((i+size)); done; }

#: 파일별 조각 크기 — 2026-09-26 실측(CPU x64)으로 조각당 ≤50초가 되게
declare -A SIZE=( [test_gpu_core.py]=9 [test_gpu_state.py]=12 [test_gpu_travel.py]=10 [test_gpu_stack.py]=6
  [test_gpu_reserve.py]=4 [test_gpu_plan.py]=4 [test_gpu_convert.py]=8 [test_gpu_phi.py]=8 [test_gpu_exact.py]=30
  [test_gpu_escape.py]=16 [test_gpu_dispatch.py]=5 [test_gpu_wake.py]=40 [test_gpu_cands3.py]=6 [test_gpu_vessel.py]=20
  [test_gpu_engine_equiv.py]=3 [test_gpu_y01.py]=2 )

if [ $REPORT -eq 0 ]; then
  echo "■ 시작 $(date +%T): ${FILES[*]}" >> "$OUT"
  for f in "${FILES[@]}"; do
    mapfile -t IDS < <(collect "tests/v6/$f")
    echo "▶ $f: ${#IDS[@]}건 (조각 ${SIZE[$f]:-4})" >> "$OUT"
    [ ${#IDS[@]} -eq 0 ] && { echo "  수집 0건 — 건너뜀" >> "$OUT"; continue; }
    run_ids "${f%.py}" "${SIZE[$f]:-4}" "${IDS[@]}"
  done
else
  echo "▶ 집계 시험 (REPORT 병합) $(date +%T)" >> "$OUT"
  python - "$WORK_WIN" >> "$OUT" 2>&1 <<'PY'
import glob, json, os, sys
d = sys.argv[1]; merged = {}
for p in sorted(glob.glob(os.path.join(d, "report_*.json"))):
    for mod, rep in json.load(open(p, encoding="utf-8")).items():
        merged.setdefault(mod.split(".")[-1], {}).update(rep)
json.dump(merged, open(os.path.join(d, "merged.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("  병합:", {k: len(v) for k, v in merged.items()})
PY
  for f in test_gpu_plan.py test_gpu_exact.py test_gpu_escape.py test_gpu_dispatch.py test_gpu_wake.py test_gpu_cands3.py test_gpu_vessel.py test_gpu_engine_equiv.py test_gpu_y01.py; do
    fresh || continue; NCHUNK=$((NCHUNK+1))
    wsl.exe -e bash -lc "cd '$REPO' && PYTHONPATH=src:tests/v6 REPORT_LOAD='$WORK/merged.json' JAX_PLATFORMS=cpu $PY -u -m pytest 'tests/v6/$f' -q -p no:cacheprovider -p report_dump -k '$REPORT_TESTS' > '$WORK/chunk_rep_$f.log' 2>&1; echo EXIT=\$? >> '$WORK/chunk_rep_$f.log'" > /dev/null 2>&1
    log=$(tr -d '\0\r' < "$WORK_WIN/chunk_rep_$f.log" 2>/dev/null)
    if echo "$log" | grep -qE '^EXIT='; then line=$(echo "$log" | grep -E '[0-9]+ passed|[0-9]+ failed|[0-9]+ error|no tests ran' | tail -1); tally "$line"; echo "  [집계 $f] $line" >> "$OUT"
    else DIED=$((DIED+1)); echo "  [집계 $f] DIED" >> "$OUT"; fi
  done
fi
echo "■ 끝 $(date +%T) — 조각 $NCHUNK · 통과 $PASS · 실패 $FAIL · 완주 못한 조각 $DIED · 건너뜀($SKIP_TESTS)" >> "$OUT"
tail -30 "$OUT"
