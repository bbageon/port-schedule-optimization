# RL 학습 전용 세션 운영 수칙

> **트리거**: RL 재학습을 별도 세션에서 돌릴 때 / 두 세션이 동시에 작업할 때.
> **사용자 결정 2026-08-18** — 학습이 코어를 점유하므로 레인을 나눈다.

## 1. 왜 나누나

| 제약 | 실측 |
|---|---|
| **코어** | 24개. 실험 하나가 `max_workers=24` 로 전부 쓴다 |
| **동시 실행** | 두 세션이 함께 무거운 실험을 돌리면 **둘 다 절반 속도** |
| **git** | 같은 저장소·같은 board 파일 → 커밋 충돌 |

**한 번에 무거운 실험은 한 세션만.** 나머지 세션은 CPU 를 안 쓰는 일을 한다.

## 2. 레인 (담당 분리)

| | **RL 세션** | **본 세션** |
|---|---|---|
| 작업 | YR-183(회차 확대) → YR-184(크레인 재학습) → YR-175·177(견적망) | 논문(YR-186 계열)·board·인프라·문서 |
| 코드 | `src/yard_rl/experiments/yr18[3-5]*_*.py` · 새 `yr19x_*` | 게이트·`repro.py`·문서·board |
| 산출물 | `outputs/reports/yr183_*`·`yr184_*` 등 자기 디렉터리 | `outputs/reports/yr153_research_gates/` |
| CPU | **점유 (기본)** | 안 씀 |

**RL 세션이 학습을 돌리는 동안 본 세션은 실험을 돌리지 않는다.** 반대도 같다.
꼭 필요하면 미리 합의하고 `max_workers` 를 나눈다.

## 3. 충돌 방지 수칙

### 3-1. worktree 를 나눈다 (권장)

```bash
git worktree add C:/Users/geonu/orca/workspaces/port_reinforcement/rl-lane -b rl-lane
```

폴더·브랜치가 분리되고 저장소(커밋 이력)는 공유되므로 서로의 결과를 볼 수 있다.
현재 구성도 이미 worktree 다 — `Desktop/port_reinforcement`(master)와
`orca/.../강화학습-판매`가 같은 저장소를 공유한다.

### 3-2. board 는 커밋 직전에만 만진다

```bash
git pull --rebase        # 항상 먼저
# board 편집은 자기 row 한 줄만
git add -A && git commit && git push
```

board 파일(`.claude/Dashboard/*.md`)은 두 세션이 다 건드린다. **작업 내내
열어두지 말고 커밋 직전에 최소 편집**한다. 자기 row 밖은 손대지 않는다.

### 3-3. 두 원격 참조를 모두 갱신한다

```bash
git push origin HEAD              # 작업 브랜치
git push origin HEAD:master       # master (게이트가 origin/master 를 본다)
```

## 4. 실행 환경 (WSL)

torch 는 Windows 에서 차단되므로 **WSL 에서 돌린다**. 이 worktree 는 `.git` 이
Windows 경로를 가리켜 WSL 에서 git 이 실패하므로 두 변수를 export 한다.

```bash
R='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
export GIT_DIR='/mnt/c/Users/geonu/Desktop/port_reinforcement/.git/worktrees/강화학습-판매'
export GIT_WORK_TREE="$R"
cd "$R" && source ~/.venvs/yard-rl/bin/activate && export PYTHONPATH=src
```

**⚠️ 이 상태로 `pytest` 를 돌리면 안 된다** — `GIT_DIR` 가 테스트의 임시
저장소까지 덮어써 게이트 시험 3건이 거짓 실패한다(2026-08-17 실측).
**테스트는 Windows 에서** 돌린다.

## 5. 실행 전 필수 (오늘 하루에 다 겪은 것들)

1. **코드를 먼저 커밋하고 깨끗한 트리에서 돌린다.** 로그 첫 줄에 `dirty=[]` 를
   찍어 확인한다. dirty 로 돌리면 그 수치를 만든 코드가 복원되지 않는다(YR-181).
2. **배선을 실행 전에 검증한다.** 정책이 요구하는 속성·의존 파일·통계 함수를
   미리 두드려 본다. 에피소드 끝에서 터지면 13분을 날린다(YR-179 실측).
3. **판정 기준을 사전등록하고 커밋한 뒤** 실행한다. 판정축은 **규칙 대비**로
   잡고 **모드(argmax/추첨)를 명시**한다 — 둘 다 YR-185 에서 실패한 지점이다.
4. **소요 시간은 물결 수를 올림해서** 계산한다. `108판 ÷ 24 = 4.5` 는 4.5물결이
   아니라 **5물결**이다(마지막 부분 물결도 한 물결 시간을 쓴다).

## 6. 보고·인계

- RL 세션은 판정이 나면 **board done row 에 evidence 를 박제**하고 push 한다.
- 본 세션은 그 결과를 논문 초안(`docs/paper/`)에 반영한다.
- 서로의 진행은 **board 와 커밋 로그로만** 확인한다. 구두 인계에 의존하지 않는다.

## 7. 하지 않는 것

- 두 세션이 동시에 무거운 실험을 돌리는 것(합의 없이).
- 상대 레인의 코드·산출물 디렉터리를 건드리는 것.
- board 파일을 오래 열어두고 여러 row 를 한꺼번에 고치는 것.
- 결과를 보고 판정축을 바꾸는 것 — 레인과 무관하게 금지다.
