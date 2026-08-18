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

**★정정 (2026-08-18 실측)**: 구 문장 "테스트는 Windows 에서"는 **틀렸다** —
Windows 에는 torch 가 없어 `tests/integrated/` 가 **수집 단계에서 19건 오류**로
전부 죽는다. 올바른 규칙은 다음 표다.

| 무엇 | 어디서 | `GIT_DIR` |
|---|---|---|
| **pytest** | **WSL** | **주지 않는다** (거짓 실패의 원인) |
| **학습·평가 실행** | WSL | **준다** (아래) |

`GIT_DIR` 를 안 주고 학습을 돌리면 `code_dirty()` 가 `False` 가 아니라
**`None`** 을 반환한다. 하드 가드 "실행 트리 `code_dirty == False`" 가
**조용히 검증 불가**가 된다 — 사전등록 가드가 있는 판정에서는 치명적이다.
실행 전에 한 줄로 확인한다:

```bash
PYTHONPATH=src python -c "from yard_rl.integrated.repro import code_dirty, git_head; print(code_dirty(), git_head())"   # False <sha> 여야 한다. None 이면 GIT_DIR 누락
```

## 5. 실행 전 필수 (오늘 하루에 다 겪은 것들)

1. **코드를 먼저 커밋하고 깨끗한 트리에서 돌린다.** 로그 첫 줄에 `dirty=[]` 를
   찍어 확인한다. dirty 로 돌리면 그 수치를 만든 코드가 복원되지 않는다(YR-181).
2. **배선을 실행 전에 검증한다.** 정책이 요구하는 속성·의존 파일·통계 함수를
   미리 두드려 본다. 에피소드 끝에서 터지면 13분을 날린다(YR-179 실측).
   · **정책은 `trail` 을 반드시 가져야 한다** — `run_episode_diurnal` 이 마지막
     줄에서 `build_joint_transitions(policy.trail, …)` 을 부른다. `KeepAllUnified`
     처럼 `trail` 이 없는 클래스를 쓰면 **9분을 다 돌고 마지막 줄에서** 터진다
     (2026-08-18 실측). 기준선은 `KeepAllTrail` 을 쓴다.
   · **긴 판정 스크립트는 짧은 관측창으로 먼저 돌린다** —
     `ObservationContract(warmup_s=3600, measure_s=7200)` 이면 한 판이 몇 분이다.
     판정 대역이 **아닌** 시드로 돌려 대역을 태우지 않는다.
   · **목표 눈금을 실행 전에 잰다.** 회귀 목표의 평균·범위를 스모크에서 찍어
     보고, 학습률 × 스텝수로 그 거리를 갈 수 있는지 계산한다. 못 가면 고정
     상수로 나눠 O(1) 로 만든다(YR-170 은 이걸 안 해서 16시간을 멈춘 채 돌았다).
3. **판정 기준을 사전등록하고 커밋한 뒤** 실행한다. 판정축은 **규칙 대비**로
   잡고 **모드(argmax/추첨)를 명시**한다 — 둘 다 YR-185 에서 실패한 지점이다.
4. **병렬 설계는 코어와 메모리를 함께 계산한다** (2026-08-18 실측 사고).
   `워커 수 × 워커당 RSS + 부모 ≤ 총 메모리 × 0.8`. 이 프로젝트는 **워커당
   약 3 GB**, 장비 62 GB 이므로 **동시 워커 16개가 상한**이다. 20개를 올렸다가
   3시드가 조용히 중단됐다 — 에러도 안 나고 파일만 안 써진다.
   **진행 확인은 회차 수가 아니라 `파일 마지막 갱신 시각`으로** 한다.
5. **소요 시간은 물결 수를 올림해서** 계산한다. `108판 ÷ 24 = 4.5` 는 4.5물결이
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
