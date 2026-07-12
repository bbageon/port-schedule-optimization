# Dashboard Board 하네스 전략 — 범용 템플릿

> **목적**: Agent(Claude Code 등)와 사람이 함께 쓰는 **파일 기반 작업 board** 를 아무 프로젝트에나 이식하기 위한 단일 문서. 이 파일 하나만 복사하면 구조·규약·자동화 훅까지 그대로 재현할 수 있다.
> **한 줄 요약**: Jira/Plane 스타일 칸반을 `상태별 md 파일 5개 + index 1개` 로 구현하고, board 는 **1줄 index 만** 유지하며(상세는 spec/일지/커밋이 원본), 훅으로 Agent 가 세션 시작·커밋 시점마다 board 를 갱신하도록 강제한다.

---

## 1. 설계 철학 (왜 이 구조인가)

| 원칙 | 내용 | 효과 |
|---|---|---|
| **상태별 파일 분리** | 상태(컬럼)마다 파일 1개. 한 파일에 전체 board 를 두지 않는다 | 상태 이동 = 잘라내기+붙여넣기 diff 가 명확. Agent 가 필요한 상태 파일만 읽어 컨텍스트 절약 |
| **Board = index, 원본은 딴 곳** | row 는 **1줄 요약 + 링크**. 상세 명세는 `task-specs/` 문서, 증거는 커밋·일지·평가 산출물 | board 비대화 방지, 중복 서술로 인한 drift 차단 |
| **Evidence 박제** | Done 은 반드시 commit 해시/산출물 링크 동반. Cancelled 는 폐기 사유 동반 | "완료 주장"과 "검증 가능한 완료"를 분리. negative result 도 보존 |
| **순서 제시 원칙** | Agent 가 "다음 할 일·우선순위"를 말할 때 **board 에 등록된 row 만** 인용. 없는 작업은 먼저 backlog 등록 | Agent 의 즉흥 계획 난립 방지, 계획의 단일 출처 유지 |
| **훅으로 습관화** | 세션 시작·git commit 시점에 board 갱신 reminder 를 자동 주입 | 규칙을 문서에만 두면 잊힌다 — 하네스가 대신 기억 |
| **ID 재사용 금지** | 닫힌(done/cancelled) ID 는 영구 폐번 | 이력 추적·링크 안정성 |

---

## 2. 디렉토리 구조

```
.claude/
├── Dashboard/                  # board (상태별 파일 분리)
│   ├── README.md               # index: 운영 규약 + 상태 파일 링크 + epic + 현황 overview
│   ├── backlog.md              # 🗂️ 미착수·미래
│   ├── ready.md                # 📋 착수 준비 (선별됨)
│   ├── in-progress.md          # 🟢 진행 중 (한 번에 1개 권장)
│   ├── done.md                 # ✅ 완료 (commit 링크 박제)
│   └── cancelled.md            # 🚫 폐기 (사유 박제)
├── docs/
│   └── dashboard-task-specs/   # row 별 상세 명세: <ID>-<slug>.md
└── settings.json               # 자동화 훅 (§6)
```

상태 흐름:

```
Backlog → Ready → In Progress → Done
                       └──────→ Cancelled (사유 박제)
```

---

## 3. 운영 규약 (규칙 전문)

1. **Issue ID**: `XX-NNN` 형식 (프로젝트 prefix + 일련번호, 예: `AR-021`). 하위 분할은 `XX-NNN-a/b/c`. **닫힌 ID 재사용 금지.**
2. **Priority**: 🔴 Urgent / 🟠 High / 🟡 Medium / ⚪ Low.
3. **Epic**: 큰 작업 묶음 라벨. README 에 epic 표(의미 + 상태: 안정/active/완료)를 유지.
4. **작업 시작(pull)**: `ready.md` 의 row 를 `in-progress.md` 로 **이동** (복사 아님). WIP(Work In Progress) 는 한 번에 1개 권장.
5. **작업 종료(commit)**: `in-progress.md` → `done.md` 이동 + **commit 해시/링크를 Evidence 열에 박제**. 파생 후속 작업은 그 자리에서 `backlog.md` 에 등록.
6. **폐기**: `cancelled.md` 로 이동 + 폐기 사유 기록 (negative result 도 자산 — 왜 접었는지가 다음 판단의 근거).
7. **1줄 index 원칙**: row 에는 제목·우선순위·한 줄 note 만. 배경·실험 설계·수용 기준은 `dashboard-task-specs/<ID>-<slug>.md` 로 분리하고 row 에서 링크.
8. **순서 제시 원칙**: 예정사항·다음 순서·우선순위는 board 에 등록된 row 기준으로만 제시. board 에 없는 작업은 먼저 backlog 에 등록한 뒤 언급.
9. **외부 대기 표기**: Agent 작업이 끝나고 사용자/외부 응답만 남은 in-progress row 는 Note 에 `⏸ 외부 대기` 를 명시 — WIP=1 규약과 새 작업 착수가 충돌하지 않음을 board 만 보고 알 수 있게.
10. **done 압축**: done row 가 20+ 누적되면 최근 ~20개만 본표에 남기고 나머지는 같은 파일 하단 `## Archive` 절로 이동.
11. **board ≠ 승인 게이트 장부**: 사용자 승인이 필요한 결정(예: 가설 status 전환, 임계값 변경)은 board 밖의 규칙(AGENTS.md 등)이 관장. board 에는 "승인 대기" 상태만 표기하고 Agent 가 단독 진행하지 않는다.

---

## 4. 파일별 템플릿 (복사용)

### 4-1. `README.md` (index)

```markdown
# `.claude/Dashboard/` — 작업 board (index)

> Jira/Plane 스타일 작업 board. **상태별로 파일 분리** (각 state = 한 파일). 본 README 가 **index**
> (운영 규약 + state 링크 + epic + 현황 overview). 상세 evidence 는 일지/평가 산출물/git commit 이
> single source — board 는 그 위의 index 일 뿐 중복 서술하지 않는다.

## 상태별 파일

| State | 파일 | 현황 |
|---|---|---|
| 🗂️ Backlog | [backlog.md](backlog.md) | 미착수·미래 |
| 📋 Ready | [ready.md](ready.md) | 착수 준비 |
| 🟢 In Progress | [in-progress.md](in-progress.md) | 진행 중 (한 번에 1개) |
| ✅ Done | [done.md](done.md) | 완료 (commit 링크 박제) |
| 🚫 Cancelled | [cancelled.md](cancelled.md) | 폐기 (사유 박제) |

흐름: `Backlog → Ready → In Progress → Done / Cancelled`.

## 사용 규약

- **Issue ID**: `XX-NNN`. 닫힌 ID/row 재사용 금지. **Priority**: 🔴/🟠/🟡/⚪. **Epic**: 아래 표.
- **작업 시작**: ready → in-progress 로 이동(pull, 한 번에 1개 권장).
- **작업 종료(commit)**: in-progress → done + commit 해시/링크 박제. 후속 작업은 backlog 에.
- **Board row 는 1줄 index**: 상세 명세는 `../docs/dashboard-task-specs/<ID>-<slug>.md` 로 분리.
- **순서 제시 원칙**: 예정사항·우선순위는 board 등록 row 기준으로만. 없는 작업은 backlog 먼저.

## 🧭 Epics

| Epic | 의미 | 상태 |
|---|---|---|
| (예) Infra | 환경·도구 | 상시 |

## 📌 현재 상태 overview (한눈에)

- (프로젝트의 핵심 결론/진행 상태를 3~5줄로 — 세션 시작 시 Agent 가 가장 먼저 읽는 요약)
```

### 4-2. `backlog.md`

```markdown
# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| XX-001 | 예시 | 예시 작업 한 줄 | 🟡 | [spec](../docs/dashboard-task-specs/XX-001-example.md) |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
```

### 4-3. `ready.md`

```markdown
# 📋 Ready (선택됨)

> 착수 준비된 작업 (착수 신호 대기). [index](README.md) · 인접: [backlog](backlog.md) → 여기 → [in-progress](in-progress.md).

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|

---

운영: [backlog.md](backlog.md) 에서 승격. 착수 시 [in-progress.md](in-progress.md) 로 이동.
```

### 4-4. `in-progress.md`

```markdown
# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
```

### 4-5. `done.md`

```markdown
# ✅ Done

> 완료 작업 (commit 링크 박제). 20+ 누적 시 하단 `## Archive` 로 압축 이동. [index](README.md).

| ID | Epic | Title | 완료 | Evidence |
|---|---|---|---|---|

---

## Archive
```

### 4-6. `cancelled.md`

```markdown
# 🚫 Cancelled (폐기)

> 폐기·중단 작업 (사유 박제 — negative result 보존). [index](README.md).

| ID | Epic | Title | 폐기 | 사유 |
|---|---|---|---|---|
| _(없음)_ | | | | |

---

운영: 폐기 시 본 파일로 이동 + 사유 기록. ID 재사용 금지.
```

### 4-7. task-spec (`dashboard-task-specs/<ID>-<slug>.md`)

```markdown
# XX-NNN — <제목>

- **Epic**: <epic> / **Priority**: <🔴|🟠|🟡|⚪> / **등록일**: YYYY-MM-DD
- **배경**: 왜 이 작업이 필요한가 (선행 작업·발견 링크).
- **목표(수용 기준)**: 무엇이 되면 done 인가 — 가능하면 정량 기준으로 사전 고정.
- **범위 밖(non-goal)**: 이번에 하지 않는 것.
- **계획**: 단계별 절차 (필요 시).
- **산출물**: 코드/스냅샷/리포트 경로.
```

---

## 5. Done row 작성 규격 (evidence 박제)

Done 의 Evidence 열은 다음 중 해당하는 것을 **전부** 링크한다:

```
<commit 해시> · [report](<평가 리포트 경로>) · [snapshot](<정량 산출물 경로>) · [일지](<일자별 일지 경로>)
```

- Title 열에는 결과 **결론 요약**을 굵게 1구절 포함 (예: "**X 가설 지지 (p<1e-4)**", "**self-STOP 기각**") — done 목록만 훑어도 무엇이 확정됐는지 보이게.
- 커밋이 없는 작업(문서·사용자 결정)은 근거 문서/대화 날짜를 대신 박제.

---

## 6. 자동화 훅 (`.claude/settings.json`)

Agent 가 board 를 잊지 않도록 두 시점에 reminder 를 주입한다. 그대로 복사 후 프로젝트에 맞게 문구만 수정:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "shell": "bash",
            "command": "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"[Project Board] 작업 시작 전: .claude/Dashboard/ (상태별 파일 분리) 를 읽어라. ready.md 에서 착수 항목을 in-progress.md 로 이동(pull). 종료(commit) 시 in-progress.md→done.md + commit 링크 박제. index=README.md.\"}}'",
            "statusMessage": "Project board reminder"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "shell": "bash",
            "command": "cat | grep -q 'git commit' && echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"[Project Board] commit 완료 = 작업 segment 종료. .claude/Dashboard/in-progress.md 의 해당 항목을 done.md 로 이동하고 방금 commit 해시/링크를 Evidence 열에 박제했는지 확인하라. 새 후속 작업은 backlog.md 에 추가.\"}}' || true",
            "statusMessage": "Project board update reminder"
          }
        ]
      }
    ]
  }
}
```

동작 원리:
- **SessionStart** — 세션이 열릴 때마다 "board 부터 읽어라" 컨텍스트를 주입 → Agent 가 항상 board 기준으로 계획.
- **PostToolUse(Bash + `git commit` 매칭)** — 커밋 직후 "done 이동 + evidence 박제" reminder 주입 → 완료 처리 누락 방지. 커밋이 아닌 Bash 호출은 `|| true` 로 조용히 통과.

---

## 7. 상위 하네스와의 관계 (4-layer 분배)

board 는 하네스 4계층 중 **상태(state) 레이어**다. 새 내용을 어디에 둘지 기준:

| layer | 역할 | 위치 | 판단 기준 |
|---|---|---|---|
| **rules** | 위반 시 결과가 무효화되는 절대 규칙·게이트 | `AGENTS.md` + phase 문서 | 위반하면 재현성·비교가능성·claim 유효성이 깨지는가 |
| **skills** | Agent 가 실행하는 절차·checklist | `.claude/skills/` | 단계별 수행 절차인가 |
| **docs** | 참조 문서 (용어·요약·spec·근거) | `.claude/docs/` | 인용·배경인가 |
| **Dashboard** | 작업 흐름·상태 board | `.claude/Dashboard/` | 진행 추적인가 |

- board row 에 규칙·절차·배경을 쌓지 마라 — 각각 rules/skills/docs 로 보내고 링크만.
- AGENTS.md(또는 상위 규칙 문서)에는 다음 한 조각만 있으면 된다:

```markdown
Dashboard row 는 1줄 index 로 유지하고 상세 작업 명세는
`.claude/docs/dashboard-task-specs/<dashboard-id>-*.md` 로 분리한다.
예정사항·다음 순서·우선순위는 Dashboard 에 등록된 row 기준으로만 제시하며,
Dashboard 에 없는 작업은 먼저 backlog 에 등록한다.
```

---

## 8. 신규 프로젝트 도입 체크리스트

1. [ ] `.claude/Dashboard/` 생성 + §4 템플릿 6개 파일 복사.
2. [ ] Issue ID prefix 결정 (예: `AR-`, `PX-`) + README 에 명시.
3. [ ] Epic 3~6개 정의 (너무 잘게 나누지 말 것 — epic 은 분기 단위 묶음).
4. [ ] `.claude/docs/dashboard-task-specs/` 생성.
5. [ ] `.claude/settings.json` 에 §6 훅 2개 등록.
6. [ ] 상위 규칙 문서(AGENTS.md/CLAUDE.md)에 §7 의 "1줄 index + 순서 제시 원칙" 조각 추가.
7. [ ] 현재 머릿속의 할 일을 전부 backlog 에 등록 (첫 grooming) → 상위 2~5개를 ready 로 승격.
8. [ ] 첫 작업을 in-progress 로 pull 하고 사이클 시작.

## 9. 운영에서 검증된 팁 (함정 회피)

- **README 의 "현황 overview" 를 살아있게** — 세션 시작 시 Agent 가 처음 읽는 3~5줄. 핵심 결론·미검증 항목을 여기 갱신해 두면 매 세션 재설명 비용이 사라진다.
- **후속 작업은 발견 즉시 backlog 등록** — "나중에 하자"는 말은 row 가 아니면 증발한다. done 처리하는 커밋과 같은 turn 에 등록하는 습관이 가장 잘 유지된다.
- **priority 는 결과가 나올 때마다 재정렬** — 실험 결과가 기존 backlog row 의 전제를 뒤집으면(예: "이 tool 불필요" 판정) 그 자리에서 하향/폐기. 방치하면 board 와 실제 판단이 어긋난 채 굳는다.
- **WIP=1 의 예외는 "외부 대기"뿐** — 사용자 응답·외부 평가 대기 row 는 `⏸` 표기 후 다음 작업 pull 가능. 그 외에 in-progress 2개면 하나는 거짓말이다.
- **done Title 에 결론을 쓰면 board 가 곧 연구 요약이 된다** — 리포트를 다시 열지 않아도 무엇이 확정/기각됐는지 done 목록 스캔만으로 복기 가능.
