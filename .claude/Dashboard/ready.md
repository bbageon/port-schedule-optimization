# 📋 Ready (선택됨)

> 착수 준비된 작업 (착수 신호 대기). [index](README.md) · 인접: [backlog](backlog.md) → 여기 → [in-progress](in-progress.md).

| ID | Epic | Title | Priority | Blocked by / Note |
|---|---|---|---|---|
| YR-091 | Sim | **비통과 크레인 물리 — idle/down 크레인 상시 장벽 + 순서·최소간격 불변식** (외부감사 결함2, 2026-07-26 기준 `4b44737`) | 🔴 | 현 이동구간 충돌검사는 "작업 예약 중 크레인끼리"만 — idle·고장·WAIT 크레인 위치가 장애물이 아니어서 **비통과 RMG 가 idle 크레인을 관통**(실측 bay20 idle 관통 허용·mid/high SF 실행당 1~7회). shared 크레인 동일 시작 bay 초기화 포함. 수정: ①초기 위치·물리 순서 명시 ②idle/down 상시 장벽 ③매 이벤트 순서·최소간격 invariant. **행동가능집합 변경 → 골든 재동결+기왕 결과 재판정** (YR-092 와 묶어 1회). [reservation.py:61](../../src/yard_rl/integrated/reservation.py#L61)·[cranes.py:36](../../src/yard_rl/integrated/cranes.py#L36). 수정 전 "물리·안전 위반 0" 주장 금지. **YR-075-c 앞 선결** |
| YR-092 | Sim | **초기 스택 규격 정합 — 시나리오 생성이 실행 적재규칙(pile 동일규격) 준수** (외부감사 결함3) | 🔴 | 실행 규칙은 pile 당 같은 규격만 허용하는데 생성기는 tier 마다 20/40ft 독립 추출 — 12 seed 전부 혼합 pile(mid/high 평균 ~75개)·재조작 용량검사 "blocker 동일규격" 가정 위반 = **시작상태부터 물리 가정 불일치**(재조작 수·서비스시간·목적지 가용성·정책 순위 영향 가능). 수정: 초기 배치가 place() 규칙 사용 + 초기 validator 에 규격 규칙 검사. **초기상태 변경 → YR-091 과 동시 골든 재동결**. [scenario_gen.py:179](../../src/yard_rl/integrated/scenario_gen.py#L179) |
| YR-075-c | RL | 재조작 목적지 **K후보·30분 국소 rollout 헤드룸** — H1 잔여격차 재판정 (행동공간 확정) | 🟠 | **승격 (2026-07-26)**: 선결 YR-089 완료. **단 감사 후 YR-091·092 뒤로 순연** (물리 정정이 재조작·목적지 가용성을 바꿈). [spec](../docs/dashboard-task-specs/YR-075-c-destination-rollout-headroom.md) |

---

운영: [backlog.md](backlog.md) 에서 승격. 착수 시 [in-progress.md](in-progress.md) 로 이동.
