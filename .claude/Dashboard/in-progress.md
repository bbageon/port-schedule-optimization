# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-064 | RL | BC 초기화 + 차분 신호 미세조정 — 효율(BC)×행동유인(차분) 결합 | 🟠 | 2026-07-18 | YR-063 파생. TD(YR-062 파괴)와 달리 차분은 순위-정렬 신호 — BC 보존·개선 검정. **YR-065 와 병행 (사용자 지시)** |
| YR-065 | RL | 차분 신호 개량 — window 사다리 1200/2400s (근시 교정) | 🟡 | 2026-07-18 | YR-063 파생. "일은 하나 순서를 모름"의 window 근시 가설 검정. rollout 비용 window 비례 — 장시간 예상. **YR-064 와 병행 (사용자 지시)** |
| YR-059 | RL | 상태 feature scale-only 정규화 + 클리핑 → QMIX 재실행 | 🟠 | 2026-07-18 | [적용전략](../docs/상태정규화-보상가중치-적용전략.md) §4·6-1 기준. **병행 세션(YR-061)과 동시 진행 — 사용자 지시**, 파일 경계: schema/encoding/qmix 하네스 (YR-061은 보상 경로) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
