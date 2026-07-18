# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-062 | RL | 단일 DQN 성능 경로: BC(모방) 초기화 + RL 미세조정 — SF_SPT 초과 시도 | 🟠 | 2026-07-18 | **YR-061 phase-3 파생**: 표현력 충분·BC 56.25(SF_SPT +3.13). lr 사다리 미세조정 — 되망가지면 신용 희석 2차 확증, 유지·개선이면 성능 경로. 알려진 위험: CE-표적(서수) Q 가 TD 비용 재척도화에 파괴될 수 있음 — lr 사다리가 완충 |
| YR-059 | RL | 상태 feature scale-only 정규화 + 클리핑 → QMIX 재실행 | 🟠 | 2026-07-18 | [적용전략](../docs/상태정규화-보상가중치-적용전략.md) §4·6-1 기준. **병행 세션(YR-061)과 동시 진행 — 사용자 지시**, 파일 경계: schema/encoding/qmix 하네스 (YR-061은 보상 경로) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
