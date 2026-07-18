# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-063 | RL | 단일 DQN 신용 개별화 — 차분(counterfactual WAIT) 귀속 1-step Q | 🟠 | 2026-07-18 | **YR-061/062 파생 — 잔여 유일 경로**. 크레인별 credit = 내 행동 rollout 비용 − 내가 WAIT 했을 때 rollout 비용 (상대 행동 고정, 600s 창·SF_SPT base — JR 기계 재사용). 학습시 특권 rollout, 실행시 Q만 (CTDE 관례) |
| YR-059 | RL | 상태 feature scale-only 정규화 + 클리핑 → QMIX 재실행 | 🟠 | 2026-07-18 | [적용전략](../docs/상태정규화-보상가중치-적용전략.md) §4·6-1 기준. **병행 세션(YR-061)과 동시 진행 — 사용자 지시**, 파일 경계: schema/encoding/qmix 하네스 (YR-061은 보상 경로) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
