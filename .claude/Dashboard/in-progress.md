# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-121 | RL | **2차 단일축 — WAIT 지속시간 벌점 (사용자 선택)** | 🔴 | 2026-07-29 | 사전등록 `5cb48cf` **동결**(결과 미열람). 유일 차이 = `WAIT_TIME_PENALTY 1.0`(크레인이 의도적으로 노는 1시간 = 트럭 1대 대기 1시간, assumed 앵커). 대조군 = YR-119 WAITON 동결 산물 재사용. 판정: 통과 = K1(전략적 WAIT 감소 3/3 시드) ∧ K2(총비용 CI 상한<0) ∧ K4(풍선 재발 없음 — 재배치<0.30·60% 장악 0건). **미통과 시 기각 보고, 같은 실험 안 벌점 튜닝 금지**. 학습 3런(시드 88000/99000/123000 × 500ep) 병렬 실행 → 평가 → 판정 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
