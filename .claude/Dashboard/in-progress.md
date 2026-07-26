# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-090 | RL | **본선 점증 학습신호 (dense potential-based) — 희소·지연 vessel 보상 밀도화 판가름** | 🟠 | 2026-07-26 | **Pull (사용자 결정 — RL 최종 지위 판가름 실험)**. 사전등록 동결(하네스 docstring, 결과 미열람): Φ=(ρ/3600)·Σ예상지연(관측가능 계획값), F=γΦ′−Φ **훈련 전용**(평가 순수 numeraire·lump 재분배=이중계상 아님), arm 2(DENSE vs CONTROL=기존 최고 레시피, 유일 차이 Φ)×학습시드 3, **주판정=재현성**(DENSE 3/3 시드 berth CI 상한<0 & CONTROL<3 ⇒ 희소보상 가설 확정 / 2/3 부분지지 / 그 외 기각). v2 계약·물리 정정 후. 기대치 명시: 재현성 확보이지 계획법 추월 아님. 하네스 `yr090_dense_vessel.py`. RL 트랙과 별개로 성능 트랙(YR-096 rollout 단일계층)은 ready 대기 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
