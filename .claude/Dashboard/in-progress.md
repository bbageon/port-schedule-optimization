# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-133 | RL | 블록 간 판매 발의·수신부담 견적·결정론 확정 — 1차(반입·event-only·1건/epoch) | P0 | 2026-08-04 | [spec](../docs/dashboard-task-specs/YR-133-blockq-sell-quote.md) · 1차 사전등록 동결 `648d0c9`(top-1 OFFER·최소 InBurden·NetGain 결정론 확정·epoch 1건·이송≤1·quote epoch 전용·fail-closed·본선 가드·κ 동결·견적 원장). 기반 조사로 재사용(원자 이송 API·브리지 테스트 22종)/신규 갭 확정. 파일럿 8쌍(906000+) — 기능 가드 판정(효과 확증은 후속 단일축). 결과 미열람 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
