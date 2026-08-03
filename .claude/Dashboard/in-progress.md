# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-147 | RL | WAIT-WAIT 정책 최적화 — 유한 DEFER 뒤 반사실 순위학습 | P0 | 2026-08-03 | [spec](../docs/dashboard-task-specs/YR-147-wait-wait-policy-optimization.md) · 1·2단계 완료. 선결 보정(`23d873b`): C 원자료·B=C hash·`완주→backlog→비용` 라벨·정책독립 K≤4 후보를 확인했다. **3단계 R 구현·학습·판정은 미실행**이며, 실행 전 손실 단위·별도 난수·전용 테스트·라벨 노출·최소검출효과(MDE)·시나리오 신뢰구간·B/R 가드를 잠근다. |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
