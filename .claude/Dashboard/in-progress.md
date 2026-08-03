# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-147 | RL | WAIT-WAIT 정책 최적화 — 1단계 기준 측정 (하드 마스크 없음·유한 DEFER 는 2단계) | P0 | 2026-08-03 | [spec](../docs/dashboard-task-specs/YR-147-wait-wait-policy-optimization.md) · 1단계 기준선(21차 정정) 완료 → **2단계 구현·학습·파일럿 완료**: 유한 DEFER(T_MAX 600s·wake 재개방) 동결 `2369af8`. 파일럿 발견 — ①**C≡B 바이트 동일**(trigger 부재 상태가 분포에서 공허 — C 축 재정의 필요) ②B 재개방 작동(A 미완주 6판→해당 초기화 0)·단 B:99000 신규 4판 = **순위 실패 잔존** ③층화 D_wait 양수 75~81%. 다음 = **3단계 반사실 순위 신호** 설계·사전등록 (판정 표본 동결 동반) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
