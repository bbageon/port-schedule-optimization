# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-147 | RL | WAIT-WAIT 정책 최적화 — 유한 DEFER 뒤 반사실 순위학습 | P0 | 2026-08-03 | [spec](../docs/dashboard-task-specs/YR-147-wait-wait-policy-optimization.md) · 1단계 기준선·2단계 파일럿 완료(`2369af8`·`ed7b02a`). **22차 감사 정정**: B는 재개방을 보장했지만 A의 기존 미완주 6판 해소 뒤 B:99000 신규 4판이 생겨 선택 오류가 남음. C≡B는 환경 발견이 아니라 기존 `FORBID_WAIT`와의 **중복 구현**이며 C 원자료 분모도 미정정 → C 종료·계측 산출물 정정 필요. 3단계 전 `완주→backlog→비용` 반사실 라벨·정책독립 후보선정·체크포인트 hash/재실행 명령을 보정한 뒤 **B vs B+순위손실** 단일축 사전등록 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
