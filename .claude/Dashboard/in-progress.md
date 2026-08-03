# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-147 | RL | WAIT-WAIT 정책 최적화 — 3단계 판정 완료·§11 재결정 대기 | P0 | 2026-08-03 | [spec](../docs/dashboard-task-specs/YR-147-wait-wait-policy-optimization.md) · 1·2단계 + 선결 보정(`23d873b`) 완료 → **3단계 실행 동결(`7d0c547`·`bc596ac`: R 배선·계약 테스트 4·훈련 대역 BASE+16..31·판정 32시나리오 MDE 1.03) → 학습 6개(노출 229~235 충족) → 판정: ★공식 기각** — 방향 1/3·R−B 평균 −0.246±0.884(CI 상한 +0.64>0)·**단 하드가드는 R 96판 전판 완주 vs B 미완주 4판**(잘못된 연기 계급 제거 방향 증거·비용 주장 금지). [report3](../../outputs/reports/yr147_defer/report3.md)·[results3](../../outputs/reports/yr147_defer/results3.json)·판정 커밋 `60e9c77` · **spec §11 발동 — 트랙 중단/부분 성과 종결/4단계 진행 사용자 재결정 대기** |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
