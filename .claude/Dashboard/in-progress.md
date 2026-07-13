# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-027-a | RL | Exp-1 Direct-Job Cost-Q 전략 명세 동결·히스토리 기록 | 🟠 | 2026-07-13 | ⏸ 외부 대기 — 문서 `9b46ae2`; `git push`가 GitHub HTTPS 인증 부재로 실패 |
| YR-015-f | UI | Three.js 실시간 3D 뷰어 — 연속 시간 재생 (크레인 활주·트럭 진입/대기/퇴장), 항만 씨너리 강화, 레이아웃 재배치 (정책 패널 상단·화면 확대) | 🟡 | 2026-07-13 | 사용자 요청 (plotly frames 크레인 정지 한계 → 엔진 교체 허용). replay.json 데이터 계약 불변 — 렌더러만 교체 |
| YR-027 | RL | 외부트럭 Direct-Job Cost-Q 구현·평가 — 선박 제외, `BLOCK_ENTRY` 이후 개별 작업 `argmin` | 🟠 | 2026-07-13 | 최소상태 v2 FAIL — fallback 0.01%로 coverage 통과, shortest-service 대비 평균 +1.195분·P95 +47.57%; push 확인 전 · [report](../../outputs/reports/exp1_direct_costq_minimal_hjnc/exp1_direct_costq_report.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
