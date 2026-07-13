# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-027-a | RL | Exp-1 Direct-Job Cost-Q 전략 명세 동결·히스토리 기록 | 🟠 | 2026-07-13 | ⏸ 외부 대기 — 문서 commit `9b46ae2`; `origin` push는 GitHub CLI 설치·인증 필요 |
| YR-027 | RL | 외부트럭 Direct-Job Cost-Q 구현·평가 — 선박 제외, `BLOCK_ENTRY` 이후 개별 작업 `argmin` | 🟠 | 2026-07-13 | full run 완료·primary FAIL: 평균 +0.039분, P95 CI 상한 +7.13%, fallback 55.0%; commit·push/evidence 정리 중 · [report](../../outputs/reports/exp1_direct_costq_hjnc/exp1_direct_costq_report.md) |
| YR-015-d | UI | 즉석 실행 패널 — 터미널 환경·정책·부하(트럭 수 등) 선택 → 실시간 시뮬·재생 + 컨테이너 배색 다양화 | 🟡 | 2026-07-13 | 사용자 요청. record→replay 원칙 유지 (04 §2.1), 즉석분은 `outputs/replays/live/` (미추적) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
