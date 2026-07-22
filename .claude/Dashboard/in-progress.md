# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-080 | RL | 목적함수 문헌정합 재설계와 채택 정책 재판정 — **1단계(계약·엔진 정렬+기준재 목적) 완료, 2단계(교사 재수집→증류→FT 재판정) 대기** | 🟠 | 2026-07-21 | [spec](../docs/dashboard-task-specs/YR-080-objective-contract-redesign.md) · **1단계 완료 (2026-07-22, 6단계 커밋 사슬 `827bba4`~`9ed4923` — [전략 v5 §9](../docs/strategy-history/2026-07-21-본선처리-전략.md))**: ①양하 방향 역전 수정(STORE·물리도착 해제) ②**인과 연결 완성** — "야드 늦추면 배 늦음" 단조성 테스트 통과(유령 pre-fill 삭제·YC→이송→안벽 게이팅·전량 정합) ③비용==KPI 등식(berth_overrun_s)·SYMPTOM 미완 정산 교정 ④기준재 config(numeraire v1: 트럭 1h=1.0·ρ_vessel 33·proxy 0·λ≡1) ⑤manifest 재동결 d04559e7(변경=의도 3파일·반출 high P50 공개범위 첫 진입) ⑥적재 seam(기본 off). 트럭트랙 스냅샷 3건 불변 유지. **2단계**: 기준재 목적 하 교사 재수집→listwise 증류→반사실 FT→재판정 (기존 student_ft 자동 승계 불가) + ρ 민감도·본선 tier 는 타이트마감 train/val 셀 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
