# 🚫 Cancelled (폐기)

> 폐기·중단 작업 (사유 박제 — negative result 보존). [index](README.md).

| ID | Epic | Title | 폐기 | 사유 |
|---|---|---|---|---|
| YR-097 | RL | 본선-임계 우선순위 경험재생 (PER) | 2026-07-26 | **사용자 결정 — RL-본선 트랙 종결의 일부** ([경계발견 문서 §3](../docs/strategy-history/2026-07-26-RL본선트랙-종결-경계발견-5축소진.md)): YR-098 천장 숫자가 투자 부정 — ①학습형이 넘어야 할 벤치마크가 한 줄 VF −2.84(~4%)로 얇고 ②**조건부==무조건(정밀 개입 상금 0)** = PER 이 표집을 고쳐도 배울 대상이 없음 ③같은 조건 rollout 이 3~4배(−9.7~−12.3) ④PER 은 표집 절반만 고침(제로섬·관측별칭 그대로). 재개 조건은 문서 §6(양하 레버 활성·조건부≠무조건 환경·별칭 해소·다중블록) — 성립 시 YR-098 재측정부터. [spec](../docs/dashboard-task-specs/YR-097-vessel-prioritized-replay.md) 보존 |
| YR-032 | RL | 계열 2 미래정보 잔차 Δ-net 별도 단계정책 | 2026-07-15 | [spec](../docs/dashboard-task-specs/YR-032-future-info-residual-rl.md) · ETA/포지셔닝/선재조작을 별도 정책으로 나누지 않고 YR-014의 동일 통합정책 ablation으로 흡수 |
| YR-026 | RL | 트럭-only 비용계수 민감도 | 2026-07-15 | 최종 Q 목표가 터미널 Total Cost로 확대되어 [YR-038](../docs/dashboard-task-specs/YR-038-total-terminal-cost.md)의 정규화·가중치 민감도로 흡수. 기존 YR-025 negative 근거는 Done에 보존 |
| YR-040 | Exp | 단일 야드 다중 test band 평가 | 2026-07-15 | 단일 야드 트랙 종료 ([결론서](../docs/strategy-history/2026-07-15-single-yard-track-closure.md)) — "greedy near-optimal" 결론이 이미 2-band(220k +0.035 / 240k +0.111)+최적선택 하한으로 확정. 다중 band 는 win-claim 방어용이었으나 win 을 추구하지 않으므로 불필요 |

---

운영: 폐기 시 본 파일로 이동 + 사유 기록. ID 재사용 금지.
