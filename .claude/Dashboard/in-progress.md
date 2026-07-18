# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-061 | RL | 단일 DQN 보상 재설계 — 퇴화 원인 축차 검정 (phase 2: 할인 정합 γ 사다리) | 🟠 | 2026-07-18 | phase 1 미완료 페널티 **기각(전제 불성립 — 완료율 구조적 1.0, 페널티 무발동)**, 퇴화 = 지연형. 새 용의자 = 할인 근시 → γ {0.95→1.0} 사다리 실행 중. [사전등록+결과](../docs/strategy-history/2026-07-18-YR-061-미완료페널티-prereg.md) |
| YR-059 | RL | 상태 feature scale-only 정규화 + 클리핑 → QMIX 재실행 | 🟠 | 2026-07-18 | [적용전략](../docs/상태정규화-보상가중치-적용전략.md) §4·6-1 기준. **병행 세션(YR-061)과 동시 진행 — 사용자 지시**, 파일 경계: schema/encoding/qmix 하네스 (YR-061은 보상 경로) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
