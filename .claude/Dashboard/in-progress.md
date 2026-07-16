# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-048 | Sim | PRE_REHANDLE(ETA 선제 재조작) 후보가 통합 실험에서 전혀 발생하지 않음 — `integrated/scenario_gen.py` 가 `provided_eta` 미설정 | 🟠 | 2026-07-16 | **YR-047 적대 리뷰 파생 발견 (2026-07-16)**: 후보 생성기는 PRE_ADVICE + `job.provided_eta` 를 요구하는데(candidates.py:172) 통합 시나리오 생성기·fixture 는 provided_eta 를 설정하지 않는다 (설정처는 단일야드 `io/scenario_gen.py` 뿐). 실측: 3개 시나리오 후보 385건 중 PRE_REHANDLE **0건**. ETA 기반 선제정리는 연구 핵심 축(가설 H2·최종전략 §8.2)이므로 이대로 YR-045 를 돌리면 해당 축이 통째로 비활성인 채 판정하게 된다. 할 일: 통합 생성기에 ETA 분포(기존 gaussian 파라미터와 정합) 주입 + PRE_REHANDLE 발생율 검증 테스트 + YR-045 사전등록에 반영. **YR-045 착수 전 처리 권장** |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
