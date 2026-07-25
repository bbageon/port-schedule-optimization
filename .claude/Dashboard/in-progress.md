# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| AUDIT-0726 | Sim | **외부감사 대응 — 평가창 결함 정정·재검증 + 물리결함 등록** | 🔴 | 2026-07-26 | 사용자 감사(기준 `4b44737`) 4결함: **①종료 후 완료를 완주로 계산(시간창 불일치)** → 즉시 수정(평가창 밖 이벤트·wake 처리 중단, RUNNING=검열·backlog, 회귀가드 테스트) + YR-089/087 박제수치 재측정 중. **②비통과 크레인 idle 관통 → YR-091 등록**(골든 재동결급 물리 정정). **③초기 스택 혼합규격 → YR-092 등록**(동급). **④예측 rollout 정보누출** → ETA 결측 fallback fail-closed 즉시 수정 + 잔여(미통지 고장·계획변경) YR-093 등록. 부수 4건 YR-094 등록. **YR-091/092 수정 전 "물리·안전 위반 0" 주장 금지** — 이후 순서: YR-091·092 → YR-075-c → 하이브리드 → YR-041 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
