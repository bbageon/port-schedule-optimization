# YR-028 — Direct-Job Cost-Q coverage 회복

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-13
- **배경**: YR-027의 validation 최저 checkpoint는 SLA_OFF fallback 56.2%였고 locked test도 55.0%였다. 후반 checkpoint는 coverage를 충족했지만 평균대기가 악화됐다.
- **목표**: greedy fallback의 성능과 학습 Q의 성능을 분리하고, 후보 signature state aliasing 또는 checkpoint 선택 규칙이 실패의 주원인인지 판별한다.
- **사전등록 후보**: validation fallback 5% 이하 checkpoint 중 평균대기 최소 선택, bucket/state 축소, train episode 증량을 각각 독립 ablation으로 둔다.
- **데이터 규율**: YR-027 test seed는 재선택에 쓰지 않는다. 새 train/validation/test seed band와 선택 규칙을 실행 전에 동결한다.
- **비교 기준**: YR-027 최강 비교군 `SHORTEST_ESTIMATED_SERVICE_TIME`과 pure Cost-Q(fallback 0%)를 우선 비교한다.
- **수용 기준**: test fallback 5% 이하, completion 100%, backlog 0, paired 평균대기 CI와 P95 +5% guardrail을 함께 보고한다.
- **비목표**: YR-027의 FAIL 결론을 사후 재해석하거나 선박·미래정보·다중 YC를 추가하지 않는다.
- **근거**: [YR-027 전략·결과](../strategy-history/2026-07-13-YR-027-exp1-direct-job-cost-q.md).
