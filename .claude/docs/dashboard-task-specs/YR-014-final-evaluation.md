# YR-014 — 통합정책 locked 평가·ablation·운영 적용판정

- **Epic**: Exp / **Priority**: 🟡 / **등록일**: 2026-07-12 / **재기준화**: 2026-07-15
- **배경**: [최종전략 전환](../strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md)은 Exp-1~4 별도 정책 대신 동일 통합정책에서 ETA·선제정리·본선위험·레인·협조 요소를 제거하는 ablation을 요구한다.
- **목표(수용 기준)**: 코드·schema·비용 scale·자료버전·seed를 재현하고 봉인된 test를 1회 평가한다. 평균대기 감소, P95·본선완료·STS/이송대기 악화 없음, 이동/재조작 중 하나 개선, 제약위반 0을 동시에 판정한다.
- **범위 밖(non-goal)**: 탄소를 근거로 한 정책 재선정, test 결과 기반 hyperparameter 수정.
- **계획**: 강한 동정보 baseline·중앙 matching·QMIX paired 평가 → 요소별 ablation → 부하/ETA오류/장애 강건성 → 실패사례·한계·사후 탄소.
- **산출물**: 통합 검증 리포트, 재현 manifest, ablation·guardrail 표, 운영 채택/보류 판정.
- **의존**: YR-002·009·013·035~039, YR-015 UI 검증.
