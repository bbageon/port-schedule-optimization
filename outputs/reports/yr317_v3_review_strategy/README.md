# YR-317 — v3 리뷰 대응 전략 작성 증거

- 작성: 2026-09-14. 추가 실험·재학습·원고 수정은 실행하지 않았다.
- 범위: [전략](../../../.claude/docs/strategy-history/2026-09-14-v3-review-response-strategy.md),
  [실험 설계](../../../.claude/docs/strategy-history/2026-09-14-v3-review-experiment-protocol.md),
  [사용자 맥락·재현성](../../../.claude/docs/strategy-history/2026-09-14-v3-review-hci-reproducibility.md).
- Dashboard: YR-317과 a~f를 backlog에 등록했다. 전략은 작성됐으며 개정 작업이 남아 있다.
- 원자료 재계산: `python outputs/reports/yr317_v3_review_strategy/recompute.py`.
- [snapshot.json](snapshot.json)은 기존 8개 실험 묶음·29개 정책 자료의 28일 합계와 원파일 해시다.
- 첫 재계산 시점 HEAD를 기록했다. 문서 수정 중 실행한 읽기 전용 집계이며 clean 실험 실행 기록이 아니다.
- 전체 14.7204%, 시간만 13.4416%, 공간만 -1.3070%라는 기존 비용 변화의 규모를 대조했다.
- 수락망 비교의 거절 제거 정책−전체 정책은 2.9491억 원이다.
- 위 수치는 합성 비용의 재합산이며 독립 시드에 대한 유의성·현장 타당성·학습 성공 판정이 아니다.
- 전략은 기존 날짜별 검정의 한계, 같은 난수의 수요 민감도, 철회된 v4 교사 실험을 명시했다.
- 현재 v3 게이트 판정은 재발행하지 않았다. 저장된 2026-08-17 공통 게이트를 승계하지 않았다.
- 알려진 기존 문제: YR-315 ID 중복, YR-296 row/spec 상태 불일치, 200줄 초과 기존 문서.
  전체 board 통과를 주장하지 않는다. 기존 이력은 이번 문서 작업에서 변경하지 않았다.
- 완료 전 검사 범위: 신규 row/spec 일치·상대 링크·변경 문서 줄 수·원자료 합계·공백 오류.
- 검사 명령: `python outputs/reports/yr317_v3_review_strategy/validate.py`.
- [검사 결과](validation.json): Markdown 14개·row/spec 7쌍·원자료 해시 29개, 지적 0건.
- 전략 산출물 커밋 `2480ba3`을 origin에 push했고 parent spec·Dashboard에 같은 증거를 기록했다.
