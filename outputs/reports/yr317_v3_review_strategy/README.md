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
- 최초 작성 검증: Markdown 14개·row/spec 7쌍·원자료 해시 29개, 지적 0건. [현재 검사 결과](validation.json).
- 전략 산출물 커밋 `2480ba3`을 origin에 push했고 parent spec·Dashboard에 같은 증거를 기록했다.

## 사용자 범위 수정과 정밀 검토

- [현행 전략](../../../.claude/docs/strategy-history/2026-09-14-v3-review-refined-strategy.md).
- [전체 후보 검증](../../../.claude/docs/strategy-history/2026-09-14-v3-review-ranking-detail.md) · [반복·제거 비교](../../../.claude/docs/strategy-history/2026-09-14-v3-review-evaluation-detail.md).
- 예약 정원·변경 부담은 결론·제약층 필요성·후속연구로 이관했다. 별도 YR-317-g는 기존 수요 보존 검사다.
- 읽기 전용 재검토 명령: `python outputs/reports/yr317_v3_review_strategy/refine_audit.py`.
- [정밀 감사 결과](refinement-audit.json): 코드·원자료 39개 해시와 함수 위치, 미투입·비용 항 합계.
- ckpt_000은 첫날 라벨 55건으로 학습한 뒤 저장됐다. 진짜 초기 모형과 구별해야 한다.
- 수락망 비교의 미투입 집계는 전체 745·거절 제거 675·재배치 없음 1,919대다. 원인·비용 영향은 미확인이다.
- NOVETO는 망의 거절만 끄고 점수에 의한 중앙 정렬은 남긴다. 전체 망 제거의 근거로 쓰지 않는다.
- 변경 후보의 목표는 공통 KEEP 대비 차이의 절반이다. 초기의 임의 a-b/b-c 설명을 실제 v3 구조로 적용하지 않는다.
- 신규 시뮬레이션·재학습·원고 편집은 0회다. 코드 실행 경로를 수정하지 않았다.
- 현행 검증은 Markdown 18개·row/spec 8쌍·고유 원자료/코드 해시 50개를 대상으로 한다.
