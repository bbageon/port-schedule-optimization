# YR-284 — 제출본 문장 교정 10건 (수치·주장 불변)

**상태**: done (2026-08-30)
**Epic**: Paper

## 왜 했나

사용자 요청: *"지금 전체적인 논문 흐름에 이상이 없는지 영어 문체 문법 어절 등 전체적으로
체크해서 줘"*. [[YR-282]] 가 **수치**를 기계로 대조했다면 이번은 **문장**을 읽었다.
수치와 주장은 한 글자도 건드리지 않는 것을 제약으로 뒀다.

## 고친 열 건

| # | 자리 | 무엇이 문제였나 | 어떻게 고쳤나 |
|---|---|---|---|
| ① | §5.3 | 시제 불일치 + §5.1 이 이미 말한 10.00/8.52 재등장 | 첫 회차 모형만 말하고 과거시제로 통일 |
| ② | §5.2 | 매달린 관계절 `on all of which` | `and the trained model is cheaper on all four` |
| ③ | §6 | 한 문장에 `also` 가 둘 | 뒤의 `also` 제거 |
| ④ | §3.1 | 독립절 둘을 쉼표 없이 `and` 로 이음 | 쉼표 추가 |
| ⑤ | §5.2 | `by the two` 의 지시대상 불명 | `by the two tests` |
| ⑥ | §3.3 | 서로 다른 두 이야기를 `and` 로 이어 붙인 긴 문장 | 문장 분리 |
| ⑦ | §3.2 | `commit` 을 자동사로 씀 | `never produces a commitment` |
| ⑧ | §5.3 | `with about 56 labelled decisions` 가 엉뚱한 절에 붙음 | 앞으로 옮기고 세미콜론으로 분리 |
| ⑨ | 그림 3 캡션 | `whose demand was drawn at` 어색 | `in which the demand level drawn was` |
| ⑩ | §1 | `which` 의 선행사 흐림 | `; this brings…` |

## 검증

12쪽 · Overfull 0 · Underfull 0 · 오류 0 · 미정의 참조 0 · 인용 18/18 ·
`scripts/v3/verify_submission.py` **41/41 일치** · `tex_escape_scan.py` 이상 없음.

## 남은 것

- §5.2 의 *"블록만 재배치하면 최고 수요에서 비용이 오른다"* 는 **대역 특정**이다.
  혼잡 민감도 실험(env-quiet 완료)에서 부호가 뒤집힌다 (+0.226 bn quiet · +0.394 bn
  판정대역 vs −0.131 bn 진단대역). 제거 제안했고 사용자 결정 대기 중 → backlog.

## Evidence

`b659dc8`
