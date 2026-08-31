# YR-285 — 교신저자 이메일 변경 (제출본만)

**상태**: done (2026-08-31)
**Epic**: Paper

## 왜 했나

사용자 지시: *"bongjun.choi@dongseo.ac.kr 이걸로 메일 변경해줘"*
→ 이어서 *"submission 만 바꾸면 돼"*.

## 무엇을 했나

`bjchoi@gdsu.dongseo.ac.kr` → `bongjun.choi@dongseo.ac.kr` 을
**`docs/paper/v3/submission/main.tex:52`** 한 곳에만 적용했다.

처음에는 "연락처는 사실 정보라 판본이 갈릴 이유가 없다"고 보고 세 원고를 모두 바꿨으나
(`9d72d49`), 사용자가 제출본만으로 한정해 작업본 둘은 원래 주소로 되돌렸다. 결과적으로
제목([[YR-283]])과 같은 규칙이 됐다 — **제출본이 앞서고 작업본은 그대로 둔다**.

| 파일 | 줄 | 주소 |
|---|---|---|
| `docs/paper/v3/submission/main.tex` | 52 | `bongjun.choi@dongseo.ac.kr` |
| `docs/paper/v3/latex-en/main.tex` | 62 | `bjchoi@gdsu.dongseo.ac.kr` (그대로) |
| `docs/paper/v3/latex/main.tex` | 73 | `bjchoi@gdsu.dongseo.ac.kr` (그대로) |

## 검증

세 원고 각 2회 빌드 후 PDF 1쪽에서 렌더된 문자열을 직접 읽었다 — LNCS 는 이메일을 소속
바로 아랫줄에 두므로 줄바꿈으로 쪼개지지 않았는지가 확인 대상이다.

| 판본 | 쪽수 | Overfull | 오류 | 1쪽 이메일 |
|---|---|---|---|---|
| 제출본 | 12 | 0 | 0 | `bongjun.choi@dongseo.ac.kr` |
| 작업본 영문 | 20 | 0 | 0 | `bjchoi@gdsu.dongseo.ac.kr` |
| 작업본 국문 | 17 | 0 | 0 | `bjchoi@gdsu.dongseo.ac.kr` |

배포본 `논문-영문.pdf`·`논문-국문.pdf` 갱신 · `scripts/v3/verify_submission.py` 대조 실패 0건.

## Evidence

`9d72d49` (세 판본 변경) · `609b5fb` (제출본으로 한정 · 작업본 둘 되돌림)
