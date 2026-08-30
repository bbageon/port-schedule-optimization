# YR-267 — 저자·소속·사사 기입 (LNCS 규약 확인 후)

- **Epic**: Paper / **Priority**: 🟠 / **등록·완료일**: 2026-08-30

## 발단

사용자 질문 셋: ①머리말의 `A. Name` 을 지울 수 있는가 ②소속이 같으면 한 번만
쓰는가 ③사사 문구는 템플릿 어디에 쓰는가.

## 조사 결과 (근거)

- `llncs.cls` **v2.24 (2024-01-29)** — 로컬 파일 직접 확인.
- 머리말은 `runningheads` 옵션이 만들고 이름은 `\authorrunning` 에서 온다.
  스프링거 LNCS 는 머리말을 쓰는 양식이므로 **지우는 대신 실제 이름을 채운다**.
  이름은 머리말에서 이니셜로 줄인다 (llncsdoc §3.2).
- 소속은 `\institute` 에 넣고 **여러 곳일 때만 `\and` 로 나눈다** — 그때만 위첨자
  번호가 붙는다 (llncsdoc §3.3, 클래스 `\ifnum\c@@inst=1`). 두 저자가 같은 학과·
  같은 대학이므로 **한 번만** 적고 `\inst{}` 를 쓰지 않는다.
- 교신저자는 이름 바로 뒤 **봉투 기호 위첨자** (스프링거 지침). `*` 나
  "Corresponding author" 같은 글자는 쓰지 않는다. 기호는 `marvosym` 의 `\Letter`.
- 사사는 **참고문헌 바로 앞**, `credits` 환경 안에 둔다 (llncsdoc §6).
  `\subsubsection{\ackname}` 은 선택, `\subsubsection{\discintname}`
  (Disclosure of Interests) 는 **필수**다.

## 넣은 내용

- 저자: `Geon-U Kim \and Bong-Jun Choi` + 교신 봉투 · `\authorrunning{G.-U. Kim and B.-J. Choi}`
- 소속: Department of Computer Engineering, Dongseo University, Busan, Republic of Korea
- 사사: 중소벤처기업부 민관공동 기술개발사업(기술이전 사업화) RS-2026-25529952
- 이해관계 공시: 경쟁 이해관계 없음 (템플릿 기본 문장 — **저자 확인 필요**)
- 국문판: 이름 `김건우 · 최봉준`(로마자에서 옮김 — 확인 필요) · 소속 동서대학교
  컴퓨터공학과 · `\ackname`→사사 · `\discintname`→이해관계 공시 ·
  `\andname`→`·` (그대로 두면 "김건우 and 최봉준" 으로 찍힌다)

## 남은 것

- 교신저자 이메일은 스프링거 필수 항목인데 아직 없다 — `\institute` 안에
  `\email{...}` 로 넣어야 한다. 두 파일에 TODO 주석으로 표시.
