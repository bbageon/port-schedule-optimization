# YR-270 — 키워드·교신 이메일·구분기호와 시스템 아키텍처 그림 교체

- **Epic**: Paper / **Priority**: 🟠 / **등록·완료일**: 2026-08-30

## 사용자 지시

- "Keywords 도 적절히 골라서 넣어주고 교신저자 메일은 bjchoi@gdsu.dongseo.ac.kr"
- "· 대신 , 쓰자"
- "S.A.png 시스템 아키텍처 해당 파일로 수정해줘 기존꺼는 지워도 돼"

## 한 일

- **이메일**: 스프링거 지침대로 소속 **바로 아랫줄**에 `\email{}` 로 넣었다
  ("Email addresses should start on a new line directly under the corresponding
  affiliation"). [[YR-268]] 의 필수 항목 하나가 닫혔다.
- **키워드**: 색인어로 더 표준적인 표현으로 바꿨다 — `Truck arrival management`
  → `Truck appointment system`(문헌 표준어, 국문 "트럭 예약제"), `Counterfactual
  learning` → `Counterfactual cost learning`(제목과 일치). 나머지 넷은 유지.
- **구분기호**: 키워드와 국문 저자 사이를 가운뎃점에서 쉼표로. llncs 의 `\and` 는
  가운뎃점으로 찍히므로 키워드는 `\and` 를 쓰지 않고, 저자는 `\andname` 을 바꾼다.
  클래스가 "이름 + 공백 + \andname" 으로 찍으므로 `\unskip` 으로 앞 공백을 지운다.
  ⚠️ 스프링거 지침은 키워드 구분에 가운뎃점을 **선호**한다고 적혀 있다(강제는 아님).
- **아키텍처 그림**: 새 파일을 `system-architecture.png` 로 이름을 바꿔 넣고 옛 파일을
  지웠다. 파일명에 점(`S.A.`)이나 공백(`System Architecture`)이 있으면 `\includegraphics`
  확장자 해석이 흔들려 안전한 이름으로 정리했다.

## 완료 조건

- 국·영문 각 2회 빌드 — Overfull hbox 0 · 오류 0 · 그림 일곱 건 캡션과 같은 쪽.
