# YR-269 — 머리말·쪽번호 제거 (스프링거가 조판 때 넣는다)

- **Epic**: Paper / **Priority**: 🟡 / **등록·완료일**: 2026-08-30

## 발단과 정정

사용자 물음: *"20 G.-U. Kim and B.-J. Choi — 이거는 왜 계속 뜨는지 모르겠네"*.

[[YR-267]] 에서 나는 *"LNCS 는 머리말을 쓰는 양식이라 지우면 안 된다"* 고 답했는데
**틀렸다**. 근거로 삼은 것은 클래스 견본이 `runningheads` 옵션을 쓴다는 사실뿐이었고,
정작 저자 지침을 확인하지 않았다. 스프링거 회의록 저자 지침 원문:

- §2.4 *"There is **no need to include page numbers or running heads**; this will be
  done at our end. If your paper title is too long to serve as a running head, it will
  be shortened. Your suggestion as to how to shorten it would be most welcome."*
- §4.1 *"In addition, running-heads, final page numbers, and a copyright line **are
  inserted**"* (출판사 조판 과정 설명).

## 편집 계약

- `\documentclass[runningheads]{llncs}` → `\documentclass{llncs}` (국·영문 둘 다).
- `\titlerunning`·`\authorrunning` 은 **남긴다** — 지침이 줄인 형태를 제안해 달라고
  하므로, 원고에 남은 줄임형이 그 제안 역할을 한다. 옵션이 꺼져 있으면 조판에는
  쓰이지 않는다.
- 쪽번호도 같이 사라진다 (llncs 에서 쪽번호는 머리말 줄에 붙는다).

## 완료 조건

- 두 판 모두 쪽 위 머리말·쪽번호 없음, 각 2회 빌드에 Overfull hbox 0 · 오류 0.
- 쪽수와 그림 배치 불변 — 영문 20쪽·국문 17쪽, 그림 일곱 건 전부 캡션과 같은 쪽.
