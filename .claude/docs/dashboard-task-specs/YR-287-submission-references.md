# YR-287 — 제출본 레퍼런스·조판 정비 (A4 · 인용순 · 서지 보강)

**상태**: done (2026-08-31)
**Epic**: Paper

## 왜 했나

사용자가 전달한 레퍼런스 피드백 7건을 검토했다. 문서만 읽고 받아들이지 않고
실제 PDF·`.tex`·템플릿 스타일 파일(`splncs04.bst`)로 대조해 판정했다.

## 판정 — 맞는 지적 5, 틀린 지적 1, 처방 과잉 1

| 지적 | 판정 | 근거 |
|---|---|---|
| 레퍼런스 누락 없음 | ✅ 맞다 | 18/18 인용 · 미인용 0 · 목록 밖 인용 0 |
| 용지가 US Letter | ✅ 맞다 | 실측 612×792pt. `a4paper` 시험빌드 = 12쪽 유지 |
| **저자 봉투 기호 깨짐** | ❌ **틀렸다** | 600dpi 렌더 결과 ✉ 정상. `ChoiB` 는 **글자 추출** 현상 (MarVoSym 이 봉투를 `B` 자리에 둔다). 제목의 `E<esc>ects` 도 같은 부류(`ff` 합자) |
| 참고문헌 정렬 | ⚠️ 문제는 맞고 **처방이 과했다** | 지침 §2.9: *"either by order of citation or by alphabetical order"* — 둘 다 허용. 우리는 등장순인데 **3자리만** 어긋났다. 알파벳(splncs04)으로 가면 18개 전부 이동 |
| [10] Foerster 쪽수 | ✅ 맞다 | AAAI-18 pp.2974–2982 (dblp 확인) |
| [11] Kourounioti 쪽수 | ✅ 맞다 | EJTIR 18(1), 76–90 (저널 페이지 확인) |
| [15][16] `(in Korean)` | ✅ **지침 요구사항** | *"If the title … is, e.g., in Russian or Chinese, then please write (in Russian)…"* |

**피드백이 놓친 것**: 설정표 캡션이 *design 표시 없는 값은 인용문헌 범위 안*이라고
하는데 블록 수는 문헌 20 vs 우리 21 이었다.

## 한 일

1. `\documentclass[a4paper]{llncs}` — **`runningheads` 는 넣지 않았다**. 피드백은
   같이 넣으라 했지만 지침 §2.4 가 *"There is no need to include page numbers or
   running heads; this will be done at our end"* 라고 못 박는다.
2. 참고문헌을 등장순으로 정렬 — Huber 를 14번으로, 두 부처 고시 순서 교체.
3. 서지 보강 4건 — Foerster 저자 5명 전부 + 쪽수 · Kourounioti 쪽수 ·
   Schwientek 학회명(ASIM SST 2020) · 두 고시에 `(in Korean)`.
4. 학회 항목 구두점을 `splncs04` 출력과 동일하게 (booktitle 뒤 마침표, `vol.~59`).
5. `20 blocks` → `a yard of roughly twenty blocks`.
6. §5.2 대역 특정 주장 제거 ([[YR-286]] 의 앞부분).

**[14] 김선용 교수 논문(ref20)은 사용자 지시로 유지** — 번호만 [15]로 밀렸다.

## ★ 쪽수 예산이 진짜 제약이었다

보강 전 12쪽 마지막 쪽 여유는 **0.9줄**이었다. 보강을 넣자 **13쪽**이 됐다.
11쪽이 37pt 비고 사사 블록(5줄)이 통째로 12쪽으로 밀린 구조여서, 결론의
중복 문장 **둘**을 덜어 사사를 11쪽으로 끌어올리자 12쪽으로 돌아왔다.

지운 두 문장(둘 다 앞 문장의 되풀이):
- *"Real-time congestion information was therefore turned into assignment actions
  and not only into monitoring."*
- *"These extensions retain the observation--proposal--acceptance--commitment--
  execution structure proposed here while defining the conditions under which
  spatial and temporal reallocation are most effective."*

## 검증

12쪽 · A4 595×842pt · Overfull 0 · Underfull 0 · 오류 0 · 미정의 참조 0 ·
인용순서 두 축 통과(`scripts/v3` 외 scratchpad 스크립트) · 18/18 인용 ·
수치 41/41 일치 · 여백 16% 초과 쪽 0 · escape 전수검사 이상 없음.

## Evidence

`fd39eb7`
