# 연구 소개 발표 자료 (2026-09-22)

슬라이드 **열두 장**을 장당 PNG 한 개로 만든다. 발표 프로그램에 그림으로 끌어다
놓으면 된다. 파일 형식은 1920×1080, 16대 9다.

발표에 무슨 말을 올릴지는 [`내용/`](내용/README.md) 에 있다 — 장마다 *슬라이드에 넣을 글 ·
말로 덧붙일 것 · 숫자와 출처 · 나올 만한 질문* 네 칸으로 적어 두었다.

## 만들기

```
cd docs/presentation/2026-09-22-연구소개
python build_all.py          # 전부 다시 그리고 모아보기까지
python build_all.py --sheet  # 이미 있는 PNG 로 모아보기만
python slide_07_result.py    # 한 장만 다시 그리기
```

Windows 쪽 `python` 을 쓴다. 한글 폰트(맑은 고딕)가 거기 있다. 리눅스에서는 안 된다.

## 파일

| 파일 | 슬라이드 | 무엇을 말하는가 |
|---|---|---|
| `deckstyle.py` | (공통) | 색·글꼴·상자·화살표·줄바꿈. **여기만 고치면 열두 장이 같이 바뀐다** |
| `slide_01_cover.py` | 1 | 표지 |
| `slide_02_summary.py` | 2 | 한 장 요약 — 무엇을 하고 무엇을 알아냈고 지금 무엇을 하는가 |
| `slide_03_problem.py` | 3 | 트럭이 올 시각과 갈 블록은 미리 정해진다 |
| `slide_04_idea.py` | 4 | 그 배정을 다시 조정한다 (시간만 / 블록까지) |
| `slide_05_method.py` | 5 | 제안하는 정책과 받아들이는 정책, 그리고 두 번 굴려 값을 매기는 법 |
| `slide_06_experiment.py` | 6 | 스무 달 × 정책 넷 = 여든 번 |
| `slide_07_result.py` | 7 | 시간 조정은 재현, 공간 조정은 재현 안 됨 |
| `slide_08_stall.py` | 8 | 그런데 아홉 번이 멈췄다 |
| `slide_09_budget.py` | 9 | **원인 — 후보 열두 칸을 못 하는 일이 다 차지한다** |
| `slide_10_evidence.py` | 10 | 고친 쪽은 비용이 터지던 날을 평범하게 넘겼다 |
| `slide_11_strategy.py` | 11 | 정작 큰 구멍은 규칙과 견준 적이 없다는 것 |
| `slide_12_status.py` | 12 | 끝난 것 · 도는 것 · 남은 것 |

`slides/contact-sheet.png` 은 열두 장을 한 장에 모은 것이다. 흐름을 한눈에 볼 때 쓴다.

## 쓸 때 지키는 것

- **한 장에 메시지 하나.** 제목이 곧 결론이고 나머지는 근거다.
- **숫자는 장당 두세 개**만 크게. 표를 길게 넣지 않는다.
- **한계를 각주로 같이 둔다.** 예: "시드 하나·블록 하나 진단이며 정책이 더 낫다는
  증거가 아니다."
- 숫자를 새로 쓰려면 **측정한 값만** 쓴다. 출처는 각 파일 머리말에 적는다.

## 숫자의 출처

| 숫자 | 어디서 |
|---|---|
| 스무 달 중 16 / 4 / 3, 절감 중앙 12.7% | `outputs/reports/yr317_v3_independent_eval/run-7e2fb14/summary.json` |
| 멈춘 실행 9건·6시드·최장 23.5일 | `outputs/reports/yr317_v3_stall_diagnosis/scan-80.json` |
| 후보 174건·탈출 3건·예산 12칸 | `.../legacy-probe-21000000-Y17-b/` |
| 3일째·7일째 비용, 밀린 일 1,941 vs 40 | `.../fix-validation/`, `.../fix-validation-9d/` |
| 시험 422건 통과 | `.../tests-final.xml` |

자세한 이야기는 [정지 결함 진단](../../paper/v3/paper-revision/31-정지-결함-진단.md)과
[확증 캠페인 사전등록](../../../outputs/reports/yr318_confirmatory/prereg.md)에 있다.
