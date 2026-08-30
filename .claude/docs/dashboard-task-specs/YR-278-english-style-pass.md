# YR-278 — 영문 어체·문체 손질 + 학습 회차 용어 통일 + Overfull 실측 정정

- **Epic**: Paper / **Priority**: 🟠 / **등록·완료일**: 2026-08-30

## ① 표현 다듬기 (전달받은 목록 그대로, 내용 불변)

| 자리 | 앞판 | 고친 뒤 |
|---|---|---|
| 초록 | cannot reflect congestion | cannot **account for** congestion |
| 초록 | It also cost less than | It also **achieved a lower operating cost** than |
| 초록 | is most useful as | **may be more useful** as |
| §1 | turns observed information into operating decisions | **translates** … into **operational** decisions |
| §1 | the working block | the **assigned yard block** |
| §1 | An immediate movement cost | **The immediate cost of reallocation** |
| §1 | Can…? Can…? Does…? Finally, can…? | **(i)~(iv) 한 문장**으로 |
| §3.2 | the freshest state the lead time allows | **the latest feasible point before arrival** |
| §3.2 | which is where the freshness comes from | **a more recent terminal state than at announcement** |
| §3.2 | becomes the accepting party | **the acceptance policy evaluates** the proposed block/slot |
| §3.3 | relative to the action it was rolled out against | relative to **its paired counterfactual action** |
| §3.5 | The alternative world always reverses what the policy actually did | **The counterfactual branch reverses the policy's observed decision** |
| §4.5 | The monetary anchor comes from | **The base monetary scale is derived from** |
| §5.3 | not statistically **clear** | not statistically **significant** |
| §5.3 | which is what a concentrated effect produces | **is consistent with** an effect concentrated in a small number of high-cost days |
| §5.4 | An action that moves waiting time acts on the dominant term | **Arrival-time adjustment acts directly on the dominant waiting-cost component** |
| §5.5 | The temporal action has value by itself | **provides a measurable benefit even without learning** |
| §5.5 | large savings on expensive days / frequent small wins | **larger savings on high-cost days / smaller savings across many days** |
| §5.5 | Both the number of wins and the total reduction | **the frequency of lower-cost outcomes and the aggregate cost reduction** |
| §5.6 | the fit extends to unseen decisions | **generalises to held-out decisions** |

국문은 **의미가 함께 바뀌는 두 곳만** 반영했다 — "분명하지 않았다" → "**통계적으로 유의하지
않았다**", "몰려 있을 때 나타나는 모습" → "**…경우와 부합한다**". 나머지는 영어 표현 문제라
국문 문장은 그대로다.

## ② 학습 회차 용어 통일 — iteration

`30 passes` · `Training epoch` · `5-epoch mean` 이 섞여 있었다. 이 구조는 회차마다 라벨을
새로 만들고 이전 라벨을 버리므로 일반적인 epoch 이 아니다 → **iteration** 으로 통일:
본문 `30 iterations`, 그림 축 `Training iteration`, 범례 `per iteration`·`5-iter. mean`.
⚠️ **`decision epoch`(60초 결정 주기)는 그대로 둔다** — 다른 뜻이다.

## ③ ★Overfull 카운트가 틀렸다 (내 검증 절차의 결함)

지금까지 빌드 확인에 `grep -c 'Overfull \hbox'` 를 썼는데, 셸 따옴표를 지나며 `\h` 가 되어
grep 이 이를 **문자 h** 로 읽었다. 즉 `Overfull hbox` 를 찾았고 로그의 `Overfull \hbox` 와는
영원히 안 맞는다 → **항상 0** 이 나왔다.

`grep -cF 'Overfull'` 로 다시 세니 영문판에 실제로 있었다:

| 빌드 | 영문 | 국문 |
|---|---|---|
| YR-264 | 0 | 0 |
| YR-266(저자·사사 추가 후) | **1** | 0 |
| YR-271(검정·용어) | **3** | 0 |
| YR-273 | **2** | 0 |
| 이번(고치기 전) | **2** | 0 |

두 자리를 고쳐 **0** 으로 만들었다:

- 사사의 `…Commercialization)(RS-2026-25529952)` — 괄호 둘이 붙어 끊을 자리가 없었다(19.2pt).
  괄호를 하나로 합쳤다: `(Technology Transfer and Commercialization, RS-2026-25529952)`.
  국문 사사도 같은 모양으로 맞췄다.
- §5.3 의 `common-random-number method` (2.9pt) → `method of common random numbers`.

앞으로 검증은 `grep -cF 'Overfull'` / `'Underfull'` 로 센다.

## 검증

- 영문 pdflatex·국문 xelatex 각 2회 — **Overfull 0 · Underfull 0 · 오류 0 · 미정의 참조 0**
  (`-F` 로 다시 셈).
- PDF 본문에서 새 표현 23종 확인, 옛 표현 14종 **0건**.
- 그림 일곱 건 캡션과 같은 쪽 유지(영문 21쪽·국문 19쪽), 학습곡선 축이 `Training iteration`.
