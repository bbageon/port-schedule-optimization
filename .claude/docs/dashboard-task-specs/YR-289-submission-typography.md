# YR-289 — 시드 표기 · 이탤릭 통일 · 통화 단위 · 괄호글 재배치

**상태**: done (2026-08-31)
**Epic**: Paper

## 1. 본문에서 시드 숫자 제거

사용자 물음: *"랜덤 시드 9900950 을 쓰고 있다고 작성했는데 이 수치를 넣을 필요가 있나?"*

**필요 없다고 판정했다.** 논문에 **코드·데이터 공개 문구가 한 줄도 없다**
(`github`·`available`·`repository` 전부 0건). 시드는 코드가 있어야 의미가 생기는
값이라 읽는 사람이 그 일곱 자리로 할 수 있는 일이 없다. 오히려 *"왜 하필 이
시드인가"* 라는 질문을 부른다 — 실현이 하나뿐이기 때문에 더 그렇다.

- 본문 `with random seed 9{,}900{,}950, which was not used in training`
  → `with a random seed held out from training`
- **설정표(Table 1)의 시드 항목은 유지** — 표는 실행 설정을 기록하는 자리다.
- `verify_submission.py` 는 시드 문자열을 대조 대상으로 잡지 않아 41건 그대로 통과.

## 2. 실현이 하나뿐이라는 한계를 결론에 명시

시드 숫자를 빼든 남기든 **결과 전부가 30일 실현 하나에 얹혀 있다**는 사실은
그대로다. 날짜 28개를 독립 단위로 보지만 그건 *같은 세계 안의* 28일이다.
그리고 혼잡 민감도에서 부호가 뒤집히는 것을 이미 확인했다([[YR-286]]).

> They rest on a single 30-day realisation and carry no estimate of variation
> across realisations.

YR-286 의 민감도 소절이 준비되면 이 문장을 실제 수치로 교체한다.

## 3. 이탤릭 통일

사용자가 3쪽의 기울어진 `after` 를 발견했다. PDF 의 **모든 글자를 글꼴별로 훑어**
(`scratchpad/fontaudit.py`) 이탤릭(`F71`)이 두 가지 일을 섞어 하고 있음을 확인했다.

| 쓰임 | 사례 | 판정 |
|---|---|---|
| 용어·이름 표시 | `design` · `least-loaded` | 유지 (관례) |
| 의도적 대비 짝 | `where` / `when` | 유지 (논문 핵심 대비) |
| 결론 구절 | `intervenes selectively when congestion is observed` | 유지 |
| **낱말 하나 강조** | **`after` · `relative to`** | **제거** |

전치사 하나만 기울어 있으면 오식처럼 읽힌다 — 실제로 사용자가 그렇게 읽었다.
문장 뜻은 어순이 이미 나르므로 이탤릭만 뺐다.

**정상인데 달라 보이는 것**: 비용식의 아래첨자(`GT`·`wait`·`move`·`rehandle`·
`vessel`)는 수식 글꼴이라 기울어 있다. 수식이므로 맞다.

## 4. bn / m 풀어쓰기

사용자 물음: *"KRW bn 이라고 쓰던데 bn 이 뭐야?"* — billion(10억)이다. `m` 은
million(100만). 스프링거가 미국식 영어를 선호하고(지침 §4 *"we prefer the use of
American English"*) 그림 축이 이미 `million KRW` 로 풀어 쓰고 있어 본문도 풀어 썼다.
`bn` 10건 → `billion`, `m` 5건 → `million`.

## 5. §5.3 괄호글 재배치

사용자 물음: *"(means over the first and last five iterations, targets in units of
KRW 100,000) 이런 괄호 글은 넣는게 이상하지 않아?"*

**내용은 필요하지만 자리가 틀렸다.** 그 괄호는 ①수치를 어떻게 계산했는지(5회차
평균) ②단위가 무엇인지 — 둘 다 숫자를 읽기 *전에* 알아야 하는 것인데, 숫자 여덟
개가 지나간 뒤에 붙어 있었다. 게다가 **단위는 §3.4 에서 이미 정의**돼 있다
(`and divided by KRW 100{,}000` + 라벨 식). 그래서 앞으로 옮기고 단위는 참조로 바꿨다.

> Losses below are reported in the scaled label units of Sect.~3.4 as means over
> the first and last five of the 30 training iterations. The proposal network's
> Huber loss decreases from …

원고의 다른 긴 괄호글은 열거·수치라 정상이다 (규칙 이름 넷, `sign test p=…`, 과제번호).

## 6. 아키텍처 그림을 관련연구 -> 제안 아키텍처로

사용자 물음: *"Fig 1 이 related research 에 있어야 하는 이유가 있어?"*

**없다.** 실측하니 그림 1이 **3쪽을 통째로** 차지하고 있었고 그 3쪽은 처음부터
끝까지 관련연구였다 (§2 는 2쪽 y=531 에서 시작해 4쪽 y=256 까지, 그림은 3쪽 상단).
남의 연구를 논하는 절 한가운데 우리 구조도가 놓일 근거가 없다.

**원인은 정의 위치였다.** 그림 블록이 서론 맨 끝 —
`\section{Related Work}` 바로 앞 — 에 있었고, `flafter` 때문에 정의 지점보다
앞서 나올 수 없으니 LaTeX 이 갈 수 있는 가장 이른 쪽머리인 3쪽으로 보낸 것이다.

정의 지점을 §3 첫머리로 옮겼다. 이제 부동체가 갈 수 있는 가장 이른 자리도 §3
안쪽이라 그림이 **4쪽 상단**에 놓인다 (§3 은 3쪽 y=445 ~ 6쪽 y=328).
서론의 `Fig.~ef{fig:arch} summarises ...` 는 앞을 가리키는 참조로 유지했다.

**부수 효과**: 관련연구가 2쪽 아래~3쪽 중간으로 압축돼 흐름이 끊기지 않는다.

**그림·표 번호와 순서는 그대로** — Fig.1 4쪽 · Table1 7쪽 · Table2 8쪽 ·
Fig.2 10쪽 · Fig.3 11쪽.

## 검증

12쪽 · A4 595×842pt · Overfull 0 · Underfull 0 · 오류 0 · 미정의 참조 0 ·
인용순서 두 축 통과 · 수치 41/41 일치 · 여백 초과 쪽 0 · escape 이상 없음.

## Evidence

`6a03e1c` (시드·한계) · `1a1e405` (이탤릭·단위·괄호글) · `bd74f4c` (그림 1 이동)
