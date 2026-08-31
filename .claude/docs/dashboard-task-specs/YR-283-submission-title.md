# YR-283 — 제출본 제목 재설정 (도메인 제거 → "지연 효과가 있는 실시간 운영")

**상태**: done (2026-08-30)
**Epic**: Paper

## 왜 했나

세 단계로 사용자 요구가 바뀌었다.

1. **길이** — 원 제목이 22단어였다. LNCS 통상 범위(8~15단어)를 넘어 부제를 덜어냈다.
   기여 둘(독립 제안·수락 정책 / 반사실 비용 학습)은 초록 첫 세 문장이 이미 말한다.
   합성환경 평가이므로 "Smart" 는 수식이 과하다고 판단해 뺐다 (keyword 에는 남겨 검색성 유지).
2. **도메인 제거** — 사사 과제(RS-2026-25529952)의 연구개발계획서와 제목이 연결돼야
   제출할 수 있다는 사용자 상황. 계획서 내용은 딥페이크·고령자 체력이고 항만·강화학습
   내용이 **전혀 없다**. 따라서 계획서에 없는 주장(LSTM·시계열 모형)을 넣지 않고,
   실제로 공유되는 성질만 제목에 남기는 방향을 택했다 — "배정을 나중에 바꿀 수 있고
   그 결과가 시간이 지난 뒤 나타나는 실시간 운영".
3. **표현 확정** — 사용자가 `Time-Coupled` 대신 `Real-Time Operations with Delayed
   Effects` 를 골랐다. 뜻이 더 직접적이다 (무엇이 결합돼 있는지가 아니라, 무엇이 늦게
   나타나는지를 말한다).

## 최종 제목

- 제출본: `A Reinforcement Learning Architecture for Spatio-Temporal Reallocation in
  Real-Time Operations with Delayed Effects`
- 작업본(영문·국문)은 중간 단계인 `Time-Coupled Real-Time Operations` 에 머물러 있다
  — 사용자 지시로 마지막 변경은 **제출본에만** 반영했다 (`docs/paper/v3/submission/README.md` 에 명시).

## 제목이 도메인을 벗으면 본문이 책임져야 하는 것 둘

1. **범위** — 어디에 구현·평가했는지를 §1 이 말해야 한다.
   > The architecture is stated for real-time operational systems in which an assignment
   > can be revised while its consequences appear only later, and it is instantiated and
   > evaluated here in a container-terminal environment.
2. **모순 해소** — 제목은 시간 결합을 말하는데 §2 는 *"입력에 시간 순서가 없다"* 고
   적혀 있었다. 환경이 시간적으로 결합돼 있다는 것과 정책 **입력**이 한 후보의 고정 길이
   벡터라는 것은 다른 말이다. 결합은 **라벨**(고정 지평을 앞으로 굴린 결과)이 나른다.
   그 구분을 문장으로 적었다.

## 검증

- 제출본 12쪽 · 작업본 영문 20쪽 · 국문 17쪽 유지
- Overfull 0 · Underfull 0 · 오류 0 · 미정의 참조 0
- `scripts/v3/verify_submission.py` 41/41 일치 (제목 변경은 수치를 건드리지 않는다)

## Evidence

`bb7adbb` (부제 제거) · `299a70a` (작업본 동기화) · `7ad0ba7` (도메인 제거 + 본문 둘)
· `e16a4ab` (제출본 최종 표현)
