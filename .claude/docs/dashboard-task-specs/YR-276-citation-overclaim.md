# YR-276 — 인용이 감당 못 하는 주장 둘 (ref1 과장 · DQN 계열 오인)

- **Epic**: Paper / **Priority**: 🟠 / **등록·완료일**: 2026-08-30

전달받은 참고문헌 피드백. **구조 감사는 통과**였다 — 문헌 18편, 인용 키 18개,
고아 문헌 0, 항목 없이 인용된 키 0, 인용 번호도 본문 첫 등장 순서와 일치.
문제는 **인용이 나르지 못하는 주장 두 개**였다.

## ① `ref1`(Yen, DEA-Tobit)을 결론에서 과하게 씀

- 앞판: *"This result supports the view that real-time smart-port information can
  guide assignment actions, not only monitoring~\cite{ref1}."*
- Yen 등은 **항만 설계 속성과 측정된 효율의 상관**을 본 연구다. *"실시간 정보가 배치
  행동을 이끈다"* 는 그 논문이 보인 것이 아니라 **우리 실험이 보인 것**이다.
- 고침: 인용을 빼고 **우리 결과로** 말한다 —
  "In this environment, therefore, real-time congestion information was turned into
  assignment actions and not only into monitoring."
  (국문: "즉 이 환경에서 실시간 혼잡 정보는 모니터링에 그치지 않고 배치 행동으로 이어졌다.")
- §1·§2.1 의 `ref1` 사용은 **그대로 둔다** — 거기서는 "자동화·지능화·환경 설계가 서로
  다른 경로로 작용한다" 만 말하는데 그건 실제 그 논문의 내용이다.

## ② `refmnih`(DQN)을 **같은 계열**로 적음

- 앞판: *"The method therefore approximates action values rather than a policy, as
  deep Q-networks do~\cite{refmnih}, but it forms its target differently…"*
- DQN 은 $r+\gamma\max Q$ 로 **자기 추정을 되먹인다**(부트스트랩). 본 연구는 3시간을
  끝까지 굴려 실제 비용을 재고 출력도 짝 중심 상대값이라 부트스트랩이 없다.
  "as deep Q-networks do" 는 같은 계열이라고 읽힌다.
- 고침: 인용을 **대비 대상**으로만 남긴다 —
  "The method therefore learns a value-like quantity rather than a policy. Unlike deep
  Q-networks, which bootstrap from their own estimates~\cite{refmnih}, the target here
  is a Monte-Carlo cost difference measured over a fixed horizon."
  ([[YR-272]] 에서 한 번 손댄 문장을 한 걸음 더 낮춘 것이다.)

## 검증

- 인용 감사 재실행: 국·영문 모두 문헌 18 · 인용 키 18 · 고아 0 · 항목없음 0.
  `ref1` 인용 3→**2회**(결론에서 빠짐), `refmnih` **1회**(대비 용도).
- 국·영문 각 2회 빌드 — Overfull hbox 0 · 오류 0 · 미정의 참조 0.
- PDF 본문 확인: 새 문장 각 1건, 옛 표현(`as deep Q-networks do`,
  `supports the view that real-time smart-port`) **0건**.
