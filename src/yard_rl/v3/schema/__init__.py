"""v3 ① 데이터 — 오더 레코드 규격 — 시각 셋을 가른다.

■ 담는 것
  - 오더 6필드: docKey · inOut(반출0/반입1) · copinoNoticeTime ·
    inOutReserveTime · conLoc · conNo
  - 시나리오 파라미터 3종 (정책 비가시): 실현 게이트인 · 진입 주행 · 출문 주행
  - 실행 기록 7종: gateIn · blockIn · serviceStart · jobDone · gateOut
    + prevConLoc · conSwapReason

■ 지금 구현과 다른 점
  현행은 통지·예정·실현 **셋이 같은 값**(`arrival_s`)이라 예약 준수·리드타임
  축이 소멸해 있다. 여기서 셋으로 가른다.

■ 하지 말 것
  파생값(turnTime·리드타임·예약 준수 오차)을 **저장하지 않는다** — 시각에서
  전부 유도된다. 중복 저장은 불일치의 원인.
■ 설계 문서: `.claude/docs/architecture/01-오더-스키마.md`
"""

__all__: list[str] = []
