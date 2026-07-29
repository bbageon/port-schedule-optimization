# YR-110 — 하드 guard 기계화
> 상태: done 2026-07-28 · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 문제
계약 0순위(완주 1.0·backlog 0)가 기록만 되고 게이트가 아니었다. yr099-mid 채널 블록은
영구 죽은 코드, 정책 예외는 무계수 삼킴.
## 조치
backlog 수집·policy_exceptions 계수·guard 실패 시 INVALID+종료코드 2(시드 제외 금지 —
검열 회피). **첫 발화가 YR-112 발견의 입구가 됨**.
## Evidence
[report](../../../outputs/reports/yr106b_gates/report.md) · evalkit.check_guards
