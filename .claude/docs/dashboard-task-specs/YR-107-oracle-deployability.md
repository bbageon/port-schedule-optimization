# YR-107 — 오라클 오표기 정정 + 배포 자격 규칙
> 상태: done 2026-07-28 · 측정계약 축
> spec 소급 작성 2026-07-29 (사용자 지시: 모든 row 는 spec 필수). 세대 표기는 [YR-099 경계](../strategy-history/2026-07-29-아키텍처-세대구분-정정-YR-099경계.md) 기준.

## 발견
JR1800 은 deepcopy 사본의 진짜 도착을 소비하는 오라클인데 "공개정보 참고군"으로 표기·
잠금평가됨. 더 나쁜 것: 학습 arm 전원 healthy 탈락 시 규칙이 오라클을 **자동 후보 지명**.
## 조치
`uses_future_information`/`is_deployable` — 오라클은 전 게이트 통과해도 채택 불가.
원자료마다 정보등급 박제(사람 라벨은 이미 1회 유실). FIFO 약누출은 표기만.
## Evidence
[report](../../../outputs/reports/yr106b_gates/report.md) · [tests](../../../tests/integrated/test_yr106b_gates.py)
