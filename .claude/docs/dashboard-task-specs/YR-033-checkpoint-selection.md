# YR-033 — checkpoint 선택 프로토콜 보완

- **Epic**: Exp / **Priority**: 🟡 / **등록일**: 2026-07-15
- **배경**: YR-012-b에서 60~120개 checkpoint 중 validation 30일 최저를 고른 순위와 locked test 순위가 역전됐다. 기존 row가 YR-032를 중복 사용해 본 ID로 바로잡았다.
- **목표(수용 기준)**: 선택용·확인용 validation 분리 또는 표본 확대 계약을 사전등록하고, 알려진 checkpoint 묶음에서 선택 안정도·test rank correlation·불확실성을 재평가한다.
- **범위 밖**: test를 사용한 checkpoint 재선택, 학습기·feature·비용 변경.
- **계획**: 기존 snapshot 재평가 → 후보 프로토콜 비교 → 계산예산과 오선택률 기준으로 하나를 동결.
- **산출물**: 선택 프로토콜 문서·재현 스크립트·winner's-curse 진단표.
