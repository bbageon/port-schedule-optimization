# YR-084 — 장기 운영화: LLM 감독형 설명·예외 관제 Shadow PoC

- **Epic**: Infra / **Priority**: ⚪ / **등록일**: 2026-07-22
- **범위 정정**: 2026-07-26 — TOS 대체·터미널 통합계획기 구축 제외
- **상태**: **backlog** — 장기 북극성만 박제. **현재 착수하지 않는다.**
- **관련 결정**: [반입·양하 우선과 현실 적재규칙 후순위](../strategy-history/2026-07-26-YR-081-반입-양하-우선-현실적재규칙-후순위.md)

## 배경과 권한 경계

TOS는 연구가 수정·대체하는 대상이 아니다. 작업 생성·최초 블록 배정 등 상위 운영기능은
외부 TOS가 수행하며 연구는 그 결과를 입력으로 받는다. 따라서 독자적인 선석·QC·YT·게이트·
야드 통합계획기를 새로 만들거나 LLM이 TOS 작업을 재배분하는 구조는 채택하지 않는다.

현재 연구의 제어 범위는 블록 내부 실행정책이다. 다중 블록 확장에서는 YR-099가
재배정 가능한 미장치 반입·본선 양하 STORE에 한해 실행 블록을 `KEEP/A→B`로 바꾸는 좁은 결정론적
`TransferResolver`를 검증한다. 이는 TOS 최초 배정 로직과 별개인 후단 실행 최적화다.

LLM은 이 검증된 정책·resolver·상태조회 도구를 감독하는 **읽기·설명·예외 처리 보조자**다.
수치 최적화기, 작업 소유권 resolver, 안전판단기, 장비제어기가 아니다.

## 동결할 목표 구조

```text
외부 TOS: 최초 작업·담당 블록 배정 (연구가 수정하지 않음)
  ↓
BlockPolicy 인스턴스들: 각 블록 내부 YC 실행 최적화
  ↓ 재배정 가능한 GATE_IN·VESSEL_DISCHARGE STORE에 한함
결정론적 TransferResolver(YR-099): KEEP 또는 atomic A→B
  ↓
안전 resolver·ECS/PLC: 물리 가능성·충돌방지·실제 장비제어

LLM 감독 Shadow:
  위 계층들의 상태·결정·예외를 읽고 설명·비교·에스컬레이션
  직접 작업배정·commit·장비명령은 하지 않음
```

## LLM 허용 범위

- 선사 이메일·관제 메모·상충 알림을 **미확정 사건 후보**로 정형화
- TOS 배정 provenance, BlockQ quote, resolver 결과와 근거 조회
- 승인된 `RUN_SCENARIO`, `COMPARE_PLANS`, `EXPLAIN_DECISION` 읽기/분석 도구 호출
- stale quote·반복 rollback·예측오차·블록 과부하를 운영자에게 설명
- 허용된 playbook 제안과 영향 KPI·불확실성 비교
- 근거가 부족하거나 권한 밖이면 기권·운영자 에스컬레이션

## 영구 금지 범위

- TOS 최초 배정·작업속성·비용계수·운영목표 변경
- TOS write API 또는 범용 SQL·셸·쓰기 API 사용
- `TransferResolver`의 KEEP/TRANSFER 결과를 우회하거나 직접 owner 변경
- YC·YT·AGV·QC에 원시 이동·모터 명령 전송
- 안전거리·물리 mask·인터록·위험물 규칙 변경
- 상태 version·TTL·결정론적 검증 없이 계획 확정
- 운영자 승인이나 fallback을 건너뛴 자동 예외조치

## 개방 선결조건

1. YR-014로 단일 블록 정책·목적·비용계약 최종 판정
2. YR-082/083/042로 구조자료·런타임·블록 일반화 범위 확정
3. YR-081 독립 다중 블록 실행과 YR-099 TransferResolver 검증
4. versioned read API, 불변 snapshot, quote TTL, 감사·rollback·rule fallback 계약
5. YR-082 Level 3 운영로그와 Level 4 읽기전용 Shadow 범위 승인
6. TOS·정책·resolver·ECS 사이 권한과 책임표 확정

선결조건이 없으면 착수하지 않는다. 특히 LLM을 먼저 만들고 검증 도구를 나중에 붙이는
순서는 금지한다.

## 비교 질문

```text
A. 검증된 BlockPolicy + TransferResolver
B. A + 규칙 기반 예외 알림·설명기
C. A + LLM 읽기전용 예외 해석·설명기
```

- B→C에서 비정형 사건 해석·원인 설명·운영자 판단시간이 개선되는가
- 오류·근거부족 상황에서 LLM이 올바르게 기권하는가
- C가 우월하지 않으면 LLM은 검색·설명 UI로만 남긴다

LLM 유무가 작업 배정·resolver 물리결과를 임의로 바꾸는 비교는 하지 않는다.

## 수용 기준 초안

- 안전·물리 위반, 무단 쓰기·도구 실행, owner 직접변경 0
- 오래된 상태·허위 장비·불가능 계획은 실행 전 100% 거부
- LLM·네트워크 장애 중에도 TOS·BlockPolicy·resolver·규칙 fallback 계속 운영
- 잘못된 사건분류·기권·운영자 수정·승인 부담·응답기한 초과 별도 보고
- 정상·복합장애·상충정보·데이터 지연·간접 prompt injection 시험
- 실제 쓰기 권한 없이 오프라인 replay → 읽기전용 Shadow 순서만 허용

정량 우월 임계값은 실제 운영자 자료를 확보한 뒤 사전등록하며 지금 임의로 정하지 않는다.

## 선행 학습 범위

- TOS–ECS–FMS–VBS–PCS의 권한·책임 경계
- rolling-horizon 결과 설명과 불확실성 전달
- LLM 도구 최소권한·정형 출력·prompt injection·memory poisoning 방어
- Human-in-the-loop 관제, 승인·반려·수동복귀 UI
- ISA-95/IEC 62264와 OT 안전·신뢰성

## 연구 근거 출발점

- [ISA-95 공식 계층·인터페이스](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard)
- [NIST SP 800-82 Rev.3 — OT 신뢰성·안전·보안](https://csrc.nist.gov/pubs/sp/800/82/r3/final)
- [NIST AI 600-1 — 생성형 AI 중요 의사결정 위험](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- [OWASP Excessive Agency — 권한·자율성 최소화](https://owasp.org/www-project-top-10-for-large-language-model-applications/2_0_vulns/LLM06_ExcessiveAgency.html)

## 범위 밖

- 현 단일 블록 작업순서 변경
- TOS 최초 배정·최초 경매·TOS 쓰기연동
- 선석·QC·YT·게이트를 다시 최적화하는 독자 터미널 계획기
- LLM의 작업 재배정·resolver 대체·실시간 장비제어
- 공개 Level 0~1 자료만으로 실운영 Shadow 또는 자동제어 주장

## 향후 산출물

- 읽기전용 운영화 참조 아키텍처와 계층별 책임·API 계약
- 돌발상황 taxonomy·승인 playbook·오프라인 사건 benchmark
- 권한표·도구 allowlist·감사·fallback 명세
- 규칙 설명기 대비 LLM 설명기 비교 보고서
