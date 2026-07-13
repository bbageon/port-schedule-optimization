# YR-002 요구정보 리스트 — HJNC·DGT 2케이스 (2026-07-13)

> D2 후보를 HJNC(신항 3부두)·DGT(신항 7부두) 2케이스로 좁혔을 때, 프로파일·시나리오·
> 학습검증 각각에 필요한 정보 전수 목록. 근거 조사: [결정자료](YR-002-터미널선정-결정자료.md).
> 상태: ✅ 공개 확보 · 📚 문헌 대체 가능(assumed→literature) · ❓ 미공개(가정 유지) · 🤝 운영사 확인 필수

## 1. TerminalProfile 필드 — 코드가 실제 소비하는 것 (profile_loader 기준)

| 필드 | 현 가정 | HJNC | DGT | 확보 경로 |
|---|---|---|---|---|
| block: row_count | 6 | ✅ 10열 | ✅ 최대 10열 | 공식 장비 페이지 |
| block: tier_max | 5 | ✅ 6단 | ✅ 6단 | 공식 장비 페이지 |
| block: bay_count | 24 | ❓ | ❓ | 전 터미널 미공개 — 문헌 10~40 bay, 위성사진 실측 추정 가능 |
| block: bay_length_m | 6.5 | 📚 | 📚 | 20ft slot+갭 일반치, TEU ground slot 문헌 |
| block: row_width_m | 2.9 | ✅유도 ~3.1 | ❓ | HJNC 레일간격 31m ÷ 10열 근사. DGT 레일간격 미공개 |
| block: tier_height_m | 2.6 | 📚 | 📚 | 컨테이너 표준 높이 + 여유 |
| block: transfer_row | 0 (측면) | ✅ 캔틸레버 측면 차선 | ✅ 캔틸레버, 육/해측 분리 | 둘 다 캔틸레버라 현 모델 호환. DGT 는 육측=외부트럭·해측=AGV 로 방향 고정 |
| crane: gantry_speed_mps | 2.0 | 📚 4~5 | 📚 4~5 | Kalmar ASC 240~300 m/min·PEMA ~5 m/s — **현 값의 2배+, 상향 필수** |
| crane: trolley_speed_mps | 1.0 | 📚 1.0~1.2 | 📚 | Kalmar 60/72 m/min, TBA 1.0 — 현 값 유지 가능 |
| crane: hoist_loaded/empty | 0.5/0.9 | 📚 0.58~0.75 / 1.17~1.5 | 📚 | Kalmar ASC 35/45·70/90 m/min — 소폭 상향 |
| crane: lock/unlock_time_s | 30/20 | 📚 | 📚+🤝 | TBA "트럭 위 정위치 30s 포괄" — 현 3분할(75s)은 무거운 편. DGT 육측 원격조종은 분산 큼(PEMA) → 확률화 검토 |
| crane: truck_positioning_s | 25 | 📚 | 📚+🤝 | 〃 |
| crane: service_bay_min/max | 1~24 | 🤝 | 🤝 | 블록당 2기의 bay 분담 규칙 (분할? 풀링?) — 공개 없음 |
| ops: long_wait_sla_s | 1800 | 📚+🤝 | 📚+🤝 | 제도 근거: 안전운임 대기료 60분 임계 → 30분 warning/60분 violation 2단계화. 내부 SLA 는 운영사 확인 |
| ops: decision_horizon_s | 1800 | 🤝 | 🤝 | TOS 계획창 길이 |
| ops: gate_travel_estimate_s | 600 | 🤝 | 🤝 | 게이트→블록 실주행 2~5분 문헌 — 하향 검토, 실측이 정답 |

## 2. 설계 예정 필드 — 코드 미소비이나 01 §2 가 요구 (수집 시 같이 확보)

| 필드 | HJNC | DGT | 비고 |
|---|---|---|---|
| crane_type | ✅ ARMGC | ✅ ARMG | 공식 확인 |
| can_cross (상호 통과) | 🤝 | 🤝 | 동일 레일 2기면 통과 불가가 일반 — 단정 금지 (01 §2 규칙) |
| safety_distance | 🤝 | 🤝 | Exp-4 핵심 파라미터 |
| acceleration | 📚 | 📚 | Kalmar ASC: gantry 0.40, hoist 0.35 m/s² 공개 — 사다리꼴 속도 프로파일 구현 가능 |
| allowed_job_types (역할 고정) | 🤝 | ✅ 육측=외부트럭 / 해측=AGV | DGT 는 공식 서술 존재. HJNC 2기 분담은 미공개 |
| transfer_points / lane_capacity | 🤝 | 🤝 | 캔틸레버 하부 차선 수 |
| simultaneous_work_rules | 🤝 | 🤝 | 동시작업 금지 규칙 |
| wind_limit | 📚 | 📚 | 제조사 일반치 (운영 임계는 🤝) |
| forbidden_zones / slot_coordinates | 🤝 | 🤝 | 협약 시 레이아웃 도면 |
| vessel_priority_rule | 🤝 | 🤝 | 본선 데드라인·우선 규칙 — CURRENT_RULE 과 함께 확인 |

## 3. 시나리오 캘리브레이션 — 합성 가정(GenParams)을 실측으로 대체할 정보

| 항목 | 현 가정 | 필요한 정보 | 협약 없이 가능한 대체 |
|---|---|---|---|
| 외부트럭 도착률·피크 | 100대/8h, peak 옵션 | 게이트 반출입 시각 로그 🤝 | 📚 크레인당 4~10대/h (Zhao&Goodchild) · 주간 피크 8h 주기 실증(2022 논문) |
| 반입/반출 비율 (gate_out_share) | 0.6 | 반출입 구분 통계 🤝 | ✅ BPA 물동량 통계 부분 대체 |
| 본선 작업량 (n_vessel) | 8건/shift | 선석 스케줄·양적하량 | ✅ 운영사 공개 berth schedule 크롤링 (ETB/ETD·van 수) |
| 장치율 (fill_ratio) | 0.45 | 블록 점유율 시계열 🤝 | ❓ 보도 개략치뿐 |
| 재조작위험 (rehandle_risk) | 0.35 | rehandle_log 🤝 | 📚 Kim(1997) 스택높이·점유율 함수로 모델화 |
| 40ft 비율 (size_mix_ft40) | 0.7 | 컨 규격 분포 | ✅ BPA·ISO 규격 통계 부분 대체 |
| ETA 오차 (eta_error_s) | ±300s 균등 | pre_advice vs 실도착 🤝 | ❓ 없음 — Exp-3 해석의 최대 가정. YR-019 시나리오 매트릭스로 감도 방어 |
| shift 구조 (horizon) | 8h+2h drain | 운영 교대 체계 🤝 | 📚 24h 3교대 일반 관행 |

## 4. 학습·검증용 로그 — 협약 필수 (두 케이스 공통, 01 §6.1·YR-009)

이벤트 타임스탬프 수준이어야 함 (분포·분위수 비교가 YR-009 게이트 요건 — 집계 통계 불가):

1. `job_events` — job_id·flow·actual_arrival·block·bay (도착 재현)
2. `pre_advice` — job_id·provided_eta·provided_at (Exp-3 정보시점)
3. `crane_tasks` — crane_id·job_id·start/end·from/to bay (서비스시간·이동 보정)
4. `container_snapshot` — 초기 장치상태 (block/bay/row/tier)
5. `rehandle_log` — source/dest slot·start/end
6. `equipment_status` — 고장·복구 시각
+ **CURRENT_RULE**: 운영사가 확인한 현행 dispatch 로직 (불명확 시 "TOS 최적화" 명명 금지)
+ **기준기간**: 날짜 단위 train/val/test 분리 가능한 연속 기간 (권장 수개월). **DGT 는 2024.4 개장이라 축적 짧음 — 초기 ramp-up 왜곡 제외 필요**
+ 익명화(비가역 hash)·반출절차 합의 — "A터미널" 관행 준용

## 5. 케이스별 특이 확인사항

**HJNC**: (a) 2기/블록 분담 규칙 🤝 — 단일 YC 모델의 "1기 담당구역" 단순화 타당성 좌우.
(b) 수평배열이라 블록이 안벽과 평행 — 게이트→블록 거리 분포 상이. (c) 유인/원격 운영 비율 🤝.

**DGT**: (a) 육측 인계가 원격조종이라 인계시간 분산 큼 (PEMA 명시) — `truck_positioning` 확률화 필요.
(b) 해측 AGV 연동 — 내부작업 도착과정이 AGV 스케줄 종속 (내부 이송 이력 🤝).
(c) ARMG 46기/AGV 60대 수치는 언론 기준 — 운영사 확인으로 승격 필요. (d) 데이터 축적 기간 짧음 (§4).

## 6. 운영사 요청 패키지 (미팅용, 우선순위순)

1. **샘플 1주치**: crane_tasks + job_events (+ 가능하면 rehandle_log) — YR-009 실현 가능성 즉시 판정
2. **CURRENT_RULE 인터뷰**: dispatch 우선순위·작업분류·본선 규칙 (§2 vessel_priority_rule 포함)
3. **장비 제원표**: 속도·가감속·역할규칙·안전거리 (§1~2 의 🤝 일괄 해소)
4. **pre_advice 스키마**: ETA 필드·제공시점 정의 (§6 체크리스트 5번)
5. **기준기간·익명화·반출절차** 합의 (§4)

— 협약 성사 전 진행 가능분: §1~3 의 ✅·📚 항목만으로 **케이스별 프로파일 v2 초안 2벌**
(`configs/terminals/hjnc_armg.yaml`·`dgt_armg.yaml`, 전 항목 근거 주석) 작성 가능 — YR-022 범위.
