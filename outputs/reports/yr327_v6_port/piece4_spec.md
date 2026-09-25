# v6 조각 4 구현 명세 — 본선·이송 — VESSEL_START/STS_MOVE/TRANSFER_ARRIVE/VESSEL_RELEASED/PLAN_CHANGE·이송차 링버퍼·양하 해제 순위·선석 초과 비용

원본: outputs/v6/port_plan_wf_9e7366e6.json (plan.pieces[3])

files_to_port:
- engine.py:948-1065 (본선 6 처리기), 929-942 (완료의 본선 훅), 806-811, 816-817 (STS 대기 적분·요율), 1072-1080 (clearout)
- integrated/transfer.py (request·dispatch_pending·integrate)
- integrated/vessel.py:20-54 (VesselPlan·VesselProcess → (V,) 열)
- engine.py:995-1006 (_release_next_discharge → release_rank 표)
- scenario.py:14-22 (InjectedEvent → (I,) 표)

hard_parts:
- 양하 job 해제 순서 = job_id 문자열 정렬 첫 PLANNED·STORE (1001-1006행) — 스트림 물량 ≥100 이면 사전식≠숫자순이라 release_rank 열이 필수
- 막힌 STS 는 자기 STS_MOVE 를 재예약하지 않고 다른 사건이 clock 에 push (971-974, 1019-1021, 939-942행) — '넣지 않음' 을 time=+inf push 로 표현하되 큐 칸을 안 먹게 push_event 를 'inf 면 무시' 로 확장
- cadence 130.909… 누적 가산(988행) vs release_time = start+k·cadence 곱셈(scenario_gen) 의 마지막 비트 경쟁 — 같은 산술을 그대로

array_technique:
(V,) 열 + alive 마스크; STS 처리기는 act/ok/finish 마스크 사슬; 이송차 free=argmax(busy_until≤now+1e-9), pending 링버퍼 (P,) + head/tail; 요율 = Σ(blocked&~done).

equivalence_test:
fixtures.build_minimal_terminal_scenario (fixtures.py:48-98) 를 크레인 1대로(YC-B 대상 주입은 무시됨 866/870행) 그대로 + 변형 (a) 양하 5 moves·YT 1대·600s 로 버퍼 cap 3 도달(sts_blocked 발생) (b) 적하 5 moves 를 트럭 5대와 섞어 굶김. 비교 = 본선 사건열(VESSEL_START/STS_MOVE/TRANSFER_ARRIVE/VESSEL_RELEASED/PLAN_CHANGE/JOB_RELEASED) 순서·시각 정확(cadence 144 는 f64 비트 일치), 배별 done/remaining/buffer/sts_wait_accum/actual_completion, cost sts_wait/transfer_wait/vessel_delay/depart_delay·kpis.berth_overrun 상대 1e-9, transfer busy_until·pending 길이.

예상 규모: ~350줄

의존: ['1']

## 관련 event_kinds (설계)
- 3 TRANSFER_ARRIVE (2) · 선박 번호 (조각 4) · 양하: buffer=max(0,b−1), 그 배의 PLANNED·STORE 작업 중 release_rank 최소를 VESSEL_RELEASED 로 clock 에 push; 적하: buffer+1; 대기 요청 재배차(링버퍼 head); 막힌 STS 재개 가능하면 STS_MOVE 를 clock 에 push(불필요 push 는 time=+inf 로 무시); yielded[:]=F (engine.py:1008-1022)
- 4 STS_MOVE (2) · 선박 번호 (조각 4) · act=started&~done; ok=where(work==DISCH, buffer<cap, buffer>0); blocked_since=where(act&~ok&blocked==inf, clock, where(act&ok, inf, old)); buffer += where(act&ok, ±1, 0); remaining −= act&ok; finish=act&ok&remaining≤0 → done, actual_completion, vessel_delay/depart_delay accrue; 아니면 STS_MOVE 를 clock+cadence 에 push; 양하면 transfer_request (engine.py:959-1039)
- 7 VESSEL_RELEASED (3) · 오더 번호 (조각 4) · status[n]=RELEASED; yielded[:]=F — 같은 시각 TRANSFER_ARRIVE(2) 뒤에 오는 것은 큐 키가 보장 (engine.py:877-881)
- 8 VESSEL_START (4) · 선박 번호 (조각 4) · do=~started[v]; started|=do; remaining=where(do,total,old); STS_MOVE 를 where(do, clock+cadence, +inf) 에 push (engine.py:948-957)
- 9 PLAN_CHANGE (5) · 선박 번호 + 주입 사건 행 번호(aux 표) (조각 4) · 주입 표 (I,) 에서 completion/basis/etd 를 NaN 아니면 덮어쓰기; (I2,) 작업마감 쌍을 마스크 산포 deadline.at[job].set (engine.py:1048-1065)