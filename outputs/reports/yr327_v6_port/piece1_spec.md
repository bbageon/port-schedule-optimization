# v6 조각 1 구현 명세 — 워크플로 wf_9e7366e6 설계 결과 발췌

원본: outputs/v6/port_plan_wf_9e7366e6.json (plan.*). 줄 번호는 src/yard_rl/v6/world/... 기준.

## step_structure

한 스텝 = v5 `run_until_decision` 의 한 순회(engine.py:279-336). 네 국면을 마스크로 고르며, 서로 배타적이다.

  nt = min(queue.time)                          # 다음 사건 시각(비면 +inf)
  alive = nt < inf ;  in_window = nt ≤ end+EPS  # 평가창 밖 사건은 없는 것으로(287행)
  due_now = alive & in_window & (nt ≤ clock+EPS)  # 동시각 사건이 남아 있나

  [국면 W: 깨우기 — 조각 3]  ~due_now 이고 wake_s[idx] ≤ clock+EPS 면 소비(armed·yield 해제·로그) → 이 스텝 종료(continue 와 같음, 291-292행)
  [국면 D: 결정]   open_k = idle_k & ~yielded_k & (any_n cand[k,n] | (armed_k & eta_opp_k))   (453-464행)
                   decide = ~due_now & ~wake_fired & any(open) & (clock < end−EPS)
                   decide 면: pending=open; 정책 호출; 배정 적용(아래); last_decision_at=clock; refresh_rates → 스텝 종료 (사건을 꺼내지 않는다)
  [국면 X: 탈출 — 조각 2] ~decide & 교착 술어 → 유휴 전원 결정 개방 (302-305, 384-409행)
  [국면 F: 종료]   fin = ~decide & ~alive_in_window & ~wake_pending → advance(max(clock,end)); wait_sample 검열(end−block_in, waiting 인 것); closed_end=end; terminal=T (1068-1084행)
  [국면 E: 사건]   pop = ~decide & ~fin & alive & in_window → (q', t, kind, tgt) = next_event(q); advance(t); 로그(t,kind,tgt); 종류별 처리기 12개를 **전부** 계산해 `where(kind==i, h_i(world), world)` 로 고른다; refresh_rates (776-782행)

due_now 인 사건도 국면 E 와 같은 경로(advance 는 같은 시각이라 무동작)라 실제 구현은 E·D·F 세 마스크만 있으면 된다: decide 우선, 아니면 pop, 아니면 fin. v5 가 결정을 "동시각 사건을 다 소진한 뒤"에만 여는 규칙(288-299행)이 `~due_now` 한 항으로 재현된다.

결정이 스텝 안에서 일어나는 방식 (decide 국면):
  1) idle/eligible (K,) → dispatchable (K,N) → 계획 (K,N): plan_serve 를 크레인×오더로 vmap → plan_ok, dur, rehandles, corridor, slots, lane → can_reserve (K,N) → cand = eligible & dispatchable & ~taken & plan_ok & can_reserve  (467-492행의 마스크판)
  2) x (K,N,9) 특징 → `choose_batch(params, x, cand)` (gpu/policy.py:108) → pick (K,). 시험 정책은 같은 서명의 `first_by_id = argmax(cand[k])`.
  3) 배정 적용은 **크레인 정렬 순 lax.scan** (K 단계, carry=예약표): 앞 크레인이 잡은 예약이 뒤 크레인의 can_reserve 에 반영된다 — v5 `commit_decisions` 가 crane_id 정렬 순으로 assign→reserve 하는 것(733-736행)과 ReferenceDispatcher 의 순차 재계산(dispatcher.py:19-32)에 대응. K=1 이면 단계 하나.
  4) 각 단계: act=SERVE 면 reserve(res_*[k], token_owner[n]=k), plan[k]=계획, assigned[k]=n, status WORKING, available_at=clock+dur, is_loaded, 오더 status RUNNING·assigned_crane·service_s=clock·waiting F·wait_sample=clock−block_in, cost.pending[crane_travel/empty_travel/rehandle] += loaded_m/empty_m/rehandles, push(JOB_COMPLETED, clock+dur, k), 로그 DISPATCH; act=WAIT 면 yielded[k]=T (679-722행). reserve 가 거절하면 violation|=16 (v5 는 예외).
  5) refresh_rates: rate=[Σ(blocked&~done), transfer_waiting, lane_mean, Σyielded, imbalance/shift_len] (815-838행).

전체 실행: `world, _ = lax.scan(step, world0, None, length=S_max)` 에 `terminal` 이면 스텝을 항등으로. 여러 세계는 `vmap(step)` — 세계마다 멈추는 국면이 달라도 마스크라 무방. 단일 세계·CPU 동등성 모드에서는 decide 국면을 `lax.cond` 로 감싸 계획 계산을 건너뛴다(vmap 아래서는 select 로 풀림).


## decision_interface

■ 입력 — 크레인 k 마다 오더 행렬 x[k] (N,9) f32 와 후보 마스크 cand[k] (N,) bool. gpu/policy.py `choose(p, x, mask)` (86-89행) 서명 그대로, 크레인 축은 `choose_batch` (108행) 로 vmap.
  특징 9칸 (크레인 k·오더 n 의 값 — 계획 결과와 오더 상태에서만; 실현 미래(actual_*)는 절대 안 읽는다):
   f0 누적 대기 = max(0, clock−block_in_s[n])/3600 (외부트럭·도착분만, engine.py:258-265 cum_wait)
   f1 계획 소요 = plan_dur[k,n]/3600
   f2 재조작 수 = plan_rehandles[k,n] (0..T−1)
   f3 빈 주행 = plan_empty_m[k,n]/(bay_count·bay_len)
   f4 신규 적재인가(STORE) 0/1
   f5 본선연계인가 0/1
   f6 마감 여유 = clip((deadline−clock)/3600, −2, 24), 없으면 0
   f7 SLA 임박(mandatory) = 외부 & 누적대기 ≥ 0.8·sla 0/1 (candidates 의 _is_mandatory 규칙)
   f8 ETA 간격 = clip((provided_eta−clock)/3600, −2, 24), 없으면 0 (PRE_ADVICE 에서만 유한)
  ★연구 설계 원칙 2(핵심 정보 우선): 첫 학습은 f0~f3 만 채우고 나머지는 0 으로 두었다가 실증 후 한 칸씩 켠다. 동등성 시험은 특징을 안 쓴다(규칙 정책).
  cand[k,n] = idle_k & ~yielded_k & ~down_k & dispatchable[k,n] & token_owner[n]<0 & plan_ok[k,n] & can_reserve[k,n]. WAIT 는 별도 칸이 없다 — 마스크가 전부 False 인 크레인은 결정 대상이 아니고(v5 도 후보 없는 크레인은 안 묻는다, 460행), 정책이 명시적 WAIT 를 낼 수 있게 하려면 열 N(가상 오더 'WAIT', 특징 0·f4~f8 0) 을 더한 (N+1) 행렬로 확장한다 — 조각 7 에서 결정.

■ 출력 — pick (K,) int32: 고른 오더 번호(−1 = WAIT). 스텝은 이를 act_kind (SERVE/WAIT) 와 act_job 으로 바꿔 배정 scan 에 넘긴다. PRE_REHANDLE/REPOSITION 은 조각 3 에서 후보 종류 열(kind)과 목표 bay 열을 (K, N+R) 로 붙인다.

■ 학습 신호 — 같은 순전파에서 q_values (67-83행) 의 Q,V,A 와 counterfactual_advantage (92-103행) 을 얻어 (K,) 로 기록; 보상은 세계 비용 누적(cost.episode / Φ) 의 결정 간 차분. 이 배열들은 스텝의 carry 에 (S_max, K, …) 로 쌓이거나 결정 순번 카운터로 고정 로그에 적힌다.

■ 결정 시점이 사건 루프 안에서 잡히는 방식 — 사건 하나를 처리한 다음 스텝의 머리에서 "동시각 사건이 더 없다(~due_now) & 유휴·비양보 크레인에 후보가 있다 & clock<end−EPS" 이면 그 스텝은 사건을 꺼내지 않고 결정만 한다(v5 engine.py:288-299 와 같은 순서). 정책망 호출(choose_batch)은 그 스텝 안의 순수 함수이고, 결정이 없는 스텝에서는 마스크로 무효화된다(cond 로 건너뜀). 결정 시각은 엄격 증가한다 — 같은 시각의 재질문은 yielded(WAIT 크레인 제외)와 바쁜 크레인 제외로 구조적으로 막히고, 탈출(조각 2)은 last_decision_at 으로 막는다(395-397행).


## array_layout

- **queue.time / kind / target / seq (+ counter, overflow)** (Q,) ×4, () ×2 time f64(x64 모드; 학습은 f32) · int32 ×3 — 미래 사건. ★꺼내기 키는 (time, PRIO[kind], seq) 3단 사전식 — PRIO 는 kind→우선순위 상수표 [0,1,1,2,2,3,3,3,4,5,6,7] 로 열 추가 없이 kind 에서 파생 · 빈칸=time=+inf, kind/target/seq=-1 (gpu/events.py:33-34) · 출처=integrated/events.py:14-26 (EventKind), 30-43 (_PRIORITY), 46-50·61-65 (힙 키·seq) ↔ gpu/events.py:37-60 (틀 유지), 82-88 (키 교체 대상)
- **clock, end_s, terminal, last_decision_at, escape_at, escape_count, steps** () 각각 f64, f64, bool, f64, f64, int32, int32 — 공용 시계·평가창 끝(horizon+drain)·끝났나·결정 시각 엄격증가 표식·같은 시각 탈출 재발화 방지·탈출 횟수·소비한 스텝 수 · 빈칸=last_decision_at/escape_at = -inf (v5 None) · 출처=engine.py:143-145, 181-184, 403-405; scenario.py:37-38
- **orders.status** (N,) int32 — 엔진 작업 상태 — JobStatus 선언 순 PLANNED0 RELEASED1 WAITING2 ASSIGNED3 RUNNING4 DONE5 CANCELLED6. ★gpu/state.py 의 stage(라이프사이클 6단, 학습·시장용)와 별개 열로 둔다 · 빈칸=없는 오더는 block=-1 로 판별 · 출처=enums.py:48-55; engine.py:849, 855, 707, 919; gpu/state.py:31,42
- **orders.flow / is_external / is_vessel / is_store** (N,) ×4 int32, bool, bool, bool — JobFlow 선언 순(GATE_IN0 GATE_OUT1 VESSEL_LOAD2 VESSEL_DISCHARGE3 TRANSSHIPMENT4 REHANDLE5) · 외부트럭 · 본선연계 · STORE(=inbound_size 있음) · 빈칸=flow=-1 · 출처=enums.py:10-16; models.py:58-75
- **orders.target_cont / inbound_cont / inbound_size / vessel / assigned_crane / rehandles** (N,) ×6 int32 — 반출 대상 컨테이너 번호 · 반입 시 낳을 컨테이너 번호(=C0+n, reset 에 고정) · 규격(FT20 0/FT40 1/FT45 2) · 소속 선박 · 맡은 크레인 · 재조작 횟수 · 빈칸=-1 · 출처=models.py:46-56; engine.py:604-606 (IN_{job_id} 동적 생성을 사전 배정으로 대체), 708, 921
- **orders.release_s / provided_eta_s / deadline_s / exit_travel_s / actual_arrival_s** (N,) ×5 f64 — 해제 시각 · 제공 ETA · 마감 · 완료→출문 소요 · 실현 블록도착(시드용 진실값 — 정책 입력 금지) · 빈칸=+inf (v5 None); exit_travel 은 -1=None(장부 모드 판별) · 출처=models.py:32-44; engine.py:127-133 (exit_travel 유무로 TimeLedger 활성), 229
- **orders.gate_in_s / block_in_s / service_s / done_s / gate_out_s** (N,) ×5 f64 — 트럭 시각 다섯 A/B/S/C/O — 이미 gpu/state.py OrderArrays 에 있음. ★단일블록 엔진에서 gate_in_s 는 reset 에 시나리오 값으로 채운다(v5 가 reset 에 등록·정렬) · 빈칸=+inf · 출처=gpu/state.py:44-49; time_contract.py:24-30, 52-58, 60-74; engine.py:132, 852, 713, 920, 927-928
- **orders.waiting / in_block / wait_sample_s** (N,) ×3 bool, bool, f64 — S−B 대기 중(kpis._waiting) · B≤t<C 블록 점유 중(ledger._in_block) · 대기 표본(서비스 시작 또는 검열 시 확정) · 빈칸=wait_sample=NaN · 출처=kpis.py:45-53, 58-63, 71-80; time_contract.py:44, 60-63, 72
- **cranes.assigned / status / available_at / bay / row / bay_min / bay_max** (K,) ×7 int32, int32, f64, f64, f64, int32, int32 — 이미 gpu/state.py CraneArrays. status 는 CR_IDLE0/CR_WORKING2 만 씀(v5 IDLE·HANDLING). bay/row 는 연속 좌표 · 빈칸=assigned=-1 · 출처=gpu/state.py:58-68; models.py:85-96; engine.py:699-701, 902-906
- **cranes.down / down_pending / yielded / yield_count / completions / served / is_loaded / loaded_m / empty_m** (K,) ×9 bool ×3, int32 ×3, bool, f64 ×2 — YcRuntime 확장 필드 — 고장·완료후고장예약·양보(WAIT 후 다음 상태변경까지 결정 제외)·경합패배 수·완료 수·서비스 수·적재 중·누적 주행 · 빈칸=없음(전부 값) · 출처=cranes.py:14-33; engine.py:686-688, 840-842, 907-910, 943-944, 1041-1046
- **cranes.spec_gantry / spec_trolley / spec_hoist_loaded / spec_hoist_empty / spec_lock / spec_unlock / spec_truck_pos** (K,) ×7 f64 — 정적 스펙 — 이동시간 10항의 분모·상수 · 빈칸=없음 · 출처=models.py:121-132; travel_time.py:56-67
- **rail_order / idle_pos** (K,) ×2 int32, f64 — 레일 물리 순서(초기 bay 오름차순, 동률 id) 순열 — 호스트 reset 1회 · 예약 없는 크레인 현재 bay 장벽(YR-091) · 빈칸=없음 · 출처=engine.py:105-118; reservation.py:40-42, 70-84, 903
- **stacks.grid** (B,R,T) int32 — 칸(bay,row)·단(tier)별 컨테이너 번호. 0-based 색인, 값은 컨테이너 번호 · 빈칸=-1 · 출처=stack.py:17-26 (_stacks dict→list), 50-70 (place/remove)
- **stacks.height / top_size** (B,R) ×2 int32 — 쌓인 단 수(=len(pile)) · 맨 위 규격(캐시: c_size[grid[b,r,height-1]], 빈 칸 -1) · 빈칸=height 0 / top_size -1 · 출처=stack.py:32-33, 42-47, 156-163
- **conts.c_bay / c_row / c_tier / c_size / c_avail / c_alive** (C,) ×6 int32 ×4, bool ×2 — 컨테이너 표 (1-based 좌표, v5 Container 필드). C = 초기 수 + N(반입 예비칸). 번호 = sorted(container_id) 순위, 반입은 C0+n · 빈칸=좌표 -1, alive False · 출처=models.py:14-24; stack.py:16, 68-69; engine.py:604-606
- **res.active / token / lo / hi / lane / release_at** (K,) ×6 bool, int32, f64, f64, int32, f64 — 크레인별 예약 한 건(Reservation) — 작업 토큰(=오더 번호)·통로 [lo,hi]·레인·해제 시각 · 빈칸=active False, token/lane -1 · 출처=reservation.py:25-32, 35-42, 101-117; engine.py:672-676
- **res.slots / token_owner** (K,B,R) · (N,) bool · int32 — 크레인별 예약 칸 마스크(frozenset slots) · 토큰→크레인 역표(_tokens) · 빈칸=False / -1 · 출처=reservation.py:39, 47-54, 97, 107-108
- **plan.kind / job / lo / hi / dur / end_bay / end_row / rehandles / loaded_m / empty_m / n_moves / start_s** (K,) ×12 int32, int32, f64, f64, f64, f64, f64, int32, f64, f64, int32, f64 — 크레인이 지금 실행 중인 JobPlan(_active_plans) — kind SERVE0 PRE1 REPO2 · 빈칸=kind -1 · 출처=jobplan.py:45-61; engine.py:156, 698, 888
- **plan.mv_cont / mv_src (3) / mv_dst (3) / mv_kind** (K,M) · (K,M,3) ×2 · (K,M) int32 — 완료 시 순서대로 실현할 이동 목록(Move). mv_kind 0=재배치(remove+place) 1=반출(remove) 2=반입(place). M = tier_max (blocker ≤ T−1 + 대상 1) · 빈칸=cont -1 / k ≥ n_moves 무효 · 출처=jobplan.py:15-27; engine.py:607-608, 638-639, 654-655, 892-900
- **kpi.queue_area / tail_area / loaded_m / empty_m / rehandles / completed_ext / completed_ves / berth_overrun** () ×8 f64 ×4, int32 ×3, f64 — KpiTracker 누적 · 빈칸=0 · 출처=kpis.py:27-45, 83-101
- **ledger.block_area / block_tail / terminal_area / closed_end** () ×4 f64 — TimeLedger 적분 셋 — 포인터·힙·정렬 리스트는 닫힌 식으로 대체(아래 advance 식) · 빈칸=closed_end=+inf(None) · 출처=time_contract.py:39-49, 77-109
- **cost.rate / pending / episode** (5,) · (13,) · (13,) f64 — 13항 raw 비용(COST_TERMS 순, 호스트가 schema 에서 순서를 상수로 굽는다) — rate 5항은 advance 에서 rate×dt · 빈칸=0 · 출처=cost.py:15-19, 43-91; engine.py:714-719, 797-812, 815-822
- **lane.n_lanes / lane_adj** () · (L,L) int32 · bool — 레인 수(lane = (bay−1) mod L)·인접 — 혼잡률 rate 계산용 · 빈칸=L=0 이면 lane=-1 · 출처=engine.py:271-273, 807-808, 819-820; lane.py
- **pending / answered / act_kind / act_job / act_bay** (K,) ×5 bool, bool, int32, int32, f64 — 열린 결정의 대상·답 — act_kind SERVE0 PRE1 REPO2 WAIT3 · 빈칸=act_job -1, act_bay NaN · 출처=engine.py:146, 155, 295-299, 679-736
- **eta_wake_s / eta_wake_job / wake_idx / eta_armed / defer_wake_s / defer_n / review_s / review_idx** (W,) (W,) () (K,) (D,) () (Rv,) () f64, int32, int32, bool, f64, int32, f64, int32 — PRE_ADVICE 깨우기(정렬 고정+포인터)·armed·유한 DEFER 예약·외부 조정자 검토 시각. 조각 1 에서는 전부 빈 배열(BLOCK_ARRIVAL 수준) · 빈칸=+inf 채움 · 출처=engine.py:166-179, 339-382, 314-322
- **log.t / log.kind / log.target / log_n** (E,) ×3 · () f64, int32, int32, int32 — 사건 흐름 기록(event_log). kind 는 큐 종류 0..11 + 로그 전용 12 DISPATCH 13 ETA_WAKE 14 DEFER_WAKE 15 DEADLOCK_ESCAPE. 해시는 호스트 · 빈칸=kind -1; 넘치면 overflow++ · 출처=engine.py:157, 370, 377, 408, 721, 845, 1147-1149
- **violation / overflow** () ×2 int32 — 위반 비트합(1 NOT_TOP 2 TIER 4 SIZE 8 PLAN_POSTCOND 16 RESERVE_REJECT 32 NEG_COST 64 TIME_BACKWARD 128 COMPLETE_NO_PLAN 256 STEPS_EXHAUSTED 512 DECISION_COVERAGE) · 칸 부족 수. 0 이 아니면 그 세계는 실격 · 빈칸=0 · 출처=stack.py:55, 64, 66; engine.py:636, 681-696, 785-786, 891; cost.py:70-71; gpu/state.py:87-88
- **policy 입력 x / mask** (K, N, 9) · (K, N) f32 · bool — 크레인별 오더 특징 행렬(ORDER_FEATURES=9)과 후보 마스크 — gpu/policy.py choose 에 크레인마다 vmap · 빈칸=mask False 칸은 점수 −inf · 출처=gpu/policy.py:37, 67-89, 106-108

## capacities

- N (블록 안 오더 칸) = 조각 1 시험 8 (오더 5) · 단일 블록 일반 1,024 · 터미널 전체 16,384 — H21 하루 7,500대/21블록 ≈ 360 + 본선 작업 ≤ 700 → 1,024 여유; 터미널은 gpu/state.py:17 의 하루 최대 15,000 상한을 2의 거듭제곱으로. 넘치면 overflow
- K (크레인) = 조각 1 시험 1 · 일반 2 — profiles H21 블록당 2기, fixtures.py:30 도 2기. 크레인 번호 = crane_id 문자열 정렬 순위(cranes.py:53-54)
- B×R×T (격자) = 시험 10×4×4 · fixture 40×4×4 · H21 24×10×6 — BlockGeometry 를 컴파일 상수로 굽는다(모양이 곧 배열 모양). find_slot 은 B·R 칸을 매번 전부 계산
- C (컨테이너 칸) = 초기 컨테이너 수 + N (시험 3+5=8; H21 최대 1,440+1,024) — 반입 오더 n 의 컨테이너 번호를 C0+n 으로 미리 잡아 engine.py:604 의 동적 생성을 없앤다. 초기 수 ≤ B·R·T
- M (계획당 이동 칸) = tier_max (시험 4, H21 6) — blocker 최대 tier_max−1 + 대상 1 (stack.py:35-40, engine.py:619-663). 재조작 scan 단계 수 = M−1
- Q (사건 큐 칸) = N + K + I + 2V + U + 8 (시험 32) — reset 에 오더 전부(BLOCK_ARRIVAL/JOB_RELEASED)가 한꺼번에 들어가고(engine.py:226-243), 그 위에 크레인당 완료 1·주입 I·선박당 STS/RELEASED·이송 U·HORIZON. 넘치면 overflow(gpu/events.py:79)
- S_max (scan 스텝 상한) = 8·N + 256 (시험 64) — 스텝 = 사건 1 또는 결정 1. 사건 ≤ 시드 N+1+I + 완료(≤ SERVE N + REPO 결정 수); 결정 ≤ 사건 수. REPOSITION 반복 정책이면 더 커질 수 있어 종료 안 됐으면 violation STEPS_EXHAUSTED
- E (사건 로그 칸) = S_max + N (시험 128) — 큐 사건 + DISPATCH(결정마다 K) + wake/escape 기록. event_stream_hash 대조는 호스트가 배열을 읽어 v5 포맷으로 재구성
- W / D / Rv (ETA wake · DEFER wake · review epoch) = W = GATE_OUT 오더 수 · D = 64 · Rv = 관측창/60+1 (24h 면 1,441) — engine.py:166-171 (GATE_OUT+ETA 만 시드), 339-347, 314-322. 조각 1 은 전부 0 칸
- L (레인) = 2 (fixture) / 프로파일 값 — engine.py:271-273 lane=(bay−1) mod L; lane.py 인접표
- P (결정당 계획 계산 상한, 선택) = 기본 = N 전부; 확장 시 64 — 결정 한 번에 K·N·(M−1) 번 find_slot(B·R 격자) — 시험 규모에선 무시. N=1,024·K=2·M=6·240칸 ≈ 2.9M 연산/결정이라 GPU 는 무방, CPU 동등성 시험은 N 작음. 커지면 dispatchable 상위 P 개만 gather 해 계획(순서는 색인 오름차순 유지)

## event_kinds

- 0 JOB_COMPLETED (우선순위 0) · target=크레인 번호 k · 처리=moves(k,·) 를 M 단계 scan 으로 실현(재배치=remove→place, 반출=remove, 반입=place) → res.active[k]=F, token_owner[res.token[k]]=-1, res.slots[k]=F → bay/row[k]=plan.end → idle_pos[k]=end_bay → assigned[k]=-1, status IDLE, available_at=clock, is_loaded F, completions+1, loaded_m/empty_m += plan → kpi.loaded/empty/rehandles += → plan.kind==SERVE: served+1, n=plan.job: status DONE, done_s=clock, rehandles[n]=plan.rehandles, completed_ext/ves += , gate_out_s = clock+max(0,exit_travel) (exit_travel≥0 인 외부트럭), in_block[n]=F → down_pending→down → yielded[:]=F → plan.kind[k]=-1 (engine.py:887-945)
- 1 EQUIPMENT_DOWN (1) · target=크레인 번호 (없는 id 는 호스트가 -1 로 → 무시) · 처리=valid=target≥0; down_pending[k] |= valid & assigned[k]≥0; down[k] |= valid & assigned[k]<0 (engine.py:865-867, 1041-1046)
- 2 EQUIPMENT_UP (1) · target=크레인 번호 · 처리=down[k]=F, down_pending[k]=F, yielded[:]=F (engine.py:868-874)
- 3 TRANSFER_ARRIVE (2) · target=선박 번호 (조각 4) · 처리=양하: buffer=max(0,b−1), 그 배의 PLANNED·STORE 작업 중 release_rank 최소를 VESSEL_RELEASED 로 clock 에 push; 적하: buffer+1; 대기 요청 재배차(링버퍼 head); 막힌 STS 재개 가능하면 STS_MOVE 를 clock 에 push(불필요 push 는 time=+inf 로 무시); yielded[:]=F (engine.py:1008-1022)
- 4 STS_MOVE (2) · target=선박 번호 (조각 4) · 처리=act=started&~done; ok=where(work==DISCH, buffer<cap, buffer>0); blocked_since=where(act&~ok&blocked==inf, clock, where(act&ok, inf, old)); buffer += where(act&ok, ±1, 0); remaining −= act&ok; finish=act&ok&remaining≤0 → done, actual_completion, vessel_delay/depart_delay accrue; 아니면 STS_MOVE 를 clock+cadence 에 push; 양하면 transfer_request (engine.py:959-1039)
- 5 BLOCK_ARRIVAL (3) · target=오더 번호 n · 처리=status[n]=WAITING; waiting[n]=T; block_in_s[n]=clock; in_block[n]=T (장부 모드); yielded[:]=F (engine.py:847-853; kpis.py:48-49; time_contract.py:60-62)
- 6 JOB_RELEASED (3) · target=오더 번호 · 처리=status[n]=RELEASED; yielded[:]=F (engine.py:854-856)
- 7 VESSEL_RELEASED (3) · target=오더 번호 (조각 4) · 처리=status[n]=RELEASED; yielded[:]=F — 같은 시각 TRANSFER_ARRIVE(2) 뒤에 오는 것은 큐 키가 보장 (engine.py:877-881)
- 8 VESSEL_START (4) · target=선박 번호 (조각 4) · 처리=do=~started[v]; started|=do; remaining=where(do,total,old); STS_MOVE 를 where(do, clock+cadence, +inf) 에 push (engine.py:948-957)
- 9 PLAN_CHANGE (5) · target=선박 번호 + 주입 사건 행 번호(aux 표) (조각 4) · 처리=주입 표 (I,) 에서 completion/basis/etd 를 NaN 아니면 덮어쓰기; (I2,) 작업마감 쌍을 마스크 산포 deadline.at[job].set (engine.py:1048-1065)
- 10 ETA_UPDATED (6) · target=오더 번호 · 처리=no-op, 로그만 (engine.py:882-883). CargoBlock 의 not_before 해제는 조각 6
- 11 HORIZON (7) · target=-1 · 처리=no-op, 로그만 (engine.py:882-883, 243)
- 12 DISPATCH (로그 전용 — 큐에 안 들어감) · target=크레인 번호 (호스트가 v5 'crane:job' 문자열로 복원) · 처리=assign 시 (clock, 12, k) 기록 (engine.py:721)
- 13 ETA_WAKE / DEFER_WAKE (13/14, 큐 밖 데이터 — 조각 3) · target=오더 번호 / -1 · 처리=clock ≥ wake_s[idx]−EPS 인 동안 idx+1·로그; fired 면 eta_armed[:]=T, yielded[:]=F (engine.py:359-382)
- 15 DEADLOCK_ESCAPE (결정 개방 — 조각 2) · target=유휴 크레인 마스크 · 처리=모든 크레인 유휴 & SERVE 후보 0 & 거절사유가 오직 CRANE_INTERFERENCE & clock>last_decision_at+EPS & escape_at≠clock → yielded[:]=F, pending=idle, escape_at=clock, count+1 (engine.py:384-451)
- 16 시계 전진 _advance(t) (사건 아님 — 모든 사건·결정 앞) · target=목표 시각 t · 처리=lo=min(clock,end), hi=min(t,end), dt=hi−lo (dt>0 일 때만): queue_area += dt·Σwaiting; tail += Σ_waiting clip(hi−max(lo, block_in+sla), 0); block_area += dt·Σin_block; block_tail += Σ_in_block clip(hi−max(lo, block_in+sla), 0); terminal_area += Σ_{gate_in<inf} clip(min(hi, gate_out)−max(lo, gate_in), 0); cost.pending[truck_wait] += Δblock_area(장부) 또는 Δqueue_area(비장부); long_wait += Δblock_tail/Δtail; pending[rate 항] += rate·dt; clock=t; t<clock−EPS 면 violation|=64 (engine.py:784-813; kpis.py:71-80; time_contract.py:77-105; cost.py:55-61)

## first_piece_detail (전문)

■ §0 준비 (구현 착수 전 30분)
  · jax[cpu] 를 이 작업트리 venv 에 설치(현재 `import jax` 실패). 시험 파일 머리에 `jax.config.update("jax_enable_x64", True)`.
  · gpu/events.py: `TIME_DTYPE = jnp.float64` 모듈 상수(학습 모드는 float32 로 바꿈), `PRIO = jnp.array([0,1,1,2,2,3,3,3,4,5,6,7], jnp.int32)` (integrated/events.py:30-43 그대로), `_sort_key` 를 지우고
      def _pick(q):                                   # (time, PRIO[kind], seq) 사전식 최소
          tmin = q.time.min(); m1 = q.time == tmin
          p = jnp.where(q.kind >= 0, PRIO[jnp.clip(q.kind, 0, 11)], 0)
          pmin = jnp.min(jnp.where(m1, p, 1 << 20)); m2 = m1 & (p == pmin)
          return jnp.argmin(jnp.where(m2, q.seq, 1 << 30))
    next_event/peek_time 는 `i = _pick(q)` 로. push_event 의 `jnp.float32(time)` → `jnp.asarray(time, TIME_DTYPE)`. 빈 큐(전부 +inf)는 alive 검사(events.py:97)가 막는다. test_gpu_core.py:29-47 을 힙 키 (t, _PRIORITY[k], i) 로 바꾸고 칸 재사용 뒤 동시각 사례를 추가.
  · 파일 배치(신규): gpu/stack_ops.py · gpu/travel.py · gpu/plan.py · gpu/reserve.py · gpu/engine_step.py · gpu/host_convert.py · tests/v6/test_gpu_engine_equiv.py. gpu/state.py 에는 아래 NamedTuple 을 추가한다(기존 열은 유지).

■ §1 호스트 변환기 (파이썬, 1회)
  `to_block_world(profile, scenario, *, n_max, q_cap, log_cap) -> (BlockWorld, IdTables)`  /  `from_block_world(world, tables) -> dict` (v5 비교용)
  번호 규칙 (전부 파이썬 기본 문자열 정렬 = v5 sorted 와 동일):
    오더 n     = sorted(job_id) 순위        (engine.py:227, 477, 1001 의 순회 순서)
    컨테이너 c = sorted(container_id) 순위  (stack.py:18); 반입 예비칸 inbound_cont[n] = C0 + n (C0 = 초기 수)
    크레인 k   = sorted(crane_id) 순위      (cranes.py:53-54)
  reset 재현: 격자 grid.at[bay−1,row−1,tier−1].set(c); height=Σ(grid≥0); top_size 파생; 초기 위치 groups→ lo+(k+0.5)(hi−lo)/n (engine.py:105-113, K=1 이면 service_bay_min); idle_pos=bay; rail_order=argsort((bay, k)); gate_in_s[n] = actual_gate_in or 0.0 (외부트럭·exit_travel 있음, engine.py:127-133); 큐 시드 = 오더 정렬순 BLOCK_ARRIVAL(외부)/JOB_RELEASED(그 외, 양하 STORE 제외) → 선박 정렬순 VESSEL_START → 주입 (time,kind,target) 정렬 → HORIZON(horizon_s, target −1) 을 push_event 로 같은 순서 호출(seq 동일, engine.py:226-243). 입력 검증 `_validate`(198-221행)는 호스트에서 그대로 실행.

■ §2 상태 묶음 (array_layout 참조) — `BlockWorld(NamedTuple)`: clock, end_s, terminal, last_decision_at, escape_at, escape_count, queue(EventArray), orders(OrderArrays+엔진 열), cranes(CraneArrays+런타임·스펙 열), stacks(grid, height, top_size), conts(c_bay,c_row,c_tier,c_size,c_avail,c_alive), res(active,token,lo,hi,lane,release_at,slots,token_owner,idle_pos), plan(kind,job,lo,hi,dur,end_bay,end_row,rehandles,loaded_m,empty_m,n_moves,start_s, mv_cont,mv_src,mv_dst,mv_kind), kpi(…), ledger(…), cost(rate5,pending13,episode13), log(t,kind,target,n), violation, overflow. 기하·스펙 상수(B,R,T, bay_len,row_w,tier_h, transfer_row, sla, shift_len, gap, n_lanes)는 `Geom` 파이썬 dataclass 로 static 인자.

■ §3 순수 함수 서명 (전부 배열 in → 배열 out, jit 가능)
  stack_ops.find_slot(height, top_size, excluded, size, bay_min, bay_max, near_bay, near_row, g: Geom) -> (found bool, bay i32 1-based, row i32 1-based)
  stack_ops.blockers_above(grid, height, c_bay, c_row, c_tier, c, g) -> (blk (T−1,) i32 위→아래, n_block i32)
  stack_ops.rehandle_capacity_ok(height, top_size, c_bay, c_row, n_block, bay_min, bay_max, g) -> bool
  stack_ops.place(stacks, conts, c, bay, row) -> (stacks', conts', viol_bits)   /  stack_ops.remove(stacks, conts, c) -> (…)
  travel.move_container(spec_k(7 스칼라), start_bay, start_row, src(3,), dst(3,), g) -> (dur, loaded_m, empty_m, end_bay, end_row)
  plan.plan_serve(world, k, n, extra_excluded (B,R) bool, g) -> PlanOut(ok, kind, lo, hi, dur, end_bay, end_row, rehandles, loaded_m, empty_m, lane, slots (B,R), n_moves, mv_cont (M,), mv_src (M,3), mv_dst (M,3), mv_kind (M,), viol)
  reserve.reject_code(res, k, token, lo, hi, lane, slots, idle_pos, gap) -> i32 (0 없음 1 DOUBLE 2 DUP 3 LANE 4 INTERF 5 SLOT)
  engine_step.advance(world, t, g) -> world  /  handle_<kind>(world, target, g) -> world  /  decide(world, params, g, policy_fn) -> world  /  step(world, _, *, params, g, policy_fn) -> (world, trace_row)  /  run(world0, params, g, policy_fn, S_max) -> world

■ §4 find_slot — 격자 전체를 한 번에 (stack.py:139-168 의 전수판)
    bays = arange(1, B+1)[:, None]; rows = arange(1, R+1)[None, :]           # (B,1),(1,R)
    gcost = abs(near_bay − bays) * bay_len                                     # 142행
    cost  = (gcost + abs(near_row − rows) * row_w) + height * tier_h           # 164행 결합순서 그대로, f64
    valid = ~excluded & (height < T) & ((height == 0) | (top_size == size)) & (bays >= bay_min) & (bays <= bay_max)   # 154, 160, 162행
    c = where(valid, cost, +inf); m = c.min(); tie = valid & (c == m)
    flat = argmax(tie.reshape(−1))                # 행우선 평탄화 = bay 우선·row 차순 → (cost, bay, row) 사전식 최소 (165-166행)
    found = isfinite(m); bay = where(found, flat // R + 1, −1); row = where(found, flat % R + 1, −1)
  근거: 조기 종료(151행)는 g > best 인 bay 만 자르고 그 후보는 비용이 엄격히 크므로 전수 argmin 과 답이 같다. 동률은 float64 비트 동일성에 달렸다 → CPU·x64.
  blockers_above: b,r = c_bay[c]−1, c_row[c]−1; h = height[b,r]; n_block = h − c_tier[c]; blk[k] = grid[b, r, h−1−k] (k < n_block 유효, 그 외 −1) — 위에서부터 (stack.py:40 reversed).
  rehandle_capacity_ok (stack.py:72-98): size_blk = top_size[b,r]; mask = (bays∈[min,max]) & ~((bays==b+1)&(rows==r+1)) & (height<T) & ((height==0)|(top_size==size_blk)); ok = (n_block==0) | (Σ where(mask, T−height, 0) ≥ n_block).
  store_slot_exists[k, size] (engine.py:508-512): find_slot(height, top_size, excluded=res.slots.any(0)?→ ★아니다: _dispatchable 은 exclude 없이(510-511행), _plan 은 reserved_slots 포함(588행) — 두 번 부른다).

■ §5 plan_serve (engine.py:552-670 의 SERVE 두 경로; REPOSITION 은 조각 3)
  공통: cur_bay,cur_row = cranes.bay[k], cranes.row[k]; excluded0 = res.slots.any(0) | extra_excluded (588행); lo=hi=cur_bay (590행); is_ext = orders.is_external[n]
  (a) STORE (596-616행): found,db,dr = find_slot(height, top_size, excluded0, inbound_size[n], bay_min[k], bay_max[k], cur_bay, cur_row); dtier = height[db−1,dr−1]+1; src=(db, transfer_row, 1); mv = move_container(spec_k, cur_bay, cur_row, src, (db,dr,dtier)); dur = mv.dur + where(is_ext, truck_pos_k, 0); loaded=mv.loaded; empty=mv.empty; end=(db,dr) as float; lo=min(cur_bay, db); hi=max(cur_bay, db); slots=onehot(db,dr); n_moves=1; mv_cont[0]=inbound_cont[n]; mv_kind[0]=2; ok=found; rehandles=0.
  (b) RETRIEVE (618-663행): tc = target_cont[n]; blk, n_block = blockers_above(...); lax.scan over k∈0..M−2 with carry=(cur_bay,cur_row,excluded,total_s,loaded,empty,rehandles,lo,hi,slots,ok, moves…):
      active = k < n_block; b = blk[k]; sb,sr,st = c_bay[b],c_row[b],c_tier[b]
      excl_k = excluded | onehot(sb,sr)                                            # 622행
      found,db,dr = find_slot(height, top_size, excl_k, c_size[b], bay_min[k], bay_max[k], float(sb), float(sr))   # 626행 — 기준점은 blocker 자기 위치
      dtier = height[db−1,dr−1] + 1                                                # 631행 (스택은 계획 중 불변, 목적지 중복은 excluded 가 막음)
      post_ok = found & (dtier ≤ T) & ((height==0)|(top_size==c_size[b]))[db−1,dr−1] & ~excluded[db−1,dr−1] & ~((db==sb)&(dr==sr))   # 634-636행 → 위반 시 viol|=8
      mv = move_container(spec_k, cur_bay, cur_row, (sb,sr,st), (db,dr,dtier))
      갱신(전부 where(active, 새값, 옛값)): total_s += mv.dur; loaded += ; empty += ; cur=(db,dr); rehandles += 1; excluded |= onehot(db,dr) (645행); lo=min(lo,sb,db); hi=max(hi,sb,db) (646행); slots |= onehot(sb,sr)|onehot(db,dr) (647행); mv_cont[k]=b; mv_src[k]=(sb,sr,st); mv_dst[k]=(db,dr,dtier); mv_kind[k]=0; ok &= found
    scan 뒤 대상 이동 (648-663행): tb,tr,tt = c_bay[tc],…; dst=(tb, transfer_row, 1); mv=move_container(spec_k, cur_bay, cur_row, (tb,tr,tt), dst); total_s += mv.dur + where(is_ext, truck_pos_k, 0); loaded/empty +=; cur=(tb, transfer_row); lo=min(lo,tb); hi=max(hi,tb); slots |= onehot(tb,tr); mv_cont[n_block]=tc; mv_kind[n_block]=1; n_moves=n_block+1.
  결과: lane = where(n_lanes>0, (bay_of_job−1) % n_lanes, −1) 로 bay_of_job = 대상 bay 또는 STORE 슬롯 bay (engine.py:271-273, 537-543); start_s=clock; ok 가 거짓이면 v5 None.
  move_container (travel_time.py:44-69): hoist(tier, v) = ((T+1)−tier)·tier_h / v;  dur = e_g/gantry + e_t/trolley + hoist(src_t, empty) + lock + hoist(src_t, loaded) + l_g/gantry + l_t/trolley + hoist(dst_t, loaded) + unlock + hoist(dst_t, empty) — 이 10항 좌결합 순서를 코드에 그대로 적는다; e_g=|start_bay−src_bay|·bay_len, l_g=|src_bay−dst_bay|·bay_len; end=(float(dst_bay), float(dst_row)).

■ §6 reject_code (reservation.py:86-99 순서 고정)
    c1 = res.active[k]                                                       # DOUBLE_RESERVE
    c2 = (token ≥ 0) & (res.token_owner[token] ≥ 0)                          # DUP_JOB
    c3 = (lane ≥ 0) & any(res.active & (res.lane == lane))                   # LANE_CONFLICT (자기 포함 — v5 도 lane_owner 는 자기 제외 안 함)
    others = res.active & (arange(K) != k)
    ov_res = ~((res.hi + gap ≤ lo) | (hi + gap ≤ res.lo))                    # Corridor.overlaps (21-22행)
    ov_idle = ~((idle_pos + gap ≤ lo) | (hi + gap ≤ idle_pos)) & ~res.active & (arange(K) != k)   # 70-76행 점 장벽
    c4 = any(others & ov_res) | any(ov_idle)                                 # CRANE_INTERFERENCE
    c5 = any(res.slots[others] & slots[None])                                # SLOT_CONFLICT
    code = where(c1,1, where(c2,2, where(c3,3, where(c4,4, where(c5,5,0)))))
  K=1 이면 c4·c5 는 항상 거짓, c3 는 자기 예약이 없으니 거짓.

■ §7 처리기 갱신식 (모두 world → world; 대상 target 은 스텝이 넘긴 정수)
  h_arrival(n) (847-853행): status[n]=2(WAITING); waiting[n]=T; block_in_s[n]=clock; in_block[n]=T; yielded[:]=F.
  h_released(n) (854-856행): status[n]=1; yielded[:]=F.
  h_completed(k) (887-945행): no_plan = plan.kind[k]<0 → viol|=128.
      moves scan (M 단계, valid = i < n_moves[k]): kind0: remove(c)→place(c,dst); kind1: remove(c); kind2: place(c,dst) — remove/place 는 §3 의 함수, 무효 단계는 항등.
      n = plan.job[k]; res.active[k]=F; token_owner = token_owner.at[res.token[k]].set(where(res.token[k]≥0, −1, …)); res.slots[k]=F (901행); bay[k],row[k]=plan.end (902행); idle_pos[k]=plan.end_bay (903행); assigned[k]=−1; status[k]=IDLE; available_at[k]=clock; is_loaded[k]=F; completions[k]+=1; loaded_m[k]+=plan.loaded; empty_m[k]+=plan.empty; kpi.loaded/empty += ; kpi.rehandles += plan.rehandles (908-912행)
      serve = plan.kind[k]==0: served[k]+=1; status[n]=5(DONE); done_s[n]=clock; rehandles[n]=plan.rehandles; kpi.completed_ext += is_ext; completed_ves += ~is_ext (deadline 지각은 본선만, 조각 4); ledger 모드 & is_ext & exit_travel≥0: gate_out_s[n]=clock+max(0,exit_travel[n]), in_block[n]=F (927-928행; time_contract.py:67-74)
      down_pending[k] → down[k]=T, down_pending[k]=F (943-944행); yielded[:]=F (945행); plan.kind[k]=−1.
  h_down(k) / h_up(k): event_kinds 참조. h_horizon: 항등.
  advance(t) (784-813행): event_kinds 의 16번 식 그대로. imbalance 등 rate 는 refresh_rates 가 스텝 끝에 다시 계산: rate = [0, 0, 0, Σyielded, I/shift_len] with loads=where(assigned≥0, max(0, available_at−clock), 0), I=where(Σloads>0 & K≥2, (max−min)/Σ, 0) (815-838행). cost.pending/episode 의 항 순서는 호스트가 contract/schema COST_TERMS 에서 읽어 상수 색인으로.

■ §8 decide — 마스크 규칙
    idle = (assigned<0) & ~down (cranes.py:33);  eligible = idle & ~yielded (456행)
    dispatchable[k,n] (494-513행) = status[n]∈{1,2} & assigned_crane[n]<0 & (block[n]≥0)
        & where(has_target, c_alive[tc] & c_avail[tc] & (bay_min[k] ≤ c_bay[tc] ≤ bay_max[k]) & rehandle_capacity_ok(tc,k), T)
        & where(is_store[n], find_slot(height, top_size, zeros(B,R), inbound_size[n], bay_min[k], bay_max[k], bay[k], row[k]).found, T)
    taken[n] = token_owner[n] ≥ 0 (481행)
    P = vmap_k(vmap_n(plan_serve))(world, extra=0) → plan_ok (K,N) …
    can_res[k,n] = reject_code(res, k, token=n, P.lo, P.hi, P.lane, P.slots, idle_pos, gap) == 0 (489행)
    cand = eligible[:,None] & dispatchable & ~taken[None,:] & P.ok & can_res
    open = cand.any(1); decide = ~due_now & open.any() & (clock < end−EPS)
    pick = policy_fn(params, x, cand) (K,)  — 시험 정책 first_by_id(params, x, mask) = where(mask.any(), argmax(mask), −1)
    배정 scan(크레인 정렬 순, K 단계; carry=(res, world)): 단계 k: a = open[k]; n = pick[k]; wait = a & (n<0)
        yielded[k] |= wait (686행)
        serve = a & (n≥0): P2 = plan_serve(world, k, n, extra=carry.res.slots.any(0)) (694행 — 재계획; K=1 이면 P 와 동일); code = reject_code(carry.res, …) (697행); viol |= where(serve & (code>0 | ~P2.ok), 16, 0)
        ok = serve & P2.ok & (code==0): res.active[k]=T, token[k]=n, lo/hi/lane/release_at=clock+dur, slots[k]=P2.slots, token_owner[n]=k (106-108행); plan[k]=P2 (698행); assigned[k]=n; status[k]=WORKING; available_at[k]=clock+P2.dur; is_loaded[k]=T (699-702행); orders: status[n]=4(RUNNING), assigned_crane[n]=k, service_s[n]=clock (706-709행); waiting[n]=F, wait_sample[n]=clock−block_in_s[n] (711행, kpis.py:51-53) — is_ext 일 때만; cost.pending[crane_travel]+=P2.loaded, [empty_travel]+=P2.empty, [rehandle]+=P2.rehandles (714-719행); queue=push_event(queue, clock+P2.dur, 0, k) (720행); log (clock, 12, k) (721행)
    close (724-731행): answered==pending 은 구조상 항상 참(전 open 크레인이 답함); last_decision_at=clock (298행); refresh_rates.

■ §9 step 합성
    def step(w, _):
        nt = w.queue.time.min(); alive = nt < inf; inwin = nt ≤ w.end_s + EPS
        due_now = alive & inwin & (nt ≤ w.clock + EPS)
        D = decide(w)                     # open/decide 계산 + 적용 (cond 로 감싸 건너뛰기 가능)
        pop = ~D.decided & alive & inwin
        E = pop 국면: (q', t, kind, tgt, _) = next_event(w.queue); w1 = advance(w._replace(queue=q'), t); w1 = log(w1, t, kind, tgt); w1 = select_by_kind(kind, [h_completed(w1,tgt), h_down, h_up, id, id, h_arrival, h_released, id, id, id, id, id]); refresh_rates
        F = ~D.decided & ~pop: advance(w, max(clock,end)); wait_sample = where(waiting & isnan(sample), max(0, end−block_in), sample) (kpis.py:58-63); closed_end=end; terminal=T (1068-1084행)
        w' = tree_where(D.decided, D.world, tree_where(pop, E, F)); w' = tree_where(w.terminal, w, w'); steps+=~w.terminal
        return w', (w'.clock, D.decided, kind)
    run: w, trace = lax.scan(step, w0, None, length=S_max); viol |= where(~w.terminal, 256, 0). 여러 세계는 vmap(run).

■ §10 시험 입력 (tests/v6/test_gpu_engine_equiv.py)
  프로파일: dataclasses.replace(fixtures.build_integrated_profile(), block=BlockGeometry('B1', 10, 4, 4, 6.5, 2.9, 2.6, 0), cranes=(replace(fixtures._spec('YC-A'), service_bay_max=10),), lane_graph=LaneGraph(('L1',), ()), transfer=TransferFleetSpec('TF1','YT', n_units=0, move_time_s=180.0)) — long_wait_sla 1800·decision_horizon 1800·safety_gap 2.0 그대로.
  시나리오: containers {C1:(5,1,1), C2:(5,1,2), C3:(8,2,1)} FT40 FULL; jobs = [Job('J-OUT-1', GATE_OUT, release 0, gate_in 0, arrival 300, target 'C1', exit_travel_s 60), Job('J-OUT-2', GATE_OUT, …, arrival 300, target 'C3', exit 60), Job('J-IN-1', GATE_IN, gate_in 100, arrival 700, inbound FT40 FULL, exit 60), Job('J-OUT-3', GATE_OUT, gate_in 900, arrival 1500, target 'C2', exit 60), Job('J-IN-2', GATE_IN, gate_in 900, arrival 1500, inbound FT40, exit 60)]; vessels [] ; injected []; horizon_s 7200, drain 0.
  정렬 결과(호스트가 확인해 시험에 박음): 오더 0 J-IN-1 · 1 J-IN-2 · 2 J-OUT-1 · 3 J-OUT-2 · 4 J-OUT-3; 컨테이너 0 C1 · 1 C2 · 2 C3 · 반입 3(=C0+0)·4(=C0+1); 시드 seq: J-IN-1(700) J-IN-2(1500) J-OUT-1(300) J-OUT-2(300) J-OUT-3(1500) HORIZON(7200). 기대 거동: t=300 에 J-OUT-1 → J-OUT-2 순 도착, 첫 결정에서 오더 2(J-OUT-1, C2 를 치우는 재조작 1회) 선택; 완료 후 오더 3; t=700 J-IN-1 STORE; t=1500 오더 1(J-IN-2, STORE) 먼저, 그다음 오더 4(J-OUT-3 — 재조작으로 옮겨진 C2 를 새 좌표에서 반출).
  v5 구동: sim = TerminalSimulator(prof, scn, check_invariants=True); while (dp := sim.run_until_decision()) is not None: sim.commit_decisions([CraneAssignment(c, SERVE, cands[0]) if (cands := sim.candidates_for(c)) else CraneAssignment(c, WAIT) for c in dp.crane_ids]); 결정마다 (dp.time, 크레인, 오더) 기록.
  배열판: w0 = to_block_world(prof, scn, n_max=8, q_cap=32, log_cap=128); w = run(w0, params=None, g, first_by_id, S_max=64).
  비교(순서대로, 앞이 깨지면 뒤는 보지 않는다): ① 로그 (round(t,6), kind, target) 전열 == v5 event_log 를 번호로 옮긴 것 ② 결정열 ③ 오더 status/assigned_crane/rehandles ④ 계획 moves ⑤ 크레인 bay/row/served/completions ⑥ kpi 정수 ⑦ grid ⑧ violation==0 & overflow==0 & terminal; 실수(1e-6): service_s/done_s/gate_out_s, loaded/empty_m, queue_area/tail_area, wait_samples 정렬열, block_area/block_tail/terminal_area, cost.episode 13항 (truck_wait·long_wait·crane_travel·empty_travel·rehandle 만 0 아님; imbalance 는 K=1 이라 0).
  단위시험 셋(엔진 없이): (i) find_slot — tests/v6/test_find_slot_equiv.py:25-49 의 _yard(seed, fill∈{0,.3,.45,.75,.95}) 를 배열로 변환해 300 질의(size, U(0,25), U(0,11), 무작위 exclude 0~10칸) → (bay,row)|None 정확 일치; 불일치 시 v5 최선·차선 비용차 <1e-9 인 근접 동률만 따로 집계·보고 (ii) move_container 500 무작위 → x64 CPU 비트 동일 (iii) 큐 200건 힙 대조.
  성공 판정 뒤 확장 사다리: (a) 크레인 2대 (조각 2) (b) PRE_ADVICE (조각 3) (c) fixtures.build_minimal_terminal_scenario 전체 (조각 4) (d) test_world_equivalence.py 하루 300대 짝비교 항목(58-64행).

## 반박 검증에서 나온 높음·치명 (조각 1 관련)

- [벡터화 ·높음] v5 골든(기준 답)에서는 조각 1 무대에서도 lane_cong(레인 혼잡률 × 시간) 항이 0 이 아니다. 시험 프로파일이 레인 1개(`LaneGraph(('L1',), ())`)라 모든 bay 가 L1 에 매핑되고, 크레인이 작업 중이면 예약이 L1 을 점유해 혼잡률 평균이 정확히 1.0 이 되어 서비스 시간 전부가 lane_cong 으로 적립된다. 배열판이 §7 대로 레인 항을 0 으로 두면 ⑧ 13항 대조가 반드시 깨진다 — 계획 자체의 기대값이 틀렸다.
  근거: engine.py:271-273 `_lane_for`: `ids[(bay-1) % len(ids)]` → L=1 이면 항상 'L1'; engine.py:544 `_jobref(... lane_id=self._lane_for(bay))` → 예약에 lane 부착; engine.py:818-820 `_refresh_rates`: `occ = frozenset(r.lane_id for r in reservations.active() ...)`, `set_rate("lane_cong", self.lanes.occupancy(occ)[0])`; lane.py:44-50 `occupancy`: 레인 1개·인접 0 이면 load=1.0, deg=0 → vals=[1.0] → mean 1.0; cost.py:55-61 `advance`: `accrue(term, rate*dt)` → 크레인 작업 시간만큼 lane_cong 누적. 같은 곳 engine.py:808 `self.lanes.integrate(lo, hi, occ)` 도 cong_area_s 를 쌓는다.
  고침: 조각 1 에 레인 점유 벡터를 넣는다: occ (L,) = any_k(res.active & res.lane==l); load = occ + adj@occ; lane_mean = mean(load/(1+deg)) (L=1 이면 lane_mean = any(res.active)). §7 의 rate 5항을 [0, 0, lane_mean, Σyielded, I/shift_len] 로 고치고, §10 기대값을 "lane_cong 도 0 아님(= 크레인 점유 시간 합)" 으로 바로잡는다. lanes.cong_area_s 도 상태에 추가.
- [벡터화 ·높음] XLA 의 CPU 백엔드도 곱셈-덧셈 융합(FMA: a·b+c 를 한 번의 반올림으로 계산)을 기본으로 허용한다. 따라서 `(g + |Δrow|·row_w) + top·tier_h` 를 x64 로 적어도 파이썬(곱 반올림 → 덧셈 반올림, 두 번)과 마지막 비트가 다를 수 있고, 두 칸의 비용이 파이썬에선 같은데(동률 → (bay,row) 순) 배열판에선 다르게(또는 그 반대) 나와 다른 칸을 고를 수 있다. find_slot 의 규칙 8개(서비스 구간·exclude·tier<T·규격 일치·비용식·결합순서·(cost,bay,row) 사전식 최소·조기종료 무영향)는 §4 가 전부 정확히 재현한다 — 문제는 규칙이 아니라 "CPU 면 비트 동일" 이라는 전제다. 이 venv 에 jax 가 없어 빈도는 실측하지 못했다(계획의 단위시험 (i) 300 질의 ×5 채움이 그 실측이다).
  근거: stack.py:164-167 `cost = g + abs(near_row - row) * row_w + top * tier_h; key = (cost, bay, row); if key < best` — 동률이 비트 동일성에 의존; stack.py:122-123 docstring 이 이를 명시. XLA 소스 xla/backends/cpu/codegen/ir_compiler.cc `RunIrPasses` 마지막: `llvm_ir::SetAllowContractOnFpArithmetic(module);` 와 주석 "TODO(b/560320144): `AllowFPOpFusion = Fast` is deliberately still set in service/cpu/cpu_aot_loader.cc:53, tools/hlo_opt/cpu_opt.cc:217 ..." — CPU JIT 는 부동소수 산술에 `contract`(융합 허용) 플래그를 붙이고 AOT 는 FPOpFusion=Fast 를 그대로 둔다. 구체 동률 예: row_w 2.9·tier_h 2.6 일 때 (Δbay+2, top 0) 칸과 (Δbay, top 5) 칸은 참값 비용이 같다(13.0 = 2·6.5 = 5·2.6) — 파이썬은 5·2.6→13.0 으로 먼저 반올림하지만 융합판은 (g+t)+13.00000000000000044 를 한 번에 반올림해 결과 비트가 갈릴 수 있다.
  고침: 비용식의 곱을 곱셈 명령이 아니라 호스트에서 파이썬으로 미리 반올림한 표의 gather 로 바꾼다: ROW_COST[|Δrow|] (near_row 은 transfer_row 또는 정수 row 라 |Δrow| 는 정수 0..R), TIER_COST[top] (0..T). g = |near_bay−bay|·bay_len 은 단독 곱이라 융합 대상이 아니다. 그러면 덧셈만 남아 FMA 가 끼어들 곳이 없다. 대안은 곱 결과를 `lax.optimization_barrier` 로 실체화. 단위시험 (i) 는 "근접 동률 따로 집계" 가 아니라 CPU x64 에서 불일치 0 을 통과 조건으로 두고, GPU 에서만 계수·보고한다.
- [벡터화 ·누락] 레인 혼잡 적분 `lanes.cong_area_s`(lane.py:52-56) 와 인접 가중 점유율(lane.py:36-50) 이 상태 배열·처리기 어디에도 없다 — 조각 1 부터 rate 5항 중 하나를 빠뜨린다(finding 1).
- [벡터화 ·누락] engine.py:325-326 '큐가 비었는데 작업 중인 크레인이 있다 → RuntimeError' 에 대응하는 위반 비트가 없다(종료 국면 F 에서 `~alive & any(assigned≥0)` 검사 추가).
- [벡터화 ·누락] kpis.vessel_delay_s(kpis.py:90-96, 본선 야드작업 마감 지각 합)가 kpi 배열에 없다 — 조각 4 대조 항목에 필요.
- [벡터화 ·누락] 초기 크레인 trolley_row = transfer_row 와 초기 bay 분산을 호스트 변환기가 채운다는 규약(finding 6) 및 reset 직후 상태 대조 단계.
- [벡터화 ·누락] jax_enable_x64 를 모든 배열 생성 전에 켜는 import 규약과 dtype 단언(finding 7) — importorskip 대신 명시 실패로 두겠다는 risks #10 과 함께 적어야 한다.
- [벡터화 ·누락] CPU 에서도 find_slot 동률이 FMA 로 갈릴 수 있으므로(finding 2) 단위시험 (i) 의 통과 조건을 'CPU 불일치 0' 으로 못박고, 곱을 호스트 표로 대체하는 설계 결정.
- [벡터화 ·누락] 동등성 모드의 스텝별 관측·불변식 검사 방법(finding 8) — 통째 lax.scan 은 학습 모드 전용으로 분리.
- [벡터화 ·누락] 조각 6 multiblock.validate 의 find_slot 기준점이 크레인 위치가 아니라 고정점 (service_bay_min, 1.0) 이라는 규칙(multiblock.py:484-488) — §4 서명은 near_bay/near_row 를 받으니 재현 가능하지만 계획 본문이 언급하지 않는다. slot_plan.py 에는 find_slot 규칙이 없다(48×9 계획표 전용) — 조각 1 대조 범위 밖임을 명시.
- [벡터화 ·누락] 조각 3 REPOSITION 목표 bay 의 원천 candidates.py:41-47 `_future_bay_of` 도 exclude 없는 find_slot 을 쓴다 — finding 4 의 lane_bay(K,N) 를 재사용하면 되나 계획에 연결이 없다.
- [벡터화 ·누락] 정책망 `choose`(policy.py:89) 는 마스크가 전부 거짓일 때 0 을 돌려주고 −1(WAIT) 을 낼 수 없다 — decision_interface 가 '−1 = WAIT' 로 쓰는 것과 서명이 어긋난다(조각 7 이전에 first_by_id 와 choose 의 반환 규약을 통일해야 한다).
- [동등성 ·높음] 레인 혼잡률(lane_cong = 예약이 잡은 레인의 점유 비율)을 0 으로 못박았다. 그런데 계획이 정한 조각 1 무대는 레인 1개(LaneGraph(('L1',),()))라 크레인이 일하는 동안 v5 는 점유율 1.0 을 매 구간 적분한다. 실제로 v5 를 돌리면 lane_cong = 870.19 이고, 계획이 '0 아님' 이라 쓴 crane_travel 과 long_wait 은 오히려 0 이다(모든 이동이 같은 bay 안·최장 대기 271초 < SLA 1800). 이대로 구현하면 13항 대조에서 조각 1 시험이 실패하고, 계획의 기대값 표를 그대로 박으면 틀린 기대값과 비교하게 된다. step_structure 5) 에는 "lane_mean" 이라 적혀 있어 계획 안에서도 서로 모순이다.
  근거: engine.py:271-273 (_lane_for: ids[(bay-1)%len] → 'L1'), 819-820 (occ = 활성 예약 레인 → lanes.occupancy(occ)[0] 을 set_rate('lane_cong')), lane.py:41-50 (레인 1개·점유 시 load=1.0 → 평균 1.0), 812 (cost.advance 가 rate×dt 누적). v5 실행 결과(조각 1 시나리오 그대로): episode_raw 0 아닌 항 = {truck_wait 1310.538889, empty_travel 65.0, rehandle 1.0, lane_cong 870.188889}; crane_travel = 0, long_wait = 0.
  고침: §7 의 rate 식을 v5 그대로: lane_cong = mean_l( (occ[l] + Σ_{m∈adj[l]} occ[m]) / (1+deg[l]) ) 로 occ (L,) = any_k(res.active & res.lane==l); L=0 이면 0. advance 에도 lanes.cong_area += mean·dt 를 넣는다(engine.py:806-808). 시험 기대값은 손으로 쓰지 말고 v5 실행값(위 4항)과 대조하고, 조각 1 무대에 다른 bay 로 옮겨지는 blocker(loaded gantry>0)와 1800초 넘는 대기 1건을 넣어 crane_travel·long_wait 식도 실제로 검증되게 한다.
- [동등성 ·높음] 두 v5 의미는 다르고, 계획의 scan 은 그중 commit_decisions 쪽이다. 계획은 정책 호출(pick (K,))을 scan **앞에서** 한 번 하고 scan 안에서는 계획만 다시 세운다(재계획·reject 검사). 반면 ReferenceDispatcher 는 크레인마다 앞 크레인 배정을 반영한 **후보를 다시 뽑아** 고른다(재선택). 크레인 2대가 같은 시각에 놀고 같은 후보를 보면(조각 2 무대의 동시각 도착 짝이 정확히 이 상황) v5 RD 는 A 가 1순위, B 가 2순위를 잡지만 배열판은 B 도 1순위를 고른 뒤 scan 에서 DUP_JOB 거절 → viol|=16 → 세계 실격. 즉 조각 2 시험은 설계상 통과할 수 없다.
  근거: dispatcher.py:25-31 (for cid in dp.crane_ids: cands = sim.candidates_for(cid)  # live (앞 배정 반영) → select), 16-17 (min(본선, -cum_wait, job_id) — 동일 후보집합이면 두 크레인이 같은 job). engine.py:733-736 (commit_decisions 는 미리 정한 배정을 순차 assign, 충돌은 697 행 reserve 예외). 계획 §8 '배정 scan … P2 = plan_serve(…, extra=carry.res.slots) … code = reject_code … viol |= where(serve & (code>0 | ~P2.ok), 16, 0)' — 재선택 없음.
  고침: 정책 호출을 배정 scan **안으로** 옮긴다: 단계 k 에서 carry 예약표로 cand[k] 행(dispatchable & ~taken(carry.token_owner) & plan_ok & reject_code(carry)==0)을 다시 계산한 뒤 policy_fn(params, x[k], cand_k) 로 pick 을 뽑는다(K=1 이면 지금과 동일). 신경망 정책(choose_batch·동시 선택)은 이 순차 의미로 두거나, 조각 7 의 resolver 의미로 갈지 명시한다. 조각 2 시험은 이 순차 의미로 ReferenceDispatcher 와 대조한다.
- [동등성 ·누락] "한 크레인 실패 = 전원 WAIT" 예외 대체 규칙(stage/episode.py:212-216)은 조각 7 에만 언급됐지만, 실제 v5 fallback 은 이미 assign 된 크레인에 다시 WAIT 를 넣으면 DECISION_COVERAGE 예외(engine.py:682-683)가 나므로 '부분 배정 뒤 실패' 상황의 v5 거동이 크래시다 — 배열판이 무엇을 재현할지(전원 WAIT vs 실격) 정해야 한다.
- [동등성 ·누락] 동시 선택 신경망 정책(choose_batch = 크레인별 독립 argmin)에서 두 크레인이 같은 오더를 고르는 경우의 처리 규칙이 조각 1·2 에 없다 — 현재 설계는 viol|=16(실격)이라 학습 배치의 상당수가 실격된다. 순차(scan 안 재선택) 또는 조각 7 resolver 중 어느 의미를 학습 경로의 정본으로 할지 명시 필요.
- [동등성 ·누락] advance 식에 lanes.cong_area_s 적분(engine.py:806-808, lane.py:52-56)이 빠져 있다 — 비용 rate 와 같은 값이지만 상태로 남는 KPI 다.
- [동등성 ·누락] 탈출(_try_escape)이 _last_decision_at 도 clock 으로 갱신한다(engine.py:404) — event_kinds 15 의 갱신 목록에 없다.
- [동등성 ·누락] cranes.yield_count(recent_yield_count) 가 언제 오르는지(yield_reason == LOST_CONTENTION 일 때만, engine.py:687-688) 배정 scan 에 규칙이 없다 — resolver 없는 조각 1·2 에서는 항상 0 이어야 한다.
- [동등성 ·누락] jax 미설치를 계획도 알고 있으나(§0), 이번 검토에서도 `import jax` 실패를 확인했다 — XLA CPU 의 FMA·정수 색인 처리 등 '비트 동일' 주장은 전부 설치 뒤 단위시험으로 먼저 굳혀야 한다(위 finding 4·5).
- [동등성 ·누락] 조각 1 시험은 crane_travel(loaded gantry)·long_wait(tail) 식을 실제로 검증하지 못한다(v5 실행값 둘 다 0). blocker 가 다른 bay 로 가는 배치(예: bay 5 의 row 2~4 를 미리 채움)와 SLA(1800초)를 넘는 대기 1건을 추가해야 다섯 적분식이 전부 살아 있는 값으로 대조된다.
- [동등성 ·누락] 동등성 시험이 최종 상태만 비교한다 — 스텝 중간에 갈렸다가 우연히 합쳐지는 경우는 사건열 ① 이 대체로 잡지만, 결정마다 cand (K,N) 마스크·plan dur·reject 코드를 v5 candidates_for 결과와 스텝 단위로 대조하는 '한 결정 단위' 시험(조각 7 에서 제안된 방식)을 조각 1 에도 두면 실패 원인 추적이 훨씬 빠르다.
- [완전성 ·높음] 학습 루프가 실제로 굴리는 세계(30일 무대)가 어느 조각에도 없다. PPO 진입점 둘 다 run_month 를 부르고, 그 안에서 MonthTerminal(=MultiBlockTerminal 의 30일판), inject_vessel(그날 아침 배를 런 중에 붙임 — VesselProcess 등록·VESSEL_START push·양하 규격 난수·JOB_RELEASED push), prune_completed(끝난 job 을 sim.jobs·원장·time_ledger.records·_a_sorted 네 곳에서 삭제), retire_done_vessels(끝난 배를 archive 로 옮기고 pop), _MonthTape(날 경계 계수기 사진첩), month_vessel_idle(스트림→배 묶기, archive 포함), _day_records(docKey 'Dnn-' 필터), 그리고 review() 안에서 ppo.boundary(t) 를 부르는 순서(투입→_sync→boundary→시장→snap→날 경계)가 전부 세계 규칙인데 계획의 8개 조각 files_to_port 에 month_engine.py·month_run.py·month.py 런타임 함수가 하나도 없다. 조각 8 의 시험(run_debug)은 이 경로 없이는 실행 자체가 안 된다.
  근거: ppo/continuous.py:78 `run_month(seed=seed, days=days, ppo=runtime, ...)`; ppo/run.py:39 `run_month(seed=seed, days=days, ppo=runtime)`; month_run.py:272-281 (seed_data None → MonthTerminal, 아니면 CargoTerminal); month_run.py:448-475 open_day→inject_vessel; month_run.py:503-505 prune_completed→retire_done_vessels; month_run.py:513-539 review(): ann.review → bridge._sync → ppo.boundary(t) → bridge.review → tape.snap → open/close_day; ppo/runtime.py:115-119 read_cost 가 month_vessel_idle(self.mbt, self.meta, self.archive) 를 직접 호출; month_engine.py:64-111 MonthTerminal.run; month_engine.py:149-257 inject_vessel; month.py:171-199 prune_completed; month.py:411-456 retire_done_vessels; month.py:371-404 month_vessel_idle
  고침: 조각 6 과 8 사이에 '조각 6b 30일 무대' 를 추가: (a) 배·연계 작업 표를 t=0 에 호스트에서 전부 만들고 active_from_s(=start_s)·day 열로 두어 inject_vessel 을 '마스크 켜기' 로 대체(양하 규격 난수는 배 시드만의 함수라 사전 계산 가능), (b) prune/retire 는 배열에서 삭제 불필요 — done 마스크와 archive 대신 sts_wait_accum 을 그대로 두고 segment_sum, (c) _MonthTape=날 경계 계수기 배열 (D+1,·), _day_records=day 열 마스크, (d) review 순서(투입→동기화→boundary→시장→snap→날 경계)를 조각 8 step 명세에 명시. 조각 8 의 equivalence_test 는 이 조각에 의존한다고 depends_on 에 적는다.
- [완전성 ·높음] free_targets 는 진단 전용이 아니라 **기본 학습 경로**가 탄다. 시드 번들(--seed-bundle) 을 주지 않으면 plan_month_vessels 가 만든 행에 'targets' 키가 없어 inject_vessel 이 적하(LOAD) 배마다 free_targets 로 '지금 야드에 있고 안 찍힌 상자' 를 시드 셔플해 고른다. 어느 상자를 싣느냐가 재조작 수·본선 유휴(Φ 항3·항4)를 정하므로 학습 결과에 직접 닿는다. 조각 8 의 동등성 드라이버 run_debug 도 시드 번들 없이 run_month 를 부르므로 이 경로를 반드시 지난다 — 계획은 스스로 제외한 규칙에 의존한 시험을 적어 두었다.
  근거: month_engine.py:184-200 `if 'targets' in row: ... else: targets = free_targets(sim, limit=asked, seed=f"{size_seed}:tgt")`; month.py:308-341 plan_month_vessels 가 만드는 행 키 = work·type_offset·ship·day·key·start_s (targets 없음); month_engine.py:142-146 `cand = [c for c in sorted(sim.stacks.containers) if c not in taken]; random.Random(seed).shuffle(cand); return cand[:limit]` (야드 재고 = 궤적 의존); ppo/continuous.py:47-54 seed_bundle 은 선택 인자; ppo/run.py:39 run_debug 는 seed_data 없이 호출
  고침: 둘 중 하나를 명시: ① 배열판 학습·동등성은 반드시 시드 번들(고정 targets) 경로만 쓴다고 조각 8 시험을 restore_input(seed_data) 기반으로 바꾸고 run_debug 에 번들을 넘기도록 수정; ② 또는 날 경계 review 는 호스트 파이썬 루프에서 열리므로(계획 조각 8 array_technique) 그 시점에 device_get 한 야드 재고로 v5 free_targets 를 호스트에서 그대로 실행해 targets 를 만들어 넣는다 — 이 경우 what_not_to_port 문구를 '호스트에서 v5 코드로 실행해 결과만 넣는다' 로 고친다.
- [완전성 ·누락] 30일 무대 런타임 전체: month_engine.py:64-111 MonthTerminal.run / 149-257 inject_vessel(런 중 본선 붙이기·양하 규격 난수 random.Random(f'{size_seed}:size')·JOB_RELEASED push·TransferError 원자성) / 133-146 free_targets; month.py:171-199 prune_completed(끝난 job 을 sim.jobs·ledger.records·time_ledger.records·_a_sorted 에서 삭제) / 411-456 retire_done_vessels(done ∧ 남은 job 0 ∧ 완료 후 3600s → archive·pop·_refresh_rates) / 371-404 month_vessel_idle; month_run.py:142-167 _MonthTape(snap·read·diff, max(0,·) 클램프) / 184-187 _day_records / 448-494 open_day·close_day / 513-539 review 순서(투입→_sync→ppo.boundary→bridge.review→snap→날 경계) — ppo/runtime.py:115-119 read_cost 가 month_vessel_idle 을 직접 부르므로 조각 8 은 이것 없이 성립하지 않는다
- [완전성 ·누락] 검토 시각(review epoch)과 같은 시각 사건의 처리 순서 규칙 — MultiBlock/Month 계열(engine.py:316-322: 검토가 동시각 사건보다 먼저)과 Cargo 계열(cargo_runtime.py:225-268: 동시각 사건·결정 소진 뒤 검토)이 반대
- [완전성 ·누락] STORE 작업 lane_id 의 원천 = 예약 제외 없는 _jobref find_slot bay (engine.py:537-546) — plan_serve 의 제외 후 슬롯 bay 와 분리해야 함
- [완전성 ·누락] ETA wake 는 reset 시 scenario.jobs 한정(engine.py:166-171); 런 중 투입 오더는 wake 없음 — 무대 세계 W=0 규칙
- [완전성 ·누락] recent_yield_count 갱신식(engine.py:686-688) — 열은 있으나 §8 에 갱신이 없음(학습 경로 미사용이라 영향 없음)
- [완전성 ·누락] engine.py:324-325 '작업 중인데 완료 이벤트 없음' RuntimeError 가 violation 비트 목록(1~512)에 없음 — 큐가 비었는데 assigned≥0 인 크레인이 있으면 표시할 코드 필요
- [완전성 ·누락] 본선 항(Φ 항4)의 월 경로 집계: 스트림→배 표(vessel_meta: month.py:376-380 add()) 와 archive 합산이 조각 5 files_to_port(episode.py:225-310 만) 에 없음 — 하루 무대의 vessel_idle_of(블록 키) 와 월의 month_vessel_idle(스트림 키) 은 묶는 열쇠가 다르다
- [완전성 ·누락] time_sell.py:96-111 의 day_plan 분기는 학습 경로에서 attach 가 호출되지 않아 항상 None(try_defer_admitted_entry 만 실행) — 계획이 day_plan.py 를 뺀 것은 옳으나 그 근거를 적어 두어야 나중에 attach 가 생겨도 드러난다
- [완전성 ·누락] 확인된 것(문제 아님): what_not_to_port 의 block_congestion·slot_plan·cost_config·cost_curve·vessel_cost·ledger 는 features/ppo/stage/reward 에서 import 0건(grep) — 학습 결과 무영향; escape_mode 'delayed'·slot_selector/store_slot_selector 설정자 0건; float32 순번 소실 실측 재현(t=1.0: seq3 소실, t≥100: seq999 소실); 이 venv(anaconda3 python) 에 jax 없음