# v6 조각 2 구현 명세 — 다중 크레인

원본: outputs/v6/port_plan_wf_9e7366e6.json (plan.pieces[1]) + 조각 1 구현 결과의 미결.

## 조각 2: 다중 크레인 — 간섭·안전거리·레인·순차 예약 scan·초기 위치 분산·레일 순서·교착 탈출·장비 고장

files_to_port:
- engine.py:105-118 (초기 위치 분산·_rail_order — 호스트)
- engine.py:384-451 (_try_escape·interference_deadlock_corridors)
- engine.py:738-763 (dry_run_commit — 크레인 순 scan)
- reservation.py:64-77 (corridor_conflict 예약·idle 장벽)
- engine.py:824-838 (load_imbalance), 815-822 (interference rate)
- engine.py:1107-1121 (레일 순서·간격 검사 → violation 비트, 시험 모드)
- integrated/dispatcher.py:14-32 (ReferenceDispatcher — K≥2 동등성 시험용 v5 정책)

hard_parts:
- 교착 술어가 후보 생성 전체를 재계산(engine.py:430, 437-450) — (K,N) 후보 행렬을 한 번 만들어 reason 행렬로 재사용
- 동시 결정 크레인 간 경합: 앞 크레인 슬롯이 뒤 크레인 exclude 가 되어 **재계획**(752행) — 배정 scan 안에서 크레인마다 plan 재계산 필요
- corridor 충돌은 (K,K) 쌍별 + 유휴 위치 점 장벽, 자기 제외 (reservation.py:65-76)

array_technique:
배정을 크레인 정렬 순 lax.scan(carry=예약 배열)으로; 단계마다 plan(extra_exclude=carry.slots.any(0)) 재계산 → reject 5-lock → 수용 시 carry 갱신. 교착 = ~any(assigned) & ~any(cand) & any(reason==INTERFERENCE); 통로 = (N,2) + 마스크(정렬·중복제거 대신).

equivalence_test:
조각 1 무대에 YC-B(service 1..10) 추가, 오더 8건(동시각 도착 짝 2개), 주입 EQUIPMENT_DOWN(2000,YC-B)/UP(2600) 추가. v5 정책 = ReferenceDispatcher.run (dispatcher.py:19-32: 크레인 순차·live 후보 재계산) — 배열판 배정 scan 과 같은 의미. 비교 = 조각 1 항목 + 초기 위치(1+(k+0.5)·9/2)·rail_order·reject 사유 코드열·down/down_pending 시각·cost imbalance/interference·deadlock_escape_count. 단위시험: reject_reason 50조(예약 2 + 유휴 위치 + 후보) 를 random.Random(0) 로 만들어 사유 코드 정확 일치.

예상 규모: ~300줄

## 조각 1 이 남긴 미결·한계 (조각 2 가 이어받는 것)

- K=1 에서는 코드 2~5 가 구조적으로 나올 수 없어(자기 예약이 있으면 항상 DOUBLE 이 먼저) 5-lock 순서 검증은 K=2·3 이 담당한다. 조각 1 시험(K=1)만으로는 c3~c5 가 실제 엔진 경로에서 검증되지 않는다 — 조각 2(크레인 2대)에서 엔진 수준 대조 필요.
- reject_code 의 lane 검사는 v5 처럼 자기 예약을 제외하지 않는다 — DOUBLE 이 먼저 잡혀 결과는 같지만, 뒤 단계가 c1 을 우회해 c3 만 따로 쓰려 하면(예: 탈출 판정에서 '거절사유가 오직 CRANE_INTERFERENCE' 검사, engine.py:447-448) 반드시 reject_code 전체를 쓰고 코드==4 로 비교해야 v5 와 같다.
- 성능 미측정: plan_serve_all 은 (k,n) 마다 STORE·RETRIEVE 둘 다 계산하고 RETRIEVE 는 (M−1) 번 find_slot(각각 FMA_GUARD optimization_barrier 포함) — N=1,024·K=2·M=6 에서 결정당 비용을 engine_step 작성 뒤 재 볼 것. 학습 모드(f32·동등성 불필요)에서는 stack_ops.FMA_GUARD 와 travel.div_exact 스위치를 끄는 것을 검토.
- 결정 구동(run_until_decision→commit) 대조는 크레인 1대에서만 했다(2대는 후보 첫째 정책이 같은 오더를 골라 DUP_JOB 예외). 크레인 2대는 t=0 계획 대조(832건)만 — 엔진 수준 K=2 는 조각 2 몫.
- DEADLOCK_ESCAPE 로그 target 인코딩을 '유휴 크레인 비트마스크(bit k)' 로 정했다. 조각 2 엔진이 같은 규약으로 적어야 하고, K>31 이면 int32 비트가 부족하다(단일 블록 K=2 라 문제 없음).
- rail_order 는 '레일 i번째 자리의 크레인 번호' 순열(rail_order[i]=k)이다 — state.py 주석과 같은 읽기지만 조각 2 의 CRANE_ORDER_SWAP 검사가 역순열(k→자리)로 읽으면 K=2 에서 우연히 같아 못 잡는다. 명시적으로 맞출 것.
- K≥2 결정 의미: 정책을 scan 앞에서 한 번 부르므로(§8 원문) 두 크레인이 같은 오더를 고르면 DUP_JOB → violation 16 (연기 시험 실측: crowded 무대 K=2 에서 violation=16). v5 ReferenceDispatcher(dispatcher.py:19-32)와 같으려면 조각 2 가 정책 호출을 배정 scan 안으로 옮기거나(단계 k 에서 carry 예약표로 cand 행 재계산 후 policy_fn) 조각 7 resolver 의미를 정본으로 정해야 한다. K=2 엔진 수준 동등성(c3~c5 거절 코드 포함)은 조각 2 몫.
- gpu/reserve.py 와 gpu/state.py 가 같은 이름의 ReservationArrays 를 따로 정의해 pytree 형이 다르다 — 엔진이 매 호출 _as_res 로 변환한다. reserve.py 담당이 state.py 의 클래스를 import 하도록 통일하면 변환을 뺄 수 있다.
- 성능 미측정: S_max=64·N=8·K=1 에서 jit 컴파일+실행 약 1.7초, run_python(eager cond/scan 재컴파일)은 17스텝에 27초. N=1,024·K=2·M=6 에서 스텝당 plan_serve_all(K·N·(M−1) find_slot)+lax.cond 비용은 재 볼 것. vmap 아래서는 cond/switch 가 select 로 풀려 결정·사건·종료 세 국면을 매 스텝 전부 계산한다.
- K≥2 동등성은 여전히 조각 2 몫: 정책 호출은 배정 scan 앞에서 한 번이라 두 크레인이 같은 오더를 고르면 DUP_JOB(16) 실격(two-cranes-smoke 가 16 을 허용하는 연기 시험). 반박 검증이 요구한 'scan 안 재선택' 의미는 미구현.

## 조각 1 검증 렌즈가 지적한 K≥2 관련

- [동등성 ·누락] 크레인 2대(K≥2) 엔진 수준 동등성(순차 재계획·LANE/INTERFERENCE/SLOT 거절·imbalance rate·rail_order)은 조각 2 몫이라 검증하지 않았다. test_two_cranes_smoke_runs_to_terminal 은 violation 16(DUP_JOB)을 명시적으로 허용하는 연기 시험이며 동등성 주장은 아니다.
- [벡터화 ·낮음] 시험의 로그 칸 log_cap = 8·n_max+256 은 S_max 와 같아 명세 E = S_max + N 보다 N 만큼 작다. 로그 수 ≤ 사건 스텝 수 + DISPATCH 수(≤ N) 이므로 K=1 에서는 스텝당 로그 ≤ 1 이라 안전하지만, K≥2 에서는 결정 스텝 하나가 DISPATCH 를 K 줄 남겨 S_max 를 넘길 수 있다(넘치면 overflow 로 표시는 되므로 조용히 지나가진 않음). 또 §10 이 정한 시험 값(q_cap 32·log_cap 128·S_max 64) 대신 320/320 을 써 명세와 다르다(더 크므로 안전).
  고침: `log_cap = s_max + n_max` 로 명세식을 그대로 쓰고, 조각 2(K=2)부터는 그 식이 상한임을 시험 주석에 적는다.
- [벡터화 ·누락] 학습 규모(N=1024·K=2·M=6·B=256, 24×10×6 격자)의 컴파일·스텝 시간은 측정하지 않았다 — 실측은 N≤256·K=1·B≤32·10×4×4 까지. vmap 아래 K=2 배정 scan 은 두 크레인 연기(단일 세계)만 통과 확인.
- [벡터화 ·누락] 실제 정책망(policy.choose_batch + PolicyParams)을 policy_fn 으로 넣어 run_jit/vmap 을 컴파일해 보지 않았다 — first_by_id 만 사용. choose 가 마스크 전부 거짓일 때 0 을 내는 문제는 decide 의 where(open_, pick, -1) 로 가려지지만 후보 있는 크레인의 명시적 WAIT 는 여전히 표현 불가(기존 open issue).
- [완전성 ·낮음] terminal_area 를 닫힌 식 Σ_{A<inf} max(0, min(hi,O)−max(lo,A)) (189-191행)으로 계산해 v5 time_contract.py:89-105 의 경계 조각 순차 적분과 합산 순서가 다르다. 값은 같지만 마지막 비트가 갈려 절대 규칙 1('v5 의 연산 결합 순서를 그대로')에 어긋나며, 담당자가 open_issues 에 적어 둔 대로 시험 허용오차(1e-6) 뒤에 숨어 있다. 같은 부류로 tail_area·block_tail 도 `jnp.sum` 뒤 한 번 더함(182·188행) 이라 대기 트럭이 많으면 갈릴 수 있다(이번 9무대에서는 비트 동일).
  고침: reset 에 정렬된 A 열과 완료 시 O 를 넣는 고정 길이 경계 배열을 두고, [lo,hi] 안의 경계를 v5 순서(A≤O 면 진입 먼저)로 도는 고정 길이 lax.scan 으로 `(nxt−t)*n_inside` 를 차례로 더한다(곱은 barrier). 당장 못 바꾸면 시험에서 terminal_area 만 명시적 허용오차로 남기고 나머지 적분은 == 로 조인다.
- [완전성 ·누락] 조각 2·3: _try_escape(384-451행, last_decision_at 갱신 포함)·_consume_due_wakes/_next_wake_time(339-382행)·review epoch(314-322행)·wt 전진(333-335행) 은 없다 — K=1·BLOCK_ARRIVAL 수준에서 발화 불가함은 코드로 확인(유휴 K=1 크레인의 거절사유가 CRANE_INTERFERENCE 일 수 없음).