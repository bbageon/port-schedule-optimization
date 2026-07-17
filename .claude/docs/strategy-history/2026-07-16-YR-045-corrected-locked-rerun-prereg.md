# YR-045 — 정정판 locked 재실험 사전등록

> 상태: **실행 전 동결**, YR-050 완료 전 실행 금지. 실행 뒤에는 이 본문을 고치지 않고
> `## 실행 결과` 절만 추가한다. 코드·Dashboard 상태가 실제 착수를 결정한다.

## 1. 결정 경위

YR-039의 승리는 두 이유로 무효가 됐다. 목적함수의 불균형 항 하나가 총비용의 97.6~99.9%를
차지했고, 비교 정책은 결정의 54~81%를 실작업이 아닌 위치 이동에 썼다. YR-043은 비용을,
YR-044는 비교 정책을 정정했다. YR-048은 빠져 있던 ETA를 넣었지만 다음 사실도 드러냈다.

- ETA는 PRE_REHANDLE과 미래 도착 bay REPOSITION을 동시에 활성화한다.
- 현재 엔진은 SERVE가 없는 한산기에 ETA만으로 결정 시점을 열지 않는다(YR-050).
- YR-044의 93.4/96.6은 ETA 주입 전 보정값이며 현재는 62.927/79.643이다.

따라서 이번 실행은 “RL이 총비용 하나를 이겼는가”가 아니라 두 ETA 경로를 분리하고 여러 운영
지표를 동시에 통과하는지를 판단한다.

## 2. 실행 선결조건

1. **비교 baseline이 신규 seed 대역에서 라이브락 없이 완주한다** (YR-051에서 조합 절단이 WAIT를
   떨궈 완료율 0%가 되던 결함을 수정했으나, 새 seed 400k/410k/420k에서 재확인한다 — 절단이 잦은
   고밀도 후보 상태에서 완주·건전성 통과를 arm별로 계측). baseline이 실패한 seed는 비교 대상에서
   제외하지 않고 원인을 규명한다.
2. YR-050이 ETA 기반 결정 시점과 연착 ETA feature 의미를 테스트로 고정한다.
3. Windows 순수 파이썬 전체와 WSL torch 테스트가 통과한다.
4. 실행 tree는 clean commit이며 manifest에 commit·dirty=false를 기록한다.
5. 비용 scale은 train 기준정책으로만 맞춘 뒤 validation/test에서 재계산하지 않는다.

## 3. 동결 seed와 시나리오

기존 300000~320059 대역은 학습·진단·보정에 사용돼 모두 폐기한다.

| 용도 | seed | 개수 |
|---|---|---:|
| train | 400000~400499 | 500 |
| validation | 410000~410019 | 20 |
| locked test | 420000~420059 | 60 |

- 프로파일: `POC-MULTI`, 가정값 표시 유지.
- 정보수준: PRE_ADVICE.
- ETA: `eta_error_s=300초`, `Random(f"eta:{seed}")` 전용 흐름.
- 평균조건 변주: TruncatedNormal, σ=12%, ±2σ 절단.
- λ_vessel=1.0 중립; 고부하·타이트 마감은 YR-041 별도.

## 4. 정책과 제거 실험

학습 정책은 Candidate DDQN을 기본으로 DQN·Dueling을 같은 예산으로 비교한다. 선택은 예상 누적
비용 `Q_cost`의 argmin이며 물리·정보 제약은 mask와 중앙 resolver가 책임진다.

비교군은 JointRolloutGreedy(600초), BeamLookahead, ServiceFirstSPT, FIFO다. 모든 정책은 같은
후보 생성기·resolver·비용 설정을 사용한다. 후보 조합 축소 횟수는 정책·seed별로 반드시 보고한다.

| arm | provided ETA | PRE_REHANDLE | 의미 |
|---|---|---|---|
| NO_ETA | 제거 | 자동 미발행 | ETA 두 경로가 모두 없는 기준 |
| ETA_NO_PRE | 유지 | 명시 차단 | ETA 기반 REPOSITION 순효과 |
| FULL | 유지 | 허용 | PRE_REHANDLE의 추가 순효과 |

**전략적 WAIT ablation (YR-052 보류 결정 판정용)**: 위 각 arm에서 `전략적 WAIT 허용/금지`를
교차한다. 금지 arm은 실작업 조합이 공동 실행가능하면 WAIT-포함 조합을 argmin 후보에서 제외하되
구조적 WAIT(경합 양보·NO_FEASIBLE)은 보존한다. val 6-seed 예비측정에서 전략적 WAIT 금지가
JointRollout 총비용을 평균 −4.24(5/6 seed) 낮췄으나 소각 안 된 대역이라 신규 seed로 재판정한다.
강제 WAIT(양보)은 계약상 필수라 어느 arm에서도 제거 대상이 아니다 (근거: YR-052 문서).

`ETA_NO_PRE - NO_ETA`는 위치선점, `FULL - ETA_NO_PRE`는 선제 재조작의 기여다. FULL과
NO_ETA만 비교해 H2를 판정하지 않는다.

## 5. 학습·선택 규칙

- variant별 500 episode, checkpoint 25 episode마다 validation 20 seed 평가.
- ε=`1/sqrt(ep)`, replay 50k, batch 64, target 동기화 500 step.
- checkpoint는 validation 총비용 최저값으로만 선택하며 locked test는 마지막에 한 번 연다.
- test 결과를 본 뒤 variant·checkpoint·가중치·seed를 바꾸면 해당 실행은 탐색 결과로 강등한다.

## 6. 동시 판정 게이트

다음을 모두 만족해야 채택 후보가 된다.

1. 평균 트럭 대기 감소.
2. P95 대기(트럭 100대 중 오래 기다린 상위 5대 수준) 비악화.
3. 본선 완료·STS 대기·이송장비 대기 비악화.
4. 크레인 이동 또는 재조작 중 하나 이상 개선.
5. 작업 완료율 100%, backlog 0, 물리·정보·공동제약 위반 0.
6. 비용 단일 항 기여율 70% 이하, 행동분포 건전성 문턱 통과.

총비용 신뢰구간만 좋아서는 채택하지 않는다. PRE_REHANDLE은 건전성 지표에서 비-SERVE로 세므로
선택 횟수·SERVE 가능 상태에서의 선택 횟수·REPOSITION과의 비율을 별도 공개한다. 문턱 실패를
사후 해석으로 면제하지 않으며, 오탐 증거가 생기면 새 사전등록으로 계약을 다시 설계한다.

## 7. 필수 산출물과 중단 조건

- seed별 원자료, paired 95% 신뢰구간, 비용 항목별 기여율.
- 행동 4종 분포, PRE_REHANDLE·REPOSITION 후보/선택 수, 조합 축소 횟수.
- 완료·backlog·제약 위반·P95·본선·STS·이송 지표.
- arm별 scenario meta와 ETA 설정, git·환경 manifest.

선결 테스트 실패, dirty tree, seed 중복, scale 재적합, 결정론 불일치가 하나라도 있으면 locked
실행을 중단한다. 결과가 baseline을 못 넘으면 QMIX로 바로 넘어가지 않고 feature·비용·후보 누락을
재감사한다.

## 실행 결과 (2026-07-18, 위 본문 무수정)

집행 코드 `f530ae4`(+검사 정밀화 `f928807`·결함 정정 `e278ab8`), 원자료
`outputs/reports/yr045_locked_rerun/`. 총 2시간 38분(병렬 12 worker). 선결: 신규 3 대역
90 seed×3 arm 라이브락 0(절단 0~333회에도 완주 100%)·지배도 최대 항 48.6%·스위트 348 passed.

### 판정 — RL 채택 실패, YR-039 무효 확증

- **RL 3 variant × 6 조건(arm×WAIT) 36개 게이트 행 중 통과 0.** FULL/allow 총비용:
  dueling 90.95 vs JointRollout 70.78 (+28%). RL 최선 조건(FULL/forbid 86.31)도 +25%.
  정정된 목적함수·건전 baseline·신규 seed에서 YR-039 의 승리는 어떤 형태로도 재현되지 않았다.
- 6중 게이트 전부 통과(채택 후보 정의)는 **SF-SPT 4조건·FIFO 1조건** — 총비용은 rollout 이
  낮지만(70.8 vs 81.7) 게이트 축(평균대기↓·P95·이동/재조작)에서는 고정규칙이 rollout 을 이긴다.
  즉 "총비용 최적"과 "다중 운영지표 우월"이 분리된다 (§6 설계가 의도한 바로 그 구분).
- RL 결함 실측: allow 조건에서만 미완주 6건(트럭 1대 방치, backlog=1) — 전략적 WAIT 오용.
  dqn·ddqn 은 행동 건전성 문턱도 다수 조건 실패.

### ETA 두 경로 분리 (paired Δ 총비용, 60-seed 95% CI)

- **위치선점(REPOSITION) 경로가 지배적**: JointRollout −29.1 [−30.7, −27.6], SF −23.6, RL −10~−27.
- **선제 재조작(PRE) 경로는 실재·유의**: 고정규칙 −1.6~−2.9 (CI 전체 음수), RL forbid −2.9~−4.0
  (CI 전체 음수; allow 는 분산 커서 dueling 만 유의 −8.6 [−16.8, −2.3]). H2 의 "선제 재조작이
  더해주는 가치" 는 방향성 있게 확인 — 단 크기는 위치선점의 1/7~1/10.
- FULL−NO_ETA 단독 비교는 사전등록대로 판정에 사용하지 않음.

### 전략적 WAIT (YR-052 판정 데이터)

전 정책·전 arm 에서 금지 ≥ 허용: JointRollout −0.8~−1.6, RL **−4.0~−15.7** (dqn ETA_NO_PRE
107.5→91.9). RL 미완주 6건은 전부 허용 조건 — RL 이 전략적 대기를 학습으로 오용함을 확정.
설계 결정(행동공간 유지 여부)은 사용자 보류 유지 — 데이터는 YR-052 문서 갱신.

### 실행 중 발견·정정 (경위 보존)

1. **run_episode 가 generator 를 capture 에 미전달** (YR-039 부터의 잠복 결함) — RL 의
   ETA_NO_PRE 평가가 조용히 FULL 과 동일해짐. locked 원자료 대조(60/60 seed 완전일치)로
   기계 검출 → `e278ab8` 정정 + 회귀 테스트 → 오염 6조건만 재계산(선택은 FULL val 기준이라
   불변 — 결과 기반 선택 아님). 정정 전 수치는 git 이력에만 있고 본 결과에 미사용.
2. **BEAM 이 JointRollout 과 60/60 seed 완전 동일** — 2단 lookahead 가 결정을 한 번도 바꾸지
   않음. 결함 또는 실질 무가치 여부는 YR-055 로 분리.
3. clean-tree 검사가 실험 자신의 산출물(추적 대상 outputs/)에 오탐 → out_dir 밖 변경만
   판정하도록 정밀화(`f928807`), manifest 에 근거 기록.

### 후속 (§7 지시 이행)

QMIX(YR-013)로 넘어가지 않는다. **YR-054 — RL 재감사**: 전략적 WAIT 오용(미완주)·짧은
학습예산 대비 rollout 격차·NO_ETA 에서 dqn 폭주(134.9)·feature/비용/후보 누락 순으로.
BEAM 동일성은 **YR-055**. 실운영 주장 게이트(YR-002/009)는 계속 유효 — 본 결과는 합성·가정
조건의 구현 증거다.
