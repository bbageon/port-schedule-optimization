"""고전 배차 규칙을 **재배치 축으로 번역**한 대조군 5종 ([[YR-211]]).

설계 정본: `.claude/docs/architecture/06b-대조군.md` §1

■ 왜 필요한가 — **주판정 축이다**
  [06 §3](../../../../.claude/docs/architecture/06-학습과-판정.md) 동결 규약:

      주판정 축 = **규칙 대비**
      안 팔기 대비는 **판별력 0** ([[YR-185]] — 대조군도 이미 넘던 축)

  [[YR-229]] 가 안 팔기 대비 이득을 냈지만 **그 축은 기각된 축**이다.
  *"RL 이 유효하다"* 를 말하려면 **이 파일의 팔들을 넘어야** 한다.

■ ★번역 — 배차 규칙 ≠ 재배치 규칙
  고전 규칙은 *"다음에 어느 작업을"*(sequencing)인데 우리 축은 *"어디로"*(assignment)다.
  그리고 재배치는 결정이 **둘**(누구를 + 어디로)이라 목적지 규칙을 짝지어야 한 팔이 된다.

  | 팔 | 누구를 | 어디로 |
  |---|---|---|
  | `FCFS` | 먼저 **통지된** 트럭 | Least-Loaded |
  | `SPT` | 예상 작업시간 짧은 트럭 | Least-Loaded |
  | `LEAST_SLACK` | SLA 여유 최소 트럭 | Least-Loaded |
  | ~~`NEAREST`~~ | ~~(트리거된 전건)~~ | ~~**주행 증가 최소**~~ **제외 — 아래** |
  | `NETGAIN` | 순이득 내림차순 | **순이득 최대 목적지** |

■ ★공통 트리거 — 고전 규칙에는 "안 옮긴다" 가 없다
  트리거가 없으면 전건을 무차별 재배치해 **실제보다 훨씬 나빠 보인다**(공정하지 않다).
  출발 블록의 **공개 예상 대기**(크레인 밀린 일)가 상위 `TRIGGER_TOP_K` 일 때만
  후보를 만든다.

  ⚠️ **설계문서와의 차이(2026-08-25)**: 06b 는 *"NetGain·RL 도 같은 트리거"* 라고
  적었으나, v3 의 RL 팔은 **`KEEP` 을 스스로 고를 수 있어** 트리거가 필요 없고,
  이미 그 없이 학습됐다([[YR-222]]·[[YR-230]]). 지금 트리거를 RL 에 걸면 재학습이
  필요하고 [[YR-229]] 와의 비교도 깨진다.
  → **트리거는 고전 팔에만 건다.** 이 방향은 **고전 팔에 유리**하므로(무차별
  재배치를 막아 준다) 우리 주장에는 **보수적**이다. 유리하게 조작한 것이 아니다.

■ ⚠️ 알고 쓰는 비대칭 — 고전 팔은 **공간 이동만** 한다
  번역표의 목적지가 전부 **블록**이다. 시간 이연은 고전 배차 규칙에 대응물이 없다.
  그래서 RL(공간+시간) 과 고전(공간만) 사이에 **행동 폭 차이**가 남는다.
  이건 v3 가 제안하는 **메커니즘의 일부**이지만, 이겼을 때 *"구조가 좋아서"* 인지
  *"고를 게 많아서"* 인지 갈리지 않는다. **논문에 명시하고**, 필요하면 RL 을
  공간 전용으로 제한한 진단 팔을 따로 돌린다.

■ ★`NEAREST` 제외 (사용자 결정 2026-08-28) — 번역이 무너졌다
  문헌의 `Nearest` 는 *"크레인이 가장 가까운 작업부터"* 다. 이를 재배치 축으로
  옮기며 *"트럭을 주행 증가가 가장 적은 블록으로"* 로 바꿨는데, **가장 가까운
  블록이 모두에게 같은 하나**라 규칙이 규칙 노릇을 못 한다.

      Y02 -> Y01 (주행 −11초)   Y03 -> Y01 (−22초)   Y04 -> Y01 (−33초) ...
      ★21개 블록 중 20개가 전부 Y01 로 보낸다 (목적지로 쓰이는 블록 2/21)

  `Y01` 이 게이트에서 가장 가까워 **어디서 옮기든 주행이 준다**(델타가 음수).
  그러니 argmin 이 늘 `Y01` 이다. 다른 넷은 `min(load)` 라 한산한 곳으로 흩어지는데
  이 팔만 `min(route)` 라 **혼잡을 풀려고 옮기면서 한 블록에 쏟아붓는다.**
  실측(9일·시드 9,900,900): 거래는 오히려 적은데(3,972 vs 다른 팔 ~6,000)
  Φ 차이가 −5.3억(부하 12,500 에서 −16.1억 · 15,000 에서 −20.5억)이었다.

  ■ 사전등록 규약과의 관계 — **정직하게 적어 둔다**
    [[YR-211]] spec §"SPT 는 판별력이 거의 없을 수 있다" 에 *"결과를 보고 빼지
    않는다"* 가 있다. 그 조항의 취지는 **판별력이 없어도 보고하라**는 것이고,
    여기 사유는 판별력이 아니라 **구조 결함**이다 — 위 목적지 표는 `yard_layout`
    만으로 계산되며 **어떤 Φ 결과와도 무관**하게 성립한다.
    그래도 **발견 시점이 결과를 본 뒤**인 것은 사실이므로,
    ①제외 사유와 증거 ②포함했을 때의 수치를 **논문 각주로 남긴다**.
    코드도 지우지 않고 `RETIRED_ARMS` 로 남겨 재현할 수 있게 둔다.

■ ★도착시각 조정을 쓰는 팔 둘 ([[YR-249]] · 2026-08-29)
  [[YR-247]] 실측이 *"우세의 원인은 정책 품질이 아니라 **행동 폭**"* 임을 보였다 —
  앞 넷은 전부 블록 재배치만 해서 절감이 −0.48~+0.53%로 사실상 0인데, 행동 폭을
  블록으로 맞춘 학습 정책(`RL_SPACE`)도 −1.31%였다. 그래서 **도착시각 조정을 쓰는
  규칙**이 없으면 *"같은 행동 폭에서도 학습이 낫다"* 를 물었을 때 답이 없다.

  | 팔 | 반입(공간) | 반출·전체(시간) |
  |---|---|---|
  | `SLOT_LL` | 안 함 | **가장 한산한 칸**으로 미룬다 |
  | `SPACE_TIME_LL` | 가장 한산한 블록 | **가장 한산한 칸** |

  ★번역이 자의적이지 않다 — 공간 축에서 이미 쓰는 **Least-Loaded** 를 축만 바꿔
  적용한 것이다. 트럭 예약제 문헌의 표준 규칙(*"가장 한산한 시간대로 배정"*)과 같은
  꼴이고, 새 방법론이 아니다. [[YR-211]] 이 *"시간 이연은 고전 **배차** 규칙에
  대응물이 없다"* 고 적은 것은 맞지만, 그 판단은 배차 규칙에 한정된 것이고 **예약제
  문헌에는 대응물이 있다** — 그때 결정을 뒤집는 게 아니라 축을 넓히는 것이다.

  ★"한산함" 은 `announced_around` 로 잰다 — 그 시각 ±창에 그 블록으로 **통지된**
  물량이다. 공개 정보이고 학습 정책의 도착압력 칸과 **같은 함수**라 잣대가 같다.

■ 정보 경계 — 전 팔이 **공개 정보만** 쓴다
  통지 시각(`copino_notice_s`)·공개 예정(`in_out_reserve_s`)·현재 내부 대수·
  크레인 밀린 일만 본다. **실현 도착·실현 완료는 한 줄도 안 읽는다**
  (FIFO 누출을 반복하지 않는다 — [[YR-107]]).
"""
from __future__ import annotations

from ..features.block import announced_around
from ..reward.krw import KRW_TRUCK_HOUR
from .market import EpochResult, Market
from .offer import RESOLVER_KEEP, SPACE, TIME
from .resolver import ResolveResult, Trade

#: 구현된 고전 팔. **`NEAREST` 는 뺐다** — 사유는 아래.
#: 앞 넷은 **블록 재배치만**, 뒤 둘은 **도착시각 조정**을 쓴다([[YR-249]]).
ARM_RULES: tuple[str, ...] = ("FCFS", "SPT", "LEAST_SLACK", "NETGAIN",
                              "SLOT_LL", "SPACE_TIME_LL")

#: ★도착시각 조정을 쓰는 팔 — 행동 폭을 학습 정책에 맞춘 대조군.
TIME_ARMS: frozenset = frozenset({"SLOT_LL", "SPACE_TIME_LL"})

#: ★제외된 팔 — 코드는 남기되 판정에서 뺀다(재현·검증을 위해 지우지 않는다).
RETIRED_ARMS: tuple[str, ...] = ("NEAREST",)

#: ★공통 트리거 — 출발 블록의 공개 예상 대기가 **상위 이 비율**일 때만 후보 생성.
#: 착수 시 동결(2026-08-25). ⚠️ 결과를 보고 바꾸지 않는다.
TRIGGER_TOP_K = 0.30

#: 순이득 계산에 쓰는 1건당 평균 서비스 시간(초) — 무대 기준값.
#: 공개 상수이지 실현값이 아니다.
SERVICE_REF_S = 180.0


def crane_backlog_s(sim, t: float) -> float:
    """블록의 **공개 예상 대기** — 크레인에 밀려 있는 일(초).

    터미널이 자기 크레인 상태를 아는 것은 공개 정보다. 실현 도착이 아니다.
    """
    return sum(max(0.0, sim.fleet.get(c.crane_id).state.available_at - t)
               for c in sim.profile.cranes)


def inside_now(mbt, bid: str, t: float, records) -> int:
    """블록 안 대수 — `features.block.inside_count` 와 같은 정의(공개)."""
    sim = mbt.blocks[bid]
    n = 0
    for jid in sim.jobs:
        rec = records.get(jid)
        if rec is None or rec.gate_in_s is None or rec.gate_in_s > t:
            continue
        if rec.gate_out_s is None or rec.gate_out_s > t:
            n += 1
    return n


class _Trail:
    """고전 팔에는 행위자가 없다 — 계수만 맞춰 준다(에피소드가 결정 수를 센다)."""

    def __init__(self):
        self.trail: list = []


class ClassicalMarket(Market):
    """고전 규칙 팔 — `Market` 과 **같은 자리에 꽂힌다**(`step` 이 같은 모양을 낸다).

    `Market` 을 상속해 `decided`·`newly_eligible`·창 규칙을 그대로 쓴다.
    다른 것은 **누가 고르느냐** 뿐이다 — 학습 망 대신 규칙이 고른다.
    """

    def __init__(self, arm: str, layout, *, window_s: float = 1800.0,
                 trigger_top_k: float = TRIGGER_TOP_K):
        if arm not in ARM_RULES:
            raise ValueError(f"고전 팔이 아니다: {arm!r} — {ARM_RULES}")
        # 부모의 seller/buyer/resolver 는 안 쓴다. 계수용 껍데기만 둔다.
        super().__init__(_Trail(), _Trail(), None, window_s=window_s)
        self.arm = arm
        self.layout = layout
        self.trigger_top_k = float(trigger_top_k)

    # ------------------------------------------------------------------ 트리거
    def _triggered_blocks(self, mbt, t: float) -> set:
        """공개 예상 대기 상위 `k` 비율의 블록들. 동점은 이름순으로 깬다."""
        pairs = sorted(((crane_backlog_s(sim, t), bid)
                        for bid, sim in mbt.blocks.items()),
                       key=lambda p: (-p[0], p[1]))
        n = max(1, int(round(len(pairs) * self.trigger_top_k)))
        return {bid for _b, bid in pairs[:n]}

    # ------------------------------------------------------------------ 우선순위
    def _priority(self, doc_key: str, o, mbt, t: float, records) -> tuple:
        """**누구를 먼저 옮길까** — 작을수록 먼저. 동점은 `doc_key` 로 깬다."""
        if self.arm == "FCFS":
            return (float(o.copino_notice_s), doc_key)
        if self.arm == "SPT":
            # ★이 무대에서 SPT 는 **판별력이 없다** — 코드로 확인한 사실이다:
            #   ① `Order` 에 작업시간 필드가 없다(공개 정보는 통지시각·예정시각뿐)
            #   ② 엔진의 서비스 시간은 **기하학**(갠트리·트롤리 주행)으로 정해지고
            #      컨테이너 규격(20/40ft)은 **적치 슬롯 탐색에만** 쓰인다 —
            #      `duration_s = dist/gantry + t_dist/trolley` 에 규격 항이 없다.
            #   → 우선순위가 `doc_key` 순, 즉 **사실상 임의 순서**가 된다.
            #
            # 야적 위치를 읽으면 기하 기반 SPT 를 만들 수는 있으나, 그러면 SPT 가
            # **RL 보다 많은 정보**를 보게 되어 비교가 반대로 기운다(RL 특징에는
            # 야적 좌표가 없다). 정보 계층을 맞추는 쪽을 택한다.
            #
            # ⚠️ **그래도 팔에서 빼지 않는다** — 06b 사전등록: *"이 무대에서는 SPT 가
            #    무의미하다" 도 보고할 결과다. 결과를 보고 빼지 않는다.*
            return (SERVICE_REF_S, doc_key)
        if self.arm == "LEAST_SLACK":
            # 여유 = (공개 예정까지 남은 시간) − (그 블록의 밀린 일)
            sim = mbt.blocks.get(o.con_loc)
            back = crane_backlog_s(sim, t) if sim is not None else 0.0
            return (float(o.in_out_reserve_s - t) - back, doc_key)
        return (0.0, doc_key)          # NEAREST·NETGAIN 은 순서를 안 정한다

    # ------------------------------------------------------------------ 목적지
    def _dest(self, mbt, src: str, t: float, records, reserved: dict):
        """**어디로 보낼까** — 못 보내면 `None`.

        `reserved` 는 이번 epoch 에 이미 보낸 대수(같은 블록에 몰리는 것을 막는다).
        """
        cands = []
        for dst in sorted(b for b in mbt.blocks if b != src):
            if mbt.free_slots(dst) - reserved.get(dst, 0) <= 0:
                continue
            load = inside_now(mbt, dst, t, records) + reserved.get(dst, 0)
            route = self.layout.pre_gate_route_delta_s(src, dst)
            cands.append((dst, load, route))
        if not cands:
            return None

        if self.arm == "NEAREST":
            best = min(cands, key=lambda c: (c[2], c[0]))          # 주행 증가 최소
        elif self.arm == "NETGAIN":
            src_load = inside_now(mbt, src, t, records)
            def gain(c):
                # 순이득 = (혼잡 해소로 아낀 대기) − (늘어난 주행). 전부 공개값.
                relief = (src_load - c[1]) * SERVICE_REF_S
                return (relief - c[2]) * KRW_TRUCK_HOUR / 3600.0
            best = max(cands, key=lambda c: (gain(c), -c[1], c[0]))
            if gain(best) <= 0.0:
                return None                                         # 이득 없으면 안 판다
        else:
            best = min(cands, key=lambda c: (c[1], c[0]))          # Least-Loaded
        return best

    # ------------------------------------------------------------------ 한 판
    def step(self, mbt, t: float, *, orders, records, end_s: float,
             time_slots_of=None, quay_of=None, slot_capacity_left=None,
             epoch_s: float = 60.0) -> EpochResult:
        res = EpochResult()
        res.newly_eligible = self.newly_eligible(
            mbt, t, orders=orders, records=records, epoch_s=epoch_s)
        res.resolve = ResolveResult()
        if not res.newly_eligible:
            return res

        hot = self._triggered_blocks(mbt, t)
        picks, time_picks = [], []
        does_time = self.arm in TIME_ARMS
        does_space = self.arm != "SLOT_LL"
        for doc_key in res.newly_eligible:
            o = orders[doc_key]
            src = o.con_loc
            if src not in mbt.blocks:
                continue
            self.seller.trail.append({"t": t, "doc_key": doc_key})   # 결정 계수
            if src not in hot:
                self.decided.add(doc_key)
                continue
            if o.is_inbound and does_space:
                # 반입 — 블록 재배치 (v3 Seller 와 같은 물리 제약)
                picks.append((self._priority(doc_key, o, mbt, t, records), doc_key))
            elif does_time:
                # ★반출(또는 시간 전용 팔) — 도착시각 조정
                time_picks.append((self._priority(doc_key, o, mbt, t, records), doc_key))
            else:
                self.decided.add(doc_key)

        reserved: dict = {}
        for _pri, doc_key in sorted(picks):
            o, rec = orders[doc_key], records[doc_key]
            src = o.con_loc
            best = self._dest(mbt, src, t, records, reserved)
            if best is None:
                self.decided.add(doc_key)
                rec.con_swap_reason = RESOLVER_KEEP
                res.resolve.kept.append(doc_key)
                continue
            dst, _load, route = best
            reserved[dst] = reserved.get(dst, 0) + 1
            res.resolve.trades.append(Trade(
                doc_key=doc_key, src_block=src, coord_key=f"SPACE@{dst}",
                kind=SPACE, dst_block=dst, slot=None,
                route_delta_s=float(route), defer_s=0.0))
            self.decided.add(doc_key)
            rec.record_swap(prev_block=src, reason="SPACE")

        # ── ★도착시각 조정 — 가장 한산한 칸으로 ([[YR-249]])
        taken: dict = {}
        for _pri, doc_key in sorted(time_picks):
            o, rec = orders[doc_key], records[doc_key]
            src = o.con_loc
            slots = list(time_slots_of(doc_key, t)) if time_slots_of else []
            best = self._slot(mbt, src, slots, orders, taken, slot_capacity_left)
            if best is None:
                self.decided.add(doc_key)
                rec.con_swap_reason = RESOLVER_KEEP
                res.resolve.kept.append(doc_key)
                continue
            k, start, defer = best
            taken[k] = taken.get(k, 0) + 1
            res.resolve.trades.append(Trade(
                doc_key=doc_key, src_block=src, coord_key=f"TIME@{k}",
                kind=TIME, dst_block=None, slot=int(k),
                route_delta_s=0.0, defer_s=float(defer)))
            self.decided.add(doc_key)
            rec.record_swap(prev_block=src, reason="TIME")
        return res

    # ------------------------------------------------------------------ 시간 칸
    def _slot(self, mbt, src, slots, orders, taken, slot_capacity_left):
        """**언제로 미룰까** — 가장 한산한 칸. 못 미루면 `None`.

        한산함은 `announced_around`(그 시각 ±창에 통지된 물량)로 잰다. 이번 epoch 에
        이미 그 칸으로 보낸 대수(`taken`)를 더해 **같은 칸에 몰리는 것을 막는다** —
        공간 축의 `reserved` 와 같은 장치다.
        """
        cands = []
        for k, start, defer in slots:
            if slot_capacity_left is not None:
                left = slot_capacity_left(k)
                if left is not None and left - taken.get(k, 0) <= 0:
                    continue
            load = announced_around(mbt, src, start, orders) + taken.get(k, 0)
            cands.append((load, defer, k, start))
        if not cands:
            return None
        load, defer, k, start = min(cands)      # 한산한 곳 → 동점이면 덜 미루는 쪽
        return k, start, defer
