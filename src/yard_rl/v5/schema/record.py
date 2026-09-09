"""② 실행 기록 — 이벤트가 끝날 때마다 터미널이 전송하는 사실.

설계 정본: `.claude/docs/architecture/01-오더-스키마.md` §3

■ 기록의 주인은 오더다
  별도 장부에 두고 오더가 참조하는 구조가 아니다. 비용·KPI 는 이 기록들을
  **읽어서** 적분한다 — **적분이 원본을 갖지 않는다.**

■ 넷은 터미널이 전송한다
  gateIn · blockIn · jobDone · gateOut. 기사가 그 지점을 통과하면 온다.
  **그 시각이 지나야 칸이 차므로** 정책이 오더를 읽어도 미래를 못 본다 —
  가려놓은 게 아니라 **아직 안 온 것**이다.

■ serviceStart 는 전송되지 않는다
  DGT 실데이터에도 없다. 시뮬레이터라서 알 뿐이고, 보상의 크레인 점유 계산과
  "누가 대기 중인가" 판정에만 쓴다. 실데이터로 가면 사라지는 항이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .lifecycle import STAGE_FIELD, LifecycleError, Stage, validate

#: **터미널이 남기는 기록 7** — 시각 5 + 교체 2. 계약(`record_fields_target = 7`)
#:
#: 아래 두 종류는 이 7 에 **안 들어간다**. 성격이 달라서다.
#:   · `copino_notice_s`  — 오더가 원본을 갖는다. 여기 것은 편의 사본이다.
#:   · `rehandles` · `yc_extra_move_s` — 이벤트가 아니라 **비용 집계용 계수기**다.
TRANSMITTED_FIELDS: tuple[str, ...] = (
    "gate_in_s", "block_in_s", "service_start_s", "job_done_s", "gate_out_s",
    "prev_con_loc", "con_swap_reason",
)
RECORD_FIELDS = len(TRANSMITTED_FIELDS)      # = 7

SWAP_SPACE = "SPACE"     # 공간 판매로 블록이 바뀜
SWAP_TIME = "TIME"       # 시간 이연 — 블록은 그대로, 예약 슬롯만 바뀜


@dataclass
class ExecutionRecord:
    """오더 1건의 실행 사실. 초기값은 전부 `None` — **미발생은 비어 있다.**"""

    doc_key: str

    # ── 라이프사이클 다섯 시각 (코피노는 오더 수신이라 오더에서 복사해 둔다)
    copino_notice_s: float | None = None
    gate_in_s: float | None = None
    block_in_s: float | None = None
    job_done_s: float | None = None
    gate_out_s: float | None = None

    # ── 터미널이 전송하지 않는 값 (시뮬레이터 내부 관측)
    service_start_s: float | None = None

    # ── 재배치 이력
    prev_con_loc: str | None = None
    con_swap_reason: str | None = None

    #: 실제로 일어난 재조작 횟수 — 비용 항3 의 원료
    rehandles: int = 0
    #: 재배치·재조작으로 **생긴** YC 추가 이동 시간(초) — 비용 항2 의 원료.
    #: 생산 사이클은 세지 않는다(넣으면 "아무것도 안 하기" 가 최적이 된다).
    yc_extra_move_s: float = 0.0

    _stamped: set = field(default_factory=set, repr=False, compare=False)

    # ------------------------------------------------------------------ 이벤트 기록
    def stamp(self, stage: Stage, t: float, *, validate_now: bool = True) -> None:
        """이벤트가 끝나는 순간 **그때의 `t` 를 그대로** 적는다.

        되돌아가기 없음 — 같은 단계를 두 번 찍으면 `LifecycleError`.
        나중에 "아마 이때쯤" 으로 되계산하지 않는다.
        """
        if stage in self._stamped:
            raise LifecycleError(
                f"{self.doc_key}: {STAGE_FIELD[stage]} 재기록 — 각 단계는 한 번뿐이다")
        setattr(self, STAGE_FIELD[stage], float(t))
        self._stamped.add(stage)
        if validate_now:
            validate(self)

    def observe_service_start(self, t: float) -> None:
        """크레인이 이 트럭을 잡은 순간 — **터미널 전송이 아니다**(시뮬레이터 관측)."""
        if self.service_start_s is not None:
            raise LifecycleError(f"{self.doc_key}: service_start 재기록")
        if self.block_in_s is None:
            raise LifecycleError(f"{self.doc_key}: 블록 진입 전에 서비스 시작 불가")
        self.service_start_s = float(t)

    def record_swap(self, *, prev_block: str, reason: str) -> None:
        """재배치가 확정될 때 옛 블록을 보관한다. 오더의 `conLoc` 은 새 블록이 된다."""
        if reason not in (SWAP_SPACE, SWAP_TIME):
            raise ValueError(f"{self.doc_key}: 알 수 없는 교체 사유 {reason!r}")
        self.prev_con_loc = prev_block
        self.con_swap_reason = reason

    # ------------------------------------------------------------------ 조회
    @property
    def occupy_s(self) -> float | None:
        """크레인 점유 = 작업완료 − 서비스시작. **이것만이 진짜 점유**다.

        `blockIn → jobDone` 을 쓰면 줄 서서 기다린 시간까지 점유로 센다.
        """
        if self.job_done_s is None or self.service_start_s is None:
            return None
        return self.job_done_s - self.service_start_s

    def waiting_at(self, t: float) -> bool:
        """시각 `t` 에 이 트럭이 **줄 서 있었나** — `blockIn ≤ t < serviceStart`.

        정책이 보는 `inside`(블록 안 전부)와 다르다. 채점은 이쪽을 쓴다.
        """
        if self.block_in_s is None or self.block_in_s > t:
            return False
        return self.service_start_s is None or t < self.service_start_s
