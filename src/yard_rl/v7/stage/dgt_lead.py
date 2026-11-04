"""DGT 표본에서 빌린 트럭 통지 리드타임 — **반입·반출 따로** ([[YR-332-a]]).

리드타임 = 사전 반출입정보(코피노) 통지 → 게이트 진입. 정책이 그 트럭을
"얼마나 미리 아나" 를 정한다.

■ 출처 (읽기만 했다 — 원 파일은 추적 제외라 저장소에 없다)
  `data/raw/dgt_visits_sample_20260818_09h.jsonl`
  (원본 체크아웃 `강화학습-판매/data/raw/` · 137,546 바이트 ·
   sha256 00816921df3393395b10491eaa09fc4f4c51bb30ba7c886a1c06c6e606ada431)
  필드: `gate_in_s` − `extra.copino_notice_s` (초 단위로 그대로 뺐다).
  100건 중 통지가 있는 71건(반입 45 · 반출 26)만 쓴다 — 통지 없는 29건은 조인 실패가
  주 원인이라(`docs/data-handover/2026-08-19-DGT-1차표본-대조.md` §2) 빼 둔다.

  대조 문서의 표와 맞는다 (분위는 문서와 같은 방식 `x[int(n·p)]`):
      반입 n=45  p25 0.34h · 중앙 0.95h · p75 2.02h · 통지 전 도착 5건
      반출 n=26  p25 0.16h · 중앙 2.45h · p75 91.2h · 통지 전 도착 5건
      전체 n=71  통지 전 도착 10건 = 14.1%

■ 어떻게 뽑나 — 모수분포를 가정하지 않는다
  ① 확률 `통지 전 도착 비율`(반입 5/45 · 반출 5/26)로 **리드 0** — 통지 시각 = 도착 시각.
     실제로는 통지보다 트럭이 먼저 왔다. 수명주기 불변식(통지 ≤ 게이트 진입)을 지키려고
     **사전 정보 없이 나타나는 트럭**으로 모델링한다(YR-332 확정 결정 2).
  ② 나머지는 양수 리드 표본을 정렬해 그 사이를 **선형보간**한다(경험 분위 함수).
  ⚠️ 표본이 **1시간·71건**뿐이다. 분포 모양을 빌리는 것이지 DGT 실측 재현이 아니다.
  ⚠️ 반입 최댓값 8,474,018초(약 98일)는 이상치로 보이지만 **지우지 않았다** — 무대가
     통지를 그날 0시보다 앞에 둘 수 없어(`orders.build_stage` 의 `min(리드, 도착)`)
     어차피 "그날 시작부터 알던 트럭" 으로 눌려 담긴다.

■ 무대에서 실제로 쓰이는 값은 표본보다 짧다
  하루 무대는 통지가 그날 0시보다 앞설 수 없다. 반출 p75 91시간 같은 긴 꼬리는
  "0시에 이미 알던 트럭" 으로 잘린다. 뽑는 분포(이 모듈)와 무대에 담긴 분포는
  `tests/v7/test_v7_scaffold.py` 가 따로 잰다.
"""
from __future__ import annotations

#: 반입(GATE_IN) — 통지 전 도착 건수와 **양수 리드(초)** 정렬 표본. 출처는 머리말.
DGT_IN_NEGATIVE = 5
DGT_IN_POSITIVE_S: tuple[int, ...] = (
    7, 206, 551, 575, 975, 997, 1234, 1340, 1352, 2022, 2262, 2324, 2405, 2838,
    3225, 3286, 3289, 3406, 3668, 4575, 4577, 4671, 5091, 5994, 6023, 6258, 6936,
    7018, 7259, 7416, 8269, 8296, 8998, 9083, 9458, 10508, 59096, 77484, 264540,
    8474018)

#: 반출(GATE_OUT) — 같은 형식.
DGT_OUT_NEGATIVE = 5
DGT_OUT_POSITIVE_S: tuple[int, ...] = (
    158, 575, 836, 2333, 3556, 6149, 7172, 8406, 8805, 32829, 327614, 327614,
    327670, 327670, 328387, 328551, 329983, 330113, 331032, 349436, 415120)

#: 원 표본의 음수 리드(통지 전 도착, 초) — 시험이 원 표본 분위를 되짚을 때만 쓴다.
DGT_IN_NEGATIVE_S: tuple[int, ...] = (-1602, -1571, -1105, -565, -91)
DGT_OUT_NEGATIVE_S: tuple[int, ...] = (-1818, -836, -749, -406, -195)

_TABLE = {
    "GATE_IN": (DGT_IN_NEGATIVE, DGT_IN_POSITIVE_S),
    "GATE_OUT": (DGT_OUT_NEGATIVE, DGT_OUT_POSITIVE_S),
}


def no_notice_share(flow: str) -> float:
    """그 흐름에서 **사전 정보 없이** 오는 트럭 비율(통지 전 도착)."""
    neg, pos = _TABLE[flow]
    return neg / (neg + len(pos))


def quantile_table(flow: str) -> tuple[tuple[float, float], ...]:
    """(분위, 리드 초) 앵커 표 — 분위 0 ~ `통지 전 비율` 은 리드 0 이다."""
    neg, pos = _TABLE[flow]
    p0 = no_notice_share(flow)
    n = len(pos)
    out = [(0.0, 0.0), (p0, 0.0)]
    for i, v in enumerate(pos):
        out.append((p0 + (1.0 - p0) * (i / (n - 1)), float(v)))
    return tuple(out)


def sample_dgt_lead_s(u: float, flow: str) -> float:
    """분위 `u`∈[0,1] → 통지 리드타임(초). 흐름(`GATE_IN`·`GATE_OUT`)마다 표가 따로다.

    난수를 안에서 뽑지 않고 분위를 **받는다** — 시드에서 미리 뽑아야 재현된다
    (`terminal_stream.sample_lead_s` 와 같은 규약).
    """
    if flow not in _TABLE:
        raise ValueError(f"DGT 리드는 GATE_IN|GATE_OUT 만 — {flow!r}")
    u = min(max(float(u), 0.0), 1.0)
    p0 = no_notice_share(flow)
    if u < p0:
        return 0.0                       # 통지 전 도착 → 통지 시각 = 도착 시각
    pos = _TABLE[flow][1]
    x = (u - p0) / (1.0 - p0) * (len(pos) - 1)
    i = min(int(x), len(pos) - 2)
    w = x - i
    return float(pos[i] + (pos[i + 1] - pos[i]) * w)
