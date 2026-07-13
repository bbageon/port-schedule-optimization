"""ReplayRepository — recorder 산출 replay.json 로딩·조회 (04 §2.1).

UI 와 분리된 순수 로직: streamlit 미의존 (테스트 04 §8.1 대상).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_ROOT = "outputs/replays"


@dataclass(frozen=True)
class RunRef:
    run_id: str
    path: Path
    terminal_id: str
    policy_id: str
    seed: int


def scan_runs(root: str | Path = DEFAULT_ROOT) -> list[RunRef]:
    """root 아래 replay.json 을 스캔 (박제분 root/*/ + 즉석 실행분 root/live/*/)."""
    out = []
    rootp = Path(root)
    if not rootp.exists():
        return out
    for p in sorted(rootp.rglob("replay.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))["manifest"]
            out.append(RunRef(m["run_id"], p, m["terminal_id"], m["policy_id"],
                              int(m["seed"])))
        except (json.JSONDecodeError, KeyError):
            continue  # 손상 파일은 목록에서 제외 (읽기 전용 — 수리하지 않음)
    return out


@lru_cache(maxsize=8)
def load_replay(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def decision_at(replay: dict, i: int) -> dict:
    ds = replay["decisions"]
    return ds[max(0, min(i, len(ds) - 1))]


def events_window(replay: dict, t: float, half_window_s: float = 600.0,
                  kinds: set[str] | None = None) -> list[tuple]:
    """현재 시각 ±window 의 이벤트 (로그 패널용)."""
    return [(et, kind, payload) for et, kind, payload in replay["events"]
            if abs(et - t) <= half_window_s and (kinds is None or kind in kinds)]


def queue_series(replay: dict) -> tuple[list[float], list[int]]:
    """의사결정 시점별 대기 트럭 수 시계열 (타임라인 차트용)."""
    ts = [d["t"] for d in replay["decisions"]]
    ns = [d["kpis"]["waiting_now"] for d in replay["decisions"]]
    return ts, ns
