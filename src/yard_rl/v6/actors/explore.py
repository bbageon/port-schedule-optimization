"""탐색 난수 — **결정마다 독립적으로 계산**한다. 순차 난수를 쓰지 않는다.

설계 정본: `.claude/docs/architecture/04b-학습-잣대.md` §3 (동일성 불변식)

■ 왜 순차 난수(`random.Random`)면 안 되나
  반사실은 분기 시점 t 에서 세계를 복제해 다시 굴린다. 그런데 난수가 **순서에
  의존**하면 복제한 세계의 난수 상태를 실제 궤적과 맞출 수가 없다 —
  실제 궤적은 0 부터 t 까지 난수를 이미 소비했고, 새로 만든 시장은 안 그랬다.
  결과: **factual 가지가 실제와 다른 결정을 내고** 동일성 불변식이 깨진다.
  (탐색을 끄면 안 터지지만, 그건 문제가 없어서가 아니라 난수를 안 뽑아서다.)

■ 그래서 좌표로 뽑는다
      u = hash(시드, docKey, 결정시각, 무엇을뽑나) → [0, 1)
  같은 결정이면 언제 어디서 물어도 같은 값이다. 분기 세계도 같은 값을 받으므로
  **탐색을 켜도 factual 가지가 실제 궤적을 그대로 재현**한다.

■ 통계적 성질
  blake2b 8바이트를 [0,1) 로 편다. 균등성·독립성은 암호 해시 성질에 기댄다 —
  몬테카를로 품질이 필요한 자리가 아니라 **행동 다양성**을 만드는 자리다.
"""
from __future__ import annotations

import hashlib

_DENOM = float(1 << 64)


def draw(seed: int, doc_key: str, t: float, tag: str) -> float:
    """`[0, 1)` 균등값. 같은 (시드·오더·시각·태그)면 항상 같은 값이다."""
    key = f"{int(seed)}:{doc_key}:{t:.6f}:{tag}".encode()
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "big") / _DENOM


def pick(seed: int, doc_key: str, t: float, tag: str, n: int) -> int:
    """`0..n-1` 중 하나. `n <= 0` 이면 0."""
    if n <= 0:
        return 0
    return min(n - 1, int(draw(seed, doc_key, t, tag) * n))
