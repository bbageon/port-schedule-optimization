"""[[YR-249]] 연기 — 새 시간 규칙 팔이 실제로 미루는가 (부하 15,000 하루)."""
from concurrent.futures import ProcessPoolExecutor


def one(a):
    from yard_rl.v3.stage.episode import run_episode
    r = run_episode(load=15_000, arm=a, seed=9_900_777)
    b = r.breakdown
    return (a, r.phi_krw, r.n_space, r.n_time, r.traded_edges,
            b["n_censored"], b["p90_turn_time_s"] / 60)


ARMS = ("NO_REALLOC", "FCFS", "SLOT_LL", "SPACE_TIME_LL")
print(f"{'팔':<15} {'Φ':>16} {'공간':>7} {'시간':>7} {'거래':>7} {'미완료':>7} {'P90분':>7}")
with ProcessPoolExecutor(max_workers=4) as ex:
    rows = list(ex.map(one, ARMS))
base = rows[0][1]
for a, phi, ns, nt, tr, nc, p90 in rows:
    d = "" if a == "NO_REALLOC" else f"  ({phi-base:+,.0f}원)"
    print(f"{a:<15} {phi:>16,.0f} {ns:>7,} {nt:>7,} {tr:>7,} {nc:>7,} {p90:>7.0f}{d}")
