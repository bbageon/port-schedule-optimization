"""학습 정책망의 **결정이 갈릴 확률**을 잰다 ([[YR-327]] 조각 7 · 2026-09-26 검증 반박의 증거).

■ 무엇을 재나 — 한 줄로
  v5(torch·float32)와 배열판(XLA·float64)은 이 층에서 **비트 일치가 불가능**하다 (tanh 근사식과 행렬곱
  축소 순서가 둘 다 우리 코드 밖이다). 그래서 동등성은 "점수가 조금 달라도 **고르는 것은 같다**" 에 기댄다.
  그 '조금' 이 1·2위 점수 격차보다 작아야 하는데, **격차는 무대가 정하는 값**이라 보장이 없다.
  이 스크립트는 그 확률을 표본으로 잰다: v5 가 실제로 만든 후보 행과 마스크에 **가중치 수천 벌**을 먹여
  v5 의 선택과 배열판의 선택을 대조하고, 뒤집힌 건수·최소 격차·최대 |Δ| 를 보고한다.

■ 왜 이 방식인가 (on-policy 완주가 아니라 '재생')
  망이 세계를 실제로 굴리면(on-policy) 한 무대의 결정이 10~100건뿐이라 6e-06 급 확률을 밟으려면
  망 수천 벌 × 완주가 필요하고 CPU 시간이 모자란다. 여기서는 **행과 마스크는 진짜**(v5 가 만든 것)로 두고
  **가중치만** 바꿔 결정 수를 수십만으로 늘린다. 재생이 on-policy 보다 가혹한 표본이다 (실측: on-policy
  64조에서는 한 건도 안 갈렸고 최소 격차가 2.3e-05 였다).

■ 실측 (2026-09-26 · Windows CPU x64 · torch 2.14 · jax 0.11.2 · `--nets 2600` · 995초)
    후보 행 328결정분(무대 8종 · 행 합계 1,922줄) × 망 2,600벌 = **결정 852,800**  →  **뒤집힘 4건 = 4.7e-06/결정**
    최소 1·2위 격차 1.583e-08 · 최대 점수 |Δ| 9.418e-08  → **여유 0.17배**
    갈린 넷: 시드 1063 feat-crowded#21(후보 4 · v5=0 배열=1) · 2551 feat-vessel#57(후보 4 · v5=0 배열=1) ·
             2599 feat-crowded#5(후보 9 · v5=1 배열=2) · 2599 feat-crowded#14(후보 11 · v5=4 배열=5)
    규모 감각: 터미널 하루 망 호출 8,350회면 하루 기대 0.04건 · 하루가 갈릴 확률 ≈3.8% · 30일 ≈69%
  ⇒ 조각 8 의 체크포인트 대조는 '하루/한 달 해시 일치' 로 판정하면 안 된다. **첫 갈린 결정까지의 접두사
    일치 + 갈린 지점의 1·2위 격차 기록** 으로 판정한다.

■ 돌리기 (Git Bash · Windows 파이썬 · 시간 제약 없음)
    PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 .venv-jax/Scripts/python.exe \
        scripts/v6/probe_net_flip_rate.py --nets 200
    # 전수(느리다 — 수십 분): --nets 2000   ·  결과 JSON: --out outputs/v6/net_flip_rate.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

#: ★numpy 를 torch 보다 **먼저** 들인다 — 거꾸로면 OMP #15(libiomp5md.dll 중복)로 프로세스가 그냥 죽는다
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS = os.path.join(_ROOT, "tests", "v6")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="학습 정책망의 결정 갈림 확률 측정")
    ap.add_argument("--nets", type=int, default=200, help="먹일 가중치 벌 수 (실측 표는 2000)")
    ap.add_argument("--seed0", type=int, default=1, help="torch.manual_seed 의 시작값 (seed0 … seed0+nets-1)")
    ap.add_argument("--collect-net", type=int, default=101, help="후보 행·마스크를 만들 v5 구동 망의 시드")
    ap.add_argument("--stages", default="", help="쉼표로 무대 이름 제한 (기본: 전부)")
    ap.add_argument("--out", default="", help="결과 JSON 경로 (비우면 화면만)")
    a = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    from yard_rl.v6.gpu import v5net as VN
    from yard_rl.v6.gpu.geom import Geom
    from yard_rl.v6.gpu.host_convert import to_block_world
    from yard_rl.v6.ppo.model import BlockPolicy, encode

    EQ = _load("_probe_equiv", os.path.join(_TESTS, "test_gpu_v5policy_equiv.py"))
    FEAT = EQ._FEAT

    names = [s for s in (a.stages.split(",") if a.stages else list(FEAT.STAGES)) if s]
    # ── ① v5 를 굴려 **실제 후보 행 + 마스크**를 모은다 (행은 진짜, 가중치만 바꿔 먹인다)
    t0 = time.perf_counter()
    rows_all: list[tuple] = []          # (무대, 결정번호, rows (m,24) f64, mask (m,) bool)
    collect = EQ.D.build_net(seed=a.collect_net, hidden=64)
    for label in names:
        prof, scn = FEAT.STAGES[label]()
        caps, _s = FEAT._caps(scn, prof)
        w0, tb = to_block_world(prof, scn, **caps)
        g = Geom.from_profile(prof)
        end_s = float(scn.horizon_s) + float(scn.drain_window_s)
        sim, recs, exc_n, rt, _n = EQ.run_v5_net(prof, scn, w0, tb, g, end_s, collect,
                                                 max_decisions=FEAT.MAX_DECISIONS)
        assert exc_n == 0, f"{label}: v5 예외 {exc_n}"
        for i, rec in enumerate(recs):
            for p in rec["picks"]:
                rows_all.append((label, i, np.asarray(p["rows"], np.float64),
                                 np.asarray(p["mask"], bool)))
    n_rows = len(rows_all)
    print(f"■ 후보 행 수집 {n_rows}줄 (무대 {len(names)}종 · {time.perf_counter() - t0:.1f}s)")

    # ── ② 가중치를 바꿔 가며 v5(torch f32) 와 배열판(XLA f64) 의 **선택**을 대조
    scores_jit = jax.jit(lambda p, x: VN.actor_scores(p, x))
    flips: list[dict] = []
    min_gap = float("inf")
    max_delta = 0.0
    n_dec = 0
    t1 = time.perf_counter()
    for s in range(a.seed0, a.seed0 + a.nets):
        torch.manual_seed(int(s))
        pol = BlockPolicy(hidden=64)
        sd = {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}
        params = VN.load_v5_params(sd)
        for (label, i, rows, mask) in rows_all:
            x32 = encode([list(r) for r in rows], "crane")          # v5 가 망에 실제로 넣는 값 (float32)
            with torch.no_grad():
                dist = pol.distribution(x32, torch.as_tensor(mask))
                act5 = int(dist.probs.argmax())                     # runtime.py:135
            xj = jnp.asarray(np.asarray(x32.numpy(), np.float64))
            sc = np.asarray(scores_jit(params, xj))
            mj = jnp.asarray(mask)
            act6 = int(VN.greedy_action(jnp.asarray(sc), mj))
            n_dec += 1
            with torch.no_grad():
                raw5 = pol.actor(pol.trunk(x32)).squeeze(-1).numpy()
            d = float(np.abs(sc - raw5).max())
            max_delta = max(max_delta, d)
            v = np.sort(sc[mask])[::-1]
            gap = float("inf") if v.size < 2 else float(v[0] - v[1])
            min_gap = min(min_gap, gap)
            if act6 != act5:
                flips.append(dict(net_seed=s, stage=label, decision=i, n_cands=int(mask.sum()),
                                  v5=act5, arr=act6, gap=gap, delta=d,
                                  v5_scores=[float(q) for q in raw5],
                                  arr_scores=[float(q) for q in sc]))
                print(f"  ★뒤집힘 — 시드 {s} · {label} 결정#{i} · 후보 {int(mask.sum())}개 · "
                      f"v5={act5} 배열={act6} · 1·2위 격차 {gap:.3e} · |Δ| {d:.3e}")
        if (s - a.seed0 + 1) % 20 == 0:
            print(f"  … 망 {s - a.seed0 + 1}/{a.nets} · 결정 {n_dec:,} · 뒤집힘 {len(flips)} "
                  f"· 최소격차 {min_gap:.3e} · 최대|Δ| {max_delta:.3e} ({time.perf_counter() - t1:.0f}s)")

    rate = len(flips) / max(1, n_dec)
    out = dict(nets=a.nets, seed0=a.seed0, collect_net=a.collect_net, stages=names,
               rows=n_rows, decisions=n_dec, flips=len(flips), flip_rate=rate,
               min_gap=min_gap, max_delta=max_delta,
               margin=(min_gap / max_delta if max_delta else None),
               examples=flips[:8], wall_s=round(time.perf_counter() - t0, 1))
    print(f"\n■ 결과 — 결정 {n_dec:,} 중 뒤집힘 {len(flips)} = {rate:.3e}/결정")
    print(f"  최소 1·2위 격차 {min_gap:.3e} · 최대 점수 |Δ| {max_delta:.3e} "
          f"· 여유 {out['margin'] if out['margin'] is None else f'{out['margin']:.2f}'}배")
    print(f"  하루 망호출 8,350회 기준 기대 {rate * 8350:.3f}건/일 · 하루가 갈릴 확률 "
          f"≈{100 * (1 - (1 - rate) ** 8350):.1f}% · 30일 ≈{100 * (1 - (1 - rate) ** (8350 * 30)):.1f}%")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
