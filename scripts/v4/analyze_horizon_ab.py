"""YR-299 B — 반사실 창 3시간 vs 6시간. **사전등록(de12b15)한 기준 그대로** 판정한다."""
from __future__ import annotations
import json, pathlib, statistics as st

ROOT = pathlib.Path("outputs/v4/horizon")

def load(tag):
    h = json.loads((ROOT / tag / "history.json").read_text(encoding="utf-8"))
    m = json.loads((ROOT / tag / "month.json").read_text(encoding="utf-8"))
    return h, m

print("=" * 74)
print("YR-299 B · 반사실 창 3시간 vs 6시간 (시드 9,900,996 · v4 무대)")
print("=" * 74)

got = {}
for tag in ("h3", "h6"):
    h, m = load(tag)
    tr = [r for r in h if r.get("n_labels")]
    last5 = tr[-5:]
    first5 = tr[:5]
    zero = [r.get("zero_label_ratio") for r in tr if r.get("zero_label_ratio") is not None]
    got[tag] = dict(
        n=len(tr), labels=sum(r.get("n_labels", 0) for r in tr),
        val_s0=st.mean([r["val_seller_loss"] for r in first5 if r.get("val_seller_loss")] or [0]),
        val_s1=st.mean([r["val_seller_loss"] for r in last5 if r.get("val_seller_loss")] or [0]),
        val_b0=st.mean([r["val_buyer_loss"] for r in first5 if r.get("val_buyer_loss")] or [0]),
        val_b1=st.mean([r["val_buyer_loss"] for r in last5 if r.get("val_buyer_loss")] or [0]),
        z0=first5[0].get("gap_ratio"), z1=last5[-1].get("gap_ratio"),
        zero_mean=st.mean(zero) if zero else None,
        space=sum(r.get("n_space", 0) for r in tr),
        time=sum(r.get("n_time", 0) for r in tr),
        phi_rl=sum(r.get("phi_rl", 0) for r in tr),
        phi_keep=sum(r.get("phi_keep", 0) for r in tr),
        secs=sum(r.get("secs", 0) for r in tr),
        worlds=sum(r.get("worlds", 0) for r in tr))

print(f"\n  {'항':<24}{'창 3시간':>14}{'창 6시간':>14}{'변화':>12}")
rows = [("학습 회차", "n", "{:.0f}"), ("라벨 수", "labels", "{:,.0f}"),
        ("제안망 검증(첫5)", "val_s0", "{:.3f}"), ("제안망 검증(끝5)", "val_s1", "{:.3f}"),
        ("수락망 검증(첫5)", "val_b0", "{:.3f}"), ("수락망 검증(끝5)", "val_b1", "{:.3f}"),
        ("★차이0 라벨 비율", "zero_mean", "{:.1%}"),
        ("공간 거래", "space", "{:,.0f}"), ("시간 거래", "time", "{:,.0f}"),
        ("반사실 세계 수", "worlds", "{:,.0f}"), ("학습 시간(초)", "secs", "{:,.0f}")]
for lbl, k, fmt in rows:
    a, b = got["h3"].get(k), got["h6"].get(k)
    if a is None or b is None:
        continue
    chg = "—" if not a else f"{(b-a)/a:+.1%}"
    print(f"  {lbl:<24}{fmt.format(a):>14}{fmt.format(b):>14}{chg:>12}")

print("\n" + "=" * 74)
a, b = got["h3"], got["h6"]
better = b["val_s1"] < a["val_s1"] and b["val_b1"] < a["val_b1"]
worse = b["val_s1"] > a["val_s1"] * 1.2 or b["val_b1"] > a["val_b1"] * 1.2
if better:
    print("판정 — **6시간이 낫다.** 더 멀리 보면 더 잘 배운다.")
elif worse:
    print("판정 — **6시간이 나쁘다.** 멀리 볼수록 신호가 흐려진다.")
else:
    print("판정 — **차이 없다.** 3시간이면 충분하다 — 지금 설정에 근거가 생긴다.")
print("=" * 74)
