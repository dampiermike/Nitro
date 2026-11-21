"""
Pyramid PMP audit — measure the cost of de-margining the pyramid.

Flag 1 fix: rescale get_pmp ceiling to 100% capital.
    original  : VIX<20 -> 1.30 , VIX<32 -> 1.10 , else 0.50   (up to 130% = margin)
    no-margin : VIX<20 -> 1.00 , VIX<32 -> 0.85 , else 0.50   (never > 100%)

We swap the PMP bands on the UNMODIFIED v16 engine and the v17 engine and
re-run the pyramid-active modes (5 = pyramid, 7 = decay+pyramid). Modes 4a/4b
have no pyramid, so they are identical to the earlier run and omitted.
"""
import os, sys
import numpy as np

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)

import nitro_v16
import nitro_v17_backtest as v17b
from nitro_v16 import load_data, compute_dew, trade_stats, fmt_eq


def make_pmp(a, b, c):
    def f(vix):
        v = vix if not np.isnan(vix) else 15.0
        if v < 20:  return a
        if v < 32:  return b
        return c
    return f

PMP_ORIG     = make_pmp(1.30, 1.10, 0.50)   # margin (current)
PMP_NOMARGIN = make_pmp(1.00, 0.85, 0.50)   # rescaled ceiling to 100%


def run_v16(df, mode, pmp):
    nitro_v16.get_pmp = pmp
    return nitro_v16.run_engine(df, mode=mode)

def run_v17(df, mode, pmp):
    v17b.get_pmp = pmp
    return v17b.run_engine_v17(df, mode=mode)


def row(tag, r):
    s = trade_stats(r["trades"], r["eq"], r["dret"])
    extra = ""
    if "n_cap" in r:
        extra = f" | Cap={r['n_cap']} MR={r['n_mr']}"
    return (s, f"  {tag:20s} {s['n']:>5d} {fmt_eq(s['final']):>20s} "
               f"{s['cagr']*100:>8.2f}% {s['sharpe']:>8.3f} {s['mdd']*100:>8.2f}%   "
               f"Q={r['n_qqq']} I={r['n_inv']} T={r['n_tqqq']}{extra}")


def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_data()
    dew = compute_dew(qqq_raw)
    df = df.merge(dew, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    df = v17b.add_v17_features(df)

    print("\n" + "=" * 104)
    print("  PYRAMID DE-MARGIN AUDIT   (2000-01-03 -> 2026-04-01)")
    print("  orig = PMP up to 1.30 (130% capital, margin) | no-margin = PMP ceiling 1.00")
    print("=" * 104)

    for mode, name in [("5", "MODE 5  pyramid"), ("7", "MODE 7  decay+pyramid (PRODUCTION)")]:
        print(f"\n  {name}")
        print(f"  {'config':20s} {'n':>5s} {'final':>20s} {'CAGR':>9s} {'Sharpe':>8s} {'MDD':>9s}   counts")

        s16o, l16o = row("v16 orig(1.30)",     run_v16(df, mode, PMP_ORIG))
        s16n, l16n = row("v16 no-margin(1.00)", run_v16(df, mode, PMP_NOMARGIN))
        s17o, l17o = row("v17 orig(1.30)",     run_v17(df, mode, PMP_ORIG))
        s17n, l17n = row("v17 no-margin(1.00)", run_v17(df, mode, PMP_NOMARGIN))
        print(l16o); print(l16n); print(l17o); print(l17n)

        # cost of de-margining (v16)
        dc16 = (s16n['cagr'] - s16o['cagr']) * 100
        dm16 = (s16n['mdd']  - s16o['mdd'])  * 100
        f16  = s16n['final'] / s16o['final']
        print(f"    de-margin cost (v16):  CAGR {dc16:+.2f}%   MDD {dm16:+.2f}pp   final x{f16:.3f}")
        # cost of de-margining (v17)
        dc17 = (s17n['cagr'] - s17o['cagr']) * 100
        dm17 = (s17n['mdd']  - s17o['mdd'])  * 100
        f17  = s17n['final'] / s17o['final']
        print(f"    de-margin cost (v17):  CAGR {dc17:+.2f}%   MDD {dm17:+.2f}pp   final x{f17:.3f}")
        # recommended (v17 no-margin) vs current production (v16 orig)
        dcp = (s17n['cagr'] - s16o['cagr']) * 100
        dmp = (s17n['mdd']  - s16o['mdd'])  * 100
        fp  = s17n['final'] / s16o['final']
        print(f"    v17 no-margin vs v16 orig:  CAGR {dcp:+.2f}%   MDD {dmp:+.2f}pp   final x{fp:.3f}")

    print("\n" + "=" * 104)


if __name__ == "__main__":
    main()
