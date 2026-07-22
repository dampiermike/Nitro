"""
v17 Cap sweep (no-margin, mode 7).

Diagnosis said: drop Mean-Rev, keep Cap but stop it catching the knife.
This sweeps Cap's stop distance and green-candle confirmation (MR off), and
includes a VIX-gate-revert variant, to find the config that keeps Cap's ~+4%
CAGR while pulling MDD back toward v16's -21.5% and Sharpe above 1.827.

Reference floor: v16 no-margin = CAGR 46.19% / Sharpe 1.827 / MDD -21.54%.
"""
import os, sys
import numpy as np
from collections import Counter

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

PMP_NOMARGIN = make_pmp(1.00, 0.85, 0.50)
nitro_v16.get_pmp = PMP_NOMARGIN
v17b.get_pmp = PMP_NOMARGIN
MODE = "7"


def apply(cfg):
    v17b.CAP_ON = cfg.get("cap_on", True)
    v17b.MR_ON = cfg.get("mr_on", False)
    v17b.CAP_ATR_MULT = cfg.get("atr", 1.0)
    v17b.MR_ATR_MULT = 1.0
    v17b.CAP_MAXHOLD = cfg.get("maxhold", 0)
    v17b.CAP_VIX_MAX = cfg.get("cap_vix", 0.0)
    v17b.CAP_CONFIRM = cfg.get("confirm", False)
    v17b.CAP_TP = cfg.get("tp", 0.05)
    v17b.CDN_VIX_MAX = cfg.get("cdn_vix", 35.0)


def cap_win(trades):
    rows = [t for t in trades if len(t) > 7 and t[7] == "cap"]
    if not rows:
        return 0, 0.0
    rets = np.array([t[2] for t in rows])
    return len(rows), float((rets > 0).mean())


def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_data()
    dew = compute_dew(qqq_raw)
    df = df.merge(dew, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    df = v17b.add_v17_features(df)

    configs = [
        ("v16 no-margin (floor)",        {"_v16": True}),
        ("baseline v17 (MR on,atr1.0)",  {"mr_on": True, "atr": 1.0, "confirm": False}),
        ("MR off · atr0.75",             {"atr": 0.75, "confirm": False}),
        ("MR off · atr1.0",              {"atr": 1.0,  "confirm": False}),
        ("MR off · atr1.5",              {"atr": 1.5,  "confirm": False}),
        ("MR off · atr2.0",              {"atr": 2.0,  "confirm": False}),
        ("MR off · confirm · atr0.75",   {"atr": 0.75, "confirm": True}),
        ("MR off · confirm · atr1.0",    {"atr": 1.0,  "confirm": True}),
        ("MR off · confirm · atr1.5",    {"atr": 1.5,  "confirm": True}),
        ("MR off · confirm · atr2.0",    {"atr": 2.0,  "confirm": True}),
        ("MR off · confirm · atr1.0 · maxhold10", {"atr": 1.0, "confirm": True, "maxhold": 10}),
        ("MR off · confirm · atr1.0 · VIXrevert27", {"atr": 1.0, "confirm": True, "cdn_vix": 27.0}),
    ]

    rows = []
    for name, cfg in configs:
        if cfg.get("_v16"):
            r = nitro_v16.run_engine(df, mode=MODE)
            ncap, cw = 0, 0.0
        else:
            apply(cfg)
            r = v17b.run_engine_v17(df, mode=MODE)
            ncap, cw = cap_win(r["trades"])
        s = trade_stats(r["trades"], r["eq"], r["dret"])
        rows.append((name, s, ncap, cw))

    print("\n" + "=" * 104)
    print("  v17 CAP SWEEP  ·  no-margin  ·  MODE 7     (floor: v16 = 46.19% / 1.827 / -21.54%)")
    print("=" * 104)
    print(f"  {'config':42s} {'CAGR':>8s} {'Sharpe':>8s} {'MDD':>9s} {'final':>18s} {'nCap':>5s} {'capWin':>7s}")
    print("  " + "-" * 100)
    for name, s, ncap, cw in rows:
        print(f"  {name:42s} {s['cagr']*100:>7.2f}% {s['sharpe']:>8.3f} {s['mdd']*100:>8.2f}% "
              f"{fmt_eq(s['final']):>18s} {ncap:>5d} {cw*100:>6.1f}%")

    # Highlight best by Sharpe and by CAGR among the tuned (non-floor, non-baseline) set
    tuned = [(n, s) for (n, s, _, _) in rows if "floor" not in n and "baseline" not in n]
    best_sharpe = max(tuned, key=lambda x: x[1]["sharpe"])
    best_cagr   = max(tuned, key=lambda x: x[1]["cagr"])
    print("  " + "-" * 100)
    print(f"  best Sharpe: {best_sharpe[0]}  ({best_sharpe[1]['sharpe']:.3f}, "
          f"CAGR {best_sharpe[1]['cagr']*100:.2f}%, MDD {best_sharpe[1]['mdd']*100:.2f}%)")
    print(f"  best CAGR:   {best_cagr[0]}  ({best_cagr[1]['cagr']*100:.2f}%, "
          f"Sharpe {best_cagr[1]['sharpe']:.3f}, MDD {best_cagr[1]['mdd']*100:.2f}%)")
    print("=" * 104)


if __name__ == "__main__":
    main()
