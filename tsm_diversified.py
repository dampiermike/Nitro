"""
Diversified Time-Series Momentum (Moskowitz-Ooi-Pedersen 2012)
==============================================================
Multi-instrument version of tsm_backtest.py.  Reproduces the paper's portfolio
construction across asset classes using liquid ETF *proxies* (the actual
futures of the paper are unavailable; ETFs stand in for the four asset classes).

Data source : the Oscillator project's EODHD history dump
              (/Users/.../Oscillator/json/history/<TICKER>_US.json).
              Each file is a list of daily bars with an `adjusted_close` field.

Per-instrument TSM, exactly as in MOP 2012 Eq. (3):

    r_TSM(t->t+1) = sign( r(t-12m, t) ) * (sigma_target / sigma_t) * r(t->t+1)

  - signal  : sign of the instrument's trailing 252-day (12-mo) return
  - sizing  : ex-ante annualized vol, EWMA com=60 (paper's estimator),
              scaled to TARGET_VOL (40%, paper's constant), weight capped
  - rebal   : monthly, 1-mo hold, strictly causal (position set at month-end
              close applies from the next bar)

Portfolio = equal-weight average of the per-instrument TSM returns across all
instruments that are "live" on a given day (an instrument goes live once it has
>=252 days of history, mirroring the paper's handling of staggered start dates).

Outputs (all on a common 2007-01-01 start):
  1. The diversified all-asset TSM portfolio
  2. Per-asset-class sub-portfolios (the paper's Table B-E panels)
  3. Benchmarks: buy & hold SPY, and the single-name QQQ TSM
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nitro_v16 import fmt_eq

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OSC_DIR    = Path("/Users/mikedampier/Documents/Development/Oscillator/json/history")
START      = pd.Timestamp("2007-01-01")     # common active start (T-bill leg gates this)
LOOKBACK   = 252      # 12-month momentum signal (trading days)
VOL_COM    = 60       # EWMA center of mass for ex-ante vol (paper: 60d)
TARGET_VOL = 0.15     # annualized vol target (lowered from paper's 40%)
MAX_WEIGHT = 4.0      # cap on the vol-scaling multiplier
TD         = 252      # annualization factor
STARTING_K = 100_000.0

# ETF proxy basket for the trend sleeves (FX sleeve removed).
BASKET = {
    "Equity":    ["SPY", "QQQ", "EWJ", "EFA", "EEM", "VGK"],
    "Bond":      ["SHY", "IEF", "TLT", "LQD"],
    "Commodity": ["GLD", "SLV", "USO", "DBC", "DBA", "GSG"],
}

# Cash / T-bill leg: held for its risk-free carry (NOT a trend signal).
# SHV (iShares 0-1yr Treasury) is the earliest non-spliced T-bill proxy
# (inception 2007-01-11), which is what pins the start at Jan 2007.
CASH_TICKER = "SHV"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_prices(ticker):
    """Return a daily adjusted-close Series (DatetimeIndex) or None if missing."""
    fp = OSC_DIR / f"{ticker}_US.json"
    if not fp.exists():
        return None
    rows = json.loads(fp.read_text())
    df = pd.DataFrame(rows)
    if "adjusted_close" not in df or "date" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"])
    s = (df[["date", "adjusted_close"]]
         .dropna()
         .drop_duplicates("date")
         .sort_values("date")
         .set_index("date")["adjusted_close"]
         .astype(float))
    s = s[s > 0]
    return s if len(s) > LOOKBACK + VOL_COM else None


# ----------------------------------------------------------------------
# Per-instrument TSM daily-return series (strictly causal, monthly rebal)
# ----------------------------------------------------------------------
def inst_tsm(p):
    r = p.pct_change()
    sigma = r.ewm(com=VOL_COM).std() * np.sqrt(TD)         # ex-ante annualized vol
    mom = p / p.shift(LOOKBACK) - 1.0                       # trailing 12-mo return

    direction = np.sign(mom).replace(0.0, 1.0)             # +1 long / -1 short
    weight = (TARGET_VOL / sigma).clip(upper=MAX_WEIGHT)
    signed = direction * weight

    valid = mom.notna() & sigma.notna() & (sigma > 0)
    month = pd.Series(p.index.to_period("M"), index=p.index)
    is_me = month != month.shift(-1)
    is_me.iloc[-1] = True

    pos = signed.where(valid).where(is_me).ffill()         # set at month-ends, hold
    strat = pos.shift(1) * r                                # apply next bar -> causal
    return strat


# ----------------------------------------------------------------------
# Stats (CAGR from the actual span)
# ----------------------------------------------------------------------
def stats(dret):
    dret = dret.dropna()
    if len(dret) == 0:
        return None
    eq = STARTING_K * (1.0 + dret).cumprod()
    final = eq.iloc[-1]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (final / STARTING_K) ** (1.0 / yrs) - 1.0
    sd = dret.std()
    sharpe = (dret.mean() * TD) / (sd * np.sqrt(TD)) if sd > 0 else float("nan")
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min()
    return dict(final=final, cagr=cagr, sharpe=sharpe, mdd=mdd, eq=eq, yrs=yrs)


def print_row(label, s, extra=""):
    if s is None:
        print(f"  {label:<28} (no data)")
        return
    print(f"  {label:<28} final={fmt_eq(s['final']):>18}  "
          f"CAGR={s['cagr']*100:7.2f}%  Sharpe={s['sharpe']:6.3f}  "
          f"MDD={s['mdd']*100:8.2f}%  {extra}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Load + compute per-instrument TSM for every ticker in the basket.
    strat = {}        # ticker -> daily TSM return series
    loaded = {cls: [] for cls in BASKET}
    missing = []
    for cls, tickers in BASKET.items():
        for t in tickers:
            p = load_prices(t)
            if p is None:
                missing.append(t)
                continue
            strat[t] = inst_tsm(p)
            loaded[cls].append(t)

    M_risk = pd.concat(strat, axis=1)            # union calendar, cols = trend tickers
    M = M_risk.copy()

    # Cash / T-bill leg: just the carry from holding the bill ETF (no trend).
    cash_p = load_prices(CASH_TICKER)
    cash_ret = cash_p.pct_change() if cash_p is not None else None
    if cash_ret is not None:
        M[CASH_TICKER] = cash_ret

    M = M[M.index >= START]
    M_risk = M_risk[M_risk.index >= START]

    print("=" * 104)
    print("DIVERSIFIED TIME-SERIES MOMENTUM  (Moskowitz-Ooi-Pedersen 2012)  -  ETF proxies")
    print("=" * 104)
    print(f"  Signal     : sign of trailing {LOOKBACK}-day (12-mo) return, per instrument")
    print(f"  Sizing     : ex-ante vol target {int(TARGET_VOL*100)}% (EWMA com={VOL_COM}, cap={MAX_WEIGHT}x)")
    print(f"  Rebalance  : monthly, 1-mo hold, strictly causal")
    print(f"  Cash leg   : {CASH_TICKER} held for T-bill carry (replaces the FX sleeve)")
    print(f"  Window     : {START.date()} -> {M.index[-1].date()}")
    for cls in BASKET:
        print(f"  {cls:<10}: {', '.join(loaded[cls]) if loaded[cls] else '(none)'}")
    print(f"  Cash      : {CASH_TICKER if cash_ret is not None else '(missing!)'}")
    if missing:
        print(f"  MISSING    : {', '.join(missing)}")
    n_active = M.notna().sum(axis=1)
    print(f"  Live count : {int(n_active.iloc[0])} at start -> {int(n_active.iloc[-1])} at end "
          f"(avg {n_active.mean():.1f})")
    print("-" * 104)

    # Build sleeve return series: 3 trend sleeves + 1 cash sleeve.
    sleeves = {}
    for cls in BASKET:
        cols = [t for t in loaded[cls] if t in M_risk.columns]
        if cols:
            sleeves[cls] = M_risk[cols].mean(axis=1, skipna=True)
    if cash_ret is not None:
        sleeves["Cash"] = M[CASH_TICKER]

    # 1) Diversified portfolio
    print("\n  [1] DIVERSIFIED PORTFOLIO")
    eq_class = pd.concat(sleeves, axis=1).mean(axis=1, skipna=True)
    print_row("All (equal-weight by sleeve)", stats(eq_class),
              extra=f"sleeves={len(sleeves)}")
    all_port = M.mean(axis=1, skipna=True)
    print_row("All (equal-weight by name)", stats(all_port),
              extra=f"instruments={M.shape[1]}")

    # 2) Per-sleeve sub-portfolios
    print("\n  [2] BY SLEEVE")
    for name, port in sleeves.items():
        n = 1 if name == "Cash" else len([t for t in loaded.get(name, []) if t in M_risk.columns])
        print_row(f"{name} ({n})", stats(port[port.index >= START]))

    # 3) Benchmarks
    print("\n  [3] BENCHMARKS (same window)")
    spy = load_prices("SPY")
    bh_spy = spy.pct_change()
    bh_spy = bh_spy[bh_spy.index >= START]
    print_row("Buy & Hold SPY", stats(bh_spy))

    qqq = load_prices("QQQ")
    qqq_tsm = inst_tsm(qqq)
    qqq_tsm = qqq_tsm[qqq_tsm.index >= START]
    print_row("QQQ single-name TSM", stats(qqq_tsm))

    print("\n  NOTES")
    print(f"   - ETF proxies, not the paper's futures. T-bill ETF ({CASH_TICKER}) inception")
    print("     gates the start at Jan 2007; earlier needs splicing.")
    print("   - FX sleeve removed; cash/T-bill leg added (held for carry, no trend signal).")
    print("   - Trend instruments enter once they have 252d of history (staggered).")
    print(f"   - vol target {int(TARGET_VOL*100)}% ann., weight capped at {MAX_WEIGHT}x.")
    print("=" * 104)


if __name__ == "__main__":
    main()
