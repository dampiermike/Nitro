"""
Time-Series Momentum (TSM) backtest
====================================
Adapts the strategy from Moskowitz, Ooi & Pedersen, "Time Series Momentum"
(Journal of Financial Economics, 2012) to the Nitro instrument set.

Paper recipe (single-instrument component), Eq. (3) of MOP 2012:

    r_TSM(t->t+1) = sign( r(t-12m, t) ) * (sigma_target / sigma_t) * r(t->t+1)

Adaptation to this codebase:
  - Signal asset .... QQQ (the underlying index).  We take the sign of QQQ's
                      trailing 12-month return.  Computing the sign on the
                      UNDERLYING avoids leveraged-ETF volatility decay
                      corrupting the momentum measure.
  - Long vehicle .... TQQQ  (3x QQQ; real >= 2010-03-31, synthetic before).
  - Short vehicle ... SQQQ  (3x inverse QQQ; real >= 2010-02-11, synthetic
                      daily-reset -3x before).
  - Sizing .......... ex-ante annualized volatility of the TRADED vehicle,
                      EWMA with a 60-day center of mass (the paper's estimator),
                      scaled to a constant target (default 40%, the paper's
                      per-instrument scaling constant).  Position weight is
                      capped at MAX_WEIGHT.
  - Rebalance ....... monthly (last trading day), 1-month holding.  The signal
                      and vol are measured at the rebalance close and applied
                      from the NEXT bar onward -> strictly causal, no look-ahead.

Output mirrors nitro_v16.print_stats formatting.  Buy & hold QQQ / TQQQ are
shown as benchmarks, plus a 1-12 month lookback sensitivity sweep.
"""
import numpy as np
import pandas as pd

from nitro_v16 import load_data, fmt_eq, START_DATE, END_DATE

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
LOOKBACK_DAYS  = 252      # 12-month momentum signal (trading days)
VOL_COM        = 60       # EWMA center of mass for ex-ante vol (paper: 60d)
TARGET_VOL     = 0.40     # annualized vol target (paper's scaling constant)
MAX_WEIGHT     = 4.0      # cap on the vol-scaling leverage multiplier
TRADING_DAYS   = 252      # annualization factor (matches nitro_v16)
VOL_SCALING    = True     # apply ex-ante vol scaling (else fixed full notional)
SHORT_FIN_COST = 0.0285 / TRADING_DAYS   # synthetic 3x-inverse daily financing drag

STARTING_K = 100_000.0


# ----------------------------------------------------------------------
# Build the vehicle daily-return series + derived signal/vol series
# ----------------------------------------------------------------------
def build_series():
    """Return a dict of numpy arrays aligned on the full (untrimmed) calendar."""
    _, df_full, _ = load_data()
    d = df_full.sort_values("Date").reset_index(drop=True).copy()
    n = len(d)

    date    = d["Date"].reset_index(drop=True)
    qclose  = d["Close"].values                 # QQQ close -> momentum signal
    c2c_qqq = d["c2c_QQQ"].values

    # Long vehicle: TQQQ close-to-close (synthetic+real stitch from load_data)
    long_c2c = d["tqqq_c2c"].values

    # Short vehicle: full 3x inverse.  Real SQQQ where available
    # (c2c_SQQQ3 is real SQQQ/3, so *3 recovers the real 3x daily-reset return);
    # synthetic daily-reset -3x QQQ (minus financing) before SQQQ existed.
    c2c_sqqq3 = d["c2c_SQQQ3"].values
    short_c2c = np.where(
        ~np.isnan(c2c_sqqq3),
        3.0 * c2c_sqqq3,
        -3.0 * c2c_qqq - SHORT_FIN_COST,
    )

    # Ex-ante annualized vol of each vehicle (EWMA, com=60 -> paper's estimator).
    long_vol  = pd.Series(long_c2c).ewm(com=VOL_COM).std().values  * np.sqrt(TRADING_DAYS)
    short_vol = pd.Series(short_c2c).ewm(com=VOL_COM).std().values * np.sqrt(TRADING_DAYS)

    # Month-end flag: True on the last trading row of each calendar month.
    period = date.dt.to_period("M")
    is_month_end = (period != period.shift(-1)).to_numpy(copy=True)
    is_month_end[-1] = True

    return dict(
        n=n, date=date, qclose=qclose, c2c_qqq=c2c_qqq,
        long_c2c=long_c2c, short_c2c=short_c2c,
        long_vol=long_vol, short_vol=short_vol,
        is_month_end=is_month_end,
    )


def momentum(qclose, lookback):
    """Trailing `lookback`-day simple return of the underlying (NaN until warm)."""
    n = len(qclose)
    mom = np.full(n, np.nan)
    for i in range(lookback, n):
        base = qclose[i - lookback]
        if base > 0:
            mom[i] = qclose[i] / base - 1.0
    return mom


# ----------------------------------------------------------------------
# Core backtest
# ----------------------------------------------------------------------
def run_tsm(S, lookback, common_start, vol_scaling=VOL_SCALING):
    """
    Run the TSM strategy. Returns (dates, dret, trades, rebalances) where
    dret is the daily strategy return restricted to [common_start, END_DATE].

    Strictly causal: at a month-end close we read momentum+vol, then the new
    position takes effect on the FOLLOWING bar.
    """
    n        = S["n"]
    date     = S["date"]
    long_c2c = S["long_c2c"]
    short_c2c= S["short_c2c"]
    long_vol = S["long_vol"]
    short_vol= S["short_vol"]
    me       = S["is_month_end"]
    mom      = momentum(S["qclose"], lookback)

    cur_dir = 0          # +1 long TQQQ, -1 short SQQQ, 0 flat
    cur_w   = 0.0
    n_capped = 0

    dates, drets = [], []
    rebalances = []      # (date, dir, weight, mom) at each acted month-end
    trades = []          # contiguous same-direction runs

    for i in range(n):
        di = date.iloc[i]

        # 1) Today's return uses the position held coming into the bar.
        if cur_dir != 0 and common_start <= di <= END_DATE:
            veh = long_c2c[i] if cur_dir > 0 else short_c2c[i]
            if not np.isnan(veh):
                dates.append(di)
                drets.append(cur_w * cur_dir_sign(cur_dir, veh))

        # 2) Month-end: form next position from info available at this close.
        if me[i] and di <= END_DATE and not np.isnan(mom[i]):
            new_dir = 1 if mom[i] >= 0 else -1
            vol = long_vol[i] if new_dir > 0 else short_vol[i]
            if np.isnan(vol) or vol <= 0:
                continue
            if vol_scaling:
                w = TARGET_VOL / vol
                if w > MAX_WEIGHT:
                    w = MAX_WEIGHT
                    n_capped += 1
            else:
                w = 1.0

            if di >= common_start:
                rebalances.append((di, new_dir, w, mom[i]))
                if not trades or trades[-1]["dir"] != new_dir:
                    trades.append(dict(dir=new_dir, start=di, end=di, n_months=1))
                else:
                    trades[-1]["end"] = di
                    trades[-1]["n_months"] += 1

            cur_dir, cur_w = new_dir, w

    return (
        np.array(dates), np.array(drets, dtype=float),
        trades, rebalances, n_capped,
    )


def cur_dir_sign(direction, vehicle_ret):
    """Vehicle already embeds the long/short direction (TQQQ vs SQQQ),
    so the position weight multiplies the vehicle return directly (sign +1)."""
    return vehicle_ret


# ----------------------------------------------------------------------
# Stats (CAGR uses the ACTUAL span, unlike nitro_v16's hardcoded exponent)
# ----------------------------------------------------------------------
def stats(dates, dret):
    eq = STARTING_K * np.cumprod(1.0 + dret)
    final = eq[-1]
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = (final / STARTING_K) ** (1.0 / yrs) - 1.0
    sd = dret.std()
    sharpe = (dret.mean() * TRADING_DAYS) / (sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return dict(final=final, cagr=cagr, sharpe=sharpe, mdd=mdd, eq=eq, yrs=yrs)


def print_row(label, s, extra=""):
    print(f"  {label:<26} final={fmt_eq(s['final']):>20}  "
          f"CAGR={s['cagr']*100:7.2f}%  Sharpe={s['sharpe']:6.3f}  "
          f"MDD={s['mdd']*100:8.2f}%  {extra}")


def benchmark(S, common_start, c2c):
    dates, drets = [], []
    for i in range(S["n"]):
        di = S["date"].iloc[i]
        if common_start <= di <= END_DATE and not np.isnan(c2c[i]):
            dates.append(di)
            drets.append(c2c[i])
    return stats(np.array(dates), np.array(drets, dtype=float))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    S = build_series()

    # Common active start = first month-end with a valid 12-month signal,
    # so every lookback in the sweep is compared over the same window.
    mom12 = momentum(S["qclose"], LOOKBACK_DAYS)
    common_start = None
    for i in range(S["n"]):
        if S["is_month_end"][i] and not np.isnan(mom12[i]) and S["date"].iloc[i] >= START_DATE:
            common_start = S["date"].iloc[i]
            break
    # advance to the next bar (position is effective AFTER the rebalance close)
    print("=" * 100)
    print("TIME-SERIES MOMENTUM  (Moskowitz-Ooi-Pedersen 2012)  -  TQQQ/SQQQ vehicles")
    print("=" * 100)
    print(f"  Signal       : sign of trailing {LOOKBACK_DAYS}-day ({LOOKBACK_DAYS//21}-mo) QQQ return")
    print(f"  Vehicles     : TQQQ (long) / SQQQ (short), 3x daily-reset")
    print(f"  Sizing       : {'ex-ante vol target ' + str(int(TARGET_VOL*100)) + '%' if VOL_SCALING else 'fixed full notional'}"
          f"  (EWMA com={VOL_COM}, cap={MAX_WEIGHT}x)")
    print(f"  Rebalance    : monthly, 1-mo hold, strictly causal")
    print(f"  Window       : {pd.Timestamp(common_start).date()}  ->  {END_DATE.date()}")
    print("-" * 100)

    # Headline: 12-month TSM
    dts, drt, trades, rebs, n_capped = run_tsm(S, LOOKBACK_DAYS, common_start)
    s = stats(dts, drt)
    n_long  = sum(1 for r in rebs if r[1] > 0)
    n_short = sum(1 for r in rebs if r[1] < 0)
    pct_long = 100.0 * n_long / max(1, len(rebs))
    flips = len(trades)
    print("\n  STRATEGY")
    print_row(f"TSM-{LOOKBACK_DAYS//21}mo (TQQQ/SQQQ)", s,
              extra=f"trades={flips} rebal={len(rebs)} long={pct_long:.0f}%")

    print("\n  BENCHMARKS (same window)")
    print_row("Buy & Hold QQQ",  benchmark(S, common_start, S["c2c_qqq"]))
    print_row("Buy & Hold TQQQ", benchmark(S, common_start, S["long_c2c"]))

    # Lookback sensitivity sweep
    print("\n  LOOKBACK SWEEP (months)")
    for mo in (1, 3, 6, 9, 12):
        lb = mo * 21
        dts_k, drt_k, tr_k, rb_k, _ = run_tsm(S, lb, common_start)
        if len(drt_k) == 0:
            continue
        sk = stats(dts_k, drt_k)
        plong = 100.0 * sum(1 for r in rb_k if r[1] > 0) / max(1, len(rb_k))
        print_row(f"TSM-{mo}mo", sk, extra=f"trades={len(tr_k)} long={plong:.0f}%")

    print("\n  NOTES")
    print(f"   - vol cap hit on {n_capped} rebalances (headline run)")
    print(f"   - vol scaling {'ON' if VOL_SCALING else 'OFF'}; target {int(TARGET_VOL*100)}% ann.")
    print(f"   - synthetic 3x SQQQ before 2010-02-11; synthetic TQQQ before 2010-03-31")
    print("=" * 100)


if __name__ == "__main__":
    main()
