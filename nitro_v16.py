"""
Dampier Nitro++ v16 - Built from Build_Guide_v16.docx
Strictly causal sizing. All checkpoints verified inline.
"""
import os
import sys
import numpy as np
import pandas as pd

# Path config: local by default (this file's directory), overridable in
# production via env vars so nothing depends on the checkout living in any
# particular (e.g. cloud-synced) location.
#   NITRO_HOME     - project root      (default: directory of this file)
#   NITRO_DATA_DIR - CSV history dir   (default: $NITRO_HOME/data/csv/history)
#   NITRO_END_DATE - backtest window end (default: today, i.e. all history).
#                    Set to freeze the window for reproducible runs, e.g.
#                    NITRO_END_DATE=2026-04-01 reproduces the v17 ship numbers.
NITRO_HOME = os.environ.get("NITRO_HOME", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NITRO_DATA_DIR", os.path.join(NITRO_HOME, "data/csv/history"))
START_DATE = pd.Timestamp("2000-01-01")
_END_ENV = os.environ.get("NITRO_END_DATE", "").strip()
END_DATE = pd.Timestamp(_END_ENV) if _END_ENV else pd.Timestamp.today().normalize()
SQQQ_CUTOFF = pd.Timestamp("2010-02-11")
TQQQ_STITCH = pd.Timestamp("2010-03-31")


# ----------------------------------------------------------------------
# Step 1 - Load + Merge
# ----------------------------------------------------------------------
def load_data():
    qqq = pd.read_csv(f"{DATA_DIR}/qqq-from-vv.csv")
    qqq.columns = [c.strip().lstrip("﻿") for c in qqq.columns]
    qqq["Date"] = pd.to_datetime(qqq["Date"], format="%m/%d/%y")
    qqq = qqq.rename(columns={"RT": "QQQ_RT_raw"}).sort_values("Date").reset_index(drop=True)

    tqqq_real = pd.read_csv(f"{DATA_DIR}/tqqq-from-vv.csv")
    tqqq_real.columns = [c.strip().lstrip("﻿") for c in tqqq_real.columns]
    tqqq_real["Date"] = pd.to_datetime(tqqq_real["Date"], format="%m/%d/%y")
    tqqq_real = tqqq_real.rename(columns={"RT": "tqqq_rt"}).sort_values("Date").reset_index(drop=True)

    tqqq_synth = pd.read_csv(f"{DATA_DIR}/synthetic-tqqq-ohlc-1999-2010.csv")
    tqqq_synth["Date"] = pd.to_datetime(tqqq_synth["Date"])
    tqqq_synth = tqqq_synth.rename(columns={"RT_v6": "tqqq_rt"}).sort_values("Date").reset_index(drop=True)

    # TQQQ stitch: synth for Date < 2010-03-31, real for Date >= 2010-03-31
    tqqq_synth_part = tqqq_synth[tqqq_synth["Date"] < TQQQ_STITCH][["Date", "Open", "High", "Low", "Close", "tqqq_rt"]]
    tqqq_real_part = tqqq_real[tqqq_real["Date"] >= TQQQ_STITCH][["Date", "Open", "High", "Low", "Close", "tqqq_rt"]]
    tqqq = pd.concat([tqqq_synth_part, tqqq_real_part], ignore_index=True).sort_values("Date").reset_index(drop=True)
    tqqq.columns = ["Date", "tqqq_open", "tqqq_high", "tqqq_low", "tqqq_close", "tqqq_rt"]

    views = pd.read_csv(f"{DATA_DIR}/vectorvest-views-w3place-precision.csv")
    views["Date"] = pd.to_datetime(views["Date"], format="%m/%d/%y")
    views = views.rename(columns={"VVC-RT": "RT", "BS Ratio": "BSR"})
    views = views[["Date", "RT", "BSR", "MTI", "Trend"]].sort_values("Date").reset_index(drop=True)

    vix = pd.read_csv(f"{DATA_DIR}/vix-from-yahoo.csv")
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%y")
    vix = vix[["Date", "Close"]].rename(columns={"Close": "VIX"}).sort_values("Date").reset_index(drop=True)

    sqqq = pd.read_csv(f"{DATA_DIR}/sqqq-from-yahoo.csv")
    sqqq["Date"] = pd.to_datetime(sqqq["Date"], format="%m/%d/%y")
    sqqq = sqqq.sort_values("Date").reset_index(drop=True)
    # SQQQ adjustment factor (handles reverse splits)
    sqqq_adj_factor = sqqq["Adj Close"] / sqqq["Close"]
    sqqq_open_adj = sqqq["Open"] * sqqq_adj_factor
    sqqq_adj_close = sqqq["Adj Close"]
    sqqq["o2c_SQQQ3"] = (sqqq_adj_close / sqqq_open_adj - 1) / 3
    sqqq["c2c_SQQQ3"] = sqqq_adj_close.pct_change() / 3
    sqqq["pc2o_SQQQ3"] = (sqqq_open_adj / sqqq_adj_close.shift(1) - 1) / 3
    sqqq = sqqq[["Date", "o2c_SQQQ3", "c2c_SQQQ3", "pc2o_SQQQ3"]]

    # Merge everything on QQQ as anchor (QQQ has full date coverage from 1999-07)
    df = qqq[["Date", "Open", "High", "Low", "Close"]].copy()
    df = df.merge(tqqq, on="Date", how="left")
    df = df.merge(views, on="Date", how="left")
    df = df.merge(vix, on="Date", how="left")
    df = df.merge(sqqq, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    # Return columns - QQQ
    df["c2c_QQQ"] = df["Close"].pct_change()
    df["o2c_QQQ"] = df["Close"] / df["Open"] - 1
    df["pc2o_QQQ"] = df["Open"] / df["Close"].shift(1) - 1

    # INV - synthetic pre-2010-02-11
    df["o2c_INV"] = -1.0 * df["o2c_QQQ"]
    df["pc2o_INV"] = -1.0 * df["pc2o_QQQ"]
    df["c2c_INV"] = -1.0 * df["c2c_QQQ"] - 0.0095 / 252.0
    # Overwrite from 2010-02-11 onward with real SQQQ/3
    mask = df["Date"] >= SQQQ_CUTOFF
    df.loc[mask, "o2c_INV"] = df.loc[mask, "o2c_SQQQ3"]
    df.loc[mask, "c2c_INV"] = df.loc[mask, "c2c_SQQQ3"]
    df.loc[mask, "pc2o_INV"] = df.loc[mask, "pc2o_SQQQ3"]

    # TQQQ returns
    df["tqqq_o2c"] = df["tqqq_close"] / df["tqqq_open"] - 1
    df["tqqq_c2c"] = df["tqqq_close"].pct_change()
    df["tqqq_pc2o"] = df["tqqq_open"] / df["tqqq_close"].shift(1) - 1

    # cc - carry-forward Confirmed Call
    cur = "C/Up"
    cc = []
    for t in df["Trend"]:
        if t == "C/Up":
            cur = "C/Up"
        elif t == "C/Dn":
            cur = "C/Dn"
        cc.append(cur)
    df["cc"] = cc

    # ATR14 - Wilder on QQQ in dollars (manual loop)
    n = len(df)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    if n >= 14:
        atr[13] = tr[0:14].mean()
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    df["ATR14"] = atr

    # tqqq_atr - simple 10-bar ATR% on TQQQ
    th = df["tqqq_high"].values
    tl = df["tqqq_low"].values
    tc = df["tqqq_close"].values
    tr_t = np.zeros(n)
    tr_t[0] = th[0] - tl[0]
    for i in range(1, n):
        tr_t[i] = max(th[i] - tl[i], abs(th[i] - tc[i-1]), abs(tl[i] - tc[i-1]))
    tqqq_atr = np.full(n, np.nan)
    tqqq_atr[0] = 7.9588  # seed
    for i in range(9, n):
        tqqq_atr[i] = tr_t[i-9:i+1].mean() / tc[i] * 100.0
    df["tqqq_atr"] = tqqq_atr

    # Keep df_full (untrimmed) for Step 6
    df_full = df.copy()

    # Trim verification window
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].reset_index(drop=True)
    return df, df_full, qqq


# ----------------------------------------------------------------------
# Step 2 - DEW oscillator (state machine on raw QQQ)
# ----------------------------------------------------------------------
def compute_dew(qqq_raw):
    """qqq_raw: full QQQ series from 1999-07-01. Returns df with Date + DEW_Signal."""
    q = qqq_raw.sort_values("Date").reset_index(drop=True).copy()
    close = q["Close"].values
    n = len(q)

    # D = (Close - Close.rolling(20).mean().shift(11)).round(2)
    sma20 = q["Close"].rolling(20).mean()
    D = (q["Close"] - sma20.shift(11)).round(2).values
    EMA10 = q["Close"].ewm(span=10, adjust=False).mean().values
    E_upper = EMA10 * 1.06
    E_lower = EMA10 * 0.94
    # W[i] = sum(Close[i-29:i+1] * weights) / 465, weights 1..30 newest=30
    weights = np.arange(1, 31)
    W = np.full(n, np.nan)
    for i in range(29, n):
        W[i] = float(np.dot(close[i-29:i+1], weights)) / 465.0

    DEW_Signal = [""] * n
    state = "Normal"
    last_fired = None
    for i in range(41, n):
        c = close[i]
        d = D[i]
        w = W[i]
        if np.isnan(d) or np.isnan(w):
            # if pierce check still possible, do it; but no candidate fires
            cand = None
        else:
            if state == "Normal":
                if c > w and d > 0:
                    cand = "Buy"
                elif c < w and d < 0:
                    cand = "Sell"
                else:
                    cand = None
            else:  # Primed - OR logic
                if c > w or d > 0:
                    cand = "Buy"
                elif c < w or d < 0:
                    cand = "Sell"
                else:
                    cand = None
        if cand == last_fired:
            cand = None
        if cand:
            DEW_Signal[i] = cand
            last_fired = cand
            state = "Normal"
        # Pierce check (takes effect next bar)
        if not np.isnan(E_upper[i]) and (c > E_upper[i] or c < E_lower[i]):
            state = "Primed"

    out = pd.DataFrame({"Date": q["Date"], "DEW_Signal": DEW_Signal})
    return out


# ----------------------------------------------------------------------
# Verification helpers
# ----------------------------------------------------------------------
def fmt_eq(x):
    return f"${x:,.0f}"


def trade_stats(trades, eq, dret_arr, dates=None):
    """Summary stats. Pass `dates` (the bar Date series) so CAGR annualizes over
    the actual window; without it the exponent falls back to the original fixed
    2000-01-03 -> 2026-04-01 span, which is only correct for that window."""
    eq_arr = np.asarray(eq, dtype=float)
    if len(trades) == 0:
        return dict(n=0, final=eq_arr[-1] if len(eq_arr) else np.nan, cagr=np.nan, sharpe=np.nan, mdd=np.nan)
    final = eq_arr[-1]
    if dates is not None and len(dates):
        d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        yrs = (d.iloc[-1] - d.iloc[0]).days / 365.25
    else:
        # Legacy spec exponent: (2026-04-01 - 2000-01-03).days / 365.25 = 9585/365.25 = 26.2423
        yrs = 9585.0 / 365.25
    cagr = (final / 100_000.0) ** (1.0 / yrs) - 1.0
    dret = np.asarray(dret_arr, dtype=float)
    dret = dret[~np.isnan(dret)]
    if dret.std() > 0:
        sharpe = dret.mean() * 252 / (dret.std() * np.sqrt(252))
    else:
        sharpe = float("nan")
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    mdd = dd.min()
    return dict(n=len(trades), final=final, cagr=cagr, sharpe=sharpe, mdd=mdd)


def print_stats(label, s, target=None):
    print(f"  {label}: n={s['n']:<5} final={fmt_eq(s['final']):>22}  CAGR={s['cagr']*100:7.4f}%  Sharpe={s['sharpe']:7.4f}  MDD={s['mdd']*100:7.4f}%")
    if target:
        print(f"  TARGET: {target}")


# ----------------------------------------------------------------------
# Step 3a - Isolated QQQ
# ----------------------------------------------------------------------
def iso_qqq(df):
    n = len(df)
    Date = df["Date"].values
    Close = df["Close"].values
    Open_ = df["Open"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    ATR14 = df["ATR14"].values
    o2c = df["o2c_QQQ"].values
    c2c = df["c2c_QQQ"].values
    pc2o = df["pc2o_QQQ"].values

    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    stop_qqq = None
    last_dew = None
    dret = np.zeros(n)
    trades = []
    eq = np.full(n, np.nan)
    cur_eq = 100_000.0
    cur_entry_date = None
    cur_entry_pc_signal = None

    for i in range(n):
        # CC transitions
        if i > 0:
            if Trend[i] == "C/Up" and cc[i-1] != "C/Up":
                last_dew = None
            if Trend[i] == "C/Dn" and cc[i-1] != "C/Dn":
                last_dew = None

        # Phase B - exit
        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1 + d)
            exited = False
            exit_reason = None
            # TP
            if cum_mult - 1 >= 0.05:
                exited = True
                exit_reason = "TP"
            # ATR stop
            elif Close[i] <= stop_qqq:
                exited = True
                exit_reason = "ATR"
            # C/Dn flip (not entry bar)
            elif (i != entry_idx) and Trend[i] == "C/Dn" and i > 0 and cc[i-1] == "C/Up":
                exited = True
                exit_reason = "Flip"
            # DEW Sell
            elif DEW[i] == "Sell":
                exited = True
                exit_reason = "DEW"
                last_dew = "Sell"
            if exited:
                if i + 1 < n:
                    pc = pc2o[i+1]
                    trade_ret = cum_mult * (1 + pc) - 1
                    dret[i] = (1 + d) * (1 + pc) - 1
                else:
                    trade_ret = cum_mult - 1
                    dret[i] = d
                trades.append((cur_entry_date, Date[i], trade_ret, exit_reason))
                cur_eq *= (1 + trade_ret) if False else cur_eq  # eq tracked via dret
                in_trade = False
                entry_idx = None
                cum_mult = 1.0
                stop_qqq = None
                if exit_reason == "DEW":
                    pass  # last_dew already set
            else:
                dret[i] = d
                eq[i] = np.nan  # leave for cumprod
                continue

        # Phase C - entry (QQQ only)
        if not in_trade and DEW[i] == "Buy" and last_dew != "Buy":
            if i + 1 < n:
                in_trade = True
                entry_idx = i + 1
                cum_mult = 1.0
                atr_mult = 2.0 if cc[i] == "C/Dn" else 1.0
                stop_qqq = Open_[i+1] - atr_mult * ATR14[i]
                last_dew = "Buy"
                cur_entry_date = Date[i+1]
                cur_entry_pc_signal = i

    # Equity curve
    eq_curve = 100_000.0 * np.cumprod(1 + dret)
    return trades, eq_curve, dret


# ----------------------------------------------------------------------
# Step 3b - Isolated INV
# ----------------------------------------------------------------------
def iso_inv(df):
    n = len(df)
    Date = df["Date"].values
    Close = df["Close"].values  # for stop check (uses QQQ close)
    Open_ = df["Open"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    ATR14 = df["ATR14"].values
    o2c = df["o2c_INV"].values
    c2c = df["c2c_INV"].values
    pc2o = df["pc2o_INV"].values

    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    stop_qqq = None
    last_dew = None
    dret = np.zeros(n)
    trades = []
    cur_entry_date = None

    for i in range(n):
        if i > 0:
            if Trend[i] == "C/Up" and cc[i-1] != "C/Up":
                last_dew = None
            if Trend[i] == "C/Dn" and cc[i-1] != "C/Dn":
                last_dew = None

        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1 + d)
            exited = False
            exit_reason = None
            if cum_mult - 1 >= 0.05:
                exited = True; exit_reason = "TP"
            elif Close[i] >= stop_qqq:
                exited = True; exit_reason = "ATR"
            elif (i != entry_idx) and Trend[i] == "C/Up" and i > 0 and cc[i-1] == "C/Dn":
                exited = True; exit_reason = "Flip"
            elif DEW[i] == "Buy":
                exited = True; exit_reason = "DEW"; last_dew = "Buy"
            if exited:
                if i + 1 < n:
                    pc = pc2o[i+1]
                    trade_ret = cum_mult * (1 + pc) - 1
                    dret[i] = (1 + d) * (1 + pc) - 1
                else:
                    trade_ret = cum_mult - 1; dret[i] = d
                trades.append((cur_entry_date, Date[i], trade_ret, exit_reason))
                in_trade = False; entry_idx = None; cum_mult = 1.0; stop_qqq = None
            else:
                dret[i] = d
                continue

        if not in_trade and DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn":
            if i + 1 < n:
                in_trade = True
                entry_idx = i + 1
                cum_mult = 1.0
                stop_qqq = Open_[i+1] + 2.0 * ATR14[i]
                last_dew = "Sell"
                cur_entry_date = Date[i+1]

    eq_curve = 100_000.0 * np.cumprod(1 + dret)
    return trades, eq_curve, dret


# ----------------------------------------------------------------------
# Step 3c - Isolated C/Up TQQQ (with v14 exits: BE@+9%, max-hold 70)
# ----------------------------------------------------------------------
def iso_cup(df):
    n = len(df)
    Date = df["Date"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    MTI = df["MTI"].values
    BSR = df["BSR"].values
    RT = df["RT"].values
    VIX = df["VIX"].values
    tqqq_rt = df["tqqq_rt"].values
    tqqq_atr = df["tqqq_atr"].values
    tqqq_open = df["tqqq_open"].values
    tqqq_close = df["tqqq_close"].values
    o2c = df["tqqq_o2c"].values
    c2c = df["tqqq_c2c"].values
    pc2o = df["tqqq_pc2o"].values

    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    tqqq_target = None
    tqqq_stop = None
    be_moved = False
    cup_entered = False  # in 3c: reset on every exit
    dret = np.zeros(n)
    trades = []
    cur_entry_date = None

    exit_breakdown = {"MTI": 0, "TP": 0, "Stop": 0, "Regime": 0, "Signal": 0, "MaxHold": 0, "Flip": 0}

    for i in range(n):
        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1 + d)
            # breakeven
            if not be_moved and cum_mult - 1 >= 0.09:
                tqqq_stop = tqqq_open[entry_idx]
                be_moved = True
            exited = False
            exit_reason = None
            # Priority: MTI > TP > Stop > MaxHold > Regime > C/Up flip > DEW Sell
            if (i != entry_idx) and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exited = True; exit_reason = "MTI"
            elif tqqq_close[i] >= tqqq_target:
                exited = True; exit_reason = "TP"
            elif tqqq_close[i] <= tqqq_stop:
                exited = True; exit_reason = "Stop"
            elif (i - entry_idx) >= 70:
                exited = True; exit_reason = "MaxHold"
            elif cc[i] == "C/Dn":
                exited = True; exit_reason = "Regime"
            elif (i != entry_idx) and i > 0 and Trend[i] == "C/Dn" and cc[i-1] == "C/Up":
                exited = True; exit_reason = "Flip"
            elif DEW[i] == "Sell":
                exited = True; exit_reason = "Signal"
            if exited:
                exit_breakdown[exit_reason] = exit_breakdown.get(exit_reason, 0) + 1
                if i + 1 < n:
                    pc = pc2o[i+1]
                    trade_ret = cum_mult * (1 + pc) - 1
                    dret[i] = (1 + d) * (1 + pc) - 1
                else:
                    trade_ret = cum_mult - 1; dret[i] = d
                trades.append((cur_entry_date, Date[i], trade_ret, exit_reason))
                in_trade = False; entry_idx = None; cum_mult = 1.0
                tqqq_target = None; tqqq_stop = None; be_moved = False
                cup_entered = False  # 3c: reset on exit
            else:
                dret[i] = d
                continue

        # Entry
        if not in_trade and i + 1 < n:
            cond = (cc[i] == "C/Up"
                    and not np.isnan(tqqq_rt[i]) and tqqq_rt[i] < 1.40
                    and not np.isnan(BSR[i]) and BSR[i] > 1.05
                    and not np.isnan(tqqq_atr[i]) and tqqq_atr[i] < 7.0
                    and not np.isnan(VIX[i]) and VIX[i] < 30
                    and not cup_entered)
            if cond:
                in_trade = True
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.50
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False
                cup_entered = True
                cur_entry_date = Date[i+1]

    eq = 100_000.0 * np.cumprod(1 + dret)
    return trades, eq, dret, exit_breakdown


# ----------------------------------------------------------------------
# Step 3d - Isolated C/Dn TQQQ
# ----------------------------------------------------------------------
def iso_cdn(df):
    n = len(df)
    Date = df["Date"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    MTI = df["MTI"].values
    RT = df["RT"].values
    VIX = df["VIX"].values  # NOT used as gate in isolated 3d
    tqqq_open = df["tqqq_open"].values
    tqqq_close = df["tqqq_close"].values
    o2c = df["tqqq_o2c"].values
    c2c = df["tqqq_c2c"].values
    pc2o = df["tqqq_pc2o"].values

    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    tqqq_target = None
    tqqq_stop = None
    be_moved = False
    last_dew = None
    dret = np.zeros(n)
    trades = []
    cur_entry_date = None

    exit_breakdown = {}

    for i in range(n):
        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1 + d)
            if not be_moved and cum_mult - 1 >= 0.09:
                tqqq_stop = tqqq_open[entry_idx]
                be_moved = True
            exited = False
            exit_reason = None
            # Priority: MTI > TP > Stop > MaxHold > DEW Sell > C/Up flip
            if (i != entry_idx) and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exited = True; exit_reason = "MTI"
            elif tqqq_close[i] >= tqqq_target:
                exited = True; exit_reason = "TP"
            elif tqqq_close[i] <= tqqq_stop:
                exited = True; exit_reason = "Stop"
            elif (i - entry_idx) >= 70:
                exited = True; exit_reason = "MaxHold"
            elif DEW[i] == "Sell":
                exited = True; exit_reason = "DEW Sell"
            elif (i != entry_idx) and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn":
                exited = True; exit_reason = "C/Up flip"
            if exited:
                exit_breakdown[exit_reason] = exit_breakdown.get(exit_reason, 0) + 1
                if i + 1 < n:
                    pc = pc2o[i+1]
                    trade_ret = cum_mult * (1 + pc) - 1
                    dret[i] = (1 + d) * (1 + pc) - 1
                else:
                    trade_ret = cum_mult - 1; dret[i] = d
                trades.append((cur_entry_date, Date[i], trade_ret, exit_reason))
                in_trade = False; entry_idx = None; cum_mult = 1.0
                tqqq_target = None; tqqq_stop = None; be_moved = False
                last_dew = None  # reset on EVERY exit per 3d spec
            else:
                dret[i] = d
                continue

        # Entry: cc=C/Dn, DEW=Buy, last_dew!=Buy, 0.95<=RT<1.00, MTI<1.00
        if not in_trade and i + 1 < n:
            cond = (cc[i] == "C/Dn"
                    and DEW[i] == "Buy"
                    and last_dew != "Buy"
                    and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                    and not np.isnan(MTI[i]) and MTI[i] < 1.00)
            if cond:
                in_trade = True
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.30  # v14: 1.30 (was 1.50)
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False
                last_dew = "Buy"
                cur_entry_date = Date[i+1]

    eq = 100_000.0 * np.cumprod(1 + dret)
    return trades, eq, dret, exit_breakdown


# ----------------------------------------------------------------------
# Combined engine - Steps 4-7
# ----------------------------------------------------------------------
def get_pmp(vix):
    v = vix if not np.isnan(vix) else 15.0
    if v < 20:
        return 1.30
    if v < 32:
        return 1.10
    return 0.50


def run_engine(df, mode):
    """
    mode:
      '4a' - no filters, no preemption, no pyramid, no decay
      '4b' - filters + preemption + VIX gate, no pyramid, no decay
      '5'  - 4b + pyramid sizing
      '7'  - 5 + VIX-conditional decay
    """
    use_filters = mode in ("4b", "5", "7")
    use_pyramid = mode in ("5", "7")
    use_decay = mode in ("7",)

    n = len(df)
    Date = df["Date"].values
    Close = df["Close"].values
    Open_ = df["Open"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    ATR14 = df["ATR14"].values
    MTI = df["MTI"].values
    BSR = df["BSR"].values
    RT = df["RT"].values
    VIX = df["VIX"].values
    tqqq_rt = df["tqqq_rt"].values
    tqqq_atr = df["tqqq_atr"].values
    tqqq_open = df["tqqq_open"].values
    tqqq_close = df["tqqq_close"].values

    o2c_q = df["o2c_QQQ"].values
    c2c_q = df["c2c_QQQ"].values
    pc2o_q = df["pc2o_QQQ"].values
    o2c_i = df["o2c_INV"].values
    c2c_i = df["c2c_INV"].values
    pc2o_i = df["pc2o_INV"].values
    o2c_t = df["tqqq_o2c"].values
    c2c_t = df["tqqq_c2c"].values
    pc2o_t = df["tqqq_pc2o"].values

    # State
    in_trade = False
    instrument = None
    entry_idx = None
    cum_mult = 1.0
    stop_qqq = None
    tqqq_target = None
    tqqq_stop = None
    be_moved = False
    last_dew = None
    fell_through = False
    cup_entered = False
    cdn_active = False
    pyramid_on = False
    entry_vix = None
    decayed_active = False

    dret = np.zeros(n)
    trades = []
    cur_entry_date = None
    cur_entry_kind = None  # 'cup' | 'cdn' | 'qqq' | 'inv'
    decay_transitions = 0
    raw_cum_mult_per_trade = []  # for TQQQ isolation

    tqqq_eq_pct = []  # eq% per TQQQ trade

    def select_returns(instr):
        if instr == "QQQ":
            return o2c_q, c2c_q, pc2o_q
        if instr == "INV":
            return o2c_i, c2c_i, pc2o_i
        if instr == "TQQQ":
            return o2c_t, c2c_t, pc2o_t
        raise ValueError("instrument not set")

    def qqq_inv_block_filter(kind, i):
        """Returns True if entry is BLOCKED."""
        if not use_filters:
            return False
        if kind == "QQQ":
            # OR - any one blocks
            cond = False
            if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90:
                cond = True
            if not np.isnan(BSR[i]) and BSR[i] > 1.50:
                cond = True
            if not np.isnan(MTI[i]) and MTI[i] > 1.15:
                cond = True
            return cond
        if kind == "INV":
            # AND - both must be true to block
            if (not np.isnan(RT[i]) and RT[i] > 0.95) and (not np.isnan(MTI[i]) and MTI[i] > 0.95):
                return True
            return False
        return False

    def vix_too_high(i):
        if not use_filters:
            return False
        return (not np.isnan(VIX[i])) and VIX[i] > 35

    def open_trade_eq_pct(i, eq_before, eq_after):
        if eq_before <= 0:
            return 0.0
        return (eq_after / eq_before) - 1

    eq_before_trade = 100_000.0
    cur_eq = 100_000.0

    for i in range(n):
        # Phase A
        fell_through = False
        if i > 0:
            if Trend[i] == "C/Up" and cc[i-1] != "C/Up":
                last_dew = None
                cup_entered = False
            if Trend[i] == "C/Dn" and cc[i-1] != "C/Dn":
                last_dew = None

        # Signal flags - cup_f / cdn_f
        cup_f = (cc[i] == "C/Up"
                 and not np.isnan(tqqq_rt[i]) and tqqq_rt[i] < 1.40
                 and not np.isnan(BSR[i]) and BSR[i] > 1.05
                 and not np.isnan(tqqq_atr[i]) and tqqq_atr[i] < 7.0
                 and not np.isnan(VIX[i]) and VIX[i] < 30)
        cdn_f = (cc[i] == "C/Dn"
                 and DEW[i] == "Buy" and last_dew != "Buy"
                 and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                 and not np.isnan(MTI[i]) and MTI[i] < 1.00
                 and not np.isnan(VIX[i]) and VIX[i] < 27)

        # Priority 1a/1b preemption (4b+)
        if use_filters and in_trade and i + 1 < n:
            # 1a: C/Up TQQQ preempts QQQ/INV
            if instrument in ("QQQ", "INV") and cup_f and not cup_entered:
                exiting = instrument
                _, _, pc_exit = select_returns(exiting)
                o2c_exit, c2c_exit, _ = select_returns(exiting)
                d = o2c_exit[i] if i == entry_idx else c2c_exit[i]
                cum_mult *= (1 + d)
                pc = pc_exit[i+1]
                trade_ret = cum_mult * (1 + pc) - 1
                dret[i] = (1 + d) * (1 + pc) - 1
                eq_after = eq_before_trade * (1 + trade_ret)
                trades.append((cur_entry_date, Date[i], trade_ret, "Preempt1a", exiting, eq_before_trade, eq_after))
                eq_before_trade = eq_after
                cur_eq = eq_after
                # Now enter TQQQ (cup variant)
                in_trade = True
                instrument = "TQQQ"
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.50
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False
                cup_entered = True
                cdn_active = False
                pyramid_on = False
                entry_vix = VIX[i+1] if i+1 < n else np.nan
                decayed_active = False
                last_dew = None
                cur_entry_date = Date[i+1]
                cur_entry_kind = "cup"
                fell_through = False
                continue
            # 1b: C/Dn switch while in QQQ
            if instrument == "QQQ" and not cdn_active:
                if (cc[i] == "C/Dn"
                    and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                    and not np.isnan(MTI[i]) and MTI[i] < 0.95):
                    o2c_exit, c2c_exit, pc_exit = select_returns("QQQ")
                    d = o2c_exit[i] if i == entry_idx else c2c_exit[i]
                    cum_mult *= (1 + d)
                    pc = pc_exit[i+1]
                    trade_ret = cum_mult * (1 + pc) - 1
                    dret[i] = (1 + d) * (1 + pc) - 1
                    eq_after = eq_before_trade * (1 + trade_ret)
                    trades.append((cur_entry_date, Date[i], trade_ret, "Preempt1b", "QQQ", eq_before_trade, eq_after))
                    eq_before_trade = eq_after
                    cur_eq = eq_after
                    in_trade = True
                    instrument = "TQQQ"
                    entry_idx = i + 1
                    cum_mult = 1.0
                    tqqq_target = tqqq_open[i+1] * 1.30  # cdn variant
                    tqqq_stop = tqqq_open[i+1] * 0.94
                    be_moved = False
                    cup_entered = False  # cdn entry; cup_entered set only on cup_f or 1a
                    cdn_active = True
                    pyramid_on = False
                    entry_vix = VIX[i+1] if i+1 < n else np.nan
                    decayed_active = False
                    last_dew = None
                    cur_entry_date = Date[i+1]
                    cur_entry_kind = "cdn"
                    fell_through = False
                    continue

        # Phase B - exit
        if in_trade:
            o2c_a, c2c_a, pc2o_a = select_returns(instrument)
            d = o2c_a[i] if i == entry_idx else c2c_a[i]
            cum_mult *= (1 + d)

            # Pyramid pre-state
            pyr_pre = pyramid_on if (use_pyramid and instrument == "TQQQ") else False
            pmp_pyr = get_pmp(VIX[i]) if pyr_pre else 1.0
            # Decay
            should_decay = False
            if use_decay and instrument == "TQQQ" and entry_vix is not None and not np.isnan(entry_vix):
                days_held = i - entry_idx
                if days_held >= 15 and (VIX[i] - entry_vix) >= 1.0:
                    should_decay = True
            decay_mult = 0.30 if should_decay else 1.0
            pmp = pmp_pyr * decay_mult

            # Track decay transitions
            if use_decay and instrument == "TQQQ":
                if should_decay != decayed_active:
                    decay_transitions += 1
                decayed_active = should_decay

            # Breakeven for TQQQ
            if instrument == "TQQQ" and not be_moved and cum_mult - 1 >= 0.09:
                tqqq_stop = tqqq_open[entry_idx]
                be_moved = True

            # Exit checks - priority order
            exited = False
            exit_reason = None
            ftype = "continue"  # continue | fall_through

            if instrument == "TQQQ" and (i != entry_idx) and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exited = True; exit_reason = "MTI"
            elif instrument == "TQQQ" and tqqq_close[i] >= tqqq_target:
                exited = True; exit_reason = "TQQQ TP"
            elif instrument == "TQQQ" and tqqq_close[i] <= tqqq_stop:
                exited = True; exit_reason = "TQQQ Stop"
            elif instrument == "TQQQ" and (i - entry_idx) >= 70:
                exited = True; exit_reason = "MaxHold"
            elif instrument == "QQQ" and cum_mult - 1 >= 0.05:
                exited = True; exit_reason = "QQQ TP"
            elif instrument == "QQQ" and Close[i] <= stop_qqq:
                exited = True; exit_reason = "QQQ ATR"
            elif instrument == "INV" and cum_mult - 1 >= 0.05:
                exited = True; exit_reason = "INV TP"
            elif instrument == "INV" and Close[i] >= stop_qqq:
                exited = True; exit_reason = "INV ATR"
            elif instrument == "TQQQ":
                # Regime exits for TQQQ
                if cur_entry_kind == "cup" and cc[i] == "C/Dn":
                    exited = True; exit_reason = "TQQQ Regime"
                elif cur_entry_kind == "cdn" and (DEW[i] == "Sell" or (i != entry_idx and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn")):
                    exited = True; exit_reason = "TQQQ Regime"
            # QQQ C/Dn flip
            if not exited and instrument == "QQQ" and (i != entry_idx) and i > 0 and Trend[i] == "C/Dn" and cc[i-1] == "C/Up":
                exited = True; exit_reason = "QQQ Flip"; ftype = "fall_through"
            # INV C/Up flip
            elif not exited and instrument == "INV" and (i != entry_idx) and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn":
                exited = True; exit_reason = "INV Flip"; ftype = "fall_through"
            # QQQ DEW Sell reversal
            elif not exited and instrument == "QQQ" and DEW[i] == "Sell":
                exited = True; exit_reason = "QQQ DEW"; ftype = "fall_through"
            # INV DEW Buy reversal
            elif not exited and instrument == "INV" and DEW[i] == "Buy":
                exited = True; exit_reason = "INV DEW"; ftype = "fall_through"

            if exited:
                # Compute exit dret with pmp scaling
                if i + 1 < n:
                    pc = pc2o_a[i+1]
                    if instrument == "TQQQ":
                        dret[i] = (1 + d * pmp) * (1 + pc * pmp) - 1
                    else:
                        dret[i] = (1 + d) * (1 + pc) - 1
                    trade_ret = cum_mult * (1 + pc) - 1  # raw trade return
                else:
                    if instrument == "TQQQ":
                        dret[i] = d * pmp
                    else:
                        dret[i] = d
                    trade_ret = cum_mult - 1
                eq_after = eq_before_trade * (1 + (dret[i] if False else trade_ret * 1))
                # For pyramid trades, equity change differs from trade_ret. Compute via dret accumulated since entry.
                # We compute eq_after via the change in cumprod(1+dret) from entry to exit inclusive.
                # Simpler: recompute eq from full dret. We'll just capture the trade and compute eq globally later.
                trades.append((cur_entry_date, Date[i], trade_ret, exit_reason, instrument, eq_before_trade, None))
                # Update state
                prev_instr = instrument
                in_trade = False
                instrument = None
                entry_idx = None
                cum_mult = 1.0
                stop_qqq = None
                tqqq_target = None
                tqqq_stop = None
                be_moved = False
                if prev_instr == "TQQQ":
                    last_dew = None
                    cdn_active = False
                    entry_vix = None
                    decayed_active = False
                    pyramid_on = False
                    if use_decay:
                        # decay reset doesn't count as a transition
                        pass
                if exit_reason == "QQQ DEW":
                    last_dew = "Sell"
                elif exit_reason == "INV DEW":
                    last_dew = "Buy"
                # Flips: don't change last_dew (Phase A already reset it this bar via CC transition)

                if ftype == "fall_through":
                    fell_through = True
                    # do NOT continue - allow Phase C
                else:
                    continue
            else:
                # Holding day
                if instrument == "TQQQ":
                    dret[i] = d * pmp
                else:
                    dret[i] = d
                # Latch pyramid AFTER dret written
                if use_pyramid and instrument == "TQQQ" and cum_mult - 1 >= 0.10 and not pyramid_on:
                    pyramid_on = True
                continue

        # Phase C - entries (only if not in_trade and (not fell_through guard checked per-entry))
        if not in_trade and i + 1 < n:
            # Priority: cup_f > cdn_f > QQQ Buy > INV Sell
            # cup_f
            if cup_f and not cup_entered and not fell_through:
                in_trade = True
                instrument = "TQQQ"
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.50
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False
                cup_entered = True
                cdn_active = False
                pyramid_on = False
                entry_vix = VIX[i+1]
                decayed_active = False
                cur_entry_date = Date[i+1]
                cur_entry_kind = "cup"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue
            # cdn_f
            if cdn_f and last_dew != "Buy" and not fell_through:
                in_trade = True
                instrument = "TQQQ"
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.30
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False
                cup_entered = False
                cdn_active = True
                pyramid_on = False
                entry_vix = VIX[i+1]
                decayed_active = False
                last_dew = "Buy"
                cur_entry_date = Date[i+1]
                cur_entry_kind = "cdn"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue
            # QQQ entry
            if DEW[i] == "Buy" and last_dew != "Buy" and not fell_through:
                if not qqq_inv_block_filter("QQQ", i) and not vix_too_high(i):
                    in_trade = True
                    instrument = "QQQ"
                    entry_idx = i + 1
                    cum_mult = 1.0
                    atr_mult = 2.0 if cc[i] == "C/Dn" else 1.0
                    stop_qqq = Open_[i+1] - atr_mult * ATR14[i]
                    last_dew = "Buy"
                    cur_entry_date = Date[i+1]
                    cur_entry_kind = "qqq"
                    eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                    continue
            # INV entry
            if DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn" and not fell_through:
                if not qqq_inv_block_filter("INV", i) and not vix_too_high(i):
                    in_trade = True
                    instrument = "INV"
                    entry_idx = i + 1
                    cum_mult = 1.0
                    stop_qqq = Open_[i+1] + 2.0 * ATR14[i]
                    last_dew = "Sell"
                    cur_entry_date = Date[i+1]
                    cur_entry_kind = "inv"
                    eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                    continue

    eq = 100_000.0 * np.cumprod(1 + dret)

    # Counts
    n_qqq = sum(1 for t in trades if t[4] == "QQQ")
    n_inv = sum(1 for t in trades if t[4] == "INV")
    n_tqqq = sum(1 for t in trades if t[4] == "TQQQ")
    return dict(
        trades=trades,
        eq=eq,
        dret=dret,
        n_qqq=n_qqq, n_inv=n_inv, n_tqqq=n_tqqq,
        decay_transitions=decay_transitions,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Step 1 - Load + Merge")
    print("=" * 80)
    df, df_full, qqq_raw = load_data()
    print(f"  df rows: {len(df)} (target: 6,606)")
    print(f"  df_full rows: {len(df_full)}")
    print(f"  ATR14[13] = {df['ATR14'].iloc[13]:.4f} (target: 4.8201)")
    print(f"  cc.head(3) = {df['cc'].head(3).tolist()} (target: ['C/Up','C/Up','C/Up'])")
    vix_check = df.loc[df["Date"] == pd.Timestamp("2020-03-16"), "VIX"]
    print(f"  VIX[2020-03-16] = {vix_check.iloc[0]:.2f} (target: 82.69)")
    tqqq_atr_first = df["tqqq_atr"].dropna().iloc[0]
    print(f"  tqqq_atr.dropna().iloc[0] = {tqqq_atr_first:.4f} (target: 7.9588)")
    spot1 = df[df["Date"] == pd.Timestamp("2005-07-06")][["RT", "MTI", "BSR"]].iloc[0]
    print(f"  Spot 2005-07-06: RT={spot1['RT']:.2f} MTI={spot1['MTI']:.2f} BSR={spot1['BSR']:.2f} (target: 1.08/1.43/2.49)")
    spot2 = df[df["Date"] == pd.Timestamp("2015-09-24")][["RT", "MTI", "BSR"]].iloc[0]
    print(f"  Spot 2015-09-24: RT={spot2['RT']:.2f} MTI={spot2['MTI']:.2f} BSR={spot2['BSR']:.2f} (target: 0.85/0.63/0.16)")

    print()
    print("=" * 80)
    print("Step 2 - DEW Signals")
    print("=" * 80)
    dew = compute_dew(qqq_raw)
    df = df.merge(dew, on="Date", how="left")
    df_full = df_full.merge(dew, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    df_full["DEW_Signal"] = df_full["DEW_Signal"].fillna("")
    n_buy = (df["DEW_Signal"] == "Buy").sum()
    n_sell = (df["DEW_Signal"] == "Sell").sum()
    n_blank = (df["DEW_Signal"] == "").sum()
    print(f"  Buy = {n_buy} (target: 194)")
    print(f"  Sell = {n_sell} (target: 195)")
    print(f"  Blank = {n_blank} (target: 6,217)")
    first3 = df[df["DEW_Signal"] != ""].head(3)
    print("  First 3 signals:")
    for _, r in first3.iterrows():
        print(f"    {r['Date'].strftime('%Y-%m-%d')} {r['DEW_Signal']} (cc={r['cc']})")

    print()
    print("=" * 80)
    print("Step 3a - Isolated QQQ")
    print("=" * 80)
    trades, eq, dret = iso_qqq(df)
    s = trade_stats(trades, eq, dret)
    print_stats("QQQ iso", s, "162 / $295,205 / CAGR 4.21% / Sharpe 0.5032 / MDD -22.89%")

    print()
    print("=" * 80)
    print("Step 3b - Isolated INV")
    print("=" * 80)
    trades, eq, dret = iso_inv(df)
    s = trade_stats(trades, eq, dret)
    print_stats("INV iso", s, "75 / $127,666 / CAGR 0.94% / Sharpe 0.1603 / MDD -27.24%")

    print()
    print("=" * 80)
    print("Step 3c - Isolated C/Up TQQQ")
    print("=" * 80)
    trades, eq, dret, brk = iso_cup(df)
    s = trade_stats(trades, eq, dret)
    print_stats("CUP iso", s, "121 / $4,600,652 / CAGR 15.71% / Sharpe 0.7051 / MDD -48.53%")
    print(f"  Exit breakdown: {brk} (target: Signal=53 Stop=31 Regime=27 TP=8 MTI=2)")

    print()
    print("=" * 80)
    print("Step 3d - Isolated C/Dn TQQQ")
    print("=" * 80)
    trades, eq, dret, brk = iso_cdn(df)
    s = trade_stats(trades, eq, dret)
    print_stats("CDN iso", s, "27 / $355,494 / CAGR 4.95% / Sharpe 0.6066 / MDD -16.05%")
    print(f"  Exit breakdown: {brk} (target: C/Up flip=16 DEW Sell=4 MTI=3 Stop=3 TP=1)")

    print()
    print("=" * 80)
    print("Step 4a - Combined (no filters)")
    print("=" * 80)
    r = run_engine(df, mode="4a")
    s = trade_stats(r["trades"], r["eq"], r["dret"])
    print_stats(f"4a (Q={r['n_qqq']} I={r['n_inv']} T={r['n_tqqq']})", s, "216 (Q=100,I=47,T=69) / $35,689,156 / CAGR 25.10% / Sharpe 1.0474 / MDD -32.76%")

    print()
    print("=" * 80)
    print("Step 4b - Combined (filters + preemption + VIX)")
    print("=" * 80)
    r = run_engine(df, mode="4b")
    s = trade_stats(r["trades"], r["eq"], r["dret"])
    print_stats(f"4b (Q={r['n_qqq']} I={r['n_inv']} T={r['n_tqqq']})", s, "187 (Q=61,I=40,T=86) / $557,706,505 / CAGR 38.92% / Sharpe 1.4601 / MDD -29.32%")

    print()
    print("=" * 80)
    print("Step 5 - Pyramid sizing")
    print("=" * 80)
    r = run_engine(df, mode="5")
    s = trade_stats(r["trades"], r["eq"], r["dret"])
    print_stats(f"5 (Q={r['n_qqq']} I={r['n_inv']} T={r['n_tqqq']})", s, "187 / $12,425,254,860 / CAGR 56.36% / Sharpe 1.8940 / MDD -21.54%")

    print()
    print("=" * 80)
    print("Step 7 - Decay (final)")
    print("=" * 80)
    r = run_engine(df, mode="7")
    s = trade_stats(r["trades"], r["eq"], r["dret"])
    print_stats(f"7 (Q={r['n_qqq']} I={r['n_inv']} T={r['n_tqqq']})", s, "187 / $12,425,254,860 / CAGR 56.36% / Sharpe 1.8940 / MDD -21.54%")
    print(f"  Decay transitions: {r['decay_transitions']} (target: 93 = 51 down + 42 up)")


if __name__ == "__main__":
    main()
