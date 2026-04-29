"""
Dampier Nitro++ v13
Build from raw CSV data files, checkpoint at every step.
Target: $1,636,197,971 final equity · CAGR 44.74% · Sharpe 1.6798 · Max DD -22.23%
        192 trades (QQQ=65, PSQ=39, TQQQ=88) · Jan 2000 – Apr 1 2026
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR    = str(Path(__file__).resolve().parent / "data" / "csv" / "history")
STARTING_K  = 100_000.0
BACKTEST_END = pd.Timestamp("2026-04-01")
SQQQ_CUTOFF  = pd.Timestamp("2010-02-11")
TQQQ_CUTOFF  = pd.Timestamp("2010-03-31")

# ── Helpers ────────────────────────────────────────────────────────────────────
def equity_curve(dret):
    return STARTING_K * np.cumprod(1.0 + np.asarray(dret, dtype=float))

def eq_stats(dret, dates):
    eq   = equity_curve(dret)
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    cagr = (eq[-1] / STARTING_K) ** (365.25 / days) - 1 if days > 0 else 0.0
    m = float(np.mean(dret)); s = float(np.std(dret, ddof=1))
    sharpe = m / s * np.sqrt(252) if s > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd  = float(((eq - peak) / peak * 100).min())
    return float(eq[-1]), cagr, sharpe, mdd

def checkpoint(label, trades, qqq_n, psq_n, tqqq_n, dret, dates):
    f, c, sh, mdd = eq_stats(dret, dates)
    print(f"\n{'='*62}")
    print(f"  🛑 {label}")
    print(f"  Trades:{trades}  QQQ:{qqq_n}  PSQ:{psq_n}  TQQQ:{tqqq_n}")
    print(f"  Final: ${f:>16,.0f}")
    print(f"  CAGR: {c*100:.2f}%   Sharpe: {sh:.4f}   Max DD: {mdd:.2f}%")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 · Load and Merge
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    def rd(fname, **kw):
        return pd.read_csv(
            os.path.join(DATA_DIR, fname),
            parse_dates=["Date"], encoding="utf-8-sig", **kw
        )

    # Load raw files
    qqq_raw = rd("qqq-from-vv.csv").rename(columns={"RT": "QQQ_RT_raw"})
    qqq_raw = qqq_raw.sort_values("Date").reset_index(drop=True)

    vv = rd("vectorvest-views-w3place-precision.csv")
    vv = vv.rename(columns={"VVC-RT": "RT", "BS Ratio": "BSR"})
    vv = vv.sort_values("Date").reset_index(drop=True)

    vix = rd("vix-from-yahoo.csv")[["Date", "Close"]].rename(columns={"Close": "VIX"})

    sqqq = rd("sqqq-from-yahoo.csv").sort_values("Date").reset_index(drop=True)

    syn   = rd("synthetic-tqqq-ohlc-1999-2010.csv").rename(columns={"RT_v6": "tqqq_rt"})
    syn   = syn.sort_values("Date").reset_index(drop=True)

    treal = rd("tqqq-from-vv.csv").rename(columns={"RT": "tqqq_rt"})
    treal = treal.sort_values("Date").reset_index(drop=True)

    # ── Stitch TQQQ: synthetic < 2010-03-31, real >= 2010-03-31
    tqqq = pd.concat(
        [syn[syn["Date"] < TQQQ_CUTOFF], treal[treal["Date"] >= TQQQ_CUTOFF]],
        ignore_index=True
    ).sort_values("Date").reset_index(drop=True)
    tqqq["tqqq_o2c"]  = tqqq["Close"] / tqqq["Open"] - 1
    tqqq["tqqq_c2c"]  = tqqq["Close"].pct_change()
    tqqq["tqqq_pc2o"] = tqqq["Open"] / tqqq["Close"].shift(1) - 1

    # tqqq_atr: 10-bar simple ATR% on FULL stitched TQQQ series
    tc, th, tl = tqqq["Close"].values, tqqq["High"].values, tqqq["Low"].values
    N_t = len(tqqq)
    tr_t = np.empty(N_t); tr_t[0] = th[0] - tl[0]
    for i in range(1, N_t):
        tr_t[i] = max(th[i]-tl[i], abs(th[i]-tc[i-1]), abs(tl[i]-tc[i-1]))
    tatr = np.full(N_t, np.nan)
    for i in range(9, N_t):
        tatr[i] = np.mean(tr_t[i-9:i+1]) / tc[i] * 100.0
    tqqq["tqqq_atr"] = tatr

    # ── Merge: QQQ ∩ VV (inner), + VIX, + TQQQ (left)
    df = qqq_raw.merge(vv[["Date","RT","BSR","MTI","Trend"]], on="Date", how="inner")
    df = df.merge(vix, on="Date", how="left")
    tqqq_m = tqqq[["Date","tqqq_rt","tqqq_o2c","tqqq_c2c","tqqq_pc2o","tqqq_atr",
                    "Open","High","Low","Close"]].rename(columns={
        "Open":"tqqq_open","High":"tqqq_high","Low":"tqqq_low","Close":"tqqq_close"})
    df = df.merge(tqqq_m, on="Date", how="left")

    # ── QQQ return columns
    df["o2c_QQQ"]  = df["Close"] / df["Open"] - 1
    df["c2c_QQQ"]  = df["Close"].pct_change()
    df["pc2o_QQQ"] = df["Open"] / df["Close"].shift(1) - 1

    # ── PSQ synthetic (pre-SQQQ cutoff)
    df["o2c_PSQ"]  = -df["o2c_QQQ"]
    df["c2c_PSQ"]  = -df["c2c_QQQ"] - 0.0095 / 252.0
    df["pc2o_PSQ"] = -df["pc2o_QQQ"]

    # ── PSQ real SQQQ/3 (post cutoff)
    sqqq = sqqq.copy()
    sqqq_adj          = sqqq["Adj Close"] / sqqq["Close"]
    sqqq["open_adj"]  = sqqq["Open"] * sqqq_adj
    sqqq["o2c_S3"]    = (sqqq["Adj Close"] / sqqq["open_adj"] - 1) / 3.0
    sqqq["c2c_S3"]    = sqqq["Adj Close"].pct_change() / 3.0
    sqqq["pc2o_S3"]   = (sqqq["open_adj"] / sqqq["Adj Close"].shift(1) - 1) / 3.0
    df = df.merge(sqqq[["Date","o2c_S3","c2c_S3","pc2o_S3"]], on="Date", how="left")
    m = df["Date"] >= SQQQ_CUTOFF
    df.loc[m, "o2c_PSQ"]  = df.loc[m, "o2c_S3"]
    df.loc[m, "c2c_PSQ"]  = df.loc[m, "c2c_S3"]
    df.loc[m, "pc2o_PSQ"] = df.loc[m, "pc2o_S3"]

    # ── cc: carry-forward Confirmed Call (computed on FULL series before filter)
    cc = []; cur = "C/Up"
    for t in df["Trend"]:
        if t == "C/Up":   cur = "C/Up"
        elif t == "C/Dn": cur = "C/Dn"
        cc.append(cur)
    df["cc"] = cc

    # ── Filter to backtest window (keep full df for Step 6)
    df_full = df[df["Date"] >= "2000-01-01"].copy().reset_index(drop=True)
    df      = df[(df["Date"] >= "2000-01-01") & (df["Date"] <= BACKTEST_END)].copy().reset_index(drop=True)

    # ── ATR14: Wilder ATR on QQQ (seeded at index 13, computed on each df)
    def add_atr14(d):
        H = d["High"].values; L = d["Low"].values; C = d["Close"].values; N = len(d)
        TR = np.empty(N); TR[0] = H[0] - L[0]
        for i in range(1, N):
            TR[i] = max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
        atr = np.zeros(N)
        atr[13] = np.mean(TR[:14])
        for i in range(14, N):
            atr[i] = (atr[i-1]*13 + TR[i]) / 14.0
        d["ATR14"] = atr

    add_atr14(df)
    add_atr14(df_full)

    return df, df_full, qqq_raw


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 · DEW Oscillator
# ──────────────────────────────────────────────────────────────────────────────
def compute_dew(qqq_raw):
    """Compute DEW on raw QQQ series (starts 1999-07-01 for warmup)."""
    C = qqq_raw["Close"].values.astype(float)
    N = len(C)

    D = (pd.Series(C) - pd.Series(C).rolling(20).mean().shift(11)).round(2).values
    EMA10   = pd.Series(C).ewm(span=10, adjust=False).mean().values
    E_upper = EMA10 * 1.06
    E_lower = EMA10 * 0.94

    weights = np.arange(1.0, 31.0)   # 1..30, newest=30
    W = np.full(N, np.nan)
    for i in range(29, N):
        W[i] = np.dot(C[i-29:i+1], weights) / 465.0

    sig   = np.full(N, "", dtype=object)
    state = "Normal"
    last_fired = None

    for i in range(41, N):
        c, d, w = C[i], D[i], W[i]
        if np.isnan(d) or np.isnan(w):
            continue

        # Step 1: candidate (state entering this bar)
        if state == "Normal":
            cand = "Buy" if (c > w and d > 0) else "Sell" if (c < w and d < 0) else None
        else:  # Primed — OR logic
            cand = "Buy" if (c > w or d > 0) else "Sell" if (c < w or d < 0) else None

        # Step 2: suppress same direction as last fired
        if cand == last_fired:
            cand = None

        # Step 3: record signal; reset to Normal
        if cand:
            sig[i]     = cand
            last_fired = cand
            state      = "Normal"

        # Step 4: pierce check → Primed next bar
        if c > E_upper[i] or c < E_lower[i]:
            state = "Primed"

    out = qqq_raw[["Date"]].copy()
    out["DEW_Signal"] = sig
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Array extractor (used by all engine functions)
# ──────────────────────────────────────────────────────────────────────────────
def arr(df):
    """Return dict of numpy arrays from df."""
    def fv(col): return df[col].values.astype(float)
    def sv(col): return df[col].values
    return dict(
        N      = len(df),
        dates  = sv("Date"),
        cc     = sv("cc"),
        Trend  = sv("Trend"),
        DEW    = sv("DEW_Signal"),
        RT     = fv("RT"),
        MTI    = fv("MTI"),
        BSR    = fv("BSR"),
        VIX    = fv("VIX"),
        ATR14  = fv("ATR14"),
        Open   = fv("Open"),
        Close  = fv("Close"),
        o2c_Q  = fv("o2c_QQQ"),
        c2c_Q  = fv("c2c_QQQ"),
        pc2o_Q = fv("pc2o_QQQ"),
        o2c_P  = fv("o2c_PSQ"),
        c2c_P  = fv("c2c_PSQ"),
        pc2o_P = fv("pc2o_PSQ"),
        to2c   = fv("tqqq_o2c"),
        tc2c   = fv("tqqq_c2c"),
        tpc2o  = fv("tqqq_pc2o"),
        tatr   = fv("tqqq_atr"),
        trt    = fv("tqqq_rt"),
        topen  = fv("tqqq_open"),
        tclose = fv("tqqq_close"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3a · Isolated QQQ Long
# ──────────────────────────────────────────────────────────────────────────────
def run_3a_qqq(df):
    a = arr(df); N = a["N"]
    dret = np.zeros(N)
    trades = []

    in_trade  = False
    entry_idx = None
    cum_mult  = 1.0
    stop_lvl  = None
    last_dew  = None

    for i in range(N - 1):
        # Phase A: CC transitions reset last_dew
        if i > 0:
            if a["cc"][i] == "C/Up" and a["cc"][i-1] != "C/Up": last_dew = None
            if a["cc"][i] == "C/Dn" and a["cc"][i-1] != "C/Dn": last_dew = None

        # Phase B: exits
        if in_trade:
            d = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
            cum_mult *= (1.0 + d)
            exited = False; exit_type = None

            if cum_mult - 1.0 >= 0.05:
                exit_type = "TP"; exited = True

            if not exited and a["Close"][i] <= stop_lvl:
                exit_type = "ATR"; exited = True

            if not exited and i != entry_idx and a["Trend"][i] == "C/Dn" and a["cc"][i-1] == "C/Up":
                exit_type = "Flip"; exited = True

            if not exited and a["DEW"][i] == "Sell":
                last_dew = "Sell"; exit_type = "DEW"; exited = True

            if exited:
                dret[i] = (1.0 + d) * (1.0 + a["pc2o_Q"][i+1]) - 1.0
                trade_ret = cum_mult * (1.0 + a["pc2o_Q"][i+1]) - 1.0
                trades.append(dict(
                    entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                    inst="QQQ", ret=trade_ret, exit=exit_type))
                in_trade = False; entry_idx = None; cum_mult = 1.0; stop_lvl = None
                continue
            else:
                dret[i] = d
                continue

        # Phase C: entry
        if a["DEW"][i] == "Buy" and last_dew != "Buy":
            in_trade  = True
            entry_idx = i + 1
            cum_mult  = 1.0
            last_dew  = "Buy"
            atr_mult  = 2.0 if a["cc"][i] == "C/Dn" else 1.0
            stop_lvl  = a["Open"][i+1] - atr_mult * a["ATR14"][i]

    return trades, dret


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3b · Isolated PSQ Short
# ──────────────────────────────────────────────────────────────────────────────
def run_3b_psq(df):
    a = arr(df); N = a["N"]
    dret = np.zeros(N)
    trades = []

    in_trade  = False
    entry_idx = None
    cum_mult  = 1.0
    stop_lvl  = None
    last_dew  = None

    for i in range(N - 1):
        if i > 0:
            if a["cc"][i] == "C/Up" and a["cc"][i-1] != "C/Up": last_dew = None
            if a["cc"][i] == "C/Dn" and a["cc"][i-1] != "C/Dn": last_dew = None

        if in_trade:
            d = a["o2c_P"][i] if i == entry_idx else a["c2c_P"][i]
            cum_mult *= (1.0 + d)
            exited = False; exit_type = None

            if cum_mult - 1.0 >= 0.05:
                exit_type = "TP"; exited = True

            # PSQ stop: QQQ Close >= stop_level (QQQ rallied above stop)
            if not exited and a["Close"][i] >= stop_lvl:
                exit_type = "ATR"; exited = True

            # C/Up flip
            if not exited and i != entry_idx and a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn":
                exit_type = "Flip"; exited = True

            if not exited and a["DEW"][i] == "Buy":
                last_dew = "Buy"; exit_type = "DEW"; exited = True

            if exited:
                dret[i] = (1.0 + d) * (1.0 + a["pc2o_P"][i+1]) - 1.0
                trade_ret = cum_mult * (1.0 + a["pc2o_P"][i+1]) - 1.0
                trades.append(dict(
                    entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                    inst="PSQ", ret=trade_ret, exit=exit_type))
                in_trade = False; entry_idx = None; cum_mult = 1.0; stop_lvl = None
                continue
            else:
                dret[i] = d
                continue

        # PSQ entry: DEW Sell AND cc == C/Dn
        if a["DEW"][i] == "Sell" and last_dew != "Sell" and a["cc"][i] == "C/Dn":
            in_trade  = True
            entry_idx = i + 1
            cum_mult  = 1.0
            last_dew  = "Sell"
            stop_lvl  = a["Open"][i+1] + 2.0 * a["ATR14"][i]

    return trades, dret


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3c · Isolated C/Up TQQQ
# ──────────────────────────────────────────────────────────────────────────────
def run_3c_cup_tqqq(df):
    a = arr(df); N = a["N"]
    dret = np.zeros(N)
    trades = []

    in_trade    = False
    entry_idx   = None
    cum_mult    = 1.0
    tqqq_target = None
    tqqq_stop   = None
    cup_entered = False   # reset on every EXIT in isolated mode

    for i in range(N - 1):
        # cup_f signal
        cup_f = (a["cc"][i] == "C/Up"
                 and a["trt"][i] < 1.40
                 and a["BSR"][i] > 1.05
                 and a["tatr"][i] < 7.0
                 and a["VIX"][i] < 30.0
                 and not cup_entered)

        if in_trade:
            d = a["to2c"][i] if i == entry_idx else a["tc2c"][i]
            cum_mult *= (1.0 + d)
            exited = False; exit_type = None

            # MTI exit (not entry bar)
            if i != entry_idx and a["MTI"][i] < 0.75:
                exit_type = "MTI"; exited = True

            if not exited and a["tclose"][i] >= tqqq_target:
                exit_type = "TP"; exited = True

            if not exited and a["tclose"][i] <= tqqq_stop:
                exit_type = "Stop"; exited = True

            # Regime exit: cc flips to C/Dn
            if not exited and a["cc"][i] == "C/Dn":
                exit_type = "Signal"; exited = True  # spec calls it "Signal" in list

            # DEW Sell
            if not exited and a["DEW"][i] == "Sell":
                exit_type = "Signal"; exited = True

            if exited:
                dret[i] = (1.0 + d) * (1.0 + a["tpc2o"][i+1]) - 1.0
                trade_ret = cum_mult * (1.0 + a["tpc2o"][i+1]) - 1.0
                trades.append(dict(
                    entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                    inst="TQQQ", ret=trade_ret, exit=exit_type))
                in_trade = False; entry_idx = None; cum_mult = 1.0
                tqqq_target = None; tqqq_stop = None
                cup_entered = False  # reset on exit in isolated mode
                continue
            else:
                dret[i] = d
                continue

        if cup_f:
            in_trade    = True
            entry_idx   = i + 1
            cum_mult    = 1.0
            cup_entered = True
            tqqq_target = a["topen"][i+1] * 1.50
            tqqq_stop   = a["topen"][i+1] * 0.94
            trades_meta = dict(entry_bar=i)

    return trades, dret


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3d · Isolated C/Dn TQQQ
# ──────────────────────────────────────────────────────────────────────────────
def run_3d_cdn_tqqq(df):
    a = arr(df); N = a["N"]
    dret = np.zeros(N)
    trades = []

    in_trade    = False
    entry_idx   = None
    cum_mult    = 1.0
    tqqq_target = None
    tqqq_stop   = None
    last_dew    = None

    for i in range(N - 1):
        if i > 0:
            if a["cc"][i] == "C/Up" and a["cc"][i-1] != "C/Up": last_dew = None
            if a["cc"][i] == "C/Dn" and a["cc"][i-1] != "C/Dn": last_dew = None

        cdn_f = (a["cc"][i] == "C/Dn"
                 and a["DEW"][i] == "Buy"
                 and last_dew != "Buy"
                 and 0.95 <= a["RT"][i] < 1.00
                 and a["MTI"][i] < 1.00
                 and not in_trade)

        if in_trade:
            d = a["to2c"][i] if i == entry_idx else a["tc2c"][i]
            cum_mult *= (1.0 + d)
            exited = False; exit_type = None

            if i != entry_idx and a["MTI"][i] < 0.75:
                exit_type = "MTI"; exited = True

            if not exited and a["tclose"][i] >= tqqq_target:
                exit_type = "TP"; exited = True

            if not exited and a["tclose"][i] <= tqqq_stop:
                exit_type = "Stop"; exited = True

            if not exited and a["DEW"][i] == "Sell":
                exit_type = "DEW Sell"; exited = True

            # C/Up flip
            if not exited and a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn":
                exit_type = "C/Up flip"; exited = True

            if exited:
                dret[i] = (1.0 + d) * (1.0 + a["tpc2o"][i+1]) - 1.0
                trade_ret = cum_mult * (1.0 + a["tpc2o"][i+1]) - 1.0
                trades.append(dict(
                    entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                    inst="TQQQ", ret=trade_ret, exit=exit_type))
                in_trade = False; entry_idx = None; cum_mult = 1.0
                tqqq_target = None; tqqq_stop = None
                last_dew = None  # reset on every exit in isolated 3d
                continue
            else:
                dret[i] = d
                continue

        if cdn_f:
            in_trade    = True
            entry_idx   = i + 1
            cum_mult    = 1.0
            last_dew    = "Buy"
            tqqq_target = a["topen"][i+1] * 1.50
            tqqq_stop   = a["topen"][i+1] * 0.94

    return trades, dret


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4a · Combined Engine — No Filters, No Preemption
# ──────────────────────────────────────────────────────────────────────────────
def run_4a_combined(df):
    a = arr(df); N = a["N"]
    dret = np.zeros(N)
    trades = []

    in_trade    = False
    instrument  = None
    entry_idx   = None
    cum_mult    = 1.0
    stop_qqq    = None
    tqqq_target = None
    tqqq_stop   = None
    last_dew    = None
    cup_entered = False
    cdn_active  = False
    pyramid_on  = False

    for i in range(N - 1):
        # Phase A: CC transitions reset state
        fell_through = False
        flip_blocked = None   # "QQQ" = QQQ is blocked, "PSQ" = PSQ is blocked

        if i > 0:
            if a["Trend"][i] == "C/Up" and a["cc"][i-1] != "C/Up":
                last_dew = None; cup_entered = False
            if a["Trend"][i] == "C/Dn" and a["cc"][i-1] != "C/Dn":
                last_dew = None

        # Signal flags
        cup_f = (a["cc"][i] == "C/Up"
                 and a["trt"][i] < 1.40
                 and a["BSR"][i] > 1.05
                 and a["tatr"][i] < 7.0
                 and a["VIX"][i] < 30.0)
        cdn_f = (a["cc"][i] == "C/Dn"
                 and a["DEW"][i] == "Buy"
                 and last_dew != "Buy"
                 and 0.95 <= a["RT"][i] < 1.00
                 and a["MTI"][i] < 1.00)

        # Phase B: exits
        if in_trade:
            if instrument == "TQQQ":
                d = a["to2c"][i] if i == entry_idx else a["tc2c"][i]
            elif instrument == "QQQ":
                d = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
            else:
                d = a["o2c_P"][i] if i == entry_idx else a["c2c_P"][i]
            cum_mult *= (1.0 + d)
            exited = False; exit_type = None; is_fall_through = False

            # 1. MTI (TQQQ, not entry bar) → continue
            if instrument == "TQQQ" and i != entry_idx and a["MTI"][i] < 0.75:
                exit_type = "MTI"; exited = True

            # 2. TQQQ TP → continue
            if not exited and instrument == "TQQQ" and a["tclose"][i] >= tqqq_target:
                exit_type = "TP"; exited = True

            # 3. TQQQ stop → continue
            if not exited and instrument == "TQQQ" and a["tclose"][i] <= tqqq_stop:
                exit_type = "Stop"; exited = True

            # 4. QQQ TP → continue
            if not exited and instrument == "QQQ" and cum_mult - 1.0 >= 0.05:
                exit_type = "TP"; exited = True

            # 5. QQQ ATR → continue
            if not exited and instrument == "QQQ" and a["Close"][i] <= stop_qqq:
                exit_type = "ATR"; exited = True

            # 6. PSQ TP → continue
            if not exited and instrument == "PSQ" and cum_mult - 1.0 >= 0.05:
                exit_type = "TP"; exited = True

            # 7. PSQ ATR → continue
            if not exited and instrument == "PSQ" and a["Close"][i] >= stop_qqq:
                exit_type = "ATR"; exited = True

            # 8. TQQQ regime → continue
            if not exited and instrument == "TQQQ":
                if cdn_active and (a["DEW"][i] == "Sell" or
                                   (a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn")):
                    exit_type = "Regime"; exited = True
                elif not cdn_active and a["cc"][i] == "C/Dn":
                    exit_type = "Regime"; exited = True

            # 9. QQQ C/Dn flip → fall through, PSQ blocked
            if not exited and instrument == "QQQ" and i != entry_idx:
                if a["Trend"][i] == "C/Dn" and a["cc"][i-1] == "C/Up":
                    fell_through = True; flip_blocked = "PSQ"
                    exit_type = "Flip"; exited = True; is_fall_through = True

            # 10. PSQ C/Up flip → fall through, QQQ blocked
            if not exited and instrument == "PSQ" and i != entry_idx:
                if a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn":
                    fell_through = True; flip_blocked = "QQQ"
                    exit_type = "Flip"; exited = True; is_fall_through = True

            # 11. QQQ DEW Sell → fall through, last_dew="Sell" blocks PSQ
            if not exited and instrument == "QQQ" and a["DEW"][i] == "Sell":
                last_dew = "Sell"; fell_through = True
                exit_type = "DEW"; exited = True; is_fall_through = True

            # 12. PSQ DEW Buy → fall through, last_dew="Buy" blocks QQQ
            if not exited and instrument == "PSQ" and a["DEW"][i] == "Buy":
                last_dew = "Buy"; fell_through = True
                exit_type = "DEW"; exited = True; is_fall_through = True

            if exited:
                pc2o = a["tpc2o"][i+1] if instrument=="TQQQ" else \
                       a["pc2o_Q"][i+1] if instrument=="QQQ" else a["pc2o_P"][i+1]
                dret[i] = (1.0 + d) * (1.0 + pc2o) - 1.0
                trade_ret = cum_mult * (1.0 + pc2o) - 1.0
                trades.append(dict(
                    entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                    inst=instrument, ret=trade_ret, exit=exit_type))
                is_tqqq = (instrument == "TQQQ")
                in_trade = False; instrument = None; entry_idx = None
                cum_mult = 1.0; stop_qqq = None; tqqq_target = None
                tqqq_stop = None; pyramid_on = False
                if is_tqqq:
                    last_dew = None; cdn_active = False
                # Exits 1-8: skip Phase C. Exits 9-12: fall through to Phase C.
                if not is_fall_through:
                    continue
            else:
                dret[i] = d
                continue   # still in trade, skip Phase C

        # Phase C (only after fall-through exits, or no trade)
        if not in_trade:
            if cup_f and not cup_entered and not fell_through:
                in_trade = True; instrument = "TQQQ"; cdn_active = False
                entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
                cup_entered = True
                tqqq_target = a["topen"][i+1] * 1.50
                tqqq_stop   = a["topen"][i+1] * 0.94
                continue
            if cdn_f and not fell_through:
                in_trade = True; instrument = "TQQQ"; cdn_active = True
                entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
                tqqq_target = a["topen"][i+1] * 1.50
                tqqq_stop   = a["topen"][i+1] * 0.94
                continue
            if a["DEW"][i] == "Buy" and last_dew != "Buy" and flip_blocked != "QQQ":
                in_trade  = True; instrument = "QQQ"
                entry_idx = i + 1; cum_mult = 1.0; last_dew = "Buy"
                atr_mult  = 2.0 if a["cc"][i] == "C/Dn" else 1.0
                stop_qqq  = a["Open"][i+1] - atr_mult * a["ATR14"][i]
                continue
            if (a["DEW"][i] == "Sell" and last_dew != "Sell"
                    and a["cc"][i] == "C/Dn" and flip_blocked != "PSQ"):
                in_trade  = True; instrument = "PSQ"
                entry_idx = i + 1; cum_mult = 1.0; last_dew = "Sell"
                stop_qqq  = a["Open"][i+1] + 2.0 * a["ATR14"][i]

    return trades, dret


# ──────────────────────────────────────────────────────────────────────────────
# Shared Phase B + Phase C engine (used by 4b and 5)
# use_hf_pyr=True enables health factor and pyramid scaling
# ──────────────────────────────────────────────────────────────────────────────
def _run_engine(df, use_hf_pyr=False):
    a = arr(df); N = a["N"]
    dret = np.zeros(N)
    trades = []

    in_trade    = False
    instrument  = None
    entry_idx   = None
    cum_mult    = 1.0
    stop_qqq    = None
    tqqq_target = None
    tqqq_stop   = None
    last_dew    = None
    cup_entered = False
    cdn_active  = False
    pyramid_on  = False

    for i in range(N - 1):
        fell_through = False
        flip_blocked = None   # "PSQ" = PSQ blocked after QQQ flip; "QQQ" = QQQ blocked after PSQ flip

        # Health factor
        if use_hf_pyr:
            vix_i = a["VIX"][i]; tatr_i = a["tatr"][i]; bsr_i = a["BSR"][i]
            if np.isnan(vix_i) or np.isnan(tatr_i):
                hf = 1.0
            else:
                hf = min(1.0, max(0.25,
                    (30.0 - vix_i) / 20.0 * (8.0 / max(tatr_i, 4.0)) * max(bsr_i, 0.5)))
        else:
            hf = 1.0

        # Phase A: CC transitions
        if i > 0:
            if a["Trend"][i] == "C/Up" and a["cc"][i-1] != "C/Up":
                last_dew = None; cup_entered = False
            if a["Trend"][i] == "C/Dn" and a["cc"][i-1] != "C/Dn":
                last_dew = None

        # Signal flags
        cup_f = (a["cc"][i] == "C/Up"
                 and a["trt"][i] < 1.40
                 and a["BSR"][i] > 1.05
                 and a["tatr"][i] < 7.0
                 and a["VIX"][i] < 30.0)
        cdn_f = (a["cc"][i] == "C/Dn"
                 and a["DEW"][i] == "Buy"
                 and last_dew != "Buy"
                 and 0.95 <= a["RT"][i] < 1.00
                 and a["MTI"][i] < 1.00)

        # ── Priority 1a: C/Up TQQQ preempts open QQQ or PSQ
        if in_trade and instrument in ("QQQ", "PSQ") and cup_f and not cup_entered:
            d = a["o2c_Q"][i] if (instrument=="QQQ" and i==entry_idx) else \
                a["c2c_Q"][i] if instrument=="QQQ" else \
                a["o2c_P"][i] if i==entry_idx else a["c2c_P"][i]
            cum_mult *= (1.0 + d)
            pc2o = a["pc2o_Q"][i+1] if instrument == "QQQ" else a["pc2o_P"][i+1]
            raw_exit = (1.0 + d) * (1.0 + pc2o) - 1.0
            dret[i] = raw_exit * hf
            trade_ret = cum_mult * (1.0 + pc2o) - 1.0
            trades.append(dict(entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                               inst=instrument, ret=trade_ret, exit="Preempt"))
            in_trade = True; instrument = "TQQQ"; cdn_active = False
            entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
            cup_entered = True; stop_qqq = None
            tqqq_target = a["topen"][i+1] * 1.50; tqqq_stop = a["topen"][i+1] * 0.94
            continue

        # ── Priority 1b: C/Dn switch while in QQQ
        if (in_trade and instrument == "QQQ" and not cdn_active
                and a["cc"][i] == "C/Dn"
                and 0.95 <= a["RT"][i] < 1.00
                and a["MTI"][i] < 0.95):
            d = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
            cum_mult *= (1.0 + d)
            pc2o_1b = a["pc2o_Q"][i+1]
            raw_exit_1b = (1.0 + d) * (1.0 + pc2o_1b) - 1.0
            dret[i] = raw_exit_1b * hf
            trade_ret = cum_mult * (1.0 + pc2o_1b) - 1.0
            trades.append(dict(entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                               inst="QQQ", ret=trade_ret, exit="Preempt"))
            in_trade = True; instrument = "TQQQ"; cdn_active = True
            entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
            stop_qqq = None; last_dew = None
            tqqq_target = a["topen"][i+1] * 1.50; tqqq_stop = a["topen"][i+1] * 0.94
            continue

        # Phase B: exits
        if in_trade:
            if instrument == "TQQQ":
                d = a["to2c"][i] if i == entry_idx else a["tc2c"][i]
            elif instrument == "QQQ":
                d = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
            else:
                d = a["o2c_P"][i] if i == entry_idx else a["c2c_P"][i]
            cum_mult *= (1.0 + d)

            # Pyramid (TQQQ only, dret scaling)
            if use_hf_pyr and instrument == "TQQQ" and cum_mult - 1.0 >= 0.20 and not pyramid_on:
                pyramid_on = True
            pyr = 1.25 if (use_hf_pyr and instrument == "TQQQ" and pyramid_on) else 1.0
            raw_d = d * hf * pyr

            exited = False; exit_type = None; is_fall_through = False

            if instrument == "TQQQ" and i != entry_idx and a["MTI"][i] < 0.75:
                exit_type = "MTI"; exited = True
            if not exited and instrument == "TQQQ" and a["tclose"][i] >= tqqq_target:
                exit_type = "TP"; exited = True
            if not exited and instrument == "TQQQ" and a["tclose"][i] <= tqqq_stop:
                exit_type = "Stop"; exited = True
            if not exited and instrument == "QQQ" and cum_mult - 1.0 >= 0.05:
                exit_type = "TP"; exited = True
            if not exited and instrument == "QQQ" and a["Close"][i] <= stop_qqq:
                exit_type = "ATR"; exited = True
            if not exited and instrument == "PSQ" and cum_mult - 1.0 >= 0.05:
                exit_type = "TP"; exited = True
            if not exited and instrument == "PSQ" and a["Close"][i] >= stop_qqq:
                exit_type = "ATR"; exited = True

            if not exited and instrument == "TQQQ":
                if cdn_active and (a["DEW"][i] == "Sell" or
                                   (a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn")):
                    exit_type = "Regime"; exited = True
                elif not cdn_active and a["cc"][i] == "C/Dn":
                    exit_type = "Regime"; exited = True

            # QQQ C/Dn flip → fall through, PSQ blocked
            if not exited and instrument == "QQQ" and i != entry_idx:
                if a["Trend"][i] == "C/Dn" and a["cc"][i-1] == "C/Up":
                    fell_through = True; flip_blocked = "PSQ"
                    exit_type = "Flip"; exited = True; is_fall_through = True

            # PSQ C/Up flip → fall through, QQQ blocked
            if not exited and instrument == "PSQ" and i != entry_idx:
                if a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn":
                    fell_through = True; flip_blocked = "QQQ"
                    exit_type = "Flip"; exited = True; is_fall_through = True

            # QQQ DEW Sell → fall through, last_dew="Sell" blocks PSQ
            if not exited and instrument == "QQQ" and a["DEW"][i] == "Sell":
                last_dew = "Sell"; fell_through = True
                exit_type = "DEW"; exited = True; is_fall_through = True

            # PSQ DEW Buy → fall through, last_dew="Buy" blocks QQQ
            if not exited and instrument == "PSQ" and a["DEW"][i] == "Buy":
                last_dew = "Buy"; fell_through = True
                exit_type = "DEW"; exited = True; is_fall_through = True

            if exited:
                pc2o = a["tpc2o"][i+1] if instrument=="TQQQ" else \
                       a["pc2o_Q"][i+1] if instrument=="QQQ" else a["pc2o_P"][i+1]
                # raw 4b exit return; scale entire exit-day return by hf * pyr
                raw_exit = (1.0 + d) * (1.0 + pc2o) - 1.0
                dret[i] = raw_exit * hf * pyr
                trade_ret = cum_mult * (1.0 + pc2o) - 1.0
                trades.append(dict(entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                                   inst=instrument, ret=trade_ret, exit=exit_type))
                is_tqqq = (instrument == "TQQQ")
                in_trade = False; instrument = None; entry_idx = None
                cum_mult = 1.0; stop_qqq = None; tqqq_target = None
                tqqq_stop = None; pyramid_on = False
                if is_tqqq:
                    last_dew = None; cdn_active = False
                # Exits 1-8 (TP/ATR/Stop/MTI/Regime): skip Phase C
                # Exits 9-12 (Flip/DEW): fall through to Phase C
                if not is_fall_through:
                    continue
            else:
                dret[i] = raw_d
                continue   # still in trade, skip Phase C

        # Phase C: only reached after fall-through exits or no trade
        if not in_trade:
            if cup_f and not cup_entered and not fell_through:
                in_trade = True; instrument = "TQQQ"; cdn_active = False
                entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
                cup_entered = True
                tqqq_target = a["topen"][i+1] * 1.50
                tqqq_stop   = a["topen"][i+1] * 0.94
                continue
            if cdn_f and not fell_through:
                in_trade = True; instrument = "TQQQ"; cdn_active = True
                entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
                tqqq_target = a["topen"][i+1] * 1.50
                tqqq_stop   = a["topen"][i+1] * 0.94
                continue
            if a["DEW"][i] == "Buy" and last_dew != "Buy" and flip_blocked != "QQQ":
                rt  = a["RT"][i]; bsr = a["BSR"][i]; mti = a["MTI"][i]
                blk = ((not np.isnan(rt)  and 0.85 <= rt  < 0.90) or
                       (not np.isnan(bsr) and bsr > 1.50) or
                       (not np.isnan(mti) and mti > 1.10))
                if not blk:
                    in_trade  = True; instrument = "QQQ"
                    entry_idx = i + 1; cum_mult = 1.0; last_dew = "Buy"
                    atr_mult  = 2.0 if a["cc"][i] == "C/Dn" else 1.0
                    stop_qqq  = a["Open"][i+1] - atr_mult * a["ATR14"][i]
                continue   # blocked QQQ: leave state unchanged and skip PSQ on same bar
            if (a["DEW"][i] == "Sell" and last_dew != "Sell"
                    and a["cc"][i] == "C/Dn" and flip_blocked != "PSQ"):
                rt  = a["RT"][i]; mti = a["MTI"][i]
                blk = (not np.isnan(rt)  and rt  > 0.95 and
                       not np.isnan(mti) and mti > 0.95)
                if not blk:
                    in_trade  = True; instrument = "PSQ"
                    entry_idx = i + 1; cum_mult = 1.0; last_dew = "Sell"
                    stop_qqq  = a["Open"][i+1] + 2.0 * a["ATR14"][i]

    return trades, dret


def run_4b_filtered(df):
    return _run_engine(df, use_hf_pyr=False)

def run_5_final(df):
    return _run_engine(df, use_hf_pyr=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def trade_counts(trades):
    qqq_n  = sum(1 for t in trades if t["inst"] == "QQQ")
    psq_n  = sum(1 for t in trades if t["inst"] == "PSQ")
    tqqq_n = sum(1 for t in trades if t["inst"] == "TQQQ")
    return qqq_n, psq_n, tqqq_n

def main():
    # ── STEP 1 ────────────────────────────────────────────────────────────────
    print("\n=== Step 1: Loading data ===")
    df, df_full, qqq_raw = load_data()

    print(f"  df.shape[0]                 → {len(df)}")
    print(f"  df.loc[13,'ATR14']          → {df.loc[13,'ATR14']:.4f}")
    vix_row = df[df["Date"] == "2020-03-16"]
    print(f"  VIX on 2020-03-16           → {vix_row['VIX'].values[0]:.2f}")
    print(f"  tqqq_atr first non-NaN      → {df['tqqq_atr'].dropna().iloc[0]:.4f}")
    print(f"  cc.head(3)                  → {df['cc'].head(3).tolist()}")
    r1 = df[df["Date"] == "2005-07-06"]
    print(f"  2005-07-06 RT={r1['RT'].values[0]:.2f} MTI={r1['MTI'].values[0]:.2f} BSR={r1['BSR'].values[0]:.2f}")
    r2 = df[df["Date"] == "2015-09-24"]
    print(f"  2015-09-24 RT={r2['RT'].values[0]:.2f} MTI={r2['MTI'].values[0]:.2f} BSR={r2['BSR'].values[0]:.2f}")

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    print("\n=== Step 2: DEW Oscillator ===")
    dew_df = compute_dew(qqq_raw)
    df      = df.merge(dew_df, on="Date", how="left")
    df_full = df_full.merge(dew_df, on="Date", how="left")
    df["DEW_Signal"]      = df["DEW_Signal"].fillna("").astype(str)
    df_full["DEW_Signal"] = df_full["DEW_Signal"].fillna("").astype(str)

    print(f"  DEW Buy signals             → {(df['DEW_Signal']=='Buy').sum()}")
    print(f"  DEW Sell signals            → {(df['DEW_Signal']=='Sell').sum()}")
    print(f"  DEW empty                   → {(df['DEW_Signal']=='').sum()}")
    first3 = df[df["DEW_Signal"].isin(["Buy","Sell"])].head(3)
    for _, r in first3.iterrows():
        print(f"    {str(r['Date'])[:10]} {r['DEW_Signal']} (cc={r['cc']})")

    # ── STEP 3a ───────────────────────────────────────────────────────────────
    print("\n=== Step 3a: Isolated QQQ ===")
    t3a, d3a = run_3a_qqq(df)
    q, p, tq = trade_counts(t3a)
    checkpoint("3a QQQ", len(t3a), q, p, tq, d3a, df["Date"].values)
    # First 5 trades
    eq = STARTING_K
    for idx, t in enumerate(t3a[:5]):
        eq *= (1.0 + t["ret"])
        print(f"    {idx+1} {str(t['entry_date'])[:10]} → {str(t['exit_date'])[:10]}  {t['ret']*100:+.4f}% {t['exit']}  ${eq:,.2f}")

    # ── STEP 3b ───────────────────────────────────────────────────────────────
    print("\n=== Step 3b: Isolated PSQ ===")
    t3b, d3b = run_3b_psq(df)
    q, p, tq = trade_counts(t3b)
    checkpoint("3b PSQ", len(t3b), q, p, tq, d3b, df["Date"].values)
    eq = STARTING_K
    for idx, t in enumerate(t3b[:5]):
        eq *= (1.0 + t["ret"])
        print(f"    {idx+1} {str(t['entry_date'])[:10]} → {str(t['exit_date'])[:10]}  {t['ret']*100:+.4f}% {t['exit']}  ${eq:,.2f}")

    # ── STEP 3c ───────────────────────────────────────────────────────────────
    print("\n=== Step 3c: Isolated C/Up TQQQ ===")
    t3c, d3c = run_3c_cup_tqqq(df)
    q, p, tq = trade_counts(t3c)
    checkpoint("3c C/Up TQQQ", len(t3c), q, p, tq, d3c, df["Date"].values)
    eq = STARTING_K
    for idx, t in enumerate(t3c[:5]):
        eq *= (1.0 + t["ret"])
        print(f"    {idx+1} {str(t['entry_date'])[:10]}  {t['ret']*100:+.4f}% {t['exit']}  ${eq:,.2f}")

    # ── STEP 3d ───────────────────────────────────────────────────────────────
    print("\n=== Step 3d: Isolated C/Dn TQQQ ===")
    t3d, d3d = run_3d_cdn_tqqq(df)
    q, p, tq = trade_counts(t3d)
    checkpoint("3d C/Dn TQQQ", len(t3d), q, p, tq, d3d, df["Date"].values)
    eq = STARTING_K
    for idx, t in enumerate(t3d[:5]):
        eq *= (1.0 + t["ret"])
        print(f"    {idx+1} {str(t['entry_date'])[:10]}  {t['ret']*100:+.4f}% {t['exit']}  ${eq:,.2f}")

    # ── STEP 4a ───────────────────────────────────────────────────────────────
    print("\n=== Step 4a: Combined (no filters) ===")
    t4a, d4a = run_4a_combined(df)
    q, p, tq = trade_counts(t4a)
    checkpoint("4a Combined", len(t4a), q, p, tq, d4a, df["Date"].values)
    print("  First 10 trades:")
    for idx, t in enumerate(t4a[:10]):
        print(f"    {idx+1} {str(t['entry_date'])[:10]} {t['inst']:4s} → {str(t['exit_date'])[:10]}  {t['ret']*100:+.4f}% {t['exit']}")

    # ── STEP 4b ───────────────────────────────────────────────────────────────
    print("\n=== Step 4b: Combined + filters + preemption ===")
    t4b, d4b = run_4b_filtered(df)
    q, p, tq = trade_counts(t4b)
    checkpoint("4b Filtered", len(t4b), q, p, tq, d4b, df["Date"].values)
    print("  First 10 trades:")
    for idx, t in enumerate(t4b[:10]):
        print(f"    {idx+1} {str(t['entry_date'])[:10]} {t['inst']:4s} → {str(t['exit_date'])[:10]}  {t['ret']*100:+.4f}% {t['exit']}")

    # TQQQ isolation check
    tqqq_trades_4b = [t for t in t4b if t["inst"] == "TQQQ"]
    eq_tqqq = STARTING_K
    for t in tqqq_trades_4b:
        eq_tqqq *= (1.0 + t["ret"])
    print(f"  TQQQ isolation (compound from $100K): ${eq_tqqq:,.2f}")

    # ── STEP 5 ────────────────────────────────────────────────────────────────
    print("\n=== Step 5: Health Factor + Pyramid ===")
    t5, d5 = run_5_final(df)
    q, p, tq = trade_counts(t5)
    checkpoint("STEP 5 — FINAL", len(t5), q, p, tq, d5, df["Date"].values)

    # ── STEP 6 ────────────────────────────────────────────────────────────────
    print("\n=== Step 6: Full Data Run ===")
    t6, d6 = run_5_final(df_full)
    q, p, tq = trade_counts(t6)
    checkpoint("STEP 6 — FULL RUN", len(t6), q, p, tq, d6, df_full["Date"].values)

    # Last completed trade
    if t6:
        last = t6[-1]
        print(f"  Last trade: {str(last['entry_date'])[:10]} {last['inst']} → {str(last['exit_date'])[:10]}  {last['ret']*100:+.4f}% {last['exit']}")

    # Last bar signal state
    last_row = df_full.iloc[-1]
    print(f"  Last bar ({str(last_row['Date'])[:10]}): cc={last_row['cc']}  DEW={last_row['DEW_Signal']}  "
          f"RT={last_row['RT']:.2f}  MTI={last_row['MTI']:.2f}  BSR={last_row['BSR']:.2f}  VIX={last_row['VIX']:.2f}")


if __name__ == "__main__":
    main()
