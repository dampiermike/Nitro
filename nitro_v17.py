"""
Dampier Nitro++ v17 - Production engine
=======================================
v17-tuned = v16 with two deliberate changes, both validated by backtest:

  1. De-margined pyramid (Flag-1 fix).  PMP ceiling lowered so the strategy
     never deploys more than 100% of capital:
         VIX<20: 1.30->1.00 , VIX<32: 1.10->0.85 , VIX>=32: 0.50 (unchanged)

  2. New "Cap" setup (Setup 5).  When flat, a 5-day QQQ drop <= -10% with a
     >=5-bar cooldown buys QQQ (1x) at the next open, exiting on the standard
     QQQ machinery (5% TP, entry_open - 1x ATR14 stop, DEW-Sell / C/Dn-flip).
     75.9% win rate over 2000-2026; adds ~+4.5% CAGR.

Deliberately NOT included (backtest showed no benefit):
  * Mean-Rev (Setup 6)      - +0.03% CAGR, -0.056 Sharpe (pure churn). Dropped.
  * C/Dn-TQQQ VIX loosening  - -0.42% CAGR. Reverted to v16's VIX < 27.

Backtest (2000-01-03..2026-04-01, mode 7, no-margin):
  CAGR ~50.7% / Sharpe ~1.81 / MDD -27.19%   vs   v16 no-margin 46.19% / 1.827 / -21.54%
  (the +4.5% CAGR is bought with ~5.7pp deeper drawdown - an intentional trade.)

Data loading and DEW are reused unchanged from nitro_v16.
"""
import os
import sys
import numpy as np
import pandas as pd

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
from nitro_v16 import (
    load_data, compute_dew, trade_stats, print_stats, fmt_eq, START_DATE,
)

# ── Cap setup parameters ─────────────────────────────────────────────────────
CAP_DROP5_THR = -0.10   # 5-day QQQ close-to-close drop that arms Cap
CAP_COOLDOWN  = 5       # bars since last exit before Cap may fire
CAP_ATR_MULT  = 1.0     # Cap stop distance: entry_open - k * ATR14
CAP_TP        = 0.05    # Cap take-profit (same as the QQQ long leg)


def get_pmp(vix):
    """De-margined pyramid multiplier (ceiling 1.00 = never > 100% capital)."""
    v = vix if not np.isnan(vix) else 15.0
    if v < 20:
        return 1.00
    if v < 32:
        return 0.85
    return 0.50


def add_cap_feature(df):
    """Add the 5-day QQQ drop column used by the Cap setup."""
    df = df.copy()
    df["drop5"] = df["Close"] / df["Close"].shift(5) - 1.0
    return df


def run_engine(df, mode):
    """
    v16 combined engine + de-margined get_pmp + Cap setup (Setup 5).
    Modes: '4a' none | '4b' filters+preempt+VIX | '5' +pyramid | '7' +decay (prod)
    """
    use_filters = mode in ("4b", "5", "7")
    use_pyramid = mode in ("5", "7")
    use_decay   = mode in ("7",)

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
    drop5 = df["drop5"].values

    o2c_q = df["o2c_QQQ"].values;  c2c_q = df["c2c_QQQ"].values;  pc2o_q = df["pc2o_QQQ"].values
    o2c_i = df["o2c_INV"].values;  c2c_i = df["c2c_INV"].values;  pc2o_i = df["pc2o_INV"].values
    o2c_t = df["tqqq_o2c"].values; c2c_t = df["tqqq_c2c"].values; pc2o_t = df["tqqq_pc2o"].values

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
    cur_entry_kind = None
    decay_transitions = 0
    last_exit_idx = None  # Cap cooldown anchor

    def select_returns(instr):
        if instr == "QQQ":  return o2c_q, c2c_q, pc2o_q
        if instr == "INV":  return o2c_i, c2c_i, pc2o_i
        if instr == "TQQQ": return o2c_t, c2c_t, pc2o_t
        raise ValueError("instrument not set")

    def qqq_inv_block_filter(kind, i):
        if not use_filters:
            return False
        if kind == "QQQ":
            cond = False
            if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90:  cond = True
            if not np.isnan(BSR[i]) and BSR[i] > 1.50:        cond = True
            if not np.isnan(MTI[i]) and MTI[i] > 1.15:        cond = True
            return cond
        if kind == "INV":
            if (not np.isnan(RT[i]) and RT[i] > 0.95) and (not np.isnan(MTI[i]) and MTI[i] > 0.95):
                return True
            return False
        return False

    def vix_too_high(i):
        if not use_filters:
            return False
        return (not np.isnan(VIX[i])) and VIX[i] > 35

    def cooldown_ok(i):
        return (last_exit_idx is None) or (i - last_exit_idx >= CAP_COOLDOWN)

    eq_before_trade = 100_000.0

    for i in range(n):
        fell_through = False
        if i > 0:
            if Trend[i] == "C/Up" and cc[i-1] != "C/Up":
                last_dew = None
                cup_entered = False
            if Trend[i] == "C/Dn" and cc[i-1] != "C/Dn":
                last_dew = None

        cup_f = (cc[i] == "C/Up"
                 and not np.isnan(tqqq_rt[i]) and tqqq_rt[i] < 1.40
                 and not np.isnan(BSR[i]) and BSR[i] > 1.05
                 and not np.isnan(tqqq_atr[i]) and tqqq_atr[i] < 7.0
                 and not np.isnan(VIX[i]) and VIX[i] < 30)
        # C/Dn TQQQ: VIX < 27 (v16 original; loosening reverted per tuning)
        cdn_f = (cc[i] == "C/Dn"
                 and DEW[i] == "Buy" and last_dew != "Buy"
                 and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                 and not np.isnan(MTI[i]) and MTI[i] < 1.00
                 and not np.isnan(VIX[i]) and VIX[i] < 27)

        # Cap flag (Setup 5): evaluated when flat (Phase C)
        cap_f = (not np.isnan(drop5[i]) and drop5[i] <= CAP_DROP5_THR)

        # Priority 1a/1b preemption (4b+)
        if use_filters and in_trade and i + 1 < n:
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
                trades.append((cur_entry_date, Date[i], trade_ret, "Preempt1a", exiting, eq_before_trade, eq_after, cur_entry_kind))
                eq_before_trade = eq_after
                in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.50
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False; cup_entered = True; cdn_active = False
                pyramid_on = False
                entry_vix = VIX[i+1] if i+1 < n else np.nan
                decayed_active = False; last_dew = None
                cur_entry_date = Date[i+1]; cur_entry_kind = "cup"
                fell_through = False
                continue
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
                    trades.append((cur_entry_date, Date[i], trade_ret, "Preempt1b", "QQQ", eq_before_trade, eq_after, cur_entry_kind))
                    eq_before_trade = eq_after
                    in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
                    cum_mult = 1.0
                    tqqq_target = tqqq_open[i+1] * 1.30
                    tqqq_stop = tqqq_open[i+1] * 0.94
                    be_moved = False; cup_entered = False; cdn_active = True
                    pyramid_on = False
                    entry_vix = VIX[i+1] if i+1 < n else np.nan
                    decayed_active = False; last_dew = None
                    cur_entry_date = Date[i+1]; cur_entry_kind = "cdn"
                    fell_through = False
                    continue

        # Phase B - exit
        if in_trade:
            o2c_a, c2c_a, pc2o_a = select_returns(instrument)
            d = o2c_a[i] if i == entry_idx else c2c_a[i]
            cum_mult *= (1 + d)

            pyr_pre = pyramid_on if (use_pyramid and instrument == "TQQQ") else False
            pmp_pyr = get_pmp(VIX[i]) if pyr_pre else 1.0
            should_decay = False
            if use_decay and instrument == "TQQQ" and entry_vix is not None and not np.isnan(entry_vix):
                days_held = i - entry_idx
                if days_held >= 15 and (VIX[i] - entry_vix) >= 1.0:
                    should_decay = True
            decay_mult = 0.30 if should_decay else 1.0
            pmp = pmp_pyr * decay_mult
            if use_decay and instrument == "TQQQ":
                if should_decay != decayed_active:
                    decay_transitions += 1
                decayed_active = should_decay

            if instrument == "TQQQ" and not be_moved and cum_mult - 1 >= 0.09:
                tqqq_stop = tqqq_open[entry_idx]
                be_moved = True

            exited = False; exit_reason = None; ftype = "continue"
            if instrument == "TQQQ" and (i != entry_idx) and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exited = True; exit_reason = "MTI"
            elif instrument == "TQQQ" and tqqq_close[i] >= tqqq_target:
                exited = True; exit_reason = "TQQQ TP"
            elif instrument == "TQQQ" and tqqq_close[i] <= tqqq_stop:
                exited = True; exit_reason = "TQQQ Stop"
            elif instrument == "TQQQ" and (i - entry_idx) >= 70:
                exited = True; exit_reason = "MaxHold"
            elif instrument == "QQQ" and cum_mult - 1 >= (CAP_TP if cur_entry_kind == "cap" else 0.05):
                exited = True; exit_reason = "QQQ TP"
            elif instrument == "QQQ" and Close[i] <= stop_qqq:
                exited = True; exit_reason = "QQQ ATR"
            elif instrument == "INV" and cum_mult - 1 >= 0.05:
                exited = True; exit_reason = "INV TP"
            elif instrument == "INV" and Close[i] >= stop_qqq:
                exited = True; exit_reason = "INV ATR"
            elif instrument == "TQQQ":
                if cur_entry_kind == "cup" and cc[i] == "C/Dn":
                    exited = True; exit_reason = "TQQQ Regime"
                elif cur_entry_kind == "cdn" and (DEW[i] == "Sell" or (i != entry_idx and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn")):
                    exited = True; exit_reason = "TQQQ Regime"
            if not exited and instrument == "QQQ" and (i != entry_idx) and i > 0 and Trend[i] == "C/Dn" and cc[i-1] == "C/Up":
                exited = True; exit_reason = "QQQ Flip"; ftype = "fall_through"
            elif not exited and instrument == "INV" and (i != entry_idx) and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn":
                exited = True; exit_reason = "INV Flip"; ftype = "fall_through"
            elif not exited and instrument == "QQQ" and DEW[i] == "Sell":
                exited = True; exit_reason = "QQQ DEW"; ftype = "fall_through"
            elif not exited and instrument == "INV" and DEW[i] == "Buy":
                exited = True; exit_reason = "INV DEW"; ftype = "fall_through"

            if exited:
                if i + 1 < n:
                    pc = pc2o_a[i+1]
                    if instrument == "TQQQ":
                        dret[i] = (1 + d * pmp) * (1 + pc * pmp) - 1
                    else:
                        dret[i] = (1 + d) * (1 + pc) - 1
                    trade_ret = cum_mult * (1 + pc) - 1
                else:
                    if instrument == "TQQQ":
                        dret[i] = d * pmp
                    else:
                        dret[i] = d
                    trade_ret = cum_mult - 1
                trades.append((cur_entry_date, Date[i], trade_ret, exit_reason, instrument, eq_before_trade, None, cur_entry_kind))
                prev_instr = instrument
                in_trade = False; instrument = None; entry_idx = None
                cum_mult = 1.0; stop_qqq = None
                tqqq_target = None; tqqq_stop = None; be_moved = False
                last_exit_idx = i  # start Cap cooldown clock
                if prev_instr == "TQQQ":
                    last_dew = None; cdn_active = False
                    entry_vix = None; decayed_active = False; pyramid_on = False
                if exit_reason == "QQQ DEW":
                    last_dew = "Sell"
                elif exit_reason == "INV DEW":
                    last_dew = "Buy"
                if ftype == "fall_through":
                    fell_through = True
                else:
                    continue
            else:
                if instrument == "TQQQ":
                    dret[i] = d * pmp
                else:
                    dret[i] = d
                if use_pyramid and instrument == "TQQQ" and cum_mult - 1 >= 0.10 and not pyramid_on:
                    pyramid_on = True
                continue

        # Phase C - entries (priority: cup > cdn > QQQ Buy > INV Sell > Cap)
        if not in_trade and i + 1 < n:
            if cup_f and not cup_entered and not fell_through:
                in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.50
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False; cup_entered = True; cdn_active = False
                pyramid_on = False; entry_vix = VIX[i+1]; decayed_active = False
                cur_entry_date = Date[i+1]; cur_entry_kind = "cup"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue
            if cdn_f and last_dew != "Buy" and not fell_through:
                in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.30
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False; cup_entered = False; cdn_active = True
                pyramid_on = False; entry_vix = VIX[i+1]; decayed_active = False
                last_dew = "Buy"
                cur_entry_date = Date[i+1]; cur_entry_kind = "cdn"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue
            if DEW[i] == "Buy" and last_dew != "Buy" and not fell_through:
                if not qqq_inv_block_filter("QQQ", i) and not vix_too_high(i):
                    in_trade = True; instrument = "QQQ"; entry_idx = i + 1
                    cum_mult = 1.0
                    atr_mult = 2.0 if cc[i] == "C/Dn" else 1.0
                    stop_qqq = Open_[i+1] - atr_mult * ATR14[i]
                    last_dew = "Buy"
                    cur_entry_date = Date[i+1]; cur_entry_kind = "qqq"
                    eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                    continue
            if DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn" and not fell_through:
                if not qqq_inv_block_filter("INV", i) and not vix_too_high(i):
                    in_trade = True; instrument = "INV"; entry_idx = i + 1
                    cum_mult = 1.0
                    stop_qqq = Open_[i+1] + 2.0 * ATR14[i]
                    last_dew = "Sell"
                    cur_entry_date = Date[i+1]; cur_entry_kind = "inv"
                    eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                    continue
            # Setup 5: Cap (capitulation buy) - LONG QQQ, lowest priority
            if cap_f and cooldown_ok(i) and not fell_through and not np.isnan(ATR14[i]):
                in_trade = True; instrument = "QQQ"; entry_idx = i + 1
                cum_mult = 1.0
                stop_qqq = Open_[i+1] - CAP_ATR_MULT * ATR14[i]
                cur_entry_date = Date[i+1]; cur_entry_kind = "cap"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue

    eq = 100_000.0 * np.cumprod(1 + dret)
    n_qqq = sum(1 for t in trades if t[4] == "QQQ")
    n_inv = sum(1 for t in trades if t[4] == "INV")
    n_tqqq = sum(1 for t in trades if t[4] == "TQQQ")
    n_cap = sum(1 for t in trades if len(t) > 7 and t[7] == "cap")
    return dict(trades=trades, eq=eq, dret=dret,
                n_qqq=n_qqq, n_inv=n_inv, n_tqqq=n_tqqq, n_cap=n_cap,
                decay_transitions=decay_transitions)


def load_v17():
    """Load data and attach the Cap + DEW features. Returns (df, df_full, qqq_raw)."""
    df, df_full, qqq_raw = load_data()
    dew = compute_dew(qqq_raw)
    df = df.merge(dew, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    df = add_cap_feature(df)
    return df, df_full, qqq_raw


def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_v17()
    print("=" * 90)
    print(f"  NITRO++ v17 (production)   {str(df['Date'].iloc[0])[:10]} -> {str(df['Date'].iloc[-1])[:10]}   {len(df)} bars")
    print(f"  De-margined pyramid (PMP<=1.00) + Cap setup; Mean-Rev omitted; C/Dn-TQQQ VIX<27")
    print("=" * 90)
    for mode in ["4a", "4b", "5", "7"]:
        r = run_engine(df, mode=mode)
        s = trade_stats(r["trades"], r["eq"], r["dret"], dates=df["Date"])
        tag = f"{mode} (Q={r['n_qqq']} I={r['n_inv']} T={r['n_tqqq']} Cap={r['n_cap']})"
        print_stats(tag, s)
    print("=" * 90)


if __name__ == "__main__":
    main()
