"""
Dampier Nitro++ v17 - Backtest / A-B vs v16
============================================
v17 = v16 plus the pasted rule set:
  * Setup 2 (C/Dn TQQQ): VIX gate loosened from  VIX < 27  ->  VIX <= 35
  * Setup 5 (Cap):     flat + 5-day QQQ drop <= -10% + cooldown>=5  -> LONG QQQ
  * Setup 6 (Mean-Rev): flat + stretch < -2.0 + vol20 < 0.30 + cooldown>=5 -> LONG QQQ

Cap / Mean-Rev take the QQQ instrument and reuse the QQQ exit machinery
(5% TP, ATR14 stop, DEW-Sell / C/Dn-flip reversal), plus Cap-specific tuning
knobs (own ATR-stop distance, optional max-hold, VIX gate, green-candle
confirmation, own take-profit).

The v16 engine is imported unmodified so the comparison is apples-to-apples.
"""
import os
import sys
import numpy as np
import pandas as pd

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
from nitro_v16 import (
    load_data, compute_dew, get_pmp, trade_stats, print_stats, fmt_eq,
    run_engine as run_engine_v16, START_DATE,
)

# ── Base tunables ────────────────────────────────────────────────────────────
DROP5_THR      = -0.10
STRETCH_THR    = -2.0
VOL20_THR      = 0.30
COOLDOWN       = 5
CAPMR_ATR_MULT = 1.0       # retained for back-compat; Cap/MR now use their own
CDN_VIX_MAX    = 35.0

# ── Cap / Mean-Rev tuning knobs (a driver may override before calling) ───────
CAP_ON       = True
MR_ON        = True
CAP_ATR_MULT = 1.0         # Cap ATR stop distance (entry_open - k*ATR14)
MR_ATR_MULT  = 1.0         # Mean-Rev ATR stop distance
CAP_MAXHOLD  = 0           # 0 = off; else force-exit Cap after N bars
CAP_VIX_MAX  = 0.0         # 0 = off; else only enter Cap when VIX <= this
CAP_CONFIRM  = False       # require prior-bar QQQ up (green candle) before Cap
CAP_TP       = 0.05        # Cap take-profit (QQQ default 0.05)


def add_v17_features(df):
    """Add MA20, vol20, drop5, stretch columns used by Cap / Mean-Rev."""
    df = df.copy()
    df["MA20"]    = df["Close"].rolling(20).mean()
    df["vol20"]   = df["c2c_QQQ"].rolling(20).std() * np.sqrt(252.0)
    df["drop5"]   = df["Close"] / df["Close"].shift(5) - 1.0
    df["stretch"] = (df["Close"] - df["MA20"]) / df["ATR14"]
    return df


def run_engine_v17(df, mode):
    """
    Faithful copy of nitro_v16.run_engine with additions:
      (1) cdn_f VIX ceiling raised to CDN_VIX_MAX (was 27)
      (2) Cap entry     (LONG QQQ) at lowest priority, with tuning knobs
      (3) Mean-Rev entry (LONG QQQ) at lowest priority
    Cooldown gates Cap/MR: >= COOLDOWN bars since the last exit.
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

    # v17 features
    drop5   = df["drop5"].values
    stretch = df["stretch"].values
    vol20   = df["vol20"].values

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
    last_exit_idx = None  # v17: cooldown anchor

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
        return (last_exit_idx is None) or (i - last_exit_idx >= COOLDOWN)

    eq_before_trade = 100_000.0
    cur_eq = 100_000.0

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
        # v17 change: VIX ceiling raised from 27 to CDN_VIX_MAX
        cdn_f = (cc[i] == "C/Dn"
                 and DEW[i] == "Buy" and last_dew != "Buy"
                 and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                 and not np.isnan(MTI[i]) and MTI[i] < 1.00
                 and not np.isnan(VIX[i]) and VIX[i] <= CDN_VIX_MAX)

        # v17 entry flags (Cap / Mean-Rev) - evaluated when flat (Phase C)
        cap_extra = True
        if CAP_VIX_MAX and not (not np.isnan(VIX[i]) and VIX[i] <= CAP_VIX_MAX):
            cap_extra = False
        if CAP_CONFIRM and not (i > 0 and not np.isnan(c2c_q[i]) and c2c_q[i] > 0):
            cap_extra = False
        cap_f = (CAP_ON and not np.isnan(drop5[i]) and drop5[i] <= DROP5_THR and cap_extra)
        mr_f  = (MR_ON and not np.isnan(stretch[i]) and stretch[i] < STRETCH_THR
                 and not np.isnan(vol20[i]) and vol20[i] < VOL20_THR)

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
                cur_eq = eq_after
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
                    cur_eq = eq_after
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
            elif instrument == "QQQ" and cur_entry_kind == "cap" and CAP_MAXHOLD and (i - entry_idx) >= CAP_MAXHOLD:
                exited = True; exit_reason = "Cap MaxHold"
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
                last_exit_idx = i  # v17: start cooldown clock
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

        # Phase C - entries
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
            # v17 Setup 5: Cap (capitulation buy) - LONG QQQ, lowest priority
            if cap_f and cooldown_ok(i) and not fell_through and not np.isnan(ATR14[i]):
                in_trade = True; instrument = "QQQ"; entry_idx = i + 1
                cum_mult = 1.0
                stop_qqq = Open_[i+1] - CAP_ATR_MULT * ATR14[i]
                cur_entry_date = Date[i+1]; cur_entry_kind = "cap"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue
            # v17 Setup 6: Mean-Rev - LONG QQQ, lowest priority
            if mr_f and cooldown_ok(i) and not fell_through and not np.isnan(ATR14[i]):
                in_trade = True; instrument = "QQQ"; entry_idx = i + 1
                cum_mult = 1.0
                stop_qqq = Open_[i+1] - MR_ATR_MULT * ATR14[i]
                cur_entry_date = Date[i+1]; cur_entry_kind = "mr"
                eq_before_trade = 100_000.0 * float(np.prod(1 + dret[:i+1])) if i > 0 else 100_000.0
                continue

    eq = 100_000.0 * np.cumprod(1 + dret)

    def cnt(kind):
        return sum(1 for t in trades if len(t) > 7 and t[7] == kind)
    n_qqq = sum(1 for t in trades if t[4] == "QQQ")
    n_inv = sum(1 for t in trades if t[4] == "INV")
    n_tqqq = sum(1 for t in trades if t[4] == "TQQQ")
    return dict(
        trades=trades, eq=eq, dret=dret,
        n_qqq=n_qqq, n_inv=n_inv, n_tqqq=n_tqqq,
        n_cap=cnt("cap"), n_mr=cnt("mr"),
        decay_transitions=decay_transitions,
    )


def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_data()
    dew = compute_dew(qqq_raw)
    df = df.merge(dew, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    df = add_v17_features(df)

    modes = ["4a", "4b", "5", "7"]
    labels = {
        "4a": "4a  no filters",
        "4b": "4b  filters+preempt+VIX",
        "5":  "5   pyramid",
        "7":  "7   decay (PRODUCTION)",
    }

    print("\n" + "=" * 100)
    print(f"  NITRO++  v16  vs  v17   A/B BACKTEST     ({str(df['Date'].iloc[0])[:10]} -> {str(df['Date'].iloc[-1])[:10]}, {len(df)} bars)")
    print("=" * 100)
    print("  v17 = v16 + C/Dn-TQQQ VIX<=35 (was 27) + Cap(QQQ) + Mean-Rev(QQQ)")
    print("-" * 100)

    for mode in modes:
        r16 = run_engine_v16(df, mode=mode)
        r17 = run_engine_v17(df, mode=mode)
        s16 = trade_stats(r16["trades"], r16["eq"], r16["dret"])
        s17 = trade_stats(r17["trades"], r17["eq"], r17["dret"])

        print(f"\n  MODE {labels[mode]}")
        print(f"  {'':10s} {'n':>5s} {'final':>20s} {'CAGR':>9s} {'Sharpe':>8s} {'MDD':>9s}   counts")
        print(f"  {'v16':10s} {s16['n']:>5d} {fmt_eq(s16['final']):>20s} {s16['cagr']*100:>8.2f}% {s16['sharpe']:>8.3f} {s16['mdd']*100:>8.2f}%   "
              f"Q={r16['n_qqq']} I={r16['n_inv']} T={r16['n_tqqq']}")
        print(f"  {'v17':10s} {s17['n']:>5d} {fmt_eq(s17['final']):>20s} {s17['cagr']*100:>8.2f}% {s17['sharpe']:>8.3f} {s17['mdd']*100:>8.2f}%   "
              f"Q={r17['n_qqq']} I={r17['n_inv']} T={r17['n_tqqq']} | Cap={r17['n_cap']} MR={r17['n_mr']}")
        d_cagr = (s17['cagr'] - s16['cagr']) * 100
        d_mdd  = (s17['mdd'] - s16['mdd']) * 100
        d_shp  = s17['sharpe'] - s16['sharpe']
        fac = s17['final'] / s16['final'] if s16['final'] else float('nan')
        print(f"  {'DELTA':10s} {'':5s} {('x%.3f'%fac):>20s} {d_cagr:>+8.2f}% {d_shp:>+8.3f} {d_mdd:>+8.2f}%")

    print("\n" + "=" * 100)
    print("  Note: Cap/Mean-Rev take QQQ; Cap uses its own tuning knobs (see top of file).")
    print("=" * 100)


if __name__ == "__main__":
    main()
