"""
Dampier Nitro++ v16 - Daily Signal Generator
Replays the full production engine (Steps 1-7: pyramid VIX schedule + decay)
and reports tomorrow's pending action based on today's close.

Live mapping:
  TQQQ: long TQQQ at full equity (×pmp_pyr × decay_mult)
  QQQ : long QQQ at full equity
  INV : short QQQ via SQQQ at 1/3 position (post 2010-02-11)
"""
import os
import sys
import smtplib
import subprocess
import numpy as np
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
from nitro_v16 import load_data, compute_dew, get_pmp

START_DATE = pd.Timestamp("2000-01-01")

# ── Email / SMS config ─────────────────────────────────────────────────────────
GMAIL_USER  = os.environ.get("GOOGLE_EMAIL", "dampiermike@gmail.com")
GMAIL_PASS  = os.environ.get("GOOGLE_APP_PASSWORD", "")
TO_EMAIL    = ["dampiermike@gmail.com", "ddampier777@gmail.com", "brooke.hoover@yahoo.com"]
SMS_NUMBERS = ["+12256144680", "+13038818222", "+18137815601"]
# Numbers that must be sent via SMS (Continuity relay through paired iPhone)
# rather than iMessage — e.g. Android/Verizon recipients where iMessage bounces.
SMS_FORCE   = {"+18137815601"}


# ----------------------------------------------------------------------
# Production engine - mirrors run_engine(mode='7') but stops at N-1 and
# returns the engine state for last-bar evaluation.
# ----------------------------------------------------------------------
def run_signal_engine(df):
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
    cur_entry_kind = None

    dret = np.zeros(n)
    trades = []
    cur_entry_date = None

    def select_returns(instr):
        if instr == "QQQ": return o2c_q, c2c_q, pc2o_q
        if instr == "INV": return o2c_i, c2c_i, pc2o_i
        if instr == "TQQQ": return o2c_t, c2c_t, pc2o_t
        raise ValueError("instrument not set")

    def vix_too_high(i):
        return (not np.isnan(VIX[i])) and VIX[i] > 35

    def qqq_block(i):
        if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90: return True
        if not np.isnan(BSR[i]) and BSR[i] > 1.50: return True
        if not np.isnan(MTI[i]) and MTI[i] > 1.15: return True
        return False

    def inv_block(i):
        return ((not np.isnan(RT[i]) and RT[i] > 0.95)
                and (not np.isnan(MTI[i]) and MTI[i] > 0.95))

    # Loop through N-1 (leave last bar for evaluate_last_bar)
    for i in range(n - 1):
        fell_through = False
        if i > 0:
            if Trend[i] == "C/Up" and cc[i-1] != "C/Up":
                last_dew = None; cup_entered = False
            if Trend[i] == "C/Dn" and cc[i-1] != "C/Dn":
                last_dew = None

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

        # Priority 1a: C/Up TQQQ preempts QQQ/INV
        if in_trade and instrument in ("QQQ", "INV") and cup_f and not cup_entered:
            o2c_e, c2c_e, pc2o_e = select_returns(instrument)
            d = o2c_e[i] if i == entry_idx else c2c_e[i]
            cum_mult *= (1 + d)
            pc = pc2o_e[i+1]
            trade_ret = cum_mult * (1 + pc) - 1
            dret[i] = (1 + d) * (1 + pc) - 1
            trades.append(dict(entry_date=cur_entry_date, exit_date=Date[i], inst=instrument,
                               ret=trade_ret, exit="Preempt1a"))
            in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
            cum_mult = 1.0
            tqqq_target = tqqq_open[i+1] * 1.50
            tqqq_stop = tqqq_open[i+1] * 0.94
            be_moved = False; cup_entered = True; cdn_active = False
            pyramid_on = False
            entry_vix = VIX[i+1]; decayed_active = False
            last_dew = None
            cur_entry_date = Date[i+1]; cur_entry_kind = "cup"
            continue

        # Priority 1b: C/Dn switch while in QQQ
        if (in_trade and instrument == "QQQ" and not cdn_active
                and cc[i] == "C/Dn"
                and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                and not np.isnan(MTI[i]) and MTI[i] < 0.95):
            o2c_e, c2c_e, pc2o_e = select_returns("QQQ")
            d = o2c_e[i] if i == entry_idx else c2c_e[i]
            cum_mult *= (1 + d)
            pc = pc2o_e[i+1]
            trade_ret = cum_mult * (1 + pc) - 1
            dret[i] = (1 + d) * (1 + pc) - 1
            trades.append(dict(entry_date=cur_entry_date, exit_date=Date[i], inst="QQQ",
                               ret=trade_ret, exit="Preempt1b"))
            in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
            cum_mult = 1.0
            tqqq_target = tqqq_open[i+1] * 1.30
            tqqq_stop = tqqq_open[i+1] * 0.94
            be_moved = False; cup_entered = False; cdn_active = True
            pyramid_on = False
            entry_vix = VIX[i+1]; decayed_active = False
            last_dew = None
            cur_entry_date = Date[i+1]; cur_entry_kind = "cdn"
            continue

        # Phase B - exits
        if in_trade:
            o2c_a, c2c_a, pc2o_a = select_returns(instrument)
            d = o2c_a[i] if i == entry_idx else c2c_a[i]
            cum_mult *= (1 + d)

            pyr_pre = pyramid_on if instrument == "TQQQ" else False
            pmp_pyr = get_pmp(VIX[i]) if pyr_pre else 1.0
            should_decay = False
            if instrument == "TQQQ" and entry_vix is not None and not np.isnan(entry_vix):
                if (i - entry_idx) >= 15 and (VIX[i] - entry_vix) >= 1.0:
                    should_decay = True
            decay_mult = 0.30 if should_decay else 1.0
            pmp = pmp_pyr * decay_mult
            if instrument == "TQQQ":
                decayed_active = should_decay

            if instrument == "TQQQ" and not be_moved and cum_mult - 1 >= 0.09:
                tqqq_stop = tqqq_open[entry_idx]
                be_moved = True

            exited = False; exit_reason = None; ftype = "continue"
            if instrument == "TQQQ" and (i != entry_idx) and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exited = True; exit_reason = "MTI"
            elif instrument == "TQQQ" and tqqq_close[i] >= tqqq_target:
                exited = True; exit_reason = "TP"
            elif instrument == "TQQQ" and tqqq_close[i] <= tqqq_stop:
                exited = True; exit_reason = "Stop"
            elif instrument == "TQQQ" and (i - entry_idx) >= 70:
                exited = True; exit_reason = "MaxHold"
            elif instrument == "QQQ" and cum_mult - 1 >= 0.05:
                exited = True; exit_reason = "TP"
            elif instrument == "QQQ" and Close[i] <= stop_qqq:
                exited = True; exit_reason = "ATR"
            elif instrument == "INV" and cum_mult - 1 >= 0.05:
                exited = True; exit_reason = "TP"
            elif instrument == "INV" and Close[i] >= stop_qqq:
                exited = True; exit_reason = "ATR"
            elif instrument == "TQQQ":
                if cur_entry_kind == "cup" and cc[i] == "C/Dn":
                    exited = True; exit_reason = "Regime"
                elif cur_entry_kind == "cdn" and (DEW[i] == "Sell"
                        or (i != entry_idx and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn")):
                    exited = True; exit_reason = "Regime"

            if not exited and instrument == "QQQ" and (i != entry_idx) and i > 0 and Trend[i] == "C/Dn" and cc[i-1] == "C/Up":
                exited = True; exit_reason = "Flip"; ftype = "fall_through"
            elif not exited and instrument == "INV" and (i != entry_idx) and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn":
                exited = True; exit_reason = "Flip"; ftype = "fall_through"
            elif not exited and instrument == "QQQ" and DEW[i] == "Sell":
                exited = True; exit_reason = "DEW"; ftype = "fall_through"
            elif not exited and instrument == "INV" and DEW[i] == "Buy":
                exited = True; exit_reason = "DEW"; ftype = "fall_through"

            if exited:
                pc = pc2o_a[i+1]
                if instrument == "TQQQ":
                    dret[i] = (1 + d * pmp) * (1 + pc * pmp) - 1
                else:
                    dret[i] = (1 + d) * (1 + pc) - 1
                trade_ret = cum_mult * (1 + pc) - 1
                trades.append(dict(entry_date=cur_entry_date, exit_date=Date[i],
                                   inst=instrument, ret=trade_ret, exit=exit_reason))
                prev_instr = instrument
                in_trade = False; instrument = None; entry_idx = None
                cum_mult = 1.0; stop_qqq = None
                tqqq_target = None; tqqq_stop = None; be_moved = False
                if prev_instr == "TQQQ":
                    last_dew = None; cdn_active = False
                    entry_vix = None; decayed_active = False; pyramid_on = False
                if exit_reason == "DEW" and prev_instr == "QQQ":
                    last_dew = "Sell"
                elif exit_reason == "DEW" and prev_instr == "INV":
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
                if instrument == "TQQQ" and cum_mult - 1 >= 0.10 and not pyramid_on:
                    pyramid_on = True
                continue

        # Phase C - entries
        if not in_trade:
            if cup_f and not cup_entered and not fell_through:
                in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.50
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False; cup_entered = True; cdn_active = False
                pyramid_on = False
                entry_vix = VIX[i+1]; decayed_active = False
                cur_entry_date = Date[i+1]; cur_entry_kind = "cup"
                continue
            if cdn_f and last_dew != "Buy" and not fell_through:
                in_trade = True; instrument = "TQQQ"; entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = tqqq_open[i+1] * 1.30
                tqqq_stop = tqqq_open[i+1] * 0.94
                be_moved = False; cup_entered = False; cdn_active = True
                pyramid_on = False
                entry_vix = VIX[i+1]; decayed_active = False
                last_dew = "Buy"
                cur_entry_date = Date[i+1]; cur_entry_kind = "cdn"
                continue
            if DEW[i] == "Buy" and last_dew != "Buy" and not fell_through:
                if not qqq_block(i) and not vix_too_high(i):
                    in_trade = True; instrument = "QQQ"; entry_idx = i + 1
                    cum_mult = 1.0
                    atr_mult = 2.0 if cc[i] == "C/Dn" else 1.0
                    stop_qqq = Open_[i+1] - atr_mult * ATR14[i]
                    last_dew = "Buy"
                    cur_entry_date = Date[i+1]; cur_entry_kind = "qqq"
                    continue
            if DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn" and not fell_through:
                if not inv_block(i) and not vix_too_high(i):
                    in_trade = True; instrument = "INV"; entry_idx = i + 1
                    cum_mult = 1.0
                    stop_qqq = Open_[i+1] + 2.0 * ATR14[i]
                    last_dew = "Sell"
                    cur_entry_date = Date[i+1]; cur_entry_kind = "inv"
                    continue

    state = dict(
        in_trade=in_trade, instrument=instrument, entry_idx=entry_idx,
        entry_date=cur_entry_date, cur_entry_kind=cur_entry_kind,
        cum_mult=cum_mult, stop_qqq=stop_qqq,
        tqqq_target=tqqq_target, tqqq_stop=tqqq_stop, be_moved=be_moved,
        last_dew=last_dew, cup_entered=cup_entered, cdn_active=cdn_active,
        pyramid_on=pyramid_on, entry_vix=entry_vix, decayed_active=decayed_active,
    )
    return trades, state


# ----------------------------------------------------------------------
# Evaluate today's bar (index N-1) for tomorrow's pending action
# ----------------------------------------------------------------------
def evaluate_last_bar(df, state):
    n = len(df)
    i = n - 1

    Date = df["Date"].values
    Close = df["Close"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    MTI = df["MTI"].values
    BSR = df["BSR"].values
    RT = df["RT"].values
    VIX = df["VIX"].values
    tqqq_rt = df["tqqq_rt"].values
    tqqq_atr = df["tqqq_atr"].values
    tqqq_close = df["tqqq_close"].values
    o2c_q = df["o2c_QQQ"].values; c2c_q = df["c2c_QQQ"].values
    o2c_i = df["o2c_INV"].values; c2c_i = df["c2c_INV"].values
    o2c_t = df["tqqq_o2c"].values; c2c_t = df["tqqq_c2c"].values

    in_trade = state["in_trade"]
    instrument = state["instrument"]
    entry_idx = state["entry_idx"]
    cum_mult = state["cum_mult"]
    stop_qqq = state["stop_qqq"]
    tqqq_target = state["tqqq_target"]
    tqqq_stop = state["tqqq_stop"]
    be_moved = state["be_moved"]
    last_dew = state["last_dew"]
    cup_entered = state["cup_entered"]
    cdn_active = state["cdn_active"]
    cur_entry_kind = state["cur_entry_kind"]
    pyramid_on = state["pyramid_on"]
    entry_vix = state["entry_vix"]
    decayed_active = state["decayed_active"]

    # Today's signal flags
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

    pending = dict(action="HOLD", reason="", exit_type=None, entry_inst=None,
                   block_reasons=[], today_mult=None, today_pmp=None,
                   today_decay=False, decay_action=None,
                   pyr_pre=False, cup_f=cup_f, cdn_f=cdn_f)

    # ── In a trade: check exits/decay ──────────────────────────────
    if in_trade:
        # Accrue today's return
        if instrument == "TQQQ":
            d = o2c_t[i] if i == entry_idx else c2c_t[i]
        elif instrument == "QQQ":
            d = o2c_q[i] if i == entry_idx else c2c_q[i]
        else:
            d = o2c_i[i] if i == entry_idx else c2c_i[i]
        today_mult = cum_mult * (1.0 + d)
        pending["today_mult"] = today_mult

        # Pyramid + decay state for today
        pyr_pre = pyramid_on if instrument == "TQQQ" else False
        pmp_pyr = get_pmp(VIX[i]) if pyr_pre else 1.0
        should_decay_today = False
        if instrument == "TQQQ" and entry_vix is not None and not np.isnan(entry_vix):
            days_held = i - entry_idx
            if days_held >= 15 and (VIX[i] - entry_vix) >= 1.0:
                should_decay_today = True
        pending["pyr_pre"] = pyr_pre
        pending["today_pmp"] = pmp_pyr * (0.30 if should_decay_today else 1.0)
        pending["today_decay"] = should_decay_today
        pending["pmp_pyr"] = pmp_pyr

        # Breakeven preview
        if instrument == "TQQQ" and not be_moved and today_mult - 1 >= 0.09:
            pending["breakeven_preview"] = True
        else:
            pending["breakeven_preview"] = False

        # Priority preemption preview (1a/1b can fire on today's close → tomorrow's open)
        # 1a: in QQQ/INV, cup_f true → exit current, enter TQQQ
        if instrument in ("QQQ", "INV") and cup_f and not cup_entered:
            pending.update(action="PREEMPT_1A",
                           exit_type="Preempt → TQQQ (cup)",
                           entry_inst="TQQQ",
                           reason=f"cup_f active while in {instrument}: SELL {instrument}, BUY TQQQ tomorrow open")
            return pending
        # 1b: in QQQ, C/Dn switch
        if (instrument == "QQQ" and not cdn_active
                and cc[i] == "C/Dn"
                and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                and not np.isnan(MTI[i]) and MTI[i] < 0.95):
            pending.update(action="PREEMPT_1B",
                           exit_type="Preempt → TQQQ (cdn)",
                           entry_inst="TQQQ",
                           reason=f"C/Dn switch while in QQQ: SELL QQQ, BUY TQQQ (C/Dn) tomorrow open")
            return pending

        # Exit checks - same priority as engine
        # Effective stop level for TQQQ (preview breakeven)
        eff_stop = tqqq_stop
        if instrument == "TQQQ" and not be_moved and today_mult - 1 >= 0.09:
            # Preview: stop would ratchet to entry_open before this bar's exit checks
            from_open = df["tqqq_open"].iloc[entry_idx]
            eff_stop = max(tqqq_stop or 0.0, from_open)

        if instrument == "TQQQ" and i != entry_idx and not np.isnan(MTI[i]) and MTI[i] < 0.75:
            pending.update(action="EXIT", exit_type="MTI",
                           reason=f"MTI={MTI[i]:.2f} < 0.75 — exit TQQQ at tomorrow's open")
        elif instrument == "TQQQ" and tqqq_close[i] >= tqqq_target:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"TQQQ close {tqqq_close[i]:.2f} ≥ target {tqqq_target:.2f} — exit at tomorrow's open")
        elif instrument == "TQQQ" and tqqq_close[i] <= eff_stop:
            label = "Stop (breakeven)" if (be_moved or pending["breakeven_preview"]) else "Stop"
            pending.update(action="EXIT", exit_type=label,
                           reason=f"TQQQ close {tqqq_close[i]:.2f} ≤ stop {eff_stop:.2f} — exit at tomorrow's open")
        elif instrument == "TQQQ" and (i - entry_idx) >= 70:
            pending.update(action="EXIT", exit_type="MaxHold",
                           reason=f"Held {i-entry_idx} bars ≥ 70 — exit TQQQ at tomorrow's open")
        elif instrument == "QQQ" and today_mult - 1 >= 0.05:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"QQQ cum {(today_mult-1)*100:.2f}% ≥ 5% — exit at tomorrow's open")
        elif instrument == "QQQ" and Close[i] <= stop_qqq:
            pending.update(action="EXIT", exit_type="ATR",
                           reason=f"QQQ close {Close[i]:.2f} ≤ ATR stop {stop_qqq:.2f} — exit at tomorrow's open")
        elif instrument == "INV" and today_mult - 1 >= 0.05:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"INV cum {(today_mult-1)*100:.2f}% ≥ 5% — exit (cover SQQQ) at tomorrow's open")
        elif instrument == "INV" and Close[i] >= stop_qqq:
            pending.update(action="EXIT", exit_type="ATR",
                           reason=f"QQQ close {Close[i]:.2f} ≥ ATR stop {stop_qqq:.2f} — exit INV at tomorrow's open")
        elif instrument == "TQQQ":
            if cur_entry_kind == "cup" and cc[i] == "C/Dn":
                pending.update(action="EXIT", exit_type="Regime",
                               reason="cc flipped to C/Dn — exit C/Up TQQQ at tomorrow's open")
            elif cur_entry_kind == "cdn" and (DEW[i] == "Sell"
                    or (i != entry_idx and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn")):
                pending.update(action="EXIT", exit_type="Regime",
                               reason="C/Dn TQQQ ended (DEW Sell or C/Up flip) — exit at tomorrow's open")
        elif instrument == "QQQ" and i != entry_idx:
            if i > 0 and Trend[i] == "C/Dn" and cc[i-1] == "C/Up":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Dn flip while long QQQ — exit at tomorrow's open")
            elif DEW[i] == "Sell":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Sell while long QQQ — exit at tomorrow's open")
        elif instrument == "INV" and i != entry_idx:
            if i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Up flip while short INV — exit at tomorrow's open")
            elif DEW[i] == "Buy":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Buy while short INV — exit at tomorrow's open")

        # Decay scale operation (only if HOLDing TQQQ)
        if pending["action"] == "HOLD" and instrument == "TQQQ":
            if should_decay_today and not decayed_active:
                pending["decay_action"] = "SCALE_DOWN"
                pending["reason"] = (
                    f"DECAY ACTIVATES — held {i-entry_idx} days, VIX {VIX[i]:.2f} − entry_VIX "
                    f"{entry_vix:.2f} = +{VIX[i]-entry_vix:.2f} ≥ 1.0. SELL 70% of TQQQ at tomorrow's open."
                )
            elif not should_decay_today and decayed_active:
                pending["decay_action"] = "SCALE_UP"
                gap = (VIX[i] - entry_vix) if entry_vix is not None else float("nan")
                pending["reason"] = (
                    f"DECAY CLEARS — VIX gap now {gap:+.2f} (< 1.0). BUY 70% of TQQQ back at tomorrow's open."
                )
            else:
                pending["reason"] = f"No exit. Holding TQQQ at {pending['today_pmp']:.2f}× sizing."

        if pending["action"] == "HOLD" and instrument != "TQQQ":
            pending["reason"] = f"No exit condition met. Holding {instrument}."

    # ── Not in a trade: check entries ──────────────────────────────
    else:
        if cup_f and not cup_entered:
            pending.update(action="ENTER", entry_inst="TQQQ",
                           reason="C/Up TQQQ (cup_f) — BUY TQQQ at tomorrow's open")
        elif cdn_f and last_dew != "Buy":
            pending.update(action="ENTER", entry_inst="TQQQ",
                           reason="C/Dn TQQQ (cdn_f, VIX<27) — BUY TQQQ (C/Dn variant, TP=1.30) at tomorrow's open")
        elif DEW[i] == "Buy" and last_dew != "Buy":
            reasons = []
            if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90: reasons.append(f"RT={RT[i]:.2f}∈[0.85,0.90)")
            if not np.isnan(BSR[i]) and BSR[i] > 1.50: reasons.append(f"BSR={BSR[i]:.2f}>1.50")
            if not np.isnan(MTI[i]) and MTI[i] > 1.15: reasons.append(f"MTI={MTI[i]:.2f}>1.15")
            if not np.isnan(VIX[i]) and VIX[i] > 35: reasons.append(f"VIX={VIX[i]:.2f}>35")
            if reasons:
                pending.update(action="BLOCKED", entry_inst="QQQ", block_reasons=reasons,
                               reason=f"DEW Buy → QQQ blocked: {', '.join(reasons)}")
            else:
                pending.update(action="ENTER", entry_inst="QQQ",
                               reason="DEW Buy — BUY QQQ at tomorrow's open")
        elif DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn":
            reasons = []
            if (not np.isnan(RT[i]) and RT[i] > 0.95) and (not np.isnan(MTI[i]) and MTI[i] > 0.95):
                reasons.append(f"RT={RT[i]:.2f}>0.95 AND MTI={MTI[i]:.2f}>0.95")
            if not np.isnan(VIX[i]) and VIX[i] > 35:
                reasons.append(f"VIX={VIX[i]:.2f}>35")
            if reasons:
                pending.update(action="BLOCKED", entry_inst="INV", block_reasons=reasons,
                               reason=f"DEW Sell → INV blocked: {', '.join(reasons)}")
            else:
                pending.update(action="ENTER", entry_inst="INV",
                               reason="DEW Sell in C/Dn — SHORT QQQ via SQQQ (1/3 position) at tomorrow's open")
        else:
            pending["reason"] = "No signal. Flat."

    return pending


# ----------------------------------------------------------------------
# Report formatter
# ----------------------------------------------------------------------
def yn(cond): return "✓" if cond else "✗"


def print_report(df, state, pending, trades):
    n = len(df)
    today = df.iloc[-1]
    today_str = str(today["Date"])[:10]

    W = 70
    def rule(ch="─"): return ch * W
    def hdr(txt, ch="═"):
        pad = (W - len(txt) - 2) // 2
        return ch * pad + f" {txt} " + ch * (W - pad - len(txt) - 2)

    out = []
    out.append(rule("═"))
    out.append(hdr(f"DAMPIER NITRO++ v16  ·  DAILY SIGNAL  ·  {today_str}"))
    out.append(rule("═"))
    out.append("")

    # ── Current position ─────────────────────────────────────
    if state["in_trade"]:
        inst = state["instrument"]
        side = "LONG" if inst != "INV" else "SHORT (via SQQQ 1/3)"
        eidx = state["entry_idx"]
        bars_held = n - 1 - eidx
        if inst == "TQQQ":
            entry_price = df["tqqq_open"].iloc[eidx]
        else:
            entry_price = df["Open"].iloc[eidx]

        pnl_yest = (state["cum_mult"] - 1.0) * 100.0
        pnl_today = ((pending.get("today_mult") or state["cum_mult"]) - 1.0) * 100.0

        out.append(f"  CURRENT POSITION :  {side} {inst}")
        out.append(f"  Entry Date       :  {str(state['entry_date'])[:10]}")
        out.append(f"  Entry Price      :  ${entry_price:.4f}  ({inst})")
        if state["entry_vix"] is not None and not np.isnan(state["entry_vix"]):
            out.append(f"  Entry VIX        :  {state['entry_vix']:.2f}")
        out.append(f"  Bars Held        :  {bars_held}")
        out.append(f"  Cum Return (raw) :  {pnl_yest:+.2f}% thru yesterday  /  {pnl_today:+.2f}% thru today's close")

        if inst == "TQQQ":
            pyr_pre = pending.get("pyr_pre", False)
            pmp_pyr = pending.get("pmp_pyr", 1.0)
            decay_now = pending.get("today_decay", False)
            today_pmp = pending.get("today_pmp", 1.0)
            pyr_str = f"ACTIVE (pyramid PMP={pmp_pyr:.2f}× — VIX band)" if pyr_pre else "inactive"
            decay_str = "ACTIVE (0.30× scale)" if decay_now else "inactive"
            out.append(f"  Pyramid          :  {pyr_str}")
            out.append(f"  Decay            :  {decay_str}")
            out.append(f"  Effective Sizing :  {today_pmp:.3f}×")
            stop_label = state["tqqq_stop"]
            if state["be_moved"]:
                stop_note = " (breakeven)"
            elif pending.get("breakeven_preview"):
                stop_note = " → ratchet to entry tomorrow"
            else:
                stop_note = ""
            out.append(f"  TQQQ Target      :  ${state['tqqq_target']:.4f}  |  Stop: ${stop_label:.4f}{stop_note}")
        elif inst in ("QQQ", "INV"):
            stop_v = state["stop_qqq"]
            comp = "<=" if inst == "QQQ" else ">="
            out.append(f"  ATR Stop (QQQ {comp}) :  ${stop_v:.4f}")
    else:
        out.append("  CURRENT POSITION :  FLAT")

    # ── Market indicators ─────────────────────────────────────
    out.append("")
    out.append("  " + rule("─"))
    out.append(f"  MARKET INDICATORS  (as of {today_str} close)")
    out.append("  " + rule("─"))

    cc_v = today.get("cc", "—")
    dew_v = today.get("DEW_Signal", "") or "—"
    rt = today.get("RT", float("nan"))
    mti = today.get("MTI", float("nan"))
    bsr = today.get("BSR", float("nan"))
    vix = today.get("VIX", float("nan"))
    tatr = today.get("tqqq_atr", float("nan"))
    trt = today.get("tqqq_rt", float("nan"))
    tcl = today.get("tqqq_close", float("nan"))
    qcl = today.get("Close", float("nan"))

    def fmt(label, val, fmt_str=":.2f", suf=""):
        if isinstance(val, float) and np.isnan(val):
            return f"  {label:<17}:  —"
        return f"  {label:<17}:  {format(val, fmt_str)}{suf}"

    out.append(f"  Confirmed Call   :  {cc_v}")
    out.append(f"  DEW Signal       :  {dew_v}")
    out.append(fmt("QQQ Close", qcl, ".2f", ""))
    out.append(fmt("QQQ RT", rt, ".2f"))
    out.append(fmt("MTI", mti, ".2f"))
    out.append(fmt("BSR", bsr, ".2f"))
    out.append(fmt("VIX", vix, ".2f"))
    out.append(fmt("TQQQ ATR%", tatr, ".2f", "%"))
    out.append(fmt("TQQQ RT", trt, ".4f"))
    out.append(fmt("TQQQ Close", tcl, ".2f", ""))
    out.append(f"  Pyramid PMP(VIX) :  {get_pmp(vix):.2f}×  (band: {'<20' if vix<20 else '<32' if vix<32 else '≥32'})")

    # ── Entry signal conditions ──────────────────────────────
    out.append("")
    out.append("  " + rule("─"))
    out.append("  ENTRY SIGNAL CONDITIONS")
    out.append("  " + rule("─"))

    cup_parts = [
        f"cc=C/Up {yn(cc_v=='C/Up')}",
        f"TQQQ_RT<1.40 {yn(not np.isnan(trt) and trt<1.40)}",
        f"BSR>1.05 {yn(not np.isnan(bsr) and bsr>1.05)}",
        f"TQQQ_ATR<7.0 {yn(not np.isnan(tatr) and tatr<7.0)}",
        f"VIX<30 {yn(not np.isnan(vix) and vix<30)}",
        f"!cup_entered {yn(not state['cup_entered'])}",
    ]
    cup_all = pending.get("cup_f", False) and not state["cup_entered"]
    out.append(f"  cup_f (C/Up TQQQ):  {'ACTIVE ✓' if cup_all else 'inactive'}")
    out.append(f"      " + " · ".join(cup_parts))

    cdn_parts = [
        f"cc=C/Dn {yn(cc_v=='C/Dn')}",
        f"DEW=Buy {yn(dew_v=='Buy')}",
        f"last_dew≠Buy {yn(state['last_dew']!='Buy')}",
        f"RT∈[0.95,1.00) {yn(not np.isnan(rt) and 0.95<=rt<1.00)}",
        f"MTI<1.00 {yn(not np.isnan(mti) and mti<1.00)}",
        f"VIX<27 {yn(not np.isnan(vix) and vix<27)}",
    ]
    cdn_all = pending.get("cdn_f", False)
    out.append(f"  cdn_f (C/Dn TQQQ):  {'ACTIVE ✓' if cdn_all else 'inactive'}")
    out.append(f"      " + " · ".join(cdn_parts))

    # QQQ
    qqq_blk = []
    if not np.isnan(rt) and 0.85 <= rt < 0.90: qqq_blk.append(f"RT={rt:.2f}∈[0.85,0.90)")
    if not np.isnan(bsr) and bsr > 1.50: qqq_blk.append(f"BSR={bsr:.2f}>1.50")
    if not np.isnan(mti) and mti > 1.15: qqq_blk.append(f"MTI={mti:.2f}>1.15")
    if not np.isnan(vix) and vix > 35: qqq_blk.append(f"VIX={vix:.2f}>35")
    qqq_signal = (dew_v == "Buy" and state["last_dew"] != "Buy")
    if qqq_signal and not qqq_blk:
        qqq_str = "ACTIVE ✓"
    elif qqq_signal and qqq_blk:
        qqq_str = f"DEW Buy BLOCKED: {', '.join(qqq_blk)}"
    else:
        qqq_str = "inactive"
    out.append(f"  DEW Buy → QQQ    :  {qqq_str}")

    # INV
    inv_blk = []
    if (not np.isnan(rt) and rt > 0.95) and (not np.isnan(mti) and mti > 0.95):
        inv_blk.append(f"RT={rt:.2f}>0.95 AND MTI={mti:.2f}>0.95")
    if not np.isnan(vix) and vix > 35:
        inv_blk.append(f"VIX={vix:.2f}>35")
    inv_signal = (dew_v == "Sell" and state["last_dew"] != "Sell" and cc_v == "C/Dn")
    if inv_signal and not inv_blk:
        inv_str = "ACTIVE ✓"
    elif inv_signal and inv_blk:
        inv_str = f"DEW Sell BLOCKED: {', '.join(inv_blk)}"
    else:
        inv_str = "inactive"
    out.append(f"  DEW Sell → INV   :  {inv_str}  (INV = SQQQ 1/3 position)")

    # ── Tomorrow's action ─────────────────────────────────────
    out.append("")
    out.append("  " + rule("═"))
    a = pending["action"]
    if a == "ENTER":
        inst = pending["entry_inst"]
        verb = "LONG" if inst != "INV" else "SHORT (via SQQQ 1/3)"
        out.append(f"  ▶  TOMORROW'S ACTION:  BUY {inst} at open  ({verb})")
    elif a == "EXIT":
        out.append(f"  ▶  TOMORROW'S ACTION:  SELL {state['instrument']} at open  ({pending['exit_type']})")
    elif a == "PREEMPT_1A":
        out.append(f"  ▶  TOMORROW'S ACTION:  PREEMPT 1a — SELL {state['instrument']}, BUY TQQQ (cup)")
    elif a == "PREEMPT_1B":
        out.append(f"  ▶  TOMORROW'S ACTION:  PREEMPT 1b — SELL QQQ, BUY TQQQ (C/Dn variant)")
    elif a == "BLOCKED":
        out.append(f"  ▶  TOMORROW'S ACTION:  HOLD FLAT  (entry blocked)")
    elif state["in_trade"]:
        if pending.get("decay_action") == "SCALE_DOWN":
            out.append(f"  ▶  TOMORROW'S ACTION:  SELL 70% of TQQQ at open  (decay activates → 0.30× sizing)")
        elif pending.get("decay_action") == "SCALE_UP":
            out.append(f"  ▶  TOMORROW'S ACTION:  BUY 70% of TQQQ back at open  (decay clears → full sizing)")
        else:
            out.append(f"  ▶  TOMORROW'S ACTION:  HOLD {state['instrument']}")
    else:
        out.append(f"  ▶  TOMORROW'S ACTION:  HOLD FLAT")

    out.append(f"     Reason: {pending['reason']}")
    out.append("  " + rule("═"))

    # ── Recent trades ─────────────────────────────────────────
    out.append("")
    out.append(f"  LAST 5 COMPLETED TRADES (of {len(trades)} total)")
    out.append("  " + rule("─"))
    for t in trades[-5:]:
        out.append(f"    {str(t['entry_date'])[:10]}  {t['inst']:4s}  →  "
                   f"{str(t['exit_date'])[:10]}  {t['ret']*100:+7.2f}%  [{t['exit']}]")

    out.append("")
    out.append(rule("═"))
    body = "\n".join(out)
    print(body)
    return body, today_str


# ──────────────────────────────────────────────────────────────────────────────
# Notifications
# ──────────────────────────────────────────────────────────────────────────────
def build_subject(state, pending, today_str):
    a = pending["action"]
    if a == "ENTER":
        return f"Nitro++ v16 Signal {today_str}: BUY {pending['entry_inst']}"
    if a == "EXIT":
        return f"Nitro++ v16 Signal {today_str}: SELL {state['instrument']} ({pending.get('exit_type','exit')})"
    if a == "PREEMPT_1A":
        return f"Nitro++ v16 Signal {today_str}: PREEMPT 1a — SELL {state['instrument']} → BUY TQQQ"
    if a == "PREEMPT_1B":
        return f"Nitro++ v16 Signal {today_str}: PREEMPT 1b — SELL QQQ → BUY TQQQ"
    if a == "BLOCKED":
        return f"Nitro++ v16 Signal {today_str}: BLOCKED ({pending.get('entry_inst','?')})"
    if state["in_trade"]:
        if pending.get("decay_action") == "SCALE_DOWN":
            return f"Nitro++ v16 Signal {today_str}: SCALE DOWN TQQQ → 0.30×"
        if pending.get("decay_action") == "SCALE_UP":
            return f"Nitro++ v16 Signal {today_str}: SCALE UP TQQQ → full"
        inst = state["instrument"]
        if inst == "TQQQ":
            tag = "C/Dn TQQQ" if state.get("cdn_active") else "C/Up TQQQ"
            if state.get("pyramid_on"):
                tag += " [PYRAMID]"
            return f"Nitro++ v16 Signal {today_str}: HOLD {tag}"
        return f"Nitro++ v16 Signal {today_str}: HOLD {inst}"
    return f"Nitro++ v16 Signal {today_str}: FLAT"


def build_sms_summary(state, pending, today_str):
    short = today_str[2:]  # YY-MM-DD
    a = pending["action"]
    if a == "ENTER":
        inst = pending["entry_inst"]
        verb = "BUY" if inst != "INV" else "SHORT"
        msg = f"Nitro {short}: {verb} {inst} at open"
    elif a == "EXIT":
        msg = f"Nitro {short}: SELL {state['instrument']} at open ({pending.get('exit_type','')})".rstrip(" ()")
    elif a == "PREEMPT_1A":
        msg = f"Nitro {short}: PREEMPT1a SELL {state['instrument']}, BUY TQQQ (cup)"
    elif a == "PREEMPT_1B":
        msg = f"Nitro {short}: PREEMPT1b SELL QQQ, BUY TQQQ (C/Dn)"
    elif a == "BLOCKED":
        msg = f"Nitro {short}: BLOCKED {pending.get('entry_inst','')}".rstrip()
    elif state["in_trade"] and pending.get("decay_action") == "SCALE_DOWN":
        msg = f"Nitro {short}: SCALE DOWN TQQQ at open (decay → 0.30×)"
    elif state["in_trade"] and pending.get("decay_action") == "SCALE_UP":
        msg = f"Nitro {short}: SCALE UP TQQQ at open (decay clears → full)"
    elif state["in_trade"]:
        inst = state["instrument"]
        verb = "PYRAMID" if state.get("pyramid_on") else "HOLD"
        msg = f"Nitro {short}: {verb} {inst}"
        tm = pending.get("today_mult")
        if tm is not None:
            pnl = (tm - 1.0) * 100.0
            msg += f"  {pnl:+.2f}%"
    else:
        msg = f"Nitro {short}: HOLD FLAT"
    return msg[:160]


def send_email(subject, body_text):
    if not GMAIL_PASS:
        print("send_email: GOOGLE_APP_PASSWORD not set — skipping")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(TO_EMAIL)
    msg.attach(MIMEText(body_text, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())


def send_imessage(numbers, body):
    for num in numbers:
        service_type = "SMS" if num in SMS_FORCE else "iMessage"
        # Body is passed as an argv argument (item 1 of argv) rather than
        # embedded in the script text — this avoids AppleScript string-literal
        # breakage on quotes, backslashes and newlines.
        script = (
            "on run argv\n"
            f"  set svc to first service whose service type = {service_type}\n"
            f'  send (item 1 of argv) to participant "{num}" of svc\n'
            "end run"
        )
        try:
            subprocess.run(["osascript", "-e", script, body],
                           check=True, capture_output=True, timeout=30)
            print(f"  iMessage sent to {num} ({service_type})")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: iMessage to {num} timed out after 30s — not delivered")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="replace").strip() if e.stderr else "(no detail)"
            print(f"  WARNING: iMessage to {num} FAILED — not delivered: {err}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_data()

    print("Computing DEW oscillator ...")
    dew_df = compute_dew(qqq_raw)
    df_full = df_full.merge(dew_df, on="Date", how="left")
    df_full["DEW_Signal"] = df_full["DEW_Signal"].fillna("").astype(str)

    # Trim to start >= 2000-01-01 per Step 6 spec note
    df_full = df_full[df_full["Date"] >= START_DATE].reset_index(drop=True)

    print(f"Running v16 production engine on {len(df_full)} bars "
          f"({str(df_full['Date'].iloc[0])[:10]} → {str(df_full['Date'].iloc[-1])[:10]}) ...\n")
    trades, state = run_signal_engine(df_full)

    pending = evaluate_last_bar(df_full, state)

    body, today_str = print_report(df_full, state, pending, trades)

    subject = build_subject(state, pending, today_str)
    sms     = build_sms_summary(state, pending, today_str)

    print(f"\nSending email to {TO_EMAIL} ...")
    send_email(subject, body)
    print("Email sent.")
    print(f"Sending iMessage/SMS to {SMS_NUMBERS}: {sms}")
    send_imessage(SMS_NUMBERS, sms)


if __name__ == "__main__":
    main()
