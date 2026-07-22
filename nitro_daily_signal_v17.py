"""
Dampier Nitro++ v17 - Daily Signal Generator
Replays the v17 production engine (de-margined pyramid + Cap setup; Mean-Rev
omitted; C/Dn-TQQQ VIX < 27) and reports tomorrow's pending action.

Live mapping:
  TQQQ: long TQQQ at full equity (x pmp_pyr x decay_mult, PMP ceiling 1.00)
  QQQ : long QQQ at full equity  (DEW-Buy leg OR Cap capitulation leg)
  INV : short QQQ via SQQQ at 1/3 position (post 2010-02-11)

Set NITRO_DRYRUN=1 to print the report without sending email / iMessage.
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
from nitro_v16 import load_data, compute_dew
from nitro_v17 import get_pmp, add_cap_feature, CAP_DROP5_THR, CAP_COOLDOWN, CAP_ATR_MULT
# Mean-Rev (Setup 6) is not traded by v17 - these are imported for the
# display-only Mean-Rev block in the daily report.
from nitro_v17_backtest import add_v17_features, STRETCH_THR, VOL20_THR

START_DATE = pd.Timestamp("2000-01-01")
DRYRUN = bool(os.environ.get("NITRO_DRYRUN"))

# ── Email / SMS config ─────────────────────────────────────────────────────────
GMAIL_USER  = os.environ.get("GOOGLE_EMAIL", "dampiermike@gmail.com")
GMAIL_PASS  = os.environ.get("GOOGLE_APP_PASSWORD", "")
TO_EMAIL    = ["dampiermike@gmail.com", "ddampier777@gmail.com", "brooke.hoover@yahoo.com"]
SMS_NUMBERS = ["+12256144680", "+13038818222", "+18137815601"]
SMS_FORCE   = {"+18137815601"}


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
    cur_entry_kind = None
    last_exit_idx = None

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
        cap_f = (not np.isnan(drop5[i]) and drop5[i] <= CAP_DROP5_THR)

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
            last_exit_idx = i
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
            last_exit_idx = i
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
                last_exit_idx = i
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

        # Phase C - entries (cup > cdn > QQQ Buy > INV Sell > Cap)
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
            # Setup 5: Cap (capitulation buy) - LONG QQQ, lowest priority
            if (cap_f and (last_exit_idx is None or i - last_exit_idx >= CAP_COOLDOWN)
                    and not fell_through and not np.isnan(ATR14[i])):
                in_trade = True; instrument = "QQQ"; entry_idx = i + 1
                cum_mult = 1.0
                stop_qqq = Open_[i+1] - CAP_ATR_MULT * ATR14[i]
                cur_entry_date = Date[i+1]; cur_entry_kind = "cap"
                continue

    state = dict(
        in_trade=in_trade, instrument=instrument, entry_idx=entry_idx,
        entry_date=cur_entry_date, cur_entry_kind=cur_entry_kind,
        cum_mult=cum_mult, stop_qqq=stop_qqq,
        tqqq_target=tqqq_target, tqqq_stop=tqqq_stop, be_moved=be_moved,
        last_dew=last_dew, cup_entered=cup_entered, cdn_active=cdn_active,
        pyramid_on=pyramid_on, entry_vix=entry_vix, decayed_active=decayed_active,
        last_exit_idx=last_exit_idx,
    )
    return trades, state


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
    drop5 = df["drop5"].values
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
    last_exit_idx = state["last_exit_idx"]

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
    cap_f = (not np.isnan(drop5[i]) and drop5[i] <= CAP_DROP5_THR)
    cap_cooldown_ok = (last_exit_idx is None or i - last_exit_idx >= CAP_COOLDOWN)

    pending = dict(action="HOLD", reason="", exit_type=None, entry_inst=None,
                   block_reasons=[], today_mult=None, today_pmp=None,
                   today_decay=False, decay_action=None,
                   pyr_pre=False, cup_f=cup_f, cdn_f=cdn_f,
                   cap_f=cap_f, cap_cooldown_ok=cap_cooldown_ok, drop5=drop5[i])

    if in_trade:
        if instrument == "TQQQ":
            d = o2c_t[i] if i == entry_idx else c2c_t[i]
        elif instrument == "QQQ":
            d = o2c_q[i] if i == entry_idx else c2c_q[i]
        else:
            d = o2c_i[i] if i == entry_idx else c2c_i[i]
        today_mult = cum_mult * (1.0 + d)
        pending["today_mult"] = today_mult

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

        if instrument == "TQQQ" and not be_moved and today_mult - 1 >= 0.09:
            pending["breakeven_preview"] = True
        else:
            pending["breakeven_preview"] = False

        if instrument in ("QQQ", "INV") and cup_f and not cup_entered:
            pending.update(action="PREEMPT_1A",
                           exit_type="Preempt -> TQQQ (cup)",
                           entry_inst="TQQQ",
                           reason=f"cup_f active while in {instrument}: SELL {instrument}, BUY TQQQ tomorrow open")
            return pending
        if (instrument == "QQQ" and not cdn_active
                and cc[i] == "C/Dn"
                and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                and not np.isnan(MTI[i]) and MTI[i] < 0.95):
            pending.update(action="PREEMPT_1B",
                           exit_type="Preempt -> TQQQ (cdn)",
                           entry_inst="TQQQ",
                           reason=f"C/Dn switch while in QQQ: SELL QQQ, BUY TQQQ (C/Dn) tomorrow open")
            return pending

        eff_stop = tqqq_stop
        if instrument == "TQQQ" and not be_moved and today_mult - 1 >= 0.09:
            from_open = df["tqqq_open"].iloc[entry_idx]
            eff_stop = max(tqqq_stop or 0.0, from_open)

        if instrument == "TQQQ" and i != entry_idx and not np.isnan(MTI[i]) and MTI[i] < 0.75:
            pending.update(action="EXIT", exit_type="MTI",
                           reason=f"MTI={MTI[i]:.2f} < 0.75 - exit TQQQ at tomorrow's open")
        elif instrument == "TQQQ" and tqqq_close[i] >= tqqq_target:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"TQQQ close {tqqq_close[i]:.2f} >= target {tqqq_target:.2f} - exit at tomorrow's open")
        elif instrument == "TQQQ" and tqqq_close[i] <= eff_stop:
            label = "Stop (breakeven)" if (be_moved or pending["breakeven_preview"]) else "Stop"
            pending.update(action="EXIT", exit_type=label,
                           reason=f"TQQQ close {tqqq_close[i]:.2f} <= stop {eff_stop:.2f} - exit at tomorrow's open")
        elif instrument == "TQQQ" and (i - entry_idx) >= 70:
            pending.update(action="EXIT", exit_type="MaxHold",
                           reason=f"Held {i-entry_idx} bars >= 70 - exit TQQQ at tomorrow's open")
        elif instrument == "QQQ" and today_mult - 1 >= 0.05:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"QQQ cum {(today_mult-1)*100:.2f}% >= 5% - exit at tomorrow's open")
        elif instrument == "QQQ" and Close[i] <= stop_qqq:
            pending.update(action="EXIT", exit_type="ATR",
                           reason=f"QQQ close {Close[i]:.2f} <= ATR stop {stop_qqq:.2f} - exit at tomorrow's open")
        elif instrument == "INV" and today_mult - 1 >= 0.05:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"INV cum {(today_mult-1)*100:.2f}% >= 5% - exit (cover SQQQ) at tomorrow's open")
        elif instrument == "INV" and Close[i] >= stop_qqq:
            pending.update(action="EXIT", exit_type="ATR",
                           reason=f"QQQ close {Close[i]:.2f} >= ATR stop {stop_qqq:.2f} - exit INV at tomorrow's open")
        elif instrument == "TQQQ":
            if cur_entry_kind == "cup" and cc[i] == "C/Dn":
                pending.update(action="EXIT", exit_type="Regime",
                               reason="cc flipped to C/Dn - exit C/Up TQQQ at tomorrow's open")
            elif cur_entry_kind == "cdn" and (DEW[i] == "Sell"
                    or (i != entry_idx and i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn")):
                pending.update(action="EXIT", exit_type="Regime",
                               reason="C/Dn TQQQ ended (DEW Sell or C/Up flip) - exit at tomorrow's open")
        elif instrument == "QQQ" and i != entry_idx:
            if i > 0 and Trend[i] == "C/Dn" and cc[i-1] == "C/Up":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Dn flip while long QQQ - exit at tomorrow's open")
            elif DEW[i] == "Sell":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Sell while long QQQ - exit at tomorrow's open")
        elif instrument == "INV" and i != entry_idx:
            if i > 0 and Trend[i] == "C/Up" and cc[i-1] == "C/Dn":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Up flip while short INV - exit at tomorrow's open")
            elif DEW[i] == "Buy":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Buy while short INV - exit at tomorrow's open")

        if pending["action"] == "HOLD" and instrument == "TQQQ":
            if should_decay_today and not decayed_active:
                pending["decay_action"] = "SCALE_DOWN"
                pending["reason"] = (
                    f"DECAY ACTIVATES - held {i-entry_idx} days, VIX {VIX[i]:.2f} - entry_VIX "
                    f"{entry_vix:.2f} = +{VIX[i]-entry_vix:.2f} >= 1.0. SELL 70% of TQQQ at tomorrow's open."
                )
            elif not should_decay_today and decayed_active:
                pending["decay_action"] = "SCALE_UP"
                gap = (VIX[i] - entry_vix) if entry_vix is not None else float("nan")
                pending["reason"] = (
                    f"DECAY CLEARS - VIX gap now {gap:+.2f} (< 1.0). BUY 70% of TQQQ back at tomorrow's open."
                )
            else:
                pending["reason"] = f"No exit. Holding TQQQ at {pending['today_pmp']:.2f}x sizing."

        if pending["action"] == "HOLD" and instrument != "TQQQ":
            kind = " (Cap)" if cur_entry_kind == "cap" else ""
            pending["reason"] = f"No exit condition met. Holding {instrument}{kind}."

    else:
        if cup_f and not cup_entered:
            pending.update(action="ENTER", entry_inst="TQQQ",
                           reason="C/Up TQQQ (cup_f) - BUY TQQQ at tomorrow's open")
        elif cdn_f and last_dew != "Buy":
            pending.update(action="ENTER", entry_inst="TQQQ",
                           reason="C/Dn TQQQ (cdn_f, VIX<27) - BUY TQQQ (C/Dn variant, TP=1.30) at tomorrow's open")
        elif DEW[i] == "Buy" and last_dew != "Buy":
            reasons = []
            if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90: reasons.append(f"RT={RT[i]:.2f} in [0.85,0.90)")
            if not np.isnan(BSR[i]) and BSR[i] > 1.50: reasons.append(f"BSR={BSR[i]:.2f}>1.50")
            if not np.isnan(MTI[i]) and MTI[i] > 1.15: reasons.append(f"MTI={MTI[i]:.2f}>1.15")
            if not np.isnan(VIX[i]) and VIX[i] > 35: reasons.append(f"VIX={VIX[i]:.2f}>35")
            if reasons:
                pending.update(action="BLOCKED", entry_inst="QQQ", block_reasons=reasons,
                               reason=f"DEW Buy -> QQQ blocked: {', '.join(reasons)}")
            else:
                pending.update(action="ENTER", entry_inst="QQQ",
                               reason="DEW Buy - BUY QQQ at tomorrow's open")
        elif DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn":
            reasons = []
            if (not np.isnan(RT[i]) and RT[i] > 0.95) and (not np.isnan(MTI[i]) and MTI[i] > 0.95):
                reasons.append(f"RT={RT[i]:.2f}>0.95 AND MTI={MTI[i]:.2f}>0.95")
            if not np.isnan(VIX[i]) and VIX[i] > 35:
                reasons.append(f"VIX={VIX[i]:.2f}>35")
            if reasons:
                pending.update(action="BLOCKED", entry_inst="INV", block_reasons=reasons,
                               reason=f"DEW Sell -> INV blocked: {', '.join(reasons)}")
            else:
                pending.update(action="ENTER", entry_inst="INV",
                               reason="DEW Sell in C/Dn - SHORT QQQ via SQQQ (1/3 position) at tomorrow's open")
        elif cap_f and cap_cooldown_ok:
            pending.update(action="ENTER", entry_inst="QQQ",
                           reason=f"Cap capitulation (5-day QQQ drop {drop5[i]*100:.1f}% <= -10%) - BUY QQQ at tomorrow's open")
        else:
            pending["reason"] = "No signal. Flat."

    return pending




def print_report(df, state, pending, trades):
    n = len(df)
    today = df.iloc[-1]
    today_str = str(today["Date"])[:10]

    W = 70
    def rule(ch="-"): return ch * W
    def hdr(txt, ch="="):
        pad = (W - len(txt) - 2) // 2
        return ch * pad + f" {txt} " + ch * (W - pad - len(txt) - 2)

    out = []
    out.append(rule("="))
    out.append(hdr(f"DAMPIER NITRO++ v17  ·  DAILY SIGNAL  ·  {today_str}"))
    out.append(rule("="))
    out.append("")

    if state["in_trade"]:
        inst = state["instrument"]
        side = "LONG" if inst != "INV" else "SHORT (via SQQQ 1/3)"
        eidx = state["entry_idx"]
        bars_held = n - 1 - eidx
        if inst == "TQQQ":
            entry_price = df["tqqq_open"].iloc[eidx]
        else:
            entry_price = df["Open"].iloc[eidx]

        kind_tag = ""
        if inst == "QQQ" and state["cur_entry_kind"] == "cap":
            kind_tag = "  [Cap]"
        pnl_yest = (state["cum_mult"] - 1.0) * 100.0
        pnl_today = ((pending.get("today_mult") or state["cum_mult"]) - 1.0) * 100.0

        out.append(f"  CURRENT POSITION :  {side} {inst}{kind_tag}")
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
            pyr_str = f"ACTIVE (pyramid PMP={pmp_pyr:.2f}x - VIX band)" if pyr_pre else "inactive"
            decay_str = "ACTIVE (0.30x scale)" if decay_now else "inactive"
            out.append(f"  Pyramid          :  {pyr_str}")
            out.append(f"  Decay            :  {decay_str}")
            out.append(f"  Effective Sizing :  {today_pmp:.3f}x")
            stop_label = state["tqqq_stop"]
            if state["be_moved"]:
                stop_note = " (breakeven)"
            elif pending.get("breakeven_preview"):
                stop_note = " -> ratchet to entry tomorrow"
            else:
                stop_note = ""
            out.append(f"  TQQQ Target      :  ${state['tqqq_target']:.4f}  |  Stop: ${stop_label:.4f}{stop_note}")
        elif inst in ("QQQ", "INV"):
            stop_v = state["stop_qqq"]
            comp = "<=" if inst == "QQQ" else ">="
            out.append(f"  ATR Stop (QQQ {comp}) :  ${stop_v:.4f}")
    else:
        out.append("  CURRENT POSITION :  FLAT")

    out.append("")
    out.append("  " + rule("-"))
    out.append(f"  MARKET INDICATORS  (as of {today_str} close)")
    out.append("  " + rule("-"))

    cc_v = today.get("cc", "-")
    dew_v = today.get("DEW_Signal", "") or "-"
    rt = today.get("RT", float("nan"))
    mti = today.get("MTI", float("nan"))
    bsr = today.get("BSR", float("nan"))
    vix = today.get("VIX", float("nan"))
    tatr = today.get("tqqq_atr", float("nan"))
    trt = today.get("tqqq_rt", float("nan"))
    tcl = today.get("tqqq_close", float("nan"))
    qcl = today.get("Close", float("nan"))
    d5 = today.get("drop5", float("nan"))

    # DEW_Signal is populated only on the bar a signal fires, so the prevailing
    # direction is the most recent non-empty value.
    def dew_prevailing():
        d = df["DEW_Signal"].values
        for j in range(len(df) - 1, -1, -1):
            if d[j] in ("Buy", "Sell"):
                return d[j], str(df["Date"].iloc[j])[:10]
        return None, None

    prev_sig, prev_date = dew_prevailing()

    def fmt(label, val, fmt_str=":.2f", suf=""):
        if isinstance(val, float) and np.isnan(val):
            return f"  {label:<17}:  -"
        return f"  {label:<17}:  {format(val, fmt_str)}{suf}"

    out.append(f"  Confirmed Call   :  {cc_v}")
    if dew_v in ("Buy", "Sell"):
        dew_disp = f"{dew_v}  (fired today)"
    elif prev_sig:
        dew_disp = f"{prev_sig}  (last fired {prev_date})"
    else:
        dew_disp = "-"
    out.append(f"  DEW Signal       :  {dew_disp}")
    out.append(fmt("QQQ Close", qcl, ".2f", ""))
    out.append(fmt("QQQ RT", rt, ".2f"))
    out.append(fmt("MTI", mti, ".2f"))
    out.append(fmt("BSR", bsr, ".2f"))
    out.append(fmt("VIX", vix, ".2f"))
    out.append(fmt("TQQQ ATR%", tatr, ".2f", "%"))
    out.append(fmt("TQQQ RT", trt, ".4f"))
    out.append(fmt("TQQQ Close", tcl, ".2f", ""))
    out.append(fmt("5-day QQQ drop", d5 * 100 if not np.isnan(d5) else d5, ".2f", "%"))
    out.append(f"  Pyramid PMP(VIX) :  {get_pmp(vix):.2f}x  (band: {'<20' if vix<20 else '<32' if vix<32 else '>=32'}; ceiling 1.00, no margin)")

    out.append("")
    out.append("  " + rule("-"))
    out.append("  ENTRY SIGNAL CONDITIONS")
    out.append("  " + rule("-"))
    out.append("")

    # ── Per-setup condition blocks ────────────────────────────────────────────
    # Each condition prints 🟢 (met) / 🔴 (not met) with the live value beside it.
    conds = []           # accumulates lines for the current setup
    def cond(ok, text, current, raw=False):
        """raw=True prints the parenthetical verbatim instead of 'Current = ...'."""
        paren = current if raw else f"Current = {current}"
        conds.append(f"     {'🟢' if ok else '🔴'} {text} ({paren})")

    def setup(num, title, all_ok, note=""):
        """Flush the accumulated conditions under a numbered heading."""
        tag = "  — ACTIVE" if all_ok else ""
        suffix = f"  {note}" if note else ""
        out.append(f"  {num}. {title}{tag}{suffix}")
        out.extend(conds)
        out.append("")
        conds.clear()

    def num(v, dp=2):
        return "-" if (isinstance(v, float) and np.isnan(v)) else f"{v:.{dp}f}"

    def dew_current(sig):
        """Display string for a 'DEW == <sig> (fresh this bar)' condition."""
        if dew_v == sig:
            # Fired today - fresh unless the engine already acted on it.
            if state["last_dew"] != sig:
                return f"{sig}, fresh this bar"
            return f"{sig} fired today, but already acted on"
        if prev_sig is None:
            return "no DEW signal yet"
        if prev_sig != sig:
            return f"{prev_sig} — last fired {prev_date}"
        return f"{prev_sig}, but not fresh — last fired {prev_date}"

    # 1. C/Up TQQQ
    cond(cc_v == "C/Up", "cc == C/Up", cc_v)
    cond(not np.isnan(trt) and trt < 1.40, "TQQQ_rt < 1.40", num(trt))
    cond(not np.isnan(bsr) and bsr > 1.05, "BSR > 1.05", num(bsr))
    cond(not np.isnan(tatr) and tatr < 7.0, "TQQQ_atr < 7.0", num(tatr))
    cond(not np.isnan(vix) and vix < 30, "VIX < 30", num(vix))
    cond(not state["cup_entered"], "C/Up leg not already traded",
         "already entered" if state["cup_entered"] else "not yet entered")
    setup(1, "C/Up TQQQ", pending.get("cup_f", False) and not state["cup_entered"])

    # 2. C/Dn TQQQ
    cond(cc_v == "C/Dn", "cc == C/Dn", cc_v)
    cond(dew_v == "Buy" and state["last_dew"] != "Buy",
         "DEW signal == Buy (fresh this bar)", dew_current("Buy"))
    cond(not np.isnan(rt) and 0.95 <= rt < 1.00, "0.95 ≤ RT < 1.00", num(rt, 3))
    cond(not np.isnan(mti) and mti < 1.00, "MTI < 1.00", num(mti))
    cond(not np.isnan(vix) and vix < 27, "VIX < 27", num(vix))
    setup(2, "C/Dn TQQQ", pending.get("cdn_f", False))

    # 3. QQQ long (DEW Buy leg)
    rt_blocked = not np.isnan(rt) and 0.85 <= rt < 0.90
    qqq_signal = (dew_v == "Buy" and state["last_dew"] != "Buy")
    cond(qqq_signal, "DEW signal == Buy (fresh this bar)", dew_current("Buy"))
    cond(not rt_blocked, "NOT (0.85 ≤ RT < 0.90)",
         f"RT {num(rt, 3)}" + (" — inside the block band" if rt_blocked else ""))
    cond(np.isnan(bsr) or bsr <= 1.50, "NOT (BSR > 1.50)", f"BSR {num(bsr)}")
    cond(np.isnan(mti) or mti <= 1.15, "NOT (MTI > 1.15)", f"MTI {num(mti)}")
    cond(np.isnan(vix) or vix <= 35, "VIX ≤ 35", num(vix))
    qqq_blocked = rt_blocked or (not np.isnan(bsr) and bsr > 1.50) \
        or (not np.isnan(mti) and mti > 1.15) or (not np.isnan(vix) and vix > 35)
    setup(3, "QQQ long", qqq_signal and not qqq_blocked)

    # 4. INV short (SQQQ 1/3 position)
    inv_blocked = ((not np.isnan(rt) and rt > 0.95) and (not np.isnan(mti) and mti > 0.95))
    inv_signal = (dew_v == "Sell" and state["last_dew"] != "Sell" and cc_v == "C/Dn")
    cond(dew_v == "Sell" and state["last_dew"] != "Sell",
         "DEW signal == Sell (fresh this bar)", dew_current("Sell"))
    cond(cc_v == "C/Dn", "cc == C/Dn", cc_v)
    cond(not inv_blocked, "NOT (RT > 0.95 AND MTI > 0.95)",
         f"RT {num(rt, 3)}, MTI {num(mti)}")
    cond(np.isnan(vix) or vix <= 35, "VIX ≤ 35", num(vix))
    setup(4, "INV short (SQQQ, 1/3 position)",
          inv_signal and not inv_blocked and not (not np.isnan(vix) and vix > 35))

    # 5. QQQ Cap (capitulation buy)
    cool_ok = pending.get("cap_cooldown_ok", False)
    last_exit_idx = state.get("last_exit_idx")
    if last_exit_idx is None:
        cool_cur = "no prior exit"
    else:
        cool_cur = f"{len(df) - 1 - last_exit_idx} bars since last exit"
    cond(not state["in_trade"], "account is flat",
         "flat" if not state["in_trade"] else f"in {state['instrument']}")
    cond(not np.isnan(d5) and d5 <= CAP_DROP5_THR,
         f"5-day QQQ drop ≤ {CAP_DROP5_THR*100:.0f}%",
         f"{d5*100:.1f}%" if not np.isnan(d5) else "-")
    cond(cool_ok, f"cooldown ≥ {CAP_COOLDOWN} bars", cool_cur)
    setup(5, "QQQ Cap",
          pending.get("cap_f", False) and cool_ok and not state["in_trade"])

    # 6. QQQ Mean-Rev - display only; dropped from the v17 engine as churn.
    ma20    = today.get("MA20", float("nan"))
    atr14   = today.get("ATR14", float("nan"))
    stretch = today.get("stretch", float("nan"))
    vol20   = today.get("vol20", float("nan"))
    if np.isnan(stretch):
        stretch_cur = "Current = -"
    else:
        stretch_cur = (f"stretch = (Close - MA20) / ATR14 = "
                       f"({num(qcl)} - {num(ma20)}) / {num(atr14)} = {num(stretch)}")
    cond(not state["in_trade"], "account is flat",
         "flat" if not state["in_trade"] else f"in {state['instrument']}")
    cond(not np.isnan(stretch) and stretch < STRETCH_THR,
         f"stretch < {STRETCH_THR:.1f}", stretch_cur, raw=True)
    cond(not np.isnan(vol20) and vol20 < VOL20_THR, f"vol20 < {VOL20_THR:.2f}", num(vol20, 3))
    cond(cool_ok, f"cooldown ≥ {CAP_COOLDOWN} bars", cool_cur)
    setup(6, "QQQ Mean-Rev", False, note="(display only — not traded in v17)")

    out.append("  " + rule("="))
    a = pending["action"]
    if a == "ENTER":
        inst = pending["entry_inst"]
        verb = "LONG" if inst != "INV" else "SHORT (via SQQQ 1/3)"
        out.append(f"  >  TOMORROW'S ACTION:  BUY {inst} at open  ({verb})")
    elif a == "EXIT":
        out.append(f"  >  TOMORROW'S ACTION:  SELL {state['instrument']} at open  ({pending['exit_type']})")
    elif a == "PREEMPT_1A":
        out.append(f"  >  TOMORROW'S ACTION:  PREEMPT 1a - SELL {state['instrument']}, BUY TQQQ (cup)")
    elif a == "PREEMPT_1B":
        out.append(f"  >  TOMORROW'S ACTION:  PREEMPT 1b - SELL QQQ, BUY TQQQ (C/Dn variant)")
    elif a == "BLOCKED":
        out.append(f"  >  TOMORROW'S ACTION:  HOLD FLAT  (entry blocked)")
    elif state["in_trade"]:
        if pending.get("decay_action") == "SCALE_DOWN":
            out.append(f"  >  TOMORROW'S ACTION:  SELL 70% of TQQQ at open  (decay activates -> 0.30x sizing)")
        elif pending.get("decay_action") == "SCALE_UP":
            out.append(f"  >  TOMORROW'S ACTION:  BUY 70% of TQQQ back at open  (decay clears -> full sizing)")
        else:
            out.append(f"  >  TOMORROW'S ACTION:  HOLD {state['instrument']}")
    else:
        out.append(f"  >  TOMORROW'S ACTION:  HOLD FLAT")

    out.append(f"     Reason: {pending['reason']}")
    out.append("  " + rule("="))

    out.append("")
    out.append(f"  LAST 5 COMPLETED TRADES (of {len(trades)} total)")
    out.append("  " + rule("-"))
    for t in trades[-5:]:
        out.append(f"    {str(t['entry_date'])[:10]}  {t['inst']:4s}  ->  "
                   f"{str(t['exit_date'])[:10]}  {t['ret']*100:+7.2f}%  [{t['exit']}]")

    out.append("")
    out.append(rule("="))
    body = "\n".join(out)
    print(body)
    return body, today_str


def build_subject(state, pending, today_str):
    a = pending["action"]
    if a == "ENTER":
        return f"Nitro++ v17 Signal {today_str}: BUY {pending['entry_inst']}"
    if a == "EXIT":
        return f"Nitro++ v17 Signal {today_str}: SELL {state['instrument']} ({pending.get('exit_type','exit')})"
    if a == "PREEMPT_1A":
        return f"Nitro++ v17 Signal {today_str}: PREEMPT 1a - SELL {state['instrument']} -> BUY TQQQ"
    if a == "PREEMPT_1B":
        return f"Nitro++ v17 Signal {today_str}: PREEMPT 1b - SELL QQQ -> BUY TQQQ"
    if a == "BLOCKED":
        return f"Nitro++ v17 Signal {today_str}: BLOCKED ({pending.get('entry_inst','?')})"
    if state["in_trade"]:
        if pending.get("decay_action") == "SCALE_DOWN":
            return f"Nitro++ v17 Signal {today_str}: SCALE DOWN TQQQ -> 0.30x"
        if pending.get("decay_action") == "SCALE_UP":
            return f"Nitro++ v17 Signal {today_str}: SCALE UP TQQQ -> full"
        inst = state["instrument"]
        if inst == "TQQQ":
            tag = "C/Dn TQQQ" if state.get("cdn_active") else "C/Up TQQQ"
            if state.get("pyramid_on"):
                tag += " [PYRAMID]"
            return f"Nitro++ v17 Signal {today_str}: HOLD {tag}"
        if inst == "QQQ" and state.get("cur_entry_kind") == "cap":
            return f"Nitro++ v17 Signal {today_str}: HOLD QQQ (Cap)"
        return f"Nitro++ v17 Signal {today_str}: HOLD {inst}"
    return f"Nitro++ v17 Signal {today_str}: FLAT"


def build_sms_summary(state, pending, today_str):
    short = today_str[2:]
    a = pending["action"]
    if a == "ENTER":
        inst = pending["entry_inst"]
        verb = "BUY" if inst != "INV" else "SHORT"
        tag = " (Cap)" if (inst == "QQQ" and "Cap" in pending.get("reason", "")) else ""
        msg = f"Nitro {short}: {verb} {inst}{tag} at open"
    elif a == "EXIT":
        msg = f"Nitro {short}: SELL {state['instrument']} at open ({pending.get('exit_type','')})".rstrip(" ()")
    elif a == "PREEMPT_1A":
        msg = f"Nitro {short}: PREEMPT1a SELL {state['instrument']}, BUY TQQQ (cup)"
    elif a == "PREEMPT_1B":
        msg = f"Nitro {short}: PREEMPT1b SELL QQQ, BUY TQQQ (C/Dn)"
    elif a == "BLOCKED":
        msg = f"Nitro {short}: BLOCKED {pending.get('entry_inst','')}".rstrip()
    elif state["in_trade"] and pending.get("decay_action") == "SCALE_DOWN":
        msg = f"Nitro {short}: SCALE DOWN TQQQ at open (decay -> 0.30x)"
    elif state["in_trade"] and pending.get("decay_action") == "SCALE_UP":
        msg = f"Nitro {short}: SCALE UP TQQQ at open (decay clears -> full)"
    elif state["in_trade"]:
        inst = state["instrument"]
        verb = "PYRAMID" if state.get("pyramid_on") else "HOLD"
        tag = " (Cap)" if (inst == "QQQ" and state.get("cur_entry_kind") == "cap") else ""
        msg = f"Nitro {short}: {verb} {inst}{tag}"
        tm = pending.get("today_mult")
        if tm is not None:
            pnl = (tm - 1.0) * 100.0
            msg += f"  {pnl:+.2f}%"
    else:
        msg = f"Nitro {short}: HOLD FLAT"
    return msg[:160]


def send_email(subject, body_text):
    if not GMAIL_PASS:
        print("send_email: GOOGLE_APP_PASSWORD not set - skipping")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(TO_EMAIL)
    # utf-8: the report body contains 🟢/🔴 status markers and ≤/≥ symbols.
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())


def send_imessage(numbers, body):
    for num in numbers:
        service_type = "SMS" if num in SMS_FORCE else "iMessage"
        script = (
            "on run argv\n"
            '  tell application "Messages"\n'
            f"    set svc to first service whose service type = {service_type}\n"
            f'    send (item 1 of argv) to participant "{num}" of svc\n'
            "  end tell\n"
            "end run"
        )
        try:
            subprocess.run(["osascript", "-e", script, body],
                           check=True, capture_output=True, timeout=30)
            print(f"  iMessage sent to {num} ({service_type})")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: iMessage to {num} timed out after 30s - not delivered")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="replace").strip() if e.stderr else "(no detail)"
            print(f"  WARNING: iMessage to {num} FAILED - not delivered: {err}")


def main():
    print("Loading data ...")
    df, df_full, qqq_raw = load_data()

    print("Computing DEW oscillator ...")
    dew_df = compute_dew(qqq_raw)
    df_full = df_full.merge(dew_df, on="Date", how="left")
    df_full["DEW_Signal"] = df_full["DEW_Signal"].fillna("").astype(str)
    df_full = add_cap_feature(df_full)
    # MA20 / vol20 / stretch - display only (Mean-Rev block); drop5 recomputed identically.
    df_full = add_v17_features(df_full)

    df_full = df_full[df_full["Date"] >= START_DATE].reset_index(drop=True)

    print(f"Running v17 production engine on {len(df_full)} bars "
          f"({str(df_full['Date'].iloc[0])[:10]} -> {str(df_full['Date'].iloc[-1])[:10]}) ...\n")
    trades, state = run_signal_engine(df_full)
    pending = evaluate_last_bar(df_full, state)
    body, today_str = print_report(df_full, state, pending, trades)

    subject = build_subject(state, pending, today_str)
    sms     = build_sms_summary(state, pending, today_str)

    if DRYRUN:
        print("\n[NITRO_DRYRUN set] - not sending email / iMessage.")
        print(f"Subject: {subject}")
        print(f"SMS: {sms}")
        return

    print(f"\nSending email to {TO_EMAIL} ...")
    send_email(subject, body)
    print("Email sent.")
    print(f"Sending iMessage/SMS to {SMS_NUMBERS}: {sms}")
    send_imessage(SMS_NUMBERS, sms)


if __name__ == "__main__":
    main()
