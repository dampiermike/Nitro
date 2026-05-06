"""
Dampier Nitro++ v13 — Daily Signal Generator
=============================================
Loads all data, replays the full production engine (hf + pyramid),
and reports:
  • Current open position (if any)
  • All market indicators for today's close
  • Pending action for tomorrow's open
"""

import os
import sys
import smtplib
import subprocess
import importlib.util
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Email / SMS config ─────────────────────────────────────────────────────────
GMAIL_USER  = os.environ.get('GOOGLE_EMAIL', 'dampiermike@gmail.com')
GMAIL_PASS  = os.environ.get('GOOGLE_APP_PASSWORD', '')
TO_EMAIL    = ['dampiermike@gmail.com', 'ddampier777@gmail.com', 'brooke.hoover@yahoo.com']
SMS_NUMBERS = ['+12256144680', '+13038818222', '+18137815601']
# Numbers that must be sent via SMS (Continuity relay through paired iPhone)
# rather than iMessage — e.g. Android/Verizon recipients where iMessage bounces.
SMS_FORCE   = {'+18137815601'}

# ── Load backtest module via importlib (++ in filename) ────────────────────────
_base = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "nitro_v13",
    os.path.join(_base, "dampier_nitro++_v13.py"),
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

load_data   = _m.load_data
compute_dew = _m.compute_dew
arr         = _m.arr


# ──────────────────────────────────────────────────────────────────────────────
# Signal Engine — identical to _run_engine(use_hf_pyr=True) but returns
# the full engine state after the loop so we can evaluate today's bar.
# ──────────────────────────────────────────────────────────────────────────────
def run_signal_engine(df):
    """Run production engine on all data. Returns (trades, final_state, arrays)."""
    a = arr(df)
    N = a["N"]
    dret   = np.zeros(N)
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
    hf          = 1.0

    for i in range(N - 1):
        fell_through = False
        flip_blocked = None

        # ── Health factor ────────────────────────────────────────────────────
        vix_i  = a["VIX"][i]
        tatr_i = a["tatr"][i]
        bsr_i  = a["BSR"][i]
        if np.isnan(vix_i) or np.isnan(tatr_i):
            hf = 1.0
        else:
            hf = min(1.0, max(0.25,
                (30.0 - vix_i) / 20.0 * (8.0 / max(tatr_i, 4.0)) * max(bsr_i, 0.5)))

        # ── Phase A: CC transitions ──────────────────────────────────────────
        if i > 0:
            if a["Trend"][i] == "C/Up" and a["cc"][i-1] != "C/Up":
                last_dew = None; cup_entered = False
            if a["Trend"][i] == "C/Dn" and a["cc"][i-1] != "C/Dn":
                last_dew = None

        # ── Signal flags ─────────────────────────────────────────────────────
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

        # ── Priority 1a: C/Up TQQQ preempts open QQQ or PSQ ─────────────────
        if in_trade and instrument in ("QQQ", "PSQ") and cup_f and not cup_entered:
            d = (a["o2c_Q"][i] if (instrument == "QQQ" and i == entry_idx)
                 else a["c2c_Q"][i] if instrument == "QQQ"
                 else a["o2c_P"][i] if i == entry_idx
                 else a["c2c_P"][i])
            cum_mult *= (1.0 + d)
            pc2o     = a["pc2o_Q"][i+1] if instrument == "QQQ" else a["pc2o_P"][i+1]
            raw_exit = (1.0 + d) * (1.0 + pc2o) - 1.0
            dret[i]  = raw_exit * hf
            trade_ret = cum_mult * (1.0 + pc2o) - 1.0
            trades.append(dict(entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                               inst=instrument, ret=trade_ret, exit="Preempt"))
            in_trade = True; instrument = "TQQQ"; cdn_active = False
            entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
            cup_entered = True; stop_qqq = None
            tqqq_target = a["topen"][i+1] * 1.50
            tqqq_stop   = a["topen"][i+1] * 0.94
            continue

        # ── Priority 1b: C/Dn switch while in QQQ ───────────────────────────
        if (in_trade and instrument == "QQQ" and not cdn_active
                and a["cc"][i] == "C/Dn"
                and 0.95 <= a["RT"][i] < 1.00
                and a["MTI"][i] < 0.95):
            d         = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
            cum_mult *= (1.0 + d)
            pc2o_1b   = a["pc2o_Q"][i+1]
            raw_exit_1b = (1.0 + d) * (1.0 + pc2o_1b) - 1.0
            dret[i]   = raw_exit_1b * hf
            trade_ret = cum_mult * (1.0 + pc2o_1b) - 1.0
            trades.append(dict(entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                               inst="QQQ", ret=trade_ret, exit="Preempt"))
            in_trade = True; instrument = "TQQQ"; cdn_active = True
            entry_idx = i + 1; cum_mult = 1.0; pyramid_on = False
            stop_qqq = None; last_dew = None
            tqqq_target = a["topen"][i+1] * 1.50
            tqqq_stop   = a["topen"][i+1] * 0.94
            continue

        # ── Phase B: exits ───────────────────────────────────────────────────
        if in_trade:
            if instrument == "TQQQ":
                d = a["to2c"][i] if i == entry_idx else a["tc2c"][i]
            elif instrument == "QQQ":
                d = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
            else:
                d = a["o2c_P"][i] if i == entry_idx else a["c2c_P"][i]
            cum_mult *= (1.0 + d)

            if instrument == "TQQQ" and cum_mult - 1.0 >= 0.20 and not pyramid_on:
                pyramid_on = True
            pyr   = 1.25 if (instrument == "TQQQ" and pyramid_on) else 1.0
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
                if cdn_active and (a["DEW"][i] == "Sell"
                                   or (a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn")):
                    exit_type = "Regime"; exited = True
                elif not cdn_active and a["cc"][i] == "C/Dn":
                    exit_type = "Regime"; exited = True

            if not exited and instrument == "QQQ" and i != entry_idx:
                if a["Trend"][i] == "C/Dn" and a["cc"][i-1] == "C/Up":
                    fell_through = True; flip_blocked = "PSQ"
                    exit_type = "Flip"; exited = True; is_fall_through = True

            if not exited and instrument == "PSQ" and i != entry_idx:
                if a["Trend"][i] == "C/Up" and a["cc"][i-1] == "C/Dn":
                    fell_through = True; flip_blocked = "QQQ"
                    exit_type = "Flip"; exited = True; is_fall_through = True

            if not exited and instrument == "QQQ" and a["DEW"][i] == "Sell":
                last_dew = "Sell"; fell_through = True
                exit_type = "DEW"; exited = True; is_fall_through = True

            if not exited and instrument == "PSQ" and a["DEW"][i] == "Buy":
                last_dew = "Buy"; fell_through = True
                exit_type = "DEW"; exited = True; is_fall_through = True

            if exited:
                pc2o = (a["tpc2o"][i+1] if instrument == "TQQQ"
                        else a["pc2o_Q"][i+1] if instrument == "QQQ"
                        else a["pc2o_P"][i+1])
                raw_exit  = (1.0 + d) * (1.0 + pc2o) - 1.0
                dret[i]   = raw_exit * hf * pyr
                trade_ret = cum_mult * (1.0 + pc2o) - 1.0
                trades.append(dict(entry_date=a["dates"][entry_idx], exit_date=a["dates"][i+1],
                                   inst=instrument, ret=trade_ret, exit=exit_type))
                is_tqqq   = (instrument == "TQQQ")
                in_trade  = False; instrument = None; entry_idx = None
                cum_mult  = 1.0; stop_qqq = None; tqqq_target = None
                tqqq_stop = None; pyramid_on = False
                if is_tqqq:
                    last_dew = None; cdn_active = False
                if not is_fall_through:
                    continue
            else:
                dret[i] = raw_d
                continue

        # ── Phase C: entries ─────────────────────────────────────────────────
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
                rt = a["RT"][i]; bsr = a["BSR"][i]; mti = a["MTI"][i]
                blk = ((not np.isnan(rt)  and 0.85 <= rt  < 0.90) or
                       (not np.isnan(bsr) and bsr > 1.50) or
                       (not np.isnan(mti) and mti > 1.10))
                if not blk:
                    in_trade  = True; instrument = "QQQ"
                    entry_idx = i + 1; cum_mult = 1.0; last_dew = "Buy"
                    atr_mult  = 2.0 if a["cc"][i] == "C/Dn" else 1.0
                    stop_qqq  = a["Open"][i+1] - atr_mult * a["ATR14"][i]
                continue
            if (a["DEW"][i] == "Sell" and last_dew != "Sell"
                    and a["cc"][i] == "C/Dn" and flip_blocked != "PSQ"):
                rt = a["RT"][i]; mti = a["MTI"][i]
                blk = (not np.isnan(rt)  and rt  > 0.95
                       and not np.isnan(mti) and mti > 0.95)
                if not blk:
                    in_trade  = True; instrument = "PSQ"
                    entry_idx = i + 1; cum_mult = 1.0; last_dew = "Sell"
                    stop_qqq  = a["Open"][i+1] + 2.0 * a["ATR14"][i]

    # ── Capture final state ──────────────────────────────────────────────────
    state = dict(
        in_trade    = in_trade,
        instrument  = instrument,
        entry_idx   = entry_idx,
        entry_date  = (str(a["dates"][entry_idx])[:10]
                       if (in_trade and entry_idx is not None) else None),
        cum_mult    = cum_mult,
        stop_qqq    = stop_qqq,
        tqqq_target = tqqq_target,
        tqqq_stop   = tqqq_stop,
        last_dew    = last_dew,
        cup_entered = cup_entered,
        cdn_active  = cdn_active,
        pyramid_on  = pyramid_on,
        hf          = hf,
    )
    return trades, state, a, N


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate Today's Bar (index N-1) for Tomorrow's Action
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_last_bar(a, N, state):
    """
    Run Phase B + Phase C logic on bar N-1 (today's close data).
    Returns a dict describing tomorrow's pending action.
    """
    i = N - 1   # today's index
    in_trade    = state["in_trade"]
    instrument  = state["instrument"]
    entry_idx   = state["entry_idx"]
    cum_mult    = state["cum_mult"]
    stop_qqq    = state["stop_qqq"]
    tqqq_target = state["tqqq_target"]
    tqqq_stop   = state["tqqq_stop"]
    last_dew    = state["last_dew"]
    cup_entered = state["cup_entered"]
    cdn_active  = state["cdn_active"]
    pyramid_on  = state["pyramid_on"]

    # Health factor for today
    vix_i  = a["VIX"][i]
    tatr_i = a["tatr"][i]
    bsr_i  = a["BSR"][i]
    if np.isnan(vix_i) or np.isnan(tatr_i):
        hf = 1.0
    else:
        hf = min(1.0, max(0.25,
            (30.0 - vix_i) / 20.0 * (8.0 / max(tatr_i, 4.0)) * max(bsr_i, 0.5)))

    # Today's signal flags
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

    pending = dict(action="HOLD", reason="", exit_type=None, entry_inst=None,
                   entry_blocked=False, block_reasons=[], hf=hf)

    # ── In a trade: check exits ───────────────────────────────────────────────
    if in_trade:
        # Accrue today's return to see if TP/Stop triggers
        if instrument == "TQQQ":
            d = a["to2c"][i] if i == entry_idx else a["tc2c"][i]
        elif instrument == "QQQ":
            d = a["o2c_Q"][i] if i == entry_idx else a["c2c_Q"][i]
        else:
            d = a["o2c_P"][i] if i == entry_idx else a["c2c_P"][i]
        today_mult = cum_mult * (1.0 + d)

        if instrument == "TQQQ" and i != entry_idx and a["MTI"][i] < 0.75:
            pending.update(action="EXIT", exit_type="MTI",
                           reason=f"MTI={a['MTI'][i]:.2f} < 0.75 — exit TQQQ at tomorrow's open")

        elif instrument == "TQQQ" and a["tclose"][i] >= tqqq_target:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"TQQQ close {a['tclose'][i]:.2f} ≥ target {tqqq_target:.2f} — exit at tomorrow's open")

        elif instrument == "TQQQ" and a["tclose"][i] <= tqqq_stop:
            pending.update(action="EXIT", exit_type="Stop",
                           reason=f"TQQQ close {a['tclose'][i]:.2f} ≤ stop {tqqq_stop:.2f} — exit at tomorrow's open")

        elif instrument == "QQQ" and today_mult - 1.0 >= 0.05:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"QQQ cumulative {(today_mult-1)*100:.2f}% ≥ 5% target — exit at tomorrow's open")

        elif instrument == "QQQ" and a["Close"][i] <= stop_qqq:
            pending.update(action="EXIT", exit_type="ATR",
                           reason=f"QQQ close {a['Close'][i]:.2f} ≤ ATR stop {stop_qqq:.2f} — exit at tomorrow's open")

        elif instrument == "PSQ" and today_mult - 1.0 >= 0.05:
            pending.update(action="EXIT", exit_type="TP",
                           reason=f"PSQ cumulative {(today_mult-1)*100:.2f}% ≥ 5% target — exit at tomorrow's open")

        elif instrument == "PSQ" and a["Close"][i] >= stop_qqq:
            pending.update(action="EXIT", exit_type="ATR",
                           reason=f"QQQ close {a['Close'][i]:.2f} ≥ ATR stop {stop_qqq:.2f} — exit PSQ at tomorrow's open")

        elif instrument == "TQQQ":
            if cdn_active and (a["DEW"][i] == "Sell"
                               or (a["Trend"][i] == "C/Up" and i > 0 and a["cc"][i-1] == "C/Dn")):
                pending.update(action="EXIT", exit_type="Regime",
                               reason="C/Dn TQQQ: DEW Sell or C/Up flip — exit at tomorrow's open")
            elif not cdn_active and a["cc"][i] == "C/Dn":
                pending.update(action="EXIT", exit_type="Regime",
                               reason="cc flipped to C/Dn — exit TQQQ at tomorrow's open")

        elif instrument == "QQQ" and i != entry_idx:
            if a["Trend"][i] == "C/Dn" and i > 0 and a["cc"][i-1] == "C/Up":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Dn flip while long QQQ — exit at tomorrow's open")
            elif a["DEW"][i] == "Sell":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Sell while long QQQ — exit at tomorrow's open")

        elif instrument == "PSQ" and i != entry_idx:
            if a["Trend"][i] == "C/Up" and i > 0 and a["cc"][i-1] == "C/Dn":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Up flip while short PSQ — exit at tomorrow's open")
            elif a["DEW"][i] == "Buy":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Buy while short PSQ — exit at tomorrow's open")

        if pending["action"] == "HOLD":
            pending["reason"] = f"No exit condition met. Holding {instrument}."

        # Carry today's updated mult into pending for display
        pending["today_mult"] = today_mult

    # ── Not in a trade: check entries ────────────────────────────────────────
    else:
        pending["today_mult"] = None

        # Priority 1a: C/Up TQQQ
        if cup_f and not cup_entered:
            pending.update(action="ENTER", entry_inst="TQQQ",
                           reason="C/Up TQQQ (cup_f) — BUY TQQQ at tomorrow's open")

        # Priority 2: C/Dn TQQQ
        elif cdn_f:
            pending.update(action="ENTER", entry_inst="TQQQ",
                           reason="C/Dn TQQQ (cdn_f) — BUY TQQQ at tomorrow's open")

        # DEW Buy → QQQ
        elif a["DEW"][i] == "Buy" and last_dew != "Buy":
            rt = a["RT"][i]; bsr = a["BSR"][i]; mti = a["MTI"][i]
            reasons = []
            if not np.isnan(rt)  and 0.85 <= rt  < 0.90: reasons.append(f"RT={rt:.2f} in [0.85,0.90)")
            if not np.isnan(bsr) and bsr > 1.50:          reasons.append(f"BSR={bsr:.2f} > 1.50")
            if not np.isnan(mti) and mti > 1.10:          reasons.append(f"MTI={mti:.2f} > 1.10")
            if reasons:
                pending.update(action="BLOCKED", entry_inst="QQQ",
                               entry_blocked=True, block_reasons=reasons,
                               reason=f"DEW Buy → QQQ blocked: {', '.join(reasons)}")
            else:
                pending.update(action="ENTER", entry_inst="QQQ",
                               reason="DEW Buy — BUY QQQ at tomorrow's open")

        # DEW Sell → PSQ
        elif a["DEW"][i] == "Sell" and last_dew != "Sell" and a["cc"][i] == "C/Dn":
            rt = a["RT"][i]; mti = a["MTI"][i]
            blk = (not np.isnan(rt) and rt > 0.95 and not np.isnan(mti) and mti > 0.95)
            if blk:
                pending.update(action="BLOCKED", entry_inst="PSQ",
                               entry_blocked=True,
                               block_reasons=[f"RT={rt:.2f}>0.95 AND MTI={mti:.2f}>0.95"],
                               reason=f"DEW Sell → PSQ blocked: RT={rt:.2f} AND MTI={mti:.2f} both > 0.95")
            else:
                pending.update(action="ENTER", entry_inst="PSQ",
                               reason="DEW Sell in C/Dn — BUY PSQ at tomorrow's open")

        else:
            pending["reason"] = "No signal. Flat."

    return pending


# ──────────────────────────────────────────────────────────────────────────────
# Report Formatter
# ──────────────────────────────────────────────────────────────────────────────
def print_report(df_full, state, pending, a, N, trades):
    today_row = df_full.iloc[-1]
    today_str = str(today_row["Date"])[:10]

    # Find entry price and bars held for open trade
    entry_price = None
    bars_held   = 0
    if state["in_trade"] and state["entry_idx"] is not None:
        eidx = state["entry_idx"]
        inst = state["instrument"]
        entry_price = (a["topen"][eidx] if inst == "TQQQ"
                       else a["Open"][eidx] if inst == "QQQ"
                       else a["Open"][eidx])   # PSQ uses QQQ open
        bars_held = N - 1 - eidx   # bars processed after entry (doesn't include today)

    W = 66
    def rule(ch="─"): return ch * W
    def hdr(txt, ch="═"):
        pad = (W - len(txt) - 2) // 2
        return ch * pad + f" {txt} " + ch * (W - pad - len(txt) - 2)

    lines = []
    lines.append(rule("═"))
    lines.append(hdr(f"DAMPIER NITRO++ v13  ·  DAILY SIGNAL  ·  {today_str}"))
    lines.append(rule("═"))

    # ── Current position ─────────────────────────────────────────────────────
    lines.append("")
    if state["in_trade"]:
        inst = state["instrument"]
        pnl  = (state["cum_mult"] - 1.0) * 100.0
        pnl_today = ((pending.get("today_mult") or state["cum_mult"]) - 1.0) * 100.0
        lines.append(f"  CURRENT POSITION :  {'LONG' if inst != 'PSQ' else 'SHORT'} {inst}")
        lines.append(f"  Entry Date       :  {state['entry_date']}")
        if entry_price:
            lines.append(f"  Entry Price      :  ${entry_price:.2f}  ({inst})")
        lines.append(f"  Bars Held        :  {bars_held}")
        lines.append(f"  Cum Return       :  {pnl:+.2f}% (thru yesterday)  /  {pnl_today:+.2f}% (thru today close)")

        if state["pyramid_on"]:
            lines.append(f"  Pyramid          :  ACTIVE (×1.25)")

        if inst == "TQQQ":
            lines.append(f"  TQQQ Target      :  ${state['tqqq_target']:.2f}  |  Stop: ${state['tqqq_stop']:.2f}")
        elif inst in ("QQQ", "PSQ"):
            stop_label = "ATR Stop (QQQ close)" if inst == "QQQ" else "ATR Stop (QQQ close)"
            lines.append(f"  {stop_label:<18} :  ${state['stop_qqq']:.2f}")
    else:
        lines.append("  CURRENT POSITION :  FLAT")

    # ── Market indicators ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("  " + rule("─"))
    lines.append(f"  MARKET INDICATORS  (as of {today_str} close)")
    lines.append("  " + rule("─"))

    cc    = today_row.get("cc", "—")
    dew   = today_row.get("DEW_Signal", "—") or "—"
    rt    = today_row.get("RT",   float("nan"))
    mti   = today_row.get("MTI",  float("nan"))
    bsr   = today_row.get("BSR",  float("nan"))
    vix   = today_row.get("VIX",  float("nan"))
    tatr  = today_row.get("tqqq_atr", float("nan"))
    trt   = today_row.get("tqqq_rt",  float("nan"))
    tclose= today_row.get("tqqq_close", float("nan"))

    lines.append(f"  Confirmed Call   :  {cc}")
    lines.append(f"  DEW Signal       :  {dew}")
    lines.append(f"  QQQ RT           :  {rt:.2f}"     if not np.isnan(rt)   else "  QQQ RT           :  —")
    lines.append(f"  MTI              :  {mti:.2f}"    if not np.isnan(mti)  else "  MTI              :  —")
    lines.append(f"  BSR              :  {bsr:.2f}"    if not np.isnan(bsr)  else "  BSR              :  —")
    lines.append(f"  VIX              :  {vix:.2f}"    if not np.isnan(vix)  else "  VIX              :  —")
    lines.append(f"  TQQQ ATR%        :  {tatr:.2f}%"  if not np.isnan(tatr) else "  TQQQ ATR%        :  —")
    lines.append(f"  TQQQ RT          :  {trt:.4f}"    if not np.isnan(trt)  else "  TQQQ RT          :  —")
    lines.append(f"  TQQQ Close       :  ${tclose:.2f}" if not np.isnan(tclose) else "  TQQQ Close       :  —")
    lines.append(f"  Health Factor    :  {pending['hf']:.3f}")

    # ── cup_f / cdn_f breakdown ───────────────────────────────────────────────
    lines.append("")
    lines.append("  " + rule("─"))
    lines.append("  ENTRY SIGNAL CONDITIONS")
    lines.append("  " + rule("─"))

    def yn(cond): return "✓" if cond else "✗"

    cup_f_parts = [
        f"cc=C/Up {yn(cc=='C/Up')}",
        f"TQQQ_RT<1.40 {yn(not np.isnan(trt) and trt<1.40)}",
        f"BSR>1.05 {yn(not np.isnan(bsr) and bsr>1.05)}",
        f"TQQQ_ATR<7.0 {yn(not np.isnan(tatr) and tatr<7.0)}",
        f"VIX<30 {yn(not np.isnan(vix) and vix<30)}",
        f"cup_entered={state['cup_entered']}",
    ]
    cup_all = (cc == "C/Up"
               and not np.isnan(trt)  and trt  < 1.40
               and not np.isnan(bsr)  and bsr  > 1.05
               and not np.isnan(tatr) and tatr < 7.0
               and not np.isnan(vix)  and vix  < 30.0
               and not state["cup_entered"])
    lines.append(f"  C/Up TQQQ (cup_f):  {'ACTIVE ✓' if cup_all else 'inactive'}  —  {', '.join(cup_f_parts)}")

    cdn_f_parts = [
        f"cc=C/Dn {yn(cc=='C/Dn')}",
        f"DEW=Buy {yn(dew=='Buy')}",
        f"last_dew≠Buy {yn(state['last_dew']!='Buy')}",
        f"RT∈[0.95,1.00) {yn(not np.isnan(rt) and 0.95<=rt<1.00)}",
        f"MTI<1.00 {yn(not np.isnan(mti) and mti<1.00)}",
    ]
    cdn_all = (cc == "C/Dn"
               and dew == "Buy"
               and state["last_dew"] != "Buy"
               and not np.isnan(rt)  and 0.95 <= rt  < 1.00
               and not np.isnan(mti) and mti < 1.00)
    lines.append(f"  C/Dn TQQQ (cdn_f):  {'ACTIVE ✓' if cdn_all else 'inactive'}  —  {', '.join(cdn_f_parts)}")

    # DEW Buy → QQQ
    qqq_blk_reasons = []
    if not np.isnan(rt)  and 0.85 <= rt  < 0.90: qqq_blk_reasons.append(f"RT={rt:.2f}∈[0.85,0.90)")
    if not np.isnan(bsr) and bsr > 1.50:          qqq_blk_reasons.append(f"BSR={bsr:.2f}>1.50")
    if not np.isnan(mti) and mti > 1.10:          qqq_blk_reasons.append(f"MTI={mti:.2f}>1.10")
    qqq_entry_ok = (dew == "Buy" and state["last_dew"] != "Buy" and not qqq_blk_reasons)
    qqq_status = "ACTIVE ✓" if qqq_entry_ok else ("DEW Buy BLOCKED: " + ", ".join(qqq_blk_reasons) if (dew == "Buy" and state["last_dew"] != "Buy") else "inactive")
    lines.append(f"  DEW Buy → QQQ   :  {qqq_status}  —  DEW={dew}, last_dew={state['last_dew']}")

    psq_blk = (not np.isnan(rt) and rt > 0.95 and not np.isnan(mti) and mti > 0.95)
    psq_entry_ok = (dew == "Sell" and state["last_dew"] != "Sell" and cc == "C/Dn" and not psq_blk)
    psq_status = "ACTIVE ✓" if psq_entry_ok else ("DEW Sell BLOCKED: RT>0.95 AND MTI>0.95" if (dew == "Sell" and psq_blk) else "inactive")
    lines.append(f"  DEW Sell → PSQ  :  {psq_status}  —  DEW={dew}, cc={cc}, last_dew={state['last_dew']}")

    # ── Tomorrow's action ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("  " + rule("═"))
    action = pending["action"]
    if action == "ENTER":
        inst = pending["entry_inst"]
        verb = "LONG" if inst != "PSQ" else "SHORT"
        lines.append(f"  ▶  TOMORROW'S ACTION:  BUY {inst} at open  ({verb})")
    elif action == "EXIT":
        lines.append(f"  ▶  TOMORROW'S ACTION:  SELL {state['instrument']} at open  ({pending['exit_type']})")
    elif action == "BLOCKED":
        lines.append(f"  ▶  TOMORROW'S ACTION:  HOLD FLAT  (entry blocked)")
    elif state["in_trade"]:
        lines.append(f"  ▶  TOMORROW'S ACTION:  HOLD {state['instrument']}")
    else:
        lines.append(f"  ▶  TOMORROW'S ACTION:  HOLD FLAT")

    lines.append(f"     Reason: {pending['reason']}")
    lines.append("  " + rule("═"))

    # ── Recent trades ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  LAST 5 COMPLETED TRADES")
    lines.append("  " + rule("─"))
    for t in trades[-5:]:
        sign = "+" if t["ret"] >= 0 else ""
        lines.append(f"    {str(t['entry_date'])[:10]}  {t['inst']:4s}  →  "
                      f"{str(t['exit_date'])[:10]}  {sign}{t['ret']*100:.2f}%  [{t['exit']}]")

    lines.append("")
    lines.append(rule("═"))
    body = "\n".join(lines)
    print(body)
    return body, today_str


# ──────────────────────────────────────────────────────────────────────────────
# Notifications
# ──────────────────────────────────────────────────────────────────────────────
def build_subject(state, pending, today_str):
    action = pending["action"]
    if action == "ENTER":
        return f"Nitro++ v13 Signal {today_str}: BUY {pending['entry_inst']}"
    if action == "EXIT":
        return f"Nitro++ v13 Signal {today_str}: SELL {state['instrument']} ({pending.get('exit_type','exit')})"
    if action == "BLOCKED":
        return f"Nitro++ v13 Signal {today_str}: BLOCKED ({pending.get('entry_inst','?')})"
    if state["in_trade"]:
        inst = state["instrument"]
        if inst == "TQQQ":
            tag = "C/Dn TQQQ" if state.get("cdn_active") else "C/Up TQQQ"
            if state.get("pyramid_on"):
                tag += " [PYRAMID]"
            return f"Nitro++ v13 Signal {today_str}: HOLD {tag}"
        return f"Nitro++ v13 Signal {today_str}: HOLD {inst}"
    return f"Nitro++ v13 Signal {today_str}: FLAT"


def build_sms_summary(state, pending, today_str):
    short = today_str[2:]  # YY-MM-DD
    action = pending["action"]
    if action == "ENTER":
        inst = pending["entry_inst"]
        verb = "BUY" if inst != "PSQ" else "SHORT"
        msg = f"Nitro {short}: {verb} {inst} at open"
    elif action == "EXIT":
        msg = f"Nitro {short}: SELL {state['instrument']} at open ({pending.get('exit_type','')})".rstrip(" ()")
    elif action == "BLOCKED":
        msg = f"Nitro {short}: BLOCKED {pending.get('entry_inst','')}".rstrip()
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
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = ', '.join(TO_EMAIL)
    msg.attach(MIMEText(body_text, 'plain'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())


def send_imessage(numbers, body):
    safe = body.replace('\\', '\\\\').replace('"', '\\"')
    for num in numbers:
        service_type = 'SMS' if num in SMS_FORCE else 'iMessage'
        script = (
            'tell application "Messages"\n'
            f'  set svc to first service whose service type = {service_type}\n'
            f'  send "{safe}" to participant "{num}" of svc\n'
            'end tell'
        )
        try:
            subprocess.run(['osascript', '-e', script], check=False, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"  warning: osascript send to {num} timed out after 30s — continuing")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    df, df_full, qqq_raw = load_data()

    print("Computing DEW oscillator …")
    dew_df      = compute_dew(qqq_raw)
    df_full     = df_full.merge(dew_df, on="Date", how="left")
    df_full["DEW_Signal"] = df_full["DEW_Signal"].fillna("").astype(str)

    print("Running production engine …")
    trades, state, a, N = run_signal_engine(df_full)

    print("Evaluating today's bar …\n")
    pending = evaluate_last_bar(a, N, state)

    body, today_str = print_report(df_full, state, pending, a, N, trades)

    subject = build_subject(state, pending, today_str)
    sms     = build_sms_summary(state, pending, today_str)

    print(f"\nSending email to {TO_EMAIL} ...")
    send_email(subject, body)
    print("Email sent.")
    send_imessage(SMS_NUMBERS, sms)
    print(f"iMessage sent to {SMS_NUMBERS}: {sms}")


if __name__ == "__main__":
    main()
