#!/usr/bin/env python3
"""
Dampier Nitro++ Daily Signal Generator
Runs after market close. Reads the latest history CSVs, replays the full
Nitro++ v8 engine through today, then emails trade instructions.

Usage:
    python3 nitro_daily_signal.py
"""

import os
import sys
import smtplib
import numpy as np
import pandas as pd
from collections import Counter
from datetime import date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ.get('GOOGLE_EMAIL', 'dampiermike@gmail.com')
GMAIL_PASS = os.environ.get('GOOGLE_APP_PASSWORD', '')
TO_EMAIL   = ['dampiermike@gmail.com', 'dampier777@gmail.com']

NITRO_DIR = Path(__file__).resolve().parent
DATA_DIR  = NITRO_DIR / 'data' / 'csv' / 'history'

QQQ_FILE  = DATA_DIR / 'qqq-from-vv.csv'
TQQQ_FILE = DATA_DIR / 'tqqq-from-vv.csv'
SYN_FILE  = DATA_DIR / 'synthetic-tqqq-ohlc-1999-2010.csv'
VV_FILE   = DATA_DIR / 'vectorvest-views-w3place-precision.csv'
VIX_FILE  = DATA_DIR / 'vix-from-yahoo.csv'

# Starting capital and live-trading start date
STARTING_CAPITAL = 100_000.0
LIVE_START       = pd.Timestamp('2026-04-07')   # first Monday we can act on a signal

# ── Load Data (same as v8 engine) ─────────────────────────────────────────────

def load_data():
    qqq_raw = pd.read_csv(QQQ_FILE, encoding='utf-8-sig')
    qqq_raw['Date'] = pd.to_datetime(qqq_raw['Date'])
    qqq_raw = qqq_raw.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    qqq_raw['c2c_QQQ']  = qqq_raw['Close'].pct_change()
    qqq_raw['o2c_QQQ']  = qqq_raw['Close'] / qqq_raw['Open'] - 1
    qqq_raw['pc2o_QQQ'] = qqq_raw['Open'] / qqq_raw['Close'].shift(1) - 1
    qqq_raw['o2c_PSQ']  = -1 * qqq_raw['o2c_QQQ']
    qqq_raw['pc2o_PSQ'] = -1 * qqq_raw['pc2o_QQQ']
    qqq_raw['c2c_PSQ']  = -1 * qqq_raw['c2c_QQQ'] - 0.0095 / 252
    qqq_full = qqq_raw.copy()
    qqq = qqq_raw[qqq_raw['Date'] >= '2000-01-01'].copy()

    syn = pd.read_csv(SYN_FILE)
    syn['Date'] = pd.to_datetime(syn['Date'])
    syn = syn.sort_values('Date').reset_index(drop=True)
    syn = syn.rename(columns={'Open':'tqqq_open','High':'tqqq_high','Low':'tqqq_low',
                               'Close':'tqqq_close','RT_v6':'tqqq_rt'})
    syn = syn[['Date','tqqq_open','tqqq_high','tqqq_low','tqqq_close','tqqq_rt']]

    tqqq_real = pd.read_csv(TQQQ_FILE, encoding='utf-8-sig')
    tqqq_real['Date'] = pd.to_datetime(tqqq_real['Date'])
    tqqq_real = tqqq_real.sort_values('Date').reset_index(drop=True)
    tqqq_real = tqqq_real.rename(columns={'Open':'tqqq_open','High':'tqqq_high','Low':'tqqq_low',
                                           'Close':'tqqq_close','RT':'tqqq_rt'})
    tqqq_real = tqqq_real[['Date','tqqq_open','tqqq_high','tqqq_low','tqqq_close','tqqq_rt']]

    real_start = tqqq_real['Date'].min()
    tqqq_stitched = pd.concat([syn[syn['Date'] < real_start], tqqq_real],
                               ignore_index=True).sort_values('Date').reset_index(drop=True)
    tqqq_stitched['tqqq_c2c']  = tqqq_stitched['tqqq_close'].pct_change()
    tqqq_stitched['tqqq_o2c']  = tqqq_stitched['tqqq_close'] / tqqq_stitched['tqqq_open'] - 1
    tqqq_stitched['tqqq_pc2o'] = tqqq_stitched['tqqq_open'] / tqqq_stitched['tqqq_close'].shift(1) - 1
    tr_t = pd.concat([
        tqqq_stitched['tqqq_high'] - tqqq_stitched['tqqq_low'],
        (tqqq_stitched['tqqq_high'] - tqqq_stitched['tqqq_close'].shift(1)).abs(),
        (tqqq_stitched['tqqq_low']  - tqqq_stitched['tqqq_close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    tqqq_stitched['tqqq_atr'] = tr_t.rolling(10).mean() / tqqq_stitched['tqqq_close'] * 100
    tqqq = tqqq_stitched[tqqq_stitched['Date'] >= '2000-01-01'].copy()

    vv = pd.read_csv(VV_FILE)
    vv['Date'] = pd.to_datetime(vv['Date'])
    vv = vv.sort_values('Date').reset_index(drop=True)
    vv = vv.rename(columns={'VVC-RT':'RT','BS Ratio':'BSR'})
    vv = vv[['Date','Trend','RT','MTI','BSR']]

    vix = pd.read_csv(VIX_FILE)
    vix['Date'] = pd.to_datetime(vix['Date'], errors='coerce')
    vix = vix.dropna(subset=['Date'])
    vix = vix[['Date','Close']].rename(columns={'Close':'VIX'})
    vix['VIX'] = pd.to_numeric(vix['VIX'], errors='coerce')

    df = vv.merge(qqq[['Date','Open','High','Low','Close',
                        'o2c_QQQ','c2c_QQQ','pc2o_QQQ',
                        'o2c_PSQ','c2c_PSQ','pc2o_PSQ']], on='Date', how='inner')
    df = df.merge(tqqq[['Date','tqqq_open','tqqq_close','tqqq_rt',
                         'tqqq_atr','tqqq_o2c','tqqq_c2c','tqqq_pc2o']], on='Date', how='left')
    df = df.merge(vix, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)

    # Regime
    cur_reg = 'C/Up'
    regimes = []
    for t in df['Trend']:
        if t == 'C/Up': cur_reg = 'C/Up'
        elif t == 'C/Dn': cur_reg = 'C/Dn'
        regimes.append(cur_reg)
    df['regime'] = regimes

    # ATR14 Wilder on QQQ
    high  = df['High'].values
    low   = df['Low'].values
    close = df['Close'].values
    tr_q  = np.zeros(len(df))
    for i in range(len(df)):
        hl = high[i] - low[i]
        if i == 0:
            tr_q[i] = hl
        else:
            tr_q[i] = max(hl, abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr14 = np.full(len(df), np.nan)
    atr14[13] = np.mean(tr_q[0:14])
    for i in range(14, len(df)):
        atr14[i] = (atr14[i-1]*13 + tr_q[i]) / 14
    df['ATR14'] = atr14

    # DEW on full QQQ history
    qqq_dew = qqq_full[qqq_full['Date'] >= '1999-07-01'].copy().reset_index(drop=True)
    close_dew = qqq_dew['Close'].values
    N_dew = len(close_dew)
    s = pd.Series(close_dew)
    sma20 = s.rolling(20).mean()
    D_arr = (s - sma20.shift(11)).round(2).values
    ema10 = s.ewm(span=10, adjust=False).mean().values
    E_upper = ema10 * 1.06
    E_lower = ema10 * 0.94
    weights = np.arange(1, 31, dtype=float)
    W_arr = np.full(N_dew, np.nan)
    for i in range(29, N_dew):
        W_arr[i] = np.dot(close_dew[i-29:i+1], weights) / weights.sum()

    DEW_Signal = [''] * N_dew
    state = 'Normal'
    last_dew = None
    for i in range(41, N_dew):
        c = close_dew[i]; d = D_arr[i]; w = W_arr[i]
        if np.isnan(d) or np.isnan(w): continue
        candidate = None
        if state == 'Normal':
            if c > w and d > 0: candidate = 'Buy'
            elif c < w and d < 0: candidate = 'Sell'
        else:
            if c > w or d > 0: candidate = 'Buy'
            elif c < w or d < 0: candidate = 'Sell'
        if candidate is not None and candidate == last_dew: candidate = None
        if candidate is not None:
            DEW_Signal[i] = candidate; last_dew = candidate; state = 'Normal'
        if c > E_upper[i] or c < E_lower[i]: state = 'Primed'

    qqq_dew['DEW_Signal'] = DEW_Signal
    df = df.merge(qqq_dew[['Date','DEW_Signal']], on='Date', how='left')
    df['DEW_Signal'] = df['DEW_Signal'].fillna('')

    return df


# ── Full Nitro++ v8 Engine ────────────────────────────────────────────────────

def run_engine(df):
    N           = len(df)
    dates_arr   = df['Date'].values
    trend_arr   = df['Trend'].values
    regime_arr  = df['regime'].values
    dew_arr     = df['DEW_Signal'].values
    o2c_q       = df['o2c_QQQ'].values
    c2c_q       = df['c2c_QQQ'].values
    pc2o_q      = df['pc2o_QQQ'].values
    atr14_arr   = df['ATR14'].values
    close_arr   = df['Close'].values
    rt_arr      = df['RT'].values
    mti_arr     = df['MTI'].values
    bsr_arr     = df['BSR'].values
    vix_arr     = df['VIX'].values
    trt_arr     = df['tqqq_rt'].values
    tatr_arr    = df['tqqq_atr'].values
    tqqq_open_arr  = df['tqqq_open'].values
    tqqq_close_arr = df['tqqq_close'].values
    tqqq_c2c_arr   = df['tqqq_c2c'].values
    tqqq_o2c_arr   = df['tqqq_o2c'].values
    tqqq_pc2o_arr  = df['tqqq_pc2o'].values
    o2c_psq     = df['o2c_PSQ'].values
    c2c_psq     = df['c2c_PSQ'].values
    pc2o_psq    = df['pc2o_PSQ'].values

    def cup_tqqq_fires(i):
        if np.isnan(trt_arr[i]) or np.isnan(tatr_arr[i]) or np.isnan(vix_arr[i]):
            return False
        return (regime_arr[i] == 'C/Up' and not cup_entered
                and trt_arr[i] < 1.40 and bsr_arr[i] > 1.05
                and tatr_arr[i] < 7.0  and vix_arr[i] < 30)

    def cdn_tqqq_fires(i):
        return (regime_arr[i] == 'C/Dn' and dew_arr[i] == 'Buy'
                and 0.95 <= rt_arr[i] < 1.00 and mti_arr[i] < 1.00)

    def qqq_skip(i):
        rt = rt_arr[i]; bsr = bsr_arr[i]; mti = mti_arr[i]
        return (0.85 <= rt < 0.90) or (bsr > 1.50) or (mti > 1.10)

    def psq_skip(i):
        return rt_arr[i] > 0.95 and mti_arr[i] > 0.95

    trades   = []
    in_trade = False; entry_idx = None; cum_mult = 1.0
    stop_level = None; tp_level = None; tqqq_stop = None
    inst = None; short_leg = False; cdn_tqqq_active = False
    last_dew = None; cup_entered = False

    # State log: per-bar snapshot of engine state (for live signal)
    state_log = []

    for i in range(N):
        trend_i  = str(trend_arr[i])
        prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
        dew_sig  = dew_arr[i]

        # Phase A
        if trend_i == 'C/Up' and prev_reg != 'C/Up':
            last_dew = None; cup_entered = False
        if trend_i == 'C/Dn' and prev_reg != 'C/Dn':
            last_dew = None

        # Priority 1a: C/Up TQQQ preemption of QQQ/PSQ
        if in_trade and inst in ('QQQ','PSQ') and cup_tqqq_fires(i):
            if i + 1 < N:
                if inst == 'QQQ':
                    d = o2c_q[i] if i == entry_idx else c2c_q[i]
                    if np.isnan(d): d = 0.0
                    cum_mult *= (1+d)
                    pc2o_next = pc2o_q[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    ret = cum_mult*(1+pc2o_next)-1
                    trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret':ret,'type':'preempt','inst':'QQQ'})
                else:
                    d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
                    if np.isnan(d): d = 0.0
                    cum_mult *= (1+d)
                    pc2o_next = pc2o_psq[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    ret = cum_mult*(1+pc2o_next)-1
                    trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret':ret,'type':'preempt','inst':'PSQ'})
                last_dew = None
                entry_idx = i+1; inst='TQQQ'; short_leg=False; cdn_tqqq_active=False
                cum_mult=1.0
                tp_level   = tqqq_open_arr[i+1]*1.50
                tqqq_stop  = tqqq_open_arr[i+1]*0.95
                cup_entered = True; in_trade = True
            continue

        # Phase B
        if in_trade:
            if inst == 'TQQQ':
                d = tqqq_o2c_arr[i] if i == entry_idx else tqqq_c2c_arr[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1+d)
                if tqqq_close_arr[i] >= tp_level:
                    trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':pd.Timestamp(dates_arr[i]).date(),
                                   'ret':cum_mult-1,'type':'TP','inst':'TQQQ'})
                    last_dew=None; in_trade=False; cum_mult=1.0
                    if not cdn_tqqq_active:
                        if regime_arr[i] == 'C/Dn': pass
                        else: continue
                    else: continue
                elif tqqq_close_arr[i] <= tqqq_stop:
                    if i+1 < N:
                        pc2o_next = tqqq_pc2o_arr[i+1]
                        if np.isnan(pc2o_next): pc2o_next = 0.0
                        ret = cum_mult*(1+pc2o_next)-1
                        trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                       'exit':pd.Timestamp(dates_arr[i+1]).date(),
                                       'ret':ret,'type':'Stop','inst':'TQQQ'})
                    last_dew=None; in_trade=False; cum_mult=1.0; continue
                elif not cdn_tqqq_active and regime_arr[i] == 'C/Dn':
                    if i+1 < N:
                        pc2o_next = tqqq_pc2o_arr[i+1]
                        if np.isnan(pc2o_next): pc2o_next = 0.0
                        ret = cum_mult*(1+pc2o_next)-1
                        trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                       'exit':pd.Timestamp(dates_arr[i+1]).date(),
                                       'ret':ret,'type':'C/Dn','inst':'TQQQ'})
                    last_dew=None; in_trade=False; cum_mult=1.0
                elif cdn_tqqq_active:
                    cdn_tqqq_exit = (dew_sig=='Sell' or (trend_i=='C/Up' and prev_reg=='C/Dn'))
                    if cdn_tqqq_exit:
                        if i+1 < N:
                            pc2o_next = tqqq_pc2o_arr[i+1]
                            if np.isnan(pc2o_next): pc2o_next = 0.0
                            ret = cum_mult*(1+pc2o_next)-1
                            trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                           'exit':pd.Timestamp(dates_arr[i+1]).date(),
                                           'ret':ret,'type':'Signal','inst':'TQQQ'})
                        last_dew=None; in_trade=False; cum_mult=1.0
                    else:
                        continue
                else:
                    continue
            else:  # QQQ or PSQ
                if inst == 'QQQ':
                    d = o2c_q[i] if i == entry_idx else c2c_q[i]
                    pc2o_next_fn = lambda: pc2o_q[i+1] if i+1<N else 0.0
                else:
                    d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
                    pc2o_next_fn = lambda: pc2o_psq[i+1] if i+1<N else 0.0
                if np.isnan(d): d = 0.0
                cum_mult *= (1+d)
                cdn_flip = (trend_i=='C/Dn' and prev_reg=='C/Up') and not short_leg
                cup_exit = (trend_i=='C/Up' and prev_reg=='C/Dn') and short_leg
                if cum_mult-1 >= 0.05:
                    trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':pd.Timestamp(dates_arr[i]).date(),
                                   'ret':cum_mult-1,'type':'TP','inst':inst})
                    in_trade=False; cum_mult=1.0
                    if not cup_exit and not cdn_flip: continue
                elif (not short_leg and close_arr[i]<=stop_level) or \
                     (short_leg  and close_arr[i]>=stop_level):
                    trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':pd.Timestamp(dates_arr[i]).date(),
                                   'ret':cum_mult-1,'type':'ATR','inst':inst})
                    in_trade=False; cum_mult=1.0
                    if not cup_exit and not cdn_flip: continue
                elif cdn_flip or cup_exit:
                    if i+1 < N:
                        pco = pc2o_next_fn()
                        if np.isnan(pco): pco = 0.0
                        ret = cum_mult*(1+pco)-1
                        trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                       'exit':pd.Timestamp(dates_arr[i+1]).date(),
                                       'ret':ret,'type':'flip','inst':inst})
                    in_trade=False; cum_mult=1.0
                elif (inst=='QQQ' and dew_sig=='Sell') or (inst=='PSQ' and dew_sig=='Buy'):
                    trades.append({'entry':pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':pd.Timestamp(dates_arr[i]).date(),
                                   'ret':cum_mult-1,'type':'DEW','inst':inst})
                    last_dew=dew_sig; in_trade=False; cum_mult=1.0
                else:
                    continue

        if in_trade: continue

        # Priority 1b: C/Up TQQQ when flat
        if cup_tqqq_fires(i) and i+1 < N:
            entry_idx=i+1; inst='TQQQ'; short_leg=False; cdn_tqqq_active=False
            cum_mult=1.0
            tp_level  = tqqq_open_arr[i+1]*1.50
            tqqq_stop = tqqq_open_arr[i+1]*0.95
            cup_entered=True; in_trade=True; continue

        # Priority 2: C/Dn TQQQ
        if cdn_tqqq_fires(i) and i+1 < N:
            entry_idx=i+1; inst='TQQQ'; short_leg=False; cdn_tqqq_active=True
            cum_mult=1.0
            tp_level  = tqqq_open_arr[i+1]*1.50
            tqqq_stop = tqqq_open_arr[i+1]*0.95
            last_dew=None; in_trade=True; continue

        # Priority 3: DEW QQQ long
        if dew_sig=='Buy' and last_dew!='Buy' and not qqq_skip(i) and i+1<N:
            sig_regime  = regime_arr[i]
            atr_mult    = 2.0 if sig_regime=='C/Dn' else 1.0
            stop_level  = df['Open'].iloc[i+1] - atr_mult*atr14_arr[i]
            entry_idx=i+1; inst='QQQ'; short_leg=False
            last_dew='Buy'; cum_mult=1.0; in_trade=True; continue

        # Priority 4: DEW SQQQ short
        if dew_sig=='Sell' and last_dew!='Sell' and not psq_skip(i) and i+1<N:
            sig_regime  = regime_arr[i]
            atr_mult    = 2.0 if sig_regime=='C/Dn' else 1.0
            stop_level  = df['Open'].iloc[i+1] + atr_mult*atr14_arr[i]
            entry_idx=i+1; inst='PSQ'; short_leg=True
            last_dew='Sell'; cum_mult=1.0; in_trade=True; continue

        # Record state at end of each bar (for live signal extraction)
        state_log.append({
            'i': i,
            'date': pd.Timestamp(dates_arr[i]).date(),
            'in_trade': in_trade,
            'inst': inst,
            'entry_idx': entry_idx,
            'cum_mult': cum_mult,
            'stop_level': stop_level,
            'tp_level': tp_level,
            'tqqq_stop': tqqq_stop,
            'last_dew': last_dew,
            'cup_entered': cup_entered,
            'cdn_tqqq_active': cdn_tqqq_active,
            'short_leg': short_leg,
        })

    # Re-run to capture final state after the last bar
    # (state_log only appends when flat; for open trade we capture differently)
    # Instead return the full engine state at end of last bar
    final_state = {
        'in_trade': in_trade,
        'inst': inst,
        'entry_idx': entry_idx,
        'cum_mult': cum_mult,
        'stop_level': stop_level,
        'tp_level': tp_level,
        'tqqq_stop': tqqq_stop,
        'last_dew': last_dew,
        'cup_entered': cup_entered,
        'cdn_tqqq_active': cdn_tqqq_active,
        'short_leg': short_leg,
    }

    return trades, final_state


# ── Determine Today's Signal ──────────────────────────────────────────────────

def get_signal(df, trades, final_state):
    """
    Look at the final bar of df and the engine state to determine:
    - What action is required at tomorrow's open (or: stay flat, hold)
    - Stop levels and targets for any new trade
    """
    N = len(df)
    last_bar = df.iloc[-1]
    last_date = last_bar['Date'].date()
    today = date.today()

    # Resolve what happened on the final bar
    in_trade  = final_state['in_trade']
    inst      = final_state['inst']
    stop_level = final_state['stop_level']
    tp_level   = final_state['tp_level']
    tqqq_stop  = final_state['tqqq_stop']
    cum_mult   = final_state['cum_mult']
    entry_idx  = final_state['entry_idx']
    cdn_active = final_state['cdn_tqqq_active']

    # Arrays for forward-looking signal (we need last bar values)
    i       = N - 1
    trend_i = str(df['Trend'].iloc[i])
    regime_i = df['regime'].iloc[i]
    dew_i   = df['DEW_Signal'].iloc[i]
    rt      = df['RT'].iloc[i]
    mti     = df['MTI'].iloc[i]
    bsr     = df['BSR'].iloc[i]
    vix     = df['VIX'].iloc[i]
    trt     = df['tqqq_rt'].iloc[i]
    tatr    = df['tqqq_atr'].iloc[i]
    atr14   = df['ATR14'].iloc[i]
    qqq_open_next = df['Open'].iloc[i]      # today's open (proxy for tomorrow)
    tqqq_open_next = df['tqqq_open'].iloc[i]
    qqq_close = df['Close'].iloc[i]
    tqqq_close = df['tqqq_close'].iloc[i]

    last_dew    = final_state['last_dew']
    cup_entered = final_state['cup_entered']

    actions = []
    notes   = []

    if in_trade:
        # Already in a position — describe it and check for any exits
        entry_date = df['Date'].iloc[entry_idx].date() if entry_idx is not None else '?'
        unrealized = (cum_mult - 1) * 100

        if inst == 'TQQQ':
            tp_pct   = (tp_level / tqqq_open_next - 1)*100 if tqqq_open_next else 0
            stop_pct = (tqqq_stop / tqqq_open_next - 1)*100 if tqqq_open_next else 0
            trade_type = 'C/Dn TQQQ' if cdn_active else 'C/Up TQQQ'
            actions.append(f"HOLD {trade_type} position (entered {entry_date})")
            notes.append(f"Unrealized P&L: {unrealized:+.2f}%")
            notes.append(f"TP target:  TQQQ closes ≥ ${tp_level:.2f} (+50% from entry open)")
            notes.append(f"Hard stop:  TQQQ closes ≤ ${tqqq_stop:.2f} (-5% from entry open) → exit at NEXT open")
            if not cdn_active:
                notes.append(f"Regime exit: if VectorVest prints C/Dn → exit at NEXT open")
            else:
                notes.append(f"Exit signal: DEW Sell fires, or VectorVest prints C/Up → exit at NEXT open")
        else:
            instr_name = 'QQQ' if inst == 'QQQ' else 'SQQQ (1/3 position)'
            dir_word   = 'LONG' if inst == 'QQQ' else 'SHORT'
            actions.append(f"HOLD {dir_word} {instr_name} (entered {entry_date})")
            notes.append(f"Unrealized P&L: {unrealized:+.2f}%")
            if inst == 'QQQ':
                notes.append(f"ATR stop:  QQQ closes ≤ ${stop_level:.2f} → exit same-day close")
                notes.append(f"TP target: cumulative gain ≥ +5%")
                notes.append(f"Regime exit: VectorVest C/Dn flip → exit at NEXT open")
                notes.append(f"DEW exit:  DEW Sell signal fires → exit same-day close")
            else:
                notes.append(f"ATR stop:  QQQ closes ≥ ${stop_level:.2f} → exit same-day close")
                notes.append(f"TP target: cumulative gain ≥ +5%")
                notes.append(f"Regime exit: VectorVest C/Up flip → exit at NEXT open")
                notes.append(f"DEW exit:  DEW Buy signal fires → exit same-day close")
    else:
        # Flat — look for a new signal on tomorrow's open
        # Check C/Up TQQQ
        cup_signal = (regime_i == 'C/Up' and not cup_entered
                      and not np.isnan(trt) and trt < 1.40
                      and bsr > 1.05
                      and not np.isnan(tatr) and tatr < 7.0
                      and not np.isnan(vix) and vix < 30)
        # Check C/Dn TQQQ
        cdn_signal = (regime_i == 'C/Dn' and dew_i == 'Buy'
                      and 0.95 <= rt < 1.00 and mti < 1.00)
        # Check QQQ
        qqq_signal = (dew_i == 'Buy' and last_dew != 'Buy'
                      and not ((0.85 <= rt < 0.90) or (bsr > 1.50) or (mti > 1.10)))
        # Check SQQQ
        psq_signal = (dew_i == 'Sell' and last_dew != 'Sell'
                      and not (rt > 0.95 and mti > 0.95))

        if cup_signal:
            atr_mult    = 1.0
            entry_open  = tqqq_close   # proxy; actual fill = tomorrow's TQQQ open
            tp_tgt      = entry_open * 1.50
            stop_tgt    = entry_open * 0.95
            actions.append("BUY TQQQ (C/Up signal) — enter at TOMORROW'S open")
            notes.append(f"Regime: C/Up  |  TQQQ-RT={trt:.3f}  BSR={bsr:.2f}  ATR={tatr:.2f}%  VIX={vix:.1f}")
            notes.append(f"TP target:  TQQQ closes ≥ 150% of your fill price (+50%)")
            notes.append(f"Hard stop:  TQQQ closes ≤ 95% of your fill price (-5%) → exit at NEXT day's open")
            notes.append(f"Regime exit: if VectorVest prints C/Dn → exit at NEXT open")
            notes.append(f"Place a GTC stop-limit order at 95% of your actual fill price after you enter.")
        elif cdn_signal:
            entry_open  = tqqq_close
            tp_tgt      = entry_open * 1.50
            stop_tgt    = entry_open * 0.95
            actions.append("BUY TQQQ (C/Dn bounce signal) — enter at TOMORROW'S open")
            notes.append(f"Regime: C/Dn  |  VVC-RT={rt:.3f}  MTI={mti:.3f}")
            notes.append(f"TP target:  TQQQ closes ≥ 150% of your fill price (+50%)")
            notes.append(f"Hard stop:  TQQQ closes ≤ 95% of your fill price (-5%) → exit at NEXT day's open")
            notes.append(f"Exit signal: DEW Sell fires or C/Up transition → exit at NEXT open")
            notes.append(f"Place a GTC stop-limit order at 95% of your actual fill price after you enter.")
        elif qqq_signal:
            atr_mult    = 2.0 if regime_i == 'C/Dn' else 1.0
            stop_tgt    = qqq_close - atr_mult * atr14   # proxy; actual = entry open − ATR_mult×ATR14
            actions.append("BUY QQQ — enter at TOMORROW'S open")
            notes.append(f"Regime: {regime_i}  |  DEW Buy fired today  |  VVC-RT={rt:.3f}  BSR={bsr:.2f}  MTI={mti:.3f}")
            notes.append(f"ATR14 today: ${atr14:.2f}  |  ATR multiplier: {atr_mult:.0f}×  ({'Bear' if regime_i=='C/Dn' else 'Bull'} regime)")
            notes.append(f"Stop level: entry open − {atr_mult:.0f}×ATR14  ≈ ${stop_tgt:.2f}")
            notes.append(f"  → Set a GTC stop-market order below your actual fill open price.")
            notes.append(f"  → Exact stop = (your fill price) − {atr_mult:.0f} × ${atr14:.2f}")
            notes.append(f"TP target:  cumulative +5% from entry open → exit at same-day close")
            notes.append(f"Regime exit: VectorVest C/Dn flip → exit at NEXT open")
            notes.append(f"DEW exit:   DEW Sell signal fires → exit at same-day close")
        elif psq_signal:
            atr_mult    = 2.0 if regime_i == 'C/Dn' else 1.0
            stop_tgt    = qqq_close + atr_mult * atr14
            actions.append("BUY SQQQ (1/3 position size) — enter at TOMORROW'S open")
            notes.append(f"Regime: {regime_i}  |  DEW Sell fired today  |  VVC-RT={rt:.3f}  BSR={bsr:.2f}  MTI={mti:.3f}")
            notes.append(f"ATR14 today: ${atr14:.2f}  |  ATR multiplier: {atr_mult:.0f}×  ({'Bear' if regime_i=='C/Dn' else 'Bull'} regime)")
            notes.append(f"QQQ stop level: entry QQQ open + {atr_mult:.0f}×ATR14  ≈ ${stop_tgt:.2f}")
            notes.append(f"  → SQQQ stop triggers when QQQ RISES to ${stop_tgt:.2f} or above.")
            notes.append(f"  → Monitor QQQ close daily; exit SQQQ if QQQ close ≥ stop level.")
            notes.append(f"  → Exact QQQ stop = (tomorrow's QQQ open) + {atr_mult:.0f} × ${atr14:.2f}")
            notes.append(f"TP target:  cumulative +5% SQQQ gain from entry → exit at same-day close")
            notes.append(f"Regime exit: VectorVest C/Up flip → exit SQQQ at NEXT open")
            notes.append(f"DEW exit:   DEW Buy signal fires → exit SQQQ at same-day close")
        else:
            actions.append("STAY FLAT — no signal today")
            # Give a brief summary of why
            reasons = []
            if regime_i == 'C/Up':
                reasons.append(f"C/Up regime  |  cup_entered={cup_entered}")
                if cup_entered:
                    reasons.append("  Waiting for new C/Up run to reset TQQQ entry")
            else:
                reasons.append(f"C/Dn regime")
            reasons.append(f"DEW: last_dew={last_dew}  today_signal={dew_i or 'none'}")
            reasons.append(f"Indicators: VVC-RT={rt:.3f}  MTI={mti:.3f}  BSR={bsr:.2f}  VIX={vix:.1f}  TQQQ-RT={trt:.3f}  TQQQ-ATR={tatr:.2f}%")
            notes.extend(reasons)

    return actions, notes, last_date


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(subject, body_text):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = ', '.join(TO_EMAIL)
    msg.attach(MIMEText(body_text, 'plain'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    today = date.today()
    if today.weekday() >= 5:
        print(f"Today is {today.strftime('%A')} — skipping.")
        sys.exit(0)

    print(f"Nitro++ Daily Signal  {today}\n")
    print("Loading data...")
    df = load_data()

    print("Running engine...")
    trades, final_state = run_engine(df)

    # Compute current equity (historical trades up to live start)
    equity = STARTING_CAPITAL
    historical_trades = [t for t in trades
                         if pd.Timestamp(t['exit']) < LIVE_START]
    for t in historical_trades:
        equity *= (1 + t['ret'])

    print(f"Historical trades through {LIVE_START.date()}: {len(historical_trades)}")
    print(f"Simulated equity at live start: ${equity:,.2f}")

    # Get signal
    actions, notes, data_date = get_signal(df, trades, final_state)

    # Build output
    lines = []
    lines.append(f"Dampier Nitro++ — Daily Signal")
    lines.append(f"Data through: {data_date}  |  Signal for: tomorrow's open")
    lines.append(f"Simulated account equity: ${equity:,.2f}")
    lines.append("=" * 55)
    lines.append("")
    for a in actions:
        lines.append(f"  ▶  {a}")
    lines.append("")
    lines.append("Details:")
    for n in notes:
        lines.append(f"  {n}")
    lines.append("")
    lines.append("=" * 55)
    lines.append(f"Last {min(5,len(trades))} completed trades:")
    for t in trades[-5:]:
        inst_label = 'SQQQ' if t['inst'] == 'PSQ' else t['inst']
        lines.append(f"  {t['entry']}→{t['exit']}  {inst_label:4s}  {t['ret']*100:+.2f}%  {t['type']}")
    lines.append("")
    lines.append("Exit fill rules (reference):")
    lines.append("  TP / ATR / DEW reversal → same-day close fill")
    lines.append("  Regime flip / TQQQ stop → NEXT day open fill")
    lines.append("  TQQQ TP → same-day close fill")

    body = "\n".join(lines)
    print("\n" + body)

    # Determine subject line
    if any("STAY FLAT" in a for a in actions):
        subject = f"Nitro++ Signal {data_date}: FLAT — no trade"
    elif any("HOLD" in a for a in actions):
        inst_word = "TQQQ" if "TQQQ" in actions[0] else ("QQQ" if "QQQ" in actions[0] else "SQQQ")
        subject = f"Nitro++ Signal {data_date}: HOLD {inst_word}"
    else:
        action_word = actions[0].split("—")[0].strip() if actions else "BUY"
        subject = f"Nitro++ Signal {data_date}: {action_word}"

    print(f"\nSending email to {TO_EMAIL} ...")
    send_email(subject, body)
    print("Email sent.")


if __name__ == '__main__':
    main()
