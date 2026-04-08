#!/usr/bin/env python3
"""
Dampier Nitro++ v10 Daily Signal Generator
Runs after market close. Reads the latest history CSVs, replays the full
Nitro++ v10 engine through today, then emails trade instructions.

v10 changes vs v8:
  - Priority 1b: C/Dn TQQQ switch while in open QQQ long
  - Fix: cdn_tqqq_active reset on ALL TQQQ exits (TP and Stop)
  - Sixth data file: sqqq-from-yahoo.csv loaded
  - Health factor: Volatility-Adaptive Defensive Sizing (dret only)
  - Updated LIVE_START to 2026-04-07

Usage:
    python3 nitro_daily_signal_v10.py
"""

import os
import sys
import smtplib
import numpy as np
import pandas as pd
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ.get('GOOGLE_EMAIL', 'dampiermike@gmail.com')
GMAIL_PASS = os.environ.get('GOOGLE_APP_PASSWORD', '')
TO_EMAIL   = ['dampiermike@gmail.com', 'dampier777@gmail.com']

NITRO_DIR = Path('/Users/mikedampier/Documents/Development/Nitro')
DATA_DIR  = NITRO_DIR / 'data' / 'csv' / 'history'

QQQ_FILE  = DATA_DIR / 'qqq-from-vv.csv'
TQQQ_FILE = DATA_DIR / 'tqqq-from-vv.csv'
SYN_FILE  = DATA_DIR / 'synthetic-tqqq-ohlc-1999-2010.csv'
VV_FILE   = DATA_DIR / 'vectorvest-views-w3place-precision.csv'
VIX_FILE  = DATA_DIR / 'vix-from-yahoo.csv'
SQQQ_FILE = DATA_DIR / 'sqqq-from-yahoo.csv'

STARTING_CAPITAL = 100_000.0
LIVE_START       = pd.Timestamp('2026-04-07')


# ── Load Data (matches v10 engine) ────────────────────────────────────────────

def load_data():
    # QQQ
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

    # Synthetic TQQQ
    syn = pd.read_csv(SYN_FILE)
    syn['Date'] = pd.to_datetime(syn['Date'])
    syn = syn.sort_values('Date').reset_index(drop=True)
    syn = syn.rename(columns={'Open': 'tqqq_open', 'High': 'tqqq_high',
                               'Low': 'tqqq_low', 'Close': 'tqqq_close', 'RT_v6': 'tqqq_rt'})
    syn = syn[['Date', 'tqqq_open', 'tqqq_high', 'tqqq_low', 'tqqq_close', 'tqqq_rt']]

    # Real TQQQ
    tqqq_real = pd.read_csv(TQQQ_FILE, encoding='utf-8-sig')
    tqqq_real['Date'] = pd.to_datetime(tqqq_real['Date'])
    tqqq_real = tqqq_real.sort_values('Date').reset_index(drop=True)
    tqqq_real = tqqq_real.rename(columns={'Open': 'tqqq_open', 'High': 'tqqq_high',
                                           'Low': 'tqqq_low', 'Close': 'tqqq_close', 'RT': 'tqqq_rt'})
    tqqq_real = tqqq_real[['Date', 'tqqq_open', 'tqqq_high', 'tqqq_low', 'tqqq_close', 'tqqq_rt']]

    # Stitch TQQQ
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
    # tqqq_atr: simple (NOT Wilder) 10-bar ATR%, computed on FULL stitched series
    tqqq_stitched['tqqq_atr'] = tr_t.rolling(10).mean() / tqqq_stitched['tqqq_close'] * 100
    tqqq = tqqq_stitched[tqqq_stitched['Date'] >= '2000-01-01'].copy()

    # VectorVest
    vv = pd.read_csv(VV_FILE)
    vv['Date'] = pd.to_datetime(vv['Date'])
    vv = vv.sort_values('Date').reset_index(drop=True)
    vv = vv.rename(columns={'VVC-RT': 'RT', 'BS Ratio': 'BSR'})
    vv = vv[['Date', 'Trend', 'RT', 'MTI', 'BSR']]

    # VIX
    vix = pd.read_csv(VIX_FILE)
    vix['Date'] = pd.to_datetime(vix['Date'], errors='coerce')
    vix = vix.dropna(subset=['Date'])
    vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX'})
    vix['VIX'] = pd.to_numeric(vix['VIX'], errors='coerce')

    # SQQQ (sixth raw data file — loaded per v10 spec)
    sqqq_raw = pd.read_csv(SQQQ_FILE)
    sqqq_raw['Date'] = pd.to_datetime(sqqq_raw['Date'], errors='coerce')
    sqqq_raw = sqqq_raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    # adj_factor = Adj Close / Close; PSQ return columns still derived from QQQ inverse

    # Merge
    df = vv.merge(qqq[['Date', 'Open', 'High', 'Low', 'Close',
                        'o2c_QQQ', 'c2c_QQQ', 'pc2o_QQQ',
                        'o2c_PSQ', 'c2c_PSQ', 'pc2o_PSQ']], on='Date', how='inner')
    df = df.merge(tqqq[['Date', 'tqqq_open', 'tqqq_close', 'tqqq_rt',
                         'tqqq_atr', 'tqqq_o2c', 'tqqq_c2c', 'tqqq_pc2o']], on='Date', how='left')
    df = df.merge(vix, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)

    # Regime
    cur_reg = 'C/Up'
    regimes = []
    for t in df['Trend']:
        if t == 'C/Up':   cur_reg = 'C/Up'
        elif t == 'C/Dn': cur_reg = 'C/Dn'
        regimes.append(cur_reg)
    df['regime'] = regimes

    # ATR14 Wilder on QQQ (on merged df)
    high  = df['High'].values
    low   = df['Low'].values
    close = df['Close'].values
    tr_q  = np.zeros(len(df))
    for i in range(len(df)):
        hl = high[i] - low[i]
        if i == 0:
            tr_q[i] = hl
        else:
            tr_q[i] = max(hl, abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr14 = np.full(len(df), np.nan)
    atr14[13] = np.mean(tr_q[0:14])
    for i in range(14, len(df)):
        atr14[i] = (atr14[i-1] * 13 + tr_q[i]) / 14
    df['ATR14'] = atr14

    # DEW Oscillator — compute on FULL QQQ history starting July 1999
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
    last_dew_init = None
    for i in range(41, N_dew):
        c = close_dew[i]; d = D_arr[i]; w = W_arr[i]
        if np.isnan(d) or np.isnan(w): continue
        candidate = None
        if state == 'Normal':
            if c > w and d > 0:   candidate = 'Buy'
            elif c < w and d < 0: candidate = 'Sell'
        else:
            if c > w or d > 0:    candidate = 'Buy'
            elif c < w or d < 0:  candidate = 'Sell'
        if candidate is not None and candidate == last_dew_init: candidate = None
        if candidate is not None:
            DEW_Signal[i] = candidate; last_dew_init = candidate; state = 'Normal'
        if c > E_upper[i] or c < E_lower[i]: state = 'Primed'

    qqq_dew['DEW_Signal'] = DEW_Signal
    df = df.merge(qqq_dew[['Date', 'DEW_Signal']], on='Date', how='left')
    df['DEW_Signal'] = df['DEW_Signal'].fillna('')

    return df


# ── Full Nitro++ v10 Engine ───────────────────────────────────────────────────

def run_engine(df):
    N              = len(df)
    dates_arr      = df['Date'].values
    trend_arr      = df['Trend'].values
    regime_arr     = df['regime'].values
    dew_arr        = df['DEW_Signal'].values
    o2c_q          = df['o2c_QQQ'].values
    c2c_q          = df['c2c_QQQ'].values
    pc2o_q         = df['pc2o_QQQ'].values
    atr14_arr      = df['ATR14'].values
    close_arr      = df['Close'].values
    rt_arr         = df['RT'].values
    mti_arr        = df['MTI'].values
    bsr_arr        = df['BSR'].values
    vix_arr        = df['VIX'].values
    trt_arr        = df['tqqq_rt'].values
    tatr_arr       = df['tqqq_atr'].values
    tqqq_open_arr  = df['tqqq_open'].values
    tqqq_close_arr = df['tqqq_close'].values
    tqqq_c2c_arr   = df['tqqq_c2c'].values
    tqqq_o2c_arr   = df['tqqq_o2c'].values
    tqqq_pc2o_arr  = df['tqqq_pc2o'].values
    o2c_psq        = df['o2c_PSQ'].values
    c2c_psq        = df['c2c_PSQ'].values
    pc2o_psq       = df['pc2o_PSQ'].values

    # Pre-compute health factor (for equity display — does NOT affect trade returns)
    health_arr = np.ones(N)
    for i in range(N):
        vix_i = vix_arr[i]; atr_i = tatr_arr[i]; bsr_i = bsr_arr[i]
        if np.isnan(vix_i) or np.isnan(atr_i):
            health_arr[i] = 1.0
        else:
            raw = ((30 - vix_i) / 20.0
                   * 8.0 / max(atr_i, 4.0)
                   * max(bsr_i, 0.5))
            health_arr[i] = min(1.0, max(0.25, raw))

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

    trades          = []
    in_trade        = False
    entry_idx       = None
    cum_mult        = 1.0
    stop_level      = None
    tp_level        = None
    tqqq_stop       = None
    inst            = None
    short_leg       = False
    cdn_tqqq_active = False
    last_dew        = None
    cup_entered     = False

    for i in range(N):
        trend_i  = str(trend_arr[i])
        prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
        dew_sig  = dew_arr[i]

        # Phase A: regime transitions
        if trend_i == 'C/Up' and prev_reg != 'C/Up':
            last_dew = None; cup_entered = False
        if trend_i == 'C/Dn' and prev_reg != 'C/Dn':
            last_dew = None

        # Priority 1a: C/Up TQQQ preempts open QQQ/PSQ
        if in_trade and inst in ('QQQ', 'PSQ') and cup_tqqq_fires(i):
            if i + 1 < N:
                if inst == 'QQQ':
                    d = o2c_q[i] if i == entry_idx else c2c_q[i]
                    if np.isnan(d): d = 0.0
                    cum_mult *= (1 + d)
                    pc2o_next = pc2o_q[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    ret = cum_mult * (1 + pc2o_next) - 1
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret': ret, 'type': 'preempt', 'inst': 'QQQ'})
                else:
                    d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
                    if np.isnan(d): d = 0.0
                    cum_mult *= (1 + d)
                    pc2o_next = pc2o_psq[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    ret = cum_mult * (1 + pc2o_next) - 1
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret': ret, 'type': 'preempt', 'inst': 'PSQ'})
                last_dew = None
                entry_idx = i + 1; inst = 'TQQQ'; short_leg = False; cdn_tqqq_active = False
                cum_mult = 1.0
                tp_level  = tqqq_open_arr[i+1] * 1.50
                tqqq_stop = tqqq_open_arr[i+1] * 0.95
                cup_entered = True; in_trade = True
            continue

        # Phase B: exit checks
        if in_trade:
            if inst == 'TQQQ':
                d = tqqq_o2c_arr[i] if i == entry_idx else tqqq_c2c_arr[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1 + d)

                # TQQQ TP (+50%)
                if tqqq_close_arr[i] >= tp_level:
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i]).date(),
                                   'ret': cum_mult - 1, 'type': 'TP', 'inst': 'TQQQ'})
                    was_cdn = cdn_tqqq_active
                    last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                    if not was_cdn:
                        if regime_arr[i] == 'C/Dn': pass
                        else: continue
                    else: continue

                # TQQQ hard stop (−5%) — next-bar open
                elif tqqq_close_arr[i] <= tqqq_stop:
                    if i + 1 < N:
                        pc2o_next = tqqq_pc2o_arr[i+1]
                        if np.isnan(pc2o_next): pc2o_next = 0.0
                        ret = cum_mult * (1 + pc2o_next) - 1
                        trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                       'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                       'ret': ret, 'type': 'Stop', 'inst': 'TQQQ'})
                    last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                    continue

                # C/Up TQQQ: exit when regime flips to C/Dn (value check, not transition)
                elif not cdn_tqqq_active and regime_arr[i] == 'C/Dn':
                    if i + 1 < N:
                        pc2o_next = tqqq_pc2o_arr[i+1]
                        if np.isnan(pc2o_next): pc2o_next = 0.0
                        ret = cum_mult * (1 + pc2o_next) - 1
                        trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                       'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                       'ret': ret, 'type': 'C/Dn', 'inst': 'TQQQ'})
                    last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                    # fall through

                # C/Dn TQQQ: exit on DEW Sell or C/Up transition
                elif cdn_tqqq_active:
                    cdn_exit = (dew_sig == 'Sell' or (trend_i == 'C/Up' and prev_reg == 'C/Dn'))
                    if cdn_exit:
                        if i + 1 < N:
                            pc2o_next = tqqq_pc2o_arr[i+1]
                            if np.isnan(pc2o_next): pc2o_next = 0.0
                            ret = cum_mult * (1 + pc2o_next) - 1
                            trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                           'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                           'ret': ret, 'type': 'Signal', 'inst': 'TQQQ'})
                        last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                        # fall through
                    else:
                        continue
                else:
                    continue

            else:  # QQQ or PSQ
                if inst == 'QQQ':
                    d = o2c_q[i] if i == entry_idx else c2c_q[i]
                else:
                    d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1 + d)

                # ── Priority 1b: C/Dn TQQQ switch while in QQQ long (v10 rule) ──
                # Fires before TP/ATR/flip. Only when in QQQ, cdn_tqqq_active=False.
                if (inst == 'QQQ'
                        and not cdn_tqqq_active
                        and regime_arr[i] == 'C/Dn'
                        and 0.95 <= rt_arr[i] < 1.00
                        and mti_arr[i] < 0.95
                        and i + 1 < N):
                    pc2o_next = pc2o_q[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    if i == entry_idx:
                        ret = cum_mult * (1 + pc2o_next) - 1
                    else:
                        ret = cum_mult * (1 + pc2o_next) - 1
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret': ret, 'type': 'p1b', 'inst': 'QQQ'})
                    # Enter C/Dn TQQQ at same next-bar open
                    entry_idx = i + 1; inst = 'TQQQ'; short_leg = False; cdn_tqqq_active = True
                    cum_mult = 1.0
                    tp_level  = tqqq_open_arr[i+1] * 1.50
                    tqqq_stop = tqqq_open_arr[i+1] * 0.95
                    last_dew = None; in_trade = True
                    continue

                cdn_flip = (trend_i == 'C/Dn' and prev_reg == 'C/Up') and not short_leg
                cup_exit = (trend_i == 'C/Up' and prev_reg == 'C/Dn') and short_leg

                if inst == 'QQQ':
                    pc2o_next_fn = lambda: pc2o_q[i+1] if i+1 < N else 0.0
                else:
                    pc2o_next_fn = lambda: pc2o_psq[i+1] if i+1 < N else 0.0

                # TP
                if cum_mult - 1 >= 0.05:
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i]).date(),
                                   'ret': cum_mult - 1, 'type': 'TP', 'inst': inst})
                    in_trade = False; cum_mult = 1.0
                    if not cup_exit and not cdn_flip: continue
                # ATR stop
                elif ((not short_leg and close_arr[i] <= stop_level) or
                      (short_leg and close_arr[i] >= stop_level)):
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i]).date(),
                                   'ret': cum_mult - 1, 'type': 'ATR', 'inst': inst})
                    in_trade = False; cum_mult = 1.0
                    if not cup_exit and not cdn_flip: continue
                # Regime flip
                elif cdn_flip or cup_exit:
                    if i + 1 < N:
                        pco = pc2o_next_fn()
                        if np.isnan(pco): pco = 0.0
                        ret = cum_mult * (1 + pco) - 1
                        trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                       'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                       'ret': ret, 'type': 'flip', 'inst': inst})
                    in_trade = False; cum_mult = 1.0
                    # fall through
                # DEW reversal
                elif ((inst == 'QQQ' and dew_sig == 'Sell') or
                      (inst == 'PSQ' and dew_sig == 'Buy')):
                    trades.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i]).date(),
                                   'ret': cum_mult - 1, 'type': 'DEW', 'inst': inst})
                    last_dew = dew_sig; in_trade = False; cum_mult = 1.0
                    # fall through
                else:
                    continue

        # Guard
        if in_trade: continue

        # Priority 1c: C/Up TQQQ fires when flat (same-bar chain)
        if cup_tqqq_fires(i) and i + 1 < N:
            entry_idx = i + 1; inst = 'TQQQ'; short_leg = False; cdn_tqqq_active = False
            cum_mult = 1.0
            tp_level  = tqqq_open_arr[i+1] * 1.50
            tqqq_stop = tqqq_open_arr[i+1] * 0.95
            cup_entered = True; in_trade = True; continue

        # Priority 2: C/Dn TQQQ (flat entry, DEW Buy required)
        if cdn_tqqq_fires(i) and i + 1 < N:
            entry_idx = i + 1; inst = 'TQQQ'; short_leg = False; cdn_tqqq_active = True
            cum_mult = 1.0
            tp_level  = tqqq_open_arr[i+1] * 1.50
            tqqq_stop = tqqq_open_arr[i+1] * 0.95
            last_dew = None; in_trade = True; continue

        # Priority 3: DEW QQQ long
        if dew_sig == 'Buy' and last_dew != 'Buy' and not qqq_skip(i) and i + 1 < N:
            atr_mult   = 2.0 if regime_arr[i] == 'C/Dn' else 1.0
            stop_level = df['Open'].iloc[i+1] - atr_mult * atr14_arr[i]
            entry_idx = i + 1; inst = 'QQQ'; short_leg = False
            last_dew = 'Buy'; cum_mult = 1.0; in_trade = True; continue

        # Priority 4: DEW SQQQ short
        if dew_sig == 'Sell' and last_dew != 'Sell' and not psq_skip(i) and i + 1 < N:
            atr_mult   = 2.0 if regime_arr[i] == 'C/Dn' else 1.0
            stop_level = df['Open'].iloc[i+1] + atr_mult * atr14_arr[i]
            entry_idx = i + 1; inst = 'PSQ'; short_leg = True
            last_dew = 'Sell'; cum_mult = 1.0; in_trade = True; continue

    # Capture final engine state
    final_state = {
        'in_trade':        in_trade,
        'inst':            inst,
        'entry_idx':       entry_idx,
        'cum_mult':        cum_mult,
        'stop_level':      stop_level,
        'tp_level':        tp_level,
        'tqqq_stop':       tqqq_stop,
        'last_dew':        last_dew,
        'cup_entered':     cup_entered,
        'cdn_tqqq_active': cdn_tqqq_active,
        'short_leg':       short_leg,
    }

    return trades, final_state, health_arr


# ── Determine Today's Signal ──────────────────────────────────────────────────

def get_signal(df, trades, final_state):
    N = len(df)
    last_bar  = df.iloc[-1]
    last_date = last_bar['Date'].date()

    in_trade        = final_state['in_trade']
    inst            = final_state['inst']
    stop_level      = final_state['stop_level']
    tp_level        = final_state['tp_level']
    tqqq_stop       = final_state['tqqq_stop']
    cum_mult        = final_state['cum_mult']
    entry_idx       = final_state['entry_idx']
    cdn_active      = final_state['cdn_tqqq_active']
    last_dew        = final_state['last_dew']
    cup_entered     = final_state['cup_entered']

    i          = N - 1
    trend_i    = str(df['Trend'].iloc[i])
    regime_i   = df['regime'].iloc[i]
    dew_i      = df['DEW_Signal'].iloc[i]
    rt         = df['RT'].iloc[i]
    mti        = df['MTI'].iloc[i]
    bsr        = df['BSR'].iloc[i]
    vix        = df['VIX'].iloc[i]
    trt        = df['tqqq_rt'].iloc[i]
    tatr       = df['tqqq_atr'].iloc[i]
    atr14      = df['ATR14'].iloc[i]
    qqq_close  = df['Close'].iloc[i]
    tqqq_close = df['tqqq_close'].iloc[i]

    # Safe entry date lookup (guard for pending entries at bar N)
    def entry_date_str():
        if entry_idx is None:      return '?'
        if entry_idx >= N:         return 'tomorrow (pending)'
        return str(df['Date'].iloc[entry_idx].date())

    actions = []
    notes   = []

    if in_trade:
        unrealized = (cum_mult - 1) * 100

        if inst == 'TQQQ':
            trade_type = 'C/Dn TQQQ' if cdn_active else 'C/Up TQQQ'
            entered    = entry_date_str()
            actions.append(f"HOLD {trade_type} position (entered {entered})")
            notes.append(f"Unrealized P&L: {unrealized:+.2f}%")
            if tp_level:
                notes.append(f"TP target:  TQQQ closes ≥ ${tp_level:.2f} (+50% from entry open)")
            if tqqq_stop:
                notes.append(f"Hard stop:  TQQQ closes ≤ ${tqqq_stop:.2f} (-5% from entry open) → exit at NEXT open")
            if not cdn_active:
                notes.append("Regime exit: if VectorVest prints C/Dn → exit at NEXT open")
            else:
                notes.append("Exit signal: DEW Sell fires, or VectorVest prints C/Up → exit at NEXT open")

        else:  # QQQ or PSQ
            instr_name = 'QQQ' if inst == 'QQQ' else 'SQQQ (1/3 position)'
            dir_word   = 'LONG' if inst == 'QQQ' else 'SHORT'
            entered    = entry_date_str()
            actions.append(f"HOLD {dir_word} {instr_name} (entered {entered})")
            notes.append(f"Unrealized P&L: {unrealized:+.2f}%")
            if inst == 'QQQ':
                if stop_level:
                    notes.append(f"ATR stop:  QQQ closes ≤ ${stop_level:.2f} → exit same-day close")
                notes.append("TP target: cumulative gain ≥ +5%")
                notes.append("Regime exit: VectorVest C/Dn flip → exit at NEXT open")
                notes.append("DEW exit:  DEW Sell signal fires → exit same-day close")
                # Priority 1b advisory (monitoring condition)
                p1b_watch = (not cdn_active
                             and regime_i == 'C/Dn'
                             and 0.95 <= rt < 1.00
                             and mti < 0.95)
                if p1b_watch:
                    notes.append(f"⚠ Priority 1b ACTIVE: regime=C/Dn, RT={rt:.3f}, MTI={mti:.3f}")
                    notes.append("  → QQQ will exit at tomorrow's open; C/Dn TQQQ enters at same open")
            else:
                if stop_level:
                    notes.append(f"ATR stop:  QQQ closes ≥ ${stop_level:.2f} → exit same-day close")
                notes.append("TP target: cumulative gain ≥ +5%")
                notes.append("Regime exit: VectorVest C/Up flip → exit at NEXT open")
                notes.append("DEW exit:  DEW Buy signal fires → exit same-day close")

    else:
        # Flat — check for new signals
        cup_signal = (regime_i == 'C/Up' and not cup_entered
                      and not np.isnan(trt) and trt < 1.40
                      and bsr > 1.05
                      and not np.isnan(tatr) and tatr < 7.0
                      and not np.isnan(vix) and vix < 30)
        cdn_signal = (regime_i == 'C/Dn' and dew_i == 'Buy'
                      and 0.95 <= rt < 1.00 and mti < 1.00)
        qqq_signal = (dew_i == 'Buy' and last_dew != 'Buy'
                      and not ((0.85 <= rt < 0.90) or (bsr > 1.50) or (mti > 1.10)))
        psq_signal = (dew_i == 'Sell' and last_dew != 'Sell'
                      and not (rt > 0.95 and mti > 0.95))

        if cup_signal:
            actions.append("BUY TQQQ (C/Up signal) — enter at TOMORROW'S open")
            notes.append(f"Regime: C/Up  |  TQQQ-RT={trt:.3f}  BSR={bsr:.2f}  ATR={tatr:.2f}%  VIX={vix:.1f}")
            notes.append("TP target:  TQQQ closes ≥ 150% of your fill price (+50%)")
            notes.append("Hard stop:  TQQQ closes ≤ 95% of your fill price (-5%) → exit at NEXT day's open")
            notes.append("Regime exit: if VectorVest prints C/Dn → exit at NEXT open")
            notes.append("Place a GTC stop-limit at 95% of your actual fill price after entry.")
        elif cdn_signal:
            actions.append("BUY TQQQ (C/Dn bounce signal) — enter at TOMORROW'S open")
            notes.append(f"Regime: C/Dn  |  VVC-RT={rt:.3f}  MTI={mti:.3f}")
            notes.append("TP target:  TQQQ closes ≥ 150% of your fill price (+50%)")
            notes.append("Hard stop:  TQQQ closes ≤ 95% of your fill price (-5%) → exit at NEXT day's open")
            notes.append("Exit signal: DEW Sell fires or C/Up transition → exit at NEXT open")
            notes.append("Place a GTC stop-limit at 95% of your actual fill price after entry.")
        elif qqq_signal:
            atr_mult = 2.0 if regime_i == 'C/Dn' else 1.0
            stop_est = qqq_close - atr_mult * atr14
            actions.append("BUY QQQ — enter at TOMORROW'S open")
            notes.append(f"Regime: {regime_i}  |  DEW Buy fired today  |  VVC-RT={rt:.3f}  BSR={bsr:.2f}  MTI={mti:.3f}")
            notes.append(f"ATR14 today: ${atr14:.2f}  |  ATR multiplier: {atr_mult:.0f}×  "
                         f"({'Bear' if regime_i == 'C/Dn' else 'Bull'} regime)")
            notes.append(f"Stop level ≈ ${stop_est:.2f}  (exact = your fill price − {atr_mult:.0f} × ${atr14:.2f})")
            notes.append("TP target:  cumulative +5% from entry open → exit same-day close")
            notes.append("Regime exit: VectorVest C/Dn flip → exit at NEXT open")
            notes.append("DEW exit:   DEW Sell signal fires → exit same-day close")
            # Priority 1b advisory when QQQ is about to be entered in C/Dn
            if regime_i == 'C/Dn' and 0.95 <= rt < 1.00 and mti < 0.95:
                notes.append(f"⚠ Priority 1b conditions are also met (RT={rt:.3f}, MTI={mti:.3f})")
                notes.append("  → On entry, watch: if C/Dn + RT in [0.95,1.00) + MTI<0.95 persists,")
                notes.append("    engine will switch QQQ → C/Dn TQQQ at next open.")
        elif psq_signal:
            atr_mult = 2.0 if regime_i == 'C/Dn' else 1.0
            stop_est = qqq_close + atr_mult * atr14
            actions.append("BUY SQQQ (1/3 position size) — enter at TOMORROW'S open")
            notes.append(f"Regime: {regime_i}  |  DEW Sell fired today  |  VVC-RT={rt:.3f}  BSR={bsr:.2f}  MTI={mti:.3f}")
            notes.append(f"ATR14 today: ${atr14:.2f}  |  ATR multiplier: {atr_mult:.0f}×  "
                         f"({'Bear' if regime_i == 'C/Dn' else 'Bull'} regime)")
            notes.append(f"QQQ stop level ≈ ${stop_est:.2f}  (exact = tomorrow's QQQ open + {atr_mult:.0f} × ${atr14:.2f})")
            notes.append("  → SQQQ exits when QQQ RISES to that level.")
            notes.append("TP target:  cumulative +5% SQQQ gain → exit same-day close")
            notes.append("Regime exit: VectorVest C/Up flip → exit SQQQ at NEXT open")
            notes.append("DEW exit:   DEW Buy signal fires → exit SQQQ same-day close")
        else:
            actions.append("STAY FLAT — no signal today")
            reasons = []
            if regime_i == 'C/Up':
                reasons.append(f"C/Up regime  |  cup_entered={cup_entered}")
                if cup_entered:
                    reasons.append("  Waiting for a new C/Up run to reset TQQQ entry gate")
            else:
                reasons.append("C/Dn regime")
            reasons.append(f"DEW: last_dew={last_dew}  today_signal={dew_i or 'none'}")
            reasons.append(f"Indicators: VVC-RT={rt:.3f}  MTI={mti:.3f}  BSR={bsr:.2f}  "
                           f"VIX={vix:.1f}  TQQQ-RT={trt:.3f}  TQQQ-ATR={tatr:.2f}%")
            notes.extend(reasons)

    # Current market conditions summary
    notes.append("")
    notes.append(f"Current indicators:  Regime={regime_i}  VVC-RT={rt:.3f}  MTI={mti:.3f}  "
                 f"BSR={bsr:.2f}  VIX={vix:.1f}  TQQQ-RT={trt:.3f}  TQQQ-ATR={tatr:.2f}%  ATR14=${atr14:.2f}")
    notes.append(f"Health factor (last bar): {_health_factor(vix, tatr, bsr):.3f}  "
                 f"(position sizing scale — 0.25 min, 1.00 max)")

    return actions, notes, last_date


def _health_factor(vix, tatr, bsr):
    """Compute health factor for the last bar (display only)."""
    if np.isnan(vix) or np.isnan(tatr):
        return 1.0
    raw = ((30 - vix) / 20.0
           * 8.0 / max(tatr, 4.0)
           * max(bsr, 0.5))
    return min(1.0, max(0.25, raw))


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

    print(f"Nitro++ v10 Daily Signal  {today}\n")
    print("Loading data...")
    df = load_data()

    print("Running engine...")
    trades, final_state, health_arr = run_engine(df)

    # Simulated equity: compound raw trade returns through live start
    # (reflects Step 7 baseline; health factor scales dret, not trade returns)
    equity = STARTING_CAPITAL
    hist_trades = [t for t in trades if pd.Timestamp(t['exit']) < LIVE_START]
    for t in hist_trades:
        equity *= (1 + t['ret'])

    n_qqq  = sum(1 for t in trades if t['inst'] == 'QQQ')
    n_psq  = sum(1 for t in trades if t['inst'] == 'PSQ')
    n_tqqq = sum(1 for t in trades if t['inst'] == 'TQQQ')
    print(f"Total trades through last bar: {len(trades)}  "
          f"(QQQ={n_qqq}, SQQQ={n_psq}, TQQQ={n_tqqq})")
    print(f"Historical trades through {LIVE_START.date()}: {len(hist_trades)}")
    print(f"Simulated equity at live start: ${equity:,.2f}")

    actions, notes, data_date = get_signal(df, trades, final_state)

    # Build report body
    lines = []
    lines.append("Dampier Nitro++ v10 — Daily Signal")
    lines.append(f"Data through: {data_date}  |  Signal for: tomorrow's open")
    lines.append(f"Simulated account equity: ${equity:,.2f}")
    lines.append("=" * 60)
    lines.append("")
    for a in actions:
        lines.append(f"  ▶  {a}")
    lines.append("")
    lines.append("Details:")
    for n in notes:
        lines.append(f"  {n}")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Last {min(5, len(trades))} completed trades:")
    for t in trades[-5:]:
        inst_label = 'SQQQ' if t['inst'] == 'PSQ' else t['inst']
        lines.append(f"  {t['entry']}→{t['exit']}  {inst_label:4s}  "
                     f"{t['ret']*100:+.2f}%  {t['type']}")
    lines.append("")
    lines.append("Exit fill rules (reference):")
    lines.append("  TP / ATR / DEW reversal   → same-day close fill")
    lines.append("  Regime flip / TQQQ stop   → NEXT day open fill")
    lines.append("  TQQQ TP                   → same-day close fill")
    lines.append("  Priority 1b (QQQ→TQQQ)    → both fill at NEXT day open")

    body = "\n".join(lines)
    print("\n" + body)

    # Subject line
    if any("STAY FLAT" in a for a in actions):
        subject = f"Nitro++ v10 Signal {data_date}: FLAT"
    elif any("HOLD" in a for a in actions):
        if "TQQQ" in actions[0]:
            inst_word = "C/Dn TQQQ" if final_state['cdn_tqqq_active'] else "C/Up TQQQ"
        elif "QQQ" in actions[0]:
            inst_word = "QQQ"
        else:
            inst_word = "SQQQ"
        subject = f"Nitro++ v10 Signal {data_date}: HOLD {inst_word}"
    else:
        action_word = actions[0].split("—")[0].strip() if actions else "SIGNAL"
        subject = f"Nitro++ v10 Signal {data_date}: {action_word}"

    print(f"\nSending email to {TO_EMAIL} ...")
    send_email(subject, body)
    print("Email sent.")


if __name__ == '__main__':
    main()
