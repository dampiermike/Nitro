import pandas as pd
import numpy as np
from pathlib import Path

# Dampier Nitro++ v12 — Backtest Engine
# Build guide: Dampier_Nitro_PlusPlus_Build_Guide_v12.docx
# Changes from v10:
#   - v12 C/Up block: SQQQ entries blocked during Confirmed Up regime
#   - MTI exit: while in any TQQQ trade (after entry bar), MTI < 0.75 → next-bar open exit
#   - TQQQ hard stop tightened: −6% (0.94×) from −5% (0.95×)
#   - Step 8: Pyramid sizing overlay — dret ×1.25 once TQQQ trade up ≥20%
#   - Data window: 2000-01-03 through 2026-04-01 (6606 rows)
# Targets: 200 trades (QQQ=59, SQQQ=51, TQQQ=90)
#          $681,536,320 (Step 7 baseline, float-sensitive — not a hard target)
#          → $1,656,705,340 (Step 8 final)
#          CAGR 44.80% / Sharpe 1.6790 / Max DD −22.39%

# ── File paths ───────────────────────────────────────────────────────────────
DATA_DIR  = str(Path(__file__).resolve().parent / 'data' / 'csv' / 'history')
QQQ_FILE  = f'{DATA_DIR}/qqq-from-vv.csv'
TQQQ_FILE = f'{DATA_DIR}/tqqq-from-vv.csv'
SYN_FILE  = f'{DATA_DIR}/synthetic-tqqq-ohlc-1999-2010.csv'
VV_FILE   = f'{DATA_DIR}/vectorvest-views-w3place-precision.csv'
VIX_FILE  = f'{DATA_DIR}/vix-from-yahoo.csv'
SQQQ_FILE = f'{DATA_DIR}/sqqq-from-yahoo.csv'

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and Merge
# ════════════════════════════════════════════════════════════════════════════

# ── 1a. QQQ ─────────────────────────────────────────────────────────────────
qqq_raw = pd.read_csv(QQQ_FILE, encoding='utf-8-sig')
qqq_raw['Date'] = pd.to_datetime(qqq_raw['Date'])
qqq_raw = qqq_raw.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
# CRITICAL: rename RT immediately to avoid collision with VVC-RT on merge
qqq_raw.rename(columns={'RT': 'QQQ_RT_raw'}, inplace=True)

qqq_raw['c2c_QQQ']  = qqq_raw['Close'].pct_change()
qqq_raw['o2c_QQQ']  = qqq_raw['Close'] / qqq_raw['Open'] - 1
qqq_raw['pc2o_QQQ'] = qqq_raw['Open'] / qqq_raw['Close'].shift(1) - 1

# PSQ columns (1× inverse QQQ; expense ratio only in c2c)
qqq_raw['o2c_PSQ']  = -1 * qqq_raw['o2c_QQQ']
qqq_raw['pc2o_PSQ'] = -1 * qqq_raw['pc2o_QQQ']
qqq_raw['c2c_PSQ']  = -1 * qqq_raw['c2c_QQQ'] - 0.0095 / 252

qqq_full = qqq_raw.copy()

# ── 1b. Synthetic TQQQ (pre-2010) ───────────────────────────────────────────
# Columns: Date, Open, High, Low, Close, RT_v6  — no Adj Close, use Close directly
syn = pd.read_csv(SYN_FILE)
syn['Date'] = pd.to_datetime(syn['Date'])
syn = syn.sort_values('Date').reset_index(drop=True)
syn = syn.rename(columns={
    'Open': 'tqqq_open', 'High': 'tqqq_high',
    'Low':  'tqqq_low',  'Close': 'tqqq_close', 'RT_v6': 'tqqq_rt'
})
syn = syn[['Date', 'tqqq_open', 'tqqq_high', 'tqqq_low', 'tqqq_close', 'tqqq_rt']]

# ── 1c. Real TQQQ (2010+) ────────────────────────────────────────────────────
# Columns: Date, Open, High, Low, Close, Volume, RT  — no Adj Close, use Close directly
tqqq_real = pd.read_csv(TQQQ_FILE, encoding='utf-8-sig')
tqqq_real['Date'] = pd.to_datetime(tqqq_real['Date'])
tqqq_real = tqqq_real.sort_values('Date').reset_index(drop=True)
tqqq_real = tqqq_real.rename(columns={
    'Open': 'tqqq_open', 'High': 'tqqq_high',
    'Low':  'tqqq_low',  'Close': 'tqqq_close', 'RT': 'tqqq_rt'
})
tqqq_real = tqqq_real[['Date', 'tqqq_open', 'tqqq_high', 'tqqq_low', 'tqqq_close', 'tqqq_rt']]

# ── 1d. Stitch TQQQ ──────────────────────────────────────────────────────────
# synthetic ends 2010-03-30; real starts 2010-03-31 — use explicit cutoff
cutoff = pd.Timestamp('2010-03-31')
tqqq_stitched = pd.concat([
    syn[syn['Date'] < cutoff],
    tqqq_real[tqqq_real['Date'] >= cutoff],
], ignore_index=True).sort_values('Date').reset_index(drop=True)

tqqq_stitched['tqqq_c2c']  = tqqq_stitched['tqqq_close'].pct_change()
tqqq_stitched['tqqq_o2c']  = tqqq_stitched['tqqq_close'] / tqqq_stitched['tqqq_open'] - 1
tqqq_stitched['tqqq_pc2o'] = tqqq_stitched['tqqq_open'] / tqqq_stitched['tqqq_close'].shift(1) - 1

# tqqq_atr: simple (NOT Wilder) 10-bar ATR% — computed on FULL stitched series
# so 2000-01-03 has a valid value (7.9588)
tc = tqqq_stitched['tqqq_close'].values
th = tqqq_stitched['tqqq_high'].values
tl = tqqq_stitched['tqqq_low'].values
n_t = len(tqqq_stitched)
tqqq_tr = np.zeros(n_t)
tqqq_tr[0] = th[0] - tl[0]
for i in range(1, n_t):
    tqqq_tr[i] = max(th[i] - tl[i], abs(th[i] - tc[i-1]), abs(tl[i] - tc[i-1]))
tqqq_atr_arr = np.full(n_t, np.nan)
for i in range(9, n_t):
    tqqq_atr_arr[i] = np.mean(tqqq_tr[i-9:i+1]) / tc[i] * 100.0
tqqq_stitched['tqqq_atr'] = tqqq_atr_arr

# ── 1e. VectorVest ───────────────────────────────────────────────────────────
vv = pd.read_csv(VV_FILE)
vv['Date'] = pd.to_datetime(vv['Date'])
vv = vv.sort_values('Date').reset_index(drop=True)
vv = vv.rename(columns={'VVC-RT': 'RT', 'BS Ratio': 'BSR'})
vv = vv[['Date', 'Trend', 'RT', 'MTI', 'BSR']]

# ── 1f. VIX ─────────────────────────────────────────────────────────────────
vix = pd.read_csv(VIX_FILE)
vix['Date'] = pd.to_datetime(vix['Date'], errors='coerce')
vix = vix.dropna(subset=['Date'])
vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX'})
vix['VIX'] = pd.to_numeric(vix['VIX'], errors='coerce')

# ── 1g. SQQQ (sixth raw data file) ──────────────────────────────────────────
# Loaded for completeness; PSQ return columns derived from QQQ inverse (1× exposure proxy)
sqqq_raw = pd.read_csv(SQQQ_FILE)
sqqq_raw['Date'] = pd.to_datetime(sqqq_raw['Date'], errors='coerce')
sqqq_raw = sqqq_raw.dropna(subset=['Date'])
sqqq_raw = sqqq_raw.sort_values('Date').reset_index(drop=True)

# ── 1h. Merge ────────────────────────────────────────────────────────────────
# v12: data window 2000-01-03 through 2026-04-01 (6606 rows)
qqq_window = qqq_raw[
    (qqq_raw['Date'] >= '2000-01-03') & (qqq_raw['Date'] <= '2026-04-01')
].copy()

df = vv.merge(qqq_window[['Date', 'Open', 'High', 'Low', 'Close',
                           'o2c_QQQ', 'c2c_QQQ', 'pc2o_QQQ',
                           'o2c_PSQ', 'c2c_PSQ', 'pc2o_PSQ']], on='Date', how='inner')

df = df.merge(tqqq_stitched[['Date', 'tqqq_open', 'tqqq_close', 'tqqq_rt',
                              'tqqq_atr', 'tqqq_o2c', 'tqqq_c2c', 'tqqq_pc2o']],
              on='Date', how='left')

df = df.merge(vix, on='Date', how='left')
df = df.sort_values('Date').reset_index(drop=True)

# ── 1i. Regime ───────────────────────────────────────────────────────────────
cur_reg = 'C/Up'
regimes = []
for t in df['Trend']:
    if t == 'C/Up':
        cur_reg = 'C/Up'
    elif t == 'C/Dn':
        cur_reg = 'C/Dn'
    regimes.append(cur_reg)
df['regime'] = regimes

# ── 1j. ATR14 Wilder on QQQ (on merged df, manual recursion) ─────────────────
high  = df['High'].values
low   = df['Low'].values
close = df['Close'].values

tr_q = np.zeros(len(df))
for i in range(len(df)):
    hl = high[i] - low[i]
    if i == 0:
        tr_q[i] = hl   # TR[0] has no prior close
    else:
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i]  - close[i-1])
        tr_q[i] = max(hl, hc, lc)

atr14 = np.full(len(df), np.nan)
atr14[13] = np.mean(tr_q[0:14])
for i in range(14, len(df)):
    atr14[i] = (atr14[i-1] * 13 + tr_q[i]) / 14
df['ATR14'] = atr14

# ════════════════════════════════════════════════════════════════════════════
# STOP AND VERIFY — Step 1
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 VERIFICATION")
print("=" * 60)
print(f"\ndf.shape[0] = {df.shape[0]}  (target: 6606)")

required_cols = ['Date', 'Open', 'High', 'Low', 'Close',
                 'tqqq_open', 'tqqq_close', 'tqqq_rt', 'tqqq_atr',
                 'o2c_QQQ', 'c2c_QQQ', 'pc2o_QQQ',
                 'o2c_PSQ', 'c2c_PSQ', 'pc2o_PSQ',
                 'tqqq_o2c', 'tqqq_c2c', 'tqqq_pc2o',
                 'Trend', 'regime', 'RT', 'MTI', 'BSR', 'VIX', 'ATR14']
missing = [c for c in required_cols if c not in df.columns]
print(f"Missing columns: {missing if missing else 'None'}")
print(f"\ndf['regime'].head(3).tolist() = {df['regime'].head(3).tolist()}")
print(f"  (target: ['C/Up', 'C/Up', 'C/Up'])")

atr_val = df.loc[13, 'ATR14']
print(f"\ndf.loc[13,'ATR14'] = {atr_val:.4f}  (target: 4.8201)  date: {df.loc[13,'Date'].date()}")

vix_val = df[df['Date'] == '2020-03-16']['VIX'].values
print(f"\ndf[Date=='2020-03-16']['VIX'] = {vix_val}  (target: 82.69)")

atr_first = df['tqqq_atr'].dropna().iloc[0]
print(f"\ndf['tqqq_atr'].dropna().iloc[0] = {atr_first:.4f}  (target: 7.9588)")

print("\nSpot check VVC-RT, MTI, BSR:")
for date, rt_t, mti_t, bsr_t in [
    ('2005-07-06', 1.08, 1.43, 2.49),
    ('2015-09-24', 0.85, 0.63, 0.16),
    ('2022-08-12', 1.11, 1.66, 2.84),
]:
    row = df[df['Date'] == date]
    if len(row):
        r = row.iloc[0]
        print(f"  {date}  RT={r['RT']:.4f} (t={rt_t})  "
              f"MTI={r['MTI']:.4f} (t={mti_t})  BSR={r['BSR']:.4f} (t={bsr_t})")
    else:
        print(f"  {date} NOT FOUND")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute Timing Signals
# ════════════════════════════════════════════════════════════════════════════

# Signal A: CC Regime — already in 'regime' column

# Signal B: DEW Oscillator — compute on FULL QQQ history starting July 1999
qqq_dew = qqq_full[qqq_full['Date'] >= '1999-07-01'].copy().reset_index(drop=True)
close_dew = qqq_dew['Close'].values
N_dew = len(close_dew)

# D: Detrended Price Oscillator — SMA20 lagged 11 bars, rounded to 2dp
s = pd.Series(close_dew)
sma20 = s.rolling(20).mean()
D_arr = (s - sma20.shift(11)).round(2).values

# E: Envelopes — 10-day EMA ±6%
ema10 = s.ewm(span=10, adjust=False).mean().values
E_upper = ema10 * 1.06
E_lower = ema10 * 0.94

# W: 30-period WMA, weights 1 (oldest) to 30 (newest)
weights = np.arange(1, 31, dtype=float)
W_arr = np.full(N_dew, np.nan)
for i in range(29, N_dew):
    W_arr[i] = np.dot(close_dew[i-29:i+1], weights) / weights.sum()

# DEW state machine — start at bar 41 to match spec exactly
DEW_Signal = [''] * N_dew
state        = 'Normal'
last_dew_dew = None

for i in range(41, N_dew):
    c = close_dew[i]
    d = D_arr[i]
    w = W_arr[i]
    if np.isnan(d) or np.isnan(w):
        continue

    # Step 1: candidate direction
    candidate = None
    if state == 'Normal':
        if c > w and d > 0:
            candidate = 'Buy'
        elif c < w and d < 0:
            candidate = 'Sell'
    else:  # Primed
        if c > w or d > 0:
            candidate = 'Buy'
        elif c < w or d < 0:
            candidate = 'Sell'

    # Step 2: alternating rule
    if candidate is not None and candidate == last_dew_dew:
        candidate = None

    # Step 3: if signal fired, reset to Normal
    if candidate is not None:
        DEW_Signal[i] = candidate
        last_dew_dew = candidate
        state = 'Normal'

    # Step 4: envelope pierce → Primed next bar
    if c > E_upper[i] or c < E_lower[i]:
        state = 'Primed'

qqq_dew['DEW_Signal'] = DEW_Signal
dew_merge = qqq_dew[['Date', 'DEW_Signal']]
df = df.merge(dew_merge, on='Date', how='left')
df['DEW_Signal'] = df['DEW_Signal'].fillna('')

# ════════════════════════════════════════════════════════════════════════════
# STOP AND VERIFY — Step 2
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 VERIFICATION")
print("=" * 60)

buy_cnt  = (df['DEW_Signal'] == 'Buy').sum()
sell_cnt = (df['DEW_Signal'] == 'Sell').sum()
blank    = (df['DEW_Signal'] == '').sum()
print(f"\nDEW Buy  count = {buy_cnt}   (target: 194)")
print(f"DEW Sell count = {sell_cnt}   (target: 195)")
print(f"DEW blank      = {blank}  (target: 6217)")

print("\nPre-2000 DEW signal warm-up check:")
pre2000 = qqq_dew[qqq_dew['Date'] < '2000-01-03']
pre_sigs = pre2000[pre2000['DEW_Signal'] != ''][['Date', 'DEW_Signal']]
for _, row in pre_sigs.iterrows():
    print(f"  {row['Date'].date()}  {row['DEW_Signal']}")

print("\nFirst 3 DEW signals in backtest window:")
first3 = df[df['DEW_Signal'] != ''][['Date', 'DEW_Signal', 'regime']].head(3)
for _, row in first3.iterrows():
    print(f"  {row['Date'].date()}  {row['DEW_Signal']}  (regime={row['regime']})")

print("\nTQQQ-RT v6 spot checks:")
for date, close_t, rt_t in [
    ('2005-08-25', 0.2946, 1.1216),
    ('2012-04-23', 0.5550, 1.2200),
    ('2017-10-18', 5.0930, 1.2400),
]:
    row = df[df['Date'] == date]
    if len(row):
        r = row.iloc[0]
        print(f"  {date}  tqqq_close={r['tqqq_close']:.4f} (t={close_t})  "
              f"RT={r['tqqq_rt']:.4f} (t={rt_t})")
    else:
        print(f"  {date} NOT FOUND")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Trade QQQ Long on DEW Buy
# ════════════════════════════════════════════════════════════════════════════

def perf_stats(dret_arr, dates):
    eq = 100_000 * np.cumprod(1 + dret_arr)
    final = eq[-1]
    n_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    years  = n_days / 365.25
    cagr   = (final / 100_000) ** (1 / years) - 1
    sharpe = np.mean(dret_arr) / np.std(dret_arr) * np.sqrt(252)
    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak) / peak
    max_dd = dd.min()
    return final, cagr, sharpe, max_dd

dates_arr  = df['Date'].values
N          = len(df)
trend_arr  = df['Trend'].values
regime_arr = df['regime'].values
dew_arr    = df['DEW_Signal'].values
o2c_q      = df['o2c_QQQ'].values
c2c_q      = df['c2c_QQQ'].values
pc2o_q     = df['pc2o_QQQ'].values
atr14_arr  = df['ATR14'].values
close_arr  = df['Close'].values
open_arr   = df['Open'].values

# ── Sub-step 3a: Entry signals only ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3a — QQQ Entry signals only")
print("=" * 60)

entries_3a  = []
last_dew_3a = None
for i in range(N - 1):
    trend_i  = str(trend_arr[i])
    prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    if trend_i == 'C/Up' and prev_reg != 'C/Up':
        last_dew_3a = None
    elif trend_i == 'C/Dn' and prev_reg != 'C/Dn':
        last_dew_3a = None
    if dew_arr[i] == 'Buy' and last_dew_3a != 'Buy':
        last_dew_3a = 'Buy'
        entries_3a.append({
            'signal_bar': pd.Timestamp(dates_arr[i]).date(),
            'entry_bar':  pd.Timestamp(dates_arr[i+1]).date(),
            'entry_open': open_arr[i+1],
            'regime':     regime_arr[i],
        })

print("First 10 QQQ entry signals:")
for e in entries_3a[:10]:
    print(f"  DEW Up={e['signal_bar']}  entry={e['entry_bar']}  "
          f"entry_open={e['entry_open']:.4f}  regime={e['regime']}")

# ── Sub-step 3b: TP exit only ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3b — TP exit only")
print("=" * 60)

dret_3b = np.zeros(N)
trades_3b = []
in_trade = False; entry_idx = None; cum_mult = 1.0; last_dew = None

for i in range(N):
    trend_i = str(trend_arr[i]); prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig = dew_arr[i]
    if trend_i == 'C/Up' and prev_reg != 'C/Up': last_dew = None
    elif trend_i == 'C/Dn' and prev_reg != 'C/Dn': last_dew = None
    if in_trade:
        d = o2c_q[i] if i == entry_idx else c2c_q[i]
        if np.isnan(d): d = 0.0
        cum_mult *= (1 + d)
        if cum_mult - 1 >= 0.05:
            trades_3b.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'TP'})
            dret_3b[i] = d; in_trade = False; cum_mult = 1.0; continue
        dret_3b[i] = d; continue
    if dew_sig == 'Buy' and last_dew != 'Buy' and i + 1 < N:
        in_trade = True; entry_idx = i + 1; last_dew = 'Buy'; cum_mult = 1.0

final_3b, cagr_3b, sharpe_3b, maxdd_3b = perf_stats(dret_3b, df['Date'].values)
print(f"Trades: {len(trades_3b)}  (target: 27)")
print(f"Final:  ${final_3b:,.0f}  (target: $403,371)")
print(f"CAGR:   {cagr_3b*100:.2f}%  (target: 5.46%)")
print(f"Sharpe: {sharpe_3b:.4f}  (target: 0.3420)")
print(f"Max DD: {maxdd_3b*100:.2f}%  (target: -76.76%)")
print("First 3 trades:")
for t in trades_3b[:3]:
    print(f"  entry={t['entry']}  exit={t['exit']}  {t['ret']*100:+.4f}%  {t['type']}")

# ── Sub-step 3c: TP + ATR stop ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3c — TP + ATR stop")
print("=" * 60)

dret_3c = np.zeros(N)
trades_3c = []
in_trade = False; entry_idx = None; cum_mult = 1.0; stop_level = None; last_dew = None

for i in range(N):
    trend_i = str(trend_arr[i]); prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig = dew_arr[i]
    if trend_i == 'C/Up' and prev_reg != 'C/Up': last_dew = None
    elif trend_i == 'C/Dn' and prev_reg != 'C/Dn': last_dew = None
    if in_trade:
        d = o2c_q[i] if i == entry_idx else c2c_q[i]
        if np.isnan(d): d = 0.0
        cum_mult *= (1 + d)
        if cum_mult - 1 >= 0.05:
            trades_3c.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'TP'})
            dret_3c[i] = d; in_trade = False; cum_mult = 1.0; continue
        if close_arr[i] <= stop_level:
            trades_3c.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'ATR'})
            dret_3c[i] = d; in_trade = False; cum_mult = 1.0; continue
        dret_3c[i] = d; continue
    if dew_sig == 'Buy' and last_dew != 'Buy' and i + 1 < N:
        sig_regime = regime_arr[i]
        atr_mult = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] - atr_mult * atr14_arr[i]
        in_trade = True; entry_idx = i + 1; last_dew = 'Buy'; cum_mult = 1.0

final_3c, cagr_3c, sharpe_3c, maxdd_3c = perf_stats(dret_3c, df['Date'].values)
print(f"Trades: {len(trades_3c)}  (target: 99)")
print(f"Final:  ${final_3c:,.0f}  (target: $101,220)")
print(f"CAGR:   {cagr_3c*100:.2f}%  (target: 0.05%)")
print(f"Sharpe: {sharpe_3c:.4f}  (target: 0.0521)")
print(f"Max DD: {maxdd_3c*100:.2f}%  (target: -52.21%)")
print("First 5 trades:")
for t in trades_3c[:5]:
    print(f"  entry={t['entry']}  exit={t['exit']}  {t['ret']*100:+.4f}%  {t['type']}")

# ── Sub-step 3d: TP + ATR + regime flip ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3d — TP + ATR + regime flip")
print("=" * 60)

dret_3d = np.zeros(N)
trades_3d = []
in_trade = False; entry_idx = None; cum_mult = 1.0; stop_level = None; last_dew = None

for i in range(N):
    trend_i = str(trend_arr[i]); prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig = dew_arr[i]
    if trend_i == 'C/Up' and prev_reg != 'C/Up': last_dew = None
    elif trend_i == 'C/Dn' and prev_reg != 'C/Dn': last_dew = None
    if in_trade:
        d = o2c_q[i] if i == entry_idx else c2c_q[i]
        if np.isnan(d): d = 0.0
        cum_mult *= (1 + d)
        # ISOLATION EXCEPTION: both C/Up and C/Dn transitions close QQQ through 3e
        any_flip = ((trend_i == 'C/Dn' and prev_reg == 'C/Up') or
                    (trend_i == 'C/Up' and prev_reg == 'C/Dn'))
        if cum_mult - 1 >= 0.05:
            trades_3d.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'TP'})
            dret_3d[i] = d; in_trade = False; cum_mult = 1.0
            if not any_flip: continue
        elif close_arr[i] <= stop_level:
            trades_3d.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'ATR'})
            dret_3d[i] = d; in_trade = False; cum_mult = 1.0
            if not any_flip: continue
        elif any_flip:
            if i + 1 < N:
                pc2o_next = pc2o_q[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_3d[i] = pc2o_next
                    ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_3d[i] = (1 + d) * (1 + pc2o_next) - 1
                    ret = cum_mult * (1 + pc2o_next) - 1
                trades_3d.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret': ret, 'type': 'flip'})
            in_trade = False; cum_mult = 1.0
        else:
            dret_3d[i] = d; continue
    if in_trade: continue
    if dew_sig == 'Buy' and last_dew != 'Buy' and i + 1 < N:
        sig_regime = regime_arr[i]
        atr_mult = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] - atr_mult * atr14_arr[i]
        in_trade = True; entry_idx = i + 1; last_dew = 'Buy'; cum_mult = 1.0

final_3d, cagr_3d, sharpe_3d, maxdd_3d = perf_stats(dret_3d, df['Date'].values)
print(f"Trades: {len(trades_3d)}  (target: 106)")
print(f"Final:  ${final_3d:,.2f}  (target: $143,989.85)")
print(f"CAGR:   {cagr_3d*100:.4f}%  (target: 1.3990%)")
print(f"Sharpe: {sharpe_3d:.4f}  (target: 0.2119)")
print(f"Max DD: {maxdd_3d*100:.4f}%  (target: -39.4932%)")

# ── Sub-step 3e: Full QQQ (add DEW reversal) ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3e — Full QQQ (DEW reversal added)")
print("=" * 60)

dret_3e = np.zeros(N)
trades_3e = []
in_trade = False; entry_idx = None; cum_mult = 1.0; stop_level = None; last_dew = None

for i in range(N):
    trend_i = str(trend_arr[i]); prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig = dew_arr[i]
    if trend_i == 'C/Up' and prev_reg != 'C/Up': last_dew = None
    elif trend_i == 'C/Dn' and prev_reg != 'C/Dn': last_dew = None
    if in_trade:
        d = o2c_q[i] if i == entry_idx else c2c_q[i]
        if np.isnan(d): d = 0.0
        cum_mult *= (1 + d)
        any_flip = ((trend_i == 'C/Dn' and prev_reg == 'C/Up') or
                    (trend_i == 'C/Up' and prev_reg == 'C/Dn'))
        # EXIT ORDER: TP → ATR → flip → DEW reversal
        if cum_mult - 1 >= 0.05:
            trades_3e.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'TP'})
            dret_3e[i] = d; in_trade = False; cum_mult = 1.0
            if not any_flip: continue
        elif close_arr[i] <= stop_level:
            trades_3e.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'ATR'})
            dret_3e[i] = d; in_trade = False; cum_mult = 1.0
            if not any_flip: continue
        elif any_flip:
            if i + 1 < N:
                pc2o_next = pc2o_q[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_3e[i] = pc2o_next
                    ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_3e[i] = (1 + d) * (1 + pc2o_next) - 1
                    ret = cum_mult * (1 + pc2o_next) - 1
                trades_3e.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                   'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                   'ret': ret, 'type': 'flip'})
            in_trade = False; cum_mult = 1.0
        elif dew_sig == 'Sell':
            trades_3e.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                               'exit':  pd.Timestamp(dates_arr[i]).date(),
                               'ret': cum_mult - 1, 'type': 'DEW'})
            dret_3e[i] = d; last_dew = dew_sig; in_trade = False; cum_mult = 1.0
        else:
            dret_3e[i] = d; continue
    if in_trade: continue
    if dew_sig == 'Buy' and last_dew != 'Buy' and i + 1 < N:
        sig_regime = regime_arr[i]
        atr_mult = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] - atr_mult * atr14_arr[i]
        in_trade = True; entry_idx = i + 1; last_dew = 'Buy'; cum_mult = 1.0

final_3e, cagr_3e, sharpe_3e, maxdd_3e = perf_stats(dret_3e, df['Date'].values)
same_bar_3e = sum(1 for t in trades_3e if t['entry'] == t['exit'])
print(f"Trades:     {len(trades_3e)}  (target: 162)")
print(f"Final:      ${final_3e:,.2f}  (target: $335,213.88)")
print(f"CAGR:       {cagr_3e*100:.4f}%  (target: 4.7172%)")
print(f"Sharpe:     {sharpe_3e:.4f}  (target: 0.6386)")
print(f"Max DD:     {maxdd_3e*100:.4f}%  (target: -14.2128%)")
print(f"Same-bar exits: {same_bar_3e}  (target: 16)")
print("First 5 trades:")
for t in trades_3e[:5]:
    print(f"  entry={t['entry']}  exit={t['exit']}  {t['ret']*100:+.4f}%  {t['type']}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Add SQQQ on DEW Sell
# v12: SQQQ entries blocked during C/Up regime
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — SQQQ-only verification (v12: C/Up block applied)")
print("=" * 60)

o2c_psq  = df['o2c_PSQ'].values
c2c_psq  = df['c2c_PSQ'].values
pc2o_psq = df['pc2o_PSQ'].values

dret_4 = np.zeros(N)
trades_4 = []
in_trade = False; entry_idx = None; cum_mult = 1.0; stop_level = None; last_dew = None

for i in range(N):
    trend_i = str(trend_arr[i]); prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig = dew_arr[i]
    if trend_i == 'C/Up' and prev_reg != 'C/Up': last_dew = None
    elif trend_i == 'C/Dn' and prev_reg != 'C/Dn': last_dew = None
    if in_trade:
        d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
        if np.isnan(d): d = 0.0
        cum_mult *= (1 + d)
        # Asymmetric flip for SQQQ: only C/Up flip closes a short
        cup_exit = (trend_i == 'C/Up' and prev_reg == 'C/Dn')
        if cum_mult - 1 >= 0.05:
            trades_4.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                              'exit':  pd.Timestamp(dates_arr[i]).date(),
                              'ret': cum_mult - 1, 'type': 'TP', 'inst': 'SQQQ'})
            dret_4[i] = d; in_trade = False; cum_mult = 1.0
            if not cup_exit: continue
        elif close_arr[i] >= stop_level:
            trades_4.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                              'exit':  pd.Timestamp(dates_arr[i]).date(),
                              'ret': cum_mult - 1, 'type': 'ATR', 'inst': 'SQQQ'})
            dret_4[i] = d; in_trade = False; cum_mult = 1.0
            if not cup_exit: continue
        elif cup_exit:
            if i + 1 < N:
                pc2o_next = pc2o_psq[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_4[i] = pc2o_next
                    ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_4[i] = (1 + d) * (1 + pc2o_next) - 1
                    ret = cum_mult * (1 + pc2o_next) - 1
                trades_4.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'flip', 'inst': 'SQQQ'})
            in_trade = False; cum_mult = 1.0
        elif dew_sig == 'Buy':
            trades_4.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                              'exit':  pd.Timestamp(dates_arr[i]).date(),
                              'ret': cum_mult - 1, 'type': 'DEW', 'inst': 'SQQQ'})
            dret_4[i] = d; last_dew = dew_sig; in_trade = False; cum_mult = 1.0
        else:
            dret_4[i] = d; continue
    if in_trade: continue
    # v12 C/Up block: SQQQ entries blocked during Confirmed Up regime
    if (dew_sig == 'Sell' and last_dew != 'Sell'
            and i + 1 < N and regime_arr[i] != 'C/Up'):
        sig_regime = regime_arr[i]
        atr_mult = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] + atr_mult * atr14_arr[i]
        in_trade = True; entry_idx = i + 1; last_dew = 'Sell'; cum_mult = 1.0

final_4, cagr_4, sharpe_4, maxdd_4 = perf_stats(dret_4, df['Date'].values)
print(f"Trades: {len(trades_4)}  (target: 75)")
print(f"Final:  ${final_4:,.0f}  (target: $126,237)")
print(f"CAGR:   {cagr_4*100:.2f}%  (target: 0.89%)")
print(f"Sharpe: {sharpe_4:.4f}  (target: 0.1623)")
print(f"Max DD: {maxdd_4*100:.2f}%  (target: -28.54%)")
print("First 5 trades (sorted by entry):")
for t in sorted(trades_4, key=lambda x: x['entry'])[:5]:
    print(f"  Entry {t['entry']}  {t['inst']}  {t['ret']*100:+.4f}%  {t['type']}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Identify C/Up TQQQ Signal Dates
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — C/Up TQQQ signal dates")
print("=" * 60)

rt_arr    = df['RT'].values
mti_arr   = df['MTI'].values
bsr_arr   = df['BSR'].values
vix_arr   = df['VIX'].values
trt_arr   = df['tqqq_rt'].values
tatr_arr  = df['tqqq_atr'].values

cup_signals   = []
cup_entered_5 = False
for i in range(N - 1):
    trend_i  = str(trend_arr[i])
    prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    if trend_i == 'C/Up' and prev_reg != 'C/Up':
        cup_entered_5 = False
    if (not cup_entered_5
            and regime_arr[i] == 'C/Up'
            and not np.isnan(trt_arr[i])  and trt_arr[i] < 1.40
            and not np.isnan(bsr_arr[i])  and bsr_arr[i] > 1.05
            and not np.isnan(tatr_arr[i]) and tatr_arr[i] < 7.0
            and not np.isnan(vix_arr[i])  and vix_arr[i] < 30):
        cup_entered_5 = True
        cup_signals.append({
            'signal': pd.Timestamp(dates_arr[i]).date(),
            'entry':  pd.Timestamp(dates_arr[i+1]).date(),
            'rt': trt_arr[i], 'bsr': bsr_arr[i],
            'atr': tatr_arr[i], 'vix': vix_arr[i],
        })

print(f"Total C/Up TQQQ signal bars: {len(cup_signals)}  (target: 60)")
print("First 5:")
for s in cup_signals[:5]:
    print(f"  Signal {s['signal']}  Entry {s['entry']}  "
          f"RT={s['rt']:.3f} BSR={s['bsr']:.2f} ATR={s['atr']:.2f} VIX={s['vix']:.1f}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Identify C/Dn TQQQ Signal Dates
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — C/Dn TQQQ signal dates")
print("=" * 60)

cdn_signals = []
for i in range(N - 1):
    if (regime_arr[i] == 'C/Dn'
            and dew_arr[i] == 'Buy'
            and not np.isnan(rt_arr[i])  and 0.95 <= rt_arr[i] < 1.00
            and not np.isnan(mti_arr[i]) and mti_arr[i] < 1.00):
        cdn_signals.append({
            'signal': pd.Timestamp(dates_arr[i]).date(),
            'rt': rt_arr[i], 'mti': mti_arr[i],
        })

print(f"Total C/Dn TQQQ signal bars: {len(cdn_signals)}  (target: 27)")
print("All 27 C/Dn TQQQ signal dates:")
for s in cdn_signals:
    print(f"  {s['signal']}  VVC-RT={s['rt']:.3f}  MTI={s['mti']:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Full System (baseline, no health factor / no pyramid)
# v12 targets: 200 trades (QQQ=59, SQQQ=51, TQQQ=90)
# Note: Step 7 base equity (~$681M) is float-sensitive; NOT a hard target.
# Hard targets are TQQQ isolation ($133,184,002.58) and Step 8 ($1,656,705,340).
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — Full System (baseline, no health factor)")
print("=" * 60)

tqqq_open_arr  = df['tqqq_open'].values
tqqq_close_arr = df['tqqq_close'].values
tqqq_c2c_arr   = df['tqqq_c2c'].values
tqqq_o2c_arr   = df['tqqq_o2c'].values
tqqq_pc2o_arr  = df['tqqq_pc2o'].values

dret_7   = np.zeros(N)
trades_7 = []

in_trade        = False
entry_idx       = None
cum_mult        = 1.0
stop_level      = None
tp_level        = None
tqqq_stop       = None
inst            = None      # 'QQQ', 'PSQ', 'TQQQ'
short_leg       = False
cdn_tqqq_active = False
last_dew        = None
cup_entered     = False


def cup_tqqq_fires(i):
    """C/Up TQQQ conditions — pure market check, no trade-state guard."""
    if (np.isnan(trt_arr[i]) or np.isnan(tatr_arr[i])
            or np.isnan(vix_arr[i]) or np.isnan(bsr_arr[i])):
        return False
    return (regime_arr[i] == 'C/Up'
            and not cup_entered
            and trt_arr[i] < 1.40
            and bsr_arr[i] > 1.05
            and tatr_arr[i] < 7.0
            and vix_arr[i] < 30)


def cdn_tqqq_fires(i):
    """C/Dn TQQQ conditions (Priority 2 — flat entry, DEW Buy required)."""
    if np.isnan(rt_arr[i]) or np.isnan(mti_arr[i]):
        return False
    return (regime_arr[i] == 'C/Dn'
            and dew_arr[i] == 'Buy'
            and 0.95 <= rt_arr[i] < 1.00
            and mti_arr[i] < 1.00)


def qqq_skip(i):
    """Entry filters that block QQQ long entry."""
    rt = rt_arr[i]; bsr = bsr_arr[i]; mti = mti_arr[i]
    return (0.85 <= rt < 0.90) or (bsr > 1.50) or (mti > 1.10)


def psq_skip(i):
    """Entry filters that block SQQQ short entry (RT > 0.95 AND MTI > 0.95)."""
    return rt_arr[i] > 0.95 and mti_arr[i] > 0.95


for i in range(N):
    trend_i  = str(trend_arr[i])
    prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig  = dew_arr[i]

    # ── Phase A: regime transitions ──────────────────────────────────────────
    if trend_i == 'C/Up' and prev_reg != 'C/Up':
        last_dew    = None
        cup_entered = False
    if trend_i == 'C/Dn' and prev_reg != 'C/Dn':
        last_dew = None

    # ── Priority 1a: C/Up TQQQ fires while QQQ/PSQ trade is open ────────────
    if in_trade and inst in ('QQQ', 'PSQ') and cup_tqqq_fires(i):
        if i + 1 < N:
            if inst == 'QQQ':
                d = o2c_q[i] if i == entry_idx else c2c_q[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1 + d)
                pc2o_next = pc2o_q[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_7[i] = pc2o_next; ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_7[i] = (1+d)*(1+pc2o_next)-1; ret = cum_mult * (1 + pc2o_next) - 1
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'preempt', 'inst': 'QQQ'})
            else:  # PSQ
                d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1 + d)
                pc2o_next = pc2o_psq[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_7[i] = pc2o_next; ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_7[i] = (1+d)*(1+pc2o_next)-1; ret = cum_mult * (1 + pc2o_next) - 1
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'preempt', 'inst': 'PSQ'})
            last_dew    = None
            entry_idx   = i + 1
            inst        = 'TQQQ'; short_leg = False; cdn_tqqq_active = False
            cum_mult    = 1.0
            tp_level    = tqqq_open_arr[i+1] * 1.50
            tqqq_stop   = tqqq_open_arr[i+1] * 0.94   # v12: −6% stop
            cup_entered = True
            in_trade    = True
        continue

    # ── Phase B: exit checks ─────────────────────────────────────────────────
    if in_trade:
        if inst == 'TQQQ':
            d = tqqq_o2c_arr[i] if i == entry_idx else tqqq_c2c_arr[i]
            if np.isnan(d): d = 0.0
            cum_mult *= (1 + d)

            # ── v12 MTI exit (fires before TP/stop; not on entry bar) ─────────
            if (i != entry_idx
                    and not np.isnan(mti_arr[i]) and mti_arr[i] < 0.75
                    and i + 1 < N):
                pc2o_next = tqqq_pc2o_arr[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                dret_7[i] = (1+d)*(1+pc2o_next)-1
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': cum_mult*(1+pc2o_next)-1,
                                  'type': 'MTI', 'inst': 'TQQQ'})
                last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                continue

            # TQQQ TP (+50%) — same-bar close fill
            elif tqqq_close_arr[i] >= tp_level:
                was_cdn = cdn_tqqq_active
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'TP', 'inst': 'TQQQ'})
                dret_7[i] = d
                last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                # Fall through only if currently in C/Dn regime
                if regime_arr[i] != 'C/Dn':
                    continue

            # TQQQ hard stop (−6%) — next-bar open fill
            elif tqqq_close_arr[i] <= tqqq_stop:
                if i + 1 < N:
                    pc2o_next = tqqq_pc2o_arr[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    if i == entry_idx:
                        dret_7[i] = pc2o_next; ret = cum_mult*(1+pc2o_next)-1
                    else:
                        dret_7[i] = (1+d)*(1+pc2o_next)-1; ret = cum_mult*(1+pc2o_next)-1
                    trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                      'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                      'ret': ret, 'type': 'Stop', 'inst': 'TQQQ'})
                last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                continue

            # C/Up TQQQ regime exit: fire on every C/Dn bar (value check)
            elif not cdn_tqqq_active and regime_arr[i] == 'C/Dn':
                if i + 1 < N:
                    pc2o_next = tqqq_pc2o_arr[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    if i == entry_idx:
                        dret_7[i] = pc2o_next; ret = cum_mult*(1+pc2o_next)-1
                    else:
                        dret_7[i] = (1+d)*(1+pc2o_next)-1; ret = cum_mult*(1+pc2o_next)-1
                    trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                      'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                      'ret': ret, 'type': 'Signal', 'inst': 'TQQQ'})
                last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                # fall through

            # C/Dn TQQQ exits: DEW Sell OR C/Up transition
            elif cdn_tqqq_active:
                cdn_tqqq_exit = (dew_sig == 'Sell' or
                                 (trend_i == 'C/Up' and prev_reg == 'C/Dn'))
                if cdn_tqqq_exit:
                    if i + 1 < N:
                        pc2o_next = tqqq_pc2o_arr[i+1]
                        if np.isnan(pc2o_next): pc2o_next = 0.0
                        if i == entry_idx:
                            dret_7[i] = pc2o_next; ret = cum_mult*(1+pc2o_next)-1
                        else:
                            dret_7[i] = (1+d)*(1+pc2o_next)-1; ret = cum_mult*(1+pc2o_next)-1
                        trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                          'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                          'ret': ret, 'type': 'Signal', 'inst': 'TQQQ'})
                    last_dew = None; cdn_tqqq_active = False; in_trade = False; cum_mult = 1.0
                    # fall through
                else:
                    dret_7[i] = d; continue

            else:
                dret_7[i] = d; continue

        else:  # QQQ or PSQ
            if inst == 'QQQ':
                d = o2c_q[i] if i == entry_idx else c2c_q[i]
            else:
                d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
            if np.isnan(d): d = 0.0
            cum_mult *= (1 + d)

            # ── Priority 1b: C/Dn TQQQ switch while in QQQ long ─────────────
            if (inst == 'QQQ'
                    and not cdn_tqqq_active
                    and regime_arr[i] == 'C/Dn'
                    and not np.isnan(rt_arr[i])  and 0.95 <= rt_arr[i] < 1.00
                    and not np.isnan(mti_arr[i]) and mti_arr[i] < 0.95
                    and i + 1 < N):
                pc2o_next = pc2o_q[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_7[i] = pc2o_next
                    ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_7[i] = (1 + d) * (1 + pc2o_next) - 1
                    ret = cum_mult * (1 + pc2o_next) - 1
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'p1b', 'inst': 'QQQ'})
                entry_idx       = i + 1
                inst            = 'TQQQ'; short_leg = False; cdn_tqqq_active = True
                cum_mult        = 1.0
                tp_level        = tqqq_open_arr[i+1] * 1.50
                tqqq_stop       = tqqq_open_arr[i+1] * 0.94   # v12: −6%
                last_dew        = None
                in_trade        = True
                continue

            # Asymmetric flip detection
            cdn_flip = (trend_i == 'C/Dn' and prev_reg == 'C/Up') and not short_leg
            cup_exit = (trend_i == 'C/Up' and prev_reg == 'C/Dn') and short_leg

            if inst == 'QQQ':
                pc2o_next_fn = lambda: pc2o_q[i+1] if i+1 < N else 0.0
            else:
                pc2o_next_fn = lambda: pc2o_psq[i+1] if i+1 < N else 0.0

            # TP
            if cum_mult - 1 >= 0.05:
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'TP', 'inst': inst})
                dret_7[i] = d; in_trade = False; cum_mult = 1.0
                if not cup_exit and not cdn_flip: continue
            # ATR stop
            elif ((not short_leg and close_arr[i] <= stop_level) or
                  (short_leg and close_arr[i] >= stop_level)):
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'ATR', 'inst': inst})
                dret_7[i] = d; in_trade = False; cum_mult = 1.0
                if not cup_exit and not cdn_flip: continue
            # Regime flip
            elif cdn_flip or cup_exit:
                if i + 1 < N:
                    pco = pc2o_next_fn()
                    if np.isnan(pco): pco = 0.0
                    if i == entry_idx:
                        dret_7[i] = pco; ret = cum_mult*(1+pco)-1
                    else:
                        dret_7[i] = (1+d)*(1+pco)-1; ret = cum_mult*(1+pco)-1
                    trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                      'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                      'ret': ret, 'type': 'flip', 'inst': inst})
                in_trade = False; cum_mult = 1.0
                # fall through
            # DEW reversal
            elif ((inst == 'QQQ' and dew_sig == 'Sell') or
                  (inst == 'PSQ' and dew_sig == 'Buy')):
                trades_7.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'DEW', 'inst': inst})
                dret_7[i] = d
                last_dew = dew_sig
                in_trade = False; cum_mult = 1.0
                # fall through
            else:
                dret_7[i] = d; continue

    # Guard
    if in_trade: continue

    # ── Priority 1c: C/Up TQQQ fires when flat (same-bar chain) ─────────────
    if cup_tqqq_fires(i) and i + 1 < N:
        entry_idx       = i + 1
        inst            = 'TQQQ'; short_leg = False; cdn_tqqq_active = False
        cum_mult        = 1.0
        tp_level        = tqqq_open_arr[i+1] * 1.50
        tqqq_stop       = tqqq_open_arr[i+1] * 0.94   # v12: −6%
        cup_entered     = True
        in_trade        = True
        continue

    # ── Priority 2: C/Dn TQQQ (flat entry, DEW Buy required) ────────────────
    if cdn_tqqq_fires(i) and i + 1 < N:
        entry_idx       = i + 1
        inst            = 'TQQQ'; short_leg = False; cdn_tqqq_active = True
        cum_mult        = 1.0
        tp_level        = tqqq_open_arr[i+1] * 1.50
        tqqq_stop       = tqqq_open_arr[i+1] * 0.94   # v12: −6%
        last_dew        = None
        in_trade        = True
        continue

    # ── Priority 3: DEW QQQ long ─────────────────────────────────────────────
    if dew_sig == 'Buy' and last_dew != 'Buy' and not qqq_skip(i) and i + 1 < N:
        sig_regime = regime_arr[i]
        atr_mult   = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] - atr_mult * atr14_arr[i]
        entry_idx  = i + 1; inst = 'QQQ'; short_leg = False
        last_dew   = 'Buy'; cum_mult = 1.0; in_trade = True
        continue

    # ── Priority 4: DEW SQQQ short (v12: blocked during C/Up) ───────────────
    if (dew_sig == 'Sell' and last_dew != 'Sell'
            and not psq_skip(i) and i + 1 < N
            and regime_arr[i] != 'C/Up'):        # v12 C/Up block
        sig_regime = regime_arr[i]
        atr_mult   = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] + atr_mult * atr14_arr[i]
        entry_idx  = i + 1; inst = 'PSQ'; short_leg = True
        last_dew   = 'Sell'; cum_mult = 1.0; in_trade = True
        continue

# ── Step 7 results (baseline, no health factor) ──────────────────────────────
final_7, cagr_7, sharpe_7, maxdd_7 = perf_stats(dret_7, df['Date'].values)
n_qqq  = sum(1 for t in trades_7 if t['inst'] == 'QQQ')
n_psq  = sum(1 for t in trades_7 if t['inst'] == 'PSQ')
n_tqqq = sum(1 for t in trades_7 if t['inst'] == 'TQQQ')
same_bar_7 = sum(1 for t in trades_7
                 if t['entry'] == t['exit'])
chains_7   = sum(1 for ci in range(len(trades_7) - 1)
                 if trades_7[ci]['exit'] == trades_7[ci+1]['entry'])

print(f"\nTrades:  {len(trades_7)}  (target: 200)")
print(f"  QQQ={n_qqq} (target 59), SQQQ={n_psq} (target 51), TQQQ={n_tqqq} (target 90)")
print(f"Same-bar exits:  {same_bar_7}  (target: 8)")
print(f"Same-bar chains: {chains_7}  (target: 39)")
print(f"Final:   ${final_7:,.0f}  (note: float-sensitive, ~$681M; not a hard target)")
print(f"CAGR:    {cagr_7*100:.2f}%  (target: 39.98%)")
print(f"Sharpe:  {sharpe_7:.4f}  (target: 1.6805)")
print(f"Max DD:  {maxdd_7*100:.2f}%  (target: -22.39%)")
print("\nFirst 10 trades:")
for t in trades_7[:10]:
    print(f"  {t['entry']}  {t['inst']:4s}  {t['ret']*100:+.4f}%  {t['type']}")

# ════════════════════════════════════════════════════════════════════════════
# SUB-STEP 6b — TQQQ trades compounded in isolation
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUB-STEP 6b — TQQQ trades in isolation")
print("=" * 60)

tqqq_only = [t for t in trades_7 if t['inst'] == 'TQQQ']
eq_6b = 100_000.0
for t in tqqq_only:
    eq_6b *= (1 + t['ret'])

tp6   = sum(1 for t in tqqq_only if t['type'] == 'TP')
stop6 = sum(1 for t in tqqq_only if t['type'] == 'Stop')
sig6  = sum(1 for t in tqqq_only if t['type'] == 'Signal')
mti6  = sum(1 for t in tqqq_only if t['type'] == 'MTI')

print(f"\nTotal TQQQ trades:   {len(tqqq_only)}  (target: 90)")
print(f"Final compounded:    ${eq_6b:,.2f}  (target: $133,184,002.58)")
print(f"\nExit type breakdown:")
print(f"  Take Profit:        {tp6}   (target: 10)")
print(f"  ATR Stop:           {stop6}  (target: 23)")
print(f"  Confirmed Down:     {sig6}  (target: 40 — includes DEW Down exits)")
print(f"  MTI Exit:           {mti6}   (target: 7)")
print(f"  Total:              {tp6+stop6+sig6+mti6}  (target: 90)")

print("\nRunning compound (first 5 TQQQ trades, starting $100,000):")
eq_run = 100_000.0
for idx, t in enumerate(tqqq_only[:5]):
    eq_run *= (1 + t['ret'])
    print(f"  {idx+1:2d}  {t['entry']}→{t['exit']}  {t['ret']*100:+.4f}%  "
          f"{t['type']:8s}  ${eq_run:,.2f}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — Volatility-Adaptive Defensive Sizing + Pyramid Overlay
# Health factor: scales dret only — does NOT touch cum_mult or exit logic.
# Pyramid: dret ×1.25 once any TQQQ trade is up ≥20%; resets on new entry.
# Trade count MUST remain 200. If it changes, health/pyramid touched cum_mult.
# Target: $1,656,705,340 / 44.80% CAGR / 1.6790 Sharpe / −22.39% Max DD
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Full System with Health Factor + Pyramid Sizing")
print("=" * 60)

# Pre-compute health factor array (one value per bar)
health_arr = np.ones(N)
for i in range(N):
    vix_i = vix_arr[i]
    atr_i = tatr_arr[i]
    bsr_i = bsr_arr[i]
    if np.isnan(vix_i) or np.isnan(atr_i) or np.isnan(bsr_i):
        health_arr[i] = 1.0
    else:
        raw = ((30 - vix_i) / 20.0
               * 8.0 / max(atr_i, 4.0)
               * max(bsr_i, 0.5))
        health_arr[i] = min(1.0, max(0.25, raw))

# Re-run full engine with health factor AND pyramid applied to dret
dret_8   = np.zeros(N)
trades_8 = []

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
pyramid_active  = False   # v12: TQQQ pyramid overlay

for i in range(N):
    trend_i  = str(trend_arr[i])
    prev_reg = regime_arr[i-1] if i > 0 else 'C/Up'
    dew_sig  = dew_arr[i]
    hf       = health_arr[i]

    # ── Phase A ──────────────────────────────────────────────────────────────
    if trend_i == 'C/Up' and prev_reg != 'C/Up':
        last_dew    = None
        cup_entered = False
    if trend_i == 'C/Dn' and prev_reg != 'C/Dn':
        last_dew = None

    # ── Priority 1a: C/Up TQQQ preempts open QQQ/PSQ ────────────────────────
    if in_trade and inst in ('QQQ', 'PSQ') and cup_tqqq_fires(i):
        if i + 1 < N:
            if inst == 'QQQ':
                d = o2c_q[i] if i == entry_idx else c2c_q[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1 + d)
                pc2o_next = pc2o_q[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_8[i] = pc2o_next * hf; ret = cum_mult*(1+pc2o_next)-1
                else:
                    dret_8[i] = ((1+d)*(1+pc2o_next)-1) * hf; ret = cum_mult*(1+pc2o_next)-1
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'preempt', 'inst': 'QQQ'})
            else:
                d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
                if np.isnan(d): d = 0.0
                cum_mult *= (1 + d)
                pc2o_next = pc2o_psq[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_8[i] = pc2o_next * hf; ret = cum_mult*(1+pc2o_next)-1
                else:
                    dret_8[i] = ((1+d)*(1+pc2o_next)-1) * hf; ret = cum_mult*(1+pc2o_next)-1
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'preempt', 'inst': 'PSQ'})
            last_dew        = None
            entry_idx       = i + 1
            inst            = 'TQQQ'; short_leg = False; cdn_tqqq_active = False
            cum_mult        = 1.0
            tp_level        = tqqq_open_arr[i+1] * 1.50
            tqqq_stop       = tqqq_open_arr[i+1] * 0.94
            cup_entered     = True
            in_trade        = True
            pyramid_active  = False
        continue

    # ── Phase B ──────────────────────────────────────────────────────────────
    if in_trade:
        if inst == 'TQQQ':
            d = tqqq_o2c_arr[i] if i == entry_idx else tqqq_c2c_arr[i]
            if np.isnan(d): d = 0.0
            cum_mult *= (1 + d)

            # v12 pyramid: set once per trade when up ≥20%; never reset mid-trade
            if not pyramid_active and cum_mult - 1 >= 0.20:
                pyramid_active = True
            psize = 1.25 if pyramid_active else 1.0

            # v12 MTI exit (before TP/stop; not on entry bar)
            if (i != entry_idx
                    and not np.isnan(mti_arr[i]) and mti_arr[i] < 0.75
                    and i + 1 < N):
                pc2o_next = tqqq_pc2o_arr[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                dret_8[i] = ((1+d)*(1+pc2o_next)-1) * hf * psize
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': cum_mult*(1+pc2o_next)-1,
                                  'type': 'MTI', 'inst': 'TQQQ'})
                last_dew = None; cdn_tqqq_active = False; in_trade = False
                cum_mult = 1.0; pyramid_active = False
                continue

            elif tqqq_close_arr[i] >= tp_level:
                was_cdn = cdn_tqqq_active
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'TP', 'inst': 'TQQQ'})
                dret_8[i] = d * hf * psize
                last_dew = None; cdn_tqqq_active = False; in_trade = False
                cum_mult = 1.0; pyramid_active = False
                if regime_arr[i] != 'C/Dn':
                    continue

            elif tqqq_close_arr[i] <= tqqq_stop:
                if i + 1 < N:
                    pc2o_next = tqqq_pc2o_arr[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    if i == entry_idx:
                        dret_8[i] = pc2o_next * hf * psize; ret = cum_mult*(1+pc2o_next)-1
                    else:
                        dret_8[i] = ((1+d)*(1+pc2o_next)-1) * hf * psize
                        ret = cum_mult*(1+pc2o_next)-1
                    trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                      'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                      'ret': ret, 'type': 'Stop', 'inst': 'TQQQ'})
                last_dew = None; cdn_tqqq_active = False; in_trade = False
                cum_mult = 1.0; pyramid_active = False
                continue

            elif not cdn_tqqq_active and regime_arr[i] == 'C/Dn':
                if i + 1 < N:
                    pc2o_next = tqqq_pc2o_arr[i+1]
                    if np.isnan(pc2o_next): pc2o_next = 0.0
                    if i == entry_idx:
                        dret_8[i] = pc2o_next * hf * psize; ret = cum_mult*(1+pc2o_next)-1
                    else:
                        dret_8[i] = ((1+d)*(1+pc2o_next)-1) * hf * psize
                        ret = cum_mult*(1+pc2o_next)-1
                    trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                      'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                      'ret': ret, 'type': 'Signal', 'inst': 'TQQQ'})
                last_dew = None; cdn_tqqq_active = False; in_trade = False
                cum_mult = 1.0; pyramid_active = False
                # fall through

            elif cdn_tqqq_active:
                cdn_tqqq_exit = (dew_sig == 'Sell' or
                                 (trend_i == 'C/Up' and prev_reg == 'C/Dn'))
                if cdn_tqqq_exit:
                    if i + 1 < N:
                        pc2o_next = tqqq_pc2o_arr[i+1]
                        if np.isnan(pc2o_next): pc2o_next = 0.0
                        if i == entry_idx:
                            dret_8[i] = pc2o_next * hf * psize; ret = cum_mult*(1+pc2o_next)-1
                        else:
                            dret_8[i] = ((1+d)*(1+pc2o_next)-1) * hf * psize
                            ret = cum_mult*(1+pc2o_next)-1
                        trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                          'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                          'ret': ret, 'type': 'Signal', 'inst': 'TQQQ'})
                    last_dew = None; cdn_tqqq_active = False; in_trade = False
                    cum_mult = 1.0; pyramid_active = False
                    # fall through
                else:
                    dret_8[i] = d * hf * psize; continue

            else:
                dret_8[i] = d * hf * psize; continue

        else:  # QQQ or PSQ
            if inst == 'QQQ':
                d = o2c_q[i] if i == entry_idx else c2c_q[i]
            else:
                d = o2c_psq[i] if i == entry_idx else c2c_psq[i]
            if np.isnan(d): d = 0.0
            cum_mult *= (1 + d)

            # Priority 1b: C/Dn TQQQ switch while in QQQ long
            if (inst == 'QQQ'
                    and not cdn_tqqq_active
                    and regime_arr[i] == 'C/Dn'
                    and not np.isnan(rt_arr[i])  and 0.95 <= rt_arr[i] < 1.00
                    and not np.isnan(mti_arr[i]) and mti_arr[i] < 0.95
                    and i + 1 < N):
                pc2o_next = pc2o_q[i+1]
                if np.isnan(pc2o_next): pc2o_next = 0.0
                if i == entry_idx:
                    dret_8[i] = pc2o_next * hf
                    ret = cum_mult * (1 + pc2o_next) - 1
                else:
                    dret_8[i] = ((1 + d) * (1 + pc2o_next) - 1) * hf
                    ret = cum_mult * (1 + pc2o_next) - 1
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                  'ret': ret, 'type': 'p1b', 'inst': 'QQQ'})
                entry_idx       = i + 1
                inst            = 'TQQQ'; short_leg = False; cdn_tqqq_active = True
                cum_mult        = 1.0
                tp_level        = tqqq_open_arr[i+1] * 1.50
                tqqq_stop       = tqqq_open_arr[i+1] * 0.94
                last_dew        = None
                in_trade        = True
                pyramid_active  = False
                continue

            cdn_flip = (trend_i == 'C/Dn' and prev_reg == 'C/Up') and not short_leg
            cup_exit = (trend_i == 'C/Up' and prev_reg == 'C/Dn') and short_leg

            if inst == 'QQQ':
                pc2o_next_fn = lambda: pc2o_q[i+1] if i+1 < N else 0.0
            else:
                pc2o_next_fn = lambda: pc2o_psq[i+1] if i+1 < N else 0.0

            if cum_mult - 1 >= 0.05:
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'TP', 'inst': inst})
                dret_8[i] = d * hf; in_trade = False; cum_mult = 1.0
                if not cup_exit and not cdn_flip: continue
            elif ((not short_leg and close_arr[i] <= stop_level) or
                  (short_leg and close_arr[i] >= stop_level)):
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'ATR', 'inst': inst})
                dret_8[i] = d * hf; in_trade = False; cum_mult = 1.0
                if not cup_exit and not cdn_flip: continue
            elif cdn_flip or cup_exit:
                if i + 1 < N:
                    pco = pc2o_next_fn()
                    if np.isnan(pco): pco = 0.0
                    if i == entry_idx:
                        dret_8[i] = pco * hf; ret = cum_mult*(1+pco)-1
                    else:
                        dret_8[i] = ((1+d)*(1+pco)-1) * hf; ret = cum_mult*(1+pco)-1
                    trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                      'exit':  pd.Timestamp(dates_arr[i+1]).date(),
                                      'ret': ret, 'type': 'flip', 'inst': inst})
                in_trade = False; cum_mult = 1.0
                # fall through
            elif ((inst == 'QQQ' and dew_sig == 'Sell') or
                  (inst == 'PSQ' and dew_sig == 'Buy')):
                trades_8.append({'entry': pd.Timestamp(dates_arr[entry_idx]).date(),
                                  'exit':  pd.Timestamp(dates_arr[i]).date(),
                                  'ret': cum_mult - 1, 'type': 'DEW', 'inst': inst})
                dret_8[i] = d * hf
                last_dew = dew_sig
                in_trade = False; cum_mult = 1.0
                # fall through
            else:
                dret_8[i] = d * hf; continue

    if in_trade: continue

    # Priority 1c: C/Up TQQQ flat chain
    if cup_tqqq_fires(i) and i + 1 < N:
        entry_idx      = i + 1
        inst           = 'TQQQ'; short_leg = False; cdn_tqqq_active = False
        cum_mult       = 1.0
        tp_level       = tqqq_open_arr[i+1] * 1.50
        tqqq_stop      = tqqq_open_arr[i+1] * 0.94
        cup_entered    = True
        in_trade       = True
        pyramid_active = False
        continue

    # Priority 2: C/Dn TQQQ flat
    if cdn_tqqq_fires(i) and i + 1 < N:
        entry_idx      = i + 1
        inst           = 'TQQQ'; short_leg = False; cdn_tqqq_active = True
        cum_mult       = 1.0
        tp_level       = tqqq_open_arr[i+1] * 1.50
        tqqq_stop      = tqqq_open_arr[i+1] * 0.94
        last_dew       = None
        in_trade       = True
        pyramid_active = False
        continue

    # Priority 3: DEW QQQ long
    if dew_sig == 'Buy' and last_dew != 'Buy' and not qqq_skip(i) and i + 1 < N:
        sig_regime = regime_arr[i]
        atr_mult   = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] - atr_mult * atr14_arr[i]
        entry_idx  = i + 1; inst = 'QQQ'; short_leg = False
        last_dew   = 'Buy'; cum_mult = 1.0; in_trade = True
        continue

    # Priority 4: DEW SQQQ short (v12: blocked during C/Up)
    if (dew_sig == 'Sell' and last_dew != 'Sell'
            and not psq_skip(i) and i + 1 < N
            and regime_arr[i] != 'C/Up'):        # v12 C/Up block
        sig_regime = regime_arr[i]
        atr_mult   = 2.0 if sig_regime == 'C/Dn' else 1.0
        stop_level = open_arr[i+1] + atr_mult * atr14_arr[i]
        entry_idx  = i + 1; inst = 'PSQ'; short_leg = True
        last_dew   = 'Sell'; cum_mult = 1.0; in_trade = True
        continue

# ── Step 8 final results ─────────────────────────────────────────────────────
final_8, cagr_8, sharpe_8, maxdd_8 = perf_stats(dret_8, df['Date'].values)
n8_qqq  = sum(1 for t in trades_8 if t['inst'] == 'QQQ')
n8_psq  = sum(1 for t in trades_8 if t['inst'] == 'PSQ')
n8_tqqq = sum(1 for t in trades_8 if t['inst'] == 'TQQQ')
same_bar_8 = sum(1 for t in trades_8 if t['entry'] == t['exit'])
chains_8   = sum(1 for ci in range(len(trades_8) - 1)
                 if trades_8[ci]['exit'] == trades_8[ci+1]['entry'])

print(f"\nTrades:  {len(trades_8)}  (target: 200)  ← MUST match Step 7")
print(f"  QQQ={n8_qqq} (target 59), SQQQ={n8_psq} (target 51), TQQQ={n8_tqqq} (target 90)")
print(f"Same-bar exits:  {same_bar_8}  (target: 8)")
print(f"Same-bar chains: {chains_8}  (target: 39)")
print(f"Final:   ${final_8:,.0f}  (target: $1,656,705,340)")
print(f"CAGR:    {cagr_8*100:.2f}%  (target: 44.80%)")
print(f"Sharpe:  {sharpe_8:.4f}  (target: 1.6790)")
print(f"Max DD:  {maxdd_8*100:.2f}%  (target: -22.39%)")
print("\nFirst 10 trades:")
for t in trades_8[:10]:
    print(f"  {t['entry']}  {t['inst']:4s}  {t['ret']*100:+.4f}%  {t['type']}")

print("\n" + "=" * 60)
print("Dampier Nitro++ v12  ·  Confidential")
print("Backtest does not guarantee future results.")
print("=" * 60)
