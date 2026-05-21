#!/usr/bin/env python3
"""
Daily VectorVest end-of-day fetch — pure REST, no browser.

SOURCE 1 — Per-stock OHLCV + RT  (api.vectorvest.com/MarketData/v3)
  For TQQQ and QQQ: resolves the symbol's StockId via /quote, then pulls a
  rolling year of daily bars from /quotehistory — Open, High, Low, Close
  (LastPrice), Volume (TotalVolume) and RT (RelativeTiming).
  Merges into:
    qqq-from-vv.csv     (Date, Open, High, Low, Close, Volume, RT)
    tqqq-from-vv.csv    (Date, Open, High, Low, Close, Volume, RT)

  This replaces the old StockViewer browser scrape, which broke and — worse —
  silently wrote mis-dated bars (stamping the prior day's data with today's
  date).  Every /quotehistory row carries an authoritative TradingDate, so the
  merge is purely date-keyed and self-heals any mis-dated/stale rows inside the
  reconcile window (see RECONCILE_DAYS).

SOURCE 2 — Market timing  (api.vectorvest.com/MarketData/v3/markettiming/US)
  Market-wide indicators: MTI, Trend, % Buys, % Sells, BS Ratio, CG-Price,
  CG-RT, CG-BSR, VVC-Price, VVC-RT.  Merges new rows into:
    vectorvest-views-w3place-precision.csv

Both sources authenticate with the same Bearer token from get_vv_token().

Usage:
    python3 fetch_vv_daily.py
    python3 fetch_vv_daily.py --symbols "TQQQ, QQQ"
    python3 fetch_vv_daily.py --skip-stocks      # timing only
    python3 fetch_vv_daily.py --skip-timing      # OHLCV/RT only
"""
import os, sys, argparse, base64
from datetime import datetime, date
from pathlib import Path

import requests
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
NITRO_DIR = Path(__file__).resolve().parent
DAILY_DIR = NITRO_DIR / 'data' / 'csv' / 'daily'
HIST_DIR  = NITRO_DIR / 'data' / 'csv' / 'history'
DAILY_DIR.mkdir(parents=True, exist_ok=True)

DATE_FMT = '%-m/%-d/%y'   # M/D/YY — matches existing history file format

# ── VectorVest REST endpoints ─────────────────────────────────────────────────
VV_TOKEN_URL  = 'https://www.vectorvest.com/identity2/issue/simple'
VV_API        = 'https://api.vectorvest.com/MarketData/v3'
VV_TIMING_URL = f'{VV_API}/markettiming/US'

# Symbol → from-vv history file.  Only symbols listed here can be fetched.
OHLCV_FILES = {
    'QQQ':  HIST_DIR / 'qqq-from-vv.csv',
    'TQQQ': HIST_DIR / 'tqqq-from-vv.csv',
}
OHLCV_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RT']

# /quotehistory is treated as authoritative for the most recent N trading days:
# rows in this window are overwritten with API data, which self-heals mis-dated
# or stale bars.  Older API dates are only ever appended-if-missing — existing
# rows older than the window are never modified.
RECONCILE_DAYS = 10

VV_VIEWS_FILE = HIST_DIR / 'vectorvest-views-w3place-precision.csv'
VV_VIEWS_COLS = ['Date', 'VVC-Price', 'VVC-RT', '% Buys', '% Sells',
                 'BS Ratio', 'CG-Price', 'CG-RT', 'CG-BSR', 'MTI', 'Trend']


# ── Small coercion helpers ────────────────────────────────────────────────────

def _round(v, nd=4):
    """Float-coerce and round; '' on failure (keeps a stray null out of the CSV)."""
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return ''


def _int(v):
    """Int-coerce; '' on failure."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return ''


# ── Date helpers ──────────────────────────────────────────────────────────────

def trading_day_from_timing(timing_df: pd.DataFrame) -> date:
    """Most recent date in the timing table — authoritative trading day
    (correctly handles market holidays a weekday check would miss)."""
    return timing_df['Date'].max().date()


# ── History load/save ─────────────────────────────────────────────────────────

def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    return df


def save_history(df: pd.DataFrame, path: Path):
    out = df.copy()
    out['Date'] = out['Date'].dt.strftime(DATE_FMT)
    bom = path.name in ('qqq-from-vv.csv', 'tqqq-from-vv.csv')
    out.to_csv(path, index=False, encoding='utf-8-sig' if bom else 'utf-8')


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_vv_token(email: str, password: str) -> str:
    """Authenticate with VectorVest and return a Bearer token. No browser needed.
    The same token unlocks /markettiming, /quote and /quotehistory."""
    creds = base64.b64encode(f"{email}:{password}".encode()).decode()
    resp = requests.get(
        VV_TOKEN_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Accept":        "application/json",
            "User-Agent":    "Mozilla/5.0",
            "Origin":        "https://app.vectorvest.com",
        },
        params={"realm": "https://www.vectorvest.com", "tokenType": "jwt"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── SOURCE 1: per-stock OHLCV + RT via REST ───────────────────────────────────

def resolve_stock_id(token: str, symbol: str) -> str:
    """Resolve a ticker to its VectorVest StockId via /quote (also logs today's RT
    as a sanity cross-check against the /quotehistory series)."""
    resp = requests.get(
        f"{VV_API}/quote",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=[
            ("fields", "StockId"), ("fields", "Symbol"),
            ("fields", "LastPrice"), ("fields", "RelativeTiming"),
            ("symbol", symbol), ("region", "US"),
            ("requestToUnlockProprietaryQuote", "true"),
        ],
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    row = data[0] if isinstance(data, list) else data
    stock_id = row.get("StockId")
    if not stock_id:
        raise RuntimeError(f"/quote returned no StockId for {symbol}: {row}")
    rt = (row.get("ProprietaryData") or {}).get("RelativeTiming")
    print(f"  {symbol}: StockId={stock_id}  (quote RT={rt}, Last={row.get('LastPrice')})")
    return stock_id


def fetch_ohlcv_rt_history(token: str, stock_id: str) -> pd.DataFrame:
    """Pull ~1 year of daily OHLCV + RT for a StockId.
    Returns a DataFrame with columns Date, Open, High, Low, Close, Volume, RT
    sorted ascending by Date."""
    resp = requests.get(
        f"{VV_API}/quotehistory/{stock_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=[
            ("fields", "TradingDate"), ("fields", "OpenPrice"),
            ("fields", "HighPrice"), ("fields", "LowPrice"),
            ("fields", "LastPrice"), ("fields", "TotalVolume"),
            ("fields", "RelativeTiming"),
            ("barFrequency", "Daily"),
            ("rangeFrequency", "Yearly"), ("rangePeriod", "1"),
        ],
        timeout=25,
    )
    resp.raise_for_status()
    items = resp.json().get("HistoryData", [])
    if not items:
        raise RuntimeError(f"/quotehistory returned no HistoryData for {stock_id}")

    rows = []
    for it in items:
        td = it.get("TradingDate")
        if not td:
            continue
        rows.append({
            "Date":   pd.Timestamp(pd.to_datetime(td).date()),
            "Open":   _round(it.get("OpenPrice")),
            "High":   _round(it.get("HighPrice")),
            "Low":    _round(it.get("LowPrice")),
            "Close":  _round(it.get("LastPrice")),
            "Volume": _int(it.get("TotalVolume")),
            "RT":     _round(it.get("RelativeTiming")),
        })
    return pd.DataFrame(rows)[OHLCV_COLS].sort_values("Date").reset_index(drop=True)


def merge_ohlcv_history(api_df: pd.DataFrame, hist_path: Path) -> tuple[int, int]:
    """Reconcile API OHLCV+RT into a from-vv history file.

    The most recent RECONCILE_DAYS trading days are overwritten with API data
    (self-heals mis-dated/stale bars).  Older API dates are appended only when
    the history file is missing them; pre-existing older rows are never touched.

    Returns (rows_added, rows_corrected)."""
    hist = load_history(hist_path)
    hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
    api = api_df.copy()
    api['Date'] = pd.to_datetime(api['Date']).dt.normalize()

    recent = set(api['Date'].tail(RECONCILE_DAYS))
    base = hist[~hist['Date'].isin(recent)]
    base_dates = set(base['Date'])
    # Bring in: everything in the reconcile window + any older API date the file lacks.
    api_bring = api[api['Date'].isin(recent) | ~api['Date'].isin(base_dates)]

    merged = pd.concat([base[OHLCV_COLS], api_bring[OHLCV_COLS]], ignore_index=True)
    merged = (merged.drop_duplicates('Date', keep='last')
                    .sort_values('Date').reset_index(drop=True))

    # ── Diff vs the pre-merge file ────────────────────────────────────────────
    old = {r['Date']: r for _, r in hist.iterrows()}
    cmp_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RT']
    added, corrected = [], []
    for _, r in merged.iterrows():
        d = r['Date']
        if d not in old:
            added.append(d)
        elif any(str(old[d][c]) != str(r[c]) for c in cmp_cols):
            corrected.append(d)

    save_history(merged, hist_path)

    print(f"  ✅ {hist_path.name}: {len(merged)} rows  "
          f"(latest {merged['Date'].max().date()})")
    if added:
        print(f"     added {len(added)}: "
              f"{[d.date().isoformat() for d in sorted(added)]}")
    if corrected:
        print(f"     corrected {len(corrected)}: "
              f"{[d.date().isoformat() for d in sorted(corrected)]}")
    if not added and not corrected:
        print(f"     no changes — already current")
    return len(added), len(corrected)


# ── SOURCE 2: VectorVest market timing via REST ───────────────────────────────

def fetch_timing_table(token: str, num_days: int = 10) -> pd.DataFrame:
    """Fetch market timing via the VectorVest REST API.

    Returns a DataFrame with columns matching vectorvest-views-w3place-precision.csv:
      Date, VVC-Price, VVC-RT, % Buys, % Sells, BS Ratio,
      CG-Price, CG-RT, CG-BSR, MTI, Trend
    """
    resp = requests.get(
        VV_TIMING_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={
            "includeAtAGlance": "true",
            "includeColorGuard": "true",
            "useExtendedHours": "false",
            "numDaysColorGuard": num_days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    at_a_glance = data.get("AtAGlance", {})
    items       = data.get("ColorGuard", {}).get("Items", [])
    if not items:
        raise RuntimeError("VectorVest REST API returned no ColorGuard items")

    today_iso = date.today().isoformat()
    buys_pct  = round(at_a_glance.get("BuysPercentage",  0) * 100, 1)
    sells_pct = round(at_a_glance.get("SellsPercentage", 0) * 100, 1)

    rows = []
    for item in items:
        trading_date = pd.to_datetime(item["TradingDate"]).date()
        is_today     = (trading_date.isoformat() == today_iso)
        rows.append({
            "Date":      pd.Timestamp(trading_date),
            "VVC-Price": round(item["Price"], 3),
            "VVC-RT":    item["RelativeTiming"],
            "% Buys":    buys_pct  if is_today else "",
            "% Sells":   sells_pct if is_today else "",
            "BS Ratio":  item["BuySellRatio"],
            "CG-Price":  item["PriceColor"],
            "CG-RT":     item["RelativeTimingColor"],
            "CG-BSR":    item["BuySellRatioColor"],
            "MTI":       item["MarketTimingIndicator"],
            "Trend":     item["Trend"],
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    print(f"  Fetched {len(df)} rows  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


def merge_vv_views_from_timing(timing_df: pd.DataFrame):
    """Append new rows from the timing table into the VV views history file."""
    df = load_history(VV_VIEWS_FILE)
    existing = set(df['Date'])
    new_rows = timing_df[~timing_df['Date'].isin(existing)].copy()

    if new_rows.empty:
        print(f"  {VV_VIEWS_FILE.name}: no new rows from timing table")
        return

    for col in VV_VIEWS_COLS:
        if col not in new_rows.columns and col != 'Date':
            new_rows[col] = ''

    new_rows = new_rows[VV_VIEWS_COLS].copy()
    df = pd.concat([df, new_rows], ignore_index=True).sort_values('Date').reset_index(drop=True)
    save_history(df, VV_VIEWS_FILE)
    dates_added = sorted(new_rows['Date'].dt.date.tolist())
    print(f"  ✅ {VV_VIEWS_FILE.name}: added {len(new_rows)} row(s) from timing table: {dates_added}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="TQQQ, QQQ",
                    help="Comma-separated symbols for the OHLCV+RT fetch")
    ap.add_argument("--skip-timing", action="store_true",
                    help="Skip the market-timing REST fetch")
    ap.add_argument("--skip-stocks", action="store_true",
                    help="Skip the per-stock OHLCV+RT fetch")
    args = ap.parse_args()

    email    = os.getenv("VECTORVEST_EMAIL")
    password = os.getenv("VECTORVEST_PASSWORD")
    if not email or not password:
        sys.exit("Set VECTORVEST_EMAIL and VECTORVEST_PASSWORD env vars.")

    date_str = date.today().strftime("%Y-%m-%d")
    print(f"[{datetime.now().isoformat()}] fetch_vv_daily  {date_str}\n", flush=True)

    token = get_vv_token(email, password)

    # ── SOURCE 2: market timing ───────────────────────────────────────────────
    if not args.skip_timing:
        print("── VectorVest market timing (REST) ───────────────────────────")
        try:
            timing_df = fetch_timing_table(token, num_days=10)

            snap = timing_df.copy()
            snap['Date'] = snap['Date'].dt.strftime(DATE_FMT)
            snap_path = DAILY_DIR / f"timing_{date_str}.csv"
            snap.to_csv(snap_path, index=False)
            print(f"  Snapshot → {snap_path.name}")

            merge_vv_views_from_timing(timing_df)
            print(f"  Trading day from API: {trading_day_from_timing(timing_df)}")
        except Exception as e:
            print(f"  ⚠️  timing fetch failed: {e}")
        print()

    # ── SOURCE 1: per-stock OHLCV + RT ────────────────────────────────────────
    if not args.skip_stocks:
        print("── Per-stock OHLCV + RT (REST /quotehistory) ─────────────────")
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        for sym in symbols:
            hist_path = OHLCV_FILES.get(sym)
            if hist_path is None:
                print(f"  ⚠️  {sym}: no history file mapped — skipping")
                continue
            try:
                stock_id = resolve_stock_id(token, sym)
                api_df   = fetch_ohlcv_rt_history(token, stock_id)
                print(f"  {sym}: fetched {len(api_df)} daily bars "
                      f"({api_df['Date'].min().date()} → {api_df['Date'].max().date()})")

                snap = api_df.tail(RECONCILE_DAYS).copy()
                snap['Date'] = snap['Date'].dt.strftime(DATE_FMT)
                snap_path = DAILY_DIR / f"ohlcv_{sym.lower()}_{date_str}.csv"
                snap.to_csv(snap_path, index=False)

                merge_ohlcv_history(api_df, hist_path)
            except Exception as e:
                print(f"  ⚠️  {sym} OHLCV+RT fetch failed: {e}")
                raise
        print()


if __name__ == "__main__":
    main()
