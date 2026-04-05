#!/usr/bin/env python3
"""
Daily VectorVest end-of-day fetch — two sources combined:

SOURCE 1 — StockViewer (VVC, TQQQ, QQQ)
  Primary source for VVC-Price (3 decimal places) and per-stock metrics.
  Merges into:
    qqq-from-vv.csv               (Open, High, Low, Close=Price, Volume, RT)
    tqqq-from-vv.csv              (Open, High, Low, Close=Price, Volume, RT)
    vectorvest-views-w3place-precision.csv  → today's VVC-Price + VVC-RT

SOURCE 2 — Views newsletter timing table (views-us.vectorvest.com)
  Source for market-wide indicators: MTI, Trend, % Buys, % Sells, BS Ratio, CG-*.
  Fetches the latest newsletter via RSS feed and merges all new rows into:
    vectorvest-views-w3place-precision.csv

For today's row in the VV views file:
  - Timing table provides all 11 columns (including VVC-Price at 2 decimals)
  - StockViewer then overwrites VVC-Price with higher-precision 3-decimal value

Usage:
    python3 fetch_vv_daily.py
    python3 fetch_vv_daily.py --symbols "VVC, TQQQ, QQQ"
"""
import os, sys, argparse, importlib.util
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

# ── Paths ─────────────────────────────────────────────────────────────────────
NITRO_DIR    = Path('/Users/mikedampier/Documents/Development/Nitro')
DAILY_DIR    = NITRO_DIR / 'data' / 'csv' / 'daily'
HIST_DIR     = NITRO_DIR / 'data' / 'csv' / 'history'
VV_SCRIPT    = NITRO_DIR / 'fetch_vectorvest_stock.py'
TIMING_SCRIPT = NITRO_DIR / 'fetch_vectorvest_timing.py'
COOKIE_FILE  = Path('/Users/mikedampier/.openclaw/workspace/views_cookie.json')
FEED_URL     = 'https://views-us.vectorvest.com/category/weekly-newsletter/feed/'
DAILY_DIR.mkdir(parents=True, exist_ok=True)

DATE_FMT = '%-m/%-d/%y'   # M/D/YY — matches existing history file format

# StockViewer DOM column order
COLS = [
    "Company","Symbol","Exch","Price","$Change","%PRC","Value",
    "RV","RS","RT","VST","REC","Stop",
    "GRT","EPS","EY","P/E","GPE",
    "DIV","DY","DS","DG","YSG",
    "Open","High","Low","Range",
    "Volume","AvgVol","%Vol",
    "Sales(M)","Sales GRT","SPS","P/S","Shares(M)","Mkt Cap($M)","CI",
    "Industry","Sector"
]

VV_VIEWS_FILE = HIST_DIR / 'vectorvest-views-w3place-precision.csv'
VV_VIEWS_COLS = ['Date','VVC-Price','VVC-RT','% Buys','% Sells',
                 'BS Ratio','CG-Price','CG-RT','CG-BSR','MTI','Trend']


# ── Date helpers ─────────────────────────────────────────────────────────────

def last_trading_day(ref: date = None) -> date:
    """Return the most recent weekday on or before ref (defaults to today).
    Saturday → Friday, Sunday → Friday, weekday → same day.
    NOTE: does not account for market holidays — prefer using
    trading_day_from_timing() when a timing DataFrame is available."""
    d = ref or date.today()
    if d.weekday() == 5:    # Saturday
        return d - timedelta(days=1)
    elif d.weekday() == 6:  # Sunday
        return d - timedelta(days=2)
    return d


def trading_day_from_timing(timing_df: pd.DataFrame) -> date:
    """Use the timing table's most recent date as the authoritative trading day.
    This correctly handles market holidays (e.g. Good Friday) that a simple
    weekday check would miss."""
    return timing_df['Date'].max().date()


# ── Module loaders ────────────────────────────────────────────────────────────

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── History helpers ───────────────────────────────────────────────────────────

def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def save_history(df: pd.DataFrame, path: Path):
    out = df.copy()
    out['Date'] = out['Date'].dt.strftime(DATE_FMT)
    bom = path.name in ('qqq-from-vv.csv', 'tqqq-from-vv.csv')
    out.to_csv(path, index=False, encoding='utf-8-sig' if bom else 'utf-8')


# ── SOURCE 1: StockViewer fetch ───────────────────────────────────────────────

def fetch_stockviewer(symbols: str, email: str, password: str) -> list[dict]:
    vv = load_module(VV_SCRIPT, 'vvstock')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx  = browser.new_context()
        page = ctx.new_page()

        page.goto("https://www.vectorvest.com/vvlogin/login.aspx",
                  wait_until="domcontentloaded")
        vv.login(page, email, password)
        page.wait_for_timeout(2000)

        page.goto("https://www.vectorvest.com/vvlogin/ApplicationFrameset.aspx"
                  "?type=3&ReturnURL=", wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        if "login" in page.url.lower():
            vv.login(page, email, password)
            page.wait_for_timeout(2000)

        page.goto("https://www.vectorvest.com/vvexpress/us/StockViewer.aspx",
                  wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        print(f"  StockViewer loaded: {page.title()}", flush=True)

        page.select_option("select[name*='cboSearch']", value="0")
        page.locator("input[name*='txtSymbol']").fill(symbols)
        page.locator("input[name*='btnSearch']").click()
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_timeout(3000)

        records  = []
        sym_list = [s.strip().upper() for s in symbols.split(",")]

        for row_el in page.locator("table tr").all():
            cells      = row_el.locator("td").all()
            clean      = [c.inner_text().strip() for c in cells]
            clean      = [t for t in clean if t]
            if len(clean) < 5:
                continue
            found_sym = None
            for val in clean[:4]:
                if val.upper() in sym_list:
                    found_sym = val.upper()
                    break
            if not found_sym:
                continue
            sym_idx = next(i for i, v in enumerate(clean) if v.upper() == found_sym)
            if sym_idx == 0:
                continue
            vals = [clean[sym_idx - 1]] + clean[sym_idx:]
            rec  = dict(zip(COLS, vals[:len(COLS)]))
            if not any(r.get('Symbol') == rec.get('Symbol') for r in records):
                records.append(rec)
                print(f"  Parsed: {rec.get('Symbol')} — {rec.get('Company')} @ {rec.get('Price')}")

        ctx.close()
        browser.close()
        return records


# ── SOURCE 2: Views newsletter timing table ───────────────────────────────────

def fetch_timing_table(email: str, password: str) -> pd.DataFrame:
    """Fetch the latest Views newsletter timing table via RSS feed. Returns a DataFrame."""
    timing = load_module(TIMING_SCRIPT, 'vvtiming')

    latest_url = timing.get_latest_link_from_feed(FEED_URL)
    print(f"  Newsletter URL: {latest_url}", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx  = browser.new_context()

        if COOKIE_FILE.exists():
            timing.load_cookies_into_context(ctx, str(COOKIE_FILE))

        page = ctx.new_page()

        # Main login first
        page.goto("https://www.vectorvest.com/vvlogin/login.aspx",
                  wait_until="domcontentloaded")
        timing.login(page, email, password)
        page.wait_for_timeout(2000)

        # Navigate to newsletter
        page.goto(latest_url, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        print(f"  Landed: {page.url}  ({page.locator('table').count()} tables)", flush=True)

        records = timing.extract_table_by_name(page, target_name="timing")
        ctx.close()
        browser.close()

    if not records:
        raise RuntimeError("Timing table returned no rows")

    df = pd.DataFrame(records)
    # Normalize date column
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ── Merge functions ───────────────────────────────────────────────────────────

def merge_ohlcv(rec: dict, hist_path: Path, trading_day: date):
    """Merge a QQQ or TQQQ StockViewer record into an OHLCV history file.
    Uses last_trading_day() date — not today — to avoid weekend/late-run misdating.
    Duplicate check: if the trading day already exists, skips without overwriting."""
    df  = load_history(hist_path)
    ts  = pd.Timestamp(trading_day)

    if ts in set(df['Date']):
        print(f"  ⚠️  {hist_path.name}: {trading_day} already exists — skipping (duplicate)")
        return

    new_row = pd.DataFrame([{
        'Date':   ts,
        'Open':   rec.get('Open',   ''),
        'High':   rec.get('High',   ''),
        'Low':    rec.get('Low',    ''),
        'Close':  rec.get('Price',  ''),
        'Volume': rec.get('Volume', ''),
        'RT':     rec.get('RT',     ''),
    }])
    df = pd.concat([df, new_row], ignore_index=True).sort_values('Date').reset_index(drop=True)
    save_history(df, hist_path)
    print(f"  ✅ {hist_path.name}: added {trading_day}  Close={rec.get('Price')}  RT={rec.get('RT')}")


def merge_vv_views_from_timing(timing_df: pd.DataFrame):
    """Merge all new rows from the timing table into the VV views history file."""
    df = load_history(VV_VIEWS_FILE)
    existing = set(df['Date'])
    new_rows = timing_df[~timing_df['Date'].isin(existing)].copy()

    if new_rows.empty:
        print(f"  {VV_VIEWS_FILE.name}: no new rows from timing table")
        return

    # Ensure columns match history file
    for col in VV_VIEWS_COLS:
        if col not in new_rows.columns and col != 'Date':
            new_rows[col] = ''

    new_rows = new_rows[VV_VIEWS_COLS].copy()
    df = pd.concat([df, new_rows], ignore_index=True).sort_values('Date').reset_index(drop=True)
    save_history(df, VV_VIEWS_FILE)
    dates_added = sorted(new_rows['Date'].dt.date.tolist())
    print(f"  ✅ {VV_VIEWS_FILE.name}: added {len(new_rows)} row(s) from timing table: {dates_added}")


def update_vvc_price(rec: dict, trading_day: date):
    """Overwrite trading_day's VVC-Price (and VVC-RT) with the StockViewer 3-decimal value.
    Uses last_trading_day() date — not today — to avoid weekend/late-run misdating.
    If the row was already added by the timing table, updates VVC-Price in place.
    If the row doesn't exist yet, inserts it (timing table may not have today yet)."""
    df = load_history(VV_VIEWS_FILE)
    ts = pd.Timestamp(trading_day)

    if ts in set(df['Date']):
        idx = df.index[df['Date'] == ts][0]
        old = df.at[idx, 'VVC-Price']
        if str(old) == str(rec.get('Price', '')):
            print(f"  ⚠️  {VV_VIEWS_FILE.name}: {trading_day} VVC-Price already {old} — no change")
            return
        df.at[idx, 'VVC-Price'] = rec.get('Price', df.at[idx, 'VVC-Price'])
        df.at[idx, 'VVC-RT']    = rec.get('RT',    df.at[idx, 'VVC-RT'])
        save_history(df, VV_VIEWS_FILE)
        print(f"  ✅ {VV_VIEWS_FILE.name}: updated {trading_day}  "
              f"VVC-Price {old} → {rec.get('Price')} (StockViewer, 3dp)  VVC-RT={rec.get('RT')}")
    else:
        # Timing table didn't have this date yet — insert with StockViewer data
        new_row = pd.DataFrame([{
            'Date':      ts,
            'VVC-Price': rec.get('Price', ''),
            'VVC-RT':    rec.get('RT',    ''),
            '% Buys':    '', '% Sells': '', 'BS Ratio': '',
            'CG-Price':  '', 'CG-RT':   '', 'CG-BSR':   '',
            'MTI':       '', 'Trend':    '',
        }])
        df = pd.concat([df, new_row], ignore_index=True).sort_values('Date').reset_index(drop=True)
        save_history(df, VV_VIEWS_FILE)
        print(f"  ✅ {VV_VIEWS_FILE.name}: inserted {trading_day}  "
              f"VVC-Price={rec.get('Price')} (StockViewer, 3dp)  VVC-RT={rec.get('RT')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="VVC, TQQQ, QQQ")
    ap.add_argument("--skip-timing", action="store_true",
                    help="Skip the Views newsletter timing fetch")
    ap.add_argument("--skip-stockviewer", action="store_true",
                    help="Skip the StockViewer fetch")
    args = ap.parse_args()

    email    = os.getenv("VECTORVEST_EMAIL")
    password = os.getenv("VECTORVEST_PASSWORD")
    if not email or not password:
        sys.exit("Set VECTORVEST_EMAIL and VECTORVEST_PASSWORD env vars.")

    today = date.today()

    if today.weekday() >= 5:
        print(f"Today is {today.strftime('%A')} — skipping (weekday only).")
        sys.exit(0)

    # trading_day resolved after timing fetch (authoritative); fallback = last weekday
    trading_day = last_trading_day(today)
    date_str    = today.strftime("%Y-%m-%d")
    print(f"[{datetime.now().isoformat()}] fetch_vv_daily  {date_str}\n", flush=True)

    # ── SOURCE 2: Views timing table ──────────────────────────────────────────
    timing_df = None
    if not args.skip_timing:
        print("── Views newsletter timing table ─────────────────────────────")
        try:
            timing_df = fetch_timing_table(email, password)
            print(f"  Fetched {len(timing_df)} rows  "
                  f"({timing_df['Date'].min().date()} → {timing_df['Date'].max().date()})")

            # Save daily snapshot of timing table
            snap_path = DAILY_DIR / f"timing_{date_str}.csv"
            snap = timing_df.copy()
            snap['Date'] = snap['Date'].dt.strftime(DATE_FMT)
            snap.to_csv(snap_path, index=False)
            print(f"  Snapshot → {snap_path.name}")

            merge_vv_views_from_timing(timing_df)
            # Use timing table's max date as authoritative trading day
            trading_day = trading_day_from_timing(timing_df)
            print(f"  Trading day resolved from timing table: {trading_day}")
        except Exception as e:
            print(f"  ⚠️  Timing fetch failed: {e} — falling back to last weekday: {trading_day}")
        print()

    # ── SOURCE 1: StockViewer ─────────────────────────────────────────────────
    if not args.skip_stockviewer:
        print("── StockViewer ───────────────────────────────────────────────")
        try:
            records = fetch_stockviewer(args.symbols, email, password)

            if not records:
                print("  ⚠️  No records returned — check login or page structure.")
            else:
                # Save daily snapshot
                df_snap  = pd.DataFrame(records)
                snap_path = DAILY_DIR / f"stockviewer_{date_str}.csv"
                df_snap.to_csv(snap_path, index=False)
                print(f"\n  Snapshot → {snap_path.name}")

                key   = ["Symbol", "Price", "RV", "RS", "RT", "VST", "REC", "Stop"]
                avail = [c for c in key if c in df_snap.columns]
                print(f"\n{df_snap[avail].to_string(index=False)}\n")

                rec_by_sym = {r['Symbol'].upper(): r for r in records}

                # QQQ → qqq-from-vv.csv
                if 'QQQ' in rec_by_sym:
                    merge_ohlcv(rec_by_sym['QQQ'],  HIST_DIR / 'qqq-from-vv.csv',  trading_day)
                else:
                    print("  ⚠️  QQQ not in results")

                # TQQQ → tqqq-from-vv.csv
                if 'TQQQ' in rec_by_sym:
                    merge_ohlcv(rec_by_sym['TQQQ'], HIST_DIR / 'tqqq-from-vv.csv', trading_day)
                else:
                    print("  ⚠️  TQQQ not in results")

                # VVC → overwrite VVC-Price with 3-decimal StockViewer value
                if 'VVC' in rec_by_sym:
                    update_vvc_price(rec_by_sym['VVC'], trading_day)
                else:
                    print("  ⚠️  VVC not in results")

        except Exception as e:
            print(f"  ⚠️  StockViewer fetch failed: {e}")
            raise


if __name__ == "__main__":
    main()
