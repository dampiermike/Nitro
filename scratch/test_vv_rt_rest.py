#!/usr/bin/env python3
"""
EXPLORATORY TEST — pull RT via pure REST (no browser).

Confirms that the Bearer token from fetch_vv_daily.get_vv_token() — already used
in production for /markettiming — also unlocks the per-stock /quote and
/quotehistory endpoints that the app.vectorvest.com detail page calls.

If this works, RT no longer needs Playwright/StockViewer at all.
"""
import os, sys, importlib.util
from pathlib import Path

import requests

NITRO_DIR = Path(__file__).resolve().parent.parent
API = "https://api.vectorvest.com/MarketData/v3"


def load_get_vv_token():
    """Reuse get_vv_token() from fetch_vv_daily.py."""
    spec = importlib.util.spec_from_file_location(
        "fvd", str(NITRO_DIR / "fetch_vv_daily.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_vv_token


def quote(token, symbol):
    """Single-stock quote — returns today's RT + StockId."""
    r = requests.get(
        f"{API}/quote",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=[
            ("fields", "StockId"), ("fields", "Symbol"), ("fields", "LastPrice"),
            ("fields", "OpenPrice"), ("fields", "HighPrice"), ("fields", "LowPrice"),
            ("fields", "RelativeTiming"), ("fields", "StopPrice"),
            ("symbol", symbol), ("region", "US"),
            ("requestToUnlockProprietaryQuote", "true"),
        ],
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def quotehistory(token, stock_id, days=5):
    """Daily history for a StockId — OHLC + RT."""
    r = requests.get(
        f"{API}/quotehistory/{stock_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=[
            ("fields", "TradingDate"), ("fields", "OpenPrice"),
            ("fields", "HighPrice"), ("fields", "LowPrice"),
            ("fields", "LastPrice"), ("fields", "RelativeTiming"),
            ("fields", "StopPrice"),
            ("barFrequency", "Daily"),
            ("rangeFrequency", "Yearly"), ("rangePeriod", "1"),
        ],
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def main():
    email = os.getenv("VECTORVEST_EMAIL")
    password = os.getenv("VECTORVEST_PASSWORD")
    if not email or not password:
        sys.exit("Set VECTORVEST_EMAIL / VECTORVEST_PASSWORD (source ~/.bash_profile).")

    get_vv_token = load_get_vv_token()
    print("Authenticating (get_vv_token) ...")
    token = get_vv_token(email, password)
    print(f"  token OK ({len(token)} chars)\n")

    for sym in ("TQQQ", "QQQ"):
        print(f"── {sym} ─────────────────────────────────")
        data = quote(token, sym)
        row = data[0] if isinstance(data, list) else data
        prop = row.get("ProprietaryData", {}) or {}
        rt = prop.get("RelativeTiming")
        stock_id = row.get("StockId")
        print(f"  /quote        RT={rt}  Last={row.get('LastPrice')}  StockId={stock_id}")
        if rt is None:
            print("  ⚠️  RT is None — token may lack the proprietary-data entitlement.")

        hist = quotehistory(token, stock_id, days=5)
        rows = hist.get("HistoryData", [])
        print(f"  /quotehistory {len(rows)} daily rows; last 3:")
        for hr in rows[-3:]:
            print(f"    {hr.get('TradingDate','')[:10]}  "
                  f"O={hr.get('OpenPrice')} H={hr.get('HighPrice')} "
                  f"L={hr.get('LowPrice')} C={hr.get('LastPrice')}  "
                  f"RT={hr.get('RelativeTiming')}  Stop={hr.get('StopPrice')}")
        print()

    print("REST path works — no browser required.")


if __name__ == "__main__":
    main()
