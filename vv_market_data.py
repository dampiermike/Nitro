"""
VectorVest Market Data Fetcher
Fetches BSR, MTI, RT, VVC from api.vectorvest.com using your VectorVest credentials.
No browser required. Does NOT conflict with VV7 desktop session.

Usage:
    python vv_market_data.py

Requirements:
    pip install requests
"""

import requests
import base64
import json
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
import os
VV_USERNAME = os.environ.get("VECTORVEST_EMAIL", "ddampier")
VV_PASSWORD = os.environ.get("VECTORVEST_PASSWORD", "vvp@55wd")

TOKEN_URL   = "https://www.vectorvest.com/identity2/issue/simple"
MARKET_URL  = "https://api.vectorvest.com/MarketData/v3/markettiming/US"

# ── Step 1: Get Bearer Token ───────────────────────────────────────────────────
def get_token(username, password):
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://app.vectorvest.com",
    }
    params = {
        "realm": "https://www.vectorvest.com",
        "tokenType": "jwt"
    }
    resp = requests.get(TOKEN_URL, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()["access_token"]

# ── Step 2: Fetch Market Data ──────────────────────────────────────────────────
def get_market_data(token, num_days=5):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    params = {
        "includeAtAGlance": "true",
        "includeColorGuard": "true",
        "useExtendedHours": "false",
        "numDaysColorGuard": num_days,
    }
    resp = requests.get(MARKET_URL, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

# ── Step 3: Display Results ────────────────────────────────────────────────────
def display_results(data):
    at_a_glance = data.get("AtAGlance", {})
    color_guard = data.get("ColorGuard", {}).get("Items", [])

    print("\n" + "="*55)
    print("  VectorVest Market Data")
    print(f"  Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    print("\n── Today's Summary ─────────────────────────────────────")
    print(f"  VVC Price : {at_a_glance.get('VvcPrice', 'N/A'):.4f}")
    print(f"  RT        : {at_a_glance.get('VvcRelativeTiming', 'N/A'):.3f}")
    print(f"  BSR       : {at_a_glance.get('BuySellRatio', 'N/A'):.3f}")
    print(f"  MTI       : {at_a_glance.get('MarketTimingIndicator', 'N/A'):.3f}")
    print(f"  Buys      : {at_a_glance.get('Buys', 'N/A')}")
    print(f"  Sells     : {at_a_glance.get('Sells', 'N/A')}")

    if color_guard:
        print("\n── Color Guard History ──────────────────────────────────")
        print(f"  {'Date':<10} {'Price':>8} {'RT':>6} {'BSR':>6} {'MTI':>6}  {'Trend':<8} {'Call'}")
        print(f"  {'-'*9} {'-'*8} {'-'*6} {'-'*6} {'-'*6}  {'-'*8} {'-'*6}")
        for row in color_guard:
            date  = row.get("FormattedDate", "")
            price = row.get("Price", 0)
            rt    = row.get("RelativeTiming", 0)
            bsr   = row.get("BuySellRatio", 0)
            mti   = row.get("MarketTimingIndicator", 0)
            trend = row.get("Trend", "")
            call  = row.get("MarketCall", "")
            print(f"  {date:<10} {price:>8.4f} {rt:>6.3f} {bsr:>6.3f} {mti:>6.3f}  {trend:<8} {call}")

    print("\n" + "="*55 + "\n")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching VectorVest token...")
    token = get_token(VV_USERNAME, VV_PASSWORD)
    print("Token acquired. Fetching market data...")
    data = get_market_data(token, num_days=5)
    display_results(data)
