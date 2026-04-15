#!/bin/bash
# Dampier Nitro++ v12 — Daily runner
# Fetches latest market data, then generates and emails the daily signal.

set -e

# Load credentials (VECTORVEST_EMAIL/PASSWORD, GOOGLE_EMAIL/APP_PASSWORD, STOCKS_DIR, etc.)
source "$HOME/.bash_profile"

# Project subdirectory under $STOCKS_DIR (local to this script)
PROJECT_SUBDIR="Nitro"

# Require STOCKS_DIR to be set (e.g. export STOCKS_DIR=/Users/mikedampier/Documents/Development)
if [ -z "$STOCKS_DIR" ]; then
    echo "ERROR: STOCKS_DIR is not set. Add 'export STOCKS_DIR=...' to ~/.bash_profile." >&2
    exit 1
fi

ROOT_DIR="$STOCKS_DIR/$PROJECT_SUBDIR"
if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: project directory does not exist: $ROOT_DIR" >&2
    exit 1
fi
cd "$ROOT_DIR"

# Use project venv
PY=".venv/bin/python3"

echo "=========================================="
echo "  Nitro++ v12 Daily Run — $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

echo
echo "── Step 1/3: Yahoo daily fetch (SQQQ, VIX) ──"
$PY fetch_yahoo_daily.py

echo
echo "── Step 2/3: VectorVest daily fetch (QQQ, TQQQ, timing) ──"
$PY fetch_vv_daily.py

echo
echo "── Step 3/3: Daily signal report ──"
$PY nitro_daily_signal_v12.py

echo
echo "=========================================="
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
