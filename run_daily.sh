#!/bin/bash
# Dampier Nitro++ v12 — Daily runner
# Fetches latest market data, then generates and emails the daily signal.
# Logs output to logs/daily-YYYY-MM-DD.log and sends heartbeat/failure email.

set -eo pipefail

# Load credentials (VECTORVEST_EMAIL/PASSWORD, GOOGLE_EMAIL/APP_PASSWORD, STOCKS_DIR, etc.)
source "$HOME/.bash_profile"

# Project subdirectory under $STOCKS_DIR (local to this script)
PROJECT_SUBDIR="Nitro"

# Require STOCKS_DIR to be set
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

PY=".venv/bin/python3"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y-%m-%d_%H%M%S')"
LOG_FILE="$LOG_DIR/daily-${TS}.log"

# ── Email alert helper ────────────────────────────────────────────────────────
# Usage: send_alert "subject" "/path/to/body_file"
send_alert() {
    local subject="$1"
    local body_file="$2"
    "$PY" -c '
import os, sys, smtplib
from email.mime.text import MIMEText
user = os.environ.get("GOOGLE_EMAIL", "")
pw   = os.environ.get("GOOGLE_APP_PASSWORD", "")
if not (user and pw):
    print("send_alert: GOOGLE_EMAIL/GOOGLE_APP_PASSWORD not set — skipping", file=sys.stderr)
    sys.exit(0)
subject = sys.argv[1]
with open(sys.argv[2], "r") as f:
    body = f.read()
msg = MIMEText(body)
msg["Subject"] = subject
msg["From"]    = user
msg["To"]      = user
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, [user], msg.as_string())
    print(f"alert sent: {subject}")
except Exception as e:
    print(f"alert email failed: {e}", file=sys.stderr)
' "$subject" "$body_file"
}

# ── Failure trap ──────────────────────────────────────────────────────────────
on_failure() {
    local rc=$?
    local alert_body="/tmp/${PROJECT_SUBDIR}_fail_$$.txt"
    {
        echo "Nitro++ daily run FAILED"
        echo "Exit code: $rc"
        echo "Timestamp: $(date)"
        echo "Log file:  $ROOT_DIR/$LOG_FILE"
        echo ""
        echo "─── Last 50 log lines ───"
        tail -n 50 "$LOG_FILE" 2>/dev/null || echo "(no log available)"
    } > "$alert_body"
    send_alert "❌ Nitro++ daily FAILED (rc=$rc)" "$alert_body"
    rm -f "$alert_body"
    exit $rc
}
trap on_failure ERR

# ── Pipeline (all output tee'd to log file) ───────────────────────────────────
{
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
} 2>&1 | tee "$LOG_FILE"

# ── Heartbeat on success ──────────────────────────────────────────────────────
alert_body="/tmp/${PROJECT_SUBDIR}_ok_$$.txt"
{
    echo "Nitro++ daily run completed successfully."
    echo "Timestamp: $(date)"
    echo "Log file:  $ROOT_DIR/$LOG_FILE"
    echo ""
    echo "─── Last 30 log lines ───"
    tail -n 30 "$LOG_FILE"
} > "$alert_body"
send_alert "✅ Nitro++ daily OK" "$alert_body"
rm -f "$alert_body"
