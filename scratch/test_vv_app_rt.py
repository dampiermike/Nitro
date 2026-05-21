#!/usr/bin/env python3
"""
EXPLORATORY TEST — pull RT (Relative Timing) for TQQQ from the new VectorVest app.

Logs into https://app.vectorvest.com/login, opens https://app.vectorvest.com/detail/TQQQ,
and tries to extract RT two ways:
  1. Network capture — records every JSON XHR/fetch the SPA makes, then searches the
     payloads for a Relative-Timing field. This is the robust path: if found, it gives
     us a clean API endpoint to call directly (no browser needed in production).
  2. DOM scrape — fallback: searches the rendered page for an "RT" label/value.

All diagnostics (screenshots, captured JSON, page HTML) are written to
scratch/vv_rt_debug/ for inspection.

Usage:
    .venv/bin/python3 scratch/test_vv_app_rt.py            # headless
    .venv/bin/python3 scratch/test_vv_app_rt.py --headed   # watch the browser
"""
import os, sys, json, re, argparse
from pathlib import Path
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SYMBOL    = "TQQQ"
LOGIN_URL = "https://app.vectorvest.com/login"
DETAIL_URL = f"https://app.vectorvest.com/detail/{SYMBOL}"
DEBUG_DIR = Path(__file__).resolve().parent / "vv_rt_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Field names that plausibly hold the Relative Timing value in a JSON payload.
RT_KEY_RE = re.compile(r"^(rt|relativetiming|relative_timing)$", re.I)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def find_rt_in_json(obj, path=""):
    """Recursively search a JSON structure for keys that look like Relative Timing."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}"
            if RT_KEY_RE.match(str(k)) and isinstance(v, (int, float, str)):
                hits.append((here, v))
            hits += find_rt_in_json(v, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):   # sample first few list items
            hits += find_rt_in_json(v, f"{path}[{i}]")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    args = ap.parse_args()

    email = os.getenv("VECTORVEST_EMAIL")
    password = os.getenv("VECTORVEST_PASSWORD")
    if not email or not password:
        sys.exit("ERROR: set VECTORVEST_EMAIL and VECTORVEST_PASSWORD env vars "
                 "(source ~/.bash_profile).")

    log(f"Email: {email}   Target: {DETAIL_URL}")

    captured = []   # list of (url, json_body)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        # ── Network capture ───────────────────────────────────────────────────
        def on_response(resp):
            try:
                if resp.request.resource_type not in ("xhr", "fetch"):
                    return
                ctype = resp.headers.get("content-type", "")
                if "json" not in ctype:
                    return
                body = resp.json()
                captured.append((resp.url, body))
            except Exception:
                pass
        page.on("response", on_response)

        # ── Step 1: login ─────────────────────────────────────────────────────
        log("Loading login page ...")
        page.goto(LOGIN_URL, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(DEBUG_DIR / "01_login.png"))

        # Dump the form structure so we can adapt selectors if login fails.
        inputs = page.eval_on_selector_all(
            "input",
            "els => els.map(e => ({type:e.type, name:e.name, id:e.id, "
            "placeholder:e.placeholder}))")
        log(f"Login page inputs: {json.dumps(inputs)}")

        # The new app uses unnamed inputs identified only by placeholder text.
        email_sel = ("input[placeholder*='user' i], input[placeholder*='email' i], "
                     "input[type='email'], input[name*='email' i], "
                     "input[name*='user' i], input#email, input#username")
        pwd_sel = "input[type='password'], input[placeholder*='password' i]"

        try:
            page.fill(email_sel, email, timeout=15000)
            page.fill(pwd_sel, password, timeout=15000)
        except PWTimeout:
            page.screenshot(path=str(DEBUG_DIR / "01b_login_fields_missing.png"))
            (DEBUG_DIR / "login_page.html").write_text(page.content())
            browser.close()
            sys.exit("ERROR: could not find login fields — see vv_rt_debug/login_page.html")

        # Submit — try an explicit button, fall back to Enter.
        for sel in ("button[type='submit']",
                    "button:has-text('Log In')",
                    "button:has-text('Sign In')",
                    "button:has-text('Login')",
                    "input[type='submit']"):
            if page.locator(sel).count():
                log(f"Clicking submit: {sel}")
                page.locator(sel).first.click()
                break
        else:
            log("No submit button found — pressing Enter")
            page.press(pwd_sel, "Enter")

        # Wait to leave the login page.
        try:
            page.wait_for_url(lambda u: "login" not in u.lower(), timeout=30000)
        except PWTimeout:
            log("WARNING: still on a /login URL after submit")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(DEBUG_DIR / "02_after_login.png"))
        log(f"Post-login URL: {page.url}")

        # ── Step 2: detail page ───────────────────────────────────────────────
        log(f"Navigating to {DETAIL_URL} ...")
        captured.clear()   # only care about XHRs from the detail page onward
        page.goto(DETAIL_URL, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(5000)
        page.screenshot(path=str(DEBUG_DIR / "03_detail.png"), full_page=True)
        (DEBUG_DIR / "detail_page.html").write_text(page.content())
        log(f"Detail page title: {page.title()}   URL: {page.url}")

        # ── Result 1: network capture ─────────────────────────────────────────
        log(f"Captured {len(captured)} JSON XHR/fetch responses on the detail page")
        api_hits = []
        for i, (url, body) in enumerate(captured):
            (DEBUG_DIR / f"xhr_{i:02d}.json").write_text(
                json.dumps({"url": url, "body": body}, indent=2, default=str)[:200000])
            rt = find_rt_in_json(body)
            if rt:
                api_hits.append((url, rt))

        print()
        print("=" * 70)
        if api_hits:
            print("RT FOUND IN API RESPONSES:")
            for url, rt in api_hits:
                print(f"  endpoint: {url}")
                for path, val in rt:
                    print(f"    {path} = {val}")
        else:
            print("No obvious RT field in captured JSON.")
            print("Captured endpoints (inspect xhr_*.json in vv_rt_debug/):")
            for url, _ in captured:
                print(f"  - {url}")

        # ── Result 2: DOM scrape fallback ─────────────────────────────────────
        print("-" * 70)
        dom_rt = None
        try:
            dom_rt = page.evaluate(r"""() => {
                // Find a node whose text is exactly 'RT' and read its sibling/parent value.
                const labels = [...document.querySelectorAll('*')].filter(
                    e => e.children.length === 0 &&
                         /^RT$/i.test((e.textContent || '').trim()));
                for (const lbl of labels) {
                    const scope = lbl.parentElement?.parentElement || lbl.parentElement;
                    if (!scope) continue;
                    const m = (scope.textContent || '').match(/RT[^0-9-]*(-?\d+\.?\d*)/i);
                    if (m) return m[1];
                }
                // Fallback: any 'Relative Timing' text on the page.
                const all = document.body.innerText || '';
                const m = all.match(/Relative\s*Timing[^0-9-]*(-?\d+\.?\d*)/i);
                return m ? m[1] : null;
            }""")
        except Exception as e:
            log(f"DOM scrape error: {e}")
        if dom_rt is not None:
            print(f"RT FROM DOM SCRAPE: {dom_rt}")
        else:
            print("RT not found via DOM scrape (see 03_detail.png / detail_page.html)")
        print("=" * 70)

        browser.close()

    log(f"Debug artifacts written to: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
