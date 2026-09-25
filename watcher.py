"""
Watcher — runs the pipeline on a schedule during NSE market hours.
Sends Telegram alerts only for NEW setups or MEANINGFUL changes.
State is persisted in watcher_state.json so restarts don't re-spam.

Change triggers (re-alerts on existing setup):
  • Setup upgrades:   BREAKOUT → PRIME
  • RVOL surge:       increase ≥ 0.4x since last alert
  • Score jump:       increase ≥ 20% since last alert

Usage:
  python watcher.py                  # every 30 min, market hours only
  python watcher.py --interval 15    # scan every 15 minutes
  python watcher.py --ai             # include Gemini levels on top 2
  python watcher.py --ai-top 3       # Gemini on top 3 instead
  python watcher.py --once           # run one scan and exit (test mode)
  python watcher.py --force          # ignore market hours check (testing)

Start in background, log to file
cd /home/priyansh/files/pro_v2.1
nohup python3 watcher.py --interval 30 > watcher.log 2>&1 &

See the log live
tail -f watcher.log

Stop it
pkill -f watcher.py

"""
import argparse
import datetime
import json
import os
import time
import warnings

import pytz

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watcher_state.json")
IST        = pytz.timezone("Asia/Kolkata")

MARKET_OPEN  = (9,  15)
MARKET_CLOSE = (15, 30)

RVOL_DELTA_THRESHOLD = 0.4    # RVOL must rise by this much to re-alert
SCORE_PCT_THRESHOLD  = 0.20   # score must rise by this % to re-alert
SETUP_RANK = {"MOMENTUM": 1, "BREAKOUT": 2, "PRIME": 3}


# ── Time helpers ──────────────────────────────────────────────────────────────

def now_ist() -> datetime.datetime:
    return datetime.datetime.now(IST)

def now_str() -> str:
    return now_ist().strftime("%H:%M:%S IST")

def in_market_hours() -> bool:
    n = now_ist()
    o = n.replace(hour=MARKET_OPEN[0],  minute=MARKET_OPEN[1],  second=0, microsecond=0)
    c = n.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return o <= n <= c

def seconds_until_open() -> int:
    n = now_ist()
    o = n.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    if n >= o:
        o += datetime.timedelta(days=1)
    return int((o - n).total_seconds())


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"date": None, "alerted": {}}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def reset_if_new_day(state: dict) -> dict:
    today = now_ist().strftime("%Y-%m-%d")
    if state.get("date") != today:
        print(f"  🗓  New trading day — clearing previous state")
        state = {"date": today, "alerted": {}}
        save_state(state)
    return state


# ── Change detection ──────────────────────────────────────────────────────────

def detect_change(old: dict, new: dict) -> tuple[bool, str]:
    """Returns (should_re_alert, reason_string)."""
    if SETUP_RANK.get(new["setup"], 0) > SETUP_RANK.get(old["setup"], 0):
        return True, f"⬆ Setup upgraded: {old['setup']} → {new['setup']}"

    rvol_delta = new["rvol"] - old["rvol"]
    if rvol_delta >= RVOL_DELTA_THRESHOLD:
        return True, f"📈 RVOL surging: {old['rvol']}x → {new['rvol']}x (+{rvol_delta:.2f})"

    if old["score"] > 0:
        score_pct = (new["score"] - old["score"]) / old["score"]
        if score_pct >= SCORE_PCT_THRESHOLD:
            return True, f"🔥 Score jumped {round(score_pct * 100)}%: {old['score']} → {new['score']}"

    return False, ""


# ── Stop-loss / target monitoring ────────────────────────────────────────────
# (disabled — uncomment to activate)

# def _current_price(symbol: str) -> float | None:
#     """Fetch latest intraday price for an alerted symbol."""
#     import yfinance as yf
#     try:
#         df = yf.download(symbol, period="1d", interval="5m", progress=False, auto_adjust=True)
#         if df.empty:
#             return None
#         if isinstance(df.columns, __import__("pandas").MultiIndex):
#             df.columns = df.columns.get_level_values(0)
#         return float(df["Close"].iloc[-1])
#     except Exception:
#         return None
#
#
# def check_sl_breaches(state: dict) -> list[str]:
#     """Check every alerted symbol against its stored SL. Sends alert and evicts on breach."""
#     from alerts import send
#
#     alerted   = state.get("alerted", {})
#     hit_syms  = []
#
#     for sym, data in list(alerted.items()):
#         sl = data.get("sl")
#         if sl is None:
#             continue
#
#         price = _current_price(sym)
#         if price is None:
#             continue
#
#         if price <= sl:
#             entry      = data.get("entry_price")
#             sl_source  = data.get("sl_source", "")
#             src_label  = {"candle": "candle low", "swing": "swing low",
#                           "ema20": "EMA 20", "atr": "ATR"}.get(sl_source, sl_source)
#             pnl_str    = ""
#             if entry:
#                 pnl = round(((price - entry) / entry) * 100, 2)
#                 pnl_str = f"  |  P&L: {pnl:+.2f}%"
#
#             msg = (
#                 f"🛑 <b>SL HIT | {sym.replace('.NS', '')}</b>\n"
#                 f"━━━━━━━━━━━━━━━━━━━━━━\n"
#                 f"Current: ₹{round(price, 2)}  ≤  SL: ₹{sl}  ({src_label})\n"
#                 f"Entry: ₹{entry or '—'}{pnl_str}\n"
#                 f"Setup: {data.get('setup', '—')}  |  Alerted: {data.get('alerted_at', '—')}"
#             )
#             send(msg)
#             hit_syms.append(sym)
#             print(f"  🛑 SL HIT: {sym.replace('.NS','')}  ₹{round(price,2)} ≤ SL ₹{sl}{pnl_str}")
#
#     for sym in hit_syms:
#         alerted.pop(sym, None)
#
#     if hit_syms:
#         save_state(state)
#
#     return hit_syms


# ── Core scan (silent — no print statements) ──────────────────────────────────

def run_scan(ai: bool = False, ai_top: int = 2) -> list[dict]:
    from config import CONFIRM_TOP_N, TICKERS
    from scanner import run_scanner
    from consolidation import check as check_consol
    from confirm import check as check_conf

    candidates = run_scanner(TICKERS)
    if not candidates:
        return []

    for c in candidates:
        consol = check_consol(c["symbol"])
        c["consol"] = consol
        if consol and consol["passes"]:
            c["score"] = round(c["score"] * (1.3 if consol["acc_days"] >= 3 else 1.1), 2)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    passed = [c for c in candidates if c.get("consol") and c["consol"]["passes"]]
    pool   = (passed if passed else candidates)[:CONFIRM_TOP_N]

    confirmed = []
    for c in pool:
        conf = check_conf(c["symbol"], setup=c.get("setup", ""))
        c["conf"] = conf
        if conf and conf["passes"]:
            confirmed.append(c)

    if ai and confirmed:
        from levels import get_levels
        for c in confirmed[:ai_top]:
            ctx = {**c, **(c.get("consol") or {}), **(c.get("conf") or {})}
            c["levels"] = get_levels(c["symbol"], ctx)

    return confirmed


# ── Alert dispatch ────────────────────────────────────────────────────────────

def dispatch(confirmed: list[dict], state: dict, nifty_chg: float, nifty_warning: str, ai: bool = False):
    """Send alerts for new/changed setups, update state. Returns (sent, skipped, exits)."""
    from alerts import send, format_alert, format_header

    alerted      = state.setdefault("alerted", {})
    current_syms = {c["symbol"] for c in confirmed}
    to_alert     = []   # list of (candidate, reason)

    for c in confirmed:
        sym = c["symbol"]
        if sym not in alerted:
            to_alert.append((c, "🆕 New setup"))
        else:
            changed, reason = detect_change(alerted[sym], c)
            if changed:
                to_alert.append((c, reason))

    exits   = [s for s in alerted if s not in current_syms]
    skipped = len(confirmed) - len(to_alert)

    if to_alert:
        from alerts import send_photo
        from chart import draw_chart
        send(format_header(len(to_alert), nifty_chg, nifty_warning))
        for c, reason in to_alert:
            msg = f"{reason}\n" + format_alert(
                scan    = c,
                consol  = c.get("consol"),
                conf    = c.get("conf"),
                levels  = c.get("levels"),
                sector  = c.get("sector", "—"),
                nifty   = nifty_chg,
                warning = nifty_warning,
            )
            lvl = c.get("levels")
            if lvl and lvl.get("ACTION", "").strip().upper() in ("BUY", "STRONG BUY"):
                conf = c.get("conf") or {}
                path = draw_chart(
                    symbol    = c["symbol"],
                    entry     = conf.get("price") or c["price"],
                    sl        = conf.get("sl", 0),
                    sl_source = conf.get("sl_source", ""),
                    targets   = lvl,
                )
                if path:
                    send_photo(path)
            if send(msg):
                conf = c.get("conf") or {}
                alerted[c["symbol"]] = {
                    "setup":       c["setup"],
                    "rvol":        c["rvol"],
                    "score":       c["score"],
                    "dist_high":   c["dist_high"],
                    "alerted_at":  now_str(),
                    "alert_count": alerted.get(c["symbol"], {}).get("alert_count", 0) + 1,
                    "entry_price": conf.get("price") or c.get("price"),
                    "sl":          conf.get("sl"),
                    "sl_source":   conf.get("sl_source", ""),
                }
                if ai:
                    from trade_log import log_trade
                    log_trade(
                        scan    = c,
                        consol  = c.get("consol"),
                        conf    = c.get("conf"),
                        levels  = c.get("levels"),
                        sector  = c.get("sector", "—"),
                        nifty   = nifty_chg,
                        warning = nifty_warning,
                    )

    for sym in exits:
        alerted.pop(sym, None)

    save_state(state)
    return len(to_alert), skipped, exits


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pro-Trader Watcher")
    parser.add_argument("--interval", type=int,   default=30,   help="Scan interval in minutes (default: 30)")
    parser.add_argument("--ai",       action="store_true",       help="Include Gemini levels")
    parser.add_argument("--ai-top",   type=int,   default=2, dest="ai_top",
                        help="Max stocks to send to Gemini (default: 2)")
    parser.add_argument("--once",     action="store_true",       help="Run one scan and exit")
    parser.add_argument("--force",    action="store_true",       help="Ignore market hours check")
    args = parser.parse_args()

    from alerts import send
    from momentum import check_nifty, check_sector

    print("\n" + "━" * 50)
    print("  🤖  Pro-Trader Watcher started")
    print(f"  Interval : every {args.interval} min")
    print(f"  Gemini   : {'on (top ' + str(args.ai_top) + ')' if args.ai else 'off'}")
    print(f"  Mode     : {'one-shot' if args.once else 'continuous'}")
    print(f"  State    : {STATE_FILE}")
    print("━" * 50)

    send(f"🤖 Pro-Trader Watcher started\nInterval: {args.interval}min | "
         f"Gemini: {'on' if args.ai else 'off'} | {now_str()}")

    state     = load_state()
    scan_num  = 0

    while True:
        state = reset_if_new_day(state)

        if not args.force and not in_market_hours():
            secs = seconds_until_open()
            h, m = divmod(secs // 60, 60)
            open_time = (now_ist() + datetime.timedelta(seconds=secs)).strftime("%H:%M IST")
            print(f"\n  ⏸  Outside market hours — sleeping until {open_time} ({h}h {m}m)")
            if args.once:
                break
            time.sleep(secs)   # sleep straight until market opens
            continue

        scan_num += 1
        print(f"\n{'━' * 50}")
        print(f"  🕐  Scan #{scan_num} — {now_str()}")
        print("━" * 50)

        # SL breach check (disabled — uncomment when ready to activate)
        # sl_hits = check_sl_breaches(state)
        # if sl_hits:
        #     print(f"  🛑 SL hit this cycle: {', '.join(s.replace('.NS','') for s in sl_hits)}")

        # Nifty check
        nifty_chg, nifty_warning = check_nifty()
        nifty_label = f"{nifty_chg:+.2f}%"
        mood = "🟢" if nifty_chg >= 0 else "🔴"
        if nifty_warning:
            print(f"  ⚠️  {nifty_warning}")
        else:
            print(f"  {mood} Nifty {nifty_label}")

        nifty_status = f"⚠️ {nifty_warning}" if nifty_warning else f"{'🟢' if nifty_chg >= 0 else '🔴'} Nifty {nifty_label}"
        send(f"🔍 Scan #{scan_num} | {now_str()} | {nifty_status}")

        # Run scan
        print(f"  ⏳ Scanning...", end="", flush=True)
        confirmed = run_scan(args.ai, args.ai_top)
        print(f" {len(confirmed)} confirmed")

        if not confirmed:
            print("  😴  No confirmed setups this scan")
            if args.once:
                break
            next_run = now_ist() + datetime.timedelta(minutes=args.interval)
            print(f"  ⏭  Next scan: {next_run.strftime('%H:%M IST')}")
            time.sleep(args.interval * 60)
            continue

        # Sector momentum
        for c in confirmed:
            c["sector"] = check_sector(c["symbol"])

        # Print confirmed table
        alerted_syms = state.get("alerted", {})
        print(f"\n  {'SYMBOL':<20} {'SETUP':<10} {'RVOL':>5} {'SCORE':>6}  STATUS")
        print(f"  {'─' * 58}")
        for c in confirmed:
            sym   = c["symbol"]
            prev  = alerted_syms.get(sym)
            if not prev:
                status = "🆕 new"
            else:
                changed, reason = detect_change(prev, c)
                status = f"🔔 changed ({reason})" if changed else f"⏭  no change (alerted {prev['alerted_at']})"
            print(f"  {sym:<20} {c['setup']:<10} {c['rvol']:>5}x {c['score']:>6}  {status}")

        # Dispatch alerts
        sent, skipped, exits = dispatch(confirmed, state, nifty_chg, nifty_warning, ai=args.ai)
        print(f"\n  📤 Sent: {sent}  |  Skipped (no change): {skipped}  |  Exited: {len(exits)}")
        if exits:
            print(f"  🚪 Exited setup: {', '.join(e.replace('.NS','') for e in exits)}")

        if args.once:
            break

        next_run = now_ist() + datetime.timedelta(minutes=args.interval)
        print(f"\n  ⏭  Next scan: {next_run.strftime('%H:%M IST')}")
        time.sleep(args.interval * 60)

    print("\n  ✅  Watcher stopped.\n")


if __name__ == "__main__":
    main()
