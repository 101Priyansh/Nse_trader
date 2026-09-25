"""
Pro-Trader Pipeline v2.1
─────────────────────────────────────────────────────────────────────────────
Usage:
  python3 pipeline.py              # full run, no Gemini
  python3 pipeline.py --ai         # full run + Gemini on top 3 confirmed
  python3 pipeline.py --ai --ai-top 5   # Gemini on top 5 confirmed
  python3 pipeline.py --stage 1    # scanner only (no confirmation, no alerts)
  python3 pipeline.py --stage 2    # scanner + consolidation
  python3 pipeline.py --stage 3    # scanner + consolidation + RSI/ADX (no alerts)
  python3 pipeline.py --top 15     # confirm top 15 candidates (default: 6)
─────────────────────────────────────────────────────────────────────────────
"""
import argparse
import sys

import warnings
warnings.filterwarnings("ignore")

from config import CONFIRM_TOP_N, RSI_MAX, RSI_MIN, TICKERS, USE_VISUAL_LEVELS


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(text: str):
    print(f"\n{'━' * 50}")
    print(f"  {text}")
    print(f"{'━' * 50}")

def _ok(sym, msg=""):    print(f"  ✅  {sym:<20} {msg}")
def _fail(sym, msg=""):  print(f"  ❌  {sym:<20} {msg}")
def _info(msg):          print(f"  →   {msg}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pro-Trader Pipeline v2.1")
    parser.add_argument("--ai",     action="store_true", help="Run Gemini levels on confirmed candidates")
    parser.add_argument("--ai-top", type=int, default=3, dest="ai_top",
                        help="Max confirmed candidates to send to Gemini (default: 3)")
    parser.add_argument("--stage",  type=int, choices=[1, 2, 3], default=None,
                        help="Stop after stage N (skips alerts)")
    parser.add_argument("--top",    type=int, default=CONFIRM_TOP_N,
                        help=f"How many top candidates to confirm in Stage 3 (default {CONFIRM_TOP_N})")
    args = parser.parse_args()

    import datetime
    from alerts import send
    now = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    send(f"🔍 <b>Pro-Trader Scan started</b>\n{now}  |  {len(TICKERS)} stocks")

    print("\n🚀  Pro-Trader Pipeline v2.1")
    print(f"    Watchlist: {len(TICKERS)} stocks")

    # ── Nifty breadth gate ────────────────────────────────────────────────────
    _header("Market Check — Nifty")
    from momentum import check_nifty
    nifty_chg, nifty_warning = check_nifty()
    if nifty_warning:
        print(f"  ⚠️   {nifty_warning}")
    else:
        print(f"  🟢  Nifty {nifty_chg:+.2f}% — market is constructive")

    # ── Stage 1: Scanner ─────────────────────────────────────────────────────
    _header("Stage 1 — Trend + RVOL + Proximity Scanner")
    from scanner import run_scanner
    candidates = run_scanner(TICKERS)

    if not candidates:
        print("  😴  No candidates passed Stage 1.")
        from alerts import send
        send("😴 Pro-Trader Scan: No setups found today.")
        return

    print(f"\n  {'RANK':<5} {'SYMBOL':<18} {'PRICE':>8} {'TYPE':<10} {'RVOL':>6} {'FROM HIGH':>10} {'SCORE':>7}")
    print(f"  {'─'*68}")
    for i, r in enumerate(candidates, 1):
        stk = "⬆" if r["stacked"] else "↗"
        print(f"  {i:<5} {r['symbol']:<18} {r['price']:>8} {r['setup']:<10} {r['rvol']:>6} {r['dist_high']:>9}% {r['score']:>7}  {stk}")

    _info(f"{len(candidates)} candidates found")

    if args.stage == 1:
        print("\n  (Stopped at Stage 1 — no alerts sent)")
        return

    # ── Stage 2: Consolidation ────────────────────────────────────────────────
    _header("Stage 2 — Base / Consolidation Check")
    from consolidation import check as check_consol

    for c in candidates:
        consol = check_consol(c["symbol"])
        c["consol"] = consol
        if consol:
            if consol["passes"]:
                bonus = 1.3 if consol["acc_days"] >= 3 else 1.1
                c["score"] = round(c["score"] * bonus, 2)
                sq = " 🔄" if consol["atr_squeeze"] else ""
                _ok(c["symbol"],
                    f"{consol['acc_days']} acc days | {consol['consol_pct']}% range | {consol['vol_trend']}{sq}")
            else:
                _fail(c["symbol"],
                      f"{consol['acc_days']} acc days | {consol['consol_pct']}% range (weak base)")
        else:
            _fail(c["symbol"], "data unavailable")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    passed_consol = [c for c in candidates if c.get("consol") and c["consol"]["passes"]]
    _info(f"{len(passed_consol)} candidates passed consolidation check")

    if args.stage == 2:
        print("\n  (Stopped at Stage 2 — no alerts sent)")
        return

    # ── Stage 3: RSI / ADX / MACD confirmation ────────────────────────────────
    _header(f"Stage 3 — RSI / ADX Confirmation (top {args.top})")
    from confirm import check as check_conf

    pool = (passed_consol if passed_consol else candidates)[:args.top]
    confirmed = []

    for c in pool:
        conf = check_conf(c["symbol"], setup=c.get("setup", ""))
        c["conf"] = conf
        if conf:
            if conf["passes"]:
                _ok(c["symbol"],
                    f"RSI {conf['rsi']} | ADX {conf['adx']} | MACD {conf['macd']} | Div {conf['div']}")
                confirmed.append(c)
            else:
                if not conf["rsi_ok"]:
                    reason = (f"RSI {conf['rsi']} (overbought >{RSI_MAX})"
                              if conf["rsi"] > RSI_MAX
                              else f"RSI {conf['rsi']} (below momentum zone <{RSI_MIN})")
                else:
                    reason = "price below daily 200 EMA"
                _fail(c["symbol"], reason)
        else:
            _fail(c["symbol"], "data unavailable")

    _info(f"{len(confirmed)} candidates confirmed")

    if args.stage == 3:
        print("\n  (Stopped at Stage 3 — no alerts sent)")
        return

    if not confirmed:
        from alerts import send
        send("😴 Pro-Trader Scan: No setups passed RSI/ADX confirmation today.")
        print("\n  No confirmed setups. Telegram notified.")
        return

    # ── Sector momentum (data enrichment, not a hard gate) ────────────────────
    _header("Sector Momentum")
    from momentum import check_sector

    for c in confirmed:
        sc = check_sector(c["symbol"])
        c["sector"] = sc
        label = f"{sc:+.2f}%" if isinstance(sc, float) else sc
        mood  = "🟢" if isinstance(sc, float) and sc > 0 else ("🔴" if isinstance(sc, float) else "⚪")
        print(f"  {mood}  {c['symbol']:<20} Sector avg: {label}")

    # ── Stage 4: Gemini levels (optional, capped at --ai-top) ────────────────
    if args.ai:
        ai_pool = confirmed[:args.ai_top]
        _header(f"Stage 4 — Gemini AI Levels (top {len(ai_pool)} of {len(confirmed)})")
        _info(f"Gemini cap: {args.ai_top}  |  use --ai-top N to change")
        from levels import get_levels
        from chart import draw_chart
        for c in ai_pool:
            ctx = {
                **c,
                **(c.get("consol") or {}),
                **(c.get("conf")   or {}),
                "sector": c.get("sector", "N/A"),
            }
            lvl = get_levels(c["symbol"], ctx)
            c["levels"] = lvl
            if lvl:
                action = lvl.get("ACTION", "").strip().upper()
                _ok(c["symbol"], f"Conviction {lvl.get('CONVICTION', '?')}/10 | {lvl.get('ACTION', '?')}")
                if action in ("BUY", "STRONG BUY"):
                    conf = c.get("conf") or {}
                    path = draw_chart(
                        symbol    = c["symbol"],
                        entry     = conf.get("price") or c["price"],
                        sl        = conf.get("sl",        0),
                        sl_source = conf.get("sl_source", ""),
                        targets   = lvl,
                    )
                    c["chart_path"] = path
                    if path:
                        _ok(c["symbol"], f"chart → {path}")
                    else:
                        _fail(c["symbol"], "chart generation failed")
            else:
                _fail(c["symbol"], "Gemini returned no levels")

    # ── Stage 4b: Gemini Vision (optional, USE_VISUAL_LEVELS in config) ─────────
    if args.ai and USE_VISUAL_LEVELS:
        _header("Stage 4b — Gemini Vision Chart Analysis")
        from visual_levels import get_visual_levels
        vis_pool = confirmed[:args.ai_top]
        for c in vis_pool:
            ctx = {
                **c,
                **(c.get("consol") or {}),
                **(c.get("conf")   or {}),
                "sector": c.get("sector", "N/A"),
            }
            vis_lvl, vis_chart = get_visual_levels(c["symbol"], ctx)
            c["vis_levels"] = vis_lvl
            c["vis_chart"]  = vis_chart
            if vis_lvl:
                _ok(c["symbol"], f"Vision: {vis_lvl.get('ACTION', '?')} | Conviction {vis_lvl.get('CONVICTION', '?')}/10")
            else:
                _fail(c["symbol"], "Vision: no levels returned")

    # ── Send Telegram alerts ──────────────────────────────────────────────────
    _header("Sending Telegram Alerts")
    from alerts import format_alert, format_header, send
    from trade_log import log_trade

    header_msg = format_header(len(confirmed), nifty_chg, nifty_warning)
    send(header_msg)

    from alerts import send_photo
    sent = 0
    for c in confirmed:
        msg = format_alert(
            scan    = c,
            consol  = c.get("consol"),
            conf    = c.get("conf"),
            levels  = c.get("levels"),
            sector  = c.get("sector", "—"),
            nifty   = nifty_chg,
            warning = nifty_warning,
        )
        if c.get("chart_path"):
            if send_photo(c["chart_path"]):
                _ok(c["symbol"], "chart sent to Telegram")
            else:
                _fail(c["symbol"], "chart send failed")

        ok = send(msg)
        if ok:
            _ok(c["symbol"], "sent")
            sent += 1
            if args.ai:
                log_trade(
                    scan    = c,
                    consol  = c.get("consol"),
                    conf    = c.get("conf"),
                    levels  = c.get("levels"),
                    sector  = c.get("sector", "—"),
                    nifty   = nifty_chg,
                    warning = nifty_warning,
                )
        else:
            _fail(c["symbol"], "send failed")

    print(f"\n  ✅  Done — {sent}/{len(confirmed)} alerts sent to Telegram.")


if __name__ == "__main__":
    main()
