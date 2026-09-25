"""
Trade logger — appends one JSON record per alerted trade to trade_log.jsonl.
Called from pipeline.py after each successful Telegram send.

Format: JSON Lines (.jsonl) — one JSON object per line, easy to append and grep.
Each record is self-contained: all data needed for later P&L simulation is included.
"""
import json
import os
import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "trade_log.jsonl")


def _parse_levels(levels: dict | None) -> dict:
    """Extract and coerce numeric fields from Gemini levels dict."""
    if not levels:
        return {
            "conviction": None, "action": None, "entry_zone": None,
            "t1": None, "t2": None, "t3": None, "t4": None,
            "ai_sl": None, "t1_rr": None, "ai_note": None,
        }

    def _num(key):
        val = levels.get(key, "")
        try:
            return float(str(val).replace("₹", "").replace(",", "").strip())
        except (ValueError, AttributeError):
            return None

    return {
        "conviction":  levels.get("CONVICTION"),
        "action":      levels.get("ACTION"),
        "entry_zone":  levels.get("ENTRY"),
        "t1":          _num("T1"),
        "t2":          _num("T2"),
        "t3":          _num("T3"),
        "t4":          _num("T4"),
        "ai_sl":       _num("SL"),
        "t1_rr":       levels.get("T1_RR"),
        "ai_note":     levels.get("NOTE"),
        "ai_analysis": levels.get("ANALYSIS"),
    }


def log_trade(
    scan:    dict,
    consol:  dict | None,
    conf:    dict | None,
    levels:  dict | None,
    sector:  str | float,
    nifty:   float,
    warning: str,
) -> None:
    """Append one trade record to trade_log.jsonl."""
    now = datetime.datetime.now()

    lvl = _parse_levels(levels)
    con = conf   or {}
    csl = consol or {}

    record = {
        # ── When ──────────────────────────────────────────────────────────────
        "timestamp":    now.strftime("%Y-%m-%dT%H:%M:%S"),
        "date":         now.strftime("%Y-%m-%d"),
        "time":         now.strftime("%H:%M:%S"),

        # ── What ──────────────────────────────────────────────────────────────
        "symbol":       scan.get("symbol"),
        "setup":        scan.get("setup"),
        "score":        scan.get("score"),
        "stacked":      scan.get("stacked", False),

        # ── Stage 1 — trend + RVOL ────────────────────────────────────────────
        "price":        scan.get("price"),
        "rvol":         scan.get("rvol"),
        "dist_high":    scan.get("dist_high"),

        # ── Stage 2 — consolidation ───────────────────────────────────────────
        "acc_days":     csl.get("acc_days"),
        "consol_pct":   csl.get("consol_pct"),
        "vol_trend":    csl.get("vol_trend"),
        "atr_squeeze":  csl.get("atr_squeeze", False),

        # ── Stage 3 — RSI / ADX / SL ──────────────────────────────────────────
        "rsi":          con.get("rsi"),
        "adx":          con.get("adx"),
        "macd":         con.get("macd"),
        "above_200":    con.get("above_200"),
        "div":          con.get("div"),
        "atr":          con.get("atr"),
        "sl":           con.get("sl"),
        "sl_source":    con.get("sl_source"),

        # ── Market context ────────────────────────────────────────────────────
        "sector":       sector if isinstance(sector, str) else (
                            f"{sector:+.2f}%" if isinstance(sector, float) else None),
        "nifty":        nifty,
        "nifty_warning": warning or None,

        # ── Stage 4 — Gemini AI levels (None if --ai not used) ───────────────
        **lvl,

        # ── Outcome fields (filled in later) ─────────────────────────────────
        "exit_date":    None,
        "exit_price":   None,
        "exit_reason":  None,   # "T1"/"T2"/"T3"/"T4"/"SL"/"manual"
        "pnl_pct":      None,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
