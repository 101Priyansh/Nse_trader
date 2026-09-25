"""Telegram alert formatter and sender."""
import requests

from config import TELE_CHAT_ID, TELE_TOKEN, TELE_CHAT_ID_SID, TELE_TOKEN_SID

SETUP_EMOJI = {"PRIME": "🏆", "BREAKOUT": "🎯", "MOMENTUM": "⚡"}

_RECIPIENTS = [
    (TELE_TOKEN, TELE_CHAT_ID),
    *([(TELE_TOKEN_SID, TELE_CHAT_ID_SID)] if TELE_TOKEN_SID and TELE_CHAT_ID_SID else []),
]


def send_photo(image_path: str) -> bool:
    ok = False
    for token, chat_id in _RECIPIENTS:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(image_path, "rb") as f:
                r = requests.post(url, data={"chat_id": chat_id},
                                  files={"photo": f}, timeout=30)
            if r.status_code == 200:
                ok = True
        except Exception:
            pass
    return ok


def send(message: str) -> bool:
    ok = False
    for token, chat_id in _RECIPIENTS:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        try:
            r = requests.post(url, data=payload, timeout=10)
            if r.status_code == 200:
                ok = True
        except Exception:
            pass
    return ok


def format_alert(
    scan:    dict,
    consol:  dict | None = None,
    conf:    dict | None = None,
    levels:  dict | None = None,
    sector:  str | float = "—",
    nifty:   float = 0.0,
    warning: str = "",
) -> str:
    emoji = SETUP_EMOJI.get(scan["setup"], "📊")
    sym   = scan["symbol"].replace(".NS", "")
    stack = "⬆ Stacked" if scan.get("stacked") else "↗ Trending"

    lines = [
        f"{emoji} <b>{scan['setup']} | {sym}</b>  [{stack}]",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"💰 Price: ₹{scan['price']}  |  Score: {scan['score']}",
        f"📊 RVOL: {scan['rvol']}x  |  From 52w High: {scan['dist_high']}%",
    ]

    if conf:
        rsi_flag = " ⚠️" if not conf["rsi_ok"] else ""
        lines.append(
            f"📈 RSI: {conf['rsi']}{rsi_flag}  |  ADX: {conf['adx']}  |  MACD: {conf['macd']}  |  Div: {conf['div']}"
        )
        sl_src = {"candle": "candle low", "swing": "swing low", "atr": "ATR"}.get(conf.get("sl_source", ""), "")
        lines.append(f"🛑 SL: ₹{conf['sl']}  ({sl_src})")

    if consol:
        sq = " 🔄 ATR squeeze" if consol.get("atr_squeeze") else ""
        lines.append(
            f"🏦 Base: {consol['acc_days']} acc days  |  Range: {consol['consol_pct']}%  |  Vol: {consol['vol_trend']}{sq}"
        )

    sector_str = f"{sector:+.2f}%" if isinstance(sector, float) else str(sector)
    nifty_str  = f"{nifty:+.2f}%" if nifty else "—"
    lines.append(f"🌐 Sector: {sector_str}  |  Nifty: {nifty_str}")

    if warning:
        lines.append(f"⚠️  {warning}")

    if levels:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"🎯 Entry: ₹{levels.get('ENTRY', '—')}",
            f"✅ T1: ₹{levels.get('T1', '—')}"
            + (f"  ({levels['T1_RR']}R)" if levels.get("T1_RR") is not None else "")
            + f"  |  T2: ₹{levels.get('T2', '—')}",
            f"✅ T3: ₹{levels.get('T3', '—')}  |  T4: ₹{levels.get('T4', '—')}",
            f"🛑 SL: ₹{levels.get('SL', '—')}",
            f"⭐ Conviction: {levels.get('CONVICTION', '—')}/10  |  {levels.get('ACTION', '—')}",
        ]
        if note := levels.get("NOTE"):
            lines.append(f"📝 {note}")

    return "\n".join(lines)


def format_header(n_found: int, nifty: float, warning: str) -> str:
    nifty_str = f"{nifty:+.2f}%" if nifty else "—"
    mood = "🟢" if nifty >= 0 else "🔴"
    lines = [
        "🚀 <b>Pro-Trader Scan</b>",
        f"{mood} Nifty: {nifty_str}  |  {n_found} setup{'s' if n_found != 1 else ''} found",
    ]
    if warning:
        lines.append(f"⚠️  {warning}")
    return "\n".join(lines)
