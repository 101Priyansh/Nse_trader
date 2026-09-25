"""Market breadth (Nifty gate) + sector momentum check."""
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from config import INDUSTRY_BY_SYMBOL, INDUSTRY_GROUPS, NIFTY_DROP_GATE

_SECTOR_PEER_CAP = 8   # max peers to sample per sector (keeps the fetch fast)


def check_nifty() -> tuple[float, str]:
    """Returns (pct_change, warning_message). Warning is empty string if market is fine."""
    try:
        df = yf.download("^NSEI", period="2d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return 0.0, ""
        closes = df["Close"].values.flatten()
        change = round(float((closes[-1] - closes[-2]) / closes[-2]) * 100, 2)
        warning = f"Nifty {change:+.2f}% — breakout signals may be weaker today" if change < NIFTY_DROP_GATE else ""
        return change, warning
    except Exception:
        return 0.0, ""


def check_sector(symbol: str) -> float | str:
    """Returns average intraday % change of sector peers (auto-grouped from NSE industry data)."""
    industry = INDUSTRY_BY_SYMBOL.get(symbol)
    if not industry:
        return "Unknown"

    peers = [s for s in INDUSTRY_GROUPS.get(industry, []) if s != symbol][:_SECTOR_PEER_CAP]
    if not peers:
        return "Unknown"

    try:
        raw = yf.download(peers, period="1d", interval="15m",
                          progress=False, auto_adjust=True, group_by="ticker")
        changes = []
        for peer in peers:
            try:
                df = raw[peer] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all")
                if len(df) > 1:
                    chg = float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
                    changes.append(round(chg, 2))
            except Exception:
                continue
        return round(sum(changes) / len(changes), 2) if changes else "Unknown"
    except Exception:
        return "Unknown"


if __name__ == "__main__":
    import sys
    nifty_chg, warning = check_nifty()
    status = "⚠️ " + warning if warning else f"✅ Nifty {nifty_chg:+.2f}%"
    print(f"Market: {status}\n")

    tickers = sys.argv[1:] or ["HINDALCO.NS", "MTARTECH.NS", "HFCL.NS"]
    for sym in tickers:
        sc = check_sector(sym)
        label = f"{sc:+.2f}%" if isinstance(sc, float) else sc
        print(f"  {sym:<20} Sector avg: {label}")
