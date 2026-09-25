"""Stage 2 — Base / accumulation detection.
Checks whether the candidate has been consolidating (tight range + rising volume)
before the current move — the 'coil before the spring' pattern.
"""
import warnings

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

warnings.filterwarnings("ignore")

from config import ACC_DAYS_MIN


def check(symbol: str) -> dict | None:
    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 15:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.ta.atr(length=10, append=True)

        df["vol_ma10"] = df["Volume"].rolling(window=10).mean()
        baseline_vol   = float(df["vol_ma10"].iloc[-6])   # avg before last 5 days

        last5 = df.tail(5).copy()

        # Days where volume meaningfully exceeded baseline (accumulation footprint)
        last5["vol_spike"] = last5["Volume"] > (baseline_vol * 1.1)
        acc_days = int(last5["vol_spike"].sum())

        # 5-day price range as % of starting price (tight = good base)
        consol_pct = round(
            float((last5["Close"].max() - last5["Close"].min()) / last5["Close"].iloc[0]) * 100, 2
        )

        # Is volume expanding over last 3 days? (accumulation building)
        vol_expanding = float(last5["Volume"].tail(3).mean()) > baseline_vol

        # ATR compression: is daily range getting tighter? (squeeze signal)
        atr_col = [c for c in df.columns if c.startswith("ATRr")]
        atr_compressed = False
        if atr_col:
            recent_atr = float(df[atr_col[0]].iloc[-1])
            prior_atr  = float(df[atr_col[0]].iloc[-6])
            atr_compressed = recent_atr < prior_atr * 0.85  # 15% tighter than 5 days ago

        passes = (acc_days >= ACC_DAYS_MIN) and (consol_pct < 6.0)

        return {
            "symbol":       symbol,
            "acc_days":     acc_days,
            "consol_pct":   consol_pct,
            "vol_trend":    "Expanding" if vol_expanding else "Flat",
            "atr_squeeze":  atr_compressed,
            "passes":       passes,
        }
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    from config import TICKERS
    tickers = sys.argv[1:] or TICKERS[:20]
    print(f"{'SYMBOL':<18} {'ACC DAYS':>9} {'RANGE %':>8} {'VOL TREND':>12} {'SQUEEZE':>8} {'PASS':>6}")
    print("─" * 65)
    for sym in tickers:
        r = check(sym)
        if r:
            sq  = "✓" if r["atr_squeeze"] else "—"
            ok  = "✅" if r["passes"] else "❌"
            print(f"{sym:<18} {r['acc_days']:>9} {r['consol_pct']:>8} {r['vol_trend']:>12} {sq:>8} {ok:>6}")
