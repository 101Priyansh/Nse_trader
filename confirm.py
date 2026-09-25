"""Stage 3 — RSI / ADX / MACD confirmation on 1h data.
Filters out overbought entries and confirms trend strength before alerting.

── Setup-aware SL logic ───────────────────────────────────────────────────────
Each setup type uses a different primary SL and a wider/tighter ATR gate:

  BREAKOUT / PRIME  → base low  (quantile 0.05 of last 35 × 1h candles, ~5 days)
                       gate: 6× ATR  — base stops are structurally wider
                       fallback: swing low → 1.5× ATR

  MOMENTUM          → EMA 20 on 1h minus 0.2× ATR buffer
                       gate: 3× ATR  — EMA can be further on fast movers
                       fallback: candle low → swing low → 1.5× ATR

  DEFAULT           → trigger candle low
                       gate: 2× ATR  — tight, suited for harami / engulfing entries
                       fallback: swing low → 1.5× ATR

  base_low  : structural floor using quantile(0.05) — ignores outlier wick spikes
  swing_low : minimum low of last 20 × 1h candles
  candle_low: low of the most recent 1h candle (trigger candle)
──────────────────────────────────────────────────────────────────────────────
"""
import warnings

import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

warnings.filterwarnings("ignore")

from config import ADX_MIN, RSI_MAX, RSI_MIN


def check(symbol: str, setup: str = "") -> dict | None:
    try:
        df_1h = yf.download(symbol, period="1mo", interval="1h", progress=False, auto_adjust=True)
        df_1d = yf.download(symbol, period="1y",  interval="1d", progress=False, auto_adjust=True)
        if df_1h.empty or len(df_1h) < 20 or df_1d.empty:
            return None

        for df in [df_1h, df_1d]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        df_1h.ta.rsi(length=14, append=True)
        df_1h.ta.adx(length=14, append=True)
        df_1h.ta.macd(append=True)
        df_1h.ta.atr(length=14, append=True)
        df_1h.ta.ema(length=20, append=True)   # needed for MOMENTUM SL
        df_1d.ta.ema(length=200, append=True)

        l1h   = df_1h.iloc[-1]
        l1d   = df_1d.iloc[-1]
        price = float(l1h["Close"])

        rsi = float(l1h["RSI_14"])
        adx = float(l1h["ADX_14"])
        atr = float(l1h["ATRr_14"])

        macd_hist_col = [c for c in df_1h.columns if "MACDh" in c]
        macd_rising = False
        if macd_hist_col:
            col = macd_hist_col[0]
            macd_rising = float(df_1h[col].iloc[-1]) > float(df_1h[col].iloc[-2])

        ema200_daily = float(l1d["EMA_200"])
        above_200    = price > ema200_daily

        # RSI bullish divergence on last 20 1h candles
        last20 = df_1h.tail(20)
        low_idx    = last20["Low"].idxmin()
        rsi_at_low = float(last20.loc[low_idx, "RSI_14"])
        curr_rsi   = float(last20["RSI_14"].iloc[-1])
        bullish_div = (float(last20["Low"].iloc[-1]) > float(last20["Low"].min())) and (curr_rsi > rsi_at_low)

        # ── Setup-aware SL ────────────────────────────────────────────────────
        # Each candidate is gated: if farther than 2× ATR from entry, skip to next.
        candle_low = float(l1h["Low"])
        swing_low  = float(last20["Low"].min())
        # base_low: structural floor of last ~5 trading days (35 × 1h candles)
        # quantile(0.05) ignores outlier wick spikes — finds the real support floor
        base_low   = float(df_1h.tail(35)["Low"].quantile(0.05))
        # ema20_sl: just below EMA 20 on 1h with a 0.2× ATR buffer — used for momentum
        ema20_val  = float(l1h["EMA_20"]) if "EMA_20" in df_1h.columns else None
        ema20_sl   = round(ema20_val - 0.2 * atr, 2) if ema20_val else None

        # Gate is wider for setups where a structurally larger stop is expected
        _gates = {"BREAKOUT": 6.0, "PRIME": 6.0, "MOMENTUM": 3.0}
        atr_gate = atr * _gates.get(setup, 2.0)

        def _within_gate(level):
            return level is not None and (price - level) <= atr_gate

        if setup in ("BREAKOUT", "PRIME"):
            # SL below the consolidation base — a pullback to the base invalidates the breakout
            candidates = [
                (base_low,   "base"),
                (swing_low,  "swing"),
                (price - 1.5 * atr, "atr"),
            ]
        elif setup == "MOMENTUM":
            # SL below EMA 20 — trend is intact as long as price holds above EMA 20
            candidates = [
                (ema20_sl,   "ema20"),
                (candle_low, "candle"),
                (swing_low,  "swing"),
                (price - 1.5 * atr, "atr"),
            ]
        else:
            # Default: tight candle-based stop (harami / engulfing entries)
            candidates = [
                (candle_low, "candle"),
                (swing_low,  "swing"),
                (price - 1.5 * atr, "atr"),
            ]

        sl, sl_source = next(
            ((round(lvl, 2), src) for lvl, src in candidates if _within_gate(lvl)),
            (round(price - 1.5 * atr, 2), "atr"),
        )

        rsi_ok   = RSI_MIN <= rsi <= RSI_MAX
        adx_ok   = adx >= ADX_MIN
        trend_ok = above_200

        return {
            "symbol":    symbol,
            "price":     round(price, 2),
            "rsi":       round(rsi, 1),
            "adx":       round(adx, 1),
            "atr":       round(atr, 4),
            "macd":      "↑" if macd_rising else "↓",
            "above_200": above_200,
            "div":       "✓" if bullish_div else "—",
            "sl":        sl,
            "sl_source": sl_source,   # "candle" | "swing" | "atr"
            # "candle_low": candle_low,  # uncomment to expose raw levels
            # "swing_low":  swing_low,
            "rsi_ok":    rsi_ok,
            "adx_ok":    adx_ok,
            "passes":    rsi_ok and adx_ok and trend_ok,
        }
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["HINDALCO.NS", "MTARTECH.NS", "DIVISLAB.NS"]
    print(f"{'SYMBOL':<18} {'RSI':>5} {'ADX':>5} {'MACD':>5} {'>200':>5} {'DIV':>4} {'SL':>10} {'SRC':>6} {'PASS':>6}")
    print("─" * 72)
    for sym in tickers:
        r = check(sym)
        if r:
            ok = "✅" if r["passes"] else "❌"
            ab = "✓" if r["above_200"] else "✗"
            print(f"{sym:<18} {r['rsi']:>5} {r['adx']:>5} {r['macd']:>5} {ab:>5} {r['div']:>4} {r['sl']:>10} {r['sl_source']:>6} {ok:>6}")
        else:
            print(f"{sym:<18} — data unavailable")
