"""Stage 1 — Trend + RVOL + proximity filter with priority scoring."""
import datetime
import logging
import warnings

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from config import (
    DIST_HIGH_MAX, RVOL_MIN_BREAKOUT, RVOL_VOLUME_ANCHOR, TICKERS
)


def _project_volume(raw_vol: float) -> float:
    """Scale partial-day volume to estimated end-of-day during market hours."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(ist)
    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if open_t <= now <= close_t:
        elapsed = max((now - open_t).seconds / 60, 15)
        return raw_vol * (375 / elapsed)
    return raw_vol


def analyze(symbol: str, df: pd.DataFrame | None = None) -> dict | None:
    try:
        if df is None:
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        if len(df) < 50:
            return None

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)

        closes  = df["Close"].values.flatten()
        volumes = df["Volume"].values.flatten()
        ema20   = df["EMA_20"].values.flatten()
        ema50   = df["EMA_50"].values.flatten()
        ema200  = df["EMA_200"].values.flatten()
        highs   = df["High"].values.flatten()

        idx = -2 if np.isnan(closes[-1]) else -1
        price      = float(closes[idx])
        curr_ema20 = float(ema20[idx])
        curr_ema50 = float(ema50[idx])
        curr_ema200 = float(ema200[idx])

        if any(np.isnan(v) for v in [price, curr_ema20, curr_ema50, curr_ema200]):
            return None

        yearly_high = float(np.nanmax(highs))
        dist_high   = ((yearly_high - price) / yearly_high) * 100

        avg_vol    = float(np.nanmean(volumes[-21:-1]))
        raw_vol    = float(volumes[idx])
        proj_vol   = _project_volume(raw_vol)
        rvol       = proj_vol / avg_vol if avg_vol else 0

        is_stacked  = price > curr_ema20 > curr_ema50 > curr_ema200
        is_trending = (curr_ema50 > curr_ema200) and (price > curr_ema50)

        near_barrier   = dist_high <= DIST_HIGH_MAX
        volume_anchor  = rvol >= RVOL_VOLUME_ANCHOR
        min_rvol_ok    = rvol >= RVOL_MIN_BREAKOUT

        qualifies  = is_stacked or is_trending
        has_signal = volume_anchor or (near_barrier and min_rvol_ok)

        if not (qualifies and has_signal):
            return None

        if near_barrier and volume_anchor:
            setup = "PRIME"
        elif near_barrier:
            setup = "BREAKOUT"
        else:
            setup = "MOMENTUM"

        stack_bonus = 1.2 if is_stacked else 1.0
        score = round(rvol * (10 / (dist_high + 1)) * stack_bonus, 2)

        return {
            "symbol":    symbol,
            "price":     round(price, 2),
            "rvol":      round(rvol, 2),
            "dist_high": round(dist_high, 2),
            "setup":     setup,
            "score":     score,
            "stacked":   is_stacked,
        }
    except Exception:
        return None


_BATCH_SIZE = 100   # tickers per yfinance batch request


def _batch_download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download a chunk of tickers in one request. Returns {symbol: df}."""
    try:
        raw = yf.download(
            tickers, period="1y", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        print(f"  [scanner] ⚠️  batch download error: {e}")
        return {}

    out = {}
    for sym in tickers:
        try:
            # Multi-ticker result has a 2-level column index; single-ticker does not
            df = raw[sym].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df = df.dropna(how="all")
            if not df.empty:
                out[sym] = df
        except Exception:
            pass
    return out


def run_scanner(tickers: list[str] = TICKERS) -> list[dict]:
    results      = []
    no_data      = []   # genuine data failures (fetch returned empty)
    not_qualify  = 0    # had data, just didn't meet Stage 1 criteria

    for batch_start in range(0, len(tickers), _BATCH_SIZE):
        chunk = tickers[batch_start:batch_start + _BATCH_SIZE]
        cache = _batch_download(chunk)

        for sym in chunk:
            df = cache.get(sym)
            if df is None:
                no_data.append(sym)
                continue
            r = analyze(sym, df)
            if r:
                results.append(r)
            else:
                not_qualify += 1

    total = len(tickers)
    print(f"  [scanner] {total} tickers — {len(results)} qualified | "
          f"{not_qualify} filtered out | {len(no_data)} no data")

    if no_data:
        fail_pct = len(no_data) / total * 100
        if fail_pct >= 20:
            print(f"  [scanner] ⚠️  {fail_pct:.0f}% data fetch failures — possible rate limit or outage")
        elif len(no_data) <= 10:
            print(f"  [scanner] missing: {', '.join(s.replace('.NS', '') for s in no_data)}")
        else:
            print(f"  [scanner] {len(no_data)} fetch failures (first 10): "
                  f"{', '.join(s.replace('.NS', '') for s in no_data[:10])}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    print(f"Scanning {len(TICKERS)} stocks...")
    hits = run_scanner()
    if not hits:
        print("No setups found.")
    else:
        print(f"\n{'RANK':<5} {'SYMBOL':<18} {'PRICE':>8} {'TYPE':<10} {'RVOL':>6} {'FROM HIGH':>10} {'SCORE':>7}")
        print("─" * 70)
        for i, r in enumerate(hits, 1):
            print(f" {i:<4} {r['symbol']:<18} {r['price']:>8} {r['setup']:<10} {r['rvol']:>6} {r['dist_high']:>9}% {r['score']:>7}")
        print(f"\n{len(hits)} candidates found.")
