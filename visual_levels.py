"""
visual_levels.py — Chart-driven Gemini Vision analysis.

Flow:
  1. Download 1h data and draw a clean chart (indicators only, no level lines)
  2. Send that chart image + technical context to Gemini Vision
  3. Gemini analyzes both the visible chart patterns AND the indicator data
  4. Parse entry / SL / T1–T4 from Gemini's response
  5. Re-draw chart with levels annotated → save as final PNG

Usage (standalone):
    python3 visual_levels.py HINDALCO.NS --price 1095 --sl 1065 --sl-source candle

Called from pipeline (replaces levels.py + draw_chart calls):
    from visual_levels import get_visual_levels
    levels, chart_path = get_visual_levels("HINDALCO.NS", ctx)
"""
import warnings
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
from google import genai
from google.genai import types

warnings.filterwarnings("ignore")

from chart import _C, _TARGET_KEYS, _TARGET_COLORS, _DARK_STYLE, draw_chart
from config import GEMINI_API_KEY, GEMINI_MODEL

_CHARTS_DIR = Path(tempfile.gettempdir()) / "pro_trader_charts"


# ── Data ───────────────────────────────────────────────────────────────────────

def _download(symbol: str) -> pd.DataFrame | None:
    df = yf.download(symbol, period="10d", interval="1h",
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 30:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index)
    return df


# ── Clean chart (Gemini input) ─────────────────────────────────────────────────

def _draw_clean_chart(symbol: str, df: pd.DataFrame) -> str | None:
    """Indicators-only chart — no entry/SL/target lines. Used as Gemini Vision input."""
    try:
        df = df.tail(60).copy()
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)

        add_plots = []
        if "EMA_20" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["EMA_20"], panel=0, color=_C["ema20"], width=1.2, label="EMA 20"))
        if "EMA_50" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["EMA_50"], panel=0, color=_C["ema50"], width=1.2, label="EMA 50"))

        rsi_col = next((c for c in df.columns if c.startswith("RSI")), None)
        if rsi_col:
            add_plots.append(mpf.make_addplot(
                df[rsi_col], panel=2, color=_C["rsi"], ylabel="RSI", ylim=(0, 100)))
            add_plots.append(mpf.make_addplot(
                [52] * len(df), panel=2, color="green", linestyle="--", width=0.7))
            add_plots.append(mpf.make_addplot(
                [75] * len(df), panel=2, color="red",   linestyle="--", width=0.7))

        macd_hist_col = next((c for c in df.columns if "MACDh" in c), None)
        if macd_hist_col:
            hist       = df[macd_hist_col].copy()
            bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in hist]
            add_plots.append(mpf.make_addplot(
                hist, panel=3, type="bar", color=bar_colors, ylabel="MACD"))

        n_panels = 4 if macd_hist_col else (3 if rsi_col else 2)
        ratios   = {2: (4, 1, 1.5), 3: (4, 1, 1.5, 1), 4: (4, 1, 1.5, 1)}.get(n_panels, (4, 1))
        sym_clean = symbol.replace(".NS", "")

        _CHARTS_DIR.mkdir(exist_ok=True)
        raw_path = str(_CHARTS_DIR / f"{sym_clean}_raw.png")

        fig, _ = mpf.plot(
            df,
            type         = "candle",
            style        = _DARK_STYLE,
            title        = f"\n{sym_clean}  |  1h  |  last 60 candles",
            volume       = True,
            addplot      = add_plots,
            panel_ratios = ratios,
            figsize      = (14, 9),
            returnfig    = True,
        )
        fig.savefig(raw_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return raw_path

    except Exception as e:
        print(f"  [visual_levels] clean chart error: {e}")
        return None


# ── Gemini Vision call ─────────────────────────────────────────────────────────

def _ask_gemini(symbol: str, context: dict, chart_path: str) -> dict | None:
    """Send chart image + technical + structural context to Gemini Vision. Returns parsed dict or None."""
    try:
        from levels import get_structural_levels
        price  = context.get("price", 0)
        lvls   = get_structural_levels(symbol, price, context)

        swing_str  = ", ".join(f"₹{l}" for l in lvls["swing_highs"]) or "none found in range"
        round_str  = ", ".join(f"₹{l}" for l in lvls["round_numbers"])
        yearly     = f"₹{lvls['yearly_high']}"   if lvls["yearly_high"]   else "N/A"
        prev_high  = f"₹{lvls['prev_day_high']}" if lvls["prev_day_high"] else "N/A"
        base_range = (f"₹{lvls['base_low']} – ₹{lvls['base_high']}"
                      if lvls["base_low"] else "N/A")
        atr_str    = f"₹{lvls['atr_price']}" if lvls["atr_price"] else "N/A"
        sl         = context.get("sl", "N/A")

        client = genai.Client(api_key=GEMINI_API_KEY)

        with open(chart_path, "rb") as f:
            image_bytes = f.read()

        prompt = f"""You are a Quantitative Trader analyzing a 1h candlestick chart for {symbol}.
The chart shows: candlesticks, EMA 20 (blue), EMA 50 (orange), Volume, RSI(14) with 52/75 reference zones, and MACD histogram.

Technical Context (from pipeline):
- Current Price   : {context.get('price')}
- Setup           : {context.get('setup')} | Score: {context.get('score')}
- RVOL            : {context.get('rvol')}x | Distance from 52w High: {context.get('dist_high')}%
- RSI (1h)        : {context.get('rsi', 'N/A')} | ADX: {context.get('adx', 'N/A')} | MACD: {context.get('macd', 'N/A')}
- ATR (1h, 14)    : {context.get('atr', 'N/A')} | Above Daily 200 EMA: {context.get('above_200', 'N/A')}
- Pre-computed SL : ₹{sl} [{context.get('sl_source', 'N/A')}]
- Base            : {context.get('acc_days', 'N/A')} accumulation days | Consolidation: {context.get('consol_pct', 'N/A')}% range
- ATR Squeeze     : {context.get('atr_squeeze', 'N/A')} | Vol Trend: {context.get('vol_trend', 'N/A')}
- Sector Momentum : {context.get('sector', 'N/A')}

Structural Resistance Levels above ₹{price} (pre-computed from 90-day OHLC):
- Swing highs   : {swing_str}
- Round numbers : {round_str}
- 52-week high  : {yearly}
- Prev day high : {prev_high}
- Base range    : {base_range}

Step 1 — Read the chart FIRST.
Look at the 1h candles for visible resistance clusters, wick rejections, EMA angle, and volume pattern.
Note any chart-visible levels that confirm or conflict with the pre-computed structural levels above.

Step 2 — Choose targets.
Cross-reference what you see on the chart with the pre-computed levels.
  - Prefer levels that appear BOTH in the structural data AND visibly on the 1h chart
  - If two levels are within 1% of each other, pick the stronger — skip the other
  - T1 must be reachable in 1–2 sessions (BTST / short-term swing context)
  - Do NOT space targets by ATR — use the structural levels provided
  - Use the pre-computed SL as-is. Do not recalculate.
  - Set ACTION to AVOID if: RSI > 74, or no clear resistance within 5%, or RVOL < 1.0 with weak sector

Step 3 — Output. Strictly one value per line, no preamble, no markdown:
ANALYSIS: [one line — what the chart confirms or changes vs the pre-computed levels]
CONVICTION: [1-10]
ACTION: [STRONG BUY / BUY / HOLD / AVOID]
ENTRY: [tight zone ±0.3% around ₹{price}]
T1: [level]
T2: [level]
T3: [level]
T4: [level]
SL: ₹{sl}
NOTE: [one sentence — specific candle pattern or chart structure confirming this trade NOW]
"""
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )

        result = {}
        for line in response.text.strip().split("\n"):
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip().upper()] = v.strip()
        return result if "T1" in result else None

    except Exception as e:
        print(f"  [visual_levels] Gemini error: {e}")
        return None


# ── Price parser ───────────────────────────────────────────────────────────────

def _parse_price(raw: str | None) -> float | None:
    """Handles '1080', '₹1080', '1080-1085' (takes lower bound)."""
    if not raw:
        return None
    try:
        clean = str(raw).replace("₹", "").replace(",", "").strip()
        if "-" in clean:
            clean = clean.split("-")[0].strip()
        return float(clean)
    except (ValueError, TypeError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_visual_levels(symbol: str, context: dict) -> tuple[dict | None, str | None]:
    """
    Full flow: clean chart → Gemini Vision → annotated chart.

    Parameters
    ----------
    symbol  : NSE ticker, e.g. "HINDALCO.NS"
    context : dict from pipeline (price, sl, sl_source, rsi, adx, atr, setup, …)

    Returns
    -------
    (levels_dict, annotated_chart_path)
    levels_dict keys: CONVICTION, ACTION, ENTRY, T1–T4, SL, NOTE
    Either value may be None on failure.
    """
    sym_clean = symbol.replace(".NS", "")

    # 1 — download data
    df = _download(symbol)
    if df is None:
        print(f"  [visual_levels] no data for {symbol}")
        return None, None

    # 2 — clean chart for Gemini input
    raw_path = _draw_clean_chart(symbol, df.copy())
    if not raw_path:
        return None, None
    print(f"  [visual_levels] raw chart  → {raw_path}")

    # 3 — Gemini Vision
    levels = _ask_gemini(symbol, context, raw_path)
    if not levels:
        print(f"  [visual_levels] no levels returned for {symbol}")
        return None, raw_path

    # 4 — re-draw with levels annotated
    entry = _parse_price(levels.get("ENTRY")) or context.get("price", 0)
    sl    = _parse_price(levels.get("SL"))    or context.get("sl", 0)

    _CHARTS_DIR.mkdir(exist_ok=True)
    annotated_path = str(_CHARTS_DIR / f"{sym_clean}_chart.png")

    draw_chart(
        symbol    = symbol,
        entry     = entry,
        sl        = sl,
        sl_source = "gemini",
        targets   = levels,
        save_path = annotated_path,
    )
    print(f"  [visual_levels] annotated  → {annotated_path}")

    return levels, annotated_path


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from confirm import check as confirm_check

    parser = argparse.ArgumentParser(description="Visual Gemini level analysis")
    parser.add_argument("symbol", help="NSE ticker, e.g. HINDALCO.NS")
    args = parser.parse_args()

    print(f"\n{'━'*50}")
    print(f"  Visual Level Analysis — {args.symbol}")
    print(f"{'━'*50}")

    print("  Running Stage 3 confirmation to build context...")
    conf = confirm_check(args.symbol)
    if not conf:
        print("  ✗  confirm.py returned no data — cannot proceed.")
        raise SystemExit(1)

    ctx = {
        "price":     conf["rsi"],   # placeholder — overwritten below
        **conf,
    }
    # confirm.py doesn't store price separately; re-derive from SL + ATR is lossy.
    # Re-download close price directly.
    import yfinance as yf
    _df = yf.download(args.symbol, period="1d", interval="1h",
                      progress=False, auto_adjust=True)
    if not _df.empty:
        if hasattr(_df.columns, "get_level_values"):
            _df.columns = _df.columns.get_level_values(0)
        ctx["price"] = round(float(_df["Close"].iloc[-1]), 2)

    print(f"  Price ₹{ctx['price']}  |  SL ₹{conf['sl']} ({conf['sl_source']})"
          f"  |  RSI {conf['rsi']}  |  ADX {conf['adx']}")

    levels, chart_path = get_visual_levels(args.symbol, ctx)

    if levels:
        print("\n  Gemini output:")
        for k, v in levels.items():
            print(f"    {k:<12}: {v}")
    else:
        print("\n  No levels returned.")

    if chart_path:
        print(f"\n  Chart: {chart_path}")
