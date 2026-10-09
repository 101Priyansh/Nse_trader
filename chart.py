"""
chart.py — annotated 1h candlestick chart generator.

Draws entry, stop loss, and targets on a candlestick chart with
EMA 20/50, volume, RSI, and MACD panels.

Usage (standalone):
    python3 chart.py HINDALCO.NS --entry 1050 --sl 1020 --sl-source candle

Called from pipeline:
    from chart import draw_chart
    path = draw_chart("HINDALCO.NS", entry=1050, sl=1020, sl_source="candle",
                      targets={"T1": 1080, "T2": 1110, "T3": 1140, "T4": 1170})
"""
import argparse
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive, safe for headless / Telegram sends
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Dark chart style ───────────────────────────────────────────────────────────
_MC = mpf.make_marketcolors(
    up     = "#26a69a",   # green
    down   = "#ef5350",   # red
    edge   = "inherit",
    wick   = "inherit",
    volume = "inherit",
)
_DARK_STYLE = mpf.make_mpf_style(
    base_mpf_style = "nightclouds",
    marketcolors   = _MC,
    facecolor      = "#1a1a2e",
    figcolor       = "#1a1a2e",
    gridcolor      = "#2a2a4a",
    gridstyle      = "--",
    y_on_right     = True,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
_C = {
    "entry":   "#00BCD4",   # cyan
    "sl":      "#F44336",   # red
    "t1":      "#4CAF50",   # green
    "t2":      "#8BC34A",
    "t3":      "#CDDC39",
    "t4":      "#FFC107",   # amber
    "ema20":   "#2196F3",   # blue
    "ema50":   "#FF9800",   # orange
    "rsi":     "#9C27B0",   # purple
}

_TARGET_KEYS   = ["T1", "T2", "T3", "T4"]
_TARGET_COLORS = [_C["t1"], _C["t2"], _C["t3"], _C["t4"]]


def draw_chart(
    symbol:     str,
    entry:      float,
    sl:         float,
    sl_source:  str  = "",
    targets:    dict | None = None,
    save_path:  str  | None = None,
) -> str | None:
    """
    Download 1h data, compute indicators, and render an annotated chart.

    Parameters
    ----------
    symbol      : NSE ticker, e.g. "HINDALCO.NS"
    entry       : entry price
    sl          : stop loss price (pre-computed by confirm.py hierarchy)
    sl_source   : label shown on chart ("candle" | "swing" | "atr")
    targets     : dict with keys T1–T4 (values as floats or "₹NNN" strings)
    save_path   : output PNG path; defaults to <system temp>/pro_trader_charts/<SYMBOL>_chart.png

    Returns
    -------
    str path of saved image, or None on failure.
    """
    try:
        df = yf.download(symbol, period="10d", interval="1h",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            print(f"  [chart] insufficient data for {symbol}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.DatetimeIndex(df.index)

        # ── Indicators ────────────────────────────────────────────────────────
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)

        df = df.tail(60).copy()   # last 60 candles keeps chart readable

        # ── Additional plots ─────────────────────────────────────────────────
        # mplfinance panel layout (volume=True auto-inserts panel 1):
        #   0 → price  |  1 → volume  |  2 → RSI  |  3 → MACD histogram
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
            # RSI momentum zone reference lines
            add_plots.append(mpf.make_addplot(
                [52] * len(df), panel=2, color="green", linestyle="--", width=0.7))
            add_plots.append(mpf.make_addplot(
                [75] * len(df), panel=2, color="red",   linestyle="--", width=0.7))

        macd_hist_col = next((c for c in df.columns if "MACDh" in c), None)
        if macd_hist_col:
            hist = df[macd_hist_col].copy()
            bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in hist]
            add_plots.append(mpf.make_addplot(
                hist, panel=3, type="bar", color=bar_colors, ylabel="MACD"))

        # ── Horizontal level lines ────────────────────────────────────────────
        h_prices = [entry, sl]
        h_colors = [_C["entry"], _C["sl"]]

        parsed_targets = {}
        if targets:
            for key, color in zip(_TARGET_KEYS, _TARGET_COLORS):
                raw = targets.get(key)
                if raw is None:
                    continue
                try:
                    val = float(str(raw).replace("₹", "").replace(",", "").strip())
                    parsed_targets[key] = val
                    h_prices.append(val)
                    h_colors.append(color)
                except (ValueError, TypeError):
                    pass

        # ── Render ────────────────────────────────────────────────────────────
        sym_clean  = symbol.replace(".NS", "")
        sl_tag     = f" [{sl_source}]" if sl_source else ""
        chart_title = f"\n{sym_clean}  |  Entry ₹{entry}  |  SL ₹{sl}{sl_tag}"

        n_panels = 4 if macd_hist_col else (3 if rsi_col else 2)
        ratios   = {2: (4, 1, 1.5), 3: (4, 1, 1.5, 1), 4: (4, 1, 1.5, 1)}.get(n_panels, (4, 1))

        fig, axes = mpf.plot(
            df,
            type         = "candle",
            style        = _DARK_STYLE,
            title        = chart_title,
            volume       = True,
            addplot      = add_plots,
            hlines       = dict(hlines=h_prices, colors=h_colors,
                                linestyle="--", linewidths=1.0),
            panel_ratios = ratios,
            figsize      = (14, 9),
            returnfig    = True,
        )

        # ── Text annotations on right edge of price panel ────────────────────
        ax   = axes[0]
        xpos = len(df) + 0.5

        ax.text(xpos, entry, f" Entry ₹{entry}",
                color=_C["entry"], fontsize=8, fontweight="bold", va="center")
        ax.text(xpos, sl,    f" SL ₹{sl}{sl_tag}",
                color=_C["sl"],    fontsize=8, fontweight="bold", va="center")

        for key, color in zip(_TARGET_KEYS, _TARGET_COLORS):
            if key in parsed_targets:
                ax.text(xpos, parsed_targets[key], f" {key} ₹{parsed_targets[key]}",
                        color=color, fontsize=8, va="center")

        # ── Save ──────────────────────────────────────────────────────────────
        if save_path is None:
            charts_dir = Path(tempfile.gettempdir()) / "pro_trader_charts"
            charts_dir.mkdir(exist_ok=True)
            save_path = str(charts_dir / f"{sym_clean}_chart.png")

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [chart] saved → {save_path}")
        return save_path

    except Exception as e:
        print(f"  [chart] error for {symbol}: {e}")
        return None


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw annotated 1h chart")
    parser.add_argument("symbol",          help="NSE ticker, e.g. HINDALCO.NS")
    parser.add_argument("--entry",  type=float, required=True)
    parser.add_argument("--sl",     type=float, required=True)
    parser.add_argument("--sl-source", default="", dest="sl_source")
    parser.add_argument("--t1",     type=float, default=None)
    parser.add_argument("--t2",     type=float, default=None)
    parser.add_argument("--t3",     type=float, default=None)
    parser.add_argument("--t4",     type=float, default=None)
    parser.add_argument("--out",    default=None, help="Output PNG path")
    args = parser.parse_args()

    tgts = {k: getattr(args, k.lower()) for k in _TARGET_KEYS
            if getattr(args, k.lower()) is not None}

    path = draw_chart(
        symbol    = args.symbol,
        entry     = args.entry,
        sl        = args.sl,
        sl_source = args.sl_source,
        targets   = tgts or None,
        save_path = args.out,
    )
    if path:
        print(f"Chart saved: {path}")
