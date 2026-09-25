"""
Backtest — Pipeline v2.1 vs known successful trades

Re-runs all 3 technical stages using data as it existed on the day BEFORE each
call was made, to audit which trades the pipeline would have caught/missed and
why — so thresholds can be calibrated.

Each trade can have an optional "call_date" (YYYY-MM-DD). The cutoff used for
that stock's data download will be call_date (exclusive end = call_date, so
data up to call_date - 1 day is used). If no call_date is set, DEFAULT_CUTOFF
is used as fallback.

Usage:
  python3 backtest.py              # full audit, all 3 stages
  python3 backtest.py --stage 1    # Stage 1 only (fast, ~30 sec)
  python3 backtest.py --stage 2    # Stage 1 + 2
"""
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf

warnings.filterwarnings("ignore")

from config import (
    ACC_DAYS_MIN, ADX_MIN, DIST_HIGH_MAX,
    RSI_MAX, RSI_MIN, RVOL_MIN_BREAKOUT, RVOL_VOLUME_ANCHOR,
)

# ── Audit config ──────────────────────────────────────────────────────────────
# Fallback cutoff used when a trade has no "call_date" set.
# Set this to the day AFTER the earliest call in your dataset.
DEFAULT_CUTOFF = "2026-05-14"

# Add "call_date": "YYYY-MM-DD" to each trade once you have the exact dates.
# The cutoff for that stock will automatically become that date (exclusive).
KNOWN_TRADES = [
    {"symbol": "JINDRILL.NS",   "entry": 575,   "gain": 9.7,  "result": "Achieved"},
    {"symbol": "SASKEN.NS",     "entry": 1745,  "gain": 12.2, "result": "Achieved"},
    {"symbol": "HINDALCO.NS",   "entry": 1050,  "gain": None, "result": "Ongoing"},
    {"symbol": "ONGC.NS",       "entry": 295.6, "gain": 1.2,  "result": "Partial"},
    {"symbol": "HFCL.NS",       "entry": 136,   "gain": 13.7, "result": "Achieved"},
    {"symbol": "TEXRAIL.NS",    "entry": 120,   "gain": 8.7,  "result": "Achieved",
     "note": "verify: caller said TEXMACO — confirm TEXRAIL.NS is correct"},
    {"symbol": "HINDCOPPER.NS", "entry": 592,   "gain": 4.4,  "result": "Achieved"},
    {"symbol": "MTARTECH.NS",   "entry": 7300,  "gain": 5.8,  "result": "Achieved"},
    {"symbol": "MCX.NS",        "entry": 3278,  "gain": 2.2,  "result": "Achieved"},
    {"symbol": "ALKYLAMINE.NS", "entry": 1865,  "gain": 1.2,  "result": "Partial"},
    {"symbol": "BAJEL.NS",      "entry": 195,   "gain": 2.4,  "result": "Partial"},
    {"symbol": "NAZARA.NS",     "entry": 280,   "gain": 3.4,  "result": "Partial"},
    {"symbol": "SAREGAMA.NS",   "entry": 402,   "gain": 6.9,  "result": "Partial"},
    {"symbol": "ARVIND.NS",     "entry": 462,   "gain": 3.6,  "result": "Partial"},
    # BHARAT COKING COAL — NSE listing unverified, add once confirmed
]


def _cutoff_for(trade: dict) -> str:
    """Return the exclusive end date for yfinance download for this trade."""
    if "call_date" in trade:
        return trade["call_date"]
    return DEFAULT_CUTOFF

# ── Data downloaders (date-based, not period-based) ───────────────────────────

def _dl(symbol, days, interval, cutoff: str):
    end   = datetime.datetime.strptime(cutoff, "%Y-%m-%d")
    start = (end - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=cutoff,
                     interval=interval, progress=False, auto_adjust=True)
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Stage runners (self-contained, threshold-aware) ───────────────────────────

def stage1(symbol: str, cutoff: str) -> tuple[dict | None, str]:
    df = _dl(symbol, 365, "1d", cutoff)
    if df.empty or len(df) < 50:
        return None, "no data"

    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    closes  = df["Close"].values.flatten()
    volumes = df["Volume"].values.flatten()
    e20     = df["EMA_20"].values.flatten()
    e50     = df["EMA_50"].values.flatten()
    e200    = df["EMA_200"].values.flatten()
    highs   = df["High"].values.flatten()

    idx   = -2 if np.isnan(closes[-1]) else -1
    price = float(closes[idx])
    curr_e20, curr_e50, curr_e200 = float(e20[idx]), float(e50[idx]), float(e200[idx])

    if any(np.isnan(v) for v in [price, curr_e20, curr_e50, curr_e200]):
        return None, "NaN in indicators"

    yearly_high = float(np.nanmax(highs))
    dist_high   = round(((yearly_high - price) / yearly_high) * 100, 2)
    avg_vol     = float(np.nanmean(volumes[-21:-1]))
    rvol        = round(float(volumes[idx]) / avg_vol, 2) if avg_vol else 0

    is_stacked  = price > curr_e20 > curr_e50 > curr_e200
    is_trending = (curr_e50 > curr_e200) and (price > curr_e50)
    near_barrier   = dist_high <= DIST_HIGH_MAX
    volume_anchor  = rvol >= RVOL_VOLUME_ANCHOR
    min_rvol_ok    = rvol >= RVOL_MIN_BREAKOUT

    qualifies  = is_stacked or is_trending
    has_signal = volume_anchor or (near_barrier and min_rvol_ok)
    passes     = qualifies and has_signal

    setup = "PRIME" if (near_barrier and volume_anchor) else ("BREAKOUT" if near_barrier else "MOMENTUM")
    info  = {"price": price, "dist_high": dist_high, "rvol": rvol,
             "is_stacked": is_stacked, "is_trending": is_trending, "setup": setup}

    if passes:
        return info, "OK"

    # Describe the failure
    if not qualifies:
        why = f"not in uptrend  (price {'>' if price > curr_e50 else '<'} EMA50, EMA50 {'>' if curr_e50 > curr_e200 else '<'} EMA200)"
    elif not near_barrier and not volume_anchor:
        why = f"dist_high {dist_high}% > {DIST_HIGH_MAX}%  AND  rvol {rvol} < {RVOL_VOLUME_ANCHOR}"
    elif not near_barrier:
        why = f"dist_high {dist_high}% > {DIST_HIGH_MAX}% threshold"
    else:
        why = f"rvol {rvol} < {RVOL_MIN_BREAKOUT} minimum"
    return info, why


def stage2(symbol: str, cutoff: str) -> tuple[dict | None, str]:
    df = _dl(symbol, 40, "1d", cutoff)
    if df.empty or len(df) < 15:
        return None, "no data"

    df.ta.atr(length=10, append=True)
    df["vol_ma10"] = df["Volume"].rolling(10).mean()
    baseline = float(df["vol_ma10"].iloc[-6])
    last5    = df.tail(5).copy()
    last5["spike"] = last5["Volume"] > (baseline * 1.1)
    acc_days    = int(last5["spike"].sum())
    consol_pct  = round(float(
        (last5["Close"].max() - last5["Close"].min()) / last5["Close"].iloc[0]
    ) * 100, 2)
    vol_expanding = float(last5["Volume"].tail(3).mean()) > baseline

    atr_col       = [c for c in df.columns if c.startswith("ATRr")]
    atr_compressed = (
        float(df[atr_col[0]].iloc[-1]) < float(df[atr_col[0]].iloc[-6]) * 0.85
        if atr_col else False
    )

    passes = (acc_days >= ACC_DAYS_MIN) and (consol_pct < 6.0)
    info   = {"acc_days": acc_days, "consol_pct": consol_pct,
              "vol_trend": "Expanding" if vol_expanding else "Flat",
              "atr_squeeze": atr_compressed}

    if passes:
        return info, "OK"
    why = (f"only {acc_days} acc day(s) (need ≥{ACC_DAYS_MIN})"
           if acc_days < ACC_DAYS_MIN
           else f"range {consol_pct}% too wide (need <6%)")
    return info, why


def stage3(symbol: str, cutoff: str) -> tuple[dict | None, str]:
    df_1h = _dl(symbol, 40,  "1h", cutoff)
    df_1d = _dl(symbol, 365, "1d", cutoff)
    if df_1h.empty or len(df_1h) < 20 or df_1d.empty:
        return None, "no data"

    df_1h.ta.rsi(length=14, append=True)
    df_1h.ta.adx(length=14, append=True)
    df_1h.ta.macd(append=True)
    df_1h.ta.atr(length=14, append=True)
    df_1d.ta.ema(length=200, append=True)

    l1h   = df_1h.iloc[-1]
    price = float(l1h["Close"])
    rsi   = float(l1h["RSI_14"])
    adx   = float(l1h["ADX_14"])
    atr   = float(l1h["ATRr_14"])

    macd_col    = [c for c in df_1h.columns if "MACDh" in c]
    macd_rising = (float(df_1h[macd_col[0]].iloc[-1]) > float(df_1h[macd_col[0]].iloc[-2])
                   if macd_col else False)

    above_200 = price > float(df_1d.iloc[-1]["EMA_200"])
    rsi_ok    = RSI_MIN <= rsi <= RSI_MAX
    passes    = rsi_ok and above_200

    info = {"rsi": round(rsi, 1), "adx": round(adx, 1), "macd": "↑" if macd_rising else "↓",
            "above_200": above_200}

    if passes:
        return info, "OK"
    why = (f"RSI {rsi:.1f} outside {RSI_MIN}–{RSI_MAX} zone"
           if not rsi_ok
           else "price below daily 200 EMA")
    return info, why


# ── Report helpers ────────────────────────────────────────────────────────────

def _bar(label, passes, detail=""):
    icon = "✅" if passes else "❌"
    return f"    {icon}  Stage {label}  {detail}"


def print_stock(i, total, trade, s1, s1_why, s2, s2_why, s3, s3_why, max_stage):
    sym    = trade["symbol"]
    entry  = trade["entry"]
    gain   = f"+{trade['gain']}%" if trade["gain"] else "ongoing"
    result = trade["result"]
    note   = trade.get("note", "")

    cutoff_used = trade.get("call_date", DEFAULT_CUTOFF)
    date_label  = f"cutoff: {cutoff_used}" if "call_date" in trade else f"fallback cutoff: {DEFAULT_CUTOFF}"
    print(f"\n  [{i}/{total}]  {sym}   entry ₹{entry}  |  gain {gain}  |  {result}  ({date_label})")
    if note:
        print(f"          ⚠️  {note}")

    if s1 is None:
        print(f"    ⚠️   Stage 1 — no data returned")
        return "no_data"

    s1_pass = (s1_why == "OK")
    d = s1
    detail1 = (f"price ₹{d['price']}  dist_high {d['dist_high']}%  "
               f"rvol {d['rvol']}x  {'⬆ stacked' if d['is_stacked'] else '↗ trending' if d['is_trending'] else '↘ weak'}"
               f"  [{d['setup']}]" if s1_pass else f"— {s1_why}")
    print(_bar("1", s1_pass, detail1))

    if not s1_pass or max_stage < 2:
        return "missed_s1" if not s1_pass else "stopped"

    s2_pass = (s2_why == "OK") if s2 else False
    if s2 is None:
        print(f"    ⚠️   Stage 2 — no data")
    else:
        d2 = s2
        detail2 = (f"{d2['acc_days']} acc days  range {d2['consol_pct']}%  "
                   f"{d2['vol_trend']}  {'🔄 ATR squeeze' if d2['atr_squeeze'] else ''}"
                   if s2_pass else f"— {s2_why}")
        print(_bar("2", s2_pass, detail2))

    if not s2_pass or max_stage < 3:
        return "missed_s2" if not s2_pass else "stopped"

    s3_pass = (s3_why == "OK") if s3 else False
    if s3 is None:
        print(f"    ⚠️   Stage 3 — no data")
    else:
        d3 = s3
        detail3 = (f"RSI {d3['rsi']}  ADX {d3['adx']}  MACD {d3['macd']}  "
                   f"{'above' if d3['above_200'] else 'BELOW'} 200 EMA"
                   if s3_pass else f"— {s3_why}")
        print(_bar("3", s3_pass, detail3))

    if s3_pass:
        print(f"    🎯  WOULD HAVE BEEN CAUGHT")
        return "caught"
    return "missed_s3"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3,
                        help="Run stages up to N (default: 3)")
    args = parser.parse_args()

    print("\n" + "═" * 68)
    print("  BACKTEST — Pipeline v2.1 vs 15 Known Successful Trades")
    print(f"  Cutoff date : May 13, 2026 (day before calls started)")
    print(f"  Stages run  : 1–{args.stage}")
    print(f"  Thresholds  : dist_high ≤{DIST_HIGH_MAX}%  rvol ≥{RVOL_MIN_BREAKOUT}  "
          f"RSI {RSI_MIN}–{RSI_MAX}  ADX ≥{ADX_MIN}")
    print("═" * 68)

    outcomes: dict[str, list] = {
        "caught":    [],
        "missed_s1": [],
        "missed_s2": [],
        "missed_s3": [],
        "no_data":   [],
        "stopped":   [],
    }
    s1_fail_reasons: list[str] = []
    stage1_details: list[dict] = []
    total = len(KNOWN_TRADES)

    for i, trade in enumerate(KNOWN_TRADES, 1):
        sym    = trade["symbol"]
        cutoff = _cutoff_for(trade)
        s1, s1_why = stage1(sym, cutoff)
        s2 = s2_why = s3 = s3_why = None
        if s1 and s1_why == "OK" and args.stage >= 2:
            s2, s2_why = stage2(sym, cutoff)
        if s2 and s2_why == "OK" and args.stage >= 3:
            s3, s3_why = stage3(sym, cutoff)

        outcome = print_stock(i, total, trade, s1, s1_why, s2, s2_why, s3, s3_why, args.stage)
        outcomes[outcome].append(sym)

        if s1:
            stage1_details.append({**s1, "symbol": sym, "gain": trade["gain"],
                                    "s1_pass": s1_why == "OK"})
        if not s1_why == "OK" and s1:
            s1_fail_reasons.append(s1_why)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "═" * 68)
    print("  SUMMARY")
    print("═" * 68)
    caught_n = len(outcomes["caught"])
    missed_n = total - caught_n - len(outcomes["stopped"]) - len(outcomes["no_data"])
    print(f"\n  Total trades audited : {total}")
    if args.stage == 3:
        print(f"  ✅  Caught (all 3 stages) : {caught_n}  ({round(caught_n/total*100)}%)")
        print(f"  ❌  Missed at Stage 1     : {len(outcomes['missed_s1'])}")
        print(f"  ❌  Missed at Stage 2     : {len(outcomes['missed_s2'])}")
        print(f"  ❌  Missed at Stage 3     : {len(outcomes['missed_s3'])}")

    # Stage 1 failures breakdown
    if outcomes["missed_s1"]:
        print(f"\n  Stage 1 failures ({len(outcomes['missed_s1'])} stocks):")
        for sym in outcomes["missed_s1"]:
            d = next((x for x in stage1_details if x["symbol"] == sym), None)
            if d:
                gain = f"+{d['gain']}%" if d.get("gain") else "ongoing"
                print(f"    {sym:<22} dist_high {d['dist_high']:>6}%  rvol {d['rvol']:>5}x  gain {gain}")

    # dist_high distribution across all stocks
    if stage1_details:
        print("\n  dist_high distribution for ALL 14 trades on May 13:")
        print(f"  {'SYMBOL':<22} {'DIST_HIGH':>10} {'RVOL':>6} {'TREND':>10} {'GAIN':>8} {'S1':>4}")
        print("  " + "─" * 65)
        for d in sorted(stage1_details, key=lambda x: x["dist_high"]):
            trend = "stacked" if d["is_stacked"] else ("trending" if d["is_trending"] else "weak")
            gain  = f"+{d['gain']}%" if d.get("gain") else "ongoing"
            ok    = "✅" if d["s1_pass"] else "❌"
            print(f"  {d['symbol']:<22} {d['dist_high']:>9}%  {d['rvol']:>5}x  {trend:>10}  {gain:>8}  {ok}")

    # Threshold recommendation
    if stage1_details:
        missed_dist = [d["dist_high"] for d in stage1_details
                       if not d["s1_pass"] and d.get("gain") and d["gain"] >= 5.0]
        if missed_dist:
            suggested = round(max(missed_dist) + 1, 0)
            print(f"\n  ⚡  High-gain trades (≥5%) missed due to dist_high threshold:")
            print(f"      Max dist_high among them: {max(missed_dist):.1f}%")
            print(f"      Suggested new threshold : {int(suggested)}%  "
                  f"(current: {DIST_HIGH_MAX}%)")
        low_rvol_missed = [d for d in stage1_details
                           if not d["s1_pass"] and d["rvol"] < RVOL_MIN_BREAKOUT]
        if low_rvol_missed:
            print(f"\n  ⚡  Trades missed due to low RVOL (<{RVOL_MIN_BREAKOUT}):")
            for d in low_rvol_missed:
                print(f"      {d['symbol']:<22} rvol {d['rvol']}x  dist_high {d['dist_high']}%")

    print("\n" + "═" * 68)
    print("  Update thresholds in config.py, then re-run to verify improvement.")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    main()
