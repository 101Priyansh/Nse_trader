"""Stage 4 (optional, --ai flag) — Gemini analysis for entry zone, 4 targets, SL.

Targets are placed at pre-computed structural resistance levels (swing highs,
round numbers, 52w high) rather than mechanical ATR increments.
Gemini's job: judge WHICH levels matter and why — not compute them.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from google import genai

warnings.filterwarnings("ignore")

from config import ENFORCE_T1_RR, GEMINI_API_KEY, GEMINI_MODEL, MIN_T1_RR


def _num(val) -> float | None:
    """Coerce a price-like value ('₹1,234.5', 1234.5, 'N/A') to float, else None."""
    try:
        return float(str(val).replace("₹", "").replace(",", "").strip())
    except (ValueError, AttributeError, TypeError):
        return None


# ── Structural level helpers ──────────────────────────────────────────────────

def _round_numbers(price: float) -> list[float]:
    """Key round-number resistance levels above current price."""
    if price < 20:
        steps = [0.5, 1]
    elif price < 50:
        steps = [1, 2]
    elif price < 150:
        steps = [5, 10]
    elif price < 500:
        steps = [10, 25]
    elif price < 1500:
        steps = [25, 50]
    elif price < 5000:
        steps = [50, 100]
    elif price < 15000:
        steps = [100, 250]
    else:
        steps = [500, 1000]

    levels: set[float] = set()
    for step in steps:
        first = (int(price / step) + 1) * step
        for i in range(5):
            levels.add(round(first + i * step, 2))

    return sorted(l for l in levels if l > price * 1.002)[:6]


def get_structural_levels(symbol: str, price: float, context: dict) -> dict:
    """
    Pre-compute structural resistance levels from 90-day daily OHLC.
    Returns swing highs, round numbers, 52w high, consolidation range, prev day OHLC.
    """
    out = {
        "swing_highs":   [],
        "round_numbers": _round_numbers(price),
        "yearly_high":   None,
        "base_high":     None,
        "base_low":      None,
        "prev_day_high": None,
        "prev_day_low":  None,
        "atr_price":     None,
    }

    # 52w high derived from dist_high (scanner computes yearly_high but doesn't pass it through)
    dist_high = context.get("dist_high") or 0
    if dist_high and price:
        out["yearly_high"] = round(price / (1 - dist_high / 100), 2)

    # ATR in ₹ (confirm.py stores ATRr = ATR/close ratio)
    atr_ratio = context.get("atr") or 0
    if atr_ratio and price:
        out["atr_price"] = round(atr_ratio * price, 2)

    # Consolidation base in ₹
    consol_pct = context.get("consol_pct") or 0
    if consol_pct and price:
        out["base_high"] = round(price, 2)
        out["base_low"]  = round(price * (1 - consol_pct / 100), 2)

    try:
        df = yf.download(symbol, period="90d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return out
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Previous session OHLC (primary BTST reference)
        if len(df) >= 2:
            out["prev_day_high"] = round(float(df["High"].iloc[-2]), 2)
            out["prev_day_low"]  = round(float(df["Low"].iloc[-2]),  2)

        # Swing highs: local maxima with 3-candle window on each side
        highs = df["High"].values.flatten()
        window = 3
        raw: list[float] = []
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i - window: i + window + 1]) and highs[i] > price * 1.003:
                raw.append(round(float(highs[i]), 2))

        # Merge levels within 1% of each other (keep the higher one in each cluster)
        merged: list[float] = []
        for lvl in sorted(set(raw)):
            if not merged or (lvl - merged[-1]) / merged[-1] > 0.01:
                merged.append(lvl)

        out["swing_highs"] = [l for l in merged if l > price][:5]

    except Exception:
        pass

    return out


# ── Prompt builder ────────────────────────────────────────────────────────────
# Two prompts available — switch by commenting/uncommenting the return at the end.
#
#   PROMPT A (active)      — open judgment. gives Gemini the data, asks for its view.
#   PROMPT B (commented)   — prescriptive rules. tells Gemini exactly how to score/pick.

def _min_t1(context: dict) -> tuple[float | None, float | None]:
    """Returns (risk_per_share, minimum acceptable T1) — (None, None) if not computable."""
    price = _num(context.get("price"))
    sl    = _num(context.get("sl"))
    if not price or not sl or price <= sl:
        return None, None
    risk = price - sl
    return risk, round(price + MIN_T1_RR * risk, 2)


def _build_prompt(symbol: str, context: dict, levels: dict) -> str:
    price = context.get("price", "N/A")
    sl    = context.get("sl",    "N/A")

    # ── Risk/reward floor ─────────────────────────────────────────────────────
    # Hand Gemini the computed minimum rather than asking it to derive one —
    # the backtest showed it consistently anchors T1 to the nearest level
    # regardless of how far the SL sits.
    risk, min_t1 = _min_t1(context)
    if min_t1:
        rr_rule = f"""
Risk/reward floor (mandatory):
  Risk per share (entry → SL) : ₹{round(risk, 2)}  ({risk / _num(price) * 100:.2f}%)
  T1 must be at or above      : ₹{min_t1}  (= {MIN_T1_RR:g}× risk)
  Choose the NEAREST level from the lists above that sits at or above ₹{min_t1}.
  T2, T3, T4 must each be progressively higher than T1.
  A target closer than ₹{min_t1} is not acceptable no matter how strong the level —
  if nothing listed reaches it, this setup has too little room: set ACTION: AVOID
  and still report the nearest levels you would otherwise have used.
"""
    else:
        rr_rule = ""

    swing_str  = ", ".join(f"₹{l}" for l in levels["swing_highs"]) or "none found in range"
    round_str  = ", ".join(f"₹{l}" for l in levels["round_numbers"])
    yearly     = f"₹{levels['yearly_high']}"   if levels["yearly_high"]   else "N/A"
    prev_high  = f"₹{levels['prev_day_high']}" if levels["prev_day_high"] else "N/A"
    base_range = (f"₹{levels['base_low']} – ₹{levels['base_high']}"
                  if levels["base_low"] else "N/A")
    atr_str    = f"₹{levels['atr_price']}" if levels["atr_price"] else "N/A"

    # ── PROMPT A: Open judgment (active) ──────────────────────────────────────
    # Gives Gemini the structural data and asks for its own assessment.
    # Conviction, target selection, and action are Gemini's judgment calls.
    prompt_a = f"""You are a professional trader analyzing {symbol} for a short-term entry.

Structural resistance levels above ₹{price} (pre-computed from 90-day OHLC):
  Swing highs   : {swing_str}
  Round numbers : {round_str}
  52-week high  : {yearly}
  Prev day high : {prev_high}
  Base range    : {base_range}  ({context.get('acc_days', 'N/A')} acc days, {context.get('consol_pct', 'N/A')}% range)

Technical context:
  Setup  : {context.get('setup')} | Score: {context.get('score')} | RVOL: {context.get('rvol')}x
  RSI    : {context.get('rsi', 'N/A')} | ADX: {context.get('adx', 'N/A')} | MACD: {context.get('macd', 'N/A')}
  ATR    : {atr_str} | Sector: {context.get('sector', 'N/A')}
  SL     : ₹{sl} ({context.get('sl_source', 'N/A')}) | Above 200 EMA: {context.get('above_200', 'N/A')}
{rr_rule}
Step 1 — From the structural levels and technical context above, identify which
resistance zones are most likely to act as targets. Use your judgment on what
makes a level strong or weak. Note any levels you are skipping and why.
Respect the risk/reward floor above — it overrides level-proximity preference.

Step 2 — Assess the full picture: indicator alignment, structure quality, and
momentum. Decide whether this setup is worth entering and with what urgency.
Use AVOID for setups where the risk clearly outweighs the opportunity.

Step 3 — Output. Strictly one value per line, no preamble, no markdown:
ANALYSIS: [one line — which levels you chose and your key reason]
CONVICTION: [1-10]
ACTION: [STRONG BUY / BUY / HOLD / AVOID]
ENTRY: [low–high zone based on base structure and nearby support — not just current price]
T1: [level]
T2: [level]
T3: [level]
T4: [level]
SL: ₹{sl}
NOTE: [one sentence — why this trade works right now]
"""

    # ── PROMPT B: Prescriptive rules (comment out return_a, uncomment return_b) ─
    # Tells Gemini exactly how to pick levels and score conviction.
    # More deterministic output but less genuine LLM judgment.
    prompt_b = f"""You are a professional swing/BTST trader analyzing {symbol}.

Structural resistance levels above ₹{price} (pre-computed from 90-day OHLC):
  Swing highs   : {swing_str}
  Round numbers : {round_str}
  52-week high  : {yearly}
  Prev day high : {prev_high}
  Base range    : {base_range}  ({context.get('acc_days', 'N/A')} acc days, {context.get('consol_pct', 'N/A')}% range)

Technical context:
  Setup  : {context.get('setup')} | Score: {context.get('score')} | RVOL: {context.get('rvol')}x
  RSI    : {context.get('rsi', 'N/A')} | ADX: {context.get('adx', 'N/A')} | MACD: {context.get('macd', 'N/A')}
  ATR    : {atr_str} | Sector: {context.get('sector', 'N/A')}
  SL     : ₹{sl} ({context.get('sl_source', 'N/A')}) | Above 200 EMA: {context.get('above_200', 'N/A')}
{rr_rule}
This is a swing setup — hold until T1 or SL resolves, not a fixed 1–3 session exit.
T1 is set by the risk/reward floor above, not by what is reachable tomorrow.

Step 1 — Analyze the resistance structure.
From the levels listed above, identify which are most meaningful as targets.
  - T1 must satisfy the risk/reward floor — this overrides every rule below
  - Prefer swing highs (tested before) over untested round numbers
  - If two levels are within 1% of each other, pick the stronger — skip the other
  - Set ACTION to AVOID if: RSI > 74, no listed level reaches the T1 floor,
    or RVOL < 1.0 with negative sector momentum

Step 2 — Score conviction.
  High (8-10) : clean swing highs, RVOL > 2×, positive sector, ADX > 25
  Med  (5-7)  : structure present but mixed signals
  Low  (1-4)  : weak setup, no clear levels, conflicting indicators

Step 3 — Output. Strictly one value per line, no preamble, no markdown:
ANALYSIS: [one line — which specific levels you chose for T1-T4 and the primary reason]
CONVICTION: [1-10]
ACTION: [STRONG BUY / BUY / HOLD / AVOID]
ENTRY: [tight zone ±0.3% around ₹{price}]
T1: [nearest listed resistance that meets the risk/reward floor]
T2: [next resistance]
T3: [next resistance]
T4: [furthest target, only if structural level exists to support it]
SL: ₹{sl}
NOTE: [one sentence — specific momentum or structure reason this trade works NOW]
"""

    return prompt_a       # ← PROMPT A active
    # return prompt_b     # ← uncomment to switch to prescriptive prompt


# ── Public API ────────────────────────────────────────────────────────────────

def get_levels(symbol: str, context: dict) -> dict | None:
    """Call Gemini with structural context. Returns parsed dict or None."""
    try:
        price  = context.get("price", 0)
        levels = get_structural_levels(symbol, price, context)
        prompt = _build_prompt(symbol, context, levels)

        client = genai.Client(api_key=GEMINI_API_KEY)
        raw    = client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text

        result = {}
        for line in raw.strip().split("\n"):
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip().upper()] = v.strip()

        if "T1" not in result:
            return None

        # ── Enforce the R:R floor ────────────────────────────────────────────
        # The prompt asks for it, but a prompt is not a guarantee — verify.
        risk, min_t1 = _min_t1(context)
        t1 = _num(result.get("T1"))
        if risk and t1:
            result["T1_RR"] = round((t1 - _num(price)) / risk, 2)
            if t1 < min_t1:
                print(f"  [levels] {symbol}: T1 ₹{t1} is {result['T1_RR']}R "
                      f"— below the {MIN_T1_RR:g}R floor (₹{min_t1})"
                      + ("  → ACTION forced to AVOID" if ENFORCE_T1_RR else ""))
                if ENFORCE_T1_RR:
                    result["ACTION"] = "AVOID"
                    result["NOTE"] = (f"Auto-rejected: T1 offers only {result['T1_RR']}R "
                                      f"against a ₹{round(risk, 2)} stop. " + result.get("NOTE", ""))

        return result

    except Exception:
        return None


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] or ["HINDALCO.NS"]
    for sym in symbols:
        print(f"\n{'━'*42}")
        print(f"  {sym}")
        print(f"{'━'*42}")
        ctx = {"price": "N/A", "setup": "PRIME", "score": "N/A", "rvol": "N/A",
               "dist_high": "N/A", "rsi": "N/A", "adx": "N/A", "macd": "N/A"}
        r = get_levels(sym, ctx)
        if r:
            for k, v in r.items():
                print(f"  {k:<12}: {v}")
        else:
            print("  No levels generated.")
