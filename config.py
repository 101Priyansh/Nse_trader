import os

# ── API Keys (loaded from .env / environment) ──────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# GEMINI_MODEL   = "gemini-2.5-flash"
# GEMINI_MODEL   = "gemini-2.5-flash-preview-05-20"
GEMINI_MODEL   = "gemini-3-flash-preview"

TELE_TOKEN       = os.environ.get("TELE_TOKEN", "")
TELE_CHAT_ID     = os.environ.get("TELE_CHAT_ID", "")
TELE_TOKEN_SID   = os.environ.get("TELE_TOKEN_SID", "")
TELE_CHAT_ID_SID = os.environ.get("TELE_CHAT_ID_SID", "")
# ── Pipeline Thresholds ────────────────────────────────────────────────────────
RVOL_MIN_BREAKOUT  = 1.0    # minimum RVOL for near-high plays (BREAKOUT/PRIME)
RVOL_VOLUME_ANCHOR = 1.8    # RVOL threshold for pure momentum plays
DIST_HIGH_MAX      = 8.0    # max % from 52-week high to qualify
RSI_MIN            = 52     # RSI lower bound — momentum zone
RSI_MAX            = 75     # RSI upper bound — not overbought
ADX_MIN            = 20     # minimum ADX to confirm a trending move
NIFTY_DROP_GATE    = -0.5   # flag market warning if Nifty < -0.5% on the day
ACC_DAYS_MIN       = 2      # minimum accumulation days in consolidation check
CONFIRM_TOP_N      = 12     # max candidates to run Stage 3 RSI/ADX on (slow step)
USE_VISUAL_LEVELS  = False  # send chart image to Gemini Vision before text analysis (slower)

# ── Risk/reward floor for AI targets ──────────────────────────────────────────
# T1 must sit at least MIN_T1_RR × the entry→SL distance above entry.
# Backfilled trade_log outcomes showed T1 averaging 0.60R (23 of 26 AI trades
# below 1R) — unprofitable at the observed ~46% win rate. Expectancy on mature
# trades peaks around 2.0-2.5R (PF 1.30 vs 0.99 at today's ~0.5R).
MIN_T1_RR     = 2.0     # minimum T1 risk:reward multiple
ENFORCE_T1_RR = True    # downgrade ACTION to AVOID if Gemini returns T1 below the floor

# ── Ticker source ─────────────────────────────────────────────────────────────
# "nifty50" | "nifty100" | "nifty200" | "nifty500" | "static"
TICKER_SOURCE = "nifty500"

_NSE_INDEX_URLS = {
    "nifty50":  "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
}

def _fetch_nse_tickers(index: str) -> tuple[list, dict, dict]:
    """Returns (tickers, industry_groups, industry_by_symbol)."""
    import requests
    from io import StringIO
    import pandas as pd
    url = _NSE_INDEX_URLS.get(index)
    if not url:
        return [], {}, {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[~df["Symbol"].astype(str).str.startswith("DUMMY")]

    tickers = [str(s).strip() + ".NS" for s in df["Symbol"].tolist()]

    # Industry column name varies slightly across NSE CSVs
    ind_col = next((c for c in df.columns if "industry" in c.lower()), None)
    industry_groups:    dict[str, list] = {}
    industry_by_symbol: dict[str, str]  = {}
    if ind_col:
        for _, row in df.iterrows():
            sym = str(row["Symbol"]).strip() + ".NS"
            ind = str(row[ind_col]).strip()
            if sym and ind and ind.lower() not in ("nan", ""):
                industry_groups.setdefault(ind, []).append(sym)
                industry_by_symbol[sym] = ind

    return tickers, industry_groups, industry_by_symbol


def _load_tickers() -> tuple[list, dict, dict]:
    if TICKER_SOURCE == "static":
        return _TICKERS_STATIC, {}, {}
    try:
        tickers, groups, by_sym = _fetch_nse_tickers(TICKER_SOURCE)
        if tickers:
            print(f"  [config] loaded {len(tickers)} tickers, {len(groups)} industries from {TICKER_SOURCE}")
            return tickers, groups, by_sym
    except Exception as e:
        print(f"  [config] ticker fetch failed ({e}), using static list")
    return _TICKERS_STATIC, {}, {}

# ── Static watchlist (fallback) ────────────────────────────────────────────────
_TICKERS_STATIC = [
    # Large Cap / Core
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "ASIANPAINT.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "WIPRO.NS",
    "TECHM.NS", "POWERGRID.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    # Defence / Railways / PSU
    "HAL.NS", "BEL.NS", "BDL.NS", "COCHINSHIP.NS", "MAZDOCK.NS",
    "GRSE.NS", "RVNL.NS", "IRFC.NS", "IRCON.NS", "RAILTEL.NS",
    "CONCOR.NS", "BEML.NS", "MIDHANI.NS", "HINDCOPPER.NS", "NBCC.NS",
    # Energy / Power / Metals
    "ONGC.NS", "COALINDIA.NS", "NTPC.NS", "NTPCGREEN.NS",
    "PFC.NS", "RECLTD.NS", "TATAPOWER.NS", "JSWENERGY.NS",
    "ADANIPOWER.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS",
    "NATIONALUM.NS", "HINDALCO.NS", "JINDALSTEL.NS", "NHPC.NS",
    # Manufacturing / Industrial / Auto
    "BOSCHLTD.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BHARATFORG.NS",
    "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS", "THERMAX.NS",
    "POLYCAB.NS", "DIXON.NS", "HAVELLS.NS", "CGPOWER.NS",
    # Tech / AI / Engineering
    "KPITTECH.NS", "TATAELXSI.NS", "PERSISTENT.NS",
    "HAPPSTMNDS.NS", "CYIENT.NS", "MPHASIS.NS",
    "COFORGE.NS", "OFSS.NS", "ZENSARTECH.NS",
    # Midcap / Emerging / Specialty
    "MTARTECH.NS", "PAYTM.NS", "BSE.NS", "MCX.NS",
    "BHARTIHEXA.NS", "GRASIM.NS", "JKCEMENT.NS",
    "LODHA.NS", "OBEROIRLTY.NS", "PRESTIGE.NS",
    "DEEPAKNTR.NS", "PIIND.NS", "AETHER.NS",
    "CLEAN.NS", "SYNGENE.NS", "LAURUSLABS.NS",
    # Banking / Financial Services
    "CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS",
    "ICICIGI.NS", "ICICIPRULI.NS", "SBICARD.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "LICI.NS",
    # Telecom / Digital / Consumption
    "SWIGGY.NS", "DMART.NS", "TRENT.NS",
    "INDHOTEL.NS", "EASEMYTRIP.NS", "NAUKRI.NS", "DABUR.NS",
    # Additional Picks
    "ARVIND.NS", "GODREJPROP.NS",
    "APOLLOHOSP.NS", "FORTIS.NS", "MAXHEALTH.NS",
    "BIOCON.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CANBK.NS", "BANKBARODA.NS", "UNIONBANK.NS",
    "IDFCFIRSTB.NS", "FEDERALBNK.NS", "PIRAMALFIN.NS", "ALOKINDS.NS",
    # Pro-trader watchlist additions
    "HFCL.NS", "SASKEN.NS", "JINDRILL.NS", "TEXRAIL.NS",
    "SAREGAMA.NS", "NAZARA.NS", "ALKYLAMINE.NS", "BAJEL.NS",
]

TICKERS, INDUSTRY_GROUPS, INDUSTRY_BY_SYMBOL = _load_tickers()
