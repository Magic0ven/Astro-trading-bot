"""
Central configuration for the Astro-Bot.
All tunable parameters live here. Loaded from environment where applicable.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Swiss Ephemeris ────────────────────────────────────────────────────────────
EPHE_PATH = os.getenv("EPHE_PATH", "./ephe")

# ── Planets tracked ───────────────────────────────────────────────────────────
import swisseph as swe

PLANETS_WESTERN = {
    "Sun":     swe.SUN,
    "Moon":    swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus":   swe.VENUS,
    "Mars":    swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn":  swe.SATURN,
    "Uranus":  swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto":   swe.PLUTO,
}

PLANETS_VEDIC = {
    **PLANETS_WESTERN,
    "Rahu": swe.MEAN_NODE,     # North Node (Rahu)
    # Ketu is always Rahu + 180°, computed in code
}

# ── Ayanamsa for Vedic (Sidereal) ─────────────────────────────────────────────
# Lahiri is the standard used in India by the Indian government ephemeris
VEDIC_AYANAMSA = swe.SIDM_LAHIRI  # Options: SIDM_RAMAN, SIDM_KRISHNAMURTI

# ── Aspect weights (optional override) ────────────────────────────────────────
# By default the bot uses raw cos(angle). Override here to use classical weights.
# Set USE_COSINE_RAW = True to use the mathematical formula (recommended).
# Set USE_COSINE_RAW = False to use the table below.
USE_COSINE_RAW = True

ASPECT_WEIGHTS = {
    # (min_angle, max_angle): weight
    (0,   8):   1.0,    # Conjunction  — unity
    (52,  68):  0.5,    # Sextile      — harmony
    (82,  98):  0.0,    # Square       — tension
    (112, 128): 0.8,    # Trine        — flow (positive override vs raw cos)
    (172, 188): -1.0,   # Opposition   — maximum tension
}

# ── Vedic Nakshatra Data ───────────────────────────────────────────────────────
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishtha",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

# ── Nakshatra Classifications ─────────────────────────────────────────────────
# NAKSHATRA TRADE FILTER: DISABLED (as of 2026-02-23)
#
# Full matrix backtest (2022–2025, realistic 4h simulation with intrabar SL/TP,
# fees, slippage and real funding rates) showed the Nakshatra block list
# consistently HURTS performance when using the EMA two-way filter:
#
#   Filter config            Avg WR   4yr CAGR   Worst DD
#   EMA 2-way only           41.1%    38.2%       27.4%   ← BEST
#   NK + EMA 2-way           40.6%    24.4%       22.2%
#   NK + EMA 1-way (prev)    40.2%    22.2%       22.8%
#   Nakshatra only           35.6%    -6.0%       44.0%
#
# The Nakshatra block adds at most 0.4pp of win rate but costs ~14pp of CAGR
# by skipping EMA-confirmed signals during blocked Nakshatras. At 4h cadence,
# the EMA filter is sufficient to remove bad trades — Nakshatra adds noise.
#
# Multipliers (1.2x / 0.8x) are still applied to score calculations internally
# for signal strength weighting — only the hard trade BLOCK is removed.
#
# To re-enable blocking:  set NAKSHATRA_FILTER = True in .env
# To re-run the full matrix analysis:
#   python scripts/final_backtest.py --compare

NAKSHATRA_FILTER = os.getenv("NAKSHATRA_FILTER", "false").lower() == "true"

TRADE_FAVORABLE_NAKSHATRAS = {
    "Rohini", "Hasta", "Punarvasu", "Shatabhisha", "Anuradha",
}

TRADE_UNFAVORABLE_NAKSHATRAS = {
    # Kept for reference and multiplier weighting — not used to BLOCK trades
    # unless NAKSHATRA_FILTER=true in .env
    "Uttara Phalguni", "Ardra", "Bharani", "Mrigashira", "Mula",
    "Purva Ashadha", "Purva Phalguni", "Revati", "Shravana", "Swati",
}

def get_nakshatra(moon_longitude_sidereal: float) -> str:
    """Return Nakshatra name for a given sidereal Moon longitude."""
    idx = int(moon_longitude_sidereal / (360 / 27)) % 27
    return NAKSHATRAS[idx]

def nakshatra_multiplier(moon_longitude_sidereal: float) -> float:
    name = get_nakshatra(moon_longitude_sidereal)
    if name in TRADE_FAVORABLE_NAKSHATRAS:
        return 1.2
    elif name in TRADE_UNFAVORABLE_NAKSHATRAS:
        return 0.8
    return 1.0

# ── EMA Trend Filter ──────────────────────────────────────────────────────────
# Controls how the EMA(200) macro trend filter is applied to signals.
#
# Full matrix backtest (2022–2025, realistic 4h):
#   "none"    → -7.6% CAGR  (no filter — loses money, DO NOT USE)
#   "one_way" → 33.8% CAGR  (skip SELL when price > EMA200)
#   "two_way" → 38.2% CAGR  (skip SELL above EMA200 AND skip BUY below EMA200)
#
# "two_way" is the recommended setting: it aligns every trade with the macro
# trend, adding ~4pp of CAGR over one_way with the same drawdown profile.
EMA_FILTER   = os.getenv("EMA_FILTER", "two_way")   # none | one_way | two_way
EMA_PERIOD   = int(os.getenv("EMA_PERIOD", "20"))

# ── Signal Thresholds ─────────────────────────────────────────────────────────
SLOPE_THRESHOLD = float(os.getenv("SLOPE_THRESHOLD", "0.5"))
SCORE_HISTORY_WINDOW = int(os.getenv("SCORE_HISTORY_WINDOW", "5"))

# ── Bot Scheduling ────────────────────────────────────────────────────────────
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "60"))
MAX_WEEKLY_TRADES = int(os.getenv("MAX_WEEKLY_TRADES", "3"))

# ── Risk Parameters ───────────────────────────────────────────────────────────
#
# CAPITAL_PCT: fraction of the live exchange balance used for position sizing.
# The bot fetches your real Hyperliquid/Binance balance every cycle and computes:
#
#   effective_capital = account_balance * CAPITAL_PCT
#   risk_per_trade    = effective_capital * RISK_PER_TRADE_PCT
#
# This means position sizes automatically scale up as the account grows
# (true compounding) and scale down after losses (natural drawdown protection).
#
# CAPITAL_PCT=0.60 keeps 40% of your balance as a free liquidation buffer.
# Worst historical drawdown was 25.9% of effective capital, so 40% buffer
# means worst-case loss = 15.5% of total balance — well clear of liquidation.
#
# CAPITAL_USDT is a static fallback used only if balance fetch fails.
# Set it to CAPITAL_PCT × your expected starting balance as a safety net.
#
CAPITAL_PCT           = float(os.getenv("CAPITAL_PCT", "0.60"))
CAPITAL_USDT          = float(os.getenv("CAPITAL_USDT", "120"))   # fallback only
RISK_PER_TRADE_PCT    = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))
MAX_DRAWDOWN_HALT_PCT = float(os.getenv("MAX_DRAWDOWN_HALT_PCT", "0.10"))
# Maximum number of 4h bars a trade may stay open before force-close.
# 12 bars × 4h = 48h — mirrors the backtest MAX_OPEN_BARS=12 setting.
# If neither SL nor TP fires after 48h the bot cancels the orders and
# closes the position at market to prevent stale exposure.
MAX_OPEN_BARS         = int(os.getenv("MAX_OPEN_BARS", "12"))
RR_RATIO              = float(os.getenv("RR_RATIO", "2.0"))
ATR_MULTIPLIER        = float(os.getenv("ATR_MULTIPLIER", "1.5"))
ATR_PERIOD            = 14

# ── Leverage ──────────────────────────────────────────────────────────────────
# How leverage interacts with this system:
#
#   Dollar risk per trade is FIXED at RISK_PER_TRADE_PCT regardless of leverage.
#   Leverage only reduces the margin (collateral) required to hold the position.
#
#   Notional position size = (risk_amount / sl_distance) * price   [unchanged]
#   Margin required        = notional / leverage                    [reduced]
#   Liquidation distance   = 1 / leverage                          [tighter with higher lev]
#
# SAFETY RULE: liquidation must be FURTHER than stop-loss.
#   With ATR_MULT=1.5, typical SL is ~2-4% from entry.
#   Max safe leverage = 1 / (sl_pct * 2)  →  for 2.5% SL: max = 20x
#   RECOMMENDED for this cosmic system: 3x–5x (ample buffer, no liquidation risk)
#
# WARNING: Setting leverage > 10x with volatile assets like BTC means
#          a 5-10% adverse move can liquidate BEFORE your stop fires.
#
LEVERAGE = int(os.getenv("LEVERAGE", "3"))          # 1 = no leverage (spot-equivalent)

# ── Exchange — Hyperliquid only ───────────────────────────────────────────────
# This bot is built exclusively for Hyperliquid (DEX perpetuals).
# Authentication: EVM wallet address + private key (set in .env).
# Use a dedicated agent sub-wallet — never your main wallet.
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
ACTIVE_ASSET = os.getenv("ACTIVE_ASSET", "BTC")

# ── Pythagorean Chart ─────────────────────────────────────────────────────────
PYTHAGOREAN_MAP = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
    'j': 1, 'k': 2, 'l': 3, 'm': 4, 'n': 5, 'o': 6, 'p': 7, 'q': 8, 'r': 9,
    's': 1, 't': 2, 'u': 3, 'v': 4, 'w': 5, 'x': 6, 'y': 7, 'z': 8,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "logs/signals.db")
