# Slope Around Medium — Astro-Bot: Build Plan & Architecture

*Last updated: Feb 2026*

---

## Table of Contents

1. [Project Philosophy](#1-project-philosophy)
2. [System Architecture](#2-system-architecture)
3. [Libraries & Dependencies](#3-libraries--dependencies)
4. [Data Sources](#4-data-sources)
5. [Module Breakdown](#5-module-breakdown)
6. [Mathematical Engine — Full Formulas](#6-mathematical-engine--full-formulas)
7. [Dual-System Gate](#7-dual-system-gate)
8. [Signal Logic Flow](#8-signal-logic-flow)
9. [Trade Execution Layer](#9-trade-execution-layer)
10. [Risk Management](#10-risk-management)
11. [Asset Configuration (DNA)](#11-asset-configuration-dna)
12. [Backtesting Architecture](#12-backtesting-architecture)
13. [Validated Optimal Settings](#13-validated-optimal-settings)
14. [Deployment Plan](#14-deployment-plan)
15. [Full File Structure](#15-full-file-structure)

---

## 1. Project Philosophy

**Core Principle: Aggregate Resonance**

This bot does not trade on individual planetary predictions. It trades on the *velocity* (slope) of a composite "Cosmic Energy Waveform" constructed from:

- Every active planet's current longitude vs the asset's natal (birth) chart longitude
- Both **Western (Tropical)** and **Vedic (Sidereal)** zodiac systems independently scored and compared
- Pythagorean numerology of the asset's genesis date vs today's Universal Day Number
- The *slope* of the composite score — not its absolute value — is the trade trigger

**Key Rule: A trade fires only when BOTH systems agree. A direct contradiction (one BUY, one SELL) = mandatory NO_TRADE.**

**Backtest Integrity Rule: Only STRONG signals are executed.** WEAK signals (one system BUY + one HOLD) are logged but never traded — this keeps live behaviour exactly consistent with the backtested strategy, which only tested STRONG signals.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        ASTRO-BOT CORE                          │
├──────────────────┬───────────────────┬─────────────────────────┤
│   DATA LAYER     │  COMPUTATION LAYER│   DECISION LAYER        │
│                  │                   │                         │
│ ┌──────────────┐ │ ┌───────────────┐ │ ┌─────────────────────┐ │
│ │ Swiss Ephem  │ │ │ Astro Engine  │ │ │  Signal Engine      │ │
│ │ (pyswisseph) │→│ │ Western Calc  │ │ │  Slope + Medium     │ │
│ └──────────────┘ │ │ Vedic Calc    │ │ │  Dual-System Gate   │ │
│                  │ │ Aspect Matrix │ │ │  EMA Filter         │ │
│ ┌──────────────┐ │ └───────────────┘ │ └─────────┬───────────┘ │
│ │  Market API  │ │                   │           │             │
│ │  (ccxt)      │ │ ┌───────────────┐ │ ┌─────────▼───────────┐ │
│ └──────────────┘ │ │ Numerology    │ │ │  Trade Executor     │ │
│                  │ │ Engine        │ │ │  48h Timeout        │ │
│ ┌──────────────┐ │ └───────────────┘ │ │  Drawdown Halt      │ │
│ │  Asset DNA   │ │                   │ └─────────────────────┘ │
│ │  Config      │ │ ┌───────────────┐ │                         │
│ └──────────────┘ │ │ Score History │ │ ┌─────────────────────┐ │
│                  │ │ (Ring Buffer) │ │ │  SQLite + JSON Logs │ │
│                  │ └───────────────┘ │ └─────────────────────┘ │
└──────────────────┴───────────────────┴─────────────────────────┘
```

### Timing Model

- Bot runs on a **4-hour scheduled timer** (APScheduler)
- `CHECK_INTERVAL_MINUTES=240` is the backtested optimal cadence
- Planetary positions are re-calculated fresh every cycle
- Score history window: **last 5 readings** (for Medium + Slope)
- Trade frequency: **1–3 STRONG signals per week** (driven by celestial agreement)

---

## 3. Libraries & Dependencies

### Astronomical

| Library | Purpose |
|---|---|
| `pyswisseph` | Swiss Ephemeris — planetary longitudes, retrograde detection, Lahiri Ayanamsa |

**Why pyswisseph?**
- The same engine used by Astrodienst (astro.com) professional software
- Supports Tropical and Sidereal in one call via `swe.FLG_SIDEREAL`
- Offline and free — no API rate limits
- Covers 3000 BCE to 3000 CE — ideal for natal chart lookups

### Market Data

| Library | Purpose |
|---|---|
| `ccxt` | Unified exchange API (Binance, Bybit, Hyperliquid, OKX, …) |
| `pandas` | OHLCV manipulation, EMA |
| `numpy` | Cosine, linear regression slope |

### Infrastructure

| Library | Purpose |
|---|---|
| `apscheduler` | 4-hour interval scheduling |
| `python-dotenv` | Load API keys + config from `.env` |
| `loguru` | Structured logging to file + terminal |
| `rich` | Terminal dashboard tables and panels |
| `python-dateutil` | Genesis datetime parsing |

---

## 4. Data Sources

### 4.1 Planetary Positions — Swiss Ephemeris (local, offline)

```
Ephemeris data files: sepl_18.se1, semo_18.se1, seas_18.se1
Location: ./ephe/
Download: https://www.astro.com/swisseph/ephe/
```

Per planet per moment:
- **Ecliptic Longitude** (0–360°) — primary value used
- **Speed** — sign determines retrograde (`speed < 0` → retrograde)

Planets tracked — Western: Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto (10)

Planets tracked — Vedic: above 10 + Rahu (Mean Node) + Ketu (Rahu + 180°) = 12

### 4.2 Natal Chart — Asset DNA (static, computed once)

Stored in `assets_dna.json`. Computed from each asset's genesis datetime using `scripts/calculate_natal.py`. Never needs to change unless the genesis date is revised.

BTC genesis: `2009-01-03T18:15:05Z` (genesis block timestamp)

### 4.3 Market Price Data — ccxt (live)

- `fetch_ohlcv(symbol, "4h", limit=N)` — used for ATR and EMA calculation
- `fetch_ticker(symbol)` — current price for signal display
- `fetch_balance()` — live account balance for dynamic capital sizing

### 4.4 Sidereal Ayanamsa (Vedic Correction)

```
Vedic Longitude = Western Longitude − Ayanamsa
Current Ayanamsa (2026): ≈ 24°07'  (Lahiri)
```

`pyswisseph` handles this automatically:

```python
swe.set_sid_mode(swe.SIDM_LAHIRI)
positions = swe.calc_ut(jd, planet_id, swe.FLG_SPEED | swe.FLG_SIDEREAL)
```

---

## 5. Module Breakdown

```
astro-bot/
├── main.py                      Entry point — loads DNA, runs scheduler
├── config.py                    All parameters (reads .env)
├── assets_dna.json              Natal charts + numerology per asset
├── .env                         API keys + overrides (never commit)
├── .env.example                 Committed safe template
├── requirements.txt
│
├── core/
│   ├── astro_engine.py          Planet positions (Western + Vedic), sky state
│   ├── numerology.py            Life Path, UDN, Ticker Vibration, multiplier
│   ├── signal_engine.py         Composite score, slope, gate, EMA/NK filters
│   └── score_history.py         Ring buffer — Medium + Slope over window
│
├── exchange/
│   ├── market_data.py           OHLCV, ATR(14), EMA(20), account balance
│   └── trade_executor.py        Order placement, 48h timeout, drawdown halt
│
├── scripts/
│   ├── backtest.py              Fetch OHLCV + compute cosmic signals → CSV
│   ├── calculate_natal.py       Compute natal chart from genesis datetime
│   ├── fetch_funding_rates.py   Pull historical funding rates from Binance
│   └── final_backtest.py        Realistic multi-year simulation + comparison suite
│
├── dashboard/
│   └── app.py                   Streamlit live signal dashboard (optional)
│
├── logs/
│   ├── signals.db               SQLite — all signals and trades
│   ├── open_positions.json      Active trade timeout tracker
│   └── equity_state.json        Peak equity record for drawdown halt
│
└── ephe/                        Swiss Ephemeris binary data files
```

---

## 6. Mathematical Engine — Full Formulas

### 6.1 Numerology

```python
def digital_root(n: int) -> int:
    n = abs(n)
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n

# Life Path Number — from genesis date
# BTC: 2009-01-03 → 2+0+0+9+0+1+0+3 = 15 → 6
life_path = digital_root(sum(int(d) for d in "20090103"))

# Ticker Vibration — Pythagorean chart
# BTC: B=2 T=2 C=3 → 7
PYTHAGOREAN = {a:1,b:2,c:3,d:4,e:5,f:6,g:7,h:8,i:9,
               j:1,k:2,l:3,m:4,n:5,o:6,p:7,q:8,r:9,
               s:1,t:2,u:3,v:4,w:5,x:6,y:7,z:8}
ticker_vib = digital_root(sum(PYTHAGOREAN[c] for c in "btc"))

# Universal Day Number — from today
# 2026-02-23 → 2+0+2+6+0+2+2+3 = 17 → 8
udn = digital_root(sum(int(d) for d in "20260223"))

# Numerology Multiplier (Pc)
Pc = 1.5  if udn == life_path  else  1.0
```

### 6.2 Aspect Score (Composite Score)

```
For each live planet i  ×  each natal planet j:

  θ_ij      = |lon_live_i  −  lon_natal_j|  mod 360°
  retro_i   = +1 if speed_i > 0  (direct)
              −1 if speed_i < 0  (retrograde)
  force_ij  = retro_i × cos(radians(θ_ij))

Cosine values at key aspects:
  0°   → +1.00  (Conjunction    — maximum constructive resonance)
  60°  → +0.50  (Sextile        — harmonious)
  90°  →  0.00  (Square         — tension, net neutral)
  120° → −0.50  (Trine          — eased tension, measured as reduced force)
  180° → −1.00  (Opposition     — maximum destructive resonance)

Note on Trine: classically "beneficial" but cos(120°) = −0.50. The formula
measures geometric closeness/alignment, not traditional benefic/malefic.

Western Composite Score (10 × 10 planet pairs):
  W(t)     = Σ_ij  force_ij

Vedic Composite Score (12 × 12 pairs, incl. Rahu/Ketu):
  V(t)     = Σ_ij  force_ij  × dasha_weight_i  × nakshatra_mult

  nakshatra_mult:
    1.2   Moon in favorable Nakshatra (Rohini, Hasta, Punarvasu, Shatabhisha, Anuradha)
    0.8   Moon in unfavorable Nakshatra (Ardra, Bharani, Mula, etc.)
    1.0   otherwise

Adjusted Scores (Numerology applied):
  W_adj(t) = W(t) × Pc
  V_adj(t) = V(t) × Pc
```

### 6.3 Medium and Slope

```
Window N = SCORE_HISTORY_WINDOW = 5

Medium  = mean( W_adj(t−4), W_adj(t−3), W_adj(t−2), W_adj(t−1), W_adj(t) )
Slope   = numpy.polyfit([0,1,2,3,4], history, deg=1)[0]
          (linear regression slope — more stable than simple t − t−1 delta)
```

### 6.4 Single-System Signal

```
BUY   ←   W_adj(t) > Medium   AND   Slope >  SLOPE_THRESHOLD   (default 0.5)
SELL  ←   W_adj(t) < Medium   OR    Slope < −SLOPE_THRESHOLD
HOLD  ←   neither condition met
```

### 6.5 Stop-Loss and Take-Profit

```
ATR(14) = rolling 14-bar average of True Range on 4h OHLC candles
         True Range = max(high−low, |high−prev_close|, |low−prev_close|)

LONG (BUY):
  stop_loss   = entry  − ATR × ATR_MULTIPLIER       (default 1.5)
  take_profit = entry  + |entry − stop_loss| × RR_RATIO   (default 2.0)

SHORT (SELL):
  stop_loss   = entry  + ATR × ATR_MULTIPLIER
  take_profit = entry  − |entry − stop_loss| × RR_RATIO

Example (BTC $90,000, ATR = $1,800):
  SL  = $90,000 − $2,700 = $87,300   (−3.0%)
  TP  = $90,000 + $5,400 = $95,400   (+6.0%)
  R:R = 2:1  ✓
```

### 6.6 Position Sizing

```
effective_capital  = account_balance × CAPITAL_PCT         (default 0.60)
risk_amount        = effective_capital × RISK_PER_TRADE_PCT (default 0.01)
sl_distance        = |entry_price − stop_loss_price|

base_coins         = risk_amount / sl_distance
notional_usdt      = base_coins × entry_price

             risk_amount × entry_price
           = ──────────────────────────
                    sl_distance

size_factor:
  STRONG_BUY / STRONG_SELL → 1.0   (full size, Pc applied)
  WEAK signals              → 0.5   (not executed; STRONG-only policy)

final_notional = notional_usdt × size_factor × Pc

Example ($200 account):
  effective_capital = $200 × 0.60 = $120
  risk_amount       = $120 × 0.01 = $1.20
  sl_distance       = $87,300 − $90,000 → $2,700 (abs)
  base_coins        = $1.20 / $2,700 = 0.000444 BTC
  notional_usdt     = 0.000444 × $90,000 = $40.00
  margin_required   = $40.00 / 3 = $13.33   (at 3× leverage)
```

### 6.7 EMA Trend Filter

```
EMA(N) on 4h closes (pandas ewm with span=N, adjust=False)
N = EMA_PERIOD = 20   →   EMA(20)/4h ≈ 3.3-day momentum filter

two_way rules:
  price < EMA(20)  →  block BUY  (don't buy into a downtrend)
  price > EMA(20)  →  block SELL (don't short an uptrend)

Applied inside signal_engine.generate_signal() after the dual-system gate.
If blocked: final_action = "NO_TRADE", filter_reason logged.
```

### 6.8 Drawdown Halt

```
peak_equity tracked in logs/equity_state.json (persists across restarts)

drawdown = (peak_equity − current_equity) / peak_equity

if drawdown ≥ MAX_DRAWDOWN_HALT_PCT (0.10):
    block all trades for the cycle
    log error with drawdown%
    require manual reset of equity_state.json to resume
```

### 6.9 48-Hour Trade Timeout

```
max_hold_minutes = MAX_OPEN_BARS × CHECK_INTERVAL_MINUTES
                 = 12 × 240 = 2880 min = 48h

Each cycle: check logs/open_positions.json
  age = now − opened_at
  if age ≥ 2880 min:
    paper mode → log TIMEOUT to signals.db, remove from tracker
    live mode  → cancel SL/TP orders, send reduceOnly market close
```

---

## 7. Dual-System Gate

```
Western Signal  +  Vedic Signal  →  Final Action
──────────────────────────────────────────────────
BUY   + BUY    →  STRONG_BUY    ← executed (full size)
SELL  + SELL   →  STRONG_SELL   ← executed (full size)
BUY   + HOLD   →  WEAK_BUY      ← logged, NOT executed
SELL  + HOLD   →  WEAK_SELL     ← logged, NOT executed
HOLD  + HOLD   →  HOLD          ← no trade
BUY   + SELL   →  NO_TRADE      ← mandatory sit-out (contradiction)
```

**Why STRONG-only?** The 4-year realistic backtest (2022–2025) used STRONG signals exclusively. Trading WEAK signals in live mode would introduce behaviour not captured in the backtested P&L figures, breaking the live/backtest consistency assumption.

---

## 8. Signal Logic Flow

```
Every 4 hours:
│
├─ 0. STARTUP CHECKS
│   ├─ check_and_close_stale_positions()   [48h timeout]
│   └─ fetch price, ATR(14), EMA(20) on 4h candles
│
├─ 1. FETCH LIVE BALANCE
│   ├─ account_balance = exchange.fetch_balance()
│   ├─ effective_capital = account_balance × CAPITAL_PCT
│   └─ update_peak_equity(account_balance)   [drawdown tracking]
│
├─ 2. ASTROLOGICAL COMPUTATION
│   ├─ Sky state: Western longitudes, Vedic longitudes, Rahu/Ketu, retrograde flags
│   └─ Nakshatra of Moon (multiplier for Vedic score weighting)
│
├─ 3. NUMEROLOGY
│   ├─ Universal Day Number (UDN)
│   └─ Numerology Multiplier Pc (1.5 or 1.0)
│
├─ 4. COMPOSITE SCORES
│   ├─ Western Score  = Σ cos(live − natal) × retrograde
│   ├─ Vedic Score    = Σ cos(live − natal) × retrograde × dasha × nk_mult
│   └─ Adjusted Score = Raw Score × Pc
│
├─ 5. UPDATE SCORE HISTORY (window = 5)
│   ├─ Medium = mean(last 5)
│   └─ Slope  = linear regression slope of last 5
│
├─ 6. SINGLE-SYSTEM SIGNALS
│   ├─ Western: BUY / SELL / HOLD
│   └─ Vedic:   BUY / SELL / HOLD
│
├─ 7. DUAL-SYSTEM GATE
│   └─ STRONG_BUY / STRONG_SELL / WEAK_BUY / WEAK_SELL / HOLD / NO_TRADE
│
├─ 8. RELIABILITY FILTERS
│   ├─ Nakshatra block (disabled by default)
│   └─ EMA two_way filter  →  may convert to NO_TRADE
│
├─ 9. CALCULATE TRADE PARAMETERS
│   ├─ stop_loss   = entry ± ATR × 1.5
│   ├─ take_profit = entry ± SL_distance × 2.0
│   └─ position_size = (risk_amount / sl_distance) × entry × Pc
│
└─ 10. DISPATCH
    ├─ WEAK signal   → log only, skip execution
    ├─ Drawdown halt → log, skip execution
    ├─ STRONG signal (paper) → paper_trade(), record in open_positions.json
    └─ STRONG signal (live)  → set_leverage, market entry, SL order, TP order
```

---

## 9. Trade Execution Layer

### Exchange Connection

```python
# Hyperliquid (recommended — DEX, self-custody)
exchange = ccxt.hyperliquid({
    "walletAddress": os.getenv("HYPERLIQUID_WALLET_ADDRESS"),
    "privateKey":    os.getenv("HYPERLIQUID_PRIVATE_KEY"),
})

# Binance Futures (CEX alternative)
exchange = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY"),
    "secret": os.getenv("BINANCE_SECRET"),
    "options": {"defaultType": "future"},
})
```

### Order Bracket

```python
# 1. Set leverage
exchange.set_leverage(LEVERAGE, symbol)

# 2. Market entry
entry = exchange.create_market_order(symbol, side, qty)

# 3. Stop-loss
sl_order = exchange.create_order(symbol, "stop_market", close_side, qty,
    params={"stopPrice": stop_loss, "reduceOnly": True})

# 4. Take-profit
tp_order = exchange.create_order(symbol, "take_profit_market", close_side, qty,
    params={"stopPrice": target, "reduceOnly": True})
```

### Leverage Safety Check

Before any order is placed:

```python
sl_pct  = abs(entry - stop_loss) / entry      # stop distance as %
liq_pct = 1.0 / LEVERAGE                       # liquidation distance as %

if sl_pct >= liq_pct:
    BLOCK — liquidation would fire before stop-loss
```

At 3× leverage: `liq_pct = 33%`. A typical ATR stop at 3% is safely inside this.

### Paper Trading

When `PAPER_TRADING=true`, all signals are logged to `signals.db` and `open_positions.json` but no exchange orders are placed. The leverage safety check still runs and warns in logs.

---

## 10. Risk Management

| Parameter | Value | Rationale |
|---|---|---|
| Risk per trade | 1% of effective capital | Conservative — Kelly optimal at observed win rates is ~3%, 1% provides buffer |
| Stop-loss | ATR(14) × 1.5 on 4h candles | Volatility-adjusted; 4h ATR clears intrabar noise that 1h stops cannot |
| Take-profit | 2:1 R/R | Only trade setups with mathematical edge |
| Max drawdown halt | −10% from peak | Automatic — bot pauses, human reviews |
| STRONG-only execution | Mandatory | Maintains live/backtest consistency |
| WEAK signal handling | Log, do not trade | WEAK not in backtest; executing them invalidates CAGR figures |
| Trade contradiction | Mandatory NO_TRADE | Disagreement between systems = real uncertainty signal |
| Position timeout | 48h (12 × 4h bars) | Prevents stale exposure; mirrors backtest force-close |
| Capital buffer | 40% of balance free | `CAPITAL_PCT=0.60` → 40% always available for liquidation margin |
| Leverage | 3× recommended | Liquidation at 33% — far outside any ATR stop |
| Weekly trade cap | None (99) | Signal quality determines frequency; artificial cap was shown to hurt CAGR |

---

## 11. Asset Configuration (DNA)

Stored in `assets_dna.json`, computed via `scripts/calculate_natal.py`:

```json
{
  "BTC": {
    "name":             "Bitcoin",
    "symbol":           "BTC/USDT",
    "genesis_datetime": "2009-01-03T18:15:05Z",
    "life_path_number": 6,
    "ticker_vibration": 7,
    "natal_western": {
      "Sun": 283.5, "Moon": 201.3, "Mercury": 301.2,
      "Venus": 329.8, "Mars": 155.4, "Jupiter": 303.1,
      "Saturn": 170.2, "Uranus": 357.1, "Neptune": 321.8, "Pluto": 271.3
    },
    "natal_vedic": {
      "Sun": 259.3, "Moon": 177.1, "Mercury": 277.0,
      "Venus": 305.6, "Mars": 131.2, "Jupiter": 278.9,
      "Saturn": 146.0, "Uranus": 333.9, "Neptune": 297.6, "Pluto": 247.1,
      "Rahu": 285.0, "Ketu": 105.0
    },
    "dasha_weights": null
  }
}
```

Life Path Number for BTC: `digital_root(2+0+0+9+0+1+0+3) = digital_root(15) = 6`

Ticker Vibration for BTC: `B=2, T=2, C=3 → 7`

---

## 12. Backtesting Architecture

### Data Flow

```
scripts/backtest.py
  └── fetches 1h OHLCV (ccxt) + computes cosmic signals
  └── outputs: logs/backtest_BTC_YYYY-MM-DD_YYYY-MM-DD.csv
        columns: timestamp, open, high, low, price(close),
                 action, western_score, vedic_score,
                 western_slope, vedic_slope, resonance_day, nakshatra

scripts/fetch_funding_rates.py
  └── fetches historical funding rates from Binance Futures API
  └── outputs: logs/funding_BTCUSDT_*.csv

scripts/final_backtest.py
  └── loads signal CSV + funding CSV
  └── resamples 1h rows into true N-hour OHLC candles
  └── runs simulate() with realistic features
  └── outputs tables, equity curve, trade list
```

### Realistic Simulation Features

| Feature | Implementation |
|---|---|
| Candle resampling | 1h CSV → N-hour OHLC: open = first open, high = max(highs), low = min(lows), close = last close |
| Intrabar SL/TP | Uses H/L of each candle: if both SL and TP triggered same bar, SL wins (worst-case conservative) |
| Entry slippage | 0.05% added to entry price (market order fill cost) |
| Exchange fees | 0.10% round-trip (0.05% entry + 0.05% exit — Hyperliquid taker) |
| Funding rates | Applied every 8h from real historical Binance Futures data |
| Trade timeout | Force-close at `MAX_OPEN_BARS` bars (default 12 = 48h) |
| EMA filter | Applied on resampled candle closes — matches live engine exactly |
| Compounding | Equity updated after each trade; subsequent position sizes scale accordingly |

---

## 13. Validated Optimal Settings

Results from the full realistic 4-year backtest (2022–2025), intrabar SL/TP, fees, slippage, real funding rates.

### Cadence Comparison (EMA 2-way, EMA 200)

| Cadence | CAGR | Worst DD | Notes |
|---|---|---|---|
| 1h | 7.2% | 52% | Intrabar wicks destroy ATR stops |
| 2h | 11.1% | 33% | Better but still too much noise |
| **4h** | **22.2%** | **23%** | **BEST — stops wider than noise** |
| 8h | 14.2% | 24% | Too few trades, misses signals |

### Filter Matrix (4h cadence, 2022–2025)

| Filter | Avg WR | CAGR | Worst DD |
|---|---|---|---|
| No filter | — | −7.6% | — |
| Nakshatra only | 35.6% | −6.0% | 44% |
| EMA one_way | 40.2% | 33.8% | 23% |
| **EMA two_way** | **41.1%** | **38.2%** | **27%** |
| NK + EMA one_way | 40.2% | 22.2% | 23% |
| NK + EMA two_way | 40.6% | 24.4% | 22% |

### EMA Period Sweep (4h, EMA two_way, 2022–2025)

| EMA Period | CAGR | Avg WR |
|---|---|---|
| **EMA(20)** | **151.9%** | **48.9%** |
| EMA(50) | 84.9% | 45.5% |
| EMA(100) | 61.4% | 43.7% |
| EMA(200) | 38.2% | 41.1% |
| EMA(350) | 27.6% | 39.9% |

EMA(20) dramatically outperforms longer periods because it filters on near-term momentum, aligning trades with the current 3–5 day trend rather than the multi-month macro trend.

### MAX_OPEN_BARS Sweep (4h, EMA two_way, EMA 20)

| Max Hold | Equivalent | CAGR | Worst DD |
|---|---|---|---|
| 6 bars | 24h | lower | lower |
| **12 bars** | **48h** | **best balance** | moderate |
| 24 bars | 96h | similar | higher |
| 48 bars | 192h | similar | highest |

### Current Optimal Configuration

```
CHECK_INTERVAL_MINUTES = 240   (4h)
EMA_FILTER             = two_way
EMA_PERIOD             = 20
NAKSHATRA_FILTER       = false
MAX_OPEN_BARS          = 12    (48h)
RISK_PER_TRADE_PCT     = 0.01  (1%)
RR_RATIO               = 2.0
ATR_MULTIPLIER         = 1.5
LEVERAGE               = 3
```

**4-year summary at these settings (2022–2025, realistic):**
- CAGR: ~151.9% (compounding, starting $10,000)
- Average win rate: ~48.9%
- Worst single-year drawdown: ~26%

---

## 14. Deployment Plan

### Phase 1 — Paper Trading (Weeks 1–4)

- Run bot locally with `PAPER_TRADING=true`
- All signals logged to `logs/signals.db`
- No real trades placed
- Review `signals.db` daily: check action, EMA filter reason, scores, sizes
- Confirm signal frequency matches expectations (1–3 STRONG per week)

### Phase 2 — Small Live (Weeks 5–8)

- Set `PAPER_TRADING=false`
- Keep capital small (e.g. $200 wallet, `CAPITAL_PCT=0.60` → $120 effective)
- Only `STRONG_BUY` / `STRONG_SELL` execute (enforced in code)
- Monitor stop-loss hits; compare to paper log

### Phase 3 — Full Deployment

VPS setup (DigitalOcean / AWS t3.small ~$10/month):

```bash
# As a background process
nohup python main.py > logs/nohup.out 2>&1 &

# As a systemd service (recommended)
[Unit]
Description=Slope Around Medium Astro-Bot

[Service]
WorkingDirectory=/opt/astro-bot
ExecStart=/opt/astro-bot/venv/bin/python main.py
Restart=always
RestartSec=30
EnvironmentFile=/opt/astro-bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
# Or as Docker
docker build -t astro-bot .
docker run -d --env-file .env \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/ephe:/app/ephe \
    astro-bot
```

### Production Checklist

- [ ] `PAPER_TRADING=false` in `.env`
- [ ] `CAPITAL_PCT=0.60` set to appropriate fraction of real balance
- [ ] API keys restricted to trade-only permissions (no withdrawal)
- [ ] Use a **dedicated API sub-wallet** — never your main wallet private key
- [ ] `logs/equity_state.json` initialized (auto-created on first balance fetch)
- [ ] `logs/open_positions.json` exists and is empty `[]`
- [ ] Log rotation set up (`logs/astro_bot_*.log` auto-rotates daily in loguru config)
- [ ] Alerts configured if `logs/astro_bot_*.log` shows DRAWDOWN HALT

---

## 15. Full File Structure

```
astro-bot/
│
├── BUILD_PLAN.md              ← This document
├── README.md                  ← Quick start + formulas reference
├── requirements.txt           ← Python dependencies
├── .env.example               ← Committed safe template (no secrets)
├── .env                       ← Real secrets (gitignored)
├── .gitignore
├── assets_dna.json            ← Asset natal charts + numerology
├── config.py                  ← All tunable parameters
├── main.py                    ← Entry point + APScheduler loop
│
├── core/
│   ├── __init__.py
│   ├── astro_engine.py        ← Planet positions (Western + Vedic)
│   ├── numerology.py          ← Life Path, UDN, Ticker Vibration
│   ├── signal_engine.py       ← Score, slope, gate, EMA filter
│   └── score_history.py       ← Ring buffer for Medium + Slope
│
├── exchange/
│   ├── __init__.py
│   ├── market_data.py         ← OHLCV, ATR(14), EMA(20), balance
│   └── trade_executor.py      ← Orders, 48h timeout, drawdown halt
│
├── scripts/
│   ├── backtest.py            ← Generate cosmic signal CSV
│   ├── calculate_natal.py     ← Compute natal charts
│   ├── fetch_funding_rates.py ← Historical funding rates
│   └── final_backtest.py      ← Realistic simulation + comparison suite
│
├── dashboard/
│   └── app.py                 ← Streamlit live signal dashboard (optional)
│
├── logs/                      ← Gitignored; created at runtime
│   ├── signals.db             ← SQLite audit log
│   ├── open_positions.json    ← Active trade timeout tracker
│   ├── equity_state.json      ← Peak equity for drawdown halt
│   └── backtest_BTC_*.csv     ← Generated signal CSVs
│
└── ephe/                      ← Gitignored; download separately
    ├── sepl_18.se1
    ├── semo_18.se1
    └── seas_18.se1
```

---

*Slope Around Medium Astro-Bot — Build Plan v2.0 | Feb 2026*
