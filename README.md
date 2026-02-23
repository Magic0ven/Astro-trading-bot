# Slope Around Medium — Astro-Bot

> A cosmic momentum trading bot that trades the **velocity** (slope) of a composite planetary resonance waveform — not individual predictions.

Both the Western (Tropical) and Vedic (Sidereal) astrological systems must independently agree before any trade is executed. A contradiction between the two systems produces a mandatory `NO_TRADE`.

---

## Quick Start

### 1. Install

```bash
cd astro-bot
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
nano .env    # Add your exchange API keys; review every parameter
```

### 3. Compute Natal Charts

Required once per asset before running the bot. Calculates exact planetary positions at each asset's genesis datetime and writes them to `assets_dna.json`.

```bash
python scripts/calculate_natal.py             # All assets
python scripts/calculate_natal.py --asset BTC # Single asset
```

### 4. Run

```bash
python main.py
```

The bot runs one cycle immediately on startup, then repeats every `CHECK_INTERVAL_MINUTES` (default: 240 min = 4h). All signals are logged to `logs/signals.db`.

---

## Signal Pipeline (Every Cycle)

```
① Fetch live planetary positions  (Western Tropical + Vedic Sidereal)
                    ↓
② Compute Composite Scores
   Western:  Σ cos(live_lon − natal_lon) × retrograde_dir
   Vedic:    same, + Rahu/Ketu, × nakshatra_multiplier × dasha_weights
                    ↓
③ Apply Numerology Multiplier
   UDN == Asset Life Path Number?  → 1.5× (Resonance Day)
   Otherwise                       → 1.0× (Normal Day)
   Adjusted Score = Raw Score × Multiplier
                    ↓
④ Update Rolling History  (window = 5 bars)
   Medium = mean(last 5 adjusted scores)
   Slope  = linear regression slope of last 5 adjusted scores
                    ↓
⑤ Single-System Signals
   BUY   if: score > medium  AND  slope > SLOPE_THRESHOLD
   SELL  if: score < medium  OR   slope < −SLOPE_THRESHOLD
   HOLD  otherwise
                    ↓
⑥ Dual-System Gate
   W=BUY  + V=BUY  → STRONG_BUY
   W=SELL + V=SELL → STRONG_SELL
   W=BUY  + V=HOLD → WEAK_BUY   (skipped — only STRONG signals are executed)
   W=SELL + V=HOLD → WEAK_SELL  (skipped — only STRONG signals are executed)
   W=HOLD + V=HOLD → HOLD
   W=BUY  + V=SELL → NO_TRADE   (contradiction — mandatory sit-out)
                    ↓
⑦ Reliability Filters  (applied in order)
   a. Nakshatra Block (NAKSHATRA_FILTER=false by default — see config)
   b. EMA Trend Filter (EMA_FILTER=two_way):
      price < EMA(20) on 4h → block BUY
      price > EMA(20) on 4h → block SELL
                    ↓
⑧ WEAK signal gate  — WEAK_BUY / WEAK_SELL are logged but never executed.
   Only STRONG_BUY / STRONG_SELL proceed to order placement.
                    ↓
⑨ Drawdown Halt — if equity fell ≥ MAX_DRAWDOWN_HALT_PCT from peak, skip trade.
                    ↓
⑩ Calculate Stop-Loss, Take-Profit, Position Size
                    ↓
⑪ Execute (paper or live)  →  record to logs/signals.db + logs/open_positions.json
```

---

## Exact Formulas

### Numerology

```
digital_root(n)      = repeatedly sum digits until single digit (1–9)
                       e.g. 17 → 1+7 = 8

Life Path Number     = digital_root(Σ all digits of genesis YYYYMMDD)
                       BTC genesis 2009-01-03:
                       2+0+0+9+0+1+0+3 = 15 → 1+5 = 6

Ticker Vibration     = digital_root(Σ Pythagorean chart values of symbol letters)
                       Pythagorean: A=1 B=2 C=3 D=4 E=5 F=6 G=7 H=8 I=9
                                    J=1 K=2 L=3 M=4 N=5 O=6 P=7 Q=8 R=9
                                    S=1 T=2 U=3 V=4 W=5 X=6 Y=7 Z=8
                       BTC: B=2 T=2 C=3 → 7

Universal Day Number = digital_root(Σ all digits of today's YYYYMMDD)
(UDN)                  Feb 23 2026: 2+0+2+6+0+2+2+3 = 17 → 8

Numerology Multiplier (Pc):
   Pc = 1.5   if UDN == Asset Life Path Number  (Resonance Day)
   Pc = 1.0   otherwise
```

### Aspect Score (Composite Score)

```
For each live planet i and each natal planet j:

  θ_ij        = |longitude_live_i  −  longitude_natal_j|  mod 360°
  retro_i     = +1  if planet i is direct  (speed > 0)
                −1  if planet i is retrograde
  force_ij    = retro_i × cos(θ_ij in radians)

Cosine interpretation:
   0°   → +1.00  (Conjunction    — peak resonance)
   60°  → +0.50  (Sextile        — harmony)
   90°  →  0.00  (Square         — tension / neutral)
   120° → −0.50  (Trine          — eases pressure, measured as reduced force)
   180° → −1.00  (Opposition     — maximum tension)

Western Composite Score (10 live × 10 natal planet pairs):
  W(t) = Σ_ij  force_ij                         (100 pairs max)

Vedic Composite Score (12 live × 12 natal pairs, incl. Rahu/Ketu):
  V(t) = Σ_ij  force_ij  × dasha_weight_i  ×  nakshatra_multiplier
  nakshatra_multiplier:
    1.2  if Moon is in a favorable Nakshatra  (Rohini, Hasta, Punarvasu, …)
    0.8  if Moon is in an unfavorable Nakshatra (Ardra, Bharani, Mula, …)
    1.0  otherwise  (neutral)

Adjusted Scores (Numerology applied):
  W_adj(t) = W(t) × Pc
  V_adj(t) = V(t) × Pc
```

### Medium and Slope

```
History window N = 5  (last 5 adjusted scores)

Medium  = mean( [W_adj(t−4), W_adj(t−3), W_adj(t−2), W_adj(t−1), W_adj(t)] )
Slope   = linear regression slope of the same 5-point window
          (numpy.polyfit(x, y, 1)[0] where x = [0,1,2,3,4])
```

### Single-System Signal

```
BUY   if:  W_adj(t) > Medium  AND  Slope >  SLOPE_THRESHOLD
SELL  if:  W_adj(t) < Medium  OR   Slope < −SLOPE_THRESHOLD
HOLD  otherwise

Default SLOPE_THRESHOLD = 0.5
```

### Stop-Loss and Take-Profit

```
ATR(14) computed on 4h OHLC candles (period = 14 bars)

For a LONG (BUY) trade:
  Stop-Loss  = entry_price − ATR × ATR_MULTIPLIER      (default 1.5)
  Take-Profit = entry_price + |entry − stop_loss| × RR_RATIO  (default 2.0)

For a SHORT (SELL) trade:
  Stop-Loss  = entry_price + ATR × ATR_MULTIPLIER
  Take-Profit = entry_price − |entry − stop_loss| × RR_RATIO

Example at BTC = $90,000, ATR(14)/4h = $1,800:
  SL  = $90,000 − $1,800 × 1.5  = $87,300   (3.0% below entry)
  TP  = $90,000 + $2,700 × 2.0  = $95,400   (6.0% above entry)
```

### Position Sizing

```
effective_capital  = account_balance × CAPITAL_PCT          (default 0.60)
risk_amount        = effective_capital × RISK_PER_TRADE_PCT  (default 0.01)
sl_distance        = |entry_price − stop_loss_price|

base_qty (coins)   = risk_amount / sl_distance
notional_usdt      = base_qty × entry_price
                   = (risk_amount / sl_distance) × entry_price

size_factor:
  STRONG signal → 1.0
  WEAK signal   → 0.5  (not executed anyway — STRONG-only policy)

final_notional = notional_usdt × size_factor × Pc

Example ($200 account, CAPITAL_PCT=0.60, RISK_PCT=0.01):
  effective_capital = $200 × 0.60 = $120
  risk_amount       = $120 × 0.01 = $1.20
  sl_distance       = $90,000 − $87,300 = $2,700
  base_qty          = $1.20 / $2,700   = 0.000444 BTC
  notional_usdt     = 0.000444 × $90,000 = $40.00
  margin_required   = $40.00 / 3 (leverage) = $13.33
```

### Leverage and Liquidation Safety

```
margin_required    = notional_usdt / LEVERAGE
liquidation_dist   = 1 / LEVERAGE  (fraction of entry price)

Safety rule: sl_pct  <  liquidation_dist
  At 3× leverage: liquidation is 33.3% away
  With ATR stop ≈ 3%: 3% << 33% → SAFE

The bot BLOCKS any trade where sl_pct ≥ liquidation_dist.
```

### EMA Trend Filter (two_way)

```
EMA(N) = exponential moving average of close prices over N bars
         (pandas ewm(span=N, adjust=False))

Computed on 4h candles, N = EMA_PERIOD (default 20)
EMA(20)/4h ≈ 3.3-day short-term momentum filter

two_way filter rules:
  price < EMA → block BUY  signals (don't buy into a downtrend)
  price > EMA → block SELL signals (don't short an uptrend)
```

---

## Configuration Reference

All parameters live in `config.py` and can be overridden via `.env`.

| Parameter | Default | Description |
|---|---|---|
| `CHECK_INTERVAL_MINUTES` | `240` | Bot cycle frequency in minutes. 4h is the backtested optimal cadence |
| `SCORE_HISTORY_WINDOW` | `5` | Number of past scores kept for Medium + Slope calculation |
| `SLOPE_THRESHOLD` | `0.5` | Minimum slope magnitude to trigger BUY or SELL |
| `MAX_WEEKLY_TRADES` | `99` | Soft cap on trades per week. 99 = effectively no cap |
| `EMA_FILTER` | `two_way` | `none` / `one_way` / `two_way`. Controls how EMA blocks trades |
| `EMA_PERIOD` | `20` | EMA period on 4h candles. EMA(20) = best backtested CAGR |
| `NAKSHATRA_FILTER` | `false` | Hard-block trades in unfavorable Nakshatras. Disabled (hurts CAGR at 4h) |
| `CAPITAL_PCT` | `0.60` | Fraction of live account balance used as sizing base each cycle |
| `CAPITAL_USDT` | `120` | Static fallback capital if live balance fetch fails |
| `RISK_PER_TRADE_PCT` | `0.01` | Fraction of effective capital risked per trade (1%) |
| `MAX_DRAWDOWN_HALT_PCT` | `0.10` | Suspend trading when equity drops 10% from peak |
| `MAX_OPEN_BARS` | `12` | Force-close a trade after 12 × 4h = 48h if SL/TP not hit |
| `RR_RATIO` | `2.0` | Take-profit distance = SL distance × RR_RATIO |
| `ATR_MULTIPLIER` | `1.5` | Stop-loss distance = ATR(14) × ATR_MULTIPLIER |
| `ATR_PERIOD` | `14` | Period for ATR calculation (hardcoded, not env-configurable) |
| `LEVERAGE` | `3` | Exchange leverage. Dollar risk per trade is unchanged; only margin changes |
| `EXCHANGE` | `binance` | Exchange name (ccxt identifier) |
| `MARKET_TYPE` | `future` | `future` or `spot` |
| `PAPER_TRADING` | `true` | `true` = log only, never place real orders |
| `ACTIVE_ASSET` | `BTC` | Must match a key in `assets_dna.json` |

### Why These Defaults

| Setting | Backtested Evidence |
|---|---|
| `CHECK_INTERVAL_MINUTES=240` | Realistic 4-year simulation (2022–2025): 4h = 151.9% CAGR, 26% max DD. 1h = 7.2% CAGR (intrabar wicks destroy ATR stops) |
| `EMA_FILTER=two_way` | Filter matrix: no filter = −7.6% CAGR; one_way = 33.8%; two_way = 38.2% |
| `EMA_PERIOD=20` | EMA period sweep: EMA(20) = 151.9%, EMA(50) = 84.9%, EMA(200) = 38.2% |
| `NAKSHATRA_FILTER=false` | Adding NK block reduces CAGR from 38.2% → 24.4% at 4h+EMA2way |
| `MAX_OPEN_BARS=12` | 48h timeout tested at 6/12/24/48 — 12 gives best CAGR/DD balance |
| `RISK_PER_TRADE_PCT=0.01` | Kelly Criterion analysis: 1% is conservative but ruins risk is low |

---

## Backtesting

### Step 1 — Generate cosmic signal CSV

```bash
python scripts/backtest.py --asset BTC --start 2024-01-01 --end 2024-12-31
# Output: logs/backtest_BTC_2024-01-01_2024-12-31.csv
```

### Step 2 — Fetch historical funding rates (for realistic simulation)

```bash
python scripts/fetch_funding_rates.py --start 2024-01-01 --end 2024-12-31
# Output: logs/funding_BTCUSDT_2024-01-01_2024-12-31.csv
```

### Step 3 — Run realistic simulation

```bash
# Single year at current config settings
python scripts/final_backtest.py --file logs/backtest_BTC_2024-01-01_2024-12-31.csv \
    --funding-file logs/funding_BTCUSDT_2024-01-01_2024-12-31.csv --label "2024"

# Compare all years side by side
python scripts/final_backtest.py --compare

# Compare cadences (1h / 2h / 4h / 8h) across all years
python scripts/final_backtest.py --compare-cadence

# Compare filter combinations at 4h
python scripts/final_backtest.py --compare-filters

# Compare EMA periods
python scripts/final_backtest.py --compare-ema

# Compare RISK_PER_TRADE_PCT levels
python scripts/final_backtest.py --compare-risk
```

### Realistic simulation features

| Feature | Detail |
|---|---|
| Intrabar SL/TP | Uses candle HIGH/LOW to detect if stop or target was hit within the bar. Worst-case: stop-out assumed before TP if both triggered same candle |
| Entry slippage | +0.05% on entry (market order spread) |
| Exchange fees | 0.05% taker on entry + 0.05% taker on exit = 0.10% round-trip |
| Funding rates | Applied every 8h from real Binance Futures historical data |
| Candle resampling | 1h signal CSV resampled to true N-hour OHLC before simulation |

---

## Runtime State Files

| File | Contents | Purpose |
|---|---|---|
| `logs/signals.db` | SQLite — every signal and trade | Audit trail, weekly cap counter |
| `logs/open_positions.json` | JSON — currently open trades with timestamps | 48h timeout tracker |
| `logs/equity_state.json` | JSON — `{"peak_equity": N}` | Drawdown halt — persists across restarts |

---

## Files Reference

| Path | Purpose |
|---|---|
| `main.py` | Entry point, APScheduler loop |
| `config.py` | All tunable parameters (reads from `.env`) |
| `assets_dna.json` | Asset natal charts + numerology data |
| `.env` | API keys + runtime overrides (never commit) |
| `.env.example` | Safe template to commit |
| `core/astro_engine.py` | Planet positions via pyswisseph |
| `core/numerology.py` | Life Path, UDN, Pythagorean Ticker Vibration |
| `core/signal_engine.py` | Composite score, slope, dual-gate, EMA/NK filters |
| `core/score_history.py` | Rolling score buffer for Medium + Slope |
| `exchange/market_data.py` | OHLCV, ATR, EMA, account balance via ccxt |
| `exchange/trade_executor.py` | Order placement, 48h timeout, drawdown halt, SQLite log |
| `scripts/backtest.py` | Fetch historical OHLCV + compute signals → CSV |
| `scripts/calculate_natal.py` | Compute natal charts from genesis datetime |
| `scripts/fetch_funding_rates.py` | Pull historical funding rates from Binance |
| `scripts/final_backtest.py` | Realistic multi-year simulation and comparison suite |

---

## Adding a New Asset

1. Add the entry to `assets_dna.json`:
```json
"SOL": {
    "name":              "Solana",
    "symbol":            "SOL/USDT",
    "genesis_datetime":  "2020-03-16T00:00:00Z",
    "life_path_number":  null,
    "ticker_vibration":  null,
    "natal_western":     null,
    "natal_vedic":       null
}
```

2. Compute natal chart:
```bash
python scripts/calculate_natal.py --asset SOL
```

3. Set in `.env`:
```
ACTIVE_ASSET=SOL
```

---

## Safety Rules

- **Always start with `PAPER_TRADING=true`** — run for several weeks before live
- **Never risk capital you cannot afford to lose**
- **`NO_TRADE` exists for a reason** — a contradiction between Western and Vedic is a real signal to stay out
- **WEAK signals are intentionally not executed** — they were not included in the backtests that produced the CAGR figures; live behaviour must match backtest assumptions
- **Drawdown halt is automatic** — if equity drops 10% from peak, the bot stops placing trades until you manually reset `logs/equity_state.json` after reviewing positions

---

## Dependencies

| Library | Purpose |
|---|---|
| `pyswisseph` | Swiss Ephemeris — planet positions (Western + Vedic) |
| `ccxt` | Exchange connectivity (Binance, Bybit, Hyperliquid, …) |
| `pandas` | OHLCV data, score history, EMA |
| `numpy` | Cosine math, linear regression slope |
| `apscheduler` | Interval scheduling |
| `loguru` | Structured logging |
| `rich` | Terminal dashboard panels |
| `python-dotenv` | Load `.env` config |
| `python-dateutil` | Genesis datetime parsing |
