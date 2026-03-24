"""
Market Data — Hyperliquid perpetuals via ccxt.

All live market data (price, OHLCV, ATR, EMA, account balance) is sourced
exclusively from Hyperliquid. Authentication uses an EVM wallet address +
private key set in .env — no traditional API key/secret required.

Note on backtest data: historical OHLCV and funding rates used in backtests
are fetched from the Binance public API (no key needed) via scripts/backtest.py
and scripts/fetch_funding_rates.py. This is only because Hyperliquid launched
in late 2023 and lacks the 2022–2023 data needed for multi-year backtests.
The live trading engine is Hyperliquid-only.
"""
import os
import random
import time
import ccxt
import numpy as np
import pandas as pd
from loguru import logger

import config


# ── Hyperliquid exchange singleton ────────────────────────────────────────────

def _build_exchange() -> ccxt.hyperliquid:
    """
    Build a ccxt Hyperliquid instance.

    Paper trading (PAPER_TRADING=true):
      No credentials required. Price and OHLCV are public endpoints.
      Balance fetch will fail and automatically fall back to CAPITAL_USDT.
      Leave HYPERLIQUID_WALLET_ADDRESS and HYPERLIQUID_PRIVATE_KEY blank.

    Live trading (PAPER_TRADING=false):
      Set both HYPERLIQUID_WALLET_ADDRESS and HYPERLIQUID_PRIVATE_KEY in .env.
      Use a dedicated agent sub-wallet — never your main wallet.
    """
    wallet  = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    privkey = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

    cfg: dict = {"enableRateLimit": True}
    if wallet:
        cfg["walletAddress"] = wallet
    if privkey:
        cfg["privateKey"] = privkey

    return ccxt.hyperliquid(cfg)


_exchange: ccxt.hyperliquid = None

# Simple in-memory ticker cache to avoid duplicated requests within a short window.
# This helps when multiple bot components query the same symbol close together.
_ticker_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, epoch_seconds)


def _is_rate_limited(exc: Exception) -> bool:
    """
    Best-effort detection for HTTP 429 rate limiting.
    ccxt error objects can vary by backend/version, so we check both attributes
    and the string message.
    """
    status = getattr(exc, "status", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg


def _sleep_backoff_seconds(attempt: int, *, base_delay_s: float, max_delay_s: float) -> None:
    # attempt is 1-based (attempt=1 -> base_delay_s)
    delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
    # Add a small jitter to avoid thundering herd across restarts.
    delay *= random.uniform(0.85, 1.15)
    time.sleep(delay)


def _with_rate_limit_retries(
    *,
    symbol: str,
    op_name: str,
    attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    fn,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if _is_rate_limited(e) and attempt < attempts:
                logger.warning(
                    f"{op_name} rate-limited on {symbol} (attempt {attempt}/{attempts}); "
                    f"retrying with backoff..."
                )
                _sleep_backoff_seconds(attempt, base_delay_s=base_delay_s, max_delay_s=max_delay_s)
                continue
            # Non-rate-limit error, or we're out of retries.
            raise
    assert last_exc is not None
    raise last_exc


def get_exchange() -> ccxt.hyperliquid:
    """Return the shared Hyperliquid exchange instance (lazy singleton)."""
    global _exchange
    if _exchange is None:
        _exchange = _build_exchange()
    return _exchange


# ── OHLCV ─────────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str = "4h", limit: int = 100) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Hyperliquid.

    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume
    """
    try:
        ex = get_exchange()
        def _fetch():
            return ex.fetch_ohlcv(symbol, timeframe, limit=limit)

        raw = _with_rate_limit_retries(
            symbol=symbol,
            op_name="OHLCV",
            attempts=3,
            base_delay_s=1.0,
            max_delay_s=8.0,
            fn=_fetch,
        )
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol} on Hyperliquid: {e}")
        return pd.DataFrame()


def get_current_price(symbol: str) -> float:
    """Fetch the latest mark/last price from Hyperliquid."""
    try:
        ttl_s = float(os.getenv("HYPERLIQUID_TICKER_CACHE_TTL_SECONDS", "20"))
        now = time.time()
        cached = _ticker_cache.get(symbol)
        if cached:
            cached_price, cached_at = cached
            if now - cached_at <= ttl_s:
                return cached_price

        ex = get_exchange()

        def _fetch():
            ticker = ex.fetch_ticker(symbol)
            return float(ticker["last"])

        price = _with_rate_limit_retries(
            symbol=symbol,
            op_name="Price",
            attempts=3,
            base_delay_s=1.0,
            max_delay_s=8.0,
            fn=_fetch,
        )

        _ticker_cache[symbol] = (price, now)
        return price
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol} on Hyperliquid: {e}")
        return 0.0


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average True Range over the last `period` candles.
    True Range = max(high−low, |high−prev_close|, |low−prev_close|)
    """
    if df.empty or len(df) < period + 1:
        logger.warning("Not enough data for ATR calculation")
        return 0.0

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def compute_ema(df: pd.DataFrame, period: int) -> float:
    """
    Exponential Moving Average of close price over `period` candles.
    Returns 0.0 if insufficient data.
    """
    if df.empty or len(df) < period:
        logger.warning(f"Not enough data for EMA({period}) — need {period}, got {len(df)}")
        return 0.0
    return float(df["close"].ewm(span=period, adjust=False).mean().iloc[-1])


# ── Account balance ───────────────────────────────────────────────────────────

def get_account_balance() -> float:
    """
    Fetch total account equity from Hyperliquid (USDC-denominated).

    Hyperliquid's margin account reports equity under:
      balance['total']['USDC']                              (ccxt normalized)
      balance['info']['marginSummary']['accountValue']      (raw API fallback)

    Falls back to config.CAPITAL_USDT if the fetch fails — the bot keeps
    running in paper mode without a valid wallet configured.
    """
    try:
        ex = get_exchange()
        balance = ex.fetch_balance()

        equity = (
            balance.get("total", {}).get("USDC")
            or balance.get("total", {}).get("USDT")
            or balance.get("info", {}).get("marginSummary", {}).get("accountValue")
        )

        if equity is None:
            raise ValueError(f"Unexpected balance structure: {balance}")

        val = float(equity)
        if val > 0:
            logger.info(f"Hyperliquid account equity: ${val:,.2f} USDC")
            return val

        # Wallet connected but has $0 — agent wallet not yet funded.
        # Fall back to CAPITAL_USDT for paper trading sizing.
        logger.warning(
            f"Hyperliquid wallet balance is $0.00 — agent wallet not funded yet. "
            f"Using CAPITAL_USDT=${config.CAPITAL_USDT:,.2f} for position sizing. "
            f"This is fine for paper trading."
        )
        return config.CAPITAL_USDT

    except Exception as e:
        logger.warning(
            f"Hyperliquid balance fetch failed ({e}) — "
            f"using CAPITAL_USDT=${config.CAPITAL_USDT:,.2f}"
        )
        return config.CAPITAL_USDT


# ── Composite market-data fetch (used by bot_cycle) ───────────────────────────

def get_effective_capital() -> float:
    """
    effective_capital = account_balance × CAPITAL_PCT

    CAPITAL_PCT (default 0.60) keeps 40% of the balance as a free
    liquidation buffer. Position sizes auto-scale with the account.
    """
    balance = get_account_balance()
    capital = balance * config.CAPITAL_PCT
    logger.info(
        f"Effective capital: ${capital:,.2f}  "
        f"({config.CAPITAL_PCT*100:.0f}% of ${balance:,.2f} balance)"
    )
    return capital


def get_price_atr_ema(symbol: str) -> tuple[float, float, float, float, float]:
    """
    Returns (current_price, atr, ema, last_candle_high, last_candle_low).

    last_candle_high/low are from the most recently closed
    config.ATR_EMA_TIMEFRAME bar. Used for
    intrabar TP/SL checks in paper mode (so we detect TP hit when price
    wicks to target then bounces within the same bar).
    """
    needed = max(config.ATR_PERIOD, config.EMA_PERIOD) + 20
    tf    = getattr(config, "ATR_EMA_TIMEFRAME", "4h")
    df    = fetch_ohlcv(symbol, timeframe=tf, limit=needed)
    price = get_current_price(symbol)
    atr   = compute_atr(df, period=config.ATR_PERIOD)
    ema   = compute_ema(df, period=config.EMA_PERIOD)

    last_high = float(df["high"].iloc[-1]) if not df.empty else price
    last_low  = float(df["low"].iloc[-1])  if not df.empty else price

    if price <= 0:
        logger.warning(
            f"{symbol} — price unavailable (<=0); skipping cycle. "
            f"ATR({config.ATR_PERIOD})/{tf}: {atr:.2f} | EMA({config.EMA_PERIOD})/{tf}: {ema:.2f}"
        )
    else:
        logger.info(
            f"{symbol} — Price: {price:.2f} | "
            f"ATR({config.ATR_PERIOD})/{tf}: {atr:.2f} | "
            f"EMA({config.EMA_PERIOD})/{tf}: {ema:.2f}"
        )
    return price, atr, ema, last_high, last_low


def get_regime_ema_state() -> tuple[float, float]:
    """
    Returns (benchmark_price, benchmark_ema) for macro regime detection.
    Fixed model: BTC/USDT EMA(282) on 45m (non-configurable).
    """
    # Hyperliquid perp symbol format (e.g. BTC/USDC:USDC).
    # BTC/USDT is not a valid market symbol on Hyperliquid.
    symbol = "BTC/USDC:USDC"
    tf = "45m"
    period = 282
    needed = max(period + 20, 320)

    df = fetch_ohlcv(symbol, timeframe=tf, limit=needed)
    price = get_current_price(symbol)
    ema = compute_ema(df, period=period)

    if price > 0 and ema > 0:
        logger.info(f"Regime {symbol} — Price: {price:.2f} | EMA({period})/{tf}: {ema:.2f}")
    else:
        logger.warning(f"Regime {symbol} unavailable — Price: {price:.2f}, EMA: {ema:.2f}")
    return price, ema
