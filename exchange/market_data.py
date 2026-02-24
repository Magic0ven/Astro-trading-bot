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
        raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
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
        ex = get_exchange()
        ticker = ex.fetch_ticker(symbol)
        return float(ticker["last"])
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


def get_price_atr_ema(symbol: str) -> tuple[float, float, float]:
    """
    Returns (current_price, atr, ema) all computed on 4h Hyperliquid candles.

    Fetches enough candles to warm up both ATR(14) and EMA(20).
    This is the primary function called by bot_cycle every 4 hours.
    """
    needed = max(config.ATR_PERIOD, config.EMA_PERIOD) + 20
    df    = fetch_ohlcv(symbol, timeframe="4h", limit=needed)
    price = get_current_price(symbol)
    atr   = compute_atr(df, period=config.ATR_PERIOD)
    ema   = compute_ema(df, period=config.EMA_PERIOD)
    logger.info(
        f"{symbol} — Price: {price:.2f} | "
        f"ATR({config.ATR_PERIOD})/4h: {atr:.2f} | "
        f"EMA({config.EMA_PERIOD})/4h: {ema:.2f}"
    )
    return price, atr, ema
