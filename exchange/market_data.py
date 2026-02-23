"""
Market Data — fetches OHLCV price data and computes ATR via ccxt.
"""
import os
import ccxt
import numpy as np
import pandas as pd
from loguru import logger

import config


def _get_exchange() -> ccxt.Exchange:
    exchange_cls = getattr(ccxt, config.EXCHANGE_NAME)

    if config.EXCHANGE_NAME == "hyperliquid":
        # Hyperliquid is a DEX — authenticates via EVM wallet private key,
        # not a traditional API key/secret pair.
        exchange = exchange_cls({
            "walletAddress": os.getenv("HYPERLIQUID_WALLET_ADDRESS", ""),
            "privateKey":    os.getenv("HYPERLIQUID_PRIVATE_KEY", ""),
            "enableRateLimit": True,
        })
    else:
        exchange = exchange_cls({
            "apiKey": os.getenv(f"{config.EXCHANGE_NAME.upper()}_API_KEY", ""),
            "secret": os.getenv(f"{config.EXCHANGE_NAME.upper()}_SECRET", ""),
            "options": {"defaultType": config.MARKET_TYPE},
            "enableRateLimit": True,
        })

    return exchange


_exchange: ccxt.Exchange = None


def get_exchange() -> ccxt.Exchange:
    global _exchange
    if _exchange is None:
        _exchange = _get_exchange()
    return _exchange


def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    """
    Fetch OHLCV candles from the exchange.
    
    Returns a DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        ex = get_exchange()
        raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def get_current_price(symbol: str) -> float:
    """Fetch latest ticker price."""
    try:
        ex = get_exchange()
        ticker = ex.fetch_ticker(symbol)
        return float(ticker["last"])
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol}: {e}")
        return 0.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Average True Range (ATR) over the last `period` candles.
    
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = rolling mean of True Range
    """
    if df.empty or len(df) < period + 1:
        logger.warning("Not enough data for ATR calculation")
        return 0.0

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


def compute_ema(df: pd.DataFrame, period: int) -> float:
    """
    Compute EMA of the close price over the last `period` candles.
    Returns the latest EMA value, or 0.0 if insufficient data.
    """
    if df.empty or len(df) < period:
        logger.warning(f"Not enough data for EMA({period}) — need {period}, got {len(df)}")
        return 0.0
    ema_series = df["close"].ewm(span=period, adjust=False).mean()
    return float(ema_series.iloc[-1])


def get_account_balance() -> float:
    """
    Fetch total USDT balance from the exchange.

    Returns the total equity (unrealised P&L included) in USDT.
    Falls back to config.CAPITAL_USDT if the fetch fails, so the bot
    keeps running even without API credentials (paper trading mode).
    """
    try:
        ex = get_exchange()
        balance = ex.fetch_balance()

        # Hyperliquid returns equity under 'total' for the margin account
        if config.EXCHANGE_NAME == "hyperliquid":
            usdt = (
                balance.get("total", {}).get("USDC")
                or balance.get("total", {}).get("USDT")
                or balance.get("info", {}).get("marginSummary", {}).get("accountValue")
            )
        else:
            # Binance / Bybit futures: look for USDT in total
            usdt = (
                balance.get("total", {}).get("USDT")
                or balance.get("USDT", {}).get("total")
            )

        val = float(usdt) if usdt is not None else 0.0
        if val > 0:
            logger.info(f"Account balance: ${val:,.2f} USDT")
            return val
        else:
            raise ValueError(f"Unexpected balance structure: {balance}")

    except Exception as e:
        logger.warning(
            f"Could not fetch account balance ({e}) — "
            f"falling back to CAPITAL_USDT=${config.CAPITAL_USDT:,.2f}"
        )
        return config.CAPITAL_USDT


def get_effective_capital() -> float:
    """
    Returns the capital base to use for position sizing this cycle.

    effective_capital = account_balance * CAPITAL_PCT

    CAPITAL_PCT (default 0.60) keeps 40% of the balance as a free
    liquidation buffer.  Position sizes grow and shrink automatically
    with the account — true proportional compounding.
    """
    balance  = get_account_balance()
    capital  = balance * config.CAPITAL_PCT
    logger.info(
        f"Effective capital: ${capital:,.2f}  "
        f"({config.CAPITAL_PCT*100:.0f}% of ${balance:,.2f} balance)"
    )
    return capital


def get_price_and_atr(symbol: str) -> tuple[float, float]:
    """
    Fetch current price and ATR using 4h candles.
    4h candles match the live trading cadence and produce stops
    wide enough to survive intrabar noise (backtested optimal).
    """
    # Need ATR_PERIOD + a few extra candles for warm-up
    df = fetch_ohlcv(symbol, timeframe="4h", limit=config.ATR_PERIOD + 10)
    price = get_current_price(symbol)
    atr = compute_atr(df, period=config.ATR_PERIOD)
    logger.info(f"{symbol} — Price: {price:.2f} | ATR(14) on 4h: {atr:.2f}")
    return price, atr


def get_price_atr_ema(symbol: str) -> tuple[float, float, float]:
    """
    Returns (current_price, atr, ema_value) all computed on 4h candles.

    EMA period is read from config.EMA_PERIOD (default 20).
    At 4h cadence, EMA(20) = ~3.3-day momentum filter.

    This is the primary function used by the live bot loop.
    """
    # Fetch enough candles for both ATR(14) and EMA(20) warm-up
    needed = max(config.ATR_PERIOD, config.EMA_PERIOD) + 20
    df = fetch_ohlcv(symbol, timeframe="4h", limit=needed)
    price = get_current_price(symbol)
    atr   = compute_atr(df, period=config.ATR_PERIOD)
    ema   = compute_ema(df, period=config.EMA_PERIOD)
    logger.info(
        f"{symbol} — Price: {price:.2f} | "
        f"ATR(14)/4h: {atr:.2f} | "
        f"EMA({config.EMA_PERIOD})/4h: {ema:.2f}"
    )
    return price, atr, ema
