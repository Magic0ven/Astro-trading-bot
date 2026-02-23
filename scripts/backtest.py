"""
Backtest — replay historical planetary positions against historical prices.

Usage:
    python scripts/backtest.py --asset BTC --start 2024-01-01 --end 2024-12-31
    python scripts/backtest.py --asset BTC --start 2023-01-01 --interval 4

The bot simulates one signal per `interval` hours over the date range.
Results are printed to console and saved to logs/backtest_{asset}_{date}.csv
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.astro_engine import datetime_to_jd, get_sky_state, compute_composite_score
from core.numerology import full_numerology_report, numerology_multiplier
from core.score_history import ScoreHistory
from core.signal_engine import _system_signal, dual_system_gate

# Suppress info logs during backtest
logger.remove()
logger.add(sys.stderr, level="WARNING")


def fetch_historical_prices(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch hourly OHLCV from Binance public API for the backtest period.

    Note: this uses Binance as a historical DATA SOURCE only — no API key needed.
    The live trading engine is Hyperliquid-only (exchange/market_data.py).
    Binance is used here because Hyperliquid launched in late 2023 and lacks
    the 2022–2023 data required for multi-year backtests.
    """
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_candles = []
    while since_ms < end_ms:
        candles = exchange.fetch_ohlcv(symbol, "1h", since=since_ms, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 3600000  # advance 1 hour
        print(f"  Fetched up to {pd.to_datetime(since_ms, unit='ms', utc=True).date()}", end="\r")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def get_candle_at(df: pd.DataFrame, dt: datetime) -> dict:
    """Return OHLC values for the candle closest to dt."""
    if df.empty:
        return {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    idx = df.index.get_indexer([dt], method="nearest")[0]
    row = df.iloc[idx]
    return {
        "open":  float(row["open"]),
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "close": float(row["close"]),
    }


def get_price_at(df: pd.DataFrame, dt: datetime) -> float:
    """Get the closest close price to a given datetime."""
    return get_candle_at(df, dt)["close"]


def run_backtest(asset_key: str, start: datetime, end: datetime, interval_hours: int = 1):
    dna_path = Path(__file__).parent.parent / "assets_dna.json"
    with open(dna_path) as f:
        all_assets = json.load(f)

    if asset_key not in all_assets:
        print(f"Asset {asset_key} not found.")
        sys.exit(1)

    dna = all_assets[asset_key]
    if not dna.get("natal_western") or not dna.get("natal_vedic"):
        print("ERROR: Natal positions not computed. Run scripts/calculate_natal.py first.")
        sys.exit(1)

    symbol = dna["symbol"]
    # Use spot_symbol for Binance historical price fetching.
    # The main symbol may be exchange-specific (e.g. Hyperliquid BTC/USDC:USDC)
    # which Binance does not recognise.
    price_symbol = dna.get("spot_symbol", symbol)

    from dateutil.parser import parse
    genesis_date = parse(dna["genesis_datetime"]).date()

    print(f"\nFetching historical prices for {price_symbol} (spot)...")
    price_df = fetch_historical_prices(price_symbol, start, end)
    print(f"  Got {len(price_df)} hourly candles.")

    score_history = ScoreHistory(window=config.SCORE_HISTORY_WINDOW)
    results = []
    current = start

    print(f"\nRunning backtest ({start.date()} → {end.date()}, every {interval_hours}h)...")

    while current <= end:
        jd = datetime_to_jd(current)
        sky = get_sky_state(jd)
        today = current.date()

        num_report = full_numerology_report(dna.get("name", ""), genesis_date, today)
        num_mult = num_report["multiplier"]
        nk_mult = sky["nakshatra_multiplier"]

        w_score = compute_composite_score(
            sky["western"], dna["natal_western"]
        ) * num_mult

        v_score = compute_composite_score(
            sky["vedic"], dna["natal_vedic"],
            dasha_weights=dna.get("dasha_weights"),
            nakshatra_mult=nk_mult,
        ) * num_mult

        score_history.push(w_score, v_score)
        hist = score_history.summary()

        candle = get_candle_at(price_df, current)
        action = "COLLECTING_DATA"

        if score_history.is_ready():
            ws = _system_signal(w_score, hist["western"]["medium"], hist["western"]["slope"])
            vs = _system_signal(v_score, hist["vedic"]["medium"], hist["vedic"]["slope"])
            action = dual_system_gate(ws, vs)

        results.append({
            "timestamp":    current.isoformat(),
            "open":         candle["open"],
            "high":         candle["high"],
            "low":          candle["low"],
            "price":        candle["close"],   # keep "price" = close for backward compat
            "action":       action,
            "western_score":  round(w_score, 4),
            "vedic_score":    round(v_score, 4),
            "western_slope":  round(hist["western"]["slope"] or 0, 4),
            "vedic_slope":    round(hist["vedic"]["slope"] or 0, 4),
            "resonance_day":  num_report["resonance_match"],
            "nakshatra":      sky["nakshatra"],
        })

        current += timedelta(hours=interval_hours)

    # ── Analysis ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df["next_price"] = df["price"].shift(-interval_hours)
    df["price_change_pct"] = (df["next_price"] - df["price"]) / df["price"] * 100

    trade_signals = df[df["action"].isin(["STRONG_BUY", "WEAK_BUY", "STRONG_SELL", "WEAK_SELL"])]
    buy_signals = df[df["action"].isin(["STRONG_BUY", "WEAK_BUY"])]
    sell_signals = df[df["action"].isin(["STRONG_SELL", "WEAK_SELL"])]

    buy_accuracy = (buy_signals["price_change_pct"] > 0).mean() if len(buy_signals) > 0 else 0
    sell_accuracy = (sell_signals["price_change_pct"] < 0).mean() if len(sell_signals) > 0 else 0
    no_trade_count = (df["action"] == "NO_TRADE").sum()

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS — {asset_key} ({start.date()} to {end.date()})")
    print(f"{'='*60}")
    print(f"Total cycles:          {len(df)}")
    print(f"Trade signals:         {len(trade_signals)}")
    print(f"  BUY signals:         {len(buy_signals)}")
    print(f"  SELL signals:        {len(sell_signals)}")
    print(f"  NO_TRADE:            {no_trade_count}")
    print(f"BUY accuracy:          {buy_accuracy:.1%}")
    print(f"SELL accuracy:         {sell_accuracy:.1%}")
    print(f"Strong signals (both): {(df['action'].isin(['STRONG_BUY','STRONG_SELL'])).sum()}")
    print(f"Resonance days:        {df['resonance_day'].sum()}")

    resonance_buys = buy_signals[buy_signals["resonance_day"] == True]
    if len(resonance_buys) > 0:
        res_accuracy = (resonance_buys["price_change_pct"] > 0).mean()
        print(f"Resonance day BUY acc: {res_accuracy:.1%} ({len(resonance_buys)} signals)")

    # Save
    out_path = Path(f"logs/backtest_{asset_key}_{start.date()}_{end.date()}.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Astro-Bot backtester")
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--interval", type=int, default=1, help="Hours between signals")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    run_backtest(args.asset, start, end, args.interval)


if __name__ == "__main__":
    main()
