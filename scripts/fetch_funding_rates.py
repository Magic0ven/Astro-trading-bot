"""
Fetch historical funding rates from Binance Futures and save to CSV.

Usage:
    python scripts/fetch_funding_rates.py --symbol BTCUSDT --start 2022-01-01 --end 2025-12-31

Output:
    logs/funding_BTCUSDT_2022-01-01_2025-12-31.csv
    Columns: timestamp (UTC), funding_rate (float, e.g. 0.0001 = 0.01%)

Binance pays/charges funding every 8 hours: 00:00, 08:00, 16:00 UTC.
If you are long and funding_rate > 0, you PAY. If you are short, you RECEIVE.
The bot's SELL signals are shorts, so positive funding is income for shorts.
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding_rates(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    since_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)
    all_rows = []

    while since_ms < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": since_ms,
            "endTime":   min(since_ms + 1000 * 8 * 3600 * 1000, end_ms),
            "limit":     1000,
        }
        resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        for r in rows:
            all_rows.append({
                "timestamp":    pd.to_datetime(r["fundingTime"], unit="ms", utc=True),
                "funding_rate": float(r["fundingRate"]),
            })

        since_ms = rows[-1]["fundingTime"] + 1
        ts_str   = pd.to_datetime(since_ms, unit="ms", utc=True).date()
        print(f"  Fetched up to {ts_str}", end="\r")
        time.sleep(0.2)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start",  default="2022-01-01")
    parser.add_argument("--end",    default="2025-12-31")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print(f"\nFetching funding rates for {args.symbol} ({args.start} → {args.end})...")
    df = fetch_funding_rates(args.symbol, start, end)

    if df.empty:
        print("No data returned.")
        return

    out = Path(f"logs/funding_{args.symbol}_{args.start}_{args.end}.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    ann_avg = df["funding_rate"].mean() * 3 * 365 * 100  # 3 payments/day * 365
    print(f"\n  Records:          {len(df)}")
    print(f"  Average rate:     {df['funding_rate'].mean()*100:.4f}% per 8h")
    print(f"  Annualised avg:   {ann_avg:.1f}%  (longs pay this to shorts when positive)")
    print(f"  Max rate:         {df['funding_rate'].max()*100:.4f}%")
    print(f"  Min rate:         {df['funding_rate'].min()*100:.4f}%")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
