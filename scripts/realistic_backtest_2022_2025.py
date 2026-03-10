#!/usr/bin/env python3
"""
Run the engine's realistic backtest and plot on 2022–2025 data.

1. Realistic backtest: same as final_backtest.py — 4h candles, intrabar SL/TP,
   fees, slippage, funding. Reads logs/backtest_BTC_YYYY-01-01_YYYY-12-31.csv.
2. Plot: 4-panel chart (price+EMA+trades, equity curve, drawdown, P&L bars, rolling WR)
   saved to logs/chart_2022_2025.png.

Prerequisites:
   - Backtest CSVs: run for each year if missing:
       python scripts/backtest.py --asset BTC --start 2022-01-01 --end 2022-12-31
       (repeat for 2023, 2024, 2025)
   - Funding CSV: logs/funding_BTCUSDT_2022-01-01_2025-12-31.csv
     (optional; simulation works without it)

Usage:
   python scripts/realistic_backtest_2022_2025.py              # backtest + plot
   python scripts/realistic_backtest_2022_2025.py --backtest-only   # tables only
   python scripts/realistic_backtest_2022_2025.py --plot-only      # chart only
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
BACKTEST_FILES = [
    str(LOG_DIR / "backtest_BTC_2022-01-01_2022-12-31.csv"),
    str(LOG_DIR / "backtest_BTC_2023-01-01_2023-12-31.csv"),
    str(LOG_DIR / "backtest_BTC_2024-01-01_2024-12-31.csv"),
    str(LOG_DIR / "backtest_BTC_2025-01-01_2025-12-31.csv"),
]
FUNDING_FILE = str(LOG_DIR / "funding_BTCUSDT_2022-01-01_2025-12-31.csv")
CHART_OUT = LOG_DIR / "chart_2022_2025.png"


def run_backtest_tables():
    """Run final_backtest.py --cadence 4 to print the 4-year tables."""
    cmd = [sys.executable, str(ROOT / "scripts" / "final_backtest.py"), "--cadence", "4"]
    subprocess.run(cmd, cwd=str(ROOT), check=False)


def run_plot():
    """Run plot_backtest with 2022-2025 files and save chart."""
    sys.path.insert(0, str(ROOT))
    from scripts.plot_backtest import run_simulations, make_chart

    missing = [f for f in BACKTEST_FILES if not Path(f).exists()]
    if missing:
        print("Missing backtest CSVs. Generate them first, e.g.:")
        for y in (2022, 2023, 2024, 2025):
            print(f"  python scripts/backtest.py --asset BTC --start {y}-01-01 --end {y}-12-31")
        print("\nSkipping plot.")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("Running realistic simulation (2022–2025, 4h cadence)...")
    summary = run_simulations(BACKTEST_FILES, FUNDING_FILE, cadence=4)
    print(f"  Trades: {summary['trades']}  Win rate: {summary['wr']:.1f}%  "
          f"Total P&L: {summary['total']:+,.0f}  End equity: {summary['end_equity']:,.0f}")
    make_chart(summary, "2022–2025 Full Backtest", CHART_OUT)
    print(f"Chart saved → {CHART_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Realistic backtest + plot for 2022–2025")
    parser.add_argument("--backtest-only", action="store_true", help="Only run final_backtest (tables)")
    parser.add_argument("--plot-only", action="store_true", help="Only generate chart")
    args = parser.parse_args()

    do_backtest = not args.plot_only
    do_plot = not args.backtest_only

    if do_backtest:
        run_backtest_tables()
    if do_plot:
        if do_backtest:
            print()
        run_plot()


if __name__ == "__main__":
    main()
