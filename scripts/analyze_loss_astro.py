#!/usr/bin/env python3
"""
Analyze which trades lost and what astro signals were present at entry.

Uses 4h cadence realistic simulation. Run after final_backtest (uses same simulate()).
Outputs:
  - Loss breakdown by result type (STOP, TIMEOUT, etc.)
  - Loss breakdown by nakshatra, resonance_day, moon phase, retrograde events
  - Comparison: astro features on losing vs winning trades

Usage:
  python scripts/analyze_loss_astro.py                    # 2026 YTD + 2022-2025
  python scripts/analyze_loss_astro.py --2026-only       # 2026 only
  python scripts/analyze_loss_astro.py --years 2022 2024 # specific years
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.final_backtest import simulate, load_funding_rates

LOG_DIR = ROOT / "logs"
FUNDING_CSV = LOG_DIR / "funding_BTCUSDT_2022-01-01_2025-12-31.csv"


def _collect_trades_with_astro(
    files: list[Path],
    cadence: int = 4,
    mercury_rx_block: bool = False,
    saturn_rx_block: bool = False,
) -> list[dict]:
    """Collect trades. Set mercury_rx_block=False, saturn_rx_block=False to include RX periods for long/short analysis."""
    funding = load_funding_rates()
    all_trades = []
    for path in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        stats = simulate(
            df, funding, cadence=cadence, use_intrabar=True,
            use_fees=True, use_slippage=True, use_funding=True,
            mercury_rx_block=mercury_rx_block, saturn_rx_block=saturn_rx_block,
        )
        for t in stats.get("trade_list", []):
            t["_file"] = path.name
        all_trades.extend(stats.get("trade_list", []))
    return all_trades


def _safe(v, default=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def run_analysis(trades: list[dict], label: str) -> None:
    if not trades:
        print(f"[{label}] No trades.\n")
        return

    losses = [t for t in trades if t["pnl"] <= 0]
    wins = [t for t in trades if t["pnl"] > 0]
    n_loss = len(losses)
    n_win = len(wins)

    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    print(f"  Total trades: {len(trades)}  |  Wins: {n_win}  |  Losses: {n_loss}")
    print()

    # Loss by result type
    loss_by_result = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in losses:
        r = t.get("result", "?")
        loss_by_result[r]["count"] += 1
        loss_by_result[r]["pnl"] += t["pnl"]
    print("  Losses by exit type:")
    for r in sorted(loss_by_result.keys()):
        rec = loss_by_result[r]
        print(f"    {r:20}  count: {rec['count']:4}   total P&L: {rec['pnl']:+,.0f}")
    print()

    # Astro at entry: losing vs winning
    astro_keys = [
        "nakshatra", "resonance_day", "western_score", "vedic_score",
        "western_slope", "vedic_slope",
        "full_moon_active", "new_moon_active",
        "jupiter_uranus_active", "saturn_pluto_active",
        "mercury_retrograde_active", "saturn_retrograde_active",
        "moon_phase_deg",
    ]

    def summarize_by_key(trade_list: list[dict], key: str) -> dict:
        out = defaultdict(int)
        for t in trade_list:
            v = t.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out["(missing)"] += 1
            elif isinstance(v, bool):
                out["True" if v else "False"] += 1
            else:
                out[str(v)] += 1
        return dict(out)

    print("  --- Nakshatra on LOSING trades (top) ---")
    naks_loss = summarize_by_key(losses, "nakshatra")
    for nk, cnt in sorted(naks_loss.items(), key=lambda x: -x[1])[:12]:
        pct = cnt / n_loss * 100 if n_loss else 0
        print(f"    {nk:20}  {cnt:4}  ({pct:.1f}%)")
    print()

    print("  --- Nakshatra on WINNING trades (top) ---")
    naks_win = summarize_by_key(wins, "nakshatra")
    for nk, cnt in sorted(naks_win.items(), key=lambda x: -x[1])[:12]:
        pct = cnt / n_win * 100 if n_win else 0
        print(f"    {nk:20}  {cnt:4}  ({pct:.1f}%)")
    print()

    print("  --- Resonance day at entry ---")
    res_loss = sum(1 for t in losses if t.get("resonance_day") in (True, "True", 1))
    res_win = sum(1 for t in wins if t.get("resonance_day") in (True, "True", 1))
    print(f"    Losing trades on resonance day:  {res_loss}/{n_loss}  ({res_loss/n_loss*100:.1f}%)" if n_loss else "    (no losses)")
    print(f"    Winning trades on resonance day: {res_win}/{n_win}  ({res_win/n_win*100:.1f}%)" if n_win else "    (no wins)")
    print()

    print("  --- Moon phase (full_moon / new_moon at entry) ---")
    full_loss = sum(1 for t in losses if t.get("full_moon_active") in (True, "True", 1))
    new_loss = sum(1 for t in losses if t.get("new_moon_active") in (True, "True", 1))
    full_win = sum(1 for t in wins if t.get("full_moon_active") in (True, "True", 1))
    new_win = sum(1 for t in wins if t.get("new_moon_active") in (True, "True", 1))
    print(f"    Losing  | full_moon: {full_loss:4}  new_moon: {new_loss:4}")
    print(f"    Winning | full_moon: {full_win:4}  new_moon: {new_win:4}")
    print()

    print("  --- Retrograde at entry (Mercury / Saturn) ---")
    mer_loss = sum(1 for t in losses if t.get("mercury_retrograde_active") in (True, "True", 1))
    sat_loss = sum(1 for t in losses if t.get("saturn_retrograde_active") in (True, "True", 1))
    mer_win = sum(1 for t in wins if t.get("mercury_retrograde_active") in (True, "True", 1))
    sat_win = sum(1 for t in wins if t.get("saturn_retrograde_active") in (True, "True", 1))
    print(f"    Losing  | Mercury RX: {mer_loss:4}  Saturn RX: {sat_loss:4}")
    print(f"    Winning | Mercury RX: {mer_win:4}  Saturn RX: {sat_win:4}")
    print()

    # Long vs Short during Mercury RX and Saturn RX (which side loses / wins)
    def _rx_stats(trade_list: list[dict], rx_key: str, rx_name: str) -> None:
        in_rx = [t for t in trade_list if t.get(rx_key) in (True, "True", 1)]
        if not in_rx:
            print(f"    {rx_name}: no trades during this period.")
            return
        longs = [t for t in in_rx if t.get("side") == "BUY"]
        shorts = [t for t in in_rx if t.get("side") == "SELL"]
        def _side_stats(side_name: str, side_trades: list) -> None:
            if not side_trades:
                print(f"      {side_name:6}  —  no trades")
                return
            w = [t for t in side_trades if t["pnl"] > 0]
            l = [t for t in side_trades if t["pnl"] <= 0]
            tot_pnl = sum(t["pnl"] for t in side_trades)
            wr = len(w) / len(side_trades) * 100
            print(f"      {side_name:6}  {len(side_trades):4} trades  |  Wins: {len(w):3}  Losses: {len(l):3}  |  Win%: {wr:5.1f}%  |  P&L: ${tot_pnl:+,.0f}")
        print(f"    {rx_name} ({len(in_rx)} trades total):")
        _side_stats("LONG", longs)
        _side_stats("SHORT", shorts)
        print()

    print("  --- LONG vs SHORT during Mercury RX (who loses / wins) ---")
    _rx_stats(trades, "mercury_retrograde_active", "Mercury retrograde")
    print("  --- LONG vs SHORT during Saturn RX (who loses / wins) ---")
    _rx_stats(trades, "saturn_retrograde_active", "Saturn retrograde")
    print()

    print("  --- Conjunctions at entry (Jupiter–Uranus / Saturn–Pluto) ---")
    ju_loss = sum(1 for t in losses if t.get("jupiter_uranus_active") in (True, "True", 1))
    sp_loss = sum(1 for t in losses if t.get("saturn_pluto_active") in (True, "True", 1))
    ju_win = sum(1 for t in wins if t.get("jupiter_uranus_active") in (True, "True", 1))
    sp_win = sum(1 for t in wins if t.get("saturn_pluto_active") in (True, "True", 1))
    print(f"    Losing  | Jupiter–Uranus: {ju_loss:4}  Saturn–Pluto: {sp_loss:4}")
    print(f"    Winning | Jupiter–Uranus: {ju_win:4}  Saturn–Pluto: {sp_win:4}")
    print()

    # Sample of worst losing trades with full astro
    print("  --- Sample of losing trades (entry time, side, result, P&L, astro) ---")
    by_pnl = sorted(losses, key=lambda t: t["pnl"])[:15]
    for t in by_pnl:
        nk = _safe(t.get("nakshatra"), "?")
        res = "resonance" if t.get("resonance_day") in (True, "True", 1) else ""
        fm = "full_moon" if t.get("full_moon_active") in (True, "True", 1) else ""
        nm = "new_moon" if t.get("new_moon_active") in (True, "True", 1) else ""
        mr = "Mercury_RX" if t.get("mercury_retrograde_active") in (True, "True", 1) else ""
        tags = "  ".join(filter(None, [res, fm, nm, mr]))
        print(f"    {t.get('open_ts', '?')}  {t.get('side', '?'):4}  {t.get('result', '?'):12}  P&L: {t['pnl']:+,.0f}  |  Nakshatra: {nk}  {tags}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Analyze losing trades and astro signals")
    ap.add_argument("--2026-only", dest="only_2026", action="store_true", help="Only 2026 YTD")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="Specific years e.g. 2022 2024")
    ap.add_argument("--cadence", type=int, default=4, help="Candle cadence (default 4h)")
    args = ap.parse_args()

    # Collect with RX block OFF so we have trades during Mercury/Saturn RX to analyze long vs short
    no_rx_block = dict(mercury_rx_block=False, saturn_rx_block=False)

    if args.years:
        files = [LOG_DIR / f"backtest_BTC_{y}-01-01_{y}-12-31.csv" for y in args.years]
        files = [f for f in files if f.exists()]
        if not files:
            print("No backtest CSVs found for given years.")
            return
        trades = _collect_trades_with_astro(files, cadence=args.cadence, **no_rx_block)
        run_analysis(trades, f"Years {args.years} ({args.cadence}h)")
        return

    # 2026 YTD
    f2026 = LOG_DIR / "backtest_BTC_2026-01-01_2026-12-31.csv"
    if f2026.exists():
        trades_2026 = _collect_trades_with_astro([f2026], cadence=args.cadence, **no_rx_block)
        run_analysis(trades_2026, "2026 YTD (4h)")
    else:
        print("2026 backtest CSV not found. Run: python scripts/backtest.py --asset BTC --start 2026-01-01 --end 2026-12-31 --interval 1")

    if args.only_2026:
        return

    # 2022–2025
    files_4y = [
        LOG_DIR / "backtest_BTC_2022-01-01_2022-12-31.csv",
        LOG_DIR / "backtest_BTC_2023-01-01_2023-12-31.csv",
        LOG_DIR / "backtest_BTC_2024-01-01_2024-12-31.csv",
        LOG_DIR / "backtest_BTC_2025-01-01_2025-12-31.csv",
    ]
    existing = [f for f in files_4y if f.exists()]
    if existing:
        trades_4y = _collect_trades_with_astro(existing, cadence=args.cadence, **no_rx_block)
        run_analysis(trades_4y, "2022–2025 (4h)")


if __name__ == "__main__":
    main()
