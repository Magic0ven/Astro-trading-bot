#!/usr/bin/env python3
"""
Backtest chart generator.

Produces a 4-panel chart:
  1. BTC price + EMA(20) with trade entry markers (▲/▼) and outcome badges
  2. Equity curve with underwater drawdown shading
  3. Per-trade P&L bars (green = win, red = loss, orange = timeout)
  4. Rolling 10-trade win rate

Usage:
    python scripts/plot_backtest.py
    python scripts/plot_backtest.py \\
        --file  logs/backtest_BTC_2026-01-01_2026-02-23.csv \\
        --funding-file logs/funding_BTCUSDT_2026-01-01_2026-02-23.csv \\
        --title "2026 YTD" \\
        --out   logs/chart_2026_ytd.png

    # Multi-year (separate simulation per file, combined equity curve)
    python scripts/plot_backtest.py \\
        --file  logs/backtest_BTC_2022-01-01_2022-12-31.csv \\
                logs/backtest_BTC_2023-01-01_2023-12-31.csv \\
                logs/backtest_BTC_2024-01-01_2024-12-31.csv \\
                logs/backtest_BTC_2025-01-01_2025-12-31.csv \\
        --funding-file logs/funding_BTCUSDT_2022-01-01_2025-12-31.csv \\
        --title "2022-2025 Full Backtest" \\
        --out   logs/chart_2022_2025.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; we save to file then open it

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Allow importing from project root ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.final_backtest import (
    simulate,
    load_funding_rates,
    EMA_PERIOD,
    EMA_FILTER,
    NK_FILTER,
    RISK_PCT,
    CAPITAL,
)

# ── Colour palette (dark theme) ────────────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
CYAN     = "#39d353"


# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["price"]
    return df


def build_equity_frames(equity_curve: list) -> pd.DataFrame:
    """Convert raw equity_curve list → tidy DataFrame with drawdown column."""
    ec = pd.DataFrame(equity_curve)
    ec["ts"] = pd.to_datetime(ec["ts"], utc=True)
    ec = ec.set_index("ts").sort_index()
    ec["peak"]     = ec["equity"].cummax()
    ec["drawdown"] = ec["equity"] - ec["peak"]          # ≤ 0
    ec["dd_pct"]   = ec["drawdown"] / ec["peak"] * 100  # ≤ 0
    return ec


def run_simulations(files: list[str], funding_file: str | None,
                    cadence: int = 4) -> tuple[dict, pd.Series]:
    """Run simulate() on one or more CSV files, chaining equity."""
    # Load funding once (covers all years if multi-year file)
    if funding_file and Path(funding_file).exists():
        import scripts.final_backtest as fb
        original = fb.FUNDING_CSV
        fb.FUNDING_CSV = Path(funding_file)
        funding = load_funding_rates()
        fb.FUNDING_CSV = original
    else:
        funding = load_funding_rates()

    all_trades    = []
    all_eq_curve  = []
    all_price     = []
    capital       = CAPITAL

    for fpath in files:
        df  = load_csv(fpath)
        res = simulate(
            df, funding,
            cadence          = cadence,
            starting_capital = capital,
        )
        all_trades   += res["trade_list"]
        all_eq_curve += res["equity_curve"]
        capital       = res["end_equity"]     # chain equity across years

        ps = res["price_series"].copy()
        ps["timestamp"] = pd.to_datetime(ps["timestamp"], utc=True)
        all_price.append(ps)

    price_df = pd.concat(all_price, ignore_index=True).set_index("timestamp").sort_index()

    summary = {
        "trade_list":  all_trades,
        "equity_curve": all_eq_curve,
        "price_series": price_df,
        "starting_capital": CAPITAL,
        "end_equity": capital,
    }
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    summary["trades"] = len(all_trades)
    summary["wins"]   = len(wins)
    summary["losses"] = len(losses)
    summary["wr"]     = len(wins) / len(all_trades) * 100 if all_trades else 0
    summary["total"]  = sum(t["pnl"] for t in all_trades)
    summary["pnl_pct"] = summary["total"] / CAPITAL * 100
    eq, pk, dd = CAPITAL, CAPITAL, 0.0
    for t in all_trades:
        eq += t["pnl"]
        pk  = max(pk, eq)
        dd  = max(dd, pk - eq)
    summary["max_dd"]   = dd
    summary["dd_pct"]   = dd / CAPITAL * 100
    return summary


# ─────────────────────────────────────────────────────────────────────────────
def make_chart(summary: dict, title: str, out_path: Path):
    trades      = summary["trade_list"]
    ec_df       = build_equity_frames(summary["equity_curve"])
    price_df    = summary["price_series"]
    start_cap   = summary["starting_capital"]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[3, 1.5, 1, 1],
        hspace=0.08,
        wspace=0.04,
    )

    ax_price  = fig.add_subplot(gs[0, :])    # top — full width
    ax_eq     = fig.add_subplot(gs[1, :])    # equity curve
    ax_dd     = fig.add_subplot(gs[2, :])    # drawdown %
    ax_pnl    = fig.add_subplot(gs[3, 0])    # per-trade P&L
    ax_wr     = fig.add_subplot(gs[3, 1])    # rolling win rate

    axes = [ax_price, ax_eq, ax_dd, ax_pnl, ax_wr]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)

    # ── [1] Price + EMA ────────────────────────────────────────────────────────
    ts_idx = price_df.index
    ax_price.plot(ts_idx, price_df["price"], color=BLUE,   linewidth=1.0, label="BTC/USDT", zorder=2)
    ax_price.plot(ts_idx, price_df["ema"],   color=ORANGE, linewidth=1.2, linestyle="--",
                  label=f"EMA({EMA_PERIOD})", alpha=0.9, zorder=2)

    # Trade entry markers
    for t in trades:
        try:
            ts = pd.to_datetime(t["open_ts"], utc=True)
        except Exception:
            continue
        entry   = t["entry"]
        is_buy  = t["side"] == "BUY"
        result  = t["result"]

        # Entry arrow
        marker = "^" if is_buy else "v"
        color  = GREEN if is_buy else RED
        offset = -entry * 0.015 if is_buy else entry * 0.015
        ax_price.scatter(ts, entry + offset, marker=marker, color=color,
                         s=60, zorder=5, linewidths=0)

        # Outcome dot on the entry bar
        dot_color = GREEN if result == "WIN" else (ORANGE if result == "TIMEOUT" else RED)
        ax_price.scatter(ts, entry, marker="o", color=dot_color,
                         s=18, zorder=6, linewidths=0, alpha=0.75)

    ax_price.set_ylabel("Price (USDT)", fontsize=9)
    ax_price.legend(loc="upper left", fontsize=8, facecolor=SURFACE, edgecolor=BORDER,
                    labelcolor=TEXT)
    ax_price.set_title(f"  {title} — Astro-Bot Backtest  ", fontsize=14,
                       fontweight="bold", color=TEXT, loc="left", pad=10)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_price.tick_params(labelbottom=False)

    # ── [2] Equity curve ───────────────────────────────────────────────────────
    ax_eq.plot(ec_df.index, ec_df["equity"], color=CYAN, linewidth=1.4, zorder=3)
    ax_eq.axhline(start_cap, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.7,
                  label=f"Start capital: {start_cap:,.0f}")
    ax_eq.fill_between(ec_df.index, ec_df["equity"], start_cap,
                       where=(ec_df["equity"] >= start_cap),
                       color=GREEN, alpha=0.12, zorder=2)
    ax_eq.fill_between(ec_df.index, ec_df["equity"], start_cap,
                       where=(ec_df["equity"] < start_cap),
                       color=RED, alpha=0.18, zorder=2)
    ax_eq.set_ylabel("Equity (USDT)", fontsize=9)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_eq.legend(loc="upper left", fontsize=8, facecolor=SURFACE, edgecolor=BORDER,
                 labelcolor=TEXT)
    ax_eq.tick_params(labelbottom=False)

    # ── [3] Drawdown % ─────────────────────────────────────────────────────────
    ax_dd.fill_between(ec_df.index, ec_df["dd_pct"], 0,
                       color=RED, alpha=0.4, zorder=2)
    ax_dd.plot(ec_df.index, ec_df["dd_pct"], color=RED, linewidth=0.8, zorder=3)
    ax_dd.axhline(0, color=BORDER, linewidth=0.6)
    ax_dd.set_ylabel("Drawdown %", fontsize=9)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_dd.tick_params(axis="x", labelsize=8, colors=MUTED)

    # ── [4] Per-trade P&L bars ─────────────────────────────────────────────────
    if trades:
        xs      = list(range(len(trades)))
        pnls    = [t["pnl"] for t in trades]
        results = [t["result"] for t in trades]
        colors  = [GREEN if p > 0 else (ORANGE if r == "TIMEOUT" else RED)
                   for p, r in zip(pnls, results)]
        ax_pnl.bar(xs, pnls, color=colors, width=0.8, zorder=3)
        ax_pnl.axhline(0, color=MUTED, linewidth=0.6)
        ax_pnl.set_xlabel("Trade #", fontsize=8)
        ax_pnl.set_ylabel("P&L (USDT)", fontsize=9)
        ax_pnl.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}"))

        patches = [
            mpatches.Patch(color=GREEN,  label="Win"),
            mpatches.Patch(color=RED,    label="Loss / SL"),
            mpatches.Patch(color=ORANGE, label="Timeout"),
        ]
        ax_pnl.legend(handles=patches, fontsize=7, facecolor=SURFACE,
                      edgecolor=BORDER, labelcolor=TEXT, loc="lower right")

    # ── [5] Rolling 10-trade win rate ──────────────────────────────────────────
    if len(trades) >= 2:
        win_flags = [1 if t["pnl"] > 0 else 0 for t in trades]
        roll      = pd.Series(win_flags).rolling(min(10, len(trades)), min_periods=1).mean() * 100
        ax_wr.plot(range(len(roll)), roll, color=PURPLE, linewidth=1.4, zorder=3)
        ax_wr.axhline(50, color=MUTED, linewidth=0.7, linestyle="--", alpha=0.7)
        ax_wr.fill_between(range(len(roll)), roll, 50,
                           where=(roll >= 50), color=GREEN, alpha=0.15)
        ax_wr.fill_between(range(len(roll)), roll, 50,
                           where=(roll < 50),  color=RED,   alpha=0.15)
        ax_wr.set_ylim(0, 100)
        ax_wr.set_xlabel("Trade #", fontsize=8)
        ax_wr.set_ylabel("Roll. Win % (10T)", fontsize=9)
        ax_wr.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_wr.tick_params(labelleft=False)
        ax_wr.yaxis.set_label_position("right")
        ax_wr.yaxis.tick_right()
        ax_wr.tick_params(axis="y", colors=MUTED, labelsize=8)

    # ── Summary stats box ──────────────────────────────────────────────────────
    s      = summary
    end_eq = s["end_equity"]
    sign   = "+" if s["total"] >= 0 else ""
    txt = (
        f"Trades: {s['trades']}    "
        f"Win Rate: {s['wr']:.1f}%    "
        f"Total P&L: USD {sign}{s['total']:,.2f} ({sign}{s['pnl_pct']:.2f}%)    "
        f"Max DD: {s['dd_pct']:.1f}%    "
        f"End Equity: USD {end_eq:,.2f}"
    )
    fig.text(0.5, 0.965, txt, ha="center", va="top", fontsize=9.5,
             color=TEXT, fontfamily="monospace",
             parse_math=False,
             bbox=dict(facecolor=SURFACE, edgecolor=BORDER, boxstyle="round,pad=0.4"))

    # ── Watermark ─────────────────────────────────────────────────────────────
    fig.text(0.99, 0.01, "Astro-Bot  |  Slope Around Medium  |  BTC/USDT  Hyperliquid",
             ha="right", va="bottom", fontsize=7, color=BORDER)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n✓  Chart saved → {out_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Astro-Bot backtest chart generator")
    parser.add_argument(
        "--file", nargs="+",
        default=[str(ROOT / "logs" / "backtest_BTC_2026-01-01_2026-02-23.csv")],
        help="One or more backtest CSV files (space-separated for multi-year)",
    )
    parser.add_argument(
        "--funding-file",
        default=str(ROOT / "logs" / "funding_BTCUSDT_2026-01-01_2026-02-23.csv"),
        help="Funding rates CSV",
    )
    parser.add_argument("--cadence", type=int, default=4,
                        help="Candle size in hours (default 4 — matches live bot)")
    parser.add_argument("--title", default="2026 YTD",  help="Chart title")
    parser.add_argument("--out",   default="",          help="Output PNG path (auto if blank)")
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "logs" / f"chart_{args.title.replace(' ', '_').lower()}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running simulation for: {args.file}  (cadence={args.cadence}h)")
    summary = run_simulations(args.file, args.funding_file, cadence=args.cadence)
    make_chart(summary, args.title, out_path)

    # Try to open the image in the default viewer
    import subprocess, platform
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            subprocess.run(["open", str(out_path)], check=False)
        elif sys_name == "Linux":
            subprocess.run(["xdg-open", str(out_path)], check=False)
        elif sys_name == "Windows":
            subprocess.run(["start", str(out_path)], shell=True, check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
