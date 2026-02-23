"""
Slope Around Medium — Astro-Bot
Entry point. Loads asset DNA, schedules the bot cycle, runs the signal pipeline.

Run:
    python main.py
"""
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()

import config
from core.signal_engine import generate_signal
from core.score_history import ScoreHistory
from exchange.market_data import get_price_atr_ema, get_account_balance
from exchange.trade_executor import (
    dispatch_signal,
    check_and_close_stale_positions,
    update_peak_equity,
)

console = Console()

# ── Load asset DNA ─────────────────────────────────────────────────────────────

def load_asset_dna(asset_key: str) -> dict:
    dna_path = Path(__file__).parent / "assets_dna.json"
    with open(dna_path) as f:
        all_assets = json.load(f)
    if asset_key not in all_assets:
        logger.error(f"Asset '{asset_key}' not found in assets_dna.json")
        sys.exit(1)
    dna = all_assets[asset_key]
    if not dna.get("natal_western") or not dna.get("natal_vedic"):
        logger.warning(
            f"Natal positions for {asset_key} are empty! "
            "Run: python scripts/calculate_natal.py to compute them."
        )
    return dna


# ── Console display ────────────────────────────────────────────────────────────

def display_signal(signal: dict):
    action = signal.get("final_action", signal.get("status", "?"))
    num = signal.get("numerology", {})

    color_map = {
        "STRONG_BUY":  "bold green",
        "WEAK_BUY":    "cyan",
        "STRONG_SELL": "bold red",
        "WEAK_SELL":   "yellow",
        "NO_TRADE":    "magenta",
        "HOLD":        "white",
        "COLLECTING_DATA": "dim white",
    }
    color = color_map.get(action, "white")

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Action",     f"[{color}]{action}[/{color}]")
    table.add_row("Asset",      signal.get("asset", ""))
    table.add_row("Price",      f"${signal.get('current_price', 0):,.2f}")
    table.add_row("Stop Loss",  f"${signal.get('stop_loss', 0):,.2f}")
    table.add_row("Target",     f"${signal.get('target', 0):,.2f}")
    table.add_row("Size (USDT)",f"${signal.get('position_size_usdt', 0):,.2f}")
    cap = signal.get("effective_capital", 0)
    table.add_row("Capital base", f"${cap:,.2f}  ({config.CAPITAL_PCT*100:.0f}% of balance)")
    table.add_row("", "")
    table.add_row("Western",    f"{signal.get('western_score', 0):.4f}  "
                                f"(med: {signal.get('western_medium', 0):.4f}  "
                                f"slope: {signal.get('western_slope', 0):+.4f})")
    table.add_row("Vedic",      f"{signal.get('vedic_score', 0):.4f}  "
                                f"(med: {signal.get('vedic_medium', 0):.4f}  "
                                f"slope: {signal.get('vedic_slope', 0):+.4f})")
    table.add_row("W Signal",   signal.get("western_signal", ""))
    table.add_row("V Signal",   signal.get("vedic_signal", ""))
    table.add_row("", "")
    table.add_row("Numerology", num.get("label", ""))
    table.add_row("UDN",        str(num.get("universal_day_number", "")))
    table.add_row("Life Path",  str(num.get("life_path_number", "")))
    table.add_row("Nakshatra",  f"{signal.get('nakshatra', '')} (×{signal.get('nakshatra_multiplier', 1):.1f})")
    table.add_row("Moon Fast",  str(signal.get("moon_fast", "")))
    ema_val = signal.get("ema_value")
    if ema_val is not None:
        table.add_row("EMA", f"${ema_val:,.2f}  (filter: {config.EMA_FILTER})")
    if signal.get("filter_reason"):
        table.add_row("[yellow]Filtered[/yellow]", signal["filter_reason"])

    retro_w = ", ".join(signal.get("retrograde_western", [])) or "None"
    retro_v = ", ".join(signal.get("retrograde_vedic", [])) or "None"
    table.add_row("Retrograde (W)", retro_w)
    table.add_row("Retrograde (V)", retro_v)

    ts = signal.get("timestamp", datetime.now(timezone.utc).isoformat())
    panel = Panel(
        table,
        title=f"[bold]Astro-Bot Signal — {ts}[/bold]",
        border_style=color.replace("bold ", ""),
        expand=False,
    )
    console.print(panel)


# ── Bot cycle ──────────────────────────────────────────────────────────────────

score_history = ScoreHistory(window=config.SCORE_HISTORY_WINDOW)
asset_dna: dict = {}


def bot_cycle():
    """Single bot execution cycle. Called by the scheduler each interval."""
    logger.info(f"=== Bot cycle started — {datetime.now(timezone.utc).isoformat()} ===")

    symbol = asset_dna.get("symbol", "BTC/USDT")

    # Gap 2: force-close any positions that have been open > MAX_OPEN_BARS × interval
    # (mirrors the backtest force-close at 12 bars × 4h = 48h)
    check_and_close_stale_positions()

    # Fetch market data (price, ATR and EMA all on 4h candles)
    price, atr, ema = get_price_atr_ema(symbol)
    if price <= 0:
        logger.error("Failed to get price — skipping cycle.")
        return

    # Fetch live balance once; derive effective capital from it.
    # This avoids two separate API calls and keeps equity tracking consistent.
    balance = get_account_balance()
    capital = balance * config.CAPITAL_PCT
    logger.info(
        f"Effective capital: ${capital:,.2f}  "
        f"({config.CAPITAL_PCT*100:.0f}% of ${balance:,.2f} balance)"
    )

    # Gap 3: update peak-equity record and arm the drawdown halt if needed
    update_peak_equity(balance)

    # Generate signal (EMA filter + Nakshatra block applied inside)
    signal = generate_signal(
        asset_dna=asset_dna,
        score_history=score_history,
        current_price=price,
        atr=atr,
        current_ema=ema,
        capital=capital,
    )

    # Display
    display_signal(signal)

    # Execute / paper trade
    # current_equity=balance passes the full account balance for drawdown check
    dispatch_signal(signal, current_equity=balance)

    logger.info("=== Cycle complete ===\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global asset_dna

    console.print(Panel.fit(
        "[bold cyan]Slope Around Medium — Astro-Bot[/bold cyan]\n"
        "Cosmic Momentum Trading System\n"
        f"Asset: [yellow]{config.ACTIVE_ASSET}[/yellow]  |  "
        f"Interval: [yellow]{config.CHECK_INTERVAL_MINUTES}min[/yellow]  |  "
        f"Mode: [yellow]{'PAPER' if config.PAPER_TRADING else 'LIVE'}[/yellow]",
        border_style="cyan",
    ))

    # Load asset
    asset_dna = load_asset_dna(config.ACTIVE_ASSET)
    logger.info(f"Loaded DNA for {asset_dna.get('name', config.ACTIVE_ASSET)}")

    # Run immediately on start
    bot_cycle()

    # Schedule recurring cycles
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        bot_cycle,
        trigger="interval",
        minutes=config.CHECK_INTERVAL_MINUTES,
        id="astro_bot_cycle",
        max_instances=1,
        coalesce=True,
    )

    logger.info(f"Scheduler started — running every {config.CHECK_INTERVAL_MINUTES} minutes.")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user.")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        "logs/astro_bot_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )

    os.makedirs("logs", exist_ok=True)
    main()
