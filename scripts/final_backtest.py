"""
Final Setup 4-Year Backtest  —  Realistic simulation
======================================================
Improvements over the naive close-only simulation:

1. Intrabar SL/TP  — uses candle HIGH and LOW to detect if stop or target
   was hit *within* the candle.  Worst-case: if both SL and TP breached in
   the same candle, stop is hit first (conservative).

2. Proper N-hour candles  — when --cadence N is set (e.g. 4), the 1h CSV
   is resampled into true N-hour OHLC candles BEFORE simulation:
     open  = first 1h open in the window
     high  = MAX of all 1h highs  (catches every wick)
     low   = MIN of all 1h lows
     close = last 1h close
   ATR is then sized against N-hour candle volatility (wider stops that
   survive intrabar noise), and signals are taken from the START of each
   N-hour window (what the bot would have seen when it woke up).

3. Entry slippage  — market order fills at open of current candle + 0.05%.

4. Exchange fees   — 0.05% taker on entry + 0.05% taker on exit = 0.10%
   round-trip.  Configurable via TAKER_FEE.

5. Funding rates   — loaded from logs/funding_BTCUSDT_*.csv.  Applied
   every 8 hours while a trade is open.

Usage:
    python scripts/final_backtest.py                  # default 1h cadence
    python scripts/final_backtest.py --cadence 4      # 4h candles
    python scripts/final_backtest.py --compare        # all cadences side-by-side
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

console = Console(width=180)

# ── Simulation constants ───────────────────────────────────────────────────────
CAPITAL       = 10_000.0
RISK_PCT      = 0.01          # fraction of equity risked per trade
ATR_LOOKBACK  = 14
ATR_MULT      = 1.5
RR_RATIO      = 2.0
INTERVAL      = 1             # rows to skip (1 = every candle = true 1h cadence)
MAX_OPEN_BARS = 48            # 48 × 1h = 48h max hold
EMA_PERIOD    = config.EMA_PERIOD          # read from config / .env
EMA_FILTER    = config.EMA_FILTER          # "none" | "one_way" | "two_way"
NK_FILTER     = config.NAKSHATRA_FILTER    # True / False
TAKER_FEE     = 0.0005        # 0.05% per side (Hyperliquid taker)
SLIPPAGE      = 0.0005        # 0.05% entry slippage (market order spread)
BLOCKED       = config.TRADE_UNFAVORABLE_NAKSHATRAS
FUNDING_CSV   = Path("logs/funding_BTCUSDT_2022-01-01_2025-12-31.csv")

BTC_MARKET = {
    2022: "Bear  −65%",
    2023: "Bull +155%",
    2024: "Bull +121%",
    2025: "Mixed",
    2026: "YTD",
}


def _atr(p):  return p.pct_change().abs().rolling(ATR_LOOKBACK).mean() * p
def _ema(p):  return p.ewm(span=EMA_PERIOD, adjust=False).mean()


def resample_to_nh(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Aggregate a 1h-resolution DataFrame into N-hour OHLC candles.

    OHLC: open=first, high=MAX, low=MIN, close=last  (captures every wick)
    Signals: taken from the FIRST row of each window  (what the bot sees
             when it wakes up at the start of the N-hour period)
    """
    if hours == 1:
        return df.copy()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    signal_cols = [c for c in
                   ["action", "nakshatra", "western_score", "vedic_score",
                    "western_slope", "vedic_slope", "resonance_day"]
                   if c in df.columns]

    freq = f"{hours}h"

    price_agg = df[["open", "high", "low", "price"]].resample(freq).agg(
        open=("open",  "first"),
        high=("high",  "max"),
        low=("low",   "min"),
        price=("price", "last"),
    )
    sig_agg = df[signal_cols].resample(freq).first()

    result = price_agg.join(sig_agg).dropna(subset=["price"]).reset_index()
    return result


def load_funding_rates() -> pd.Series:
    """Load funding rates indexed by timestamp (every 8h)."""
    if not FUNDING_CSV.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(FUNDING_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return df.set_index("timestamp")["funding_rate"]


def simulate(df: pd.DataFrame, funding: pd.Series,
             cadence: int = INTERVAL,
             use_intrabar: bool = True,
             use_fees: bool = True,
             use_slippage: bool = True,
             use_funding: bool = True,
             ema_period: int = EMA_PERIOD,
             ema_filter: str = EMA_FILTER,
             nk_filter: bool = NK_FILTER,
             risk_pct: float = RISK_PCT,
             starting_capital: float = CAPITAL) -> dict:

    # Resample to the requested cadence so ATR and intrabar highs/lows
    # reflect N-hour candles, not 1h candles.
    df     = resample_to_nh(df, cadence)
    max_open_bars = max(1, 48 // cadence)   # always ~48h max hold

    prices = df["price"]
    at     = _atr(prices)
    em     = prices.ewm(span=ema_period, adjust=False).mean()
    rows   = df.copy()   # already at cadence resolution

    has_hl = "high" in df.columns and "low" in df.columns

    equity        = starting_capital
    trades        = []
    ot            = None
    total_bars    = 0
    bars_in_trade = 0

    for i, row in rows.iterrows():
        action = row["action"]
        close  = row["price"]
        high   = float(row["high"]) if has_hl else close
        low    = float(row["low"])  if has_hl else close
        naks   = row.get("nakshatra", "")
        ts     = pd.to_datetime(row["timestamp"], utc=True)
        month  = str(ts)[:7]
        total_bars += 1
        if ot:
            bars_in_trade += 1

        # ── Exit check on open trade ───────────────────────────────────────
        if ot:
            sl, tp, age = ot["sl"], ot["tp"], ot["age"]

            if use_intrabar and has_hl:
                # Determine which level was hit first inside the candle.
                # Conservative: if both SL and TP are breached, stop wins.
                if ot["side"] == "BUY":
                    sl_hit = low  <= sl
                    tp_hit = high >= tp
                else:
                    sl_hit = high >= sl
                    tp_hit = low  <= tp
            else:
                # Legacy close-only check
                sl_hit = (ot["side"] == "BUY"  and close <= sl) or \
                         (ot["side"] == "SELL" and close >= sl)
                tp_hit = (ot["side"] == "BUY"  and close >= tp) or \
                         (ot["side"] == "SELL" and close <= tp)

            # Funding applied every 8h while trade is open
            if use_funding and len(funding) > 0:
                hour = ts.hour
                if hour % 8 == 0 and ts in funding.index:
                    rate = funding.get(ts, 0.0)
                    notional = ot["notional"]
                    if ot["side"] == "BUY":
                        # Long pays positive rate, receives negative rate
                        equity -= notional * rate
                    else:
                        # Short receives positive rate, pays negative rate
                        equity += notional * rate

            # Resolve exit
            def _rec(result, pnl):
                return {
                    "pnl": pnl, "result": result,
                    "month": ot["month"], "side": ot["side"],
                    "signal": ot["signal"], "entry": ot["entry"],
                    "open_ts": ot["open_ts"],
                    "bars": age + 1, "notional": ot["notional"],
                    "risk": ot["risk"],
                }
            if sl_hit and not tp_hit:
                pnl = -ot["risk"]
                if use_fees: pnl -= ot["notional"] * TAKER_FEE
                equity += pnl
                trades.append(_rec("STOP", pnl))
                ot = None
            elif tp_hit and not sl_hit:
                pnl = ot["risk"] * RR_RATIO
                if use_fees: pnl -= ot["notional"] * TAKER_FEE
                equity += pnl
                trades.append(_rec("WIN", pnl))
                ot = None
            elif sl_hit and tp_hit:
                pnl = -ot["risk"]
                if use_fees: pnl -= ot["notional"] * TAKER_FEE
                equity += pnl
                trades.append(_rec("STOP_INTRABAR", pnl))
                ot = None
            elif age >= max_open_bars:
                exit_price = close
                if ot["side"] == "BUY":
                    pnl = (exit_price - ot["entry"]) / ot["entry"] * ot["notional"]
                else:
                    pnl = (ot["entry"] - exit_price) / ot["entry"] * ot["notional"]
                if use_fees: pnl -= ot["notional"] * TAKER_FEE
                equity += pnl
                trades.append(_rec("TIMEOUT", pnl))
                ot = None
            else:
                ot["age"] += 1

        # ── Signal filters ─────────────────────────────────────────────────
        if action not in ("STRONG_BUY", "STRONG_SELL"):
            continue
        if nk_filter and naks in BLOCKED:
            continue
        if ot:
            continue

        ev = em.get(i)
        if ev and not np.isnan(float(ev)) and ema_filter != "none":
            if "SELL" in action and close > float(ev):
                continue
            if ema_filter == "two_way" and "BUY" in action and close < float(ev):
                continue

        # ── Size and open trade ────────────────────────────────────────────
        at_v = at.get(i, close * 0.015)
        if pd.isna(at_v) or at_v <= 0:
            at_v = close * 0.015

        side = "BUY" if "BUY" in action else "SELL"
        risk = equity * risk_pct
        sld  = at_v * ATR_MULT

        # Entry price: next open + slippage
        entry = close  # fallback
        if has_hl and use_slippage:
            entry = float(row.get("open", close))
            entry *= (1 + SLIPPAGE) if side == "BUY" else (1 - SLIPPAGE)

        sl = entry - sld if side == "BUY" else entry + sld
        tp = entry + sld * RR_RATIO if side == "BUY" else entry - sld * RR_RATIO

        # Notional = risk / stop_distance_pct
        stop_pct = abs(entry - sl) / entry
        notional = risk / stop_pct if stop_pct > 0 else risk * 10

        fee_cost = notional * TAKER_FEE if use_fees else 0.0
        equity  -= fee_cost   # entry fee deducted immediately

        ot = {
            "side": side, "signal": action, "entry": entry, "sl": sl, "tp": tp,
            "risk": risk, "notional": notional,
            "age": 0, "month": month, "open_ts": str(ts)[:16],
        }

    # ── Aggregate stats ────────────────────────────────────────────────────────
    wins    = [t for t in trades if t["pnl"] > 0]
    losses  = [t for t in trades if t["pnl"] <= 0]
    total   = sum(t["pnl"] for t in trades)
    wr      = len(wins) / len(trades) * 100 if trades else 0
    avg_w   = float(np.mean([t["pnl"] for t in wins]))   if wins   else 0.0
    avg_l   = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    exp     = (wr / 100 * avg_w + (1 - wr / 100) * avg_l) if trades else 0.0

    eq, pk, dd = starting_capital, starting_capital, 0.0
    for t in trades:
        eq += t["pnl"]
        pk  = max(pk, eq)
        dd  = max(dd, pk - eq)

    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = t["month"]
        monthly.setdefault(m, {"pnl": 0, "wins": 0, "total": 0})
        monthly[m]["pnl"]   += t["pnl"]
        monthly[m]["total"] += 1
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1

    # ── Capital efficiency stats ───────────────────────────────────────────────
    time_in_market_pct = bars_in_trade / total_bars * 100 if total_bars else 0.0

    avg_hold_bars  = float(np.mean([t["bars"]    for t in trades])) if trades else 0.0
    avg_hold_hours = avg_hold_bars * cadence

    avg_notional   = float(np.mean([t["notional"] for t in trades])) if trades else 0.0
    total_risk_dep = sum(t["risk"] for t in trades)                  # cumulative risk dollars
    ror            = total / total_risk_dep * 100 if total_risk_dep else 0.0  # return on risk

    # Margin locked per trade = notional / leverage (from config)
    lev            = getattr(__import__("config"), "LEVERAGE", 3)
    avg_margin     = avg_notional / lev
    avg_margin_pct = avg_margin / starting_capital * 100

    idle_pct    = 1 - time_in_market_pct / 100
    idle_days   = total_bars * cadence / 24
    idle_yield  = starting_capital * idle_pct * 0.045 * idle_days / 365

    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "wr": wr, "avg_win": avg_w, "avg_loss": avg_l, "exp": exp,
        "total": total, "pnl_pct": total / starting_capital * 100,
        "end_equity": starting_capital + total,
        "max_dd": dd, "dd_pct": dd / CAPITAL * 100,
        "trade_list": trades, "monthly": monthly,
        # capital efficiency
        "time_in_market_pct": time_in_market_pct,
        "total_bars": total_bars,
        "bars_in_trade": bars_in_trade,
        "avg_hold_bars": avg_hold_bars,
        "avg_hold_hours": avg_hold_hours,
        "avg_notional": avg_notional,
        "avg_margin": avg_margin,
        "avg_margin_pct": avg_margin_pct,
        "total_risk_deployed": total_risk_dep,
        "return_on_risk": ror,
        "idle_yield_est": idle_yield,
        "cadence": cadence,
    }


def print_summary_table(results: dict, title: str):
    t = Table(
        "Year", "BTC Market", "Trades", "Wins", "Losses", "Win %",
        "Avg Win", "Avg Loss", "Exp/trade",
        "Total P&L", "P&L %", "Max DD %", "End Equity",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title=f"[bold]{title}[/bold]",
        show_lines=True,
    )
    for y, s in results.items():
        pc  = "green" if s["total"]   >= 0  else "red"
        wc  = "green" if s["wr"]      >= 50 else ("yellow" if s["wr"] >= 40 else "red")
        dc  = "green" if s["dd_pct"]  <= 15 else ("yellow" if s["dd_pct"] <= 25 else "red")
        ec  = "green" if s["exp"]     >= 0  else "red"
        t.add_row(
            f"[bold]{y}[/bold]",
            BTC_MARKET.get(y, ""),
            str(s["trades"]),
            f"[green]{s['wins']}[/green]",
            f"[red]{s['losses']}[/red]",
            f"[{wc}]{s['wr']:.1f}%[/{wc}]",
            f"[green]+${s['avg_win']:.0f}[/green]",
            f"[red]-${abs(s['avg_loss']):.0f}[/red]",
            f"[{ec}]${s['exp']:+.0f}[/{ec}]",
            f"[{pc}]{s['total']:+,.0f}[/{pc}]",
            f"[{pc}]{s['pnl_pct']:+.1f}%[/{pc}]",
            f"[{dc}]{s['dd_pct']:.1f}%[/{dc}]",
            f"${s['end_equity']:,.0f}",
        )
    console.print(t)


def print_monthly_table(results: dict):
    t = Table(
        "Month", "Trades", "Wins", "Win %", "P&L", "Cumulative",
        box=box.ROUNDED,
        header_style="bold white on dark_green",
        title="[bold]Monthly P&L — All 4 Years[/bold]",
    )
    cumul = 0.0
    for y, s in results.items():
        for m in sorted(s["monthly"].keys()):
            md   = s["monthly"][m]
            mpnl = md["pnl"]
            cumul += mpnl
            wr   = md["wins"] / md["total"] * 100 if md["total"] else 0
            pc   = "green" if mpnl  >= 0 else "red"
            cc   = "green" if cumul >= 0 else "red"
            wc   = "green" if wr    >= 50 else ("yellow" if wr >= 40 else "red")
            t.add_row(
                m, str(md["total"]), str(md["wins"]),
                f"[{wc}]{wr:.0f}%[/{wc}]",
                f"[{pc}]{'+' if mpnl>=0 else ''}{mpnl:,.0f}[/{pc}]",
                f"[{cc}]{'+' if cumul>=0 else ''}{cumul:,.0f}[/{cc}]",
            )
    console.print(t)


def print_compare_table(naive: dict, realistic: dict):
    t = Table(
        "Year", "Naive P&L%", "Realistic P&L%", "Δ",
        "Naive DD%", "Realistic DD%", "Naive WR", "Realistic WR",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_red",
        title="[bold]Naive (close-only, no fees) vs Realistic (intrabar + fees + slippage + funding)[/bold]",
        show_lines=True,
    )
    for y in naive:
        n, r = naive[y], realistic[y]
        diff = r["pnl_pct"] - n["pnl_pct"]
        dc   = "red" if diff < 0 else "green"
        t.add_row(
            str(y),
            f"[green]{n['pnl_pct']:+.1f}%[/green]",
            f"{'[green]' if r['pnl_pct']>=0 else '[red]'}{r['pnl_pct']:+.1f}%"
            f"{'[/green]' if r['pnl_pct']>=0 else '[/red]'}",
            f"[{dc}]{diff:+.1f}%[/{dc}]",
            f"{n['dd_pct']:.1f}%",
            f"{r['dd_pct']:.1f}%",
            f"{n['wr']:.1f}%",
            f"{r['wr']:.1f}%",
        )
    console.print(t)


def compounded_growth(results: dict) -> tuple:
    compound = CAPITAL
    for s in results.values():
        compound *= (1 + s["pnl_pct"] / 100)
    n    = max(len(results), 1)
    cagr = ((compound / CAPITAL) ** (1 / n) - 1) * 100
    return cagr, compound


def _run_year(path, funding, cadence):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return simulate(df, funding, cadence=cadence,
                    use_intrabar=True, use_fees=True,
                    use_slippage=True, use_funding=True)


def print_cadence_compare(all_results: dict):
    """all_results = {cadence: {year: stats}}"""
    cadences = sorted(all_results.keys())

    t = Table(
        "Cadence", "Candle ATR", "Trades/yr",
        "2022", "2023", "2024", "2025",
        "4yr P&L", "CAGR", "Worst DD",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title="[bold]Realistic Cadence Comparison — 4 Years  (intrabar + fees + slippage + funding)[/bold]",
        show_lines=True,
    )

    atr_note = {1: "1h ≈ narrow", 2: "2h", 4: "4h ≈ wide", 8: "8h ≈ widest"}

    for c in cadences:
        res = all_results[c]
        compound = CAPITAL
        worst_dd = 0.0
        total_trades = 0
        year_cells = []
        for y in [2022, 2023, 2024, 2025]:
            s = res.get(y)
            if not s:
                year_cells.append("—")
                continue
            compound     *= (1 + s["pnl_pct"] / 100)
            worst_dd      = max(worst_dd, s["dd_pct"])
            total_trades += s["trades"]
            pc = "green" if s["pnl_pct"] >= 0 else "red"
            year_cells.append(f"[{pc}]{s['pnl_pct']:+.1f}%[/{pc}]\n[dim]{s['wr']:.0f}% WR[/dim]")

        gain = compound - CAPITAL
        cagr = ((compound / CAPITAL) ** (1 / 4) - 1) * 100
        gc   = "green" if gain >= 0 else "red"
        dc   = "green" if worst_dd <= 20 else ("yellow" if worst_dd <= 35 else "red")
        avg_trades = total_trades // 4

        t.add_row(
            f"[bold]{c}h[/bold]",
            atr_note.get(c, f"{c}h"),
            str(avg_trades),
            *year_cells,
            f"[{gc}]+${gain:,.0f} (+{gain/CAPITAL*100:.0f}%)[/{gc}]",
            f"[{gc}]{cagr:.1f}%[/{gc}]",
            f"[{dc}]{worst_dd:.1f}%[/{dc}]",
        )
    console.print(t)


def print_capital_efficiency(stats: dict, label: str):
    """
    Print a capital efficiency panel for a simulation result.

    Covers four dimensions:
      1. Time in market   — how much of the time is capital working
      2. Margin per trade — how much equity is locked per position (at leverage)
      3. Return on risk   — P&L earned per $1 of actual risk deployed
      4. Idle capital     — estimated yield from deploying idle cash in stablecoins
    """
    t = stats["time_in_market_pct"]
    tc = "green" if t >= 40 else ("yellow" if t >= 20 else "red")

    m_pct  = stats["avg_margin_pct"]
    mc     = "green" if m_pct <= 30 else ("yellow" if m_pct <= 60 else "red")

    ror    = stats["return_on_risk"]
    rc     = "green" if ror >= 100 else ("yellow" if ror >= 0 else "red")

    idle_y = stats["idle_yield_est"]
    ic     = "green" if idle_y >= 0 else "red"

    hold_h = stats["avg_hold_hours"]
    hold_b = stats["avg_hold_bars"]
    cad    = stats["cadence"]

    console.print(Panel(
        f"  [bold]1. Time in Market[/bold]\n"
        f"     Bars in trade  :  {stats['bars_in_trade']} / {stats['total_bars']} candles\n"
        f"     Time in market :  [{tc}]{t:.1f}%[/{tc}]  "
        f"(capital working {t:.1f}% of the time)\n"
        f"     Avg hold       :  {hold_h:.1f}h  ({hold_b:.1f} × {cad}h candles)\n\n"
        f"  [bold]2. Margin Efficiency[/bold]\n"
        f"     Avg notional   :  ${stats['avg_notional']:,.0f}  per trade\n"
        f"     Avg margin     :  ${stats['avg_margin']:,.0f}  per trade  "
        f"([{mc}]{m_pct:.1f}% of capital locked[/{mc}])\n"
        f"     Leverage used  :  {stats['avg_notional']/stats['avg_margin']:.0f}x  "
        f"(notional / margin)\n"
        f"     [dim]Lower margin% = more capital free for other assets or strategies[/dim]\n\n"
        f"  [bold]3. Return on Risk Deployed[/bold]\n"
        f"     Total risk dep :  ${stats['total_risk_deployed']:,.0f}  "
        f"(sum of $-at-risk per trade)\n"
        f"     Net P&L        :  ${stats['total']:+,.0f}\n"
        f"     Return / risk  :  [{rc}]{ror:+.1f}%[/{rc}]  "
        f"(earned {ror/100:.2f}x the capital actually risked)\n\n"
        f"  [bold]4. Idle Capital Opportunity[/bold]\n"
        f"     Idle time      :  {100 - t:.1f}%  of candles with no position\n"
        f"     Yield est.     :  [{ic}]+${idle_y:,.0f}[/{ic}]  "
        f"[dim](idle cash at 4.5% APY stablecoin — additive to trading P&L)[/dim]\n"
        f"     All-in P&L est :  [{ic}]+${stats['total'] + idle_y:,.0f}[/{ic}]  "
        f"[dim](trading + idle yield combined)[/dim]",
        title=f"[bold yellow]Capital Efficiency — {label}[/bold yellow]",
        border_style="yellow",
    ))


def print_single_period(stats: dict, label: str, cadence: int,
                         start_date: str, end_date: str):
    """
    Rich display for a single partial-period backtest (e.g. 2026 YTD).
    Shows per-trade detail, monthly P&L, annualised return, and a
    'what-would-$10k-become' equity panel.
    """
    from datetime import date as date_cls

    # Compute holding period in days for annualisation
    try:
        d0 = date_cls.fromisoformat(start_date)
        d1 = date_cls.fromisoformat(end_date)
        days = max((d1 - d0).days, 1)
    except Exception:
        days = 53   # fallback

    raw_pct = stats["pnl_pct"]
    ann_pct = ((1 + raw_pct / 100) ** (365 / days) - 1) * 100

    # ── Summary panel ─────────────────────────────────────────────────────────
    gc  = "green" if stats["total"] >= 0 else "red"
    wc  = "green" if stats["wr"] >= 50 else ("yellow" if stats["wr"] >= 40 else "red")
    dc  = "green" if stats["dd_pct"] <= 15 else ("yellow" if stats["dd_pct"] <= 25 else "red")
    ac  = "green" if ann_pct >= 0 else "red"

    console.print(Panel(
        f"  Period  :  [bold]{start_date}  →  {end_date}[/bold]  "
        f"([dim]{days} days[/dim])\n"
        f"  Cadence :  [bold]{cadence}h[/bold] candles   |   "
        f"EMA({EMA_PERIOD}) {EMA_FILTER}   |   "
        f"NK filter: {'ON' if NK_FILTER else 'OFF'}\n\n"
        f"  Trades  :  {stats['trades']}  "
        f"([green]{stats['wins']} wins[/green] / [red]{stats['losses']} losses[/red])   "
        f"Win rate: [{wc}]{stats['wr']:.1f}%[/{wc}]\n"
        f"  Avg Win :  [green]+${stats['avg_win']:.0f}[/green]   "
        f"Avg Loss: [red]-${abs(stats['avg_loss']):.0f}[/red]   "
        f"Expectancy/trade: [{'green' if stats['exp']>=0 else 'red'}]"
        f"${stats['exp']:+.0f}[/{'green' if stats['exp']>=0 else 'red'}]\n\n"
        f"  Raw P&L :  [{gc}]{raw_pct:+.2f}%  (${stats['total']:+,.0f})[/{gc}]   "
        f"End equity: [{gc}]${stats['end_equity']:,.0f}[/{gc}]\n"
        f"  Ann. ret:  [{ac}]{ann_pct:+.1f}% / yr[/{ac}]  "
        f"[dim](linear annualisation over {days}d)[/dim]\n"
        f"  Max DD  :  [{dc}]{stats['dd_pct']:.1f}%[/{dc}]",
        title=f"[bold cyan]{label} — Realistic Simulation[/bold cyan]",
        border_style="cyan",
    ))

    # ── Monthly breakdown ──────────────────────────────────────────────────────
    monthly = stats.get("monthly", {})
    if monthly:
        t = Table(
            "Month", "Trades", "Wins", "Win %", "P&L ($)", "Cumulative",
            box=box.ROUNDED,
            header_style="bold white on dark_green",
            title=f"[bold]Monthly Breakdown — {label}[/bold]",
        )
        cumul = 0.0
        for m in sorted(monthly.keys()):
            md   = monthly[m]
            mpnl = md["pnl"]
            cumul += mpnl
            wr   = md["wins"] / md["total"] * 100 if md["total"] else 0
            pc   = "green" if mpnl  >= 0 else "red"
            cc   = "green" if cumul >= 0 else "red"
            wc2  = "green" if wr    >= 50 else ("yellow" if wr >= 40 else "red")
            t.add_row(
                m, str(md["total"]), str(md["wins"]),
                f"[{wc2}]{wr:.0f}%[/{wc2}]",
                f"[{pc}]{'+' if mpnl>=0 else ''}{mpnl:,.0f}[/{pc}]",
                f"[{cc}]{'+' if cumul>=0 else ''}{cumul:,.0f}[/{cc}]",
            )
        console.print(t)

    # ── Trade-by-trade detail ──────────────────────────────────────────────────
    trades = stats.get("trade_list", [])
    if trades:
        t2 = Table(
            "#", "Side", "Result", "Hold (bars)", "P&L ($)",
            box=box.SIMPLE,
            header_style="bold",
            title=f"[bold]All Trades — {label}[/bold]",
        )
        for n, tr in enumerate(trades, 1):
            pc   = "green" if tr["pnl"] > 0 else "red"
            res  = {"WIN": "[green]WIN[/green]",
                    "STOP": "[red]STOP[/red]",
                    "STOP_INTRABAR": "[red]STOP (intrabar)[/red]",
                    "TIMEOUT": "[yellow]TIMEOUT[/yellow]"}.get(tr["result"], tr["result"])
            t2.add_row(
                str(n),
                f"[cyan]{tr['side']}[/cyan]",
                res,
                f"{tr.get('bars', '?')} ({tr.get('bars', 0)*cadence}h)",
                f"[{pc}]{'+' if tr['pnl']>=0 else ''}{tr['pnl']:,.0f}[/{pc}]",
            )
        console.print(t2)

    # ── Capital efficiency ─────────────────────────────────────────────────────
    print_capital_efficiency(stats, label)
    console.print()


def run_risk_comparison(dfs: dict, funding: pd.Series,
                        cadence: int, risk_levels: list,
                        label_prefix: str = ""):
    """
    Run simulate() at each risk level and print a side-by-side comparison.

    dfs: {year_or_label: DataFrame}  — one or more periods to aggregate
    Returns the table for display.
    """
    # ── Compute results for every risk level ──────────────────────────────────
    results_by_risk = {}
    for r in risk_levels:
        period_stats = {}
        for key, df in dfs.items():
            period_stats[key] = simulate(
                df, funding, cadence=cadence,
                use_intrabar=True, use_fees=True,
                use_slippage=True, use_funding=True,
                risk_pct=r,
            )
        results_by_risk[r] = period_stats

    # ── Pre-compute row data for all risk levels ──────────────────────────────
    period_labels = list(list(results_by_risk.values())[0].keys())
    multi_period  = len(period_labels) > 1
    rows_data = []

    for r, period_stats in results_by_risk.items():
        all_trades = [tr for s in period_stats.values() for tr in s.get("trade_list", [])]
        total_pnl  = sum(s["total"]   for s in period_stats.values())
        total_risk = sum(s["total_risk_deployed"] for s in period_stats.values())

        compound = CAPITAL
        worst_dd = 0.0
        for s in period_stats.values():
            compound *= (1 + s["pnl_pct"] / 100)
            worst_dd  = max(worst_dd, s["dd_pct"])

        gain_pct  = (compound - CAPITAL) / CAPITAL * 100
        n_periods = max(len(period_stats), 1)

        if multi_period:
            cagr = ((compound / CAPITAL) ** (1 / n_periods) - 1) * 100
            growth_label = f"{cagr:.1f}% CAGR"
        else:
            s0   = list(period_stats.values())[0]
            days = s0.get("total_bars", 1) * cadence / 24
            ann  = ((1 + s0["pnl_pct"] / 100) ** (365 / max(days, 1)) - 1) * 100
            growth_label = f"{ann:.0f}%/yr"

        total_bars    = sum(s.get("total_bars",    0) for s in period_stats.values())
        bars_in_trade = sum(s.get("bars_in_trade", 0) for s in period_stats.values())
        tim_pct = bars_in_trade / total_bars * 100 if total_bars else 0
        ror     = total_pnl / total_risk * 100 if total_risk else 0

        wins   = [tr for tr in all_trades if tr["pnl"] > 0]
        losses = [tr for tr in all_trades if tr["pnl"] <= 0]
        n_t    = len(all_trades)
        wr     = len(wins) / n_t * 100 if n_t else 0
        avg_w  = float(np.mean([tr["pnl"] for tr in wins]))   if wins   else 0
        avg_l  = float(np.mean([tr["pnl"] for tr in losses])) if losses else 0
        exp    = wr / 100 * avg_w + (1 - wr / 100) * avg_l   if n_t    else 0

        per_year = {}
        if multi_period:
            for p in period_labels:
                per_year[p] = period_stats[p]["pnl_pct"]

        rows_data.append({
            "r": r, "n_t": n_t, "wr": wr, "avg_w": avg_w, "avg_l": avg_l,
            "exp": exp, "total_pnl": total_pnl, "gain_pct": gain_pct,
            "growth_label": growth_label, "worst_dd": worst_dd,
            "compound": compound, "ror": ror, "tim_pct": tim_pct,
            "per_year": per_year,
        })

    title_str = (f"[bold]Risk per Trade Comparison — {label_prefix}  |  "
                 f"{cadence}h cadence  |  EMA({EMA_PERIOD}) {EMA_FILTER}[/bold]")

    # ── Table 1: Performance metrics ──────────────────────────────────────────
    t1_cols = ["Risk/trade", "Trades", "Win %", "Avg Win ($)", "Avg Loss ($)",
               "Exp/trade ($)"]
    if multi_period:
        t1_cols += [str(p) for p in period_labels]
    t1_cols += ["Total P&L ($)", "P&L %", "Ann./CAGR", "End Equity ($)"]

    t1 = Table(*t1_cols, box=box.MINIMAL_HEAVY_HEAD,
               header_style="bold white on dark_blue",
               title=title_str + "\n[dim]Part 1 — Returns[/dim]",
               show_lines=True, min_width=100)

    for d in rows_data:
        r = d["r"]
        pc = "green" if d["total_pnl"] >= 0 else "red"
        wc = "green" if d["wr"] >= 50  else ("yellow" if d["wr"] >= 40 else "red")
        gc = "green" if d["gain_pct"] >= 0 else "red"
        ec = "green" if d["exp"] >= 0  else "red"
        is_current = abs(r - RISK_PCT) < 1e-9
        rk = (f"[bold cyan]{r*100:.1f}% ←[/bold cyan]"
              if is_current else f"[bold]{r*100:.1f}%[/bold]")

        year_cells = []
        if multi_period:
            for p in period_labels:
                yv = d["per_year"].get(p, 0)
                yc = "green" if yv >= 0 else "red"
                year_cells.append(f"[{yc}]{yv:+.1f}%[/{yc}]")

        t1.add_row(
            rk, str(d["n_t"]),
            f"[{wc}]{d['wr']:.1f}%[/{wc}]",
            f"[green]+{d['avg_w']:,.0f}[/green]",
            f"[red]-{abs(d['avg_l']):,.0f}[/red]",
            f"[{ec}]{d['exp']:+,.0f}[/{ec}]",
            *year_cells,
            f"[{pc}]{d['total_pnl']:+,.0f}[/{pc}]",
            f"[{pc}]{d['gain_pct']:+.1f}%[/{pc}]",
            f"[{gc}]{d['growth_label']}[/{gc}]",
            f"[{gc}]{d['compound']:,.0f}[/{gc}]",
        )

    # ── Table 2: Risk / efficiency metrics ────────────────────────────────────
    t2 = Table(
        "Risk/trade", "Max DD %", "DD : Gain ratio",
        "Return on Risk", "Time in Market", "Verdict",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_orange",
        title="[dim]Part 2 — Risk & Capital Efficiency[/dim]",
        show_lines=True, min_width=100,
    )

    for d in rows_data:
        r = d["r"]
        dc  = "green" if d["worst_dd"] <= 15 else ("yellow" if d["worst_dd"] <= 30 else "red")
        rc  = "green" if d["ror"] >= 50       else ("yellow" if d["ror"] >= 0 else "red")
        tc  = "green" if d["tim_pct"] >= 35   else ("yellow" if d["tim_pct"] >= 20 else "red")
        is_current = abs(r - RISK_PCT) < 1e-9
        rk = (f"[bold cyan]{r*100:.1f}% ←[/bold cyan]"
              if is_current else f"[bold]{r*100:.1f}%[/bold]")

        dd_ratio = d["worst_dd"] / max(abs(d["gain_pct"]), 0.1)
        drc = "green" if dd_ratio <= 0.3 else ("yellow" if dd_ratio <= 0.6 else "red")

        # Simple verdict
        if d["worst_dd"] > 50:
            verdict = "[red]Ruin risk[/red]"
        elif d["worst_dd"] > 30:
            verdict = "[red]High risk[/red]"
        elif d["worst_dd"] > 20:
            verdict = "[yellow]Aggressive[/yellow]"
        elif d["worst_dd"] > 10:
            verdict = "[yellow]Moderate[/yellow]"
        else:
            verdict = "[green]Conservative[/green]"

        t2.add_row(
            rk,
            f"[{dc}]{d['worst_dd']:.1f}%[/{dc}]",
            f"[{drc}]{dd_ratio:.2f}x[/{drc}]",
            f"[{rc}]{d['ror']:+.1f}%[/{rc}]",
            f"[{tc}]{d['tim_pct']:.1f}%[/{tc}]",
            verdict,
        )

    console.print()
    console.print(t1)
    console.print()
    console.print(t2)

    # ── Kelly criterion panel ──────────────────────────────────────────────────
    # Use the baseline 1% result to compute full-Kelly and half-Kelly
    base_stats   = results_by_risk.get(RISK_PCT) or list(results_by_risk.values())[0]
    base_trades  = [tr for s in base_stats.values() for tr in s.get("trade_list", [])]
    if base_trades:
        bw  = [tr for tr in base_trades if tr["pnl"] > 0]
        bl  = [tr for tr in base_trades if tr["pnl"] <= 0]
        bwr = len(bw) / len(base_trades)
        avg_bw = float(np.mean([tr["pnl"] for tr in bw]))   if bw else 0
        avg_bl = float(np.mean([tr["pnl"] for tr in bl]))   if bl else 0
        # Kelly: f* = W/|L| - (1-W)/W   (where W=win%, W/|L| = payoff ratio)
        avg_bw_r = avg_bw / (CAPITAL * RISK_PCT)   # avg win as multiple of 1R
        avg_bl_r = abs(avg_bl) / (CAPITAL * RISK_PCT)
        if avg_bl_r > 0:
            kelly = bwr / avg_bl_r - (1 - bwr) / avg_bw_r
            half_kelly = kelly / 2
        else:
            kelly = half_kelly = 0.0

        console.print(Panel(
            f"  Win rate (base):  {bwr*100:.1f}%\n"
            f"  Avg Win  (base):  {avg_bw_r:.2f}R\n"
            f"  Avg Loss (base):  {avg_bl_r:.2f}R\n\n"
            f"  Full Kelly f*  :  [bold yellow]{kelly*100:.1f}% per trade[/bold yellow]\n"
            f"  Half Kelly     :  [bold green]{half_kelly*100:.1f}% per trade[/bold green]  "
            f"[dim](recommended — half-Kelly balances growth & drawdown)[/dim]\n\n"
            f"  [dim]Kelly tells you the mathematically optimal fraction to risk per trade "
            f"to maximise long-run growth without going broke.\n"
            f"  Full Kelly maximises growth but drawdowns can be severe — "
            f"Half Kelly is the practical standard.[/dim]",
            title="[bold]Kelly Criterion — Optimal Risk Sizing[/bold]",
            border_style="yellow",
        ))
    console.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadence", type=int, default=1,
                        help="Candle size in hours (1, 2, 4, 8). Default 1.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all cadences (1h, 2h, 4h, 8h) side-by-side")
    parser.add_argument("--file", type=str, default=None,
                        help="Run a single-file realistic simulation (e.g. 2026 YTD).")
    parser.add_argument("--funding-file", type=str, default=None,
                        help="Funding rates CSV for --file mode. Auto-detected if omitted.")
    parser.add_argument("--label", type=str, default=None,
                        help="Display label for --file mode (e.g. '2026 YTD').")
    parser.add_argument("--compare-risk", action="store_true",
                        help="Compare multiple risk-per-trade levels side by side.")
    parser.add_argument("--risk-levels", type=str, default="0.5,1,2,3,5,7,10",
                        help="Comma-separated risk %% values to compare (default: 0.5,1,2,3,5,7,10).")
    args = parser.parse_args()

    # ── Single-file mode (partial year / custom period) ────────────────────────
    if args.file:
        cadence = args.cadence if args.cadence != 1 else 4   # default 4h for single-file
        fpath   = Path(args.file)
        if not fpath.exists():
            console.print(f"[red]File not found: {fpath}[/red]")
            sys.exit(1)

        # Auto-detect funding CSV from filename date range
        if args.funding_file:
            fcsvpath = Path(args.funding_file)
        else:
            # Try to match logs/funding_BTCUSDT_{start}_{end}.csv from filename
            stem = fpath.stem  # e.g. backtest_BTC_2026-01-01_2026-02-22
            parts = stem.split("_")
            # filename pattern: backtest_BTC_START_END
            # funding pattern:  funding_BTCUSDT_START_END
            try:
                start_d = parts[-2]
                end_d   = parts[-1]
                fcsvpath = Path(f"logs/funding_BTCUSDT_{start_d}_{end_d}.csv")
            except Exception:
                fcsvpath = FUNDING_CSV

        if fcsvpath.exists():
            fund_df = pd.read_csv(fcsvpath)
            fund_df["timestamp"] = pd.to_datetime(fund_df["timestamp"], utc=True, format="mixed")
            funding = fund_df.set_index("timestamp")["funding_rate"]
            console.print(f"[dim]Funding rates loaded: {fcsvpath.name} ({len(funding)} entries)[/dim]")
        else:
            funding = pd.Series(dtype=float)
            console.print(f"[yellow]Funding CSV not found at {fcsvpath} — skipping funding impact[/yellow]")

        df = pd.read_csv(fpath)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Extract date range from data
        start_date = str(df["timestamp"].min().date())
        end_date   = str(df["timestamp"].max().date())
        label = args.label or f"BTC {start_date} → {end_date}"

        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Realistic Simulation — {label}[/bold cyan]\n"
            "[dim]Intrabar SL/TP  |  Slippage 0.05%  |  Fee 0.05%/side  |  "
            "Real funding rates[/dim]\n"
            f"[dim]EMA({EMA_PERIOD}) {EMA_FILTER}  |  "
            f"Nakshatra filter: {'ON' if NK_FILTER else 'OFF'}  |  "
            f"Cadence: {cadence}h[/dim]",
            border_style="cyan",
        ))

        risk_levels = [float(x) / 100 for x in args.risk_levels.split(",")]

        if args.compare_risk:
            run_risk_comparison(
                {label: df}, funding,
                cadence=cadence,
                risk_levels=risk_levels,
                label_prefix=label,
            )
            return

        stats = simulate(df, funding, cadence=cadence,
                         use_intrabar=True, use_fees=True,
                         use_slippage=True, use_funding=True)
        print_single_period(stats, label, cadence, start_date, end_date)

        if args.compare_risk:
            run_risk_comparison(
                {label: df}, funding,
                cadence=cadence,
                risk_levels=risk_levels,
                label_prefix=label,
            )
        return

    cadences  = [1, 2, 4, 8] if args.compare else [args.cadence]
    years     = [2022, 2023, 2024, 2025]
    funding   = load_funding_rates()

    if funding.empty:
        console.print("[yellow]Warning: funding rate CSV not found — funding impact skipped.[/yellow]")

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Realistic 4-Year Backtest[/bold cyan]\n"
        "[dim]Proper N-hour OHLC candles  |  Intrabar SL/TP  |  "
        "Slippage 0.05%  |  Fee 0.05%/side  |  Real funding rates[/dim]\n"
        f"[dim]EMA({EMA_PERIOD}) {EMA_FILTER}  |  "
        f"Nakshatra filter: {'ON' if NK_FILTER else 'OFF'}  |  "
        f"Cadence: {', '.join(f'{c}h' for c in cadences)}[/dim]",
        border_style="cyan",
    ))

    # Load all CSVs once
    raw_dfs = {}
    for y in years:
        path = Path(f"logs/backtest_BTC_{y}-01-01_{y}-12-31.csv")
        if not path.exists():
            console.print(f"[red]Missing {path} — skipping {y}[/red]")
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        raw_dfs[y] = df

    # Run simulations
    all_results = {}
    for c in cadences:
        all_results[c] = {}
        for y, df in raw_dfs.items():
            console.print(f"[dim]Simulating {y} @ {c}h...[/dim]", end="\r")
            all_results[c][y] = simulate(df, funding, cadence=c,
                                         use_intrabar=True, use_fees=True,
                                         use_slippage=True, use_funding=True)
    console.print(" " * 50, end="\r")

    if args.compare:
        # ── Full cadence comparison table ──────────────────────────────────
        print_cadence_compare(all_results)
        console.print()

    # ── Detailed table for chosen cadence(s) ──────────────────────────────
    for c in cadences:
        results = all_results[c]
        if not args.compare or c == cadences[-1]:
            console.rule(f"[bold]Detailed: {c}h candles[/bold]")
            print_summary_table(results,
                f"Realistic {c}h — Intrabar + Fees + Slippage + Funding")
            console.print()
            print_monthly_table(results)
            console.print()
            # Aggregate capital efficiency across all years for this cadence
            all_trades = [t for s in results.values() for t in s.get("trade_list", [])]
            total_b    = sum(s.get("total_bars", 0)    for s in results.values())
            b_in_trade = sum(s.get("bars_in_trade", 0) for s in results.values())
            agg = {
                "time_in_market_pct": b_in_trade / total_b * 100 if total_b else 0,
                "total_bars": total_b,
                "bars_in_trade": b_in_trade,
                "avg_hold_bars": float(np.mean([t["bars"] for t in all_trades])) if all_trades else 0,
                "avg_hold_hours": float(np.mean([t["bars"] for t in all_trades])) * c if all_trades else 0,
                "avg_notional": float(np.mean([t["notional"] for t in all_trades])) if all_trades else 0,
                "avg_margin": float(np.mean([t["notional"] for t in all_trades])) / getattr(__import__("config"), "LEVERAGE", 3) if all_trades else 0,
                "avg_margin_pct": float(np.mean([t["notional"] for t in all_trades])) / getattr(__import__("config"), "LEVERAGE", 3) / CAPITAL * 100 if all_trades else 0,
                "total_risk_deployed": sum(t["risk"] for t in all_trades),
                "total": sum(s.get("total", 0) for s in results.values()),
                "return_on_risk": sum(s.get("total", 0) for s in results.values()) / max(sum(t["risk"] for t in all_trades), 1) * 100,
                "idle_yield_est": sum(s.get("idle_yield_est", 0) for s in results.values()),
                "cadence": c,
            }
            print_capital_efficiency(agg, f"4-Year Aggregate — {c}h cadence")

    # ── Compounded growth panel ────────────────────────────────────────────
    for c in cadences:
        results = all_results[c]
        compound = CAPITAL
        steps = ""
        eq = CAPITAL
        for y, s in results.items():
            prev = eq
            eq  *= (1 + s["pnl_pct"] / 100)
            col  = "green" if eq >= prev else "red"
            steps += (f"  {y}  ${prev:>10,.0f}  →  "
                      f"[{col}]${eq:>10,.0f}[/{col}]  ({s['pnl_pct']:+.1f}%)\n")
            compound = eq

        gain     = compound - CAPITAL
        cagr, _  = compounded_growth(results)
        worst_dd = max(s["dd_pct"] for s in results.values())
        gc       = "green" if gain >= 0 else "red"

        console.print(Panel(
            f"[bold]Compounded growth — {c}h cadence (starting $10,000):[/bold]\n\n"
            + steps +
            f"\n  [bold {gc}]4-year end balance:  ${compound:,.0f}[/bold {gc}]\n"
            f"  [bold {gc}]Total gain:          +${gain:,.0f}"
            f"  (+{gain/CAPITAL*100:.1f}%)[/bold {gc}]\n"
            f"  [bold {gc}]CAGR:                {cagr:.1f}% per year[/bold {gc}]\n"
            f"  [yellow]Worst drawdown:      {worst_dd:.1f}%[/yellow]",
            title=f"[bold]Compounded 4-Year Growth — {c}h[/bold]",
            border_style="green" if gain >= 0 else "red",
        ))
        console.print()

    # ── Risk comparison (4-year mode) ──────────────────────────────────────
    if args.compare_risk:
        risk_levels = [float(x) / 100 for x in args.risk_levels.split(",")]
        c = cadences[-1]   # use the last (or only) cadence chosen
        console.rule(f"[bold]Risk per Trade Comparison — {c}h cadence[/bold]")
        run_risk_comparison(
            raw_dfs, funding,
            cadence=c,
            risk_levels=risk_levels,
            label_prefix=f"4-Year ({', '.join(str(y) for y in raw_dfs.keys())})",
        )


if __name__ == "__main__":
    main()
