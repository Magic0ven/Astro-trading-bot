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
    python scripts/final_backtest.py                  # default 4h cadence
    python scripts/final_backtest.py --cadence 1      # 1h candles
    python scripts/final_backtest.py --compare        # all cadences side-by-side
    python scripts/final_backtest.py --long-short-periods   # 2022-2025 vs 2026-YTD, long vs short (shorts negative highlight)
    python scripts/final_backtest.py --year 2026      # 2026 YTD through 17 Mar; real-world macro, SL/TP, funding, market fulfillment
    python scripts/final_backtest.py --year 2026 --compare-nakshatra   # Nakshatra OFF vs ON, detailed long/short table (2026 YTD)
    python scripts/final_backtest.py --compare-nakshatra               # same comparison over 2022–2025
    python scripts/final_backtest.py --cadence-min 15 --long-short-periods   # 15m cadence, 2022–2025 + 2026 YTD (requires *_15m.csv from backtest.py --interval-min 15)
    python scripts/final_backtest.py --compare-intrahour --log logs/cadence_comparison_15m_30m_1h.txt   # 15m vs 30m vs 1h, output to log file
"""
import argparse
import atexit
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
try:
    from core.trend_predictor import predictor_actions
except ImportError:
    predictor_actions = None  # optional; used only for --compare-predictor

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
NK_FILTER         = config.NAKSHATRA_FILTER    # True / False
MERCURY_RX_BLOCK  = getattr(config, "MERCURY_RX_BLOCK", True)
SATURN_RX_BLOCK   = getattr(config, "SATURN_RX_BLOCK", True)
# Short-block: skip SHORT when nakshatra/events historically lose (backtest 2022–2025)
SHORT_BLOCK_NAKSHATRAS   = getattr(config, "SHORT_BLOCK_NAKSHATRAS", frozenset())
SHORT_BLOCK_JUPITER_URANUS = getattr(config, "SHORT_BLOCK_JUPITER_URANUS", True)
SHORT_BLOCK_NEW_MOON      = getattr(config, "SHORT_BLOCK_NEW_MOON", True)
SHORT_BLOCK_MERCURY_RX    = getattr(config, "SHORT_BLOCK_MERCURY_RX", True)
TAKER_FEE         = 0.0005        # 0.05% per side (Hyperliquid taker)
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


def cadence_label(cadence) -> str:
    """Display string for cadence (hours or minutes). cadence in hours (e.g. 0.25 = 15m, 0.5 = 30m)."""
    if isinstance(cadence, (int, float)) and cadence < 1 and cadence > 0:
        return f"{int(round(cadence * 60))}m"
    return f"{int(cadence)}h"


def resample_to_nh(df: pd.DataFrame, hours) -> pd.DataFrame:
    """
    Aggregate a 1h-resolution DataFrame into N-hour OHLC candles.
    When hours < 1 (e.g. 0.25 for 15m), no resampling — data is already at that resolution.

    OHLC: open=first, high=MAX, low=MIN, close=last  (captures every wick)
    Signals: taken from the FIRST row of each window  (what the bot sees
             when it wakes up at the start of the N-hour period)
    """
    if hours == 1 or (isinstance(hours, (int, float)) and hours < 1):
        return df.copy()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    signal_cols = [c for c in
                   ["action", "nakshatra", "western_score", "vedic_score",
                    "western_slope", "vedic_slope", "resonance_day",
                    "full_moon_active", "new_moon_active",
                    "jupiter_uranus_active", "saturn_pluto_active",
                    "mercury_retrograde_active", "saturn_retrograde_active",
                    "moon_phase_deg"]
                   if c in df.columns]

    freq = f"{int(hours)}h" if hours >= 1 else f"{int(hours * 60)}min"

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


def parse_int_float_map(spec: str) -> dict:
    """
    Parse "k=v,k=v" into {int(k): float(v)}. Ignores invalid tokens.
    Example: "1=1.5,3=1.5,5=1.25" -> {1: 1.5, 3: 1.5, 5: 1.25}
    """
    out = {}
    if not spec:
        return out
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        k_s, v_s = tok.split("=", 1)
        try:
            k = int(k_s.strip())
            v = float(v_s.strip())
        except Exception:
            continue
        if k <= 0 or v <= 0:
            continue
        out[k] = v
    return out


def parse_weekday_map(spec: str) -> dict:
    """
    Parse "mon=1.1,sun=1.2" into {0:1.1, 6:1.2} where Monday=0 ... Sunday=6.
    """
    if not spec:
        return {}
    name_to_idx = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "weds": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    out = {}
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        k_s, v_s = tok.split("=", 1)
        k = name_to_idx.get(k_s.strip().lower())
        if k is None:
            continue
        try:
            v = float(v_s.strip())
        except Exception:
            continue
        if v <= 0:
            continue
        out[int(k)] = v
    return out


def simulate(df: pd.DataFrame, funding: pd.Series,
             cadence = INTERVAL,
             use_intrabar: bool = True,
             use_fees: bool = True,
             use_slippage: bool = True,
             use_funding: bool = True,
             ema_period: int = EMA_PERIOD,
             ema_filter: str = EMA_FILTER,
             nk_filter: bool = NK_FILTER,
             mercury_rx_block: bool = MERCURY_RX_BLOCK,
             saturn_rx_block: bool = SATURN_RX_BLOCK,
             risk_pct: float = RISK_PCT,
             starting_capital: float = CAPITAL,
             rr_ratio: float = None,
             book_profit_at_r: float = 0,
             allowed_entry_days: set = None,
             day_risk_mult: dict = None,
             weekday_risk_mult: dict = None,
             buy_day_risk_mult: dict = None,
             sell_day_risk_mult: dict = None,
             buy_weekday_risk_mult: dict = None,
             sell_weekday_risk_mult: dict = None) -> dict:
    """
    allowed_entry_days: if provided, only open new trades on candles whose
    calendar day-of-month is in this set (e.g. {1, 3, 7, 9}).
    Open trades are always allowed to continue / exit on any day.
    book_profit_at_r: if > 0, close trade when unrealized PnL >= this many R.
    """
    if rr_ratio is None:
        rr_ratio = RR_RATIO
    day_risk_mult = day_risk_mult or {}
    weekday_risk_mult = weekday_risk_mult or {}
    buy_day_risk_mult = buy_day_risk_mult or {}
    sell_day_risk_mult = sell_day_risk_mult or {}
    buy_weekday_risk_mult = buy_weekday_risk_mult or {}
    sell_weekday_risk_mult = sell_weekday_risk_mult or {}

    # Resample to the requested cadence so ATR and intrabar highs/lows
    # reflect N-hour candles, not 1h candles. (cadence can be float for 15m = 0.25)
    df     = resample_to_nh(df, cadence)
    max_open_bars = max(1, int(48 / cadence))   # always ~48h max hold

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
    equity_curve  = []   # (timestamp, equity) sampled every bar

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

            # Early book-profit: close when unrealized PnL >= book_profit_at_r R
            book_profit_hit = False
            if book_profit_at_r > 0 and not sl_hit:
                # R-multiple profit level from entry
                one_r_dist = abs(ot["entry"] - sl)
                if one_r_dist <= 0:
                    one_r_dist = ot["entry"] * 0.01
                if ot["side"] == "BUY":
                    profit_price = ot["entry"] + book_profit_at_r * one_r_dist
                    if use_intrabar and has_hl:
                        book_profit_hit = high >= profit_price
                    else:
                        book_profit_hit = close >= profit_price
                else:
                    profit_price = ot["entry"] - book_profit_at_r * one_r_dist
                    if use_intrabar and has_hl:
                        book_profit_hit = low <= profit_price
                    else:
                        book_profit_hit = close <= profit_price

            # Resolve exit
            def _rec(result, pnl):
                out = {
                    "pnl": pnl, "result": result,
                    "month": ot["month"], "side": ot["side"],
                    "signal": ot["signal"], "entry": ot["entry"],
                    "open_ts": ot["open_ts"],
                    "bars": age + 1, "notional": ot["notional"],
                    "risk": ot["risk"],
                }
                for k in ("nakshatra", "resonance_day", "western_score", "vedic_score",
                         "western_slope", "vedic_slope", "full_moon_active", "new_moon_active",
                         "jupiter_uranus_active", "saturn_pluto_active",
                         "mercury_retrograde_active", "saturn_retrograde_active", "moon_phase_deg"):
                    if k in ot:
                        out[k] = ot[k]
                return out
            if sl_hit and not tp_hit:
                pnl = -ot["risk"]
                if use_fees: pnl -= ot["notional"] * TAKER_FEE
                equity += pnl
                trades.append(_rec("STOP", pnl))
                ot = None
            elif tp_hit and not sl_hit:
                pnl = ot["risk"] * rr_ratio
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
            elif book_profit_hit:
                pnl = ot["risk"] * book_profit_at_r
                if use_fees: pnl -= ot["notional"] * TAKER_FEE
                equity += pnl
                trades.append(_rec("BOOK_PROFIT", pnl))
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

        equity_curve.append({"ts": ts, "equity": equity})

        # ── Signal filters ─────────────────────────────────────────────────
        if action not in ("STRONG_BUY", "STRONG_SELL"):
            continue

        # Day-of-month filter: only enter on allowed calendar days
        if allowed_entry_days and ts.day not in allowed_entry_days:
            continue
        if nk_filter and naks in BLOCKED:
            continue
        # Block longs only during Mercury/Saturn RX (shorts allowed)
        if mercury_rx_block and "BUY" in action and row.get("mercury_retrograde_active") in (True, "True", 1):
            continue
        if saturn_rx_block and "BUY" in action and row.get("saturn_retrograde_active") in (True, "True", 1):
            continue
        # Block shorts when astro conditions historically lose (align with signal_engine 7c)
        if "SELL" in action and SHORT_BLOCK_NAKSHATRAS:
            naks_val = (row.get("nakshatra") or "").strip() if pd.notna(row.get("nakshatra")) else ""
            ju = row.get("jupiter_uranus_active") in (True, "True", 1)
            nm = row.get("new_moon_active") in (True, "True", 1)
            mr = row.get("mercury_retrograde_active") in (True, "True", 1)
            if (naks_val in SHORT_BLOCK_NAKSHATRAS or
                (ju and SHORT_BLOCK_JUPITER_URANUS) or
                (nm and SHORT_BLOCK_NEW_MOON) or
                (mr and SHORT_BLOCK_MERCURY_RX)):
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
        dom = int(ts.day)
        dow = int(ts.dayofweek)  # Mon=0 ... Sun=6

        base_dom_mult = float(day_risk_mult.get(dom, 1.0)) if day_risk_mult else 1.0
        base_dow_mult = float(weekday_risk_mult.get(dow, 1.0)) if weekday_risk_mult else 1.0

        side_dom_mult = float((buy_day_risk_mult if side == "BUY" else sell_day_risk_mult).get(dom, 1.0))
        side_dow_mult = float((buy_weekday_risk_mult if side == "BUY" else sell_weekday_risk_mult).get(dow, 1.0))

        dom_mult = base_dom_mult * side_dom_mult
        dow_mult = base_dow_mult * side_dow_mult
        mult = dom_mult * dow_mult
        if mult <= 0:
            mult = 1.0

        risk = equity * risk_pct * mult
        sld  = at_v * ATR_MULT

        # Entry price: next open + slippage
        entry = close  # fallback
        if has_hl and use_slippage:
            entry = float(row.get("open", close))
            entry *= (1 + SLIPPAGE) if side == "BUY" else (1 - SLIPPAGE)

        sl = entry - sld if side == "BUY" else entry + sld
        tp = entry + sld * rr_ratio if side == "BUY" else entry - sld * rr_ratio

        # Notional = risk / stop_distance_pct
        stop_pct = abs(entry - sl) / entry
        notional = risk / stop_pct if stop_pct > 0 else risk * 10

        fee_cost = notional * TAKER_FEE if use_fees else 0.0
        equity  -= fee_cost   # entry fee deducted immediately

        ot = {
            "side": side, "signal": action, "entry": entry, "sl": sl, "tp": tp,
            "risk": risk, "notional": notional,
            "age": 0, "month": month, "open_ts": str(ts)[:16],
            "day_risk_mult": mult,
        }
        for k in ("nakshatra", "resonance_day", "western_score", "vedic_score",
                  "western_slope", "vedic_slope", "full_moon_active", "new_moon_active",
                  "jupiter_uranus_active", "saturn_pluto_active",
                  "mercury_retrograde_active", "saturn_retrograde_active", "moon_phase_deg"):
            if k in row.index and pd.notna(row.get(k)):
                ot[k] = row[k]

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

    # Build price / EMA series for plotting
    price_series = rows[["timestamp", "price"]].copy()
    price_series["ema"] = em.values

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
        # chart data
        "equity_curve": equity_curve,
        "price_series": price_series,
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


def print_intrahour_comparison(agg_2022_2025: dict, stats_2026: dict, cadence_configs: list):
    """Print comparison table for 15m, 30m, 1h (cadence_configs = [(0.25,'15m'), (0.5,'30m'), (1,'1h')])."""
    t = Table(
        "Cadence", "Period", "Trades", "Win %", "P&L %", "Max DD %", "End Equity",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title="[bold]15m vs 30m vs 1h — Comparison[/bold]",
        show_lines=True,
    )
    for c, label in cadence_configs:
        a = agg_2022_2025.get(c)
        s6 = stats_2026.get(c)
        if a:
            pc = "green" if a.get("pnl_pct", 0) >= 0 else "red"
            t.add_row(
                cadence_label(c), "2022–2025",
                str(a.get("trades", 0)), f"{a.get('wr', 0):.1f}%",
                f"[{pc}]{a.get('pnl_pct', 0):+.1f}%[/{pc}]",
                f"{a.get('dd_pct', 0):.1f}%",
                f"${a.get('end_equity', CAPITAL):,.0f}",
            )
        else:
            t.add_row(cadence_label(c), "2022–2025", "—", "—", "—", "—", "—")
        if s6:
            pc = "green" if s6.get("pnl_pct", 0) >= 0 else "red"
            t.add_row(
                cadence_label(c), "2026 YTD",
                str(s6.get("trades", 0)), f"{s6.get('wr', 0):.1f}%",
                f"[{pc}]{s6.get('pnl_pct', 0):+.1f}%[/{pc}]",
                f"{s6.get('dd_pct', 0):.1f}%",
                f"${s6.get('end_equity', CAPITAL):,.0f}",
            )
        else:
            t.add_row(cadence_label(c), "2026 YTD", "—", "—", "—", "—", "—")
    console.print(t)


def print_overlay_suite_table(results_by_scenario: dict, cadence_configs: list):
    """
    results_by_scenario = {
      "base": {"2022_2025": {cad: agg}, "2026": {cad: stats_or_none}},
      "dayboost": {...}, "seasonality": {...}, "combined": {...}
    }
    """
    scenarios = list(results_by_scenario.keys())
    t = Table(
        "Cadence", "Period",
        *[f"{s} P&L%" for s in scenarios],
        *[f"{s} DD%" for s in scenarios],
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title="[bold]Overlay suite — base vs dayboost vs seasonality vs combined[/bold]",
        show_lines=True,
    )

    def _cell_pnl(v):
        if v is None:
            return "—"
        pc = "green" if v >= 0 else "red"
        return f"[{pc}]{v:+.1f}%[/{pc}]"

    def _cell_dd(v):
        if v is None:
            return "—"
        return f"{v:.1f}%"

    for c, _label in cadence_configs:
        for period in ("2022–2025", "2026 YTD"):
            pnl_row = []
            dd_row = []
            for s in scenarios:
                pack = results_by_scenario[s]
                stats = (pack["2022_2025"].get(c) if period == "2022–2025" else pack["2026"].get(c))
                pnl_row.append(_cell_pnl(stats.get("pnl_pct") if stats else None))
                dd_row.append(_cell_dd(stats.get("dd_pct") if stats else None))
            t.add_row(cadence_label(c), period, *pnl_row, *dd_row)
    console.print(t)

    # Delta vs base table
    if "base" in results_by_scenario:
        dt = Table(
            "Cadence", "Period",
            *[f"{s} ΔP&L%" for s in scenarios if s != "base"],
            box=box.MINIMAL_HEAVY_HEAD,
            header_style="bold white on dark_blue",
            title="[bold]Overlay suite — delta vs base[/bold]",
            show_lines=True,
        )
        for c, _label in cadence_configs:
            for period in ("2022–2025", "2026 YTD"):
                base_stats = (results_by_scenario["base"]["2022_2025"].get(c)
                              if period == "2022–2025" else results_by_scenario["base"]["2026"].get(c))
                base_pnl = base_stats.get("pnl_pct") if base_stats else None
                deltas = []
                for s in scenarios:
                    if s == "base":
                        continue
                    stats = (results_by_scenario[s]["2022_2025"].get(c)
                             if period == "2022–2025" else results_by_scenario[s]["2026"].get(c))
                    pnl = stats.get("pnl_pct") if stats else None
                    if base_pnl is None or pnl is None:
                        deltas.append("—")
                        continue
                    d = pnl - base_pnl
                    dc = "green" if d >= 0 else "red"
                    deltas.append(f"[{dc}]{d:+.1f}%[/{dc}]")
                dt.add_row(cadence_label(c), period, *deltas)
        console.print(dt)


def _aggregate_yearly_results(results: dict) -> dict:
    """Merge per-year stats {year: stats} into one aggregate stats dict."""
    if not results:
        return {}
    all_trades = []
    total_pnl = 0.0
    total_bars = 0
    bars_in_trade = 0
    total_risk_dep = 0.0
    monthly = {}
    equity_curve = []
    for s in results.values():
        all_trades.extend(s.get("trade_list", []))
        total_pnl += s.get("total", 0)
        total_bars += s.get("total_bars", 0)
        bars_in_trade += s.get("bars_in_trade", 0)
        total_risk_dep += s.get("total_risk_deployed", 0)
        for m, md in s.get("monthly", {}).items():
            monthly.setdefault(m, {"pnl": 0, "wins": 0, "total": 0})
            monthly[m]["pnl"] += md["pnl"]
            monthly[m]["wins"] += md["wins"]
            monthly[m]["total"] += md["total"]
        equity_curve.extend(s.get("equity_curve", []))
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    n = len(all_trades)
    wr = len(wins) / n * 100 if n else 0
    avg_w = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
    avg_l = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    exp = (wr / 100 * avg_w + (1 - wr / 100) * avg_l) if n else 0.0
    pnl_pct = total_pnl / CAPITAL * 100
    end_equity = CAPITAL + total_pnl
    eq, pk, dd = CAPITAL, CAPITAL, 0.0
    for t in all_trades:
        eq += t["pnl"]
        pk = max(pk, eq)
        dd = max(dd, pk - eq)
    dd_pct = dd / CAPITAL * 100
    c = list(results.values())[0].get("cadence", 4)
    avg_hold_bars = float(np.mean([t["bars"] for t in all_trades])) if all_trades else 0
    avg_notional = float(np.mean([t["notional"] for t in all_trades])) if all_trades else 0
    lev = getattr(__import__("config"), "LEVERAGE", 3)
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "wr": wr, "avg_win": avg_w, "avg_loss": avg_l, "exp": exp,
        "total": total_pnl, "pnl_pct": pnl_pct, "end_equity": end_equity,
        "max_dd": dd, "dd_pct": dd_pct, "trade_list": all_trades, "monthly": monthly,
        "time_in_market_pct": bars_in_trade / total_bars * 100 if total_bars else 0,
        "total_bars": total_bars, "bars_in_trade": bars_in_trade,
        "avg_hold_bars": avg_hold_bars, "avg_hold_hours": avg_hold_bars * c,
        "avg_notional": avg_notional, "avg_margin": avg_notional / lev,
        "avg_margin_pct": avg_notional / lev / CAPITAL * 100,
        "total_risk_deployed": total_risk_dep,
        "return_on_risk": total_pnl / total_risk_dep * 100 if total_risk_dep else 0,
        "idle_yield_est": sum(s.get("idle_yield_est", 0) for s in results.values()),
        "cadence": c, "equity_curve": equity_curve,
    }


def print_monthly_days_report(stats_all: dict, stats_days: dict,
                               day_set: set, period_label: str, cadence: int):
    """
    Side-by-side: all-days (normal) vs restricted to specific calendar days.
    Shows overall stats + long/short breakdown for both.
    """
    days_label = ", ".join(str(d) for d in sorted(day_set))
    a, d = stats_all, stats_days

    def _row(label, key, fmt="{}", higher_better=True):
        va = a.get(key, 0)
        vd = d.get(key, 0)
        diff = vd - va
        if key == "dd_pct":
            diff = -diff  # lower DD is better
        dc = "green" if (diff >= 0 and higher_better) or (diff < 0 and not higher_better) else "red"
        try:
            return (label, fmt.format(va), fmt.format(vd), f"[{dc}]{diff:+.1f}[/{dc}]")
        except Exception:
            return (label, str(va), str(vd), "")

    # ── Overall comparison ────────────────────────────────────────────────────
    t = Table(
        "Metric", "All days", f"Days {days_label}", "Δ",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title=f"[bold]All days vs Day-of-month filter ({days_label}) — {period_label}[/bold]",
        show_lines=True,
    )
    for row_args in [
        _row("Trades",           "trades",   "{:.0f}",  True),
        _row("Win rate %",       "wr",        "{:.1f}",  True),
        _row("Total P&L $",      "total",     "${:+,.0f}", True),
        _row("P&L %",            "pnl_pct",   "{:+.1f}%", True),
        _row("End equity $",     "end_equity","${:,.0f}", True),
        _row("Max drawdown %",   "dd_pct",    "{:.1f}%",  False),
        _row("Avg win $",        "avg_win",   "${:,.0f}", True),
        _row("Avg loss $",       "avg_loss",  "${:,.0f}", False),
        _row("Expectancy $",     "exp",       "${:+,.0f}", True),
    ]:
        t.add_row(*row_args)
    console.print(t)

    # ── Long vs short for the day-filtered run ────────────────────────────────
    console.rule(f"[bold]Day-filter ({days_label}) — Long vs Short breakdown[/bold]")
    print_long_short_report(f"Days {days_label} — {period_label}", stats_days, cadence)


def _long_short_breakdown(trade_list: list) -> dict:
    """Split trade_list by side (BUY=long, SELL=short). Return counts, P&L, win rate per side."""
    longs  = [t for t in trade_list if t.get("side") == "BUY"]
    shorts = [t for t in trade_list if t.get("side") == "SELL"]
    def _side_stats(trades):
        if not trades:
            return {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "wr": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        pnl = sum(t["pnl"] for t in trades)
        wr = len(wins) / len(trades) * 100 if trades else 0
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "pnl": pnl,
            "wr": wr,
            "avg_win": float(np.mean([t["pnl"] for t in wins])) if wins else 0.0,
            "avg_loss": float(np.mean([t["pnl"] for t in losses])) if losses else 0.0,
        }
    return {
        "long":  _side_stats(longs),
        "short": _side_stats(shorts),
        "long_trades": longs,
        "short_trades": shorts,
    }


def print_long_short_report(period_label: str, stats: dict, cadence: int = 4):
    """Print overall stats and long vs short breakdown; highlight when shorts are mostly negative."""
    trade_list = stats.get("trade_list", [])
    breakdown = _long_short_breakdown(trade_list)
    long_s, short_s = breakdown["long"], breakdown["short"]

    console.rule(f"[bold]{period_label}[/bold]")
    # Overall
    t = Table(
        "Metric", "All", "Long only", "Short only",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title=f"[bold]Long vs Short — {period_label}[/bold]",
        show_lines=True,
    )
    t.add_row("Trades", str(stats.get("trades", 0)), str(long_s["trades"]), str(short_s["trades"]))
    t.add_row("Wins", str(stats.get("wins", 0)), str(long_s["wins"]), str(short_s["wins"]))
    t.add_row("Losses", str(stats.get("losses", 0)), str(long_s["losses"]), str(short_s["losses"]))
    t.add_row("Win %", f"{stats.get('wr', 0):.1f}%", f"{long_s['wr']:.1f}%", f"{short_s['wr']:.1f}%")
    total = stats.get("total", 0)
    long_pnl_s = f"{long_s['pnl']:+,.0f}"
    short_pnl_s = f"{short_s['pnl']:+,.0f}"
    if long_s["pnl"] >= 0:
        long_pnl_s = f"[green]{long_pnl_s}[/green]"
    else:
        long_pnl_s = f"[red]{long_pnl_s}[/red]"
    if short_s["pnl"] >= 0:
        short_pnl_s = f"[green]{short_pnl_s}[/green]"
    else:
        short_pnl_s = f"[red]{short_pnl_s}[/red]"
    t.add_row("Total P&L ($)", f"{total:+,.0f}", long_pnl_s, short_pnl_s)
    t.add_row("P&L %", f"{stats.get('pnl_pct', 0):.1f}%",
              f"{(long_s['pnl']/CAPITAL*100):+.1f}%",
              f"{(short_s['pnl']/CAPITAL*100):+.1f}%")
    t.add_row("Max DD %", f"{stats.get('dd_pct', 0):.1f}%", "—", "—")
    console.print(t)

    short_negative = short_s["pnl"] < 0 and short_s["trades"] > 0
    if short_negative:
        console.print(Panel(
            f"  [bold]Shorts are mostly negative[/bold] in this period: "
            f"{short_s['trades']} short trades, total P&L [red]{short_s['pnl']:+,.0f}[/red] "
            f"(win rate {short_s['wr']:.0f}%).\n"
            f"  Longs: {long_s['trades']} trades, P&L [{'green' if long_s['pnl']>=0 else 'red'}]{long_s['pnl']:+,.0f}[/].",
            title="[bold]Short performance[/bold]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            f"  Shorts contributed [green]{short_s['pnl']:+,.0f}[/green] in this period "
            f"({short_s['trades']} trades, {short_s['wr']:.0f}% win rate).",
            title="[bold]Short performance[/bold]",
            border_style="green",
        ))
    console.print()


# Astro dimensions we store per trade (for short-by-astro breakdown)
ASTRO_DIMENSIONS = [
    "nakshatra",
    "full_moon_active",
    "new_moon_active",
    "jupiter_uranus_active",
    "saturn_pluto_active",
    "mercury_retrograde_active",
    "saturn_retrograde_active",
]


def _short_pnl_by_astro(short_trades: list) -> dict:
    """Group short trades by each astro dimension. Return {dim: [(value, count, pnl, wr), ...]}."""
    if not short_trades:
        return {}
    out = {}
    for dim in ASTRO_DIMENSIONS:
        groups = {}
        for t in short_trades:
            raw = t.get(dim)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                val = "—"
            elif raw in (True, "True", 1, "1"):
                val = "Yes"
            elif raw in (False, "False", 0, "0"):
                val = "No"
            else:
                val = str(raw).strip()
            groups.setdefault(val, []).append(t)
        rows = []
        for val, trades in groups.items():
            pnl = sum(x["pnl"] for x in trades)
            wins = sum(1 for x in trades if x["pnl"] > 0)
            wr = wins / len(trades) * 100 if trades else 0
            rows.append((val, len(trades), pnl, wr))
        rows.sort(key=lambda r: r[2])  # sort by P&L ascending (most negative first)
        out[dim] = rows
    return out


def _exp_for_side(s: dict) -> float:
    """Expectancy for a side stats dict: wr/100*avg_win + (1-wr/100)*avg_loss."""
    if s["trades"] == 0:
        return 0.0
    return (s["wr"] / 100 * s["avg_win"]) + ((1 - s["wr"] / 100) * s["avg_loss"])


def print_nakshatra_comparison(stats_off: dict, stats_on: dict, label: str, cadence: int = 4):
    """Very detailed comparison: Nakshatra filter OFF vs ON, with full long/short breakdown for both."""
    def _break(stats):
        b = _long_short_breakdown(stats.get("trade_list", []))
        all_s = {
            "trades": stats.get("trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "wr": stats.get("wr", 0),
            "pnl": stats.get("total", 0),
            "pnl_pct": stats.get("pnl_pct", 0),
            "avg_win": stats.get("avg_win", 0),
            "avg_loss": stats.get("avg_loss", 0),
            "exp": stats.get("exp", 0),
            "end_equity": stats.get("end_equity", CAPITAL),
            "dd_pct": stats.get("dd_pct", 0),
        }
        long_s = b["long"]
        short_s = b["short"]
        long_s["exp"] = _exp_for_side(long_s)
        short_s["exp"] = _exp_for_side(short_s)
        long_s["pnl_pct"] = (long_s["pnl"] / CAPITAL * 100) if CAPITAL else 0
        short_s["pnl_pct"] = (short_s["pnl"] / CAPITAL * 100) if CAPITAL else 0
        return all_s, long_s, short_s

    off_all, off_long, off_short = _break(stats_off)
    on_all, on_long, on_short = _break(stats_on)

    console.rule(f"[bold]Nakshatra filter OFF vs ON — {label}[/bold]")
    # Detailed table: 6 columns (OFF All, OFF Long, OFF Short | ON All, ON Long, ON Short)
    cols = [
        "Nakshatra OFF\n(All)",
        "Nakshatra OFF\n(Long)",
        "Nakshatra OFF\n(Short)",
        "Nakshatra ON\n(All)",
        "Nakshatra ON\n(Long)",
        "Nakshatra ON\n(Short)",
    ]
    t = Table(
        "Metric",
        *cols,
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title="[bold]Detailed comparison — Long & Short in both modes[/bold]",
        show_lines=True,
    )
    row_data = [
        ("Trades", str(off_all["trades"]), str(off_long["trades"]), str(off_short["trades"]),
         str(on_all["trades"]), str(on_long["trades"]), str(on_short["trades"])),
        ("Wins", str(off_all["wins"]), str(off_long["wins"]), str(off_short["wins"]),
         str(on_all["wins"]), str(on_long["wins"]), str(on_short["wins"])),
        ("Losses", str(off_all["losses"]), str(off_long["losses"]), str(off_short["losses"]),
         str(on_all["losses"]), str(on_long["losses"]), str(on_short["losses"])),
        ("Win %", f"{off_all['wr']:.1f}%", f"{off_long['wr']:.1f}%", f"{off_short['wr']:.1f}%",
         f"{on_all['wr']:.1f}%", f"{on_long['wr']:.1f}%", f"{on_short['wr']:.1f}%"),
        ("Total P&L ($)",
         f"{off_all['pnl']:+,.0f}", f"{off_long['pnl']:+,.0f}", f"{off_short['pnl']:+,.0f}",
         f"{on_all['pnl']:+,.0f}", f"{on_long['pnl']:+,.0f}", f"{on_short['pnl']:+,.0f}"),
        ("P&L %",
         f"{off_all['pnl_pct']:+.1f}%", f"{off_long['pnl_pct']:+.1f}%", f"{off_short['pnl_pct']:+.1f}%",
         f"{on_all['pnl_pct']:+.1f}%", f"{on_long['pnl_pct']:+.1f}%", f"{on_short['pnl_pct']:+.1f}%"),
        ("Avg Win ($)",
         f"{off_all['avg_win']:,.0f}", f"{off_long['avg_win']:,.0f}", f"{off_short['avg_win']:,.0f}",
         f"{on_all['avg_win']:,.0f}", f"{on_long['avg_win']:,.0f}", f"{on_short['avg_win']:,.0f}"),
        ("Avg Loss ($)",
         f"{off_all['avg_loss']:,.0f}", f"{off_long['avg_loss']:,.0f}", f"{off_short['avg_loss']:,.0f}",
         f"{on_all['avg_loss']:,.0f}", f"{on_long['avg_loss']:,.0f}", f"{on_short['avg_loss']:,.0f}"),
        ("Expectancy ($)",
         f"{off_all['exp']:+,.0f}", f"{off_long['exp']:+,.0f}", f"{off_short['exp']:+,.0f}",
         f"{on_all['exp']:+,.0f}", f"{on_long['exp']:+,.0f}", f"{on_short['exp']:+,.0f}"),
        ("End Equity ($)",
         f"{off_all['end_equity']:,.0f}", "—", "—",
         f"{on_all['end_equity']:,.0f}", "—", "—"),
        ("Max DD %",
         f"{off_all['dd_pct']:.1f}%", "—", "—",
         f"{on_all['dd_pct']:.1f}%", "—", "—"),
    ]
    for row in row_data:
        metric = row[0]
        vals = [str(x) for x in row[1:]]
        # Color P&L and Win % columns
        if "P&L" in metric or "Expectancy" in metric or "End Equity" in metric:
            for i, v in enumerate(vals):
                if v in ("—", ""):
                    continue
                try:
                    num = float(v.replace(",", "").replace("+", "").replace("%", "").replace("$", ""))
                    if num < 0:
                        vals[i] = f"[red]{v}[/red]"
                    elif num > 0 and ("P&L" in metric or "Expectancy" in metric or "Equity" in metric):
                        vals[i] = f"[green]{v}[/green]"
                except ValueError:
                    pass
        t.add_row(metric, *vals)
    console.print(t)
    # Verdict panel
    better_equity = "Nakshatra ON" if on_all["end_equity"] >= off_all["end_equity"] else "Nakshatra OFF"
    console.print(Panel(
        f"  [bold]Higher end equity:[/bold] [green]{better_equity}[/green]\n"
        f"  Nakshatra OFF: {off_all['trades']} trades (Long: {off_long['trades']}, Short: {off_short['trades']})  "
        f"P&L [{'green' if off_all['pnl']>=0 else 'red'}]{off_all['pnl']:+,.0f}[/]  DD {off_all['dd_pct']:.1f}%\n"
        f"  Nakshatra ON:  {on_all['trades']} trades (Long: {on_long['trades']}, Short: {on_short['trades']})  "
        f"P&L [{'green' if on_all['pnl']>=0 else 'red'}]{on_all['pnl']:+,.0f}[/]  DD {on_all['dd_pct']:.1f}%",
        title="[bold]Verdict[/bold]",
        border_style="green",
    ))
    console.print()


def print_short_astro_report(period_label: str, trade_list: list):
    """Break down SHORT trades by astro condition; highlight where shorts are mostly negative."""
    shorts = [t for t in trade_list if t.get("side") == "SELL"]
    if not shorts:
        console.print(f"[dim]No short trades in {period_label} — skipping astro breakdown.[/dim]\n")
        return
    by_astro = _short_pnl_by_astro(shorts)
    console.rule(f"[bold]Shorts by astro — {period_label}[/bold]")
    negative_conditions = []
    for dim in ASTRO_DIMENSIONS:
        rows = by_astro.get(dim, [])
        if not rows:
            continue
        dim_label = dim.replace("_", " ").title()
        t = Table(
            dim_label, "Trades", "Short P&L ($)", "Win %",
            box=box.SIMPLE,
            header_style="bold",
            title=f"[bold]{dim_label}[/bold]",
        )
        for val, count, pnl, wr in rows:
            pnl_str = f"{pnl:+,.0f}"
            if pnl < 0:
                pnl_str = f"[red]{pnl_str}[/red]"
                negative_conditions.append((dim, val, count, pnl, wr))
            else:
                pnl_str = f"[green]{pnl_str}[/green]"
            t.add_row(str(val), str(count), pnl_str, f"{wr:.0f}%")
        console.print(t)
        console.print()
    if negative_conditions:
        console.print(Panel(
            "\n".join(
                f"  • [bold]{d}[/bold] = [red]{v}[/red]: {c} shorts, P&L [red]{p:+,.0f}[/red] ({wr:.0f}% WR)"
                for d, v, c, p, wr in negative_conditions[:15]
            )
            + ("\n  …" if len(negative_conditions) > 15 else ""),
            title="[bold]Astro conditions where shorts are negative[/bold]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            "  No astro bucket has negative short P&L in this period.",
            title="[bold]Shorts by astro[/bold]",
            border_style="green",
        ))
    console.print()


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

    atr_note = {1: "1h ≈ narrow", 2: "2h", 4: "4h ≈ wide", 8: "8h ≈ widest", 0.25: "15m", 0.5: "30m"}

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
            f"[bold]{cadence_label(c)}[/bold]",
            atr_note.get(c, cadence_label(c)),
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
        f"     Avg hold       :  {hold_h:.1f}h  ({hold_b:.1f} × {cadence_label(cad)} candles)\n\n"
        f"  [bold]2. Margin Efficiency[/bold]\n"
        f"     Avg notional   :  ${stats['avg_notional']:,.0f}  per trade\n"
        f"     Avg margin     :  ${stats['avg_margin']:,.0f}  per trade  "
        f"([{mc}]{m_pct:.1f}% of capital locked[/{mc}])\n"
        f"     Leverage used  :  {stats['avg_notional']/(stats['avg_margin'] or 1):.0f}x  "
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


def print_overlay_comparison(agg_with: dict, agg_without: dict, cadence: int = 4):
    """Side-by-side: with decisive overlay (best-call rules) vs without (gate only)."""
    t = Table(
        "Metric", "Without overlay (gate only)", "With overlay (best-call)",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title=f"[bold]Decisive overlay comparison — {cadence_label(cadence)} cadence[/bold]",
        show_lines=True,
    )
    for (name, key) in [
        ("Trades", "trades"),
        ("Wins", "wins"),
        ("Win %", "wr"),
        ("Total P&L ($)", "total"),
        ("P&L %", "pnl_pct"),
        ("Max DD %", "dd_pct"),
        ("End Equity ($)", "end_equity"),
        ("Expectancy/trade ($)", "exp"),
    ]:
        v_no = agg_without.get(key, 0)
        v_yes = agg_with.get(key, 0)
        if key in ("wr", "pnl_pct", "dd_pct"):
            v_no_s = f"{v_no:.1f}%" if key == "wr" else f"{v_no:.2f}%" if key == "pnl_pct" else f"{v_no:.1f}%"
            v_yes_s = f"{v_yes:.1f}%" if key == "wr" else f"{v_yes:.2f}%" if key == "pnl_pct" else f"{v_yes:.1f}%"
        elif key in ("total", "exp"):
            v_no_s = f"{v_no:+,.0f}"
            v_yes_s = f"{v_yes:+,.0f}"
        elif key == "end_equity":
            v_no_s = f"{v_no:,.0f}"
            v_yes_s = f"{v_yes:,.0f}"
        else:
            v_no_s = str(v_no)
            v_yes_s = str(v_yes)
        better = (key in ("total", "pnl_pct", "end_equity", "exp", "wr") and v_yes >= v_no) or (key == "dd_pct" and v_yes <= v_no)
        c = "green" if better else "red"
        t.add_row(name, v_no_s, f"[{c}]{v_yes_s}[/{c}]")
    console.print(t)
    n = 4
    cagr_no = ((agg_without["end_equity"] / CAPITAL) ** (1 / n) - 1) * 100
    cagr_yes = ((agg_with["end_equity"] / CAPITAL) ** (1 / n) - 1) * 100
    console.print(f"\n  [dim]CAGR (4y):  Without overlay: {cagr_no:.1f}%   |   With overlay: {cagr_yes:.1f}%[/dim]")
    winner = "With overlay" if agg_with["end_equity"] >= agg_without["end_equity"] else "Without overlay"
    console.print(Panel(f"  [bold]Higher end equity:[/bold] [green]{winner}[/green]", title="[bold]Verdict[/bold]", border_style="green"))
    console.print()


def print_book_profit_comparison(stats_no: dict, stats_yes: dict,
                                  label: str, cadence: int, book_r: float):
    """Side-by-side: no early book profit vs book profit at book_r R."""
    t = Table(
        "Metric", "No early book profit", f"Book profit @ {book_r}R",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold white on dark_blue",
        title=f"[bold]Early profit booking comparison — {label}[/bold]",
        show_lines=True,
    )
    for (name, k_no, k_yes) in [
        ("Trades", "trades", "trades"),
        ("Wins", "wins", "wins"),
        ("Losses", "losses", "losses"),
        ("Win %", "wr", "wr"),
        ("Total P&L ($)", "total", "total"),
        ("P&L %", "pnl_pct", "pnl_pct"),
        ("Max DD %", "dd_pct", "dd_pct"),
        ("End Equity ($)", "end_equity", "end_equity"),
        ("Expectancy/trade ($)", "exp", "exp"),
    ]:
        v_no = stats_no.get(k_no, 0)
        v_yes = stats_yes.get(k_yes, 0)
        if k_no in ("wr", "pnl_pct", "dd_pct"):
            v_no_s = f"{v_no:.1f}%" if k_no == "wr" else f"{v_no:.2f}%" if k_no == "pnl_pct" else f"{v_no:.1f}%"
            v_yes_s = f"{v_yes:.1f}%" if k_no == "wr" else f"{v_yes:.2f}%" if k_no == "pnl_pct" else f"{v_yes:.1f}%"
        elif k_no in ("total", "exp"):
            v_no_s = f"{v_no:+,.0f}"
            v_yes_s = f"{v_yes:+,.0f}"
        elif k_no == "end_equity":
            v_no_s = f"{v_no:,.0f}"
            v_yes_s = f"{v_yes:,.0f}"
        else:
            v_no_s = str(v_no)
            v_yes_s = str(v_yes)
        diff_ok = (k_no in ("total", "pnl_pct", "end_equity", "exp", "wr") and v_yes >= v_no) or \
                  (k_no == "dd_pct" and v_yes <= v_no)
        c = "green" if diff_ok else "red"
        t.add_row(name, v_no_s, f"[{c}]{v_yes_s}[/{c}]")
    console.print(t)

    # Count BOOK_PROFIT exits in stats_yes
    trades_yes = stats_yes.get("trade_list", [])
    n_book = sum(1 for tr in trades_yes if tr.get("result") == "BOOK_PROFIT")
    if n_book:
        console.print(f"\n[dim]With early book profit: {n_book} trades closed at {book_r}R (BOOK_PROFIT).[/dim]\n")


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
        f"  Cadence :  [bold]{cadence_label(cadence)}[/bold] candles   |   "
        f"EMA({EMA_PERIOD}) {EMA_FILTER}   |   "
        f"NK filter: {'ON' if NK_FILTER else 'OFF'}  |  Mercury RX: {'ON' if MERCURY_RX_BLOCK else 'OFF'}  Saturn RX: {'ON' if SATURN_RX_BLOCK else 'OFF'}\n\n"
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
                    "TIMEOUT": "[yellow]TIMEOUT[/yellow]",
                    "BOOK_PROFIT": "[green]BOOK_PROFIT[/green]"}.get(tr["result"], tr["result"])
            t2.add_row(
                str(n),
                f"[cyan]{tr['side']}[/cyan]",
                res,
                f"{tr.get('bars', '?')} ({tr.get('bars', 0)*cadence:.1f}h)" if cadence >= 1 else f"{tr.get('bars', '?')} ({tr.get('bars', 0)*cadence*60:.0f}m)",
                f"[{pc}]{'+' if tr['pnl']>=0 else ''}{tr['pnl']:,.0f}[/{pc}]",
            )
        console.print(t2)

    # ── Capital efficiency ─────────────────────────────────────────────────────
    print_capital_efficiency(stats, label)
    console.print()


def run_risk_comparison(dfs: dict, funding: pd.Series,
                        cadence: int, risk_levels: list,
                        label_prefix: str = "", rr_ratio: float = RR_RATIO):
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
                risk_pct=r, rr_ratio=rr_ratio,
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
                 f"{cadence_label(cadence)} cadence  |  EMA({EMA_PERIOD}) {EMA_FILTER}[/bold]")

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
    parser.add_argument("--cadence", type=int, default=4,
                        help="Candle size in hours (1, 2, 4, 8). Default 4. Ignored if --cadence-min set.")
    parser.add_argument("--cadence-min", type=int, default=None, metavar="MIN",
                        help="Use 15m cadence (e.g. 15). Loads *_15m.csv from backtest.py --interval-min 15. Runs 2022-2025 + 2026-YTD with --long-short-periods.")
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
    parser.add_argument("--rr", type=float, default=2.0,
                        help="Risk:reward ratio for TP (e.g. 1.5 = TP at 1.5× SL distance). Default 2.0.")
    parser.add_argument("--book-profit", type=float, default=0, metavar="R",
                        help="If > 0, close trade when unrealized PnL reaches this many R (early book profit). 0 = off.")
    parser.add_argument("--compare-book-profit", action="store_true",
                        help="Run with and without early book-profit and print side-by-side (uses --book-profit value).")
    parser.add_argument("--compare-overlay", action="store_true",
                        help="Compare with vs without decisive overlay (best-call rules). Needs CSVs with action_no_overlay column (re-run backtest.py).")
    parser.add_argument("--long-short-periods", action="store_true",
                        help="Run backtest for 2022-2025 and 2026-YTD separately; report long vs short breakdown (highlight when shorts are mostly negative).")
    parser.add_argument("--year", type=int, default=None, metavar="YYYY",
                        help="Run real-world backtest for a single year (e.g. 2026). Uses logs/backtest_BTC_YYYY*.csv.")
    parser.add_argument("--end", type=str, default=None, metavar="YYYY-MM-DD",
                        help="End date for --year (inclusive). Default for 2026: 2026-03-17 (YTD through 17 Mar). Real-world: macro, SL/TP, funding, market fulfillment.")
    parser.add_argument("--compare-nakshatra", action="store_true",
                        help="Run backtest with Nakshatra filter OFF and ON; print detailed long/short comparison table for both.")
    parser.add_argument("--monthly-days", type=str, default=None, metavar="DAYS",
                        help="Comma-separated calendar day-of-month numbers to allow entries on "
                             "(e.g. '1,3,7,9'). Compares all-days vs day-filtered, with long/short breakdown. "
                             "Works with default 4-year run or --year.")
    parser.add_argument("--compare-intrahour", action="store_true",
                        help="Compare 15m, 30m, 1h cadences over 2022-2025 + 2026 YTD; use with --log to save to file.")
    parser.add_argument("--log", type=str, default=None, metavar="FILE",
                        help="Write all output to this file as well as stdout (e.g. logs/cadence_comparison.txt).")
    parser.add_argument("--day-size", type=str,
                        default="1=1.5,3=1.5,5=1.25,6=1.25,7=1.25,30=1.5,31=1.5",
                        help="Day-of-month risk multipliers, applied to risk_pct when opening a trade (astro signal unchanged). "
                             "Example: '1=1.5,3=1.5,5=1.25,6=1.25,7=1.25'.")
    parser.add_argument("--compare-day-size", action="store_true",
                        help="Run base sizing vs day-boost sizing side-by-side for the selected run mode (often with --compare-intrahour).")
    parser.add_argument("--seasonality", action="store_true",
                        help="Enable monthly/weekday seasonality sizing overlay (15th buy-low, 1/30/31 sell-high, Sundays buy, Mondays active).")
    parser.add_argument("--buy-day-size", type=str, default="15=1.25",
                        help="BUY-only day-of-month multipliers (e.g. '15=1.3'). Applied on top of --day-size.")
    parser.add_argument("--sell-day-size", type=str, default="1=1.25,30=1.25,31=1.25",
                        help="SELL-only day-of-month multipliers (e.g. '1=1.3,30=1.3,31=1.3'). Applied on top of --day-size.")
    parser.add_argument("--buy-weekday-size", type=str, default="sun=1.2",
                        help="BUY-only weekday multipliers (e.g. 'sun=1.2').")
    parser.add_argument("--weekday-size", type=str, default="mon=1.05",
                        help="All-sides weekday multipliers (e.g. 'mon=1.05').")
    parser.add_argument("--overlay-suite", action="store_true",
                        help="In --compare-intrahour: run base vs dayboost-only vs seasonality-only vs combined; print side-by-side + deltas.")
    args = parser.parse_args()

    # Optional: tee output to log file (global console so all print_* use it)
    _log_file_handle = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file_handle = open(log_path, "w", encoding="utf-8")
        def _close_log():
            if _log_file_handle is not None:
                _log_file_handle.close()
        atexit.register(_close_log)
        class _Tee:
            def __init__(self, stdout, f):
                self._out, self._f = stdout, f
            def write(self, s): self._out.write(s); self._f.write(s)
            def flush(self): self._out.flush(); self._f.flush()
            def isatty(self): return getattr(self._out, "isatty", lambda: False)()
        _tee = _Tee(sys.stdout, _log_file_handle)
        globals()["console"] = Console(file=_tee, width=180, force_terminal=True)

    # Effective cadence and CSV suffix (15m/30m use _15m.csv / _30m.csv from backtest.py --interval-min 15/30)
    use_cadence_min = args.cadence_min is not None
    cadence = (args.cadence_min / 60.0) if use_cadence_min else args.cadence
    csv_suffix = f"_{args.cadence_min}m" if use_cadence_min else ""
    use_15m = use_cadence_min  # for backward compat in branches that check use_15m
    day_mult_map = parse_int_float_map(args.day_size)
    buy_day_mult_map = parse_int_float_map(args.buy_day_size) if args.seasonality else {}
    sell_day_mult_map = parse_int_float_map(args.sell_day_size) if args.seasonality else {}
    weekday_mult_map = parse_weekday_map(args.weekday_size) if args.seasonality else {}
    buy_weekday_mult_map = parse_weekday_map(args.buy_weekday_size) if args.seasonality else {}
    # Minute cadence backtest for 2022-2025 + 2026 YTD: default to long-short-periods report
    if use_cadence_min and not args.long_short_periods and args.year is None and not args.file and not args.compare_nakshatra and not args.compare_intrahour:
        args.long_short_periods = True

    # ── Compare 15m, 30m, 1h and optionally log to file ────────────────────────
    if args.compare_intrahour:
        INTRACADENCE_CONFIGS = [(0.25, "_15m"), (0.5, "_30m"), (1, "")]
        funding = load_funding_rates()
        book_r = max(0.0, float(args.book_profit))
        kw_base = dict(use_intrabar=True, use_fees=True, use_slippage=True, use_funding=True,
                       rr_ratio=args.rr, book_profit_at_r=book_r)
        kw_dayboost = dict(**kw_base, day_risk_mult=day_mult_map)
        kw_seasonality = dict(
            **kw_base,
            weekday_risk_mult=weekday_mult_map,
            buy_day_risk_mult=buy_day_mult_map,
            sell_day_risk_mult=sell_day_mult_map,
            buy_weekday_risk_mult=buy_weekday_mult_map,
        )
        kw_combined = dict(
            **kw_base,
            day_risk_mult=day_mult_map,
            weekday_risk_mult=weekday_mult_map,
            buy_day_risk_mult=buy_day_mult_map,
            sell_day_risk_mult=sell_day_mult_map,
            buy_weekday_risk_mult=buy_weekday_mult_map,
        )
        kw_boost = dict(
            **kw_base,
            day_risk_mult=day_mult_map,
            weekday_risk_mult=weekday_mult_map,
            buy_day_risk_mult=buy_day_mult_map,
            sell_day_risk_mult=sell_day_mult_map,
            buy_weekday_risk_mult=buy_weekday_mult_map,
        ) if (args.compare_day_size and (day_mult_map or args.seasonality)) else dict(**kw_base)
        agg_2022_2025 = {}
        stats_2026 = {}
        agg_2022_2025_boost = {}
        stats_2026_boost = {}
        overlay_suite = None
        if args.overlay_suite:
            overlay_suite = {
                "base": {"kw": kw_base, "2022_2025": {}, "2026": {}},
                "dayboost": {"kw": (kw_dayboost if day_mult_map else kw_base), "2022_2025": {}, "2026": {}},
                "seasonality": {"kw": (kw_seasonality if args.seasonality else kw_base), "2022_2025": {}, "2026": {}},
                "combined": {"kw": (kw_combined if (day_mult_map or args.seasonality) else kw_base), "2022_2025": {}, "2026": {}},
            }
        log_dir = Path("logs")
        years = [2022, 2023, 2024, 2025]
        for cadence_val, suffix in INTRACADENCE_CONFIGS:
            raw_dfs = {}
            for y in years:
                path = Path(f"logs/backtest_BTC_{y}-01-01_{y}-12-31{suffix}.csv")
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                raw_dfs[y] = df
            if not raw_dfs:
                console.print(f"[yellow]No 2022-2025 data for {cadence_label(cadence_val)} (missing {suffix or '1h'} CSVs).[/yellow]")
                continue
            all_y = {}
            all_y_boost = {}
            for y, df in raw_dfs.items():
                all_y[y] = simulate(df, funding, cadence=cadence_val, **kw_base)
                if args.compare_day_size and (day_mult_map or args.seasonality):
                    all_y_boost[y] = simulate(df, funding, cadence=cadence_val, **kw_boost)
            agg_2022_2025[cadence_val] = _aggregate_yearly_results(all_y)
            if args.compare_day_size and day_mult_map:
                agg_2022_2025_boost[cadence_val] = _aggregate_yearly_results(all_y_boost)
            ytd_paths = sorted(log_dir.glob(f"backtest_BTC_2026*{suffix}.csv"))
            if ytd_paths:
                df_2026 = pd.read_csv(ytd_paths[-1])
                df_2026["timestamp"] = pd.to_datetime(df_2026["timestamp"], utc=True)
                stats_2026[cadence_val] = simulate(df_2026, funding, cadence=cadence_val, **kw_base)
                if args.compare_day_size and (day_mult_map or args.seasonality):
                    stats_2026_boost[cadence_val] = simulate(df_2026, funding, cadence=cadence_val, **kw_boost)
            else:
                stats_2026[cadence_val] = None
                stats_2026_boost[cadence_val] = None

            # Overlay suite scenarios
            if overlay_suite is not None:
                for name, pack in overlay_suite.items():
                    kw = pack["kw"]
                    all_s = {y: simulate(df, funding, cadence=cadence_val, **kw) for y, df in raw_dfs.items()}
                    pack["2022_2025"][cadence_val] = _aggregate_yearly_results(all_s)
                    if ytd_paths:
                        pack["2026"][cadence_val] = simulate(df_2026, funding, cadence=cadence_val, **kw)
                    else:
                        pack["2026"][cadence_val] = None
        console.print()
        console.rule("[bold]Cadence comparison: 15m vs 30m vs 1h[/bold]")
        print_intrahour_comparison(agg_2022_2025, stats_2026, [(c, cadence_label(c)) for c, _ in INTRACADENCE_CONFIGS])

        if overlay_suite is not None:
            console.print()
            console.rule("[bold]Overlay suite (base vs dayboost vs seasonality vs combined)[/bold]")
            if args.seasonality:
                console.print(f"[dim]Seasonality ON  |  day_size={day_mult_map}  buy_day={buy_day_mult_map}  sell_day={sell_day_mult_map}  weekday={weekday_mult_map}  buy_weekday={buy_weekday_mult_map}[/dim]")
            else:
                console.print(f"[dim]Seasonality OFF |  day_size={day_mult_map}[/dim]")
            results_by_scenario = {k: {"2022_2025": v["2022_2025"], "2026": v["2026"]} for k, v in overlay_suite.items()}
            print_overlay_suite_table(results_by_scenario, [(c, cadence_label(c)) for c, _ in INTRACADENCE_CONFIGS])
        if args.compare_day_size and (day_mult_map or args.seasonality):
            console.print()
            console.rule("[bold]Cadence comparison (day-of-month boosted sizing)[/bold]")
            if args.seasonality:
                console.print(f"[dim]Seasonality ON  |  day_size={day_mult_map}  buy_day={buy_day_mult_map}  sell_day={sell_day_mult_map}  weekday={weekday_mult_map}  buy_weekday={buy_weekday_mult_map}[/dim]")
            else:
                console.print(f"[dim]Day size map: {day_mult_map}[/dim]")
            print_intrahour_comparison(agg_2022_2025_boost, stats_2026_boost, [(c, cadence_label(c)) for c, _ in INTRACADENCE_CONFIGS])
            console.print()
            t = Table("Cadence", "Period", "Base P&L %", "Boosted P&L %", "Delta", box=box.MINIMAL_HEAVY_HEAD,
                      header_style="bold white on dark_blue", title="[bold]Sizing impact (boost - base)[/bold]", show_lines=True)
            for c, _ in INTRACADENCE_CONFIGS:
                b = agg_2022_2025.get(c)
                z = agg_2022_2025_boost.get(c)
                if b and z:
                    delta = z.get("pnl_pct", 0) - b.get("pnl_pct", 0)
                    dc = "green" if delta >= 0 else "red"
                    t.add_row(cadence_label(c), "2022–2025", f"{b.get('pnl_pct',0):+.1f}%", f"{z.get('pnl_pct',0):+.1f}%",
                              f"[{dc}]{delta:+.1f}%[/{dc}]")
                b6 = stats_2026.get(c)
                z6 = stats_2026_boost.get(c)
                if b6 and z6:
                    delta = z6.get("pnl_pct", 0) - b6.get("pnl_pct", 0)
                    dc = "green" if delta >= 0 else "red"
                    t.add_row(cadence_label(c), "2026 YTD", f"{b6.get('pnl_pct',0):+.1f}%", f"{z6.get('pnl_pct',0):+.1f}%",
                              f"[{dc}]{delta:+.1f}%[/{dc}]")
            console.print(t)
        for cadence_val, suffix in INTRACADENCE_CONFIGS:
            a = agg_2022_2025.get(cadence_val)
            if a:
                console.print()
                print_long_short_report(f"2022–2025 (4-year) — {cadence_label(cadence_val)}", a, cadence_val)
            s6 = stats_2026.get(cadence_val)
            if s6:
                print_long_short_report(f"2026 YTD — {cadence_label(cadence_val)}", s6, cadence_val)
            if args.compare_day_size and (day_mult_map or args.seasonality):
                ab = agg_2022_2025_boost.get(cadence_val)
                if ab:
                    console.print()
                    print_long_short_report(f"2022–2025 (4-year) — {cadence_label(cadence_val)} (boosted sizing)", ab, cadence_val)
                s6b = stats_2026_boost.get(cadence_val)
                if s6b:
                    print_long_short_report(f"2026 YTD — {cadence_label(cadence_val)} (boosted sizing)", s6b, cadence_val)
        if args.log:
            console.print(f"\n[dim]Full output also written to {args.log}[/dim]")
        return

    # ── Real-world single-year backtest (e.g. 2026 YTD through 17 Mar) ─────────
    if args.year is not None:
        log_dir = Path("logs")
        paths = sorted(log_dir.glob(f"backtest_BTC_{args.year}*.csv"))
        if use_cadence_min:
            paths = [p for p in paths if p.stem.endswith(f"_{args.cadence_min}m")]
        else:
            paths = [p for p in paths if not any(p.stem.endswith(f"_{m}m") for m in (15, 30))]
        if not paths:
            console.print(f"[red]No backtest CSV found for {args.year}. Add logs/backtest_BTC_{args.year}-01-01_*.csv (e.g. from scripts/backtest.py).[/red]")
            sys.exit(1)
        fpath = paths[-1]
        df = pd.read_csv(fpath)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # YTD cap: default 2026 to 17 March 2026 only
        end_date_str = args.end
        if end_date_str is None and args.year == 2026:
            end_date_str = "2026-03-17"
        if end_date_str:
            end_date_only = pd.Timestamp(end_date_str).date()
            df = df[df["timestamp"].dt.date <= end_date_only]
            if df.empty:
                console.print(f"[red]No data on or before {end_date_str}. Check CSV date range.[/red]")
                sys.exit(1)
        start_d = str(df["timestamp"].min().date())
        end_d = str(df["timestamp"].max().date())
        label = args.label or f"{args.year} YTD ({start_d} → {end_d})"
        book_r = max(0.0, float(args.book_profit))
        # Funding for this period
        fcsv = Path(f"logs/funding_BTCUSDT_{start_d}_{end_d}.csv")
        if not fcsv.exists():
            fcsv = Path(f"logs/funding_BTCUSDT_{args.year}-01-01_{args.year}-12-31.csv")
        if fcsv.exists():
            fund_df = pd.read_csv(fcsv)
            fund_df["timestamp"] = pd.to_datetime(fund_df["timestamp"], utc=True, format="mixed")
            funding = fund_df.set_index("timestamp")["funding_rate"]
        else:
            funding = pd.Series(dtype=float)
            console.print(f"[dim]No funding CSV for {args.year} — funding impact skipped.[/dim]")
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Real-world {args.year} YTD backtest[/bold cyan]\n"
            f"[dim]Macro (EMA)  |  Intrabar SL/TP  |  Funding cost  |  Market fulfillment (slippage 0.05%, fee 0.05%/side)  |  Short-block  |  {cadence_label(cadence)}[/dim]\n"
            f"[dim]{start_d} → {end_d}  ({fpath.name})[/dim]",
            border_style="cyan",
        ))
        if args.compare_nakshatra:
            stats_off = simulate(df, funding, cadence=cadence,
                                 use_intrabar=True, use_fees=True,
                                 use_slippage=True, use_funding=True,
                                 rr_ratio=args.rr, book_profit_at_r=book_r,
                                 nk_filter=False)
            stats_on = simulate(df, funding, cadence=cadence,
                                use_intrabar=True, use_fees=True,
                                use_slippage=True, use_funding=True,
                                rr_ratio=args.rr, book_profit_at_r=book_r,
                                nk_filter=True)
            console.print()
            print_nakshatra_comparison(stats_off, stats_on, label, cadence)
            return
        kw = dict(use_intrabar=True, use_fees=True, use_slippage=True, use_funding=True,
                  rr_ratio=args.rr, book_profit_at_r=book_r)
        # ── Monthly-days comparison for --year mode ────────────────────────
        if args.monthly_days:
            day_set = {int(x.strip()) for x in args.monthly_days.split(",") if x.strip().isdigit()}
            console.print(f"\n[bold cyan]Monthly-days filter: entries only on day(s) {sorted(day_set)} of each month[/bold cyan]\n")
            stats_all  = simulate(df, funding, cadence=cadence, **kw)
            stats_days = simulate(df, funding, cadence=cadence, **kw, allowed_entry_days=day_set)
            print_monthly_days_report(stats_all, stats_days, day_set, label, cadence)
            return
        stats = simulate(df, funding, cadence=cadence, **kw)
        console.print()
        print_single_period(stats, label, cadence, start_d, end_d)
        print_long_short_report(label, stats, cadence)
        print_short_astro_report(label, stats.get("trade_list", []))
        return

    # ── Single-file mode (partial year / custom period) ────────────────────────
    if args.file:
        if not use_15m:
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
            stem = fpath.stem  # e.g. backtest_BTC_2026-01-01_2026-02-22 or ..._2026-02-22_15m
            parts = stem.split("_")
            # filename pattern: backtest_BTC_START_END or backtest_BTC_START_END_15m
            try:
                if len(parts) >= 4 and parts[-1] == "15m":
                    start_d, end_d = parts[-3], parts[-2]
                else:
                    start_d, end_d = parts[-2], parts[-1]
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
        book_r = max(0.0, float(args.book_profit))
        book_line = f"  |  Book profit: {book_r}R (early exit)" if book_r else ""
        console.print(Panel.fit(
            f"[bold cyan]Realistic Simulation — {label}[/bold cyan]\n"
            "[dim]Intrabar SL/TP  |  Slippage 0.05%  |  Fee 0.05%/side  |  "
            "Real funding rates[/dim]\n"
            f"[dim]EMA({EMA_PERIOD}) {EMA_FILTER}  |  RR {args.rr}:1  |  "
            f"Nakshatra filter: {'ON' if NK_FILTER else 'OFF'}  |  Mercury RX block: {'ON' if MERCURY_RX_BLOCK else 'OFF'}  Saturn RX block: {'ON' if SATURN_RX_BLOCK else 'OFF'}  |  Cadence: {cadence_label(cadence)}{book_line}[/dim]",
            border_style="cyan",
        ))

        risk_levels = [float(x) / 100 for x in args.risk_levels.split(",")]

        if args.compare_risk:
            run_risk_comparison(
                {label: df}, funding,
                cadence=cadence,
                risk_levels=risk_levels,
                label_prefix=label,
                rr_ratio=args.rr,
            )
            return

        if args.compare_book_profit and book_r > 0:
            stats_no = simulate(df, funding, cadence=cadence,
                                use_intrabar=True, use_fees=True,
                                use_slippage=True, use_funding=True,
                                rr_ratio=args.rr, book_profit_at_r=0)
            stats_yes = simulate(df, funding, cadence=cadence,
                                 use_intrabar=True, use_fees=True,
                                 use_slippage=True, use_funding=True,
                                 rr_ratio=args.rr, book_profit_at_r=book_r)
            print_book_profit_comparison(stats_no, stats_yes, label, cadence, book_r)
            return

        stats = simulate(df, funding, cadence=cadence,
                         use_intrabar=True, use_fees=True,
                         use_slippage=True, use_funding=True,
                         rr_ratio=args.rr, book_profit_at_r=book_r)
        print_single_period(stats, label, cadence, start_date, end_date)

        if args.compare_risk:
            run_risk_comparison(
                {label: df}, funding,
                cadence=cadence,
                risk_levels=risk_levels,
                label_prefix=label,
                rr_ratio=args.rr,
            )
        return

    cadences  = [1, 2, 4, 8] if args.compare else [cadence]
    years     = [2022, 2023, 2024, 2025]
    funding   = load_funding_rates()
    book_r    = max(0.0, float(args.book_profit))

    if funding.empty:
        console.print("[yellow]Warning: funding rate CSV not found — funding impact skipped.[/yellow]")

    console.print()
    book_line_4y = f"  |  Book profit: {book_r}R (early exit)" if book_r else ""
    console.print(Panel.fit(
        "[bold cyan]Realistic 4-Year Backtest[/bold cyan]\n"
        "[dim]Proper N-hour OHLC candles  |  Intrabar SL/TP  |  "
        "Slippage 0.05%  |  Fee 0.05%/side  |  Real funding rates[/dim]\n"
        f"[dim]EMA({EMA_PERIOD}) {EMA_FILTER}  |  "
        f"Nakshatra filter: {'ON' if NK_FILTER else 'OFF'}  |  Mercury RX: {'ON' if MERCURY_RX_BLOCK else 'OFF'}  Saturn RX: {'ON' if SATURN_RX_BLOCK else 'OFF'}  |  "
        f"Cadence: {', '.join(cadence_label(c) for c in cadences)}{book_line_4y}[/dim]",
        border_style="cyan",
    ))

    # Load all CSVs once (use _15m suffix when --cadence-min 15)
    raw_dfs = {}
    for y in years:
        path = Path(f"logs/backtest_BTC_{y}-01-01_{y}-12-31{csv_suffix}.csv")
        if not path.exists():
            console.print(f"[red]Missing {path} — skipping {y}[/red]")
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        raw_dfs[y] = df

    # ── Nakshatra OFF vs ON — detailed long/short comparison ───────────────────
    if args.compare_nakshatra and raw_dfs:
        funding = load_funding_rates()
        book_r = max(0.0, float(args.book_profit))
        all_off = {}
        all_on = {}
        for y, df in raw_dfs.items():
            console.print(f"[dim]Comparing Nakshatra for {y} @ {cadence_label(cadence)}...[/dim]", end="\r")
            all_off[y] = simulate(df, funding, cadence=cadence,
                                  use_intrabar=True, use_fees=True,
                                  use_slippage=True, use_funding=True,
                                  rr_ratio=args.rr, book_profit_at_r=book_r,
                                  nk_filter=False)
            all_on[y] = simulate(df, funding, cadence=cadence,
                                 use_intrabar=True, use_fees=True,
                                 use_slippage=True, use_funding=True,
                                 rr_ratio=args.rr, book_profit_at_r=book_r,
                                 nk_filter=True)
        console.print(" " * 50, end="\r")
        agg_off = _aggregate_yearly_results(all_off)
        agg_on = _aggregate_yearly_results(all_on)
        console.rule("[bold]Nakshatra filter OFF vs ON — 2022–2025 (4-year)[/bold]")
        print_nakshatra_comparison(agg_off, agg_on, "2022–2025 (4-year)", cadence)
        return

    # ── Long vs short by period: 2022-2025 vs 2026-YTD ────────────────────────
    if args.long_short_periods:
        funding = load_funding_rates()
        book_r = max(0.0, float(args.book_profit))

        # 2022-2025
        if raw_dfs:
            console.print("[dim]Simulating 2022-2025...[/dim]")
            all_2022_2025 = {}
            for y, df in raw_dfs.items():
                all_2022_2025[y] = simulate(df, funding, cadence=cadence,
                                            use_intrabar=True, use_fees=True,
                                            use_slippage=True, use_funding=True,
                                            rr_ratio=args.rr, book_profit_at_r=book_r)
            agg_2022_2025 = _aggregate_yearly_results(all_2022_2025)
            print_long_short_report("2022–2025 (4-year)", agg_2022_2025, cadence)
            print_short_astro_report("2022–2025 (4-year)", agg_2022_2025.get("trade_list", []))
        else:
            console.print(f"[yellow]No 2022-2025 data (missing logs/backtest_BTC_YYYY-01-01_YYYY-12-31{csv_suffix}.csv).[/yellow]")

        # 2026-YTD: look for any backtest CSV that starts in 2026 (with same cadence suffix)
        log_dir = Path("logs")
        ytd_2026_paths = sorted(log_dir.glob(f"backtest_BTC_2026*{csv_suffix}.csv"))
        if ytd_2026_paths:
            # Use the file that covers 2026-01-01 to latest (often one file)
            path_2026 = ytd_2026_paths[-1]
            df_2026 = pd.read_csv(path_2026)
            df_2026["timestamp"] = pd.to_datetime(df_2026["timestamp"], utc=True)
            start_d = str(df_2026["timestamp"].min().date())
            end_d = str(df_2026["timestamp"].max().date())
            console.print(f"[dim]Simulating 2026-YTD ({path_2026.name})...[/dim]")
            funding_2026 = load_funding_rates()
            stats_2026 = simulate(df_2026, funding_2026, cadence=cadence,
                                 use_intrabar=True, use_fees=True,
                                 use_slippage=True, use_funding=True,
                                 rr_ratio=args.rr, book_profit_at_r=book_r)
            print_long_short_report(f"2026-YTD ({start_d} → {end_d})", stats_2026, cadence)
            print_short_astro_report(f"2026-YTD ({start_d} → {end_d})", stats_2026.get("trade_list", []))
        else:
            console.print(Panel(
                "[yellow]No 2026-YTD data.[/yellow]\n"
                "Add logs/backtest_BTC_2026-01-01_2026-MM-DD.csv (or similar) and re-run with --long-short-periods.",
                title="[bold]2026-YTD[/bold]",
                border_style="yellow",
            ))
        return

    # ── Compare with vs without decisive overlay ───────────────────────────────
    if args.compare_overlay and raw_dfs:
        if not all("action_no_overlay" in df.columns for df in raw_dfs.values()):
            console.print(
                "[yellow]All CSVs need 'action_no_overlay' column. "
                "Re-run backtest for 2022–2025 to regenerate CSVs, then run again with --compare-overlay.[/yellow]"
            )
            return
        cadence = args.cadence
        all_with = {}
        all_without = {}
        for y, df in raw_dfs.items():
            console.print(f"[dim]Comparing overlay for {y} @ {cadence_label(cadence)}...[/dim]", end="\r")
            all_with[y] = simulate(df, funding, cadence=cadence,
                                  use_intrabar=True, use_fees=True,
                                  use_slippage=True, use_funding=True,
                                  rr_ratio=args.rr, book_profit_at_r=book_r)
            df_no = df.copy()
            df_no["action"] = df_no["action_no_overlay"]
            all_without[y] = simulate(df_no, funding, cadence=cadence,
                                     use_intrabar=True, use_fees=True,
                                     use_slippage=True, use_funding=True,
                                     rr_ratio=args.rr, book_profit_at_r=book_r)
        console.print(" " * 50, end="\r")
        agg_with = _aggregate_yearly_results(all_with)
        agg_without = _aggregate_yearly_results(all_without)
        console.rule("[bold]With vs without decisive overlay (best-call rules)[/bold]")
        print_overlay_comparison(agg_with, agg_without, cadence)
        return

    # Run simulations (with optional compare-book-profit: run two sets)
    if args.compare_book_profit and book_r > 0:
        all_results_no = {}
        all_results_yes = {}
        for c in cadences:
            all_results_no[c] = {}
            all_results_yes[c] = {}
            for y, df in raw_dfs.items():
                console.print(f"[dim]Simulating {y} @ {c}h (no book / {book_r}R book)...[/dim]", end="\r")
                all_results_no[c][y] = simulate(df, funding, cadence=c,
                                                use_intrabar=True, use_fees=True,
                                                use_slippage=True, use_funding=True,
                                                rr_ratio=args.rr, book_profit_at_r=0)
                all_results_yes[c][y] = simulate(df, funding, cadence=c,
                                                 use_intrabar=True, use_fees=True,
                                                 use_slippage=True, use_funding=True,
                                                 rr_ratio=args.rr, book_profit_at_r=book_r)
        console.print(" " * 60, end="\r")
        for c in cadences:
            agg_no = _aggregate_yearly_results(all_results_no[c])
            agg_yes = _aggregate_yearly_results(all_results_yes[c])
            console.rule(f"[bold]Early book profit @ {book_r}R vs no early book — {cadence_label(c)} cadence[/bold]")
            print_book_profit_comparison(agg_no, agg_yes, f"4-Year Historical ({c}h)", c, book_r)
            # Per-year table
            t = Table(
                "Year", "No book: P&L%", f"Book @ {book_r}R: P&L%", "No book: WR", f"Book @ {book_r}R: WR",
                box=box.SIMPLE, header_style="bold",
            )
            for y in sorted(all_results_no[c].keys()):
                sn, sy = all_results_no[c][y], all_results_yes[c][y]
                pn, py = sn["pnl_pct"], sy["pnl_pct"]
                wn, wy = sn["wr"], sy["wr"]
                pc = "green" if py >= pn else "red"
                wc = "green" if wy >= wn else "red"
                t.add_row(str(y), f"{pn:+.1f}%", f"[{pc}]{py:+.1f}%[/{pc}]", f"{wn:.0f}%", f"[{wc}]{wy:.0f}%[/{wc}]")
            console.print(t)
            console.print()
        return

    # ── Monthly-days filter comparison (4-year) ───────────────────────────────
    if args.monthly_days:
        day_set = {int(x.strip()) for x in args.monthly_days.split(",") if x.strip().isdigit()}
        cadence = cadences[-1]
        console.print(f"\n[bold cyan]Monthly-days filter: entries only on day(s) {sorted(day_set)} of each month — 2022–2025 @ {cadence_label(cadence)}[/bold cyan]\n")
        all_all  = {}
        all_days = {}
        sim_kw_days = dict(use_intrabar=True, use_fees=True, use_slippage=True, use_funding=True,
                           rr_ratio=args.rr, book_profit_at_r=book_r)
        for y, df in raw_dfs.items():
            console.print(f"[dim]Simulating {y} @ {cadence_label(cadence)} (all days + day-filter)...[/dim]", end="\r")
            all_all[y]  = simulate(df, funding, cadence=cadence, **sim_kw_days)
            all_days[y] = simulate(df, funding, cadence=cadence, **sim_kw_days, allowed_entry_days=day_set)
        console.print(" " * 60, end="\r")
        agg_all  = _aggregate_yearly_results(all_all)
        agg_days = _aggregate_yearly_results(all_days)
        days_label = ",".join(str(d) for d in sorted(day_set))
        print_monthly_days_report(agg_all, agg_days, day_set, f"2022–2025 (4-year, {cadence_label(cadence)})", cadence)
        # Per-year table
        t = Table(
            "Year", "All days P&L%", f"Days {days_label} P&L%", "All WR", f"Days {days_label} WR",
            box=box.SIMPLE, header_style="bold",
            title=f"[bold]Per-year: All days vs Days {days_label}[/bold]",
        )
        for y in sorted(all_all.keys()):
            sn, sy = all_all[y], all_days[y]
            pn, py = sn["pnl_pct"], sy["pnl_pct"]
            wn, wy = sn["wr"], sy["wr"]
            pc = "green" if py >= pn else "red"
            wc = "green" if wy >= wn else "red"
            t.add_row(str(y), f"{pn:+.1f}%", f"[{pc}]{py:+.1f}%[/{pc}]",
                      f"{wn:.0f}%", f"[{wc}]{wy:.0f}%[/{wc}]")
        console.print(t)
        console.print()
        return

    # Run simulations (single run, possibly with book_profit)
    sim_kw = dict(use_intrabar=True, use_fees=True, use_slippage=True, use_funding=True,
                  rr_ratio=args.rr, book_profit_at_r=book_r)
    all_results = {}
    for c in cadences:
        all_results[c] = {}
        for y, df in raw_dfs.items():
            console.print(f"[dim]Simulating {y} @ {c}h...[/dim]", end="\r")
            all_results[c][y] = simulate(df, funding, cadence=c, **sim_kw)
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
