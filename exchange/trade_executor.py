"""
Trade Executor — places and manages orders on the exchange.

Supports:
  - Paper trading mode (logs orders without executing)
  - Live trading via ccxt
  - Market entry + Stop-loss + Take-profit bracket

Backtest-aligned safeguards (added 2026-02-23):
  1. STRONG-only dispatch — WEAK_BUY / WEAK_SELL are skipped, matching the
     backtest which only trades STRONG signals (final_backtest.py line 238).
  2. 48h trade timeout — open positions older than MAX_OPEN_BARS × interval
     are force-closed each cycle, matching the backtest MAX_OPEN_BARS logic.
  3. Drawdown halt — trading is suspended when equity drops MAX_DRAWDOWN_HALT_PCT
     from its peak, matching the config parameter that was previously undefined.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

import config
from exchange.market_data import get_exchange

# ── Postgres adapter (used when DATABASE_URL is set) ──────────────────────────

_USE_PG     = bool(os.getenv("DATABASE_URL"))
_USER_ID    = os.getenv("BOT_USER_ID", "default")   # unique per Railway service

if _USE_PG:
    from db.postgres import (
        init_schema as _pg_init,
        pg_log_signal,
        pg_log_close,
        pg_load_positions,
        pg_save_positions,
        pg_load_equity,
        pg_save_equity,
    )
    _pg_init()
    logger.info(f"PostgreSQL storage enabled (user_id='{_USER_ID}').")
else:
    logger.info("Using SQLite/JSON file storage (DATABASE_URL not set).")


# ── Persistent state file paths ───────────────────────────────────────────────

_LOGS_DIR            = Path(config.DB_PATH).parent
OPEN_POSITIONS_FILE  = _LOGS_DIR / "open_positions.json"
EQUITY_STATE_FILE    = _LOGS_DIR / "equity_state.json"


# ── SQLite trade log ───────────────────────────────────────────────────────────

def _init_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            asset           TEXT,
            symbol          TEXT,
            action          TEXT,
            entry_price     REAL,
            stop_loss       REAL,
            target          REAL,
            position_usdt   REAL,
            western_score   REAL,
            vedic_score     REAL,
            western_slope   REAL,
            vedic_slope     REAL,
            numerology_mult REAL,
            nakshatra       TEXT,
            paper           INTEGER,
            notes           TEXT,
            close_price     REAL,
            pnl             REAL,
            result          TEXT
        )
    """)
    # Add columns to existing DBs that pre-date the pnl tracking feature
    for col, coltype in [("close_price", "REAL"), ("pnl", "REAL"), ("result", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


_init_db()


def log_signal_to_db(signal: dict, paper: bool = True, notes: str = ""):
    if _USE_PG:
        pg_log_signal(signal, paper, notes, user_id=_USER_ID)
        return
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("""
        INSERT INTO signals (
            timestamp, asset, symbol, action,
            entry_price, stop_loss, target, position_usdt,
            western_score, vedic_score, western_slope, vedic_slope,
            numerology_mult, nakshatra, paper, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal.get("timestamp"),
        signal.get("asset"),
        signal.get("symbol"),
        signal.get("final_action"),
        signal.get("current_price"),
        signal.get("stop_loss"),
        signal.get("target"),
        signal.get("position_size_usdt"),
        signal.get("western_score"),
        signal.get("vedic_score"),
        signal.get("western_slope"),
        signal.get("vedic_slope"),
        signal.get("numerology", {}).get("multiplier"),
        signal.get("nakshatra"),
        1 if paper else 0,
        notes,
    ))
    conn.commit()
    conn.close()


# ── Weekly trade counter ───────────────────────────────────────────────────────

def trades_this_week() -> int:
    """Count live (non-paper) trades placed in the current ISO week."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.execute("""
        SELECT COUNT(*) FROM signals
        WHERE paper = 0
          AND action NOT IN ('HOLD', 'NO_TRADE', 'COLLECTING_DATA')
          AND timestamp >= date('now', 'weekday 0', '-7 days')
    """)
    count = cursor.fetchone()[0]
    conn.close()
    return count


# ── Leverage helpers ───────────────────────────────────────────────────────────

def compute_leverage_info(notional_usdt: float, entry_price: float, sl_price: float) -> dict:
    """
    Calculate all leverage-related values for a trade.

    Returns a dict with:
      margin_required   — collateral locked on exchange = notional / leverage
      sl_pct            — stop-loss distance as % of entry
      liq_pct           — liquidation distance as % of entry (= 1 / leverage)
      liq_price_long    — estimated liquidation price if LONG
      liq_price_short   — estimated liquidation price if SHORT
      safe              — True if stop fires before liquidation
    """
    lev = config.LEVERAGE
    margin = notional_usdt / lev
    sl_pct = abs(entry_price - sl_price) / entry_price
    liq_pct = 1.0 / lev                               # simplified: ignores maintenance margin
    liq_long  = entry_price * (1 - liq_pct)
    liq_short = entry_price * (1 + liq_pct)
    safe = sl_pct < liq_pct                           # stop fires before liq if sl% < liq%
    return {
        "leverage":        lev,
        "notional_usdt":   round(notional_usdt, 2),
        "margin_required": round(margin, 2),
        "sl_pct":          round(sl_pct * 100, 2),
        "liq_pct":         round(liq_pct * 100, 2),
        "liq_price_long":  round(liq_long, 2),
        "liq_price_short": round(liq_short, 2),
        "safe":            safe,
    }


def set_leverage_on_exchange(symbol: str):
    """Set leverage on the exchange before placing an order."""
    if config.LEVERAGE == 1:
        return
    try:
        ex = get_exchange()
        ex.set_leverage(config.LEVERAGE, symbol)
        logger.info(f"Leverage set to {config.LEVERAGE}x for {symbol}")
    except Exception as e:
        logger.warning(f"Could not set leverage automatically: {e} — set {config.LEVERAGE}x manually on exchange")


# ── Gap 2: Open position tracker (48h timeout) ────────────────────────────────

def _load_open_positions() -> list:
    if _USE_PG:
        return pg_load_positions(user_id=_USER_ID)
    if OPEN_POSITIONS_FILE.exists():
        with open(OPEN_POSITIONS_FILE) as f:
            return json.load(f)
    return []


def _save_open_positions(positions: list):
    if _USE_PG:
        pg_save_positions(positions, user_id=_USER_ID)
        return
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OPEN_POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def _record_open_position(signal: dict, order_ids: dict = None):
    """
    Persist a newly opened trade so the timeout checker can find it.
    order_ids should be {"sl": "<id>", "tp": "<id>"} for live trades.
    """
    positions = _load_open_positions()
    positions.append({
        "symbol":      signal.get("symbol"),
        "action":      signal.get("final_action"),
        "entry_price": signal.get("current_price"),
        "stop_loss":   signal.get("stop_loss"),
        "target":      signal.get("target"),
        "notional":    signal.get("position_size_usdt"),
        "opened_at":   datetime.now(timezone.utc).isoformat(),
        "sl_order_id": (order_ids or {}).get("sl"),
        "tp_order_id": (order_ids or {}).get("tp"),
        "paper":       config.PAPER_TRADING,
    })
    _save_open_positions(positions)


def _force_close_live_position(pos: dict):
    """Cancel SL/TP orders and market-close a live position that timed out."""
    symbol = pos.get("symbol", "")
    action = pos.get("action", "")
    notional = pos.get("notional", 0.0)
    entry_price = pos.get("entry_price", 0.0)
    qty = notional / entry_price if entry_price > 0 else 0.0

    try:
        ex = get_exchange()

        for order_key in ("sl_order_id", "tp_order_id"):
            oid = pos.get(order_key)
            if oid:
                try:
                    ex.cancel_order(oid, symbol)
                    logger.info(f"Cancelled order {oid} for timed-out {symbol} position")
                except Exception as e:
                    logger.warning(f"Could not cancel order {oid}: {e}")

        if qty > 0:
            close_side = "sell" if "BUY" in action else "buy"
            close_order = ex.create_market_order(
                symbol, close_side, qty,
                params={"reduceOnly": True}
            )
            logger.info(f"TIMEOUT close: {symbol} {action} — order {close_order.get('id')}")
    except Exception as e:
        logger.error(f"Failed to force-close live position {symbol}: {e}")


def check_and_close_stale_positions():
    """
    Force-close any tracked position open longer than MAX_OPEN_BARS × interval.

    Called at the start of every bot_cycle() to mirror the backtest's
    MAX_OPEN_BARS force-close that was found to be optimal at 12 bars (48h).

    Paper mode  → logs a TIMEOUT signal to the DB and removes from tracker.
    Live mode   → cancels exchange SL/TP orders and sends a reduceOnly market close.
    """
    positions = _load_open_positions()
    if not positions:
        return

    max_hold_minutes = config.MAX_OPEN_BARS * config.CHECK_INTERVAL_MINUTES
    now = datetime.now(timezone.utc)
    still_open = []

    for pos in positions:
        try:
            opened_at = datetime.fromisoformat(pos["opened_at"])
        except (KeyError, ValueError):
            still_open.append(pos)
            continue

        age_minutes = (now - opened_at).total_seconds() / 60

        if age_minutes >= max_hold_minutes:
            age_hours = age_minutes / 60
            logger.warning(
                f"TIMEOUT: {pos.get('action')} {pos.get('symbol')} "
                f"open for {age_hours:.1f}h (max {max_hold_minutes/60:.0f}h) — force-closing."
            )
            if pos.get("paper", True):
                log_signal_to_db(
                    {
                        "timestamp":        now.isoformat(),
                        "asset":            (pos.get("symbol") or "").split("/")[0],
                        "symbol":           pos.get("symbol"),
                        "final_action":     "TIMEOUT",
                        "current_price":    pos.get("entry_price"),
                        "stop_loss":        pos.get("stop_loss"),
                        "target":           pos.get("target"),
                        "position_size_usdt": pos.get("notional"),
                        "numerology":       {},
                    },
                    paper=True,
                    notes=f"Force-closed after {age_hours:.1f}h (MAX_OPEN_BARS={config.MAX_OPEN_BARS})",
                )
            else:
                _force_close_live_position(pos)
        else:
            still_open.append(pos)

    _save_open_positions(still_open)


# ── Gap 3: Drawdown halt tracker ──────────────────────────────────────────────

def _load_equity_state() -> dict:
    if _USE_PG:
        return pg_load_equity(user_id=_USER_ID)
    if EQUITY_STATE_FILE.exists():
        with open(EQUITY_STATE_FILE) as f:
            return json.load(f)
    return {
        "peak_equity":    0.0,
        "paper_pnl":      0.0,
        "paper_trades":   0,
        "paper_wins":     0,
        "paper_losses":   0,
        "paper_timeouts": 0,
    }


def _save_equity_state(state: dict):
    if _USE_PG:
        pg_save_equity(state, user_id=_USER_ID)
        return
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EQUITY_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_paper_summary() -> dict:
    """Return the current paper trading P&L summary from equity_state.json."""
    return _load_equity_state()


# ── Paper position P&L monitor ────────────────────────────────────────────────

TAKER_FEE = 0.0005  # 0.05% per side = 0.10% round-trip (Hyperliquid taker)


def _calc_paper_pnl(pos: dict, close_price: float) -> float:
    """
    Calculate realised P&L for a closed paper position.

    P&L = (price_change_per_coin × coins) − round-trip fees
    Coins = notional / entry_price
    """
    entry    = pos.get("entry_price", 0.0)
    notional = pos.get("notional", 0.0)
    action   = pos.get("action", "")
    if entry <= 0 or notional <= 0:
        return 0.0

    coins = notional / entry
    if "BUY" in action:
        raw_pnl = (close_price - entry) * coins
    else:
        raw_pnl = (entry - close_price) * coins

    fees = notional * TAKER_FEE * 2  # entry + exit
    return raw_pnl - fees


def check_paper_positions(current_price: float) -> list[dict]:
    """
    Check all open paper positions against current_price.

    For each position:
      - LONG  (BUY):  TP hit if price ≥ target, SL hit if price ≤ stop_loss
      - SHORT (SELL): TP hit if price ≤ target, SL hit if price ≥ stop_loss

    Returns a list of closed-position dicts (may be empty) so bot_cycle can
    display P&L to the console.
    """
    positions = _load_open_positions()
    if not positions:
        return []

    closed   = []
    still_open = []
    state    = _load_equity_state()

    for pos in positions:
        if not pos.get("paper", True):
            still_open.append(pos)
            continue

        action     = pos.get("action", "")
        stop_loss  = pos.get("stop_loss",  0.0)
        target     = pos.get("target",     0.0)
        entry      = pos.get("entry_price", 0.0)

        result     = None
        close_px   = None

        if "BUY" in action:
            if current_price >= target:
                result, close_px = "TP", target
            elif current_price <= stop_loss:
                result, close_px = "SL", stop_loss
        else:  # SELL / SHORT
            if current_price <= target:
                result, close_px = "TP", target
            elif current_price >= stop_loss:
                result, close_px = "SL", stop_loss

        if result:
            pnl = _calc_paper_pnl(pos, close_px)

            # Update running totals
            state["paper_pnl"]    = state.get("paper_pnl", 0.0) + pnl
            state["paper_trades"] = state.get("paper_trades", 0) + 1
            if pnl > 0:
                state["paper_wins"]   = state.get("paper_wins", 0) + 1
            else:
                state["paper_losses"] = state.get("paper_losses", 0) + 1

            # Persist close record to DB
            _ts  = datetime.now(timezone.utc).isoformat()
            _sym = pos.get("symbol", "")
            _notes = f"Paper close — {result} hit at {close_px:.2f}"
            if _USE_PG:
                pg_log_close(
                    symbol=_sym,
                    action=f"CLOSE_{result}",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    target=target,
                    notional=pos.get("notional", 0.0),
                    close_price=close_px,
                    pnl=round(pnl, 4),
                    result=result,
                    notes=_notes,
                    ts=_ts,
                    user_id=_USER_ID,
                )
            else:
                conn = sqlite3.connect(config.DB_PATH)
                conn.execute("""
                    INSERT INTO signals (
                        timestamp, asset, symbol, action,
                        entry_price, stop_loss, target, position_usdt,
                        paper, notes, close_price, pnl, result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    _ts,
                    _sym.split("/")[0],
                    _sym,
                    f"CLOSE_{result}",
                    entry,
                    stop_loss,
                    target,
                    pos.get("notional"),
                    1,
                    _notes,
                    close_px,
                    round(pnl, 4),
                    result,
                ))
                conn.commit()
                conn.close()

            closed.append({**pos, "result": result, "close_price": close_px, "pnl": pnl})
            logger.info(
                f"[PAPER CLOSE] {result} hit — {action} {pos.get('symbol')} | "
                f"Entry: {entry:.2f} → Close: {close_px:.2f} | "
                f"P&L: {'+'if pnl>=0 else ''}{pnl:.4f} USDT"
            )
        else:
            still_open.append(pos)

    if closed:
        _save_open_positions(still_open)
        _save_equity_state(state)

    return closed


def update_peak_equity(current_equity: float):
    """
    Update the persisted peak-equity record if current_equity is a new high.
    Call this every cycle BEFORE the drawdown check so the peak is always current.
    """
    state = _load_equity_state()
    if current_equity > state.get("peak_equity", 0.0):
        state["peak_equity"] = current_equity
        _save_equity_state(state)
        logger.debug(f"New peak equity: ${current_equity:,.2f}")


def check_drawdown_halt(current_equity: float) -> bool:
    """
    Returns True and logs an alert if drawdown from peak >= MAX_DRAWDOWN_HALT_PCT.
    Trading is skipped for the current cycle when this returns True.
    """
    state = _load_equity_state()
    peak = state.get("peak_equity", 0.0)
    if peak <= 0:
        return False

    drawdown = (peak - current_equity) / peak
    if drawdown >= config.MAX_DRAWDOWN_HALT_PCT:
        logger.error(
            f"DRAWDOWN HALT: equity fell {drawdown*100:.1f}% from peak "
            f"(${peak:,.2f} → ${current_equity:,.2f}). "
            f"Threshold: {config.MAX_DRAWDOWN_HALT_PCT*100:.0f}%. "
            f"No trades will be placed. Manually reset logs/equity_state.json "
            f"after reviewing positions."
        )
        return True
    return False


# ── Paper Trade Logger ─────────────────────────────────────────────────────────

def paper_trade(signal: dict):
    action   = signal["final_action"]
    notional = signal["position_size_usdt"]
    lev_info = compute_leverage_info(notional, signal["current_price"], signal["stop_loss"])
    logger.info(
        f"[PAPER TRADE] {action} {signal['symbol']} | "
        f"Entry: {signal['current_price']} | "
        f"SL: {signal['stop_loss']} ({lev_info['sl_pct']}%) | "
        f"TP: {signal['target']} | "
        f"Notional: ${notional:.2f} | "
        f"Leverage: {lev_info['leverage']}x | "
        f"Margin: ${lev_info['margin_required']:.2f} | "
        f"Liq @ ${lev_info['liq_price_long'] if 'BUY' in action else lev_info['liq_price_short']} "
        f"({'SAFE' if lev_info['safe'] else 'WARNING: LIQ BEFORE SL!'})"
    )
    if not lev_info["safe"]:
        logger.error(
            f"LEVERAGE SAFETY VIOLATION: SL is {lev_info['sl_pct']}% away but "
            f"liquidation is only {lev_info['liq_pct']}% away at {lev_info['leverage']}x leverage. "
            f"Reduce leverage or widen stop-loss."
        )
    log_signal_to_db(signal, paper=True)
    _record_open_position(signal)


# ── Live Trade Executor ────────────────────────────────────────────────────────

def execute_trade(signal: dict):
    """
    Execute a trade on the live exchange.

    Places:
      1. Set leverage on exchange
      2. Market order (entry)
      3. Stop-loss order
      4. Take-profit order
    """
    action = signal["final_action"]
    symbol = signal["symbol"]
    price  = signal["current_price"]

    # Safety check before sending any order
    notional = signal["position_size_usdt"]
    lev_info = compute_leverage_info(notional, price, signal["stop_loss"])
    if not lev_info["safe"]:
        logger.error(
            f"TRADE BLOCKED — leverage safety violation: "
            f"liquidation ({lev_info['liq_pct']}%) is closer than stop-loss ({lev_info['sl_pct']}%). "
            f"Lower LEVERAGE in .env or increase ATR_MULTIPLIER."
        )
        log_signal_to_db(signal, paper=False, notes="BLOCKED: liq before SL")
        return
    stop_loss = signal["stop_loss"]
    target = signal["target"]
    usdt_size = signal["position_size_usdt"]

    if usdt_size <= 0:
        logger.warning(f"Position size is 0 — skipping trade execution.")
        return

    if trades_this_week() >= config.MAX_WEEKLY_TRADES:
        logger.warning(f"Weekly trade cap ({config.MAX_WEEKLY_TRADES}) reached — skipping.")
        log_signal_to_db(signal, paper=False, notes="Weekly cap reached")
        return

    try:
        ex = get_exchange()
        side = "buy" if "BUY" in action else "sell"
        qty  = usdt_size / price  # notional ÷ price = coin quantity

        logger.info(
            f"[LIVE] {side.upper()} {qty:.6f} {symbol} | "
            f"Notional: ${usdt_size:.2f} | "
            f"Leverage: {lev_info['leverage']}x | "
            f"Margin: ${lev_info['margin_required']:.2f} | "
            f"SL: {lev_info['sl_pct']}% | "
            f"Liq: {lev_info['liq_pct']}%"
        )

        # 1. Set leverage on exchange
        set_leverage_on_exchange(symbol)

        # 2. Market entry
        entry_order = ex.create_market_order(symbol, side, qty)
        logger.info(f"Entry order placed: {entry_order.get('id')}")

        # 3. Stop-loss
        sl_side = "sell" if side == "buy" else "buy"
        sl_order = ex.create_order(
            symbol, "stop_market", sl_side, qty,
            params={"stopPrice": stop_loss, "reduceOnly": True}
        )
        logger.info(f"Stop-loss order placed: {sl_order.get('id')} @ {stop_loss}")

        # 4. Take-profit
        tp_order = ex.create_order(
            symbol, "take_profit_market", sl_side, qty,
            params={"stopPrice": target, "reduceOnly": True}
        )
        logger.info(f"Take-profit order placed: {tp_order.get('id')} @ {target}")

        log_signal_to_db(signal, paper=False, notes=f"Entry:{entry_order.get('id')}")
        _record_open_position(signal, order_ids={
            "sl": sl_order.get("id"),
            "tp": tp_order.get("id"),
        })

    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        log_signal_to_db(signal, paper=False, notes=f"ERROR: {e}")


# ── Main dispatch ──────────────────────────────────────────────────────────────

def dispatch_signal(signal: dict, current_equity: float = 0.0):
    """
    Route a signal to paper trading or live execution.

    Backtest-aligned gates applied in order:
      1. Skip COLLECTING_DATA / HOLD / NO_TRADE status signals.
      2. Skip WEAK_BUY / WEAK_SELL — only STRONG signals were traded in the
         backtest (final_backtest.py line 238). Weak signals are logged but
         not executed to keep live behaviour consistent with backtest results.
      3. Drawdown halt — if equity has fallen >= MAX_DRAWDOWN_HALT_PCT from
         the peak recorded in logs/equity_state.json, no trade is placed.
    """
    action = signal.get("final_action", "HOLD")
    status = signal.get("status", "SIGNAL")

    if status != "SIGNAL":
        return

    if action in ("HOLD", "NO_TRADE"):
        logger.info(f"Action is {action} — no trade placed.")
        log_signal_to_db(signal, paper=True, notes="No trade — HOLD/NO_TRADE")
        return

    # Gap 1 — STRONG-only filter
    if action in ("WEAK_BUY", "WEAK_SELL"):
        logger.info(
            f"WEAK signal skipped ({action}) — backtest only trades STRONG signals. "
            f"Log this signal? It will be recorded as NO_TRADE."
        )
        log_signal_to_db(signal, paper=True, notes="Skipped — WEAK signal (STRONG-only policy)")
        return

    # Gap 3 — Drawdown halt
    if current_equity > 0 and check_drawdown_halt(current_equity):
        log_signal_to_db(signal, paper=True, notes="Skipped — drawdown halt active")
        return

    if config.PAPER_TRADING:
        paper_trade(signal)
    else:
        execute_trade(signal)
