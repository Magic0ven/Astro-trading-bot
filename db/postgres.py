"""
PostgreSQL adapter for Astro-Bot.

Multi-user support:  every read/write is scoped to a BOT_USER_ID.
This allows multiple bot instances (each a separate Railway service) to share
one Postgres database while keeping their data fully isolated.

Activated automatically when DATABASE_URL is set.
Falls back to SQLite/JSON when DATABASE_URL is absent (local development).

Tables
------
  signals   — trade / signal log   (one row per signal, scoped by user_id)
  kv_store  — key/value blobs for  open_positions + equity_state
              key format:  "{user_id}:{data_type}"
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
BOT_USER_ID:  str           = os.getenv("BOT_USER_ID", "default")


# ── Connection ─────────────────────────────────────────────────────────────────

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — cannot connect to Postgres")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# ── Schema bootstrap ───────────────────────────────────────────────────────────

def init_schema():
    """Create tables if they do not yet exist. Safe to call on every startup."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id              SERIAL PRIMARY KEY,
                    user_id         TEXT NOT NULL DEFAULT 'default',
                    timestamp       TEXT,
                    asset           TEXT,
                    symbol          TEXT,
                    action          TEXT,
                    entry_price     DOUBLE PRECISION,
                    stop_loss       DOUBLE PRECISION,
                    target          DOUBLE PRECISION,
                    position_usdt   DOUBLE PRECISION,
                    western_score   DOUBLE PRECISION,
                    vedic_score     DOUBLE PRECISION,
                    western_slope   DOUBLE PRECISION,
                    vedic_slope     DOUBLE PRECISION,
                    numerology_mult DOUBLE PRECISION,
                    nakshatra       TEXT,
                    paper           INTEGER DEFAULT 1,
                    notes           TEXT,
                    close_price     DOUBLE PRECISION,
                    pnl             DOUBLE PRECISION,
                    result          TEXT,
                    full_signal     TEXT
                )
            """)
            try:
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default'")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS full_signal TEXT")
            except Exception:
                pass
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            # Index for fast per-user signal queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_user_id
                ON signals (user_id, id DESC)
            """)
        conn.commit()


# ── Signal / trade log ─────────────────────────────────────────────────────────

def pg_log_signal(signal: dict, paper: bool, notes: str = "",
                  user_id: str = None, full_signal_json: str = None):
    uid = user_id or BOT_USER_ID
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (
                    user_id,
                    timestamp, asset, symbol, action,
                    entry_price, stop_loss, target, position_usdt,
                    western_score, vedic_score, western_slope, vedic_slope,
                    numerology_mult, nakshatra, paper, notes, full_signal
                ) VALUES (
                    %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
            """, (
                uid,
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
                full_signal_json or "",
            ))
        conn.commit()


def pg_log_close(
    symbol: str,
    action: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    notional: float,
    close_price: float,
    pnl: float,
    result: str,
    notes: str = "",
    ts: Optional[str] = None,
    user_id: str = None,
):
    uid = user_id or BOT_USER_ID
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    asset = (symbol or "").split("/")[0]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (
                    user_id,
                    timestamp, asset, symbol, action,
                    entry_price, stop_loss, target, position_usdt,
                    paper, notes, close_price, pnl, result
                ) VALUES (
                    %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    1,  %s, %s,        %s, %s
                )
            """, (
                uid,
                ts, asset, symbol, action,
                entry_price, stop_loss, target, notional,
                notes, close_price, pnl, result,
            ))
        conn.commit()


def pg_query_signals(limit: int = 200, closed_only: bool = False,
                     user_id: str = None) -> list[dict]:
    uid   = user_id or BOT_USER_ID
    where = f"WHERE user_id = %s" + (" AND pnl IS NOT NULL" if closed_only else "")
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT
                    id, user_id,
                    timestamp, symbol, action,
                    western_score, vedic_score,
                    western_slope  AS western_signal,
                    vedic_slope    AS vedic_signal,
                    nakshatra, entry_price, stop_loss, target,
                    position_usdt  AS position_size_usdt,
                    paper, close_price, pnl, result, notes,
                    full_signal
                FROM signals
                {where}
                ORDER BY id DESC
                LIMIT %s
            """, (uid, limit))
            return [dict(r) for r in cur.fetchall()]


# ── Open positions  (kv key: "{user_id}:open_positions") ─────────────────────

def pg_load_positions(user_id: str = None) -> list:
    key = f"{user_id or BOT_USER_ID}:open_positions"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM kv_store WHERE key = %s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else []


def pg_save_positions(positions: list, user_id: str = None):
    key = f"{user_id or BOT_USER_ID}:open_positions"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kv_store (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (key, json.dumps(positions)))
        conn.commit()


# ── Equity state  (kv key: "{user_id}:equity_state") ─────────────────────────

_DEFAULT_EQUITY: dict = {
    "peak_equity":    0.0,
    "paper_pnl":      0.0,
    "paper_trades":   0,
    "paper_wins":     0,
    "paper_losses":   0,
    "paper_timeouts": 0,
}


def pg_load_equity(user_id: str = None) -> dict:
    key = f"{user_id or BOT_USER_ID}:equity_state"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM kv_store WHERE key = %s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else dict(_DEFAULT_EQUITY)


def pg_save_equity(state: dict, user_id: str = None):
    key = f"{user_id or BOT_USER_ID}:equity_state"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kv_store (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (key, json.dumps(state)))
        conn.commit()
