"""
Dashboard API — list all bots and query signals/positions by user_id.

Frontend (astro-bot-frontend) expects:
  - GET /api/users           → list of User { id, name, bot_dir, color }
  - GET /api/users/{uid}/signals, /positions, /equity, /trades, /stats, /latest-signal

Run with: uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
"""
import os
from datetime import datetime, timezone

# Load .env when running as main (cwd = project root)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if not os.getenv("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL must be set for the dashboard API")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db.postgres as pg

app = FastAPI(title="Astro-Bot Dashboard API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend User type: { id, name, bot_dir, color }. Colors for bot dropdown.
_USER_COLORS = ["#58a6ff", "#3fb950", "#d2a8ff", "#f85149", "#f0883e"]


def _configured_user_ids() -> list[str]:
    """
    Optional static bot list from env so dropdown can show bots before they
    write any rows to Postgres.
    Supported vars:
      - DASHBOARD_BOT_USERS="default,nakshatraon,bot3"
      - BOT_USERS="default,nakshatraon,bot3"
    """
    raw = os.getenv("DASHBOARD_BOT_USERS") or os.getenv("BOT_USERS") or ""
    out: list[str] = []
    for tok in raw.split(","):
        uid = tok.strip()
        if uid:
            out.append(uid)
    return out


def _bot_users_to_frontend():
    """Map pg_list_bot_users() to frontend User[] shape."""
    discovered = [b["user_id"] for b in pg.pg_list_bot_users()]
    configured = _configured_user_ids()

    merged: list[str] = []
    seen: set[str] = set()
    for uid in configured + discovered:
        if uid not in seen:
            merged.append(uid)
            seen.add(uid)
    if "default" not in seen:
        merged.insert(0, "default")

    bots = [{"user_id": uid} for uid in merged]
    return [
        {
            "id": b["user_id"],
            "name": b["user_id"].replace("_", " ").title(),
            "bot_dir": b["user_id"],
            "color": _USER_COLORS[i % len(_USER_COLORS)],
        }
        for i, b in enumerate(bots)
    ]


def _equity_with_computed_pnl(uid: str) -> dict:
    """
    Build equity payload with PnL counters computed from closed trades.
    This avoids stale/missing KV snapshots causing paper PnL to show 0.
    """
    equity = pg.pg_load_equity(user_id=uid) or {}
    agg = pg.pg_compute_pnl_from_signals(user_id=uid)
    merged = dict(equity)
    merged["paper_pnl"] = float(agg.get("paper_pnl", 0.0))
    merged["paper_trades"] = int(agg.get("paper_trades", 0))
    merged["paper_wins"] = int(agg.get("paper_wins", 0))
    merged["paper_losses"] = int(agg.get("paper_losses", 0))
    return merged


# ── /api/* routes (used by frontend proxy) ────────────────────────────────────

@app.get("/api/users")
def api_list_users():
    """List all bot user_ids as User[] for the frontend header dropdown."""
    return _bot_users_to_frontend()


@app.get("/api/users/{uid}/signals")
def api_get_signals(uid: str, limit: int = Query(100, ge=1, le=500)):
    """Signals (trade log) for this bot. Returns array directly."""
    rows = pg.pg_query_signals(limit=limit, closed_only=False, user_id=uid)
    return [dict(r) for r in rows]


@app.get("/api/users/{uid}/positions")
def api_get_positions(uid: str):
    """Open positions for this bot. Returns array directly."""
    return pg.pg_load_positions(user_id=uid)


@app.get("/api/users/{uid}/equity")
def api_get_equity(uid: str):
    """Equity state for this bot."""
    return _equity_with_computed_pnl(uid)


@app.get("/api/users/{uid}/trades")
def api_get_trades(uid: str, limit: int = Query(200, ge=1, le=500)):
    """Closed trades (signals with pnl) for this bot. Returns array directly."""
    rows = pg.pg_query_signals(limit=limit, closed_only=True, user_id=uid)
    return [dict(r) for r in rows]


@app.get("/api/users/{uid}/stats")
def api_get_stats(uid: str):
    """Dashboard stats: trades, wins, losses, win_rate, total_pnl, avg_win, avg_loss, etc."""
    s = pg.pg_get_stats(user_id=uid)
    eq = _equity_with_computed_pnl(uid)
    pos = pg.pg_load_positions(user_id=uid)
    raw_start = eq.get("paper_starting_equity")
    try:
        paper_starting_equity = float(raw_start) if raw_start is not None else 0.0
    except (TypeError, ValueError):
        paper_starting_equity = 0.0
    return {
        **s,
        "peak_equity": eq.get("peak_equity", 0.0),
        "paper_pnl": eq.get("paper_pnl", 0.0),
        "paper_starting_equity": paper_starting_equity,
        "open_positions": len(pos),
    }


class PaperTradeIn(BaseModel):
    user_id: str
    side: str
    entry: float
    sl: float
    tp: float
    notional: float
    signal: str = "MANUAL"


@app.post("/api/paper/trade")
def api_open_paper_trade(body: PaperTradeIn):
    side = (body.side or "").upper().strip()
    if side not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
    if body.entry <= 0 or body.sl <= 0 or body.tp <= 0 or body.notional <= 0:
        raise HTTPException(status_code=400, detail="entry/sl/tp/notional must be > 0")
    if side == "BUY" and body.sl >= body.entry:
        raise HTTPException(status_code=400, detail="SL must be below entry for BUY")
    if side == "SELL" and body.sl <= body.entry:
        raise HTTPException(status_code=400, detail="SL must be above entry for SELL")

    positions = pg.pg_load_positions(user_id=body.user_id)
    new_pos = {
        "side": side,
        "signal": body.signal or "MANUAL",
        "entry": float(body.entry),
        "sl": float(body.sl),
        "tp": float(body.tp),
        "notional": float(body.notional),
        "risk": abs(body.entry - body.sl) / body.entry * body.notional,
        "age": 0,
        "open_ts": datetime.now(timezone.utc).isoformat()[:16],
        "paper": True,
    }
    positions.append(new_pos)
    pg.pg_save_positions(positions, user_id=body.user_id)
    return {"status": "ok", "position": new_pos}


@app.delete("/api/paper/trade/{uid}/{index}")
def api_close_paper_trade(uid: str, index: int):
    positions = pg.pg_load_positions(user_id=uid)
    if index < 0 or index >= len(positions):
        raise HTTPException(status_code=400, detail="Invalid position index")
    removed = positions.pop(index)
    pg.pg_save_positions(positions, user_id=uid)
    return {"status": "ok", "removed": removed}


@app.get("/api/users/{uid}/latest-signal")
def api_get_latest_signal(uid: str):
    """Most recent signal for this bot. Returns single object or null."""
    rows = pg.pg_query_signals(limit=1, closed_only=False, user_id=uid)
    if not rows:
        return None
    return dict(rows[0])


# ── Legacy routes (optional; frontend uses /api/*) ────────────────────────────

@app.get("/bots")
def list_bots():
    """List all bot user_ids (legacy)."""
    return {"bots": pg.pg_list_bot_users()}


@app.get("/signals")
def get_signals(
    user_id: str = Query(..., description="BOT_USER_ID of the bot"),
    limit: int = Query(100, ge=1, le=500),
    closed_only: bool = Query(False),
):
    rows = pg.pg_query_signals(limit=limit, closed_only=closed_only, user_id=user_id)
    return {"user_id": user_id, "signals": [dict(r) for r in rows]}


@app.get("/positions")
def get_positions(user_id: str = Query(..., description="BOT_USER_ID of the bot")):
    return {"user_id": user_id, "positions": pg.pg_load_positions(user_id=user_id)}


@app.get("/equity")
def get_equity(user_id: str = Query(..., description="BOT_USER_ID of the bot")):
    return {"user_id": user_id, "equity": pg.pg_load_equity(user_id=user_id)}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "database": "postgres"}
