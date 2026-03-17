"""
Dashboard API — list all bots and query signals/positions by user_id.

Frontend (astro-bot-frontend) expects:
  - GET /api/users           → list of User { id, name, bot_dir, color }
  - GET /api/users/{uid}/signals, /positions, /equity, /trades, /stats, /latest-signal

Run with: uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
"""
import os

# Load .env when running as main (cwd = project root)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if not os.getenv("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL must be set for the dashboard API")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

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


def _bot_users_to_frontend():
    """Map pg_list_bot_users() to frontend User[] shape."""
    bots = pg.pg_list_bot_users()
    return [
        {
            "id": b["user_id"],
            "name": b["user_id"].replace("_", " ").title(),
            "bot_dir": b["user_id"],
            "color": _USER_COLORS[i % len(_USER_COLORS)],
        }
        for i, b in enumerate(bots)
    ]


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
    return pg.pg_load_equity(user_id=uid)


@app.get("/api/users/{uid}/trades")
def api_get_trades(uid: str, limit: int = Query(200, ge=1, le=500)):
    """Closed trades (signals with pnl) for this bot. Returns array directly."""
    rows = pg.pg_query_signals(limit=limit, closed_only=True, user_id=uid)
    return [dict(r) for r in rows]


@app.get("/api/users/{uid}/stats")
def api_get_stats(uid: str):
    """Dashboard stats: trades, wins, losses, win_rate, total_pnl, avg_win, avg_loss, etc."""
    s = pg.pg_get_stats(user_id=uid)
    eq = pg.pg_load_equity(user_id=uid)
    pos = pg.pg_load_positions(user_id=uid)
    return {
        **s,
        "peak_equity": eq.get("peak_equity", 0.0),
        "paper_pnl": eq.get("paper_pnl", 0.0),
        "open_positions": len(pos),
    }


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
