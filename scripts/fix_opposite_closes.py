"""
Repair historical CLOSE_OPPOSITE / OPPOSITE_SIGNAL rows in Postgres.

For each such row that does NOT have a matching opposite open trade afterwards,
we replay price action on Hyperliquid candles and convert the close into a
TP/SL/TIMEOUT result with recomputed close_price + pnl.

Run once:
    DATABASE_URL=... python -m scripts.fix_opposite_closes
"""
from __future__ import annotations

import json
import os
import ssl
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import certifi
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib import request as urlrequest


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:mjxEujUmilDKizIWqjsWsvQsCeYEckSE@turntable.proxy.rlwy.net:31653/railway",
)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _post_json(payload: dict) -> list:
    req = urlrequest.Request(
        HL_INFO_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=20, context=SSL_CTX) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected candleSnapshot response type: {type(data).__name__}")
    return data


def _symbol_to_coin(symbol: str) -> str:
    # "BTC/USDC:USDC" -> "BTC"
    left = (symbol or "").split("/", 1)[0]
    return left.strip().upper()


def _fetch_hl_candles(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Fetch Hyperliquid candles via /info candleSnapshot.
    Returns list of (t, T, o, h, l, c) sorted by t.
    """
    coin = _symbol_to_coin(symbol)
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    raw = _post_json(payload)
    out: List[Tuple[int, int, float, float, float, float]] = []
    for c in raw:
        try:
            t = int(c["t"])
            T = int(c["T"])
            o = float(c["o"])
            h = float(c["h"])
            l = float(c["l"])
            cl = float(c["c"])
        except Exception:
            continue
        out.append((t, T, o, h, l, cl))
    out.sort(key=lambda z: z[0])
    return out


def _infer_side(entry: float, sl: float, tp: float) -> str | None:
    if tp < entry < sl:
        return "short"
    if sl < entry < tp:
        return "long"
    return None


def _repair_outcome(
    side: str,
    entry: float,
    sl: float,
    tp: float,
    candles: List[Tuple[int, int, float, float, float, float]],
) -> Tuple[str, float]:
    """
    Given side+levels and future candles, return (result, exit_price).

    Logic mirrors the Hyperliquid execution sims:
      - Walk candles forward from the *next* candle after the open time.
      - If both TP and SL are in the same candle, we take SL-first
        (conservative) here to avoid overstating performance.
      - If neither hit, TIMEOUT at last close.
    """
    if not candles:
        return "TIMEOUT", entry

    outcome = "TIMEOUT"
    exit_price = candles[-1][5]

    for _t, _T, _o, h, l, _c in candles:
        if side == "short":
            sl_hit = h >= sl
            tp_hit = l <= tp
            if sl_hit and tp_hit:
                outcome = "SL"
                exit_price = sl
                break
            if sl_hit:
                outcome = "SL"
                exit_price = sl
                break
            if tp_hit:
                outcome = "TP"
                exit_price = tp
                break
        else:
            sl_hit = l <= sl
            tp_hit = h >= tp
            if sl_hit and tp_hit:
                outcome = "SL"
                exit_price = sl
                break
            if sl_hit:
                outcome = "SL"
                exit_price = sl
                break
            if tp_hit:
                outcome = "TP"
                exit_price = tp
                break

    return outcome, float(exit_price)


def _compute_pnl(side: str, entry: float, exit_price: float, notional: float) -> float:
    if entry <= 0 or notional <= 0:
        return 0.0
    coins = notional / entry
    if side == "long":
        raw = (exit_price - entry) * coins
    else:
        raw = (entry - exit_price) * coins
    # Approximate Hyperliquid taker fees (round trip)
    fee = notional * 0.0005 * 2
    return round(raw - fee, 4)


def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be set")

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    conn.autocommit = False

    print("Loading OPPOSITE_SIGNAL closes from Postgres…")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id, user_id, timestamp, symbol, action,
                entry_price, stop_loss, target, position_usdt,
                close_price, pnl, result, notes
            FROM signals
            WHERE result = 'OPPOSITE_SIGNAL'
            ORDER BY id
            """
        )
        rows = cur.fetchall()

    if not rows:
        print("No OPPOSITE_SIGNAL rows found — nothing to repair.")
        conn.close()
        return

    # Preload all STRONG open signals (pnl IS NULL) to detect real flips.
    print("Indexing open STRONG signals to detect true flips…")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, user_id, symbol, action, pnl
            FROM signals
            WHERE pnl IS NULL
              AND action IN ('STRONG_BUY', 'STRONG_SELL')
            """
        )
        opens = cur.fetchall()

    opens_by_user_symbol: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in opens:
        key = (r["user_id"] or "default", r["symbol"])
        opens_by_user_symbol[key].append(int(r["id"]))
    for key in opens_by_user_symbol:
        opens_by_user_symbol[key].sort()

    # Decide which opposite closes are "fake" (no later opposite open).
    repairs: List[dict] = []
    min_ts = None
    for r in rows:
        uid = (r["user_id"] or "default") or "default"
        sym = r["symbol"]
        close_id = int(r["id"])
        key = (uid, sym)
        open_ids = opens_by_user_symbol.get(key, [])
        has_later_open = any(oid > close_id for oid in open_ids)
        if has_later_open:
            # This close belongs to a real flip — leave it as-is.
            continue

        ts = datetime.fromisoformat((r["timestamp"] or "").replace("Z", "+00:00"))
        if (min_ts is None) or ts < min_ts:
            min_ts = ts

        repairs.append(r)

    if not repairs:
        print("All OPPOSITE_SIGNAL rows have matching later opens — nothing to repair.")
        conn.close()
        return

    print(f"Found {len(repairs)} OPPOSITE_SIGNAL rows to repair.")

    # Fetch Hyperliquid candles once per symbol for a broad time range.
    # For simplicity we assume BTC-only here, matching current bot config.
    start_ms = int((min_ts.timestamp() - 6 * 3600) * 1000)
    end_ms = int((datetime.now(timezone.utc).timestamp() + 6 * 3600) * 1000)

    # Use 1h candles as a compromise between precision and call volume.
    print("Fetching Hyperliquid candles…")
    candles_1h = _fetch_hl_candles("BTC/USDC:USDC", "1h", start_ms, end_ms)
    if not candles_1h:
        print("No Hyperliquid candles fetched — aborting.")
        conn.close()
        return

    # Index candles by time for fast slicing.
    ts_list = [c[0] for c in candles_1h]

    def slice_from(ts_iso: str) -> List[Tuple[int, int, float, float, float, float]]:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        ts_ms = int(dt.timestamp() * 1000)
        # Find first candle with open time >= ts_ms
        lo, hi = 0, len(ts_list)
        while lo < hi:
            mid = (lo + hi) // 2
            if ts_list[mid] >= ts_ms:
                hi = mid
            else:
                lo = mid + 1
        return candles_1h[lo:]

    # Build and apply updates.
    print("Replaying and preparing updates…")
    updates: List[Tuple[float, float, str, int]] = []
    for r in repairs:
        entry = float(r["entry_price"] or 0.0)
        sl = float(r["stop_loss"] or 0.0)
        tp = float(r["target"] or 0.0)
        notional = float(r["position_usdt"] or 0.0)
        side = _infer_side(entry, sl, tp)
        if side is None or entry <= 0 or sl <= 0 or tp <= 0 or notional <= 0:
            # Cannot repair reliably — leave as-is.
            continue

        future = slice_from(r["timestamp"])
        result, exit_price = _repair_outcome(side, entry, sl, tp, future)
        pnl = _compute_pnl("long" if side == "long" else "short", entry, exit_price, notional)
        updates.append((exit_price, pnl, result, int(r["id"])))

    if not updates:
        print("No repairable rows after geometry/level checks.")
        conn.close()
        return

    print(f"Applying {len(updates)} updates to Postgres…")
    with conn.cursor() as cur:
        for close_price, pnl, result, row_id in updates:
            cur.execute(
                """
                UPDATE signals
                SET close_price = %s,
                    pnl         = %s,
                    result      = %s,
                    notes       = COALESCE(notes, '') || ' | Repaired from OPPOSITE_SIGNAL via Hyperliquid replay'
                WHERE id = %s
                """,
                (close_price, pnl, result, row_id),
            )
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()

