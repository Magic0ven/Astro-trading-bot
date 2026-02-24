"""
Signal Engine — the brain of the bot.

Combines:
  1. Astrological composite scores (Western + Vedic)
  2. Numerology multiplier
  3. Score history (Medium + Slope)
  4. Dual-system agreement gate

Outputs a final trade signal: STRONG_BUY | WEAK_BUY | STRONG_SELL | WEAK_SELL | NO_TRADE | HOLD
"""
from datetime import date, datetime, timezone
from typing import Optional
from loguru import logger

import config
from core.astro_engine import (
    compute_composite_score,
    get_sky_state,
)
from core.numerology import full_numerology_report, numerology_multiplier
from core.score_history import ScoreHistory


# ── Single-system signal ───────────────────────────────────────────────────────

def _system_signal(score: float, medium: float, slope: Optional[float]) -> str:
    """
    Evaluate a single system (Western or Vedic) independently.
    
    Returns: "BUY" | "SELL" | "HOLD"
    """
    if slope is None:
        return "HOLD"

    above_medium = score > medium
    slope_positive = slope > config.SLOPE_THRESHOLD
    slope_sharply_negative = slope < -config.SLOPE_THRESHOLD

    if above_medium and slope_positive:
        return "BUY"
    elif not above_medium or slope_sharply_negative:
        return "SELL"
    else:
        return "HOLD"


# ── Dual-system agreement gate ─────────────────────────────────────────────────

def dual_system_gate(western_signal: str, vedic_signal: str) -> str:
    """
    Combine the two system signals into a single action.
    
    Agreement Table:
      BUY  + BUY  → STRONG_BUY
      SELL + SELL → STRONG_SELL
      BUY  + HOLD → WEAK_BUY
      SELL + HOLD → WEAK_SELL
      HOLD + HOLD → HOLD
      BUY  + SELL → NO_TRADE  (contradiction — sit out)
    """
    pair = frozenset({western_signal, vedic_signal})

    if pair == {"BUY"}:
        return "STRONG_BUY"
    elif pair == {"SELL"}:
        return "STRONG_SELL"
    elif pair == {"BUY", "HOLD"}:
        return "WEAK_BUY"
    elif pair == {"SELL", "HOLD"}:
        return "WEAK_SELL"
    elif pair == {"HOLD"}:
        return "HOLD"
    else:
        return "NO_TRADE"


# ── Position size calculator ───────────────────────────────────────────────────

def calculate_position_size(
    entry_price: float,
    stop_loss_price: float,
    num_multiplier: float,
    signal_strength: str,
    capital: float = 0.0,
) -> float:
    """
    Returns position size in USDT notional value.

    - Full size on STRONG signals
    - Half size on WEAK signals
    - Zero on HOLD / NO_TRADE

    capital: live effective capital (account_balance * CAPITAL_PCT).
             Falls back to config.CAPITAL_USDT if not provided.
    """
    effective = capital if capital > 0 else config.CAPITAL_USDT
    risk_amount = effective * config.RISK_PER_TRADE_PCT
    distance = abs(entry_price - stop_loss_price)
    if distance == 0:
        return 0.0

    base_qty = (risk_amount / distance) * entry_price  # notional USDT

    if signal_strength in ("STRONG_BUY", "STRONG_SELL"):
        size_factor = 1.0
    elif signal_strength in ("WEAK_BUY", "WEAK_SELL"):
        size_factor = 0.5
    else:
        return 0.0

    return base_qty * size_factor * num_multiplier


# ── Master Signal Generator ────────────────────────────────────────────────────

def generate_signal(
    asset_dna: dict,
    score_history: ScoreHistory,
    current_price: float,
    atr: float,
    today: Optional[date] = None,
    jd: Optional[float] = None,
    current_ema: float = 0.0,
    capital: float = 0.0,
) -> dict:
    """
    Full signal generation pipeline for one bot cycle.

    Args:
        asset_dna:      Asset configuration dict from assets_dna.json
        score_history:  Rolling ScoreHistory instance (persists across cycles)
        current_price:  Latest market price
        atr:            ATR(14) on 4h candles for stop-loss calculation
        today:          Override date (for backtesting)
        jd:             Override Julian Date (for backtesting)
        current_ema:    Live EMA value for trend filter (0.0 = skip filter)
        capital:        Live effective capital (balance * CAPITAL_PCT).
                        Uses config.CAPITAL_USDT as fallback if 0.

    Returns:
        Comprehensive signal dict with all intermediate values
    """
    now = datetime.now(timezone.utc)
    if today is None:
        today = now.date()

    # 1. Sky state
    sky = get_sky_state(jd)

    # 2. Numerology
    from dateutil.parser import parse as parse_date
    genesis_date = parse_date(asset_dna["genesis_datetime"]).date()
    num_report = full_numerology_report(asset_dna.get("name", ""), genesis_date, today)
    num_mult = num_report["multiplier"]

    # 3. Natal positions
    natal_western = asset_dna.get("natal_western") or {}
    natal_vedic = asset_dna.get("natal_vedic") or {}

    if not natal_western:
        logger.warning("Natal Western positions missing — run scripts/calculate_natal.py")
    if not natal_vedic:
        logger.warning("Natal Vedic positions missing — run scripts/calculate_natal.py")

    # 4. Composite scores
    dasha_weights = asset_dna.get("dasha_weights")
    nk_mult = sky["nakshatra_multiplier"]

    western_score = compute_composite_score(
        live_positions=sky["western"],
        natal_positions=natal_western,
    )

    vedic_score = compute_composite_score(
        live_positions=sky["vedic"],
        natal_positions=natal_vedic,
        dasha_weights=dasha_weights,
        nakshatra_mult=nk_mult,
    )

    # Apply numerology multiplier to scores
    western_score_adj = western_score * num_mult
    vedic_score_adj = vedic_score * num_mult

    # 5. Update history
    score_history.push(western_score_adj, vedic_score_adj)
    hist = score_history.summary()

    # 6. Generate individual signals (only if history is ready)
    if not score_history.is_ready():
        bars = score_history.bars_collected()
        needed = 2
        logger.info(
            f"Collecting history — bar {bars}/{needed}. "
            f"Signal will fire on next cycle."
        )
        return {
            "status":           "COLLECTING_DATA",
            "final_action":     "COLLECTING_DATA",
            "asset":            asset_dna.get("name", ""),
            "symbol":           asset_dna.get("symbol", ""),
            "current_price":    current_price,
            "ema_value":        round(current_ema, 2) if current_ema else None,
            "effective_capital": round(capital if capital > 0 else config.CAPITAL_USDT, 2),
            "western_score":    round(western_score_adj, 4),
            "vedic_score":      round(vedic_score_adj, 4),
            "western_medium":   0.0,
            "vedic_medium":     0.0,
            "western_slope":    0.0,
            "vedic_slope":      0.0,
            "western_signal":   "",
            "vedic_signal":     "",
            "nakshatra":        sky.get("nakshatra", ""),
            "nakshatra_multiplier": sky.get("nakshatra_multiplier", 1.0),
            "moon_fast":        sky.get("moon_fast", False),
            "retrograde_western": sky.get("retrograde_planets_western", []),
            "retrograde_vedic": sky.get("retrograde_planets_vedic", []),
            "stop_loss":        0.0,
            "target":           0.0,
            "position_size_usdt": 0.0,
            "numerology":       num_report,
            "history":          hist,
            "bars_collected":   bars,
            "bars_needed":      needed,
        }

    w_medium = hist["western"]["medium"]
    w_slope = hist["western"]["slope"]
    v_medium = hist["vedic"]["medium"]
    v_slope = hist["vedic"]["slope"]

    western_signal = _system_signal(western_score_adj, w_medium, w_slope)
    vedic_signal = _system_signal(vedic_score_adj, v_medium, v_slope)
    final_action = dual_system_gate(western_signal, vedic_signal)

    # 7. Reliability filters (must match backtest logic)
    filter_reason = None

    # 7a. Nakshatra block — only active when NAKSHATRA_FILTER=true in .env
    if config.NAKSHATRA_FILTER and final_action not in ("HOLD", "NO_TRADE"):
        nk = sky["nakshatra"]
        if nk in config.TRADE_UNFAVORABLE_NAKSHATRAS:
            filter_reason = f"NAKSHATRA_BLOCK:{nk}"
            final_action = "NO_TRADE"

    # 7b. EMA trend filter — skip trades that fight the short-term trend
    # Uses EMA(20) on 4h candles (backtested optimal: 151.9% CAGR 2022–2025)
    if (
        filter_reason is None
        and final_action not in ("HOLD", "NO_TRADE")
        and current_ema > 0
        and config.EMA_FILTER != "none"
    ):
        ema = current_ema
        if "SELL" in final_action and current_price > ema:
            filter_reason = f"EMA_FILTER:price({current_price:.2f})>EMA({ema:.2f})_blocked_SELL"
            final_action = "NO_TRADE"
        elif config.EMA_FILTER == "two_way" and "BUY" in final_action and current_price < ema:
            filter_reason = f"EMA_FILTER:price({current_price:.2f})<EMA({ema:.2f})_blocked_BUY"
            final_action = "NO_TRADE"

    if filter_reason:
        logger.info(f"Signal filtered — {filter_reason}")

    # 9. Stop-loss & target
    stop_loss = current_price - (atr * config.ATR_MULTIPLIER) if "BUY" in final_action \
        else current_price + (atr * config.ATR_MULTIPLIER)
    target = current_price + (abs(current_price - stop_loss) * config.RR_RATIO) if "BUY" in final_action \
        else current_price - (abs(current_price - stop_loss) * config.RR_RATIO)

    # 10. Position size  (uses live capital if provided, else config fallback)
    effective_capital = capital if capital > 0 else config.CAPITAL_USDT
    position_size = calculate_position_size(
        entry_price=current_price,
        stop_loss_price=stop_loss,
        num_multiplier=num_mult,
        signal_strength=final_action,
        capital=effective_capital,
    )

    signal = {
        "timestamp": now.isoformat(),
        "asset": asset_dna.get("name", "Unknown"),
        "symbol": asset_dna.get("symbol", ""),
        "status": "SIGNAL",
        "final_action": final_action,
        "filter_reason": filter_reason,
        "ema_value": round(current_ema, 2) if current_ema else None,
        "effective_capital": round(effective_capital, 2),
        "western_signal": western_signal,
        "vedic_signal": vedic_signal,
        "western_score": round(western_score_adj, 4),
        "vedic_score": round(vedic_score_adj, 4),
        "western_medium": round(w_medium, 4),
        "vedic_medium": round(v_medium, 4),
        "western_slope": round(w_slope, 4),
        "vedic_slope": round(v_slope, 4),
        "numerology": num_report,
        "current_price": current_price,
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "position_size_usdt": round(position_size, 2),
        "moon_fast": sky["moon_fast"],
        "nakshatra": sky["nakshatra"],
        "nakshatra_multiplier": nk_mult,
        "retrograde_western": sky["retrograde_planets_western"],
        "retrograde_vedic": sky["retrograde_planets_vedic"],
        "history": hist,
    }

    _log_signal(signal)
    return signal


def _log_signal(signal: dict):
    action = signal["final_action"]
    color_map = {
        "STRONG_BUY": "green",
        "WEAK_BUY": "cyan",
        "STRONG_SELL": "red",
        "WEAK_SELL": "yellow",
        "NO_TRADE": "magenta",
        "HOLD": "white",
    }
    logger.info(
        f"[{signal['timestamp']}] {signal['asset']} | "
        f"ACTION: {action} | "
        f"W: {signal['western_score']:.2f} (slope {signal['western_slope']:+.2f}) | "
        f"V: {signal['vedic_score']:.2f} (slope {signal['vedic_slope']:+.2f}) | "
        f"Num: {signal['numerology']['label']} | "
        f"Nakshatra: {signal['nakshatra']} | "
        f"Entry: {signal['current_price']} SL: {signal['stop_loss']} TP: {signal['target']}"
    )
