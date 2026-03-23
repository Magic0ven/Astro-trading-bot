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


# ── Decisive overlay (best-call rules from win-rate analysis) ──────────────────

def apply_decisive_overlay(action: str, astro_events: Optional[dict] = None) -> str:
    """
    When an astro event is active, upgrade the weak signal in the event's best-call
    direction to strong. Based on backtest win-rate:
      Jupiter–Uranus conjunction → best call LONG
      Saturn–Pluto conjunction   → best call SHORT
      Mercury retrograde        → best call LONG
      Saturn retrograde         → best call LONG
      Full moon / new moon      → best call SHORT
    """
    if not astro_events:
        return action

    out = action

    if astro_events.get("jupiter_uranus_active") and astro_events.get("jupiter_uranus_best_call") == "LONG":
        if out == "WEAK_BUY":
            out = "STRONG_BUY"
    if astro_events.get("saturn_pluto_active") and astro_events.get("saturn_pluto_best_call") == "SHORT":
        if out == "WEAK_SELL":
            out = "STRONG_SELL"
    if astro_events.get("mercury_retrograde_active") and astro_events.get("mercury_retrograde_best_call") == "LONG":
        if out == "WEAK_BUY":
            out = "STRONG_BUY"
    if astro_events.get("saturn_retrograde_active") and astro_events.get("saturn_retrograde_best_call") == "LONG":
        if out == "WEAK_BUY":
            out = "STRONG_BUY"
    if astro_events.get("full_moon_active") and astro_events.get("moon_phase_best_call") == "SHORT":
        if out == "WEAK_SELL":
            out = "STRONG_SELL"
    if astro_events.get("new_moon_active") and astro_events.get("moon_phase_best_call") == "SHORT":
        if out == "WEAK_SELL":
            out = "STRONG_SELL"

    return out


def short_confidence_score(score: float, medium: float, slope: Optional[float]) -> float:
    """
    Heuristic confidence score for SELL quality in [0, 1].
    Higher means stronger bearish structure (score below medium + negative slope).
    """
    if slope is None:
        return 0.0

    # Bearish distance below medium (normalized, robust to scale drift).
    denom = max(abs(medium), 1.0)
    gap = max(0.0, (medium - score) / denom)
    gap_term = min(1.0, gap / 0.25)  # full credit once score is 25% below medium

    # Bearish slope strength beyond threshold.
    slope_excess = max(0.0, (-slope) - config.SLOPE_THRESHOLD)
    slope_term = min(1.0, slope_excess / max(config.SLOPE_THRESHOLD, 1e-6))

    # Weighted blend: score regime + momentum.
    return (0.6 * gap_term) + (0.4 * slope_term)


# ── Position size calculator ───────────────────────────────────────────────────

def calculate_position_size(
    entry_price: float,
    stop_loss_price: float,
    num_multiplier: float,
    signal_strength: str,
    capital: float = 0.0,
    now: Optional[datetime] = None,
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """
    Returns position size in USDT notional value.

    - Full size on STRONG signals
    - Half size on WEAK signals
    - Zero on HOLD / NO_TRADE

    capital: live effective capital (account_balance * CAPITAL_PCT).
             Falls back to config.CAPITAL_USDT if not provided.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    effective = capital if capital > 0 else config.CAPITAL_USDT
    # Optional sizing overlay (dayboost + seasonality)
    mult = 1.0
    if getattr(config, "SIZE_OVERLAY_ENABLED", False):
        dom = int(now.day)
        dow = int(now.weekday())  # Mon=0 ... Sun=6
        base_dom = float(getattr(config, "DAY_MULTIPLIERS", {}).get(dom, 1.0))
        base_dow = float(getattr(config, "WEEKDAY_MULTIPLIERS", {}).get(dow, 1.0))
        if "BUY" in (signal_strength or ""):
            side_dom = float(getattr(config, "BUY_DAY_MULTIPLIERS", {}).get(dom, 1.0))
            side_dow = float(getattr(config, "BUY_WEEKDAY_MULTIPLIERS", {}).get(dow, 1.0))
        elif "SELL" in (signal_strength or ""):
            side_dom = float(getattr(config, "SELL_DAY_MULTIPLIERS", {}).get(dom, 1.0))
            side_dow = 1.0
        else:
            side_dom = 1.0
            side_dow = 1.0
        mult = base_dom * base_dow * side_dom * side_dow
        if mult <= 0:
            mult = 1.0

    if signal_strength in ("STRONG_BUY", "STRONG_SELL"):
        base_signal_factor = 1.0
    elif signal_strength in ("WEAK_BUY", "WEAK_SELL"):
        base_signal_factor = 0.72
    else:
        if return_details:
            return 0.0, {"skip_reason": "NO_ACTIONABLE_SIGNAL"}
        return 0.0

    raw_sl_distance = abs(entry_price - stop_loss_price)
    if entry_price <= 0 or raw_sl_distance <= 0:
        if return_details:
            return 0.0, {"skip_reason": "INVALID_ENTRY_OR_STOP"}
        return 0.0

    # Non-configurable dynamic sizing constants (intentionally hardcoded).
    BASE_RISK_PCT = 0.0125
    MIN_RISK_PCT = 0.005
    MAX_RISK_PCT = 0.020
    MIN_SL_PCT = 0.0035
    TARGET_SL_PCT = 0.012
    MAX_NOTIONAL_CAPITAL_MULT = 3.5
    MIN_MARGIN_USDT = 8.0

    sl_pct_raw = raw_sl_distance / entry_price
    sl_pct_used = max(sl_pct_raw, MIN_SL_PCT)
    used_sl_distance = entry_price * sl_pct_used

    # Wider stops get slightly less risk; tighter stops get slightly more risk.
    # Keeps risk adaptive while preventing overreaction at extremes.
    volatility_adjust = (TARGET_SL_PCT / sl_pct_used) ** 0.5
    volatility_adjust = min(max(volatility_adjust, 0.70), 1.35)

    # Numerology multiplier is damped for sizing so it cannot over-amplify risk.
    num_for_sizing = min(max(num_multiplier, 0.70), 1.30)
    numerology_adjust = 0.85 + (num_for_sizing - 0.70) * (0.30 / 0.60)  # maps 0.70..1.30 -> 0.85..1.15

    raw_risk_pct = BASE_RISK_PCT * base_signal_factor * mult * volatility_adjust * numerology_adjust
    risk_pct = min(max(raw_risk_pct, MIN_RISK_PCT), MAX_RISK_PCT)
    risk_amount = effective * risk_pct

    base_notional = (risk_amount / used_sl_distance) * entry_price  # USDT notional
    sized_notional = base_notional

    max_notional = effective * MAX_NOTIONAL_CAPITAL_MULT
    final_notional = min(sized_notional, max_notional) if max_notional > 0 else sized_notional

    lev = max(1, int(getattr(config, "LEVERAGE", 1)))
    margin_required = final_notional / lev if final_notional > 0 else 0.0

    details = {
        "risk_pct_used": risk_pct,
        "risk_amount_usdt": risk_amount,
        "base_signal_factor": base_signal_factor,
        "overlay_multiplier": mult,
        "sl_pct_raw": sl_pct_raw,
        "sl_pct_used": sl_pct_used,
        "min_sl_pct_applied": sl_pct_used > sl_pct_raw,
        "base_notional_usdt": base_notional,
        "num_multiplier": num_multiplier,
        "numerology_adjust": numerology_adjust,
        "volatility_adjust": volatility_adjust,
        "max_notional_cap": max_notional,
        "margin_required_usdt": margin_required,
        "skip_reason": None,
    }

    if margin_required < MIN_MARGIN_USDT:
        details["skip_reason"] = (
            f"MIN_MARGIN:{margin_required:.2f}<{MIN_MARGIN_USDT:.2f}"
        )
        if return_details:
            return 0.0, details
        return 0.0

    if return_details:
        return final_notional, details
    return final_notional


# ── Master Signal Generator ────────────────────────────────────────────────────

def generate_signal(
    asset_dna: dict,
    score_history: ScoreHistory,
    current_price: float,
    atr: float,
    today: Optional[date] = None,
    jd: Optional[float] = None,
    current_ema: float = 0.0,
    regime_price: float = 0.0,
    regime_ema: float = 0.0,
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
    final_action = apply_decisive_overlay(final_action, sky.get("astro_events"))

    # 7. Reliability filters (must match backtest logic)
    filter_reason = None

    # 7a. Nakshatra block — only active when NAKSHATRA_FILTER=true in .env
    if config.NAKSHATRA_FILTER and final_action not in ("HOLD", "NO_TRADE"):
        nk = sky["nakshatra"]
        if nk in config.TRADE_UNFAVORABLE_NAKSHATRAS:
            filter_reason = f"NAKSHATRA_BLOCK:{nk}"
            final_action = "NO_TRADE"

    # 7a2. Mercury / Saturn retrograde — block LONG only (shorts allowed; loss analysis)
    if filter_reason is None and "BUY" in (final_action or ""):
        rx_western = sky.get("retrograde_planets_western") or []
        if config.MERCURY_RX_BLOCK and "Mercury" in rx_western:
            filter_reason = "MERCURY_RX_BLOCK_LONG"
            final_action = "NO_TRADE"
        elif config.SATURN_RX_BLOCK and "Saturn" in rx_western:
            filter_reason = "SATURN_RX_BLOCK_LONG"
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

    # 7c. Short-confidence filter — reject low-quality SELL setups first.
    short_conf = 0.0
    if "SELL" in (final_action or ""):
        short_conf = short_confidence_score(vedic_score_adj, v_medium, v_slope)
        # Non-configurable floor to reduce short bias from marginal SELLs.
        if short_conf < 0.55:
            filter_reason = f"SHORT_CONFIDENCE:{short_conf:.2f}<0.55"
            final_action = "NO_TRADE"

    # 7d. Short-block filter — skip SHORT when astro conditions historically lose (backtest 2022–2025)
    if (
        filter_reason is None
        and "SELL" in (final_action or "")
        and getattr(config, "SHORT_BLOCK_NAKSHATRAS", None)
    ):
        nk = sky.get("nakshatra", "")
        astro = sky.get("astro_events") or {}
        block_reasons = []
        if nk in config.SHORT_BLOCK_NAKSHATRAS:
            block_reasons.append(f"nakshatra={nk}")
        if getattr(config, "SHORT_BLOCK_JUPITER_URANUS", True) and astro.get("jupiter_uranus_active"):
            block_reasons.append("Jupiter–Uranus")
        if getattr(config, "SHORT_BLOCK_NEW_MOON", True) and astro.get("new_moon_active"):
            block_reasons.append("new_moon")
        if getattr(config, "SHORT_BLOCK_MERCURY_RX", True) and astro.get("mercury_retrograde_active"):
            block_reasons.append("Mercury_RX")
        if block_reasons:
            filter_reason = "SHORT_BLOCK:" + ",".join(block_reasons)
            final_action = "NO_TRADE"

    # 7e. Macro regime filter (fixed: BTC EMA282 on 45m)
    if (
        filter_reason is None
        and final_action not in ("HOLD", "NO_TRADE")
        and regime_price > 0
        and regime_ema > 0
    ):
        if "BUY" in final_action and regime_price < regime_ema:
            filter_reason = (
                f"REGIME_BLOCK_LONG:price({regime_price:.2f})<EMA({regime_ema:.2f})"
            )
            final_action = "NO_TRADE"
        elif "SELL" in final_action and regime_price > regime_ema:
            filter_reason = (
                f"REGIME_BLOCK_SHORT:price({regime_price:.2f})>EMA({regime_ema:.2f})"
            )
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
    position_size, sizing = calculate_position_size(
        entry_price=current_price,
        stop_loss_price=stop_loss,
        num_multiplier=num_mult,
        signal_strength=final_action,
        capital=effective_capital,
        now=now,
        return_details=True,
    )

    signal = {
        "timestamp": now.isoformat(),
        "asset": asset_dna.get("name", "Unknown"),
        "symbol": asset_dna.get("symbol", ""),
        "status": "SIGNAL",
        "final_action": final_action,
        "filter_reason": filter_reason,
        "ema_value": round(current_ema, 2) if current_ema else None,
        "regime_price": round(regime_price, 2) if regime_price else None,
        "regime_ema": round(regime_ema, 2) if regime_ema else None,
        "regime_timeframe": "45m",
        "regime_period": 282,
        "effective_capital": round(effective_capital, 2),
        "size_overlay_enabled": bool(getattr(config, "SIZE_OVERLAY_ENABLED", False)),
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
        "short_confidence": round(short_conf, 4),
        "risk_pct_used": round(sizing.get("risk_pct_used", 0.0), 6),
        "risk_amount_usdt": round(sizing.get("risk_amount_usdt", 0.0), 4),
        "margin_required_usdt": round(sizing.get("margin_required_usdt", 0.0), 4),
        "sizing_skip_reason": sizing.get("skip_reason"),
        "sl_pct_raw": round(sizing.get("sl_pct_raw", 0.0), 6),
        "sl_pct_used": round(sizing.get("sl_pct_used", 0.0), 6),
        "min_sl_pct_applied": bool(sizing.get("min_sl_pct_applied", False)),
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
