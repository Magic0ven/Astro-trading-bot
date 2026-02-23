"""
Astrology Engine — computes live planet positions in both Western (Tropical)
and Vedic (Sidereal/Lahiri) zodiac systems using pyswisseph.

Outputs:
  - Ecliptic longitudes for each planet
  - Retrograde status (+1 direct, -1 retrograde)
  - Moon velocity vs average
  - Composite aspect score vs asset natal chart
"""
import math
from datetime import datetime, timezone
from typing import Optional

import swisseph as swe
from loguru import logger

import config


def _init_ephe():
    swe.set_ephe_path(config.EPHE_PATH)


_init_ephe()


# ── Julian Date helpers ────────────────────────────────────────────────────────

def datetime_to_jd(dt: datetime) -> float:
    """Convert a UTC-aware datetime to Julian Day Number."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    dt_utc = dt.astimezone(timezone.utc)
    return swe.julday(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    )


def now_jd() -> float:
    return datetime_to_jd(datetime.now(timezone.utc))


# ── Planet Position Fetchers ───────────────────────────────────────────────────

def _calc_planet(jd: float, planet_id: int, flags: int) -> dict:
    """Low-level: fetch position and speed for one planet."""
    result, ret_flag = swe.calc_ut(jd, planet_id, flags)
    longitude = result[0]
    speed = result[3]           # degrees/day; negative = retrograde
    return {
        "longitude": longitude % 360,
        "speed": speed,
        "retrograde": -1 if speed < 0 else 1,
    }


def get_western_positions(jd: Optional[float] = None) -> dict:
    """
    Return Tropical (Western) positions for all tracked planets.
    
    Returns:
        {
          "Sun": {"longitude": 123.45, "speed": 0.985, "retrograde": 1},
          "Moon": {...},
          ...
        }
    """
    if jd is None:
        jd = now_jd()

    flags = swe.FLG_SWIEPH | swe.FLG_SPEED
    positions = {}

    for name, planet_id in config.PLANETS_WESTERN.items():
        try:
            positions[name] = _calc_planet(jd, planet_id, flags)
        except Exception as e:
            logger.error(f"Western position error for {name}: {e}")
            positions[name] = {"longitude": 0.0, "speed": 0.0, "retrograde": 1}

    return positions


def get_vedic_positions(jd: Optional[float] = None) -> dict:
    """
    Return Sidereal (Vedic/Lahiri) positions including Rahu & Ketu.
    
    Returns:
        {
          "Sun": {"longitude": 99.38, "speed": 0.985, "retrograde": 1},
          ...
          "Rahu": {"longitude": 285.0, "speed": -0.05, "retrograde": -1},
          "Ketu": {"longitude": 105.0, "speed": -0.05, "retrograde": -1},
        }
    """
    if jd is None:
        jd = now_jd()

    swe.set_sid_mode(config.VEDIC_AYANAMSA)
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_SIDEREAL
    positions = {}

    for name, planet_id in config.PLANETS_VEDIC.items():
        try:
            positions[name] = _calc_planet(jd, planet_id, flags)
        except Exception as e:
            logger.error(f"Vedic position error for {name}: {e}")
            positions[name] = {"longitude": 0.0, "speed": 0.0, "retrograde": 1}

    # Ketu = Rahu + 180°
    if "Rahu" in positions:
        positions["Ketu"] = {
            "longitude": (positions["Rahu"]["longitude"] + 180.0) % 360,
            "speed": positions["Rahu"]["speed"],
            "retrograde": positions["Rahu"]["retrograde"],
        }

    swe.set_sid_mode(swe.SIDM_FAGAN_BRADLEY)  # Reset to default
    return positions


# ── Moon Velocity Check ────────────────────────────────────────────────────────

MOON_AVERAGE_SPEED_DEG_DAY = 13.176

def is_moon_fast(positions: dict) -> bool:
    """True if Moon is moving faster than average (bullish lunar energy)."""
    moon_speed = abs(positions.get("Moon", {}).get("speed", 0))
    return moon_speed > MOON_AVERAGE_SPEED_DEG_DAY


# ── Aspect Score Calculator ────────────────────────────────────────────────────

def _aspect_force(live_lon: float, natal_lon: float, retrograde: int) -> float:
    """
    Compute the 'force' of one aspect between a live and natal planet.
    
    Formula: retrograde × cos(angle_in_radians)
    
    angle = |live - natal| mod 360
    cos(0°)   = +1.0  (Conjunction — unity)
    cos(90°)  =  0.0  (Square — neutral/tension)
    cos(180°) = -1.0  (Opposition — maximum tension)
    """
    angle = abs(live_lon - natal_lon) % 360
    if angle > 180:
        angle = 360 - angle       # Normalize to [0, 180]
    radians = math.radians(angle)
    return retrograde * math.cos(radians)


def compute_composite_score(
    live_positions: dict,
    natal_positions: dict,
    dasha_weights: Optional[dict] = None,
    nakshatra_mult: float = 1.0,
) -> float:
    """
    Compute the total composite score: Σ cos(live_i − natal_j) × retro_i
    
    Args:
        live_positions: Dict of planet → {longitude, speed, retrograde}
        natal_positions: Dict of planet → longitude (float, degrees)
        dasha_weights: Optional per-planet weight multiplier (Vedic Dasha system)
        nakshatra_mult: Nakshatra bonus/penalty for the Moon's current position
    
    Returns:
        Float composite score (can be positive or negative)
    """
    total = 0.0
    pair_count = 0

    for live_name, live_data in live_positions.items():
        if live_name not in natal_positions:
            continue

        natal_lon = natal_positions[live_name]
        if natal_lon is None:
            continue

        live_lon = live_data["longitude"]
        retrograde = live_data["retrograde"]
        force = _aspect_force(live_lon, natal_lon, retrograde)

        # Cross-planet pairs: each live planet vs ALL natal planets
        for natal_name, n_lon in natal_positions.items():
            if n_lon is None:
                continue
            cross_force = _aspect_force(live_lon, n_lon, retrograde)
            weight = 1.0
            if dasha_weights and live_name in dasha_weights:
                weight = dasha_weights[live_name]
            total += cross_force * weight
            pair_count += 1

        _ = force  # Same-planet force is included in cross-planet loop

    # Apply Nakshatra bonus to the Moon's contribution
    # (already included above, nakshatra_mult adjusts the Moon row retroactively)
    # Simple approach: scale full score by nakshatra mult (minor effect)
    if pair_count > 0:
        total *= nakshatra_mult

    logger.debug(f"Composite score: {total:.4f} over {pair_count} pairs")
    return total


# ── Retrograde Summary ─────────────────────────────────────────────────────────

def get_retrograde_planets(positions: dict) -> list[str]:
    """Return list of planet names currently in retrograde."""
    return [name for name, data in positions.items() if data["retrograde"] == -1]


# ── Full Sky State ─────────────────────────────────────────────────────────────

def get_sky_state(jd: Optional[float] = None) -> dict:
    """
    Single call to get the complete sky state for both systems.
    
    Returns:
        {
          "jd": float,
          "western": {planet: {longitude, speed, retrograde}, ...},
          "vedic":   {planet: {longitude, speed, retrograde}, ...},
          "moon_fast": bool,
          "retrograde_planets_western": [...],
          "retrograde_planets_vedic": [...],
          "nakshatra": str,
          "nakshatra_multiplier": float,
        }
    """
    if jd is None:
        jd = now_jd()

    western = get_western_positions(jd)
    vedic = get_vedic_positions(jd)

    moon_vedic_lon = vedic.get("Moon", {}).get("longitude", 0.0)
    nakshatra = config.get_nakshatra(moon_vedic_lon)
    nk_mult = config.nakshatra_multiplier(moon_vedic_lon)

    return {
        "jd": jd,
        "western": western,
        "vedic": vedic,
        "moon_fast": is_moon_fast(western),
        "retrograde_planets_western": get_retrograde_planets(western),
        "retrograde_planets_vedic": get_retrograde_planets(vedic),
        "nakshatra": nakshatra,
        "nakshatra_multiplier": nk_mult,
    }
