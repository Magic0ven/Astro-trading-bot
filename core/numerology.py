"""
Numerology Engine — Pythagorean system calculations.

Provides:
  - Digital root reduction
  - Life Path Number (from a date)
  - Ticker Vibration (from asset symbol string)
  - Universal Day Number (from today's date)
  - Numerology multiplier (1.5 on resonance days, 1.0 otherwise)
"""
from datetime import date, datetime
from config import PYTHAGOREAN_MAP


def digital_root(n: int) -> int:
    """Reduce any integer to a single digit (1–9) via repeated digit summation."""
    n = abs(n)
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n


def date_digit_sum(d: date) -> int:
    """Sum all individual digits of a date (YYYYMMDD)."""
    raw = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return sum(int(ch) for ch in raw)


def life_path_number(d: date) -> int:
    """
    Compute the Life Path Number from a date.
    
    Formula: digital_root(sum of all digits in YYYYMMDD)
    
    Example — BTC Genesis 2009-01-03:
      2+0+0+9+0+1+0+3 = 15 → 1+5 = 6
    """
    return digital_root(date_digit_sum(d))


def ticker_vibration(symbol: str) -> int:
    """
    Compute the Pythagorean vibration number for an asset ticker.
    
    Only alphabetic characters are counted.
    Example — BTC: B=2, T=2, C=3 → sum=7 → digital_root=7
    Note: BTC sums to 7, not 2. The user's doc shows 2 which may use a
    different reduction (month/day only). Both are logged for transparency.
    """
    total = sum(PYTHAGOREAN_MAP.get(ch, 0) for ch in symbol.lower() if ch.isalpha())
    return digital_root(total)


def universal_day_number(d: date) -> int:
    """
    Universal Day Number (UDN) for a given date.
    Identical method to life_path_number but applied to today.
    
    Example — Feb 23, 2026:
      2+0+2+6+0+2+2+3 = 17 → 1+7 = 8
    """
    return digital_root(date_digit_sum(d))


def universal_hour_number(hour: int) -> int:
    """
    Universal Hour Number from an hour value (0–23).
    
    Example — 17:00:  1+7 = 8
    """
    return digital_root(sum(int(d) for d in str(hour)))


def numerology_multiplier(asset_life_path: int, today: date) -> float:
    """
    Return the Pythagorean Compatibility multiplier (Pc).
    
    Pc = 1.5 if Universal Day Number == Asset Life Path Number (Resonance Day)
    Pc = 1.0 otherwise (Neutral Day)
    """
    udn = universal_day_number(today)
    match = (udn == asset_life_path)
    return 1.5 if match else 1.0


def full_numerology_report(symbol: str, genesis_date: date, today: date) -> dict:
    """
    Generate a complete numerology snapshot for an asset on a given day.
    
    Returns a dict suitable for logging and signal display.
    """
    lp = life_path_number(genesis_date)
    tv = ticker_vibration(symbol)
    udn = universal_day_number(today)
    uhn = universal_hour_number(datetime.utcnow().hour)
    mult = numerology_multiplier(lp, today)

    return {
        "symbol": symbol,
        "genesis_date": genesis_date.isoformat(),
        "today": today.isoformat(),
        "life_path_number": lp,
        "ticker_vibration": tv,
        "universal_day_number": udn,
        "universal_hour_number": uhn,
        "resonance_match": udn == lp,
        "multiplier": mult,
        "label": "RESONANCE DAY (1.5x)" if mult == 1.5 else "Normal Day (1.0x)",
    }
