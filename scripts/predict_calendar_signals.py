#!/usr/bin/env python3
"""
Add astro-bot buy/sell signals to each day in predictions_calendar.json.

Reads the calendar JSON (e.g. from frontend backend/data/predictions_calendar.json),
runs the signal engine (Western + Vedic + numerology + score history) for each day
at noon UTC, and writes back action, western_signal, vedic_signal, scores, etc.

Usage:
  python scripts/predict_calendar_signals.py [--calendar path/to/predictions_calendar.json] [--output path]
  If --output is omitted, updates the calendar file in place.
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dateutil.parser import parse as parse_date

import config
from core.astro_engine import (
    compute_composite_score,
    datetime_to_jd,
    get_sky_state,
)
from core.numerology import full_numerology_report, universal_day_number
from core.score_history import ScoreHistory
from core.signal_engine import (
    _system_signal,
    apply_decisive_overlay,
    dual_system_gate,
)


def load_assets_dna() -> dict:
    root = Path(__file__).resolve().parent.parent
    path = root / "assets_dna.json"
    if not path.exists():
        raise FileNotFoundError(f"assets_dna.json not found at {path}")
    with open(path) as f:
        return json.load(f)


def run():
    parser = argparse.ArgumentParser(description="Add astro signals to predictions calendar")
    parser.add_argument(
        "--calendar",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "astro-bot-frontend" / "backend" / "data" / "predictions_calendar.json",
        help="Path to predictions_calendar.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: update calendar in place)",
    )
    args = parser.parse_args()

    calendar_path = args.calendar
    out_path = args.output or calendar_path

    if not calendar_path.exists():
        print(f"Error: calendar file not found: {calendar_path}", file=sys.stderr)
        sys.exit(1)

    with open(calendar_path) as f:
        data = json.load(f)

    asset_key = (data.get("asset") or "BTC").strip().upper()
    days = data.get("days") or []
    if not days:
        print("No days in calendar.", file=sys.stderr)
        sys.exit(0)

    assets = load_assets_dna()
    asset_dna = assets.get(asset_key)
    if not asset_dna:
        print(f"Error: asset '{asset_key}' not in assets_dna.json", file=sys.stderr)
        sys.exit(1)

    natal_western = asset_dna.get("natal_western") or {}
    natal_vedic = asset_dna.get("natal_vedic") or {}
    dasha_weights = asset_dna.get("dasha_weights")
    genesis_date = parse_date(asset_dna["genesis_datetime"]).date()
    symbol = asset_dna.get("symbol") or asset_key
    name = asset_dna.get("name", asset_key)

    window = getattr(config, "SCORE_HISTORY_WINDOW", 5)
    score_history = ScoreHistory(window=window, persist_path=None)

    for i, day_entry in enumerate(days):
        day_str = day_entry.get("date")
        if not day_str:
            continue
        try:
            day_date = date.fromisoformat(day_str)
        except ValueError:
            continue

        noon_utc = datetime(day_date.year, day_date.month, day_date.day, 12, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_jd(noon_utc)

        sky = get_sky_state(jd)
        num_report = full_numerology_report(
            asset_key,
            genesis_date,
            day_date,
        )
        num_mult = num_report["multiplier"]
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
        western_score_adj = western_score * num_mult
        vedic_score_adj = vedic_score * num_mult

        score_history.push(western_score_adj, vedic_score_adj)

        day_entry["nakshatra"] = sky["nakshatra"]
        day_entry["retrograde_western"] = sky["retrograde_planets_western"]
        day_entry["retrograde_vedic"] = sky["retrograde_planets_vedic"]
        day_entry["western_score"] = round(western_score_adj, 4)
        day_entry["vedic_score"] = round(vedic_score_adj, 4)
        day_entry["numerology_label"] = num_report["label"]
        day_entry["numerology_mult"] = num_report["multiplier"]
        day_entry["udn"] = num_report["universal_day_number"]
        day_entry["resonance"] = num_report.get("resonance_match", False)

        if not score_history.is_ready():
            day_entry["action"] = "COLLECTING_DATA"
            day_entry["western_signal"] = ""
            day_entry["vedic_signal"] = ""
            day_entry["western_medium"] = None
            day_entry["vedic_medium"] = None
            day_entry["western_slope"] = None
            day_entry["vedic_slope"] = None
            continue

        hist = score_history.summary()
        w_medium = hist["western"]["medium"]
        w_slope = hist["western"]["slope"]
        v_medium = hist["vedic"]["medium"]
        v_slope = hist["vedic"]["slope"]

        western_signal = _system_signal(western_score_adj, w_medium, w_slope)
        vedic_signal = _system_signal(vedic_score_adj, v_medium, v_slope)
        final_action = dual_system_gate(western_signal, vedic_signal)
        final_action = apply_decisive_overlay(final_action, sky.get("astro_events"))

        if config.NAKSHATRA_FILTER and final_action not in ("HOLD", "NO_TRADE"):
            nk = sky["nakshatra"]
            if nk in config.TRADE_UNFAVORABLE_NAKSHATRAS:
                final_action = "NO_TRADE"
        if "BUY" in (final_action or ""):
            rx_w = sky.get("retrograde_planets_western") or []
            if getattr(config, "MERCURY_RX_BLOCK", True) and "Mercury" in rx_w:
                final_action = "NO_TRADE"
            elif getattr(config, "SATURN_RX_BLOCK", True) and "Saturn" in rx_w:
                final_action = "NO_TRADE"

        day_entry["action"] = final_action
        day_entry["western_signal"] = western_signal
        day_entry["vedic_signal"] = vedic_signal
        day_entry["western_medium"] = round(w_medium, 4)
        day_entry["vedic_medium"] = round(v_medium, 4)
        day_entry["western_slope"] = round(w_slope, 4)
        day_entry["vedic_slope"] = round(v_slope, 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {len(days)} days with astro signals. Written to {out_path}")


if __name__ == "__main__":
    run()
