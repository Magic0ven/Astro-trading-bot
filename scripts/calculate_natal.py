"""
Calculate natal chart positions for an asset's genesis datetime.

Usage:
    python scripts/calculate_natal.py              # All assets
    python scripts/calculate_natal.py --asset BTC  # Single asset
    python scripts/calculate_natal.py --asset ETH

Writes computed positions back into assets_dna.json.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import swisseph as swe
from dateutil.parser import parse as parse_date

import config
from core.astro_engine import datetime_to_jd, get_western_positions, get_vedic_positions


def compute_natal(genesis_dt_str: str) -> tuple[dict, dict]:
    """
    Compute Western and Vedic natal longitudes from a genesis datetime string.
    
    Returns:
        (natal_western, natal_vedic) — dicts of planet_name: longitude
    """
    dt = parse_date(genesis_dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    jd = datetime_to_jd(dt)

    western_positions = get_western_positions(jd)
    vedic_positions = get_vedic_positions(jd)

    natal_western = {p: round(data["longitude"], 4) for p, data in western_positions.items()}
    natal_vedic = {p: round(data["longitude"], 4) for p, data in vedic_positions.items()}

    return natal_western, natal_vedic


def main():
    parser = argparse.ArgumentParser(description="Calculate natal charts for asset genesis dates")
    parser.add_argument("--asset", type=str, default=None, help="Asset key (e.g. BTC, ETH). Omit for all.")
    args = parser.parse_args()

    dna_path = Path(__file__).parent.parent / "assets_dna.json"
    with open(dna_path) as f:
        all_assets = json.load(f)

    assets_to_process = [args.asset] if args.asset else [
        k for k in all_assets if not k.startswith("_")
    ]

    for key in assets_to_process:
        if key not in all_assets:
            print(f"Asset '{key}' not found in assets_dna.json — skipping.")
            continue

        dna = all_assets[key]
        genesis = dna.get("genesis_datetime")
        if not genesis:
            print(f"{key}: No genesis_datetime — skipping.")
            continue

        print(f"\nCalculating natal chart for {key} ({genesis})...")
        natal_w, natal_v = compute_natal(genesis)

        all_assets[key]["natal_western"] = natal_w
        all_assets[key]["natal_vedic"] = natal_v

        print(f"  Western natal:")
        for planet, lon in natal_w.items():
            print(f"    {planet:12s}: {lon:.4f}°")
        print(f"  Vedic natal (Lahiri):")
        for planet, lon in natal_v.items():
            print(f"    {planet:12s}: {lon:.4f}°")

    # Write back
    with open(dna_path, "w") as f:
        json.dump(all_assets, f, indent=2)

    print(f"\nassets_dna.json updated successfully.")


if __name__ == "__main__":
    main()
