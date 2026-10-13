#!/usr/bin/env python3
"""
ingest_to_tabular.py
Converts raw JSON datasets (data/cards.json, data/pricing/latest.json, data/sets.json)
into a structured, flattened tabular dataset (Parquet and CSV) ready for feature engineering
and similarity/KNN modeling.
"""

import os
import re
import ssl
import json
import urllib.request
import concurrent.futures
from typing import Dict, Any, Optional, Tuple
import pandas as pd


def fetch_set_metadata(sets_file: str, cache_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches and caches set release dates and series names from the TCGdex API.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached set metadata from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Fetching set release dates from TCGdex API...")
    with open(sets_file, "r", encoding="utf-8") as f:
        sets = json.load(f)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def fetch_single(s: dict) -> Tuple[str, dict]:
        sid = s["id"]
        url = f"https://api.tcgdex.net/v2/en/sets/{sid}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                release_date = data.get("releaseDate")
                release_year = int(release_date[:4]) if release_date else None
                serie = data.get("serie", {}).get("name") if data.get("serie") else None
                return sid, {
                    "id": sid,
                    "name": data.get("name", s.get("name")),
                    "releaseDate": release_date,
                    "releaseYear": release_year,
                    "serie": serie,
                    "officialCardCount": data.get("cardCount", {}).get("official"),
                    "totalCardCount": data.get("cardCount", {}).get("total")
                }
        except Exception as e:
            print(f"Warning: Failed to fetch metadata for set '{sid}': {e}")
            return sid, {
                "id": sid,
                "name": s.get("name"),
                "releaseDate": None,
                "releaseYear": None,
                "serie": None,
                "officialCardCount": s.get("cardCount", {}).get("official"),
                "totalCardCount": s.get("cardCount", {}).get("total")
            }

    metadata = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for sid, set_info in executor.map(fetch_single, sets):
            metadata[sid] = set_info

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved set metadata cache to {cache_file} ({len(metadata)} sets).")
    return metadata


def parse_damage_value(damage_val: Any) -> Optional[float]:
    """
    Parses damage strings like '40+', '10x', '120-', '30' into numeric values.
    Returns None if no damage or non-numeric.
    """
    if damage_val is None:
        return None
    if isinstance(damage_val, (int, float)):
        return float(damage_val)
    if isinstance(damage_val, str):
        # Extract digits
        digits = re.findall(r"\d+", damage_val)
        if digits:
            try:
                return float(digits[0])
            except ValueError:
                return None
    return None


def extract_pricing(pricing_entry: Optional[dict]) -> dict:
    """
    Extracts preferred USD and EUR prices from a card's pricing object.
    Prioritizes Holofoil and Normal market prices.
    """
    res = {
        "tcgplayer_market_price": None,
        "tcgplayer_low_price": None,
        "tcgplayer_mid_price": None,
        "tcgplayer_high_price": None,
        "tcgplayer_variant_used": None,
        "cardmarket_avg_eur": None,
        "cardmarket_trend_eur": None,
    }

    if not pricing_entry:
        return res

    p = pricing_entry.get("pricing", {}) or {}

    # 1. TCGPlayer (USD)
    tcp = p.get("tcgplayer")
    if isinstance(tcp, dict):
        # Priority order for finishes to determine primary market price
        priority = [
            "holofoil",
            "normal",
            "reverse-holofoil",
            "1st-edition-holofoil",
            "1st-edition",
            "unlimited-holofoil",
            "unlimited",
        ]
        for finish in priority:
            sub = tcp.get(finish)
            if isinstance(sub, dict) and sub.get("marketPrice") is not None:
                res["tcgplayer_market_price"] = float(sub["marketPrice"])
                res["tcgplayer_low_price"] = float(sub.get("lowPrice")) if sub.get("lowPrice") is not None else None
                res["tcgplayer_mid_price"] = float(sub.get("midPrice")) if sub.get("midPrice") is not None else None
                res["tcgplayer_high_price"] = float(sub.get("highPrice")) if sub.get("highPrice") is not None else None
                res["tcgplayer_variant_used"] = finish
                break

    # 2. Cardmarket (EUR)
    cm = p.get("cardmarket")
    if isinstance(cm, dict):
        if cm.get("avg") is not None:
            res["cardmarket_avg_eur"] = float(cm["avg"])
        if cm.get("trend") is not None:
            res["cardmarket_trend_eur"] = float(cm["trend"])

    return res


def flatten_card(
    card: dict,
    pricing_entry: Optional[dict],
    set_meta: Optional[dict]
) -> dict:
    """
    Flattens a single card dictionary into a tabular row dictionary.
    """
    cid = card.get("id")
    name = card.get("name")
    category = card.get("category", "Unknown")
    set_obj = card.get("set", {}) or {}
    sid = set_obj.get("id")

    # Release / Era
    rel_date = set_meta.get("releaseDate") if set_meta else None
    rel_year = set_meta.get("releaseYear") if set_meta else None
    series = set_meta.get("serie") if set_meta else None

    # Variants
    variants = card.get("variants", {}) or {}
    has_normal = bool(variants.get("normal"))
    has_holo = bool(variants.get("holo"))
    has_reverse = bool(variants.get("reverse"))
    has_first_edition = bool(variants.get("firstEdition"))

    # Types
    types = card.get("types") or []
    primary_type = types[0] if len(types) > 0 else None
    secondary_type = types[1] if len(types) > 1 else None

    # HP & Retreat
    hp = card.get("hp")
    hp = float(hp) if hp is not None else None
    retreat = card.get("retreat")
    retreat = float(retreat) if retreat is not None else None

    # Attacks
    attacks = card.get("attacks") or []
    attack_count = len(attacks)
    damages = [parse_damage_value(a.get("damage")) for a in attacks]
    valid_damages = [d for d in damages if d is not None]
    max_attack_damage = max(valid_damages) if valid_damages else None
    min_attack_damage = min(valid_damages) if valid_damages else None

    total_attack_energy = 0
    for a in attacks:
        cost = a.get("cost") or []
        total_attack_energy += len(cost)

    # Abilities
    abilities = card.get("abilities") or []
    ability_count = len(abilities)

    # Weaknesses & Resistances
    weaknesses = card.get("weaknesses") or []
    weakness_type = weaknesses[0].get("type") if (weaknesses and isinstance(weaknesses[0], dict)) else None
    resistances = card.get("resistances") or []
    resistance_type = resistances[0].get("type") if (resistances and isinstance(resistances[0], dict)) else None

    # Dex ID
    dex_ids = card.get("dexId") or []
    dex_id = int(dex_ids[0]) if (dex_ids and isinstance(dex_ids[0], (int, float))) else None

    # Legalities
    legal = card.get("legal", {}) or {}
    legal_standard = bool(legal.get("standard"))
    legal_expanded = bool(legal.get("expanded"))

    # Pricing
    price_info = extract_pricing(pricing_entry)

    return {
        # Identifiers
        "card_id": cid,
        "name": name,
        "local_id": card.get("localId"),
        "set_id": sid,
        "set_name": set_obj.get("name"),
        "series": series,
        "release_date": rel_date,
        "release_year": rel_year,
        "is_older_era": (rel_year < 2016) if rel_year is not None else None,

        # Classification / Categorical
        "category": category,
        "rarity": card.get("rarity"),
        "stage": card.get("stage"),
        "primary_type": primary_type,
        "secondary_type": secondary_type,
        "types_count": len(types),
        "dex_id": dex_id,
        "evolve_from": card.get("evolveFrom"),
        "illustrator": card.get("illustrator"),

        # Numeric Stats
        "hp": hp,
        "retreat": retreat,
        "attack_count": attack_count,
        "max_attack_damage": max_attack_damage,
        "min_attack_damage": min_attack_damage,
        "total_attack_energy_cost": total_attack_energy,
        "ability_count": ability_count,
        "weakness_type": weakness_type,
        "resistance_type": resistance_type,

        # Printings / Collectibility
        "has_normal": has_normal,
        "has_holo": has_holo,
        "has_reverse": has_reverse,
        "has_first_edition": has_first_edition,
        "legal_standard": legal_standard,
        "legal_expanded": legal_expanded,

        # Pricing Labels
        "market_price": price_info["tcgplayer_market_price"],
        "low_price": price_info["tcgplayer_low_price"],
        "mid_price": price_info["tcgplayer_mid_price"],
        "high_price": price_info["tcgplayer_high_price"],
        "pricing_variant": price_info["tcgplayer_variant_used"],
        "cardmarket_avg_eur": price_info["cardmarket_avg_eur"],
        "cardmarket_trend_eur": price_info["cardmarket_trend_eur"],
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sets_file = os.path.join(base_dir, "data", "sets.json")
    cards_file = os.path.join(base_dir, "data", "cards.json")
    pricing_file = os.path.join(base_dir, "data", "pricing", "latest.json")
    sets_cache = os.path.join(base_dir, "data", "sets_metadata.json")

    parquet_out = os.path.join(base_dir, "data", "cards_tabular.parquet")
    csv_out = os.path.join(base_dir, "data", "cards_tabular.csv")

    print(f"Loading data from {base_dir}/data ...")
    set_metadata = fetch_set_metadata(sets_file, sets_cache)

    print("Loading cards.json...")
    with open(cards_file, "r", encoding="utf-8") as f:
        cards_raw = json.load(f)

    print("Loading pricing/latest.json...")
    with open(pricing_file, "r", encoding="utf-8") as f:
        pricing_raw = json.load(f)

    print("Flattening card records...")
    rows = []
    for sid, set_cards in cards_raw.items():
        s_meta = set_metadata.get(sid)
        for cid, card_obj in set_cards.items():
            p_entry = pricing_raw.get(cid)
            rows.append(flatten_card(card_obj, p_entry, s_meta))

    df = pd.DataFrame(rows)
    print(f"\nConstructed DataFrame: {df.shape[0]} rows, {df.shape[1]} columns.")

    print(f"Saving to Parquet: {parquet_out} ...")
    df.to_parquet(parquet_out, index=False)

    print(f"Saving to CSV: {csv_out} ...")
    df.to_csv(csv_out, index=False)

    print("\n=== Dataset Summary ===")
    print(f"Total cards: {len(df)}")
    print(f"Pokemon cards: {(df['category'] == 'Pokemon').sum()}")
    print(f"Older era cards (< 2016): {(df['is_older_era'] == True).sum()}")
    print(f"Newer era cards (>= 2016): {(df['is_older_era'] == False).sum()}")
    print(f"Cards with market_price: {df['market_price'].notnull().sum()} ({df['market_price'].notnull().mean():.1%})")
    print(f"Average market price: ${df['market_price'].mean():.2f}")
    print(f"Median market price:  ${df['market_price'].median():.2f}")
    print("Columns:", list(df.columns))
    print("\nConversion successfully completed!")


if __name__ == "__main__":
    main()
