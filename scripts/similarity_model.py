#!/usr/bin/env python3
"""
similarity_model.py
CLI interface for the V2 K-Nearest Neighbors (KNN) collector-similarity and price-maturation engine.
Matches modern Pokémon cards (2016+) to historically similar vintage cards (<2016) across
31 unit-range [0, 1] collector features (reconstructed rarity tier, secret depth, empirical
Bayesian species & artist prestige, biological evolution finality, trainer subtypes, and type affinities).
"""

import os
import sys
import argparse
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.inference import CardSimilarityEngine


def print_similarity_report(res: Dict[str, Any]):
    t = res["target_card"]
    p = res["projections"]

    print("\n" + "=" * 78)
    print(f"🎯 TARGET MODERN CARD: {t['name']} ({t['card_id']})")
    print(f"   Set: {t['set_name']} | Release: {int(t['release_year'] if t['release_year'] else 0)}")
    print(f"   Rarity: {t['rarity']} (Prestige Tier {t['rarity_tier']}/5)")
    print(f"   Secret Rare: {'Yes' if t['is_secret_rare'] else 'No'} | Holo: {'Yes' if t['has_holo'] else 'No'} | Artist: {t['illustrator']}")
    print(f"   Current Market Price: ${t['current_market_price']:.2f}" if t['current_market_price'] else "   Current Market Price: N/A")
    print("-" * 78)
    print("🏆 TOP SIMILAR VINTAGE COUNTERPARTS (<2016):")
    for i, n in enumerate(res["vintage_neighbors"], 1):
        yr = int(n['release_year']) if n['release_year'] else 'N/A'
        print(f"  {i}. [{n['similarity_score']:>5.1f}% match] {n['name']:<24} | {n['set_name']:<22} ({yr}) | {str(n['rarity']):<14} | ${n['market_price']:>7.2f}")
    print("-" * 78)
    print("📊 PROJECTED MATURE VINTAGE VALUE:")
    print(f"   Average Vintage Price:  ${p['vintage_mean_price']:.2f}")
    print(f"   Median Vintage Price:   ${p['vintage_median_price']:.2f}")
    print(f"   Vintage Price Range:    ${p['vintage_min_price']:.2f} – ${p['vintage_max_price']:.2f}")
    if p["upside_multiplier"]:
        print(f"   Estimated Maturation:   {p['upside_multiplier']}x current price")
    print("=" * 78 + "\n")


def print_top_gems_report(gems: List[Dict[str, Any]]):
    print("\n" + "=" * 84)
    print("💎 TOP MODERN CHASE CARDS WITH HIGHEST VINTAGE MATURATION VALUE")
    print("=" * 84)
    for i, row in enumerate(gems, 1):
        yr = row['release_year'] if row['release_year'] else 'N/A'
        print(f"  {i:>2}. {row['name']:<24} | {row['set_name']:<22} ({yr}) | {row['rarity']:<24}")
        print(f"      Current: ${row['current_price']:<7.2f} ──▶ Projected Vintage Median: ${row['vintage_median_price']:<7.2f} (Upside: {row['upside_multiplier']}x)")
    print("=" * 84 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pokémon Card Collector Similarity & Maturation Model (V2)")
    parser.add_argument("--card", type=str, default=None, help="Card name to query (e.g. Charizard ex)")
    parser.add_argument("--card_id", type=str, default=None, help="Specific card ID to query (e.g. sv03.5-199)")
    parser.add_argument("--top-gems", type=int, default=None, help="Scan modern sets and output top N highest-value gems")
    parser.add_argument("-k", type=int, default=5, help="Number of vintage neighbors to retrieve")
    args = parser.parse_args()

    engine = CardSimilarityEngine()

    if args.top_gems:
        gems = engine.get_top_gems(min_price=5.0, limit=args.top_gems, k=args.k)
        print_top_gems_report(gems)
    else:
        if args.card_id:
            target_id = args.card_id
        else:
            q_name = args.card if args.card else "Charizard ex"
            matches = engine.search_cards(query=q_name, limit=1)
            if not matches:
                print(f"No modern cards found matching '{q_name}'")
                return
            target_id = matches[0]["card_id"]

        result = engine.find_similar(target_id, k=args.k)
        if result:
            print_similarity_report(result)
        else:
            print(f"Card '{target_id}' not found.")


if __name__ == "__main__":
    main()
