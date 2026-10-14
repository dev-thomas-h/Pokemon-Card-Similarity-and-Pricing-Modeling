#!/usr/bin/env python3
"""
similarity_model.py
A K-Nearest Neighbors (KNN) collector-similarity model that matches modern Pokémon cards (2016+)
to historically similar vintage cards (<2016) based on collectibility archetypes (rarity tier,
secret rare status, holo finishes, subject prestige, artist renown) rather than gameplay stats.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Official National Pokédex IDs for canonical subject archetypes
STARTERS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9,          # Gen 1 (Bulbasaur, Charmander, Squirtle lines)
    152, 153, 154, 155, 156, 157, 158, 159, 160, # Gen 2 (Chikorita, Cyndaquil, Totodile)
    252, 253, 254, 255, 256, 257, 258, 259, 260, # Gen 3 (Treecko, Torchic, Mudkip)
    387, 388, 389, 390, 391, 392, 393, 394, 395, # Gen 4 (Turtwig, Chimchar, Piplup)
    495, 496, 497, 498, 499, 500, 501, 502, 503, # Gen 5 (Snivy, Tepig, Oshawott)
    650, 651, 652, 653, 654, 655, 656, 657, 658, # Gen 6 (Chespin, Fennekin, Froakie)
    722, 723, 724, 725, 726, 727, 728, 729, 730, # Gen 7 (Rowlet, Litten, Popplio)
    810, 811, 812, 813, 814, 815, 816, 817, 818, # Gen 8 (Grookey, Scorbunny, Sobble)
    906, 907, 908, 909, 910, 911, 912, 913, 914  # Gen 9 (Sprigatito, Fuecoco, Quaxly)
}

LEGENDARIES = {
    144, 145, 146, 150, 151,             # Birds, Mewtwo, Mew
    243, 244, 245, 249, 250, 251,        # Beasts, Lugia, Ho-Oh, Celebi
    377, 378, 379, 380, 381, 382, 383, 384, 385, 386, # Regis, Eon Duo, Weather Trio, Jirachi, Deoxys
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, # Sinnoh Legends & Mythicals
    638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, # Swords of Justice, Forces of Nature, Tao Trio
    716, 717, 718, 719, 720, 721,        # Aura Trio, Mythicals
    785, 786, 787, 788, 789, 790, 791, 792, 800, 801, 802, # Tapus, Light Trio, Necrozma
    888, 889, 890, 891, 892, 894, 895, 896, 897, 898, # Hero Duo, Eternatus, Urshifu, Calyrex
    1001, 1002, 1003, 1004, 1007, 1008, 1014, 1015, 1016, 1017, 1024, 1025 # Treasures of Ruin, Paradox Legends, Terapagos
}

FAN_FAVORITES = {
    25, 26,                              # Pikachu, Raichu
    133, 134, 135, 136, 196, 197, 470, 471, 700, # Eevee and all 8 Eeveelutions
    92, 93, 94,                          # Gastly, Haunter, Gengar
    147, 148, 149,                       # Dratini, Dragonair, Dragonite
    246, 247, 248,                       # Larvitar, Pupitar, Tyranitar
    443, 444, 445,                       # Gible, Gabite, Garchomp
    447, 448                             # Riolu, Lucario
}

RENOWNED_ARTISTS = {
    "Ken Sugimori", "Mitsuhiro Arita", "Naoki Saito", "Kouki Saitou",
    "Kagemaru Himeno", "Atsuko Nishida", "Masakazu Fukuda", "Tomokazu Komiya"
}

RARITY_TIER_MAP = {
    # Tier 5: Top Grail / Masterpiece / Chase
    "Secret Rare": 5,
    "Special illustration rare": 5,
    "Hyper rare": 5,
    "LEGEND": 5,
    "Black White Rare": 5,
    "Mega Hyper Rare": 5,

    # Tier 4: High Chase / Alternate Arts / Premium Subtypes
    "Illustration rare": 4,
    "Rare Holo LV.X": 4,
    "Rare PRIME": 4,
    "Ultra Rare": 4,
    "Shiny rare": 4,
    "Shiny Ultra Rare": 4,
    "Amazing Rare": 4,

    # Tier 3: Holo Rares & Modern V/ex mechanics
    "Double rare": 3,
    "Rare Holo": 3,
    "Holo Rare": 3,
    "Holo Rare V": 3,
    "Holo Rare VMAX": 3,
    "Holo Rare VSTAR": 3,
    "Radiant Rare": 3,
    "ACE SPEC Rare": 3,

    # Tier 2: Regular Rares & Standard Promos
    "Rare": 2,
    "Promo": 2,

    # Tier 1: Bulk
    "Uncommon": 1,
    "Common": 1
}

STAGE_TIER_MAP = {
    "Basic": 1,
    "Stage1": 2,
    "Stage2": 3,
    "VMAX": 4,
    "VSTAR": 4,
    "MEGA": 4,
    "LEVEL-UP": 4,
    "BREAK": 4,
    "V-UNION": 4,
    "RESTORED": 2
}


class PokemonCardSimilarityModel:
    def __init__(self, data_path: str, sets_meta_path: str):
        self.data_path = data_path
        self.sets_meta_path = sets_meta_path
        self.df = None
        self.sets_meta = {}
        self.train_df = None
        self.query_df = None
        self.knn = None
        self.scaler = StandardScaler()

        # Collector feature weights to emphasize key price-drivers
        self.feature_weights = {
            "rarity_tier": 3.0,
            "is_secret_rare": 2.5,
            "has_holo": 1.5,
            "has_first_edition": 1.0,
            "has_reverse": 0.5,
            "arch_legend": 2.5,
            "arch_fav": 2.0,
            "arch_starter": 1.5,
            "arch_regular": 1.0,
            "arch_trainer": 1.5,
            "arch_energy": 1.0,
            "stage_tier": 1.0,
            "is_renowned_artist": 1.0
        }
        self.feature_cols = list(self.feature_weights.keys())

    def load_and_prepare_data(self):
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)

        with open(self.sets_meta_path, "r", encoding="utf-8") as f:
            self.sets_meta = json.load(f)

        print("Engineering collectibility features...")
        self._engineer_features()

        # Older era cards (< 2016) with valid market prices form the training baseline
        train_mask = (self.df["is_older_era"] == True) & (self.df["market_price"].notnull()) & (self.df["market_price"] > 0)
        self.train_df = self.df[train_mask].copy().reset_index(drop=True)

        # Newer era cards (>= 2016) form the query pool
        self.query_df = self.df[self.df["is_older_era"] == False].copy().reset_index(drop=True)

        print(f"Training pool (Older vintage cards with prices): {len(self.train_df):,} cards")
        print(f"Query pool (Modern cards): {len(self.query_df):,} cards")

    def _engineer_features(self):
        # 1. Rarity Tier
        self.df["rarity_tier"] = self.df["rarity"].map(RARITY_TIER_MAP).fillna(1).astype(float)

        # 2. Secret Rare flag (local_id > official set card count)
        def check_secret(row):
            meta = self.sets_meta.get(row["set_id"])
            if not meta or not meta.get("officialCardCount"):
                return 0
            try:
                lid = int(row["local_id"])
                return 1 if lid > meta["officialCardCount"] else 0
            except (ValueError, TypeError):
                return 0
        self.df["is_secret_rare"] = self.df.apply(check_secret, axis=1).astype(float)

        # 3. Finishes
        self.df["has_holo"] = self.df["has_holo"].astype(float)
        self.df["has_first_edition"] = self.df["has_first_edition"].astype(float)
        self.df["has_reverse"] = self.df["has_reverse"].astype(float)

        # 4. Subject Archetype
        def get_archetype_flags(row):
            cat = row["category"]
            did = row["dex_id"]
            is_legend = 1.0 if (cat == "Pokemon" and did in LEGENDARIES) else 0.0
            is_fav = 1.0 if (cat == "Pokemon" and did in FAN_FAVORITES) else 0.0
            is_starter = 1.0 if (cat == "Pokemon" and did in STARTERS) else 0.0
            is_reg = 1.0 if (cat == "Pokemon" and not is_legend and not is_fav and not is_starter) else 0.0
            is_trainer = 1.0 if cat == "Trainer" else 0.0
            is_energy = 1.0 if cat == "Energy" else 0.0
            return pd.Series([is_legend, is_fav, is_starter, is_reg, is_trainer, is_energy])

        arch_cols = ["arch_legend", "arch_fav", "arch_starter", "arch_regular", "arch_trainer", "arch_energy"]
        self.df[arch_cols] = self.df.apply(get_archetype_flags, axis=1)

        # 5. Stage Tier
        self.df["stage_tier"] = self.df["stage"].map(STAGE_TIER_MAP).fillna(0).astype(float)

        # 6. Renowned Artist
        self.df["is_renowned_artist"] = self.df["illustrator"].isin(RENOWNED_ARTISTS).astype(float)

    def fit(self, n_neighbors: int = 5):
        print(f"Fitting KNN model (k={n_neighbors}, weighted cosine metric)...")
        X_train_raw = self.train_df[self.feature_cols].values
        X_train_scaled = self.scaler.fit_transform(X_train_raw)

        # Apply feature weights
        weights_vec = np.array([self.feature_weights[col] for col in self.feature_cols])
        X_train_weighted = X_train_scaled * weights_vec

        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.knn.fit(X_train_weighted)
        print("Model successfully trained on vintage card baseline!")

    def _transform_card_features(self, card_row: pd.Series) -> np.ndarray:
        raw_vals = card_row[self.feature_cols].values.reshape(1, -1)
        scaled = self.scaler.transform(raw_vals)
        weights_vec = np.array([self.feature_weights[col] for col in self.feature_cols])
        return scaled * weights_vec

    def find_similar_vintage_cards(
        self,
        card_query: str,
        k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Finds a modern card by name or card_id, and retrieves its top-k similar vintage counterparts.
        """
        # Match by exact card_id or case-insensitive name match
        matches = self.df[self.df["card_id"].str.lower() == card_query.lower()]
        if matches.empty:
            matches = self.df[self.df["name"].str.lower() == card_query.lower()]

        if matches.empty:
            # Partial search
            matches = self.df[self.df["name"].str.contains(card_query, case=False, na=False)]

        if matches.empty:
            print(f"No cards found matching query: '{card_query}'")
            return None

        # Take the most recent printing if multiple matches found
        target_card = matches.sort_values(by="release_year", ascending=False).iloc[0]

        # Calculate neighbors
        q_vec = self._transform_card_features(target_card)
        dists, indices = self.knn.kneighbors(q_vec, n_neighbors=k)

        vintage_matches = self.train_df.iloc[indices[0]].copy()
        vintage_matches["similarity_score"] = np.round((1 - dists[0]) * 100, 1)

        prices = vintage_matches["market_price"].dropna().tolist()
        proj_mean = float(np.mean(prices)) if prices else 0.0
        proj_median = float(np.median(prices)) if prices else 0.0
        proj_min = float(np.min(prices)) if prices else 0.0
        proj_max = float(np.max(prices)) if prices else 0.0

        current_price = target_card["market_price"]
        upside_ratio = (proj_median / current_price) if (current_price and current_price > 0) else None

        return {
            "target_card": {
                "card_id": target_card["card_id"],
                "name": target_card["name"],
                "set_name": target_card["set_name"],
                "release_year": target_card["release_year"],
                "rarity": target_card["rarity"],
                "rarity_tier": int(target_card["rarity_tier"]),
                "is_secret_rare": bool(target_card["is_secret_rare"]),
                "has_holo": bool(target_card["has_holo"]),
                "illustrator": target_card["illustrator"],
                "current_market_price": current_price
            },
            "vintage_neighbors": vintage_matches[[
                "card_id", "name", "set_name", "release_year", "rarity",
                "market_price", "similarity_score"
            ]].to_dict(orient="records"),
            "projections": {
                "sample_size": len(prices),
                "vintage_mean_price": round(proj_mean, 2),
                "vintage_median_price": round(proj_median, 2),
                "vintage_min_price": round(proj_min, 2),
                "vintage_max_price": round(proj_max, 2),
                "upside_multiplier": round(upside_ratio, 1) if upside_ratio else None
            }
        }

    def print_similarity_report(self, res: Dict[str, Any]):
        t = res["target_card"]
        p = res["projections"]

        print("\n" + "=" * 70)
        print(f"🎯 TARGET MODERN CARD: {t['name']} ({t['card_id']})")
        print(f"   Set: {t['set_name']} | Release: {int(t['release_year'] if t['release_year'] else 0)}")
        print(f"   Rarity: {t['rarity']} (Prestige Tier {t['rarity_tier']}/5)")
        print(f"   Secret Rare: {'Yes' if t['is_secret_rare'] else 'No'} | Holo: {'Yes' if t['has_holo'] else 'No'} | Artist: {t['illustrator']}")
        print(f"   Current Market Price: ${t['current_market_price']:.2f}" if t['current_market_price'] else "   Current Market Price: N/A")
        print("-" * 70)
        print("🏆 TOP SIMILAR VINTAGE COUNTERPARTS (<2016):")
        for i, n in enumerate(res["vintage_neighbors"], 1):
            yr = int(n['release_year']) if n['release_year'] else 'N/A'
            print(f"  {i}. [{n['similarity_score']}% match] {n['name']:<22} | {n['set_name']:<22} ({yr}) | {n['rarity']:<14} | ${n['market_price']:>7.2f}")
        print("-" * 70)
        print("📊 PROJECTED MATURE VINTAGE VALUE:")
        print(f"   Average Vintage Price:  ${p['vintage_mean_price']:.2f}")
        print(f"   Median Vintage Price:   ${p['vintage_median_price']:.2f}")
        print(f"   Vintage Price Range:    ${p['vintage_min_price']:.2f} – ${p['vintage_max_price']:.2f}")
        if p["upside_multiplier"]:
            print(f"   Estimated Maturation:   {p['upside_multiplier']}x current price")
        print("=" * 70 + "\n")


    def find_top_gems(self, min_price: float = 5.0, top_n: int = 10, k: int = 5) -> pd.DataFrame:
        """
        Scans all modern chase cards (Special Illustration Rare, Secret Rare, Illustration Rare)
        and ranks them by the ratio between similar vintage card median price and current price.
        """
        chase_rarities = ["Special illustration rare", "Secret Rare", "Illustration rare", "Hyper rare"]
        candidates = self.query_df[
            (self.query_df["rarity"].isin(chase_rarities)) &
            (self.query_df["market_price"] >= min_price)
        ].copy().reset_index(drop=True)

        if candidates.empty:
            return pd.DataFrame()

        weights_vec = np.array([self.feature_weights[col] for col in self.feature_cols])
        X_cand = self.scaler.transform(candidates[self.feature_cols].values) * weights_vec
        dists, indices = self.knn.kneighbors(X_cand, n_neighbors=k)

        records = []
        for i in range(len(candidates)):
            row = candidates.iloc[i]
            v_matches = self.train_df.iloc[indices[i]]
            v_prices = v_matches["market_price"].dropna().tolist()
            if not v_prices:
                continue

            v_mean = float(np.mean(v_prices))
            v_median = float(np.median(v_prices))
            curr = float(row["market_price"])
            upside = round(v_median / curr, 1)

            records.append({
                "card_id": row["card_id"],
                "name": row["name"],
                "set_name": row["set_name"],
                "release_year": int(row["release_year"]) if row["release_year"] else None,
                "rarity": row["rarity"],
                "current_price": round(curr, 2),
                "vintage_median_price": round(v_median, 2),
                "vintage_mean_price": round(v_mean, 2),
                "upside_potential": f"{upside}x"
            })

        res_df = pd.DataFrame(records).sort_values(by="vintage_median_price", ascending=False)
        return res_df.head(top_n)

    def print_top_gems_report(self, gems_df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("💎 TOP MODERN CHASE CARDS WITH HIGHEST VINTAGE MATURATION VALUE")
        print("=" * 80)
        for i, (_, row) in enumerate(gems_df.iterrows(), 1):
            yr = row['release_year'] if row['release_year'] else 'N/A'
            print(f"  {i:>2}. {row['name']:<22} | {row['set_name']:<20} ({yr}) | {row['rarity']:<24}")
            print(f"      Current: ${row['current_price']:<7.2f} ──▶ Projected Vintage Median: ${row['vintage_median_price']:<7.2f} (Upside: {row['upside_potential']})")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pokémon Card Collector Similarity & Maturation Model")
    parser.add_argument("--card", type=str, default=None, help="Card name to query (e.g. Charizard)")
    parser.add_argument("--card_id", type=str, default=None, help="Specific card ID to query (e.g. sv03.5-199)")
    parser.add_argument("--top-gems", type=int, default=None, help="Scan modern sets and output top N highest-value gems")
    parser.add_argument("-k", type=int, default=5, help="Number of vintage neighbors to retrieve")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cards_tabular.parquet")
    sets_meta_path = os.path.join(base_dir, "data", "sets_metadata.json")

    model = PokemonCardSimilarityModel(data_path, sets_meta_path)
    model.load_and_prepare_data()
    model.fit(n_neighbors=args.k)

    if args.top_gems:
        gems = model.find_top_gems(top_n=args.top_gems, k=args.k)
        model.print_top_gems_report(gems)
    else:
        query = args.card_id if args.card_id else (args.card if args.card else "Charizard ex")
        result = model.find_similar_vintage_cards(query, k=args.k)
        if result:
            model.print_similarity_report(result)


if __name__ == "__main__":
    main()
