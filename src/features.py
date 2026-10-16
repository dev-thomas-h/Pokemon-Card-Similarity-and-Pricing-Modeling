"""
src/features.py
Feature engineering definitions and transformation functions for the Pokémon card similarity engine.
"""

import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Canonical National Pokédex IDs for subject archetypes
STARTERS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    152, 153, 154, 155, 156, 157, 158, 159, 160,
    252, 253, 254, 255, 256, 257, 258, 259, 260,
    387, 388, 389, 390, 391, 392, 393, 394, 395,
    495, 496, 497, 498, 499, 500, 501, 502, 503,
    650, 651, 652, 653, 654, 655, 656, 657, 658,
    722, 723, 724, 725, 726, 727, 728, 729, 730,
    810, 811, 812, 813, 814, 815, 816, 817, 818,
    906, 907, 908, 909, 910, 911, 912, 913, 914
}

LEGENDARIES = {
    144, 145, 146, 150, 151,
    243, 244, 245, 249, 250, 251,
    377, 378, 379, 380, 381, 382, 383, 384, 385, 386,
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
    638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
    716, 717, 718, 719, 720, 721,
    785, 786, 787, 788, 789, 790, 791, 792, 800, 801, 802,
    888, 889, 890, 891, 892, 894, 895, 896, 897, 898,
    1001, 1002, 1003, 1004, 1007, 1008, 1014, 1015, 1016, 1017, 1024, 1025
}

FAN_FAVORITES = {
    25, 26,                              # Pikachu, Raichu
    133, 134, 135, 136, 196, 197, 470, 471, 700, # Eevee and evolutions
    92, 93, 94,                          # Gengar line
    147, 148, 149,                       # Dragonite line
    246, 247, 248,                       # Tyranitar line
    443, 444, 445,                       # Garchomp line
    447, 448                             # Lucario line
}

RENOWNED_ARTISTS = {
    "Ken Sugimori", "Mitsuhiro Arita", "Naoki Saito", "Kouki Saitou",
    "Kagemaru Himeno", "Atsuko Nishida", "Masakazu Fukuda", "Tomokazu Komiya",  "Yuka Morii"
}

RARITY_TIER_MAP = {
    "Secret Rare": 5,
    "Special illustration rare": 5,
    "Hyper rare": 5,
    "LEGEND": 5,
    "Black White Rare": 5,
    "Mega Hyper Rare": 5,
    "Illustration rare": 4,
    "Rare Holo LV.X": 4,
    "Rare PRIME": 4,
    "Ultra Rare": 4,
    "Shiny rare": 4,
    "Shiny Ultra Rare": 4,
    "Amazing Rare": 4,
    "Double rare": 3,
    "Rare Holo": 3,
    "Holo Rare": 3,
    "Holo Rare V": 3,
    "Holo Rare VMAX": 3,
    "Holo Rare VSTAR": 3,
    "Radiant Rare": 3,
    "ACE SPEC Rare": 3,
    "Rare": 2,
    "Promo": 2,
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

FEATURE_WEIGHTS = {
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

FEATURE_COLS = list(FEATURE_WEIGHTS.keys())


def build_image_url(card_id: str, set_id: str, series: str, local_id: str) -> str:
    """
    Constructs the high-resolution CDN image URL from TCGdex conventions.
    """
    s_clean = (series or "").lower().replace(" ", "").replace("&", "")
    if not s_clean:
        s_clean = "base"
    return f"https://assets.tcgdex.net/en/{s_clean}/{set_id}/{local_id}/high.webp"


def engineer_features(df: pd.DataFrame, sets_meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms the raw tabular card dataframe into an engineered feature dataframe.
    """
    df = df.copy()

    # 1. Rarity Tier (1 - 5)
    df["rarity_tier"] = df["rarity"].map(RARITY_TIER_MAP).fillna(1).astype(float)

    # 2. Secret Rare flag
    def check_secret(row):
        meta = sets_meta.get(row["set_id"])
        if not meta or not meta.get("officialCardCount"):
            return 0.0
        try:
            lid = int(row["local_id"])
            return 1.0 if lid > meta["officialCardCount"] else 0.0
        except (ValueError, TypeError):
            return 0.0

    df["is_secret_rare"] = df.apply(check_secret, axis=1)

    # 3. Finishes
    df["has_holo"] = df["has_holo"].astype(float)
    df["has_first_edition"] = df["has_first_edition"].astype(float)
    df["has_reverse"] = df["has_reverse"].astype(float)

    # 4. Subject Archetype
    def get_arch_flags(row):
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
    df[arch_cols] = df.apply(get_arch_flags, axis=1)

    # 5. Stage Tier
    df["stage_tier"] = df["stage"].map(STAGE_TIER_MAP).fillna(0).astype(float)

    # 6. Renowned Artist
    df["is_renowned_artist"] = df["illustrator"].isin(RENOWNED_ARTISTS).astype(float)

    # 7. Image URL
    df["image_url"] = df.apply(
        lambda r: build_image_url(r["card_id"], r["set_id"], r.get("series", ""), r["local_id"]),
        axis=1
    )

    return df
