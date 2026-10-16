"""
src/features.py
V2 Collector Feature Engineering Engine (31 features) for the Pokémon Card Similarity & Maturation Model.
Transforms raw card attributes into unit-range [0, 1] collector prestige dimensions.
"""

import re
from typing import Dict, Any, Tuple
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
    25, 26,                                       # Pikachu, Raichu
    133, 134, 135, 136, 196, 197, 470, 471, 700,  # Eevee and all 8 Eeveelutions
    59, 65, 68, 92, 93, 94, 130, 131, 143,        # Arcanine, Alakazam, Machamp, Gengar line, Gyarados, Lapras, Snorlax
    147, 148, 149,                                # Dragonite line
    212, 246, 247, 248,                           # Scizor, Tyranitar line
    282, 306, 359, 373, 376,                      # Gardevoir, Aggron, Absol, Salamence, Metagross
    443, 444, 445, 447, 448, 475,                 # Garchomp line, Lucario line, Gallade
    571, 609, 635, 637,                           # Zoroark, Chandelure, Hydreigon, Volcarona
    706, 778,                                     # Goodra, Mimikyu
    885, 886, 887, 937, 959, 998                  # Dragapult line, Ceruledge, Tinkaton, Baxcalibur
}

RENOWNED_ARTISTS = {
    "Ken Sugimori", "Mitsuhiro Arita", "Naoki Saito", "Kouki Saitou",
    "Kagemaru Himeno", "Atsuko Nishida", "Masakazu Fukuda", "Tomokazu Komiya", "Yuka Morii",
    "Ryo Ueda", "Shinji Kanda", "Akira Egawa", "sowsow", "HYOGONOSUKE"
}

CGI_STUDIOS = {
    "5ban Graphics", "Planeta", "Planeta Mochizuki", "Planeta Tsuji",
    "Planeta Igarashi", "aky CG Works", "Eske Yoshinob"
}

ITEM_TRAINER_KEYWORDS = {
    "Ball", "Potion", "Switch", "Candy", "Rod", "Rope", "Stretcher", "Pad", "Catcher",
    "Hammer", "Blower", "Scoop", "Belt", "Band", "Helmet", "Charm", "Cloak", "Cape",
    "Stone", "Board", "Energy", "Mail", "Receiver", "Pokedex", "Pokédex", "Flute",
    "Doll", "Fossil", "Shard", "Herb", "Berry", "Plume", "Armor", "Box", "Whistle",
    "Laser", "Bangle", "Policy", "Compressor", "Seeker", "Mainten", "Retrieval",
    "Recycle", "Search", "Finder", "Map", "Ticket", "Pass", "Cart", "Specs", "Goggles",
    "Boots", "Gloves", "Mask", "Memory", "Spirit Link", "Shrine", "Temple", "Stadium",
    "Gym", "Lab", "City", "Town", "Forest", "Mountain", "Valley", "Cave", "Ruins",
    "Tower", "Factory", "Plant", "Garden", "Park", "Beach", "Island", "Lake", "Sea",
    "Brook", "Hill", "Altar", "Court", "Arena", "Hall", "Dimension", "Field", "Path",
    "Basin", "Square", "Plaza", "Market", "Store", "Machine", "Lithograph", "Poffin",
    "Vessel", "Generator", "Seal", "Patch", "Urn", "Horn", "Crystal", "Trolley"
}

GEN_BOUNDS = [
    (1, 1, 151, 1.00),
    (2, 152, 251, 0.90),
    (3, 252, 386, 0.70),
    (4, 387, 493, 0.55),
    (5, 494, 649, 0.25),
    (6, 650, 721, 0.25),
    (7, 722, 809, 0.35),
    (8, 810, 905, 0.35),
    (9, 906, 1025, 0.35),
]

STAGE_NORM_MAP = {
    "Basic": 0.25,
    "Stage1": 0.50,
    "RESTORED": 0.50,
    "Stage2": 0.75,
    "VMAX": 1.00,
    "VSTAR": 1.00,
    "MEGA": 1.00,
    "LEVEL-UP": 1.00,
    "BREAK": 1.00,
    "V-UNION": 1.00,
}

# V2 Collector Feature Weights (applied to unit-range [0, 1] features)
FEATURE_WEIGHTS = {
    "rarity_norm": 4.5,
    "is_secret_rare": 1.8,
    "set_position_ratio": 1.4,
    "set_size_norm": 0.5,
    "species_prestige": 3.5,
    "is_fully_evolved": 2.5,
    "gen_nostalgia_score": 1.8,
    "gen_relative_dex": 0.6,
    "arch_legend": 2.2,
    "arch_fav": 2.0,
    "arch_starter": 2.2,
    "arch_regular": 1.5,
    "arch_supporter": 3.0,
    "arch_item": 2.5,
    "arch_energy": 2.5,
    "is_special_mechanic": 1.8,
    "is_character_or_owner": 1.0,
    "has_holo": 1.8,
    "has_first_edition": 0.8,
    "has_reverse": 0.5,
    "stage_norm": 1.2,
    "artist_prestige": 1.4,
    "type_fire": 1.2,
    "type_darkness": 1.0,
    "type_dragon": 1.0,
    "type_psychic": 1.0,
    "type_lightning": 1.0,
    "type_water": 1.0,
    "type_colorless": 0.8,
    "type_grass": 0.8,
    "type_other_elem": 0.8,
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


def _is_supporter_trainer(row: pd.Series) -> bool:
    if row["category"] != "Trainer":
        return False
    name = str(row.get("name") or "")
    name_lower = name.lower()
    for kw in ITEM_TRAINER_KEYWORDS:
        if kw.lower() in name_lower:
            return False
    return True


def _parse_set_position(row: pd.Series, sets_meta: Dict[str, Any]) -> Tuple[float, float, float]:
    sid = str(row.get("set_id") or "")
    lid = str(row.get("local_id") or "")
    meta = sets_meta.get(sid, {})
    off = meta.get("officialCardCount") or meta.get("totalCardCount") or 100
    if off <= 0:
        off = 100

    size_norm = float(np.clip((np.log10(off) - 1.0) / 1.5, 0.0, 1.0))

    # e-Card Holo series (H1..H32 in Aquapolis / Skyridge)
    if lid.startswith("H") and lid[1:].isdigit():
        return 1.0, 1.0, size_norm

    m = re.search(r"(\d+)$", lid)
    if m:
        num = int(m.group(1))
        ratio = float(min(1.25, num / float(off)) / 1.25)
        is_sec = 1.0 if num > off else 0.0
        return ratio, is_sec, size_norm

    return 0.4, 0.0, size_norm


def _compute_true_rarity_tier(row: pd.Series) -> float:
    """
    Reconstructs the true collector rarity tier (1.0 - 5.0) across all eras,
    resolving the 1999-2010 TCGdex catalog quirk where Gold Stars, Shining cards,
    Crystal secret rares, e-Card H-Holos, vintage ex cards, and WOTC Holo Rares
    were labeled simply as 'Rare'.
    """
    r = str(row.get("rarity") or "")
    name = str(row.get("name") or "")
    lid = str(row.get("local_id") or "")
    sid = str(row.get("set_id") or "")
    yr = row.get("release_year") or 2020
    has_holo = bool(row.get("has_holo")) or ("holofoil" in str(row.get("pricing_variant") or ""))
    has_normal = bool(row.get("has_normal"))
    is_sec = row.get("is_secret_rare", 0.0) > 0
    is_supp = bool(row.get("is_supporter"))

    # Vintage Full-Art / Holo Gym Leader Supporters were the top Supporter chase tier of their era
    if is_supp and yr < 2016 and r in {"Ultra Rare", "Rare Holo", "Secret Rare"}:
        return 5.0

    # Tier 5: Grails / Secret Rares / Gold Stars / Shining / SIR / Hyper Rare / e-Card H-Holos
    if (
        r in {"Secret Rare", "Special illustration rare", "Hyper rare", "LEGEND", "Black White Rare", "Mega Hyper Rare"}
        or is_sec
        or name.endswith(" Star")
        or name.startswith("Shining ")
        or (lid.startswith("H") and sid.startswith("ecard"))
    ):
        return 5.0

    # Tier 4: High Chase / Ultra Rare / IR / Vintage ex / LV.X / Prime
    if (
        r in {"Illustration rare", "Rare Holo LV.X", "Rare PRIME", "Ultra Rare", "Shiny rare", "Shiny Ultra Rare", "Amazing Rare"}
        or name.endswith(" ex")
        or name.endswith(" EX")
        or name.endswith(" LV.X")
        or name.endswith(" Prime")
    ):
        return 4.0

    # In WOTC / e-Card sets (1999-2003), Holo-Only Rares (#1-#18) were the primary chase cards in sets
    if yr <= 2003 and r in {"Rare", "Rare Holo"} and has_holo and not has_normal:
        return 4.5

    # Mid-era Holo-Only Rares (2004-2010)
    if r in {"Rare", "Rare Holo"} and has_holo and not has_normal:
        return 3.5

    # Tier 3: Standard Holo Rares & Modern Double Rares / V / VMAX / VSTAR
    if (
        r in {"Double rare", "Rare Holo", "Holo Rare", "Holo Rare V", "Holo Rare VMAX", "Holo Rare VSTAR", "Radiant Rare", "ACE SPEC Rare"}
        or (r == "Rare" and has_holo)
    ):
        return 3.0

    # Tier 2: Non-Holo Rare / Promo
    if r in {"Rare", "Promo"}:
        return 2.0

    return 1.0


def _parse_generation_info(did: Any) -> Tuple[float, float]:
    if pd.isna(did):
        return 0.0, 0.0
    d = int(did)
    for _, lo, hi, score in GEN_BOUNDS:
        if lo <= d <= hi:
            return score, (d - lo) / float(max(1, hi - lo))
    return 0.30, 0.50


def engineer_features(df: pd.DataFrame, sets_meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms the raw tabular card dataframe into the V2 31-feature collector prestige space.
    All model features in FEATURE_COLS are bounded in [0, 1].
    """
    df = df.copy()

    # 1. Identify biological pre-evolutions from Stage1/Stage2 evolution graph
    std_evolutions = set(
        df[df["stage"].isin(["Stage1", "Stage2"])]["evolve_from"]
        .dropna()
        .str.strip()
        .unique()
    )
    std_evolutions = {n for n in std_evolutions if not n.endswith((" V", " EX", " ex", " GX"))}
    pre_evo_dex_ids = set(df[df["name"].isin(std_evolutions)]["dex_id"].dropna().astype(int).unique())

    # 2. Supporter vs Item/Tool/Stadium Trainer classification
    df["is_supporter"] = df.apply(_is_supporter_trainer, axis=1)

    # 3. Continuous Set Position & Set Size
    pos_res = df.apply(lambda r: _parse_set_position(r, sets_meta), axis=1)
    df["set_position_ratio"] = [r[0] for r in pos_res]
    df["is_secret_rare"] = [r[1] for r in pos_res]
    df["set_size_norm"] = [r[2] for r in pos_res]

    # 4. True Collector Rarity Tier (1.0 - 5.0) & Normalized [0, 1] Rarity
    df["rarity_tier"] = df.apply(_compute_true_rarity_tier, axis=1)
    df["rarity_norm"] = (df["rarity_tier"] - 1.0) / 4.0

    # 5. Cleaned Holo & Print Finishes
    df["has_holo"] = (
        df["has_holo"].astype(bool)
        | df["pricing_variant"].fillna("").str.contains("holofoil")
        | (df["rarity_tier"] >= 3.0)
    ).astype(float)
    df["has_first_edition"] = df["has_first_edition"].astype(float)
    df["has_reverse"] = df["has_reverse"].astype(float)

    # 6. Subject Archetype Flags
    def get_arch_key(row):
        cat = row["category"]
        if cat == "Trainer":
            return "supporter" if row["is_supporter"] else "item"
        if cat == "Energy":
            return "energy"
        did = row["dex_id"]
        if did in LEGENDARIES:
            return "legend"
        if did in FAN_FAVORITES:
            return "fav"
        if did in STARTERS:
            return "starter"
        return "regular"

    df["arch_key"] = df.apply(get_arch_key, axis=1)
    for k in ["legend", "fav", "starter", "regular", "supporter", "item", "energy"]:
        df[f"arch_{k}"] = (df["arch_key"] == k).astype(float)

    # 7. Biological Evolution Status (Fully Evolved / Standalone Apex vs Pre-Evolution)
    def get_is_fully_evolved(row):
        if row["category"] != "Pokemon":
            return 0.0
        did = row["dex_id"]
        if pd.notna(did) and int(did) in pre_evo_dex_ids:
            return 0.0
        return 1.0

    df["is_fully_evolved"] = df.apply(get_is_fully_evolved, axis=1)

    # 8. Special Chase Subtype & Character/Owner Flags
    df["is_special_mechanic"] = (
        df["name"].str.contains(
            r" ex$| EX$| GX$| V$| VMAX$| VSTAR$| LV\.X$| Prime$| LEGEND$| Star$|^Shining |^Radiant ",
            regex=True,
            na=False,
        )
        | df["stage"].isin(["VMAX", "VSTAR", "MEGA", "LEVEL-UP", "BREAK", "V-UNION"])
    ).astype(float)

    df["is_character_or_owner"] = (
        df["name"].str.contains(r"'s |’s |^Dark |^Light ", regex=True, na=False)
        | df["set_id"].str.endswith(("tg", "gg", "rc"), na=False)
        | df["rarity"].isin(["Illustration rare", "Special illustration rare"])
    ).astype(float)

    # 9. Generation Nostalgia & Relative Dex Position
    gen_res = df["dex_id"].apply(_parse_generation_info)
    df["gen_nostalgia_score"] = [r[0] for r in gen_res]
    df["gen_relative_dex"] = [r[1] for r in gen_res]

    # 10. Elemental Type Affinity
    for t_val, col_name in [
        ({"Fire"}, "type_fire"),
        ({"Darkness"}, "type_darkness"),
        ({"Dragon"}, "type_dragon"),
        ({"Psychic"}, "type_psychic"),
        ({"Lightning"}, "type_lightning"),
        ({"Water"}, "type_water"),
        ({"Colorless"}, "type_colorless"),
        ({"Grass"}, "type_grass"),
        ({"Fighting", "Metal", "Fairy"}, "type_other_elem"),
    ]:
        df[col_name] = df["primary_type"].isin(t_val).astype(float)

    # 11. Stage Normalized
    df["stage_norm"] = df["stage"].map(STAGE_NORM_MAP).fillna(0.0).astype(float)

    # 12. Empirical Vintage Species & Artist Prestige (Zero Leakage: computed strictly on <2016 vintage pool)
    train_mask = (df["is_older_era"] == True) & (df["market_price"].notnull()) & (df["market_price"] > 0)
    tr = df[train_mask].copy()
    tr["log_price"] = np.log1p(tr["market_price"])
    tier_medians = tr.groupby("rarity_tier")["log_price"].median().to_dict()
    tr["price_residual"] = tr["log_price"] - tr["rarity_tier"].map(tier_medians)

    arch_priors = tr.groupby("arch_key")["price_residual"].mean().to_dict()
    dex_grp = tr.groupby(["dex_id", "arch_key"])["price_residual"].agg(["count", "mean"]).reset_index()
    dex_prestige_map = {}
    for _, r in dex_grp.iterrows():
        prior = arch_priors.get(r["arch_key"], 0.0)
        dex_prestige_map[r["dex_id"]] = (r["count"] * r["mean"] + 3.0 * prior) / (r["count"] + 3.0)

    def lookup_species_prestige(row):
        did = row["dex_id"]
        raw = dex_prestige_map[did] if (pd.notna(did) and did in dex_prestige_map) else arch_priors.get(row["arch_key"], 0.0)
        return float(np.clip((raw + 1.5) / 3.5, 0.0, 1.0))

    df["species_prestige"] = df.apply(lookup_species_prestige, axis=1)

    art_grp = tr.groupby("illustrator")["price_residual"].agg(["count", "mean"])
    art_prestige_map = ((art_grp["count"] * art_grp["mean"]) / (art_grp["count"] + 5.0)).to_dict()

    def lookup_artist_prestige(row):
        ill = str(row.get("illustrator") or "")
        raw = art_prestige_map.get(ill, 0.0)
        if ill in RENOWNED_ARTISTS:
            raw += 0.35
        elif ill in CGI_STUDIOS:
            raw -= 0.35
        return float(np.clip((raw + 1.0) / 2.5, 0.0, 1.0))

    df["artist_prestige"] = df.apply(lookup_artist_prestige, axis=1)

    # 13. High-Res Image URL
    df["image_url"] = df.apply(
        lambda r: build_image_url(r["card_id"], r["set_id"], r.get("series", ""), r["local_id"]),
        axis=1,
    )

    return df
