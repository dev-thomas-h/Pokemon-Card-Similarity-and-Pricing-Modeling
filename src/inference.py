import os
import sys
import json
from typing import Dict, List, Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd
from src.features import FEATURE_COLS, FEATURE_WEIGHTS, engineer_features


class CardSimilarityEngine:
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        self.models_dir = models_dir

        self.knn = None
        self.scaler = None
        self.train_df: pd.DataFrame = None
        self.query_df: pd.DataFrame = None
        self.query_vectors: np.ndarray = None
        self.card_id_to_query_idx: Dict[str, int] = {}
        self.weights_vec = np.array([FEATURE_WEIGHTS[c] for c in FEATURE_COLS])

        self._load_artifacts()

    def _load_artifacts(self):
        knn_file = os.path.join(self.models_dir, "knn_model.joblib")
        scaler_file = os.path.join(self.models_dir, "scaler.joblib")
        train_file = os.path.join(self.models_dir, "train_cards.parquet")
        query_file = os.path.join(self.models_dir, "query_cards.parquet")
        vectors_file = os.path.join(self.models_dir, "query_vectors.npy")

        if not os.path.exists(knn_file):
            raise FileNotFoundError(f"Model artifacts not found in {self.models_dir}. Please run src/train.py first.")

        self.knn = joblib.load(knn_file)
        self.scaler = joblib.load(scaler_file)
        self.train_df = pd.read_parquet(train_file)
        self.query_df = pd.read_parquet(query_file)
        self.query_vectors = np.load(vectors_file)

        # Index modern cards for O(1) lookup
        for idx, cid in enumerate(self.query_df["card_id"]):
            self.card_id_to_query_idx[cid.lower()] = idx

    def get_filter_options(self) -> Dict[str, Any]:
        """
        Returns sets and rarities available in the modern query dataset.
        """
        # Sets
        sets_series = self.query_df[["set_id", "set_name", "release_year"]].drop_duplicates().sort_values(
            by=["release_year", "set_name"], ascending=[False, True]
        )
        sets_list = [
            {"id": r["set_id"], "name": r["set_name"], "year": int(r["release_year"]) if pd.notnull(r["release_year"]) else None}
            for _, r in sets_series.iterrows()
        ]

        # Rarities
        rarities = self.query_df["rarity"].dropna().unique().tolist()
        rarities.sort()

        return {
            "sets": sets_list,
            "rarities": rarities,
            "total_cards": len(self.query_df)
        }

    def search_cards(
        self,
        query: str = "",
        set_id: Optional[str] = None,
        rarity: Optional[str] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Searches modern cards with optional set and rarity filters.
        """
        df = self.query_df

        if set_id and set_id != "all":
            df = df[df["set_id"] == set_id]

        if rarity and rarity != "all":
            df = df[df["rarity"] == rarity]

        if query:
            q_clean = query.strip().lower()
            # ID match or Name contains
            mask = df["card_id"].str.lower().str.contains(q_clean, na=False) | \
                   df["name"].str.lower().str.contains(q_clean, na=False)
            df = df[mask]

        # Prioritize cards with market price and modern chase rarities
        df = df.sort_values(by=["release_year", "market_price"], ascending=[False, False])
        results = df.head(limit)[[
            "card_id", "name", "set_id", "set_name", "release_year", "rarity",
            "rarity_tier", "is_secret_rare", "has_holo", "image_url", "market_price"
        ]].copy()

        # Sanitize NaNs to None for valid JSON serialization
        results_records = results.replace({np.nan: None}).to_dict(orient="records")
        for r in results_records:
            if r.get("release_year") is not None:
                r["release_year"] = int(r["release_year"])
            if r.get("rarity_tier") is not None:
                r["rarity_tier"] = int(r["rarity_tier"])
            if r.get("is_secret_rare") is not None:
                r["is_secret_rare"] = bool(r["is_secret_rare"])
            if r.get("has_holo") is not None:
                r["has_holo"] = bool(r["has_holo"])

        return results_records

    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single card's metadata by card_id.
        """
        idx = self.card_id_to_query_idx.get(card_id.lower())
        if idx is not None:
            return self.query_df.iloc[idx].to_dict()

        # Check vintage cards
        v_match = self.train_df[self.train_df["card_id"].str.lower() == card_id.lower()]
        if not v_match.empty:
            return v_match.iloc[0].to_dict()

        return None

    def find_similar(self, card_id: str, k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Given a modern card_id, returns its top-K similar vintage counterparts and valuation projections.
        """
        idx = self.card_id_to_query_idx.get(card_id.lower())
        if idx is not None:
            target_card = self.query_df.iloc[idx]
            q_vec = self.query_vectors[idx].reshape(1, -1)
        else:
            # On-the-fly transform if not in query_df
            card_meta = self.get_card(card_id)
            if not card_meta:
                return None
            target_card = pd.Series(card_meta)
            raw_vals = np.array([target_card.get(col, 0.0) for col in FEATURE_COLS]).reshape(1, -1)
            scaled = self.scaler.transform(raw_vals)
            q_vec = scaled * self.weights_vec

        # Sub-millisecond KNN query
        dists, indices = self.knn.kneighbors(q_vec, n_neighbors=k)

        vintage_matches = self.train_df.iloc[indices[0]].copy()
        vintage_matches["similarity_score"] = np.round((1 - dists[0]) * 100, 1)

        prices = vintage_matches["market_price"].dropna().tolist()
        proj_mean = float(np.mean(prices)) if prices else 0.0
        proj_median = float(np.median(prices)) if prices else 0.0
        proj_min = float(np.min(prices)) if prices else 0.0
        proj_max = float(np.max(prices)) if prices else 0.0

        current_price = target_card["market_price"]
        current_price_val = float(current_price) if (pd.notnull(current_price) and current_price > 0) else None
        upside = round(proj_median / current_price_val, 1) if current_price_val else None

        neighbors_df = vintage_matches[[
            "card_id", "name", "set_id", "set_name", "series", "release_year",
            "rarity", "rarity_tier", "is_secret_rare", "has_holo", "image_url",
            "market_price", "similarity_score"
        ]].copy().replace({np.nan: None})
        neighbors_list = neighbors_df.to_dict(orient="records")
        for n in neighbors_list:
            if n.get("release_year") is not None:
                n["release_year"] = int(n["release_year"])
            if n.get("rarity_tier") is not None:
                n["rarity_tier"] = int(n["rarity_tier"])
            if n.get("is_secret_rare") is not None:
                n["is_secret_rare"] = bool(n["is_secret_rare"])
            if n.get("has_holo") is not None:
                n["has_holo"] = bool(n["has_holo"])

        return {
            "target_card": {
                "card_id": target_card["card_id"],
                "name": target_card["name"],
                "set_id": target_card["set_id"],
                "set_name": target_card["set_name"],
                "series": target_card.get("series"),
                "release_year": int(target_card["release_year"]) if pd.notnull(target_card["release_year"]) else None,
                "rarity": target_card["rarity"],
                "rarity_tier": int(target_card["rarity_tier"]),
                "is_secret_rare": bool(target_card["is_secret_rare"]),
                "has_holo": bool(target_card["has_holo"]),
                "stage": target_card.get("stage"),
                "primary_type": target_card.get("primary_type"),
                "illustrator": target_card.get("illustrator"),
                "image_url": target_card["image_url"],
                "current_market_price": current_price_val
            },
            "vintage_neighbors": neighbors_list,
            "projections": {
                "sample_size": len(prices),
                "vintage_mean_price": round(proj_mean, 2),
                "vintage_median_price": round(proj_median, 2),
                "vintage_min_price": round(proj_min, 2),
                "vintage_max_price": round(proj_max, 2),
                "upside_multiplier": upside
            }
        }

    def get_top_gems(
        self,
        min_price: float = 3.0,
        rarity_filter: Optional[str] = None,
        set_filter: Optional[str] = None,
        limit: int = 12,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Evaluates modern chase cards and ranks them by projected vintage upside.
        """
        chase_rarities = ["Special illustration rare", "Secret Rare", "Illustration rare", "Hyper rare", "Ultra Rare"]
        mask = (self.query_df["market_price"] >= min_price)

        if rarity_filter and rarity_filter != "all":
            mask &= (self.query_df["rarity"] == rarity_filter)
        else:
            mask &= (self.query_df["rarity"].isin(chase_rarities))

        if set_filter and set_filter != "all":
            mask &= (self.query_df["set_id"] == set_filter)

        candidates = self.query_df[mask].copy().reset_index(drop=True)
        if candidates.empty:
            return []

        # Vectorized batch neighbor search
        indices_list = [self.card_id_to_query_idx[cid.lower()] for cid in candidates["card_id"]]
        cand_vectors = self.query_vectors[indices_list]

        dists, nbr_indices = self.knn.kneighbors(cand_vectors, n_neighbors=k)

        records = []
        for i in range(len(candidates)):
            row = candidates.iloc[i]
            v_matches = self.train_df.iloc[nbr_indices[i]]
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
                "set_id": row["set_id"],
                "release_year": int(row["release_year"]) if pd.notnull(row["release_year"]) else None,
                "rarity": row["rarity"],
                "rarity_tier": int(row["rarity_tier"]),
                "is_secret_rare": bool(row["is_secret_rare"]),
                "image_url": row["image_url"],
                "current_price": round(curr, 2),
                "vintage_median_price": round(v_median, 2),
                "vintage_mean_price": round(v_mean, 2),
                "upside_multiplier": upside
            })

        # Sort by highest vintage median price
        records.sort(key=lambda r: (r["vintage_median_price"], r["upside_multiplier"]), reverse=True)
        return records[:limit]
