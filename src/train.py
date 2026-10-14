import os
import sys
import json
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.features import engineer_features, FEATURE_COLS, FEATURE_WEIGHTS


def train_and_export_artifacts(
    data_path: str,
    sets_meta_path: str,
    models_dir: str
):
    start_time = time.time()
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading raw tabular cards from {data_path}...")
    df = pd.read_parquet(data_path)

    print(f"Loading set metadata from {sets_meta_path}...")
    with open(sets_meta_path, "r", encoding="utf-8") as f:
        sets_meta = json.load(f)

    print("Executing feature engineering pipeline...")
    df_feat = engineer_features(df, sets_meta)

    # 1. Vintage Baseline (<2016 with prices)
    train_mask = (df_feat["is_older_era"] == True) & (df_feat["market_price"].notnull()) & (df_feat["market_price"] > 0)
    train_df = df_feat[train_mask].copy().reset_index(drop=True)

    # 2. Modern Cards (>=2016)
    query_mask = (df_feat["is_older_era"] == False)
    query_df = df_feat[query_mask].copy().reset_index(drop=True)

    print(f"Vintage training set size: {len(train_df):,} cards")
    print(f"Modern query set size:    {len(query_df):,} cards")

    # 3. Fit StandardScaler on vintage features
    print("Fitting StandardScaler and applying feature weights...")
    scaler = StandardScaler()
    X_train_raw = train_df[FEATURE_COLS].values
    X_train_scaled = scaler.fit_transform(X_train_raw)

    weights_vec = np.array([FEATURE_WEIGHTS[c] for c in FEATURE_COLS])
    X_train_weighted = X_train_scaled * weights_vec

    # 4. Fit NearestNeighbors model
    print("Fitting NearestNeighbors (metric='cosine', k=10)...")
    knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn.fit(X_train_weighted)

    # 5. Pre-transform query cards for zero-latency inference
    print("Pre-transforming modern feature vectors...")
    X_query_raw = query_df[FEATURE_COLS].values
    X_query_scaled = scaler.transform(X_query_raw)
    X_query_weighted = X_query_scaled * weights_vec

    # Save query feature vectors as numpy array in models_dir
    query_vectors_file = os.path.join(models_dir, "query_vectors.npy")
    np.save(query_vectors_file, X_query_weighted)

    # 6. Save model and scaler
    knn_file = os.path.join(models_dir, "knn_model.joblib")
    scaler_file = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(knn, knn_file)
    joblib.dump(scaler, scaler_file)

    # 7. Save indexed tables
    train_cards_file = os.path.join(models_dir, "train_cards.parquet")
    query_cards_file = os.path.join(models_dir, "query_cards.parquet")
    train_df.to_parquet(train_cards_file, index=False)
    query_df.to_parquet(query_cards_file, index=False)

    # 8. Save metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_cards_count": len(train_df),
        "query_cards_count": len(query_df),
        "feature_cols": FEATURE_COLS,
        "feature_weights": FEATURE_WEIGHTS,
        "elapsed_seconds": round(time.time() - start_time, 2)
    }
    meta_file = os.path.join(models_dir, "model_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts successfully exported to {models_dir} in {meta['elapsed_seconds']}s:")
    print(f"  • {knn_file}")
    print(f"  • {scaler_file}")
    print(f"  • {train_cards_file}")
    print(f"  • {query_cards_file}")
    print(f"  • {query_vectors_file}")
    print(f"  • {meta_file}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cards_tabular.parquet")
    sets_meta_path = os.path.join(base_dir, "data", "sets_metadata.json")
    models_dir = os.path.join(base_dir, "models")

    train_and_export_artifacts(data_path, sets_meta_path, models_dir)


if __name__ == "__main__":
    main()
