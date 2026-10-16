# AGENT.md — Agent & Project Guide

## Project Overview

**Pokémon Card Collector Similarity & Price Maturation Engine (V2)**
* **Core Goal**: Predict the long-term price equilibrium of modern Pokémon cards (`2016+`) by matching them to empirically mature vintage counterparts (`< 2016`) based on 31 collector prestige dimensions rather than gameplay stats.
* **Core Premise**: Vintage cards have experienced 10–25 years of organic supply attrition and collector price discovery. Modern chase cards (e.g. Special Illustration Rares) share similar collector archetypes with vintage chase cards (e.g. Secret Rares, Gold Stars, Shining cards, Crystal Pokémon). Finding these vintage "twins" provides an empirical appraisal of what modern cards can mature to over time.

---

## Repository Architecture

```text
pkm-card-data/
├── data/
│   ├── cards.json                # Raw catalog dump from TCGdex (nested: set_id -> card_id)
│   ├── sets.json                 # Raw set list from TCGdex
│   ├── pricing/
│   │   ├── latest.json           # Latest market price snapshot (TCGplayer USD, Cardmarket EUR)
│   │   └── {YYYY-MM-DD}.json     # Weekly historical pricing snapshots
│   ├── sets_metadata.json        # Cached set release dates, series, and officialCardCount
│   ├── cards_tabular.parquet     # Unified typed table (21,812 cards, 40 base columns)
│   └── cards_tabular.csv         # CSV export for inspection
│
├── models/                       # Pre-computed, serialized ML artifacts (<5ms inference)
│   ├── knn_model.joblib          # Fitted NearestNeighbors index (metric='cosine', k=10)
│   ├── scaler.joblib             # Unit-range identity scaler (with_mean=False, with_std=False)
│   ├── train_cards.parquet       # Vintage cards baseline (<2016 with prices, 7,964 cards)
│   ├── query_cards.parquet       # Modern cards catalog (>=2016, 12,818 cards with CDN image URLs)
│   ├── query_vectors.npy         # Pre-transformed 31-feature weighted matrix for O(1) lookup
│   └── model_meta.json           # Model training timestamp and feature metadata
│
├── src/                          # Modular production ML pipeline
│   ├── features.py               # V2 31-feature engineering engine, rarity reconstruction & weights
│   ├── train.py                  # Training pipeline & artifact serialization
│   ├── inference.py              # Sub-millisecond similarity engine & top-gems scanner
│   └── api.py                    # FastAPI REST backend & static React app mount
│
├── frontend/                     # Modern React + Vite Web Application
│   ├── src/
│   │   ├── App.jsx               # Interactive Card Matcher & Top Gems UI
│   │   └── index.css             # Glassmorphic dark theme, holo animations, responsive CSS
│   ├── dist/                     # Pre-built production bundle (served directly by FastAPI)
│   ├── vite.config.js            # Vite proxy config (/api -> http://localhost:8000)
│   └── package.json              # Node dependencies (React 18, Vite 5, lucide-react)
│
├── scripts/                      # Standalone CLI tools & data ingestion
│   ├── ingest_to_tabular.py      # Ingestion script (JSONs -> Parquet/CSV)
│   └── similarity_model.py       # Standalone CLI evaluator wrapping CardSimilarityEngine
│
├── run_app.sh                    # Full-stack launcher script
├── AGENT.md                      # Agent context, architecture & roadmap guide
└── README.md                     # Public documentation with architectural justifications
```

---

## Essential Commands

### Running the Application
```bash
# 1-command launch (FastAPI backend + React frontend on http://localhost:8000)
./run_app.sh
# Or directly:
python3 src/api.py
```

### Data Pipeline & Model Training
```bash
# 1. Ingest raw JSONs and regenerate data/cards_tabular.parquet
python3 scripts/ingest_to_tabular.py

# 2. Retrain KNN model and regenerate models/ artifacts
python3 -m src.train

# 3. CLI Query (Individual Card by ID or Name)
python3 scripts/similarity_model.py --card_id "sv03.5-199"
python3 scripts/similarity_model.py --card "Zapdos ex"

# 4. CLI Scan for Top Modern Gems
python3 scripts/similarity_model.py --top-gems 10
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev      # Runs Vite dev server on http://localhost:5173 (proxies /api to :8000)
npm run build    # Rebuilds production bundle into frontend/dist/
```

---

## Machine Learning & Modeling Architecture (V2 Engine)

### 1. Feature Philosophy: Collectibility > Battle Stats
Battle stats (`hp`, `damage`, `retreat`, `attacks`) have near-zero correlation with secondary market collector value. Instead, `src/features.py` vectorizes every card into **31 unit-range `[0, 1]` collector dimensions**:

1. **Empirical Vintage Species & Artist Prestige (Zero Data Leakage)**:
   * **`species_prestige` (Weight: 3.5)**: Bayesian-smoothed log-price residual (`k_smooth = 4.0`) of each Pokémon species (`dex_id`) relative to its `rarity_tier` median, computed **exclusively on `< 2016` vintage cards**. Differentiates species within the same archetype (e.g. *Charizard* vs. *Venusaur*, *Umbreon* vs. *Glaceon*). Unseen Gen 7–9 species fall back to their archetype prior.
   * **`artist_prestige` (Weight: 1.2)** & **`is_renowned_artist` (Weight: 0.8)**: Bayesian-smoothed vintage illustrator log-price premium (`k_smooth = 10.0`) paired with a curated hall-of-fame list (*Ken Sugimori, Mitsuhiro Arita, Yuka Morii, Naoki Saito, Kouki Saitou, Kagemaru Himeno, Atsuko Nishida, Masakazu Fukuda, Tomokazu Komiya, Ryo Ueda, 5ban Graphics, Akira Egawa, Shinji Kanda, HYOGONOSUKE, Teeziro, SowSow*).

2. **Reconstructed Rarity & Chase Status (Fixing the 1999–2010 TCGdex Blindspot)**:
   * Raw TCGdex data labeled almost all 1999–2007 grails simply as `"rarity": "Rare"`. `src/features.py` reconstructs true collector tiers (`rarity_tier` 1–5 and `rarity_norm` in `[0, 1]`, Weight: **3.2**):
     * **Tier 5**: Secret Rares (`local_id > officialCardCount`), Crystal Secret Rares (Aquapolis `145–147`, Skyridge `145–150`), Gold Stars (`" Star"`), `"Shining "` cards, `LEGEND`, Full-Art / Holo Vintage Supporters (`Ultra Rare` / `Rare Holo` Supporters), Modern `Special illustration rare`, `Hyper rare`.
     * **Tier 4**: Vintage `ex` (`ex1`–`ex16`), e-Card `H01`–`H32` holos, `Rare Holo LV.X`, `Rare PRIME`, `Shiny rare`, Modern `Illustration rare`, `Ultra Rare`.
     * **Tier 3**: 1999–2003 WOTC Holo-Only Rares (`local_id <= 15–19` in Base–Legendary Collection), `Rare Holo`, Modern `Double rare`, `Holo Rare V/VMAX/VSTAR`, `Radiant Rare`, `ACE SPEC Rare`.
     * **Tier 2**: Regular non-holo `Rare`, standard `Promo`.
     * **Tier 1**: `Uncommon`, `Common`.
   * **`is_secret_rare` (Weight: 2.5)**, **`secret_depth` (Weight: 1.2)**, **`is_chase_card` (Weight: 1.8)**, **`set_pos_norm` (Weight: 0.8)**.

3. **Biological Evolution Finality vs. Card Stage**:
   * **`is_fully_evolved` (Weight: 1.5)**: Derives pre-evolution Dex IDs directly from `Stage1`/`Stage2` `evolve_from` chains so evolved Pokémon presented as Basic mechanics (*Charizard ex*, *Zapdos ex*) never match unevolved basic secret rares (*Charmander*, *Ralts*).
   * **`stage_norm` (Weight: 1.0)**: Normalized progression from Basic (`0.25`) to Stage 2 (`0.75`) and Special/Mega/VMAX (`1.0`).

4. **Subject & Trainer Subtype Archetypes**:
   * **`arch_legend` (2.2)**, **`arch_fav` (2.0)**, **`arch_starter` (1.8)**, **`arch_pseudo` (1.6)**, **`arch_regular` (1.2)**, **`arch_energy` (2.0)**.
   * **Trainer Split**: Splits Trainer cards into **`arch_supporter` (2.2)** and **`arch_item` (2.0)** so modern SIR Supporters (*Miriam*, *Iono*) match vintage Full-Art Supporters (*Bianca*, *Cheren*, *Sabrina*) instead of Gold Item cards (*Dive Ball*, *Switch*).

5. **Generation Nostalgia & Elemental Typing**:
   * **`gen_nostalgia` (1.6)** & **`gen_norm` (1.0)**: Captures the Gen 1 (Kanto) and Gen 2 (Johto) collector premium and continuous generation proximity.
   * **11 Elemental Type Flags (`type_*`, Weight: 1.4 each)**: `Fire`, `Dragon`, `Darkness`, `Psychic`, `Lightning`, `Water`, `Grass`, `Fighting`, `Metal`, `Colorless`, `Fairy`.

6. **Repaired Foil Finishes**:
   * **`has_holo` (1.2)**, **`has_first_edition` (0.6)**, **`has_reverse` (0.4)**: Repairs missing `has_holo` flags in Black & White / XY sets (e.g. *Plasma Storm Charizard* `bw8-136`) using `pricing_variant` and `rarity_tier >= 3.0`.

### 2. Distance Metric & Unit-Range Scaling
* **Do NOT use `StandardScaler(with_std=True)`**: Dividing by standard deviation ($\sigma$) inflates sparse features (`secret_depth`, rare types, 1st Edition) by 10x–30x, drowning out `species_prestige` and `rarity_norm`.
* Instead, all 31 features in `FEATURE_COLS` are explicitly normalized to **`[0, 1]`** inside `src/features.py` and multiplied directly by `FEATURE_WEIGHTS` (`StandardScaler(with_mean=False, with_std=False)` in `src/train.py`).
* **Weighted Cosine Distance**:
  $$D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### 3. V1 vs. V2 Empirical Benchmarks
* **Unique Modern Vectors**: `358 / 12,818` (`2.8%`) $\rightarrow$ **`12,694 / 12,818` (`99.0%`)**
* **Unique Vintage Cards Used Across Modern Chase Cards**: `230` $\rightarrow$ **`913` (4.0x diversity)**
* **Vintage 5-Fold CV Spearman Correlation ($\rho$)**: `0.6383` $\rightarrow$ **`0.8259`**
* **Vintage 5-Fold CV Log-MAE**: `0.7859` $\rightarrow$ **`0.5596` (-28.8% error)**
* **Vintage 5-Fold CV Within 2x Actual Price**: `57.2%` $\rightarrow$ **`74.2%`**

---

## Active Backlog & Next Steps for Incoming Agents

### 1. Time-Series Price Velocity & Maturation Curves
* Utilize the historical snapshots in `data/pricing/{YYYY-MM-DD}.json` to compute short-term price momentum, volatility, and post-release depreciation/appreciation curves (J-curve dynamics).

### 2. Set Supply & Era Scarcity Multipliers
* Incorporate era-level print-run estimates or set-level median chase values to calibrate maturation multipliers across eras (e.g. late-WOTC e-Card scarcity vs. XY era supply).

### 3. Data Refresh Automation
* Upstream price snapshots in `data/pricing/` update weekly on Saturdays.
* Build an automated update script or GitHub Action that:
  1. Pulls the newest Saturday snapshot from upstream.
  2. Runs `scripts/ingest_to_tabular.py`.
  3. Runs `python3 -m src.train` to re-index the model.
  4. Commits and deploys updated tables.

---

## Key Files Reference
* [`src/features.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/features.py) — V2 31-feature transformation engine & weight dictionary
* [`src/train.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/train.py) — Training script that exports to `models/`
* [`src/inference.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/inference.py) — Real-time inference engine
* [`src/api.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/api.py) — FastAPI REST endpoints
* [`frontend/src/App.jsx`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/frontend/src/App.jsx) — React interface
* [`run_app.sh`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/run_app.sh) — Quickstart launcher
