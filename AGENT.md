# CLAUDE.md — Agent & Project Guide

## Project Overview

**Pokémon Card Collector Similarity & Price Maturation Engine**
* **Core Goal**: Predict the long-term price equilibrium of modern Pokémon cards (`2016+`) by matching them to empirically mature vintage counterparts (`< 2016`) based on collector prestige archetypes rather than gameplay stats.
* **Core Premise**: Vintage cards have experienced 10–25 years of organic supply attrition and collector price discovery. Modern chase cards (e.g. Special Illustration Rares) share similar collector archetypes with vintage chase cards (e.g. Secret Rares, Gold Stars, Shining cards). Finding these vintage "twins" provides an empirical appraisal of what modern cards can mature to over time.

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
│   ├── cards_tabular.parquet     # Unified typed table (21,812 cards, 40 features)
│   └── cards_tabular.csv         # CSV export for inspection
│
├── models/                       # Pre-computed, serialized ML artifacts (<5ms inference)
│   ├── knn_model.joblib          # Fitted NearestNeighbors index (metric='cosine', k=10)
│   ├── scaler.joblib             # Fitted StandardScaler on vintage feature space
│   ├── train_cards.parquet       # Vintage cards baseline (<2016 with prices, 7,964 cards)
│   ├── query_cards.parquet       # Modern cards catalog (>=2016, 12,818 cards with CDN image URLs)
│   ├── query_vectors.npy         # Pre-transformed feature matrix for instant query lookup
│   └── model_meta.json           # Model training timestamp and feature metadata
│
├── src/                          # Modular production ML pipeline
│   ├── features.py               # Feature definitions, weights, Pokédex archetype maps
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
│   └── similarity_model.py       # Standalone all-in-one CLI evaluator
│
├── run_app.sh                    # Full-stack launcher script
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
python3 scripts/similarity_model.py --card "Gengar"

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

## Machine Learning & Modeling Architecture

### 1. Feature Philosophy: Collectibility > Battle Stats
Battle stats (`hp`, `damage`, `retreat`, `attacks`) have near-zero correlation with secondary market collector value. A 30 HP Base Set Magikarp or a 120 HP Base Set Charizard sells for hundreds of dollars despite being unplayable in modern tournaments.

Instead, the model vectorizes cards along **Collector Prestige Dimensions**:

* **`rarity_tier` (Weight: 3.0)**: Standardizes 25 years of changing rarity labels into a 1–5 prestige scale:
  * *Tier 5*: Vintage `Secret Rare`, `LEGEND`, `Shining`, `Gold Star` $\leftrightarrow$ Modern `Special illustration rare`, `Hyper rare`
  * *Tier 4*: Vintage `Rare Holo LV.X`, `Rare PRIME`, `Ultra Rare` $\leftrightarrow$ Modern `Illustration rare`, `Ultra Rare` (Alt Arts)
  * *Tier 3*: Vintage `Rare Holo` $\leftrightarrow$ Modern `Double rare`, `Holo Rare`, `Radiant Rare`, `ACE SPEC`
  * *Tier 2*: Regular `Rare`, standard `Promo`
  * *Tier 1*: `Uncommon`, `Common`, Bulk
* **`is_secret_rare` (Weight: 2.5)**: Binary flag (`local_id > officialCardCount`). Secret rares command a 100x median price multiplier in vintage sets.
* **`subject_archetype` (Weight: 2.0–2.5)**: Derived via National Pokédex number (`dexId`):
  * `arch_legend`: Legendary / Mythical Pokémon (e.g. Mewtwo, Lugia, Rayquaza)
  * `arch_fav`: Fan Favorites / Mascots (Pikachu, Eeveelutions, Gengar, Dragonite, Tyranitar)
  * `arch_starter`: Starter evolutionary lines (Charizard, Blastoise, Venusaur, etc.)
  * `arch_regular`: Standard wild Pokémon
  * `arch_trainer`: Trainer cards
  * `arch_energy`: Energy cards
* **Finishes**: `has_holo` (1.5), `has_first_edition` (1.0), `has_reverse` (0.5)
* **Stage & Form**: `stage_tier` (1.0) — Basic (1), Stage 1 (2), Stage 2 (3), Special mechanic (4: VMAX, LV.X, Prime, etc.)
* **`is_renowned_artist` (Weight: 1.0)**: Renowned illustrators (Sugimori, Arita, Naoki Saito, etc.)

### 2. Distance Metric & Scaling
* Features are standardized via `StandardScaler` ($\mu = 0, \sigma = 1$) fitted **only** on vintage cards.
* Scaled vectors are multiplied by `FEATURE_WEIGHTS`.
* Similarity metric: **Weighted Cosine Distance**:
  $$D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
* Pre-computed modern vectors (`models/query_vectors.npy`) enable sub-2ms query responses in `src/inference.py`.

### 3. Valuation Formula
When given a modern card $C$:
$$\text{Projected Vintage Value} = \text{median}(\{P(N_1), \dots, P(N_k)\})$$
$$\text{Maturation Upside} = \frac{\text{Projected Vintage Value}}{P(C)}$$

---

## Active Backlog & Next Steps for Incoming Agents

### 1. Feature Distinctiveness & Eliminating Neighbor Collisions
* **The Problem**: Because initial features are mostly categorical flags (`is_secret_rare=1`, `rarity_tier=5`), some modern chase cards map to identical vector coordinates, yielding similar vintage neighbor lists.
* **Actionable Features to Add** in `src/features.py`:
  1. **Elemental / Typing Affinity (`primary_type`)**:
     * Fire, Water, Lightning, Psychic, Dark, etc.
     * Charizard and Fire-type chase cards historically carry distinct collector followings compared to Grass or Psychic staples.
  2. **Generation of Origin (`generation_num`)**:
     * Gen 1 (Kanto) and Gen 2 (Johto) cards command a heavy nostalgia multiplier over later generations.
  3. **Continuous Set Position Ratio (`set_position_ratio`)**:
     * Calculate $\text{Ratio} = \frac{\text{local\_id}}{\text{officialCardCount}}$. A card numbered 204/165 (deep secret) behaves differently from 166/165.
  4. **Card Frame & Layout (`is_full_art`)**:
     * Distinguish full-bleed textured art from standard framed cards.

### 2. Data Refresh Automation
* Upstream price snapshots in `data/pricing/` update weekly on Saturdays.
* Build an automated update script or GitHub Action that:
  1. Pulls the newest Saturday snapshot from upstream.
  2. Runs `scripts/ingest_to_tabular.py`.
  3. Runs `python3 -m src.train` to re-index the model.
  4. Commits and deploys updated tables.

### 3. Visual & Qualitative Evaluation
* Card image URLs follow TCGdex CDN conventions:
  `https://assets.tcgdex.net/en/{series}/{set_id}/{local_id}/high.webp`
* When testing image rendering or similarity comps, always verify high-res image availability and graceful fallback behavior.

---

## Key Files Reference
* [`src/features.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/features.py) — Core feature transformation & weight dictionary
* [`src/train.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/train.py) — Training script that exports to `models/`
* [`src/inference.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/inference.py) — Real-time inference engine
* [`src/api.py`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/src/api.py) — FastAPI REST endpoints
* [`frontend/src/App.jsx`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/frontend/src/App.jsx) — React interface
* [`run_app.sh`](file:///Users/thorgy17/Top%20Secret%20Projects/Pokemon%20Modeling/pkm-card-data/run_app.sh) — Quickstart launcher
