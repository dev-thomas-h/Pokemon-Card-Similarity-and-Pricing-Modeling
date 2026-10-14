# Pokémon TCG card data

Local snapshot of English Pokémon Trading Card Game data: set lists, full card objects, weekly market prices, and (optionally) card images.

Sourced from the [TCGdex API](https://tcgdex.net). **TCG Pocket sets are excluded.** There are no fetch scripts in this repo; regenerate or update files from the sibling `tcgdex-data` project.

As of the files in this tree:

| Dataset | Count |
| --- | ---: |
| Sets (`data/sets.json`) | 206 |
| Cards (`data/cards.json`) | 21,812 |
| Pricing snapshot (`data/pricing/latest.json`, 2026-10-10) | 21,355 |
| Indexed images (`pokemon-cards/index.json`) | 20,051 |

Card and pricing counts differ because they are produced by different jobs at different times. Pricing walks the live API on snapshot day; `cards.json` is a catalog dump. Some cards have no image URL, so the image index is smaller still.

## Layout

```
just-card-data/
├── data/
│   ├── sets.json                 # Brief set list
│   ├── cards.json                # Full cards, keyed set_id → card_id
│   └── pricing/
│       ├── {YYYY-MM-DD}.json     # Dated snapshots, keyed by card_id
│       ├── latest.json           # Most recent complete snapshot
│       └── status.json           # Per-day completion + run history
└── pokemon-cards/                # gitignored (images are ~1.4 GB)
    ├── index.json                # key → {path, name, set, cached_at}
    └── images/{set_id}/en/{localId}.webp
```

`pokemon-cards/` is gitignored. Keep images locally (or unpack them from a sidecar archive); they are not part of a normal clone.

## `data/sets.json`

Array of brief set objects:

```json
{
  "id": "base1",
  "name": "Base Set",
  "logo": "https://assets.tcgdex.net/en/base/base1/logo",
  "symbol": "https://assets.tcgdex.net/univ/base/base1/symbol",
  "cardCount": { "total": 102, "official": 102 }
}
```

`logo` and `symbol` are omitted when TCGdex has none. `cardCount.total` includes extras (secret rares, etc.); `official` is the printed set size.

## `data/cards.json`

Nested object `{ set_id: { card_id: <card> } }`. Look up by TCGdex ids:

```python
import json

cards = json.load(open("data/cards.json"))
alakazam = cards["base1"]["base1-1"]
```

Card objects are stored as the API returned them. Typical fields:

- Identity: `id`, `localId`, `name`, `category` (`Pokemon` / `Trainer` / `Energy`), `rarity`, `illustrator`, `image`
- Pokémon: `hp`, `types`, `stage`, `evolveFrom`, `dexId`, `attacks`, `abilities`, `weaknesses`, `retreat`
- Printings: `variants`, `variants_detailed`
- Set brief: `set` (`id`, `name`, `cardCount`, optional `logo`)

`image` is a TCGdex asset URL **without** a file extension. High-quality webp is `{image}/high.webp`.

## Pricing

Snapshots live under `data/pricing/`. Each file is keyed by card id:

```json
{
  "base1-1": {
    "name": "Alakazam",
    "set": "base1",
    "localId": "1",
    "dexId": [65],
    "fetched_at": "2026-10-10T22:02:17+00:00",
    "pricing": {
      "cardmarket": { "unit": "EUR", "avg": 51.3, "low": 9.99, "trend": 43.41 },
      "tcgplayer": { "unit": "USD", "holofoil": { "marketPrice": 70.76 } }
    }
  }
}
```

The `pricing` block is stored **verbatim** from TCGdex (Cardmarket EUR, TCGplayer USD, including variant shapes such as `-holo` suffixes and `holofoil` / `reverse` / `normal` sub-objects). Missing markets are `null`. `dexId` is omitted for trainers and energy.

- **`latest.json`** — copy of the newest *complete* snapshot. Prefer this for “current prices.”
- **`{YYYY-MM-DD}.json`** — history. Files dated **2026-09-05 and earlier** are daily; from **2026-09-12** they are **Saturdays only**. The format did not change.
- **`status.json`** — per-day run log, expected vs fetched counts, coverage (`full` / `cardmarket_only` / `tcgplayer_only` / `none`), and upstream-date flags.

Gzipped snapshots (if present) are the same JSON:

```python
import gzip, json
snap = json.load(gzip.open("data/pricing/2026-07-06.json.gz", "rt"))
```

Card text, names, and images are © Pokémon / Nintendo / Creatures / GAME FREAK / The Pokémon Company. Data is assembled via [TCGdex](https://tcgdex.net). This repo is a personal local cache, not an official Pokémon product.

---

# Pokémon Card Collector Similarity & Price Maturation Engine

This repository extends the raw card snapshot data with a machine-learning pipeline and **K-Nearest Neighbors (KNN)** similarity engine. The model is trained on mature vintage cards (`< 2016`) to find the historical counterparts of modern cards (`2016+`) and project long-term collector value based on empirical vintage precedents.

## Added Files & Layout

```
pkm-card-data/
├── data/
│   ├── cards_tabular.parquet     # Unified, typed tabular dataset (21,812 cards, 40 columns)
│   ├── cards_tabular.csv         # CSV export for inspection and analysis
│   └── sets_metadata.json        # Cached set release dates, series, and official counts
├── models/                       # Serialized ML artifacts for sub-5ms inference
│   ├── knn_model.joblib          # Trained NearestNeighbors index
│   ├── scaler.joblib             # Fitted StandardScaler
│   ├── train_cards.parquet       # Fast vintage benchmark index
│   ├── query_cards.parquet       # Modern cards catalog with image URLs
│   └── query_vectors.npy         # Pre-computed feature vectors
├── src/                          # Modular production ML pipeline
│   ├── features.py               # Feature definitions & transformation functions
│   ├── train.py                  # Trains and exports artifacts to models/
│   ├── inference.py              # Sub-millisecond similarity engine & gem scanner
│   └── api.py                    # FastAPI REST server & static UI mount
├── frontend/                     # Modern React + Vite web application
│   ├── src/App.jsx               # Interactive Card Matcher & Top Gems UI
│   ├── src/index.css             # Glassmorphic dark-mode collector theme
│   └── dist/                     # Production React bundle
├── run_app.sh                    # One-command launcher for the full-stack app
└── scripts/                      # Prototyping & one-off utility scripts
    ├── ingest_to_tabular.py      # Flattens nested JSONs and merges TCGplayer pricing
    └── similarity_model.py       # Standalone CLI similarity evaluator
```

---

## Interactive React Web Application

To launch the full interactive web application (FastAPI backend + React frontend on `http://localhost:8000`):

```bash
./run_app.sh
```
*(Or run `python3 src/api.py`)*

Then open your browser to **[http://localhost:8000](http://localhost:8000)**.

### Features of the Web App:
* **Interactive Card Search**: Instant debounced search across 12,800+ modern cards with set and rarity dropdown filters.
* **Empirical Maturation Appraisal**: Displays high-resolution card artwork, collector prestige badges, current market price, and projected mature vintage value with upside multipliers.
* **Historical Vintage Comps**: Displays the top 5 closest historical benchmark cards with image cards, release years, match percentages, and vintage market values.
* **Top Gems Scanner**: Filterable discovery dashboard ranking modern chase cards with the highest historical appreciation potential.

---

## Model Architecture & Design Justifications

### 1. Why K-Nearest Neighbors (KNN)?
* **Real-World Appraisal Grounding**: Predicting the future price of speculative collectibles using black-box regression or deep learning yields single unexplainable numbers. KNN works identically to professional appraisal comps: *"Card X is projected to reach ~$375 because these 5 historical cards with the exact same prestige archetype sell for $375."*
* **Era Asymmetry**: Vintage cards (Base Set through Black & White / early XY) have experienced 10–25 years of organic supply attrition and collector price discovery. Modern cards (Sun & Moon through Scarlet & Violet) are still experiencing active pack openings and short-term market fluctuations. KNN directly connects modern card traits to mature historical baselines.

### 2. Feature Selection: Collectibility over Gameplay Stats
Traditional card attributes like **HP, attack power, retreat costs, and damage** have near-zero correlation with secondary market collector value. A 30 HP Base Set Magikarp or a 120 HP Base Set Charizard sells for hundreds of dollars despite being unplayable in modern tournaments.

Instead, the model vectorizes cards along **Collector Prestige Dimensions**:

| Feature | Type | Weight | Valuation Rationale |
| :--- | :--- | :---: | :--- |
| **`rarity_tier`** | Ordinal (1–5) | **3.0** | Maps vintage grails (*Secret Rare, LEGEND, Shining*) to modern equivalents (*Special Illustration Rare, Hyper Rare*). |
| **`is_secret_rare`** | Binary (0/1) | **2.5** | Cards numbered beyond the official set count (e.g. #199/165) command a 100x median price multiplier in vintage sets. |
| **`subject_archetype`** | Categorical | **2.0–2.5** | Mapped via National Pokédex IDs (`dexId`): *Legendaries/Mythicals*, *Fan Favorites/Mascots* (Pikachu, Eeveelutions, Gengar), *Starter Lines* (Charizard, Blastoise), and *Regular Pokémon*. |
| **`has_holo`** | Binary (0/1) | **1.5** | Holographic foil finish premium. |
| **`stage_tier`** | Ordinal (0–4) | **1.0** | Stage 2 and special mechanics (VMAX, LV.X, Prime) carry distinct chase premiums over basic wild Pokémon. |
| **`is_renowned_artist`**| Binary (0/1) | **1.0** | Celebrated artists (Ken Sugimori, Mitsuhiro Arita, Naoki Saito, etc.). |
| **`has_first_edition`**| Binary (0/1) | **1.0** | Vintage 1st-edition scarcity flag. |
| **`has_reverse`** | Binary (0/1) | **0.5** | Reverse holofoil finish. |

### 3. Scaling & Distance Choice
* **Standardization**: Features are scaled using `StandardScaler` ($\mu = 0, \sigma = 1$) to prevent numerical scales from overshadowing binary flags.
* **Weighted Cosine Metric**:
  $$D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  Cosine distance measures the **directional profile** of the card in weighted collector space. It guarantees that cards with proportional feature alignments (e.g. high-prestige holo chase starters) cluster together regardless of magnitude offsets.

### 4. Training & Inference
* **Training Set**: 7,964 vintage cards (`release_year < 2016`) with active market sales in `data/pricing/latest.json`.
* **Inference Set**: 12,818 modern cards (`release_year >= 2016`).
* **Price Projection**: When evaluating card $C$, the engine queries the $k$ nearest historical neighbors $N_1, \dots, N_k$, computing:
  $$\text{Projected Median Value} = \text{median}(\{P(N_1), \dots, P(N_k)\})$$
  $$\text{Maturation Upside} = \frac{\text{Projected Median Value}}{P(C)}$$

---

## Quickstart & Usage

### 1. Ingest Raw Data to Tabular Parquet/CSV
```bash
python3 scripts/ingest_to_tabular.py
```

### 2. Query a Modern Card by ID or Name
```bash
# Query by specific card ID (e.g. 151 Charizard ex)
python3 scripts/similarity_model.py --card_id "sv03.5-199"

# Query by Pokémon name (e.g. Gengar, Zapdos, Tinkaton)
python3 scripts/similarity_model.py --card "Zapdos ex"
```

### 3. Scan Modern Sets for Top Undervalued Gems
```bash
python3 scripts/similarity_model.py --top-gems 10
```

---

## Roadmap

- [ ] **Feature Distinctiveness**: Introduce elemental typing (`Fire`, `Psychic`), Generation ID (`Gen 1–9`), set position ratio, and full-art frame flags to eliminate tie-collisions between unrelated cards.
- [ ] **Interactive Web UI**: Streamlit application with dropdown card selectors, high-res card image previews (`tcgdex` assets), and filterable top-gems dashboard.
- [ ] **Data Refresh Automation**: GitHub Action / cron job to pull new Saturday pricing snapshots and auto-regenerate parquet tables.

