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
│   ├── cards_tabular.parquet     # Unified, typed tabular dataset (21,812 cards, 41 columns)
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
├── docs/                         # Deep-dive modeling & architecture guides
│   ├── README.md                 # Architecture overview & end-to-end card trace
│   ├── 01-data-ingestion.md      # Deep-dive into scripts/ingest_to_tabular.py
│   ├── 02-feature-engineering.md # Deep-dive into src/features.py (31 features)
│   └── 03-training-and-inference.md # Deep-dive into train.py, inference.py & API
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

## Model Architecture & Design Justifications (V2 Engine)

> **Detailed Modeling Documentation**: See **[`docs/README.md`](docs/README.md)** for a complete module-by-module walkthrough of the Python pipeline ([`01-data-ingestion.md`](docs/01-data-ingestion.md), [`02-feature-engineering.md`](docs/02-feature-engineering.md), and [`03-training-and-inference.md`](docs/03-training-and-inference.md)).

### 1. Why K-Nearest Neighbors (KNN)?
* **Real-World Appraisal Grounding**: Predicting the future price of speculative collectibles using black-box regression or deep learning yields single unexplainable numbers. KNN works identically to professional appraisal comps: *"Card X is projected to reach ~$737 because these 5 historical cards with the exact same prestige archetype sell for a median of $737."*
* **Era Asymmetry**: Vintage cards (Base Set through Black & White / early XY) have experienced 10–25 years of organic supply attrition and collector price discovery. Modern cards (Sun & Moon through Scarlet & Violet) are still experiencing active pack openings and short-term market fluctuations. KNN directly connects modern card traits to mature historical baselines.

### 2. Feature Selection: 31 Collector Prestige Dimensions (V2)
Traditional card attributes like **HP, attack power, retreat costs, and damage** have near-zero correlation with secondary market collector value. A 30 HP Magikarp or a 120 HP Base Set Charizard sells for hundreds of dollars despite being unplayable in modern tournaments.

To prevent unrelated chase cards from colliding onto identical feature vectors, the V2 feature engine (`src/features.py`) encodes cards across **31 domain-calibrated features** bounded in **`[0, 1]`**:

| Feature Group | Features (`FEATURE_COLS`) | Weight | Valuation & Distinctiveness Rationale |
| :--- | :--- | :---: | :--- |
| **Empirical Species Prestige** | `species_prestige` | **3.5** | Bayesian-smoothed historical log-price premium (`k = 3.0`) of the exact Pokémon (`dex_id`) over its rarity-tier median, computed **strictly on `< 2016` vintage cards** (zero modern leakage). Differentiates *Charizard* from *Venusaur* and *Umbreon* from *Leafeon*. |
| **Reconstructed Rarity & Set Position** | `rarity_norm` (4.5), `is_secret_rare` (1.8), `set_position_ratio` (1.4), `set_size_norm` (0.5) | **0.5–4.5** | Reconstructs true collector rarity across 25+ years — rescuing 1999–2007 Gold Stars (`" Star"`), `"Shining "` cards, Crystal Secret Rares, e-Card `H01–H32` holos, and WOTC Holo-Only Rares (`4.5` tier) that raw TCGdex data labeled simply as `"Rare"`. |
| **Subject & Trainer Archetypes** | `arch_supporter` (3.0), `arch_item` (2.5), `arch_energy` (2.5), `arch_legend` (2.2), `arch_starter` (2.2), `arch_fav` (2.0), `arch_regular` (1.5) | **1.5–3.0** | Separates human Character Supporter cards (*Miriam*, *Iono*, *Sabrina*) from Item/Tool/Stadium Trainer cards (*Nest Ball*, *Ultra Ball*) and maps mutually exclusive Pokédex archetypes. |
| **Generation, Mechanics & Evolution** | `is_fully_evolved` (2.5), `gen_nostalgia_score` (1.8), `is_special_mechanic` (1.8), `stage_norm` (1.2), `is_character_or_owner` (1.0), `gen_relative_dex` (0.6) | **0.6–2.5** | Encodes Generation 1–9 nostalgia decay, rule-box / character gallery mechanics, and biological evolution finality (`is_fully_evolved`) derived from `evolve_from` chains so final evolutions (*Charizard ex*, *Alakazam ex*) never match unevolved basic secret rares (*Charmander*, *Ralts*). |
| **Elemental Typing Affinity** | 9 `type_*` flags: `type_fire` (1.2), `type_darkness` (1.0), `type_dragon` (1.0), `type_psychic` (1.0), `type_lightning` (1.0), `type_water` (1.0), `type_colorless` (0.8), `type_grass` (0.8), `type_other_elem` (0.8) | **0.8–1.2** | Separates collector sub-markets by elemental affinity so Fire, Darkness, Dragon, Psychic, and Lightning chase cards match within their respective elemental lineages. |
| **Artist Prestige & Finishes** | `has_holo` (1.8), `artist_prestige` (1.4), `has_first_edition` (0.8), `has_reverse` (0.5) | **0.5–1.8** | Combines empirical vintage illustrator log-price residuals (`k = 5.0`) with curated adjustments for `RENOWNED_ARTISTS` (`+0.35`) and `CGI_STUDIOS` (`-0.35`), and repairs missing holo flags via pricing variants and rarity tiers. |

### 3. Why Unit-Range `[0, 1]` Scaling Beats `StandardScaler`
* **The `StandardScaler` Pitfall**: Dividing features by their standard deviation ($\sigma$) inflates sparse features (`secret_depth`, rare types, or 1st-Edition flags where $\sigma \approx 0.03\text{–}0.10$) by **10x–30x**, drowning out core signals like `species_prestige` and `rarity_norm`.
* **Unit-Range Normalization**: Every feature in `FEATURE_COLS` is explicitly scaled to **`[0, 1]`** before multiplying directly by `FEATURE_WEIGHTS`. A feature with weight `3.5` has an exact maximum directional leverage of `3.5` in cosine space.
* **Weighted Cosine Metric**:
  $$D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### 4. V1 vs. V2 Empirical Benchmark Summary
Evaluated across 12,818 modern cards and 5-fold out-of-sample cross-validation on 7,964 vintage cards:

| Metric | V1 Baseline (13 Features) | V2 Collector Engine (31 Features) | Improvement |
| :--- | :---: | :---: | :--- |
| **Unique Modern Vectors** | `358 / 12,818` (`2.8%`) | **`12,694 / 12,818` (`99.0%`)** | **+35.5x granularity** (eliminates tie collisions) |
| **Vintage Pool Used Across Chase Cards** | `230` cards | **`913` cards** | **+4.0x diversity** |
| **Vintage CV Spearman Rank Correlation ($\rho$)** | `0.6383` | **`0.8259`** | **+0.1876 rank correlation** |
| **Vintage CV Log-MAE** | `0.7859` | **`0.5596`** | **-28.8% pricing error** |
| **Vintage CV Within 2x Actual Price** | `57.2%` | **`74.2%`** | **+17.0 percentage points** |

### 5. Training & Inference
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

### 2. Train Model & Export Serialized Artifacts
```bash
python3 -m src.train
```

### 3. Query a Modern Card by ID or Name (CLI)
```bash
# Query by specific card ID (e.g. 151 Charizard ex)
python3 scripts/similarity_model.py --card_id "sv03.5-199"

# Query by Pokémon name (e.g. Gengar, Zapdos, Miriam)
python3 scripts/similarity_model.py --card "Zapdos ex"
```

### 4. Scan Modern Sets for Top Undervalued Gems (CLI)
```bash
python3 scripts/similarity_model.py --top-gems 10
```

---

## Roadmap

- [x] **Interactive Full-Stack Web UI**: FastAPI + React/Vite glassmorphic web application with instant search, set/rarity filters, high-res artwork, and Top Gems discovery scanner.
- [x] **V2 Feature Distinctiveness & Vintage Rarity Reconstruction**: 31 unit-range `[0, 1]` collector features including Bayesian vintage species/artist prestige, 1999–2010 Gold Star / Shining / Crystal rarity reconstruction, biological evolution finality, Supporter vs. Item Trainer split, and elemental typing.
- [ ] **Time-Series Maturation Curves**: Incorporate historical dated snapshots in `data/pricing/{YYYY-MM-DD}.json` to model price velocity and post-release depreciation/appreciation curves (J-curve dynamics).
- [ ] **Data Refresh Automation**: GitHub Action / cron job to pull new Saturday pricing snapshots, regenerate parquet tables, and re-run `python3 -m src.train`.


