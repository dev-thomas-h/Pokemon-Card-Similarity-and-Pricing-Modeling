# AGENTS.md — Project Rules & Session Context

## 1. Operational Rules & Agent Directives

### Version Control & Git Policy (Strict)
* **Do NOT commit changes at the end of responses**: Never run `git commit`, `git push`, or create commits automatically after completing a task. Always leave modifications in the working tree so the user can inspect and commit them manually, unless the user explicitly asks you to commit in the current prompt.
* **No destructive git commands**: Never run `git reset --hard`, `git clean -fd`, `git checkout -- .`, or force pushes without explicit user confirmation.
* **Respect ignored data**: Never force-add gitignored directories (such as `pokemon-cards/` image caches or `frontend/node_modules/`).

### Workflow & Code Quality Best Practices
* **Working directory awareness**: Ensure commands are executed from the `pkm-card-data/` repository root (if the session starts in a parent workspace directory, `cd pkm-card-data` before running Python, npm, or git commands).
* **Read before modifying**: Always read the target source files before proposing or applying edits. Do not guess function signatures, column names, or API response schemas.
* **Surgical, minimal edits**: Keep changes focused on the user's request. Avoid unrelated refactoring, reformatting, or adding unnecessary dependencies.
* **Preserve documentation & comments**: Maintain existing docstrings, inline comments, and architectural explanations unless they are directly invalidated by your changes.
* **Keep artifacts synchronized**:
  * If you modify feature engineering in [`src/features.py`](src/features.py) or training logic in [`src/train.py`](src/train.py), re-run `python3 -m src.train` so serialized artifacts in `models/` stay in sync, and verify results with `python3 scripts/similarity_model.py`.
  * If you modify the React UI in [`frontend/src/`](frontend/src), run `npm run build` inside `frontend/` so the production static bundle in `frontend/dist/` (served by FastAPI) stays updated.
* **Clarify ambiguity early**: When requirements or modeling trade-offs have multiple valid interpretations, ask concise clarifying questions before making large architectural changes.

### Modeling & Domain Guardrails (Must Not Violate)
* **Zero price leakage into modern cards**: Empirical prestige priors (`species_prestige`, `artist_prestige`) must be fitted **exclusively** on `< 2016` vintage cards. Never leak `2016+` prices into feature construction.
* **Unit-range scaling only (`[0, 1]`)**: Never apply `StandardScaler(with_std=True)` to the feature matrix. Dividing by standard deviation ($\sigma$) inflates sparse binary/tail indicators (`secret_depth`, rare types, 1st Edition) by 10x–30x and drowns out `species_prestige` and `rarity_norm`. Every feature in `FEATURE_COLS` must be explicitly normalized to `[0, 1]` in [`src/features.py`](src/features.py) and weighted directly via `FEATURE_WEIGHTS` (`StandardScaler(with_mean=False, with_std=False)`).
* **Collectibility over gameplay**: Do not introduce battle/gameplay stats (`hp`, `damage`, `retreat`, `attacks`) into similarity vectors; they have near-zero correlation with secondary market collector value.

---

## 2. Project Overview

**Pokémon Card Collector Similarity & Price Maturation Engine (V2)**
* **Core Goal**: Predict the long-term price equilibrium of modern Pokémon cards (`2016+`) by matching them to empirically mature vintage counterparts (`< 2016`) across 31 collector prestige dimensions.
* **Core Premise**: Vintage cards have experienced 10–25 years of organic supply attrition and collector price discovery. Modern chase cards (e.g., Special Illustration Rares) share collector archetypes with vintage chase cards (e.g., Secret Rares, Gold Stars, Shining cards, Crystal Pokémon). Finding these vintage "twins" provides an empirical appraisal of what modern cards can mature to over time.

---

## 3. Repository Architecture

```text
pkm-card-data/
├── AGENTS.md                     # Model-agnostic agent instructions & session context
├── README.md                     # Public documentation & dataset specifications
├── run_app.sh                    # Full-stack launcher script (FastAPI + static React UI)
│
├── data/
│   ├── cards.json                # Raw catalog dump from TCGdex (nested: set_id -> card_id)
│   ├── sets.json                 # Raw set list from TCGdex (206 sets)
│   ├── sets_metadata.json        # Cached set release dates, series, and officialCardCount
│   ├── cards_tabular.parquet     # Unified typed table (21,812 cards, 40 base columns)
│   ├── cards_tabular.csv         # CSV export for inspection
│   └── pricing/
│       ├── latest.json           # Latest complete market price snapshot (TCGplayer USD, Cardmarket EUR)
│       ├── {YYYY-MM-DD}.json     # Historical pricing snapshots (weekly Saturdays)
│       └── status.json           # Per-day snapshot completion metadata
│
├── models/                       # Pre-computed, serialized ML artifacts (<5ms inference)
│   ├── knn_model.joblib          # Fitted NearestNeighbors index (metric='cosine', k=10)
│   ├── scaler.joblib             # Unit-range identity scaler (with_mean=False, with_std=False)
│   ├── train_cards.parquet       # Vintage cards baseline (<2016 with prices, 7,964 cards)
│   ├── query_cards.parquet       # Modern cards catalog (>=2016, 12,818 cards with CDN image URLs)
│   ├── query_vectors.npy         # Pre-transformed 31-feature weighted matrix for O(1) lookup
│   └── model_meta.json           # Model training timestamp and feature metadata
│
├── src/                          # Modular production ML pipeline & backend
│   ├── features.py               # V2 31-feature engineering engine, rarity reconstruction & weights
│   ├── train.py                  # Training pipeline & artifact serialization
│   ├── inference.py              # Sub-millisecond similarity engine & top-gems scanner
│   └── api.py                    # FastAPI REST backend & static React app mount
│
├── frontend/                     # Modern React 18 + Vite 5 Web Application
│   ├── src/
│   │   ├── App.jsx               # Interactive Card Matcher & Top Gems UI
│   │   └── index.css             # Glassmorphic dark theme, holo animations, responsive CSS
│   ├── dist/                     # Pre-built production bundle (served directly by FastAPI)
│   ├── vite.config.js            # Vite proxy config (/api -> http://localhost:8000)
│   └── package.json              # Node dependencies (React 18, Vite 5, lucide-react)
│
├── docs/                         # Deep-dive modeling & code justification guides
│   ├── README.md                 # Architecture overview & end-to-end card trace
│   ├── 01-data-ingestion.md      # Deep-dive into scripts/ingest_to_tabular.py
│   ├── 02-feature-engineering.md # Deep-dive into src/features.py (31 features)
│   └── 03-training-and-inference.md # Deep-dive into train.py, inference.py & API
│
└── scripts/                      # Standalone CLI tools & data ingestion
    ├── ingest_to_tabular.py      # Ingestion script (JSONs -> Parquet/CSV)
    └── similarity_model.py       # Standalone CLI evaluator wrapping CardSimilarityEngine
```

---

## 4. Essential Commands

All commands below assume working directory `pkm-card-data/`:

### Run the Application
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
npm run dev      # Vite dev server on http://localhost:5173 (proxies /api to :8000)
npm run build    # Rebuild production bundle into frontend/dist/
```

---

## 5. Machine Learning Architecture (V2 Engine)

### 31 Collector Dimensions (`src/features.py`)
1. **Empirical Vintage Species & Artist Prestige (Zero Data Leakage)**:
   * **`species_prestige` (Weight: 3.5)**: Bayesian-smoothed log-price residual (`k_smooth = 3.0`) of each Pokémon species (`dex_id`) relative to its `rarity_tier` median, computed **exclusively on `< 2016` vintage cards**. Unseen Gen 7–9 species fall back to their archetype prior (`arch_priors`).
   * **`artist_prestige` (Weight: 1.4)**: Bayesian-smoothed vintage illustrator log-price premium (`k_smooth = 5.0`) adjusted by curated `RENOWNED_ARTISTS` (`+0.35`) and `CGI_STUDIOS` (`-0.35`) sets.
2. **Reconstructed Rarity & Set Position (Fixing the 1999–2010 TCGdex Blindspot)**:
   * Reconstructs true collector tiers (`rarity_tier` `1.0–5.0` and `rarity_norm` in `[0, 1]`, Weight: **4.5**) where raw TCGdex labeled 1999–2007 grails simply as `"Rare"`:
     * **Tier 5.0**: Secret Rares (`local_id > officialCardCount`), Crystal Secret Rares (Aquapolis `145–147`, Skyridge `145–150`), Gold Stars (`" Star"`), `"Shining "` cards, e-Card `H01`–`H32` holos, `LEGEND`, Full-Art / Holo Vintage Supporters, Modern `Special illustration rare`, `Hyper rare`.
     * **Tier 4.5**: 1999–2003 WOTC Holo-Only Rares (`yr <= 2003` with `has_holo` and `not has_normal`).
     * **Tier 4.0**: Vintage `ex`, `Rare Holo LV.X`, `Rare PRIME`, `Shiny rare`, Modern `Illustration rare`, `Ultra Rare`.
     * **Tier 3.5**: 2004–2010 Mid-Era Holo-Only Rares (`has_holo` and `not has_normal`).
     * **Tier 3.0**: Standard `Rare Holo`, Modern `Double rare`, `Holo Rare V/VMAX/VSTAR`, `Radiant Rare`, `ACE SPEC Rare`.
     * **Tier 2.0**: Regular non-holo `Rare`, standard `Promo`.
     * **Tier 1.0**: `Uncommon`, `Common`.
   * Supporting rarity & set signals: **`is_secret_rare` (1.8)**, **`set_position_ratio` (1.4)**, **`set_size_norm` (0.5)**.
3. **Biological Evolution Finality, Card Stage & Special Mechanics**:
   * **`is_fully_evolved` (Weight: 2.5)**: Derives pre-evolution Dex IDs from `Stage1`/`Stage2` `evolve_from` chains so evolved Pokémon printed as Basic mechanics (*Charizard ex*, *Zapdos ex*) never match unevolved basic secret rares (*Charmander*, *Ralts*).
   * **`is_special_mechanic` (Weight: 1.8)** & **`is_character_or_owner` (Weight: 1.0)**: Captures rule-box mechanics (`ex`, `EX`, `GX`, `V`, `VMAX`, `VSTAR`, `LV.X`, `Prime`, `Star`, `Shining`, `Radiant`) and character/gallery subsets (`'s`, `Dark `, `Light `, `tg/gg/rc`, `IR/SIR`).
   * **`stage_norm` (Weight: 1.2)**: Normalized progression from Basic (`0.25`) to Stage 2 (`0.75`) and Special/Mega/VMAX (`1.0`).
4. **Subject & Trainer Subtype Archetypes**:
   * **`arch_legend` (2.2)**, **`arch_starter` (2.2)**, **`arch_fav` (2.0)**, **`arch_regular` (1.5)**, **`arch_energy` (2.5)**.
   * **Trainer Split**: Splits Trainer cards into **`arch_supporter` (3.0)** and **`arch_item` (2.5)** so modern SIR Supporters (*Miriam*, *Iono*) match vintage Full-Art Supporters (*Bianca*, *Cheren*, *Sabrina*) instead of Gold Item cards.
5. **Generation Nostalgia & Elemental Typing**:
   * **`gen_nostalgia_score` (1.8)** & **`gen_relative_dex` (0.6)**: Captures Gen 1 (Kanto) and Gen 2 (Johto) collector premium and intra-generation Pokédex position.
   * **9 Elemental Type Flags (`type_*`)**: `type_fire` (**1.2**), `type_darkness` (**1.0**), `type_dragon` (**1.0**), `type_psychic` (**1.0**), `type_lightning` (**1.0**), `type_water` (**1.0**), `type_colorless` (**0.8**), `type_grass` (**0.8**), and `type_other_elem` (**0.8**, grouping `Fighting`, `Metal`, `Fairy`).
6. **Repaired Foil Finishes**:
   * **`has_holo` (1.8)**, **`has_first_edition` (0.8)**, **`has_reverse` (0.5)**: Repairs missing `has_holo` flags in Black & White / XY sets (e.g., *Plasma Storm Charizard* `bw8-136`) using `pricing_variant` and `rarity_tier >= 3.0`.

### Distance Metric & V2 Benchmarks
* **Weighted Cosine Distance**: $D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
* **Unique Modern Vectors**: `358 / 12,818` (`2.8%`) $\rightarrow$ **`12,694 / 12,818` (`99.0%`)**
* **Unique Vintage Cards Used Across Modern Chase Cards**: `230` $\rightarrow$ **`913` (4.0x diversity)**
* **Vintage 5-Fold CV Spearman Correlation ($\rho$)**: `0.6383` $\rightarrow$ **`0.8259`**
* **Vintage 5-Fold CV Log-MAE**: `0.7859` $\rightarrow$ **`0.5596` (-28.8% error)**
* **Vintage 5-Fold CV Within 2x Actual Price**: `57.2%` $\rightarrow$ **`74.2%`**

---

## 6. Active Backlog & Next Steps

1. **Time-Series Price Velocity & Maturation Curves**:
   * Use historical snapshots in `data/pricing/{YYYY-MM-DD}.json` to compute short-term price momentum, volatility, and post-release depreciation/appreciation curves (J-curve dynamics).
2. **Set Supply & Era Scarcity Multipliers**:
   * Incorporate era-level print-run estimates or set-level median chase values to calibrate maturation multipliers across eras (e.g., late-WOTC e-Card scarcity vs. XY era supply).
3. **Data Refresh Automation**:
   * Upstream price snapshots in `data/pricing/` update weekly on Saturdays. Build an update workflow that pulls the newest Saturday snapshot, runs `scripts/ingest_to_tabular.py`, and runs `python3 -m src.train` to re-index the model.

---

## 7. Key Files Reference
* [`docs/README.md`](docs/README.md) — Deep-dive modeling & architecture documentation index
* [`src/features.py`](src/features.py) — V2 31-feature transformation engine & weight dictionary
* [`src/train.py`](src/train.py) — Training script that exports serialized artifacts to `models/`
* [`src/inference.py`](src/inference.py) — Real-time inference engine (`CardSimilarityEngine`)
* [`src/api.py`](src/api.py) — FastAPI REST endpoints
* [`frontend/src/App.jsx`](frontend/src/App.jsx) — React interface
* [`run_app.sh`](run_app.sh) — Quickstart launcher
