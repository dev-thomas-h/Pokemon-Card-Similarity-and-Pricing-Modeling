import os
import sys
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference import CardSimilarityEngine

app = FastAPI(
    title="Pokémon Card Similarity & Price Maturation API",
    description="Machine Learning similarity engine matching modern Pokémon cards to mature vintage collector counterparts.",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine globally once on startup
engine = CardSimilarityEngine()


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "modern_cards_indexed": len(engine.query_df),
        "vintage_cards_indexed": len(engine.train_df)
    }


@app.get("/api/options")
def get_options():
    return engine.get_filter_options()


@app.get("/api/cards/search")
def search_cards(
    q: str = Query("", description="Search term for card name or card ID"),
    set_id: Optional[str] = Query(None, description="Set ID filter"),
    rarity: Optional[str] = Query(None, description="Rarity filter"),
    limit: int = Query(24, ge=1, le=100)
):
    cards = engine.search_cards(query=q, set_id=set_id, rarity=rarity, limit=limit)
    return {"results": cards, "count": len(cards)}


@app.get("/api/similarity/{card_id}")
def get_similarity(
    card_id: str,
    k: int = Query(5, ge=1, le=20)
):
    result = engine.find_similar(card_id=card_id, k=k)
    if not result:
        raise HTTPException(status_code=404, detail=f"Card '{card_id}' not found.")
    return result


@app.get("/api/top-gems")
def get_top_gems(
    min_price: float = Query(3.0, ge=0.0),
    rarity: Optional[str] = Query(None),
    set_id: Optional[str] = Query(None),
    limit: int = Query(12, ge=1, le=50),
    k: int = Query(5, ge=1, le=20)
):
    gems = engine.get_top_gems(
        min_price=min_price,
        rarity_filter=rarity,
        set_filter=set_id,
        limit=limit,
        k=k
    )
    return {"gems": gems, "count": len(gems)}


# Mount React build output if present
import os
from starlette.staticfiles import StaticFiles

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("\nStarting Pokémon Card Appraisal Web App on http://localhost:8000 ...")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

