from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os

from .recommender import BookKNNRecommender, Recommendation

app = FastAPI(title="Book Recommendation KNN")

# CORS for local static UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender: Optional[BookKNNRecommender] = None


class RecommendRequest(BaseModel):
    title: str
    k: int = 5


class RecommendResponseItem(BaseModel):
    title: str
    distance: float


class RecommendResponse(BaseModel):
    query: str
    results: List[RecommendResponseItem]


@app.on_event("startup")
async def startup_event() -> None:
    global recommender
    books_csv = os.getenv("BOOKS_CSV")
    ratings_csv = os.getenv("RATINGS_CSV")

    recommender = BookKNNRecommender(
        books_csv=books_csv,
        ratings_csv=ratings_csv,
    )
    recommender.fit()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/suggest")
async def api_suggest(q: str, limit: int = 10) -> List[str]:
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return recommender.suggest(q, limit)


@app.post("/api/recommend", response_model=RecommendResponse)
async def api_recommend(req: RecommendRequest) -> RecommendResponse:
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        results = recommender.recommend(req.title, k=req.k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendResponse(
        query=req.title,
        results=[RecommendResponseItem(title=r.title, distance=r.distance) for r in results],
    )

# Mount static frontend if present
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "web")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
