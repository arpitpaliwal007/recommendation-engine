from fastapi import FastAPI, HTTPException

from .core import ItemCFRecommender, generate_interactions

events, catalog = generate_interactions()
model = ItemCFRecommender(catalog).fit(events)
app = FastAPI(title="Recommendation API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "users": len(model.user_items), "items": len(catalog)}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10) -> dict:
    if not 1 <= k <= 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")
    return {"user": user_id, "recommendations": [
        {"item": item, "category": catalog[item], "score": score}
        for item, score in model.recommend(user_id, k)
    ]}

