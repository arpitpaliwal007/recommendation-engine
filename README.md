# Recommendation & Ranking Engine

A complete implicit-feedback recommender with candidate retrieval, personalized ranking,
cold-start fallbacks, diversity-aware reranking, and temporal offline evaluation.

## Highlights

- Synthetic user-item interaction generator with latent preferences
- Popularity baseline and item-item collaborative filtering
- Recency-weighted user profiles and cosine similarity retrieval
- Category-aware maximal-marginal-relevance reranking
- Leave-last-out evaluation with Recall@K, NDCG@K, coverage, and diversity
- FastAPI serving contract, tests, Docker, and CI

## Quickstart

```bash
python scripts/run_demo.py
python -m unittest discover -s tests
```

Results and a serializable catalog artifact are written to `artifacts/`.

Optional API:

```bash
pip install -e '.[api]'
uvicorn recommender.api:app --reload --port 8001
```

## Evaluation

For every user, the most recent interaction is held out. Models see only earlier events;
therefore future clicks cannot leak into item similarities. Reported metrics include both
relevance and catalog health because an accurate recommender that repeatedly serves the
same items is not a useful product.

## Architecture

```text
events -> temporal split -> item similarity retrieval -> personalized score
                                                   -> diversity reranker -> API
```

## Next experiments

Add implicit ALS and a two-tower retrieval model, then use the existing evaluator to
compare Recall@K, cold-start quality, catalog coverage, model size, and p95 latency.

