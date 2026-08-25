from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


def generate_interactions(users: int = 160, items: int = 90, seed: int = 19) -> tuple[list[dict], dict[str, str]]:
    rng = random.Random(seed)
    categories = ["books", "fitness", "gaming", "home", "music", "tech"]
    catalog = {f"I{i:03d}": categories[i % len(categories)] for i in range(items)}
    events = []
    timestamp = 0
    for user_idx in range(users):
        user = f"U{user_idx:04d}"
        preferred = set(rng.sample(categories, 2))
        candidates = list(catalog)
        weights = [5.0 if catalog[item] in preferred else 0.55 for item in candidates]
        seen = set()
        for _ in range(rng.randint(12, 24)):
            available = [item for item in candidates if item not in seen]
            if not available:
                break
            available_weights = [weights[candidates.index(item)] for item in available]
            item = rng.choices(available, weights=available_weights, k=1)[0]
            seen.add(item)
            timestamp += rng.randint(1, 5)
            events.append({"user": user, "item": item, "timestamp": timestamp, "weight": 1.0})
    return events, catalog


def temporal_leave_one_out(events: list[dict]) -> tuple[list[dict], dict[str, str]]:
    by_user = defaultdict(list)
    for event in events:
        by_user[event["user"]].append(event)
    train, test = [], {}
    for user, history in by_user.items():
        ordered = sorted(history, key=lambda e: e["timestamp"])
        train.extend(ordered[:-1])
        test[user] = ordered[-1]["item"]
    return train, test


class ItemCFRecommender:
    def __init__(self, catalog: dict[str, str]):
        self.catalog = catalog
        self.user_items: dict[str, list[str]] = {}
        self.item_users: dict[str, set[str]] = {}
        self.popularity: Counter = Counter()

    def fit(self, events: list[dict]) -> "ItemCFRecommender":
        user_items = defaultdict(list)
        item_users = defaultdict(set)
        for event in sorted(events, key=lambda e: e["timestamp"]):
            user_items[event["user"]].append(event["item"])
            item_users[event["item"]].add(event["user"])
            self.popularity[event["item"]] += 1
        self.user_items = dict(user_items)
        self.item_users = dict(item_users)
        return self

    def similarity(self, left: str, right: str) -> float:
        a, b = self.item_users.get(left, set()), self.item_users.get(right, set())
        if not a or not b:
            return 0.0
        return len(a & b) / math.sqrt(len(a) * len(b))

    def recommend(self, user: str, k: int = 10, diversity: float = 0.18) -> list[tuple[str, float]]:
        history = self.user_items.get(user, [])
        seen = set(history)
        if not history:
            return [(item, float(score)) for item, score in self.popularity.most_common(k)]
        raw_scores = defaultdict(float)
        recent = history[-8:]
        for position, source in enumerate(recent):
            recency = 0.82 ** (len(recent) - position - 1)
            for candidate in self.catalog:
                if candidate not in seen:
                    raw_scores[candidate] += recency * self.similarity(source, candidate)
        if not any(raw_scores.values()):
            raw_scores.update({item: float(score) for item, score in self.popularity.items() if item not in seen})
        selected = []
        pool = dict(raw_scores)
        while pool and len(selected) < k:
            def objective(item: str) -> float:
                duplicate_category = sum(self.catalog[item] == self.catalog[chosen] for chosen, _ in selected)
                return pool[item] - diversity * duplicate_category
            best = max(pool, key=objective)
            selected.append((best, round(pool.pop(best), 6)))
        return selected


class PopularityRecommender(ItemCFRecommender):
    def recommend(self, user: str, k: int = 10, diversity: float = 0.0) -> list[tuple[str, float]]:
        seen = set(self.user_items.get(user, []))
        return [(item, float(score)) for item, score in self.popularity.most_common() if item not in seen][:k]


def ranking_metrics(model: ItemCFRecommender, test: dict[str, str], k: int = 10) -> dict:
    recalls, ndcgs, recommended, diversities = [], [], set(), []
    for user, target in test.items():
        recs = [item for item, _ in model.recommend(user, k)]
        recommended.update(recs)
        recalls.append(float(target in recs))
        rank = recs.index(target) + 1 if target in recs else None
        ndcgs.append(1 / math.log2(rank + 1) if rank else 0.0)
        diversities.append(len({model.catalog[item] for item in recs}) / max(len(recs), 1))
    return {
        f"recall@{k}": round(sum(recalls) / len(recalls), 4),
        f"ndcg@{k}": round(sum(ndcgs) / len(ndcgs), 4),
        "catalog_coverage": round(len(recommended) / len(model.catalog), 4),
        "category_diversity": round(sum(diversities) / len(diversities), 4),
        "evaluated_users": len(test),
    }
