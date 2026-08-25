from __future__ import annotations

import json
from pathlib import Path

from .core import ItemCFRecommender, PopularityRecommender, generate_interactions, ranking_metrics, temporal_leave_one_out


def run_pipeline(output_dir: str | Path = "artifacts") -> dict:
    events, catalog = generate_interactions()
    train, test = temporal_leave_one_out(events)
    model = ItemCFRecommender(catalog).fit(train)
    report = ranking_metrics(model, test)
    baseline = ranking_metrics(PopularityRecommender(catalog).fit(train), test)
    report.update({f"popularity_{key}": value for key, value in baseline.items() if key != "evaluated_users"})
    report.update({"train_events": len(train), "catalog_items": len(catalog)})
    samples = {user: model.recommend(user, 5) for user in sorted(test)[:5]}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(report, indent=2))
    (target / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (target / "sample_recommendations.json").write_text(json.dumps(samples, indent=2))
    return report
