import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender.core import ItemCFRecommender, generate_interactions, ranking_metrics, temporal_leave_one_out


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        events, cls.catalog = generate_interactions(users=30, items=30)
        cls.train, cls.test = temporal_leave_one_out(events)
        cls.model = ItemCFRecommender(cls.catalog).fit(cls.train)

    def test_temporal_split_holds_out_one_per_user(self):
        self.assertEqual(len(self.test), 30)
        self.assertEqual(len(self.train) + len(self.test), len(self.train) + 30)

    def test_recommendations_exclude_seen_items(self):
        user = next(iter(self.test))
        recs = {item for item, _ in self.model.recommend(user, 10)}
        self.assertTrue(recs.isdisjoint(self.model.user_items[user]))

    def test_metrics_are_bounded(self):
        result = ranking_metrics(self.model, self.test, 5)
        self.assertTrue(0 <= result["recall@5"] <= 1)
        self.assertTrue(0 <= result["catalog_coverage"] <= 1)

    def test_unknown_user_gets_popular_items(self):
        self.assertEqual(len(self.model.recommend("new-user", 5)), 5)


if __name__ == "__main__":
    unittest.main()

