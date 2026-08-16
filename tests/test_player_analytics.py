import unittest

import pandas as pd

from cricket_analytics_core import CricketAnalytics
from entity_resolution_agent import EntityResolutionAgent


def make_sample_df():
    return pd.DataFrame([
        {"match_id": "m1", "start_date": "2026-08-01", "venue": "Venue A", "innings": 1,
         "batting_team": "Team X", "bowling_team": "Team Y", "batsman": "JC Buttler",
         "bowler": "Bowler One", "runs_of_bat": 40, "isBowlerWk": 0, "total_run": 40,
         "player_dismissed": "", "isFour": 3, "isSix": 2},
        {"match_id": "m1", "start_date": "2026-08-01", "venue": "Venue A", "innings": 1,
         "batting_team": "Team X", "bowling_team": "Team Y", "batsman": "Batter Two",
         "bowler": "Bowler One", "runs_of_bat": 2, "isBowlerWk": 0, "total_run": 4,
         "player_dismissed": "Batter Two", "isFour": 0, "isSix": 0},
        {"match_id": "m2", "start_date": "2026-08-02", "venue": "Venue A", "innings": 2,
         "batting_team": "Team Y", "bowling_team": "Team X", "batsman": "JC Buttler",
         "bowler": "Bowler Two", "runs_of_bat": 10, "isBowlerWk": 0, "total_run": 10,
         "player_dismissed": "JC Buttler", "isFour": 1, "isSix": 0},
    ])


class PlayerAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.analytics = CricketAnalytics.__new__(CricketAnalytics)
        self.analytics.df = make_sample_df()

    def test_get_player_form_builds_compact_summary(self):
        form = self.analytics.get_player_form("JC Buttler", last_n=5)
        self.assertIn(form["role"], ("batsman", "allrounder"))
        self.assertEqual(len(form["batting"]), 2)
        self.assertEqual(form["aggregate"]["runs"], 50)

    def test_get_player_vs_team_batsman(self):
        result = self.analytics.get_player_vs_team("JC Buttler", "Team Y", ptype="batsman")
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["runs"], 40)

    def test_get_player_at_venue(self):
        result = self.analytics.get_player_at_venue("JC Buttler", "Venue A", ptype="batsman")
        self.assertEqual(result["matches"], 2)
        self.assertEqual(result["runs"], 50)

    def test_get_player_innings_split(self):
        split = self.analytics.get_player_innings_split("JC Buttler", ptype="batsman")
        self.assertEqual(split["batting"][1]["runs"], 40)
        self.assertEqual(split["batting"][2]["runs"], 10)

    def test_get_player_match_analytics_bundles_signals(self):
        bundle = self.analytics.get_player_match_analytics("JC Buttler", team="Team Y", venue="Venue A")
        self.assertEqual(bundle["name"], "JC Buttler")
        self.assertIn(bundle["role"], ("batsman", "allrounder"))
        self.assertIn("form", bundle)
        self.assertIn("h2h", bundle)
        self.assertIn("venue", bundle)
        self.assertIn("innings_split", bundle)

    def test_resolver_maps_provider_name_to_csv_name(self):
        available = sorted(set(
            self.analytics.df["batsman"].dropna().astype(str).tolist()
            + self.analytics.df["bowler"].dropna().astype(str).tolist()
        ))
        resolver = EntityResolutionAgent()
        res = resolver.resolve_player("Jos Buttler", available, "the100")
        self.assertEqual(res["resolved"], "JC Buttler")
        self.assertGreaterEqual(res["confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()