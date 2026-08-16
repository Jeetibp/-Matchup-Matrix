import unittest

from match_intelligence import build_match_intelligence


class FakeAnalytics:
    def get_venue_team_options(self):
        return ["Test Ground"], ["Team A", "Team B"]

    def get_team_vs_team(self, team_a, team_b):
        return {"overall": {"completed": 5, "a_wins": 3, "b_wins": 2}}

    def get_phase_analysis(self, team=None):
        return {"team": team}

    def get_winning_patterns(self, team, venue=None):
        return {"team": team, "venue": venue}

    def get_venue_characteristics(self, venue):
        return {"completed_matches": 10, "chase_win_pct": 60.0}


class MatchIntelligenceTests(unittest.TestCase):
    def test_builds_traceable_facts_from_analytics_results(self):
        result = build_match_intelligence(
            FakeAnalytics(),
            {"team_a": "Team A", "team_b": "Team B", "venue": "Test Ground"},
        )
        self.assertTrue(result["coverage"]["teams_ready"])
        self.assertEqual(len(result["facts"]), 2)
        self.assertEqual(result["facts"][0]["source"], "get_team_vs_team")
        self.assertIn("3", result["facts"][0]["text"])
        self.assertEqual(result["facts"][1]["sample_size"], 10)

    def test_resolves_saint_abbreviation_for_cpl_teams(self):
        class CplAnalytics(FakeAnalytics):
            def get_venue_team_options(self):
                return ["Daren Sammy National Cricket Stadium"], ["St Lucia Kings", "Barbados Royals"]

        result = build_match_intelligence(
            CplAnalytics(),
            {
                "team_a": "Saint Lucia Kings",
                "team_b": "Barbados Royals",
                "venue": "Daren Sammy National Cricket Stadium, Gros Islet, St Lucia",
            },
        )
        self.assertEqual(result["coverage"]["team_a"], "St Lucia Kings")
        self.assertTrue(result["coverage"]["teams_ready"])
        self.assertEqual(result["facts"][0]["label"], "Historical head-to-head")

    def test_does_not_analyse_unresolved_teams(self):
        result = build_match_intelligence(
            FakeAnalytics(),
            {"team_a": "Unknown A", "team_b": "Unknown B", "venue": "Unknown"},
        )
        self.assertFalse(result["coverage"]["teams_ready"])
        self.assertEqual(result["facts"], [])


if __name__ == "__main__":
    unittest.main()