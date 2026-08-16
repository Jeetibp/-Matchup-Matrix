import unittest

import pandas as pd

from franchise_lineage_agent import FranchiseLineageAgent


class FakeAnalytics:
    pass


class FranchiseLineageAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = FranchiseLineageAgent({
            "cpl": {
                "St Lucia Kings": ["St Lucia Kings", "St Lucia Stars", "St Lucia Zouks"],
                "Barbados Royals": ["Barbados Royals", "Barbados Tridents"],
            }
        })

    def test_returns_verified_aliases(self):
        aliases = self.agent.aliases_for("cpl", "St Lucia Kings")
        self.assertEqual(aliases, ["St Lucia Kings", "St Lucia Stars", "St Lucia Zouks"])

    def test_canonicalizes_both_team_columns_without_mutating_source(self):
        analytics = FakeAnalytics()
        analytics.df = pd.DataFrame({
            "batting_team": ["St Lucia Zouks", "Barbados Tridents"],
            "bowling_team": ["Barbados Royals", "St Lucia Stars"],
        })
        view = self.agent.analytics_view(
            analytics,
            "cpl",
            ["St Lucia Kings", "Barbados Royals"],
        )
        self.assertEqual(view.df["batting_team"].tolist(), ["St Lucia Kings", "Barbados Royals"])
        self.assertEqual(view.df["bowling_team"].tolist(), ["Barbados Royals", "St Lucia Kings"])
        self.assertEqual(analytics.df["batting_team"].iloc[0], "St Lucia Zouks")

    def test_does_not_merge_unverified_franchises(self):
        mapping = self.agent.canonical_map("cpl", ["Jamaica Kingsmen"])
        self.assertEqual(mapping, {})


if __name__ == "__main__":
    unittest.main()