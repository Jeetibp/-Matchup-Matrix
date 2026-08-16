import unittest

from entity_resolution_agent import EntityResolutionAgent


class EntityResolutionAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = EntityResolutionAgent(
            player_aliases={"global": {"virat kohli": "V Kohli"}},
            entity_aliases={
                "teams": {"cpl": {"saint lucia kings": "St Lucia Kings"}},
                "venues": {"global": {}},
            },
        )

    def test_resolves_verified_player_alias(self):
        result = self.agent.resolve_player("Virat Kohli", ["V Kohli", "RG Sharma"], "ipl")
        self.assertEqual(result["resolved"], "V Kohli")
        self.assertEqual(result["method"], "verified_alias")
        self.assertEqual(result["confidence"], 1.0)

    def test_resolves_full_name_to_initial_and_surname(self):
        result = self.agent.resolve_player("Rohit Sharma", ["RG Sharma", "V Kohli"], "ipl")
        self.assertEqual(result["resolved"], "RG Sharma")
        self.assertEqual(result["method"], "initial_surname")

    def test_refuses_ambiguous_surname(self):
        result = self.agent.resolve_player("Sharma", ["RG Sharma", "I Sharma"], "ipl")
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["resolved"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_resolves_verified_team_alias(self):
        result = self.agent.resolve_team("Saint Lucia Kings", ["St Lucia Kings", "Barbados Royals"], "cpl")
        self.assertEqual(result["resolved"], "St Lucia Kings")
        self.assertEqual(result["status"], "resolved")


if __name__ == "__main__":
    unittest.main()