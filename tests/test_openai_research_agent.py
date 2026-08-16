import json
import unittest

from openai_research_agent import OpenAIResearchAgent


class OpenAIResearchAgentTests(unittest.TestCase):
    def test_requires_private_api_key(self):
        agent = OpenAIResearchAgent(api_key="")
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            agent.research_franchise("cpl", "Saint Lucia Kings", ["St Lucia Kings"])

    def test_builds_web_search_with_structured_output(self):
        agent = OpenAIResearchAgent(api_key="test-key")
        payload = agent.build_payload("cpl", "Saint Lucia Kings", ["St Lucia Zouks", "St Lucia Kings"])
        self.assertEqual(payload["tools"][0]["type"], "web_search")
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertFalse(payload["store"])

    def test_returns_cited_proposal_without_applying_it(self):
        proposal = {
            "canonical_name": "St Lucia Kings",
            "historical_names": ["St Lucia Zouks", "St Lucia Stars", "St Lucia Kings"],
            "confidence": "high",
            "evidence_summary": "Official history confirms the changes.",
            "conflicts": [],
            "needs_human_review": False,
        }
        response = {
            "model": "test-model",
            "usage": {"total_tokens": 50},
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"sources": [{"url": "https://example.com/history", "title": "History"}]},
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(proposal), "annotations": []}],
                },
            ],
        }
        agent = OpenAIResearchAgent(api_key="test-key", transport=lambda payload: response)
        result = agent.research_franchise("cpl", "Saint Lucia Kings", ["St Lucia Kings"])
        self.assertEqual(result["status"], "proposal_only")
        self.assertFalse(result["applied"])
        self.assertTrue(result["proposal"]["needs_human_review"])
        self.assertEqual(result["sources"][0]["url"], "https://example.com/history")

    def test_predict_probable_xi_returns_structured_xi(self):
        squad = [
            {"name": "Player One", "role": "batter"},
            {"name": "Player Two", "role": "bowler"},
        ]
        prediction = {
            "xi_team_a": ["Player One"],
            "xi_team_b": ["Player Two"],
            "rationale": "Based on form.",
            "confidence": "medium",
            "needs_human_review": True,
            "sources": ["https://example.com"],
        }
        response = {
            "model": "test-model",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(prediction), "annotations": []}],
                }
            ],
        }

        def transport(payload):
            self.assertEqual(payload["model"], "gpt-4o")
            self.assertFalse(payload["store"])
            self.assertNotIn("reasoning", payload)  # gpt-4o must not get reasoning.effort
            return response

        agent = OpenAIResearchAgent(api_key="test-key", model="gpt-4o", transport=transport)
        result = agent.predict_probable_xi(
            league="cpl",
            team_a="Saint Lucia Kings",
            team_b="Barbados Tridents",
            start_time_ist="2026-08-17 10:00 IST",
            hours_to_start=8,
            squad_a=squad,
            squad_b=squad,
            analytics_a={},
            analytics_b={},
        )
        self.assertEqual(result["xi_team_a"], ["Player One"])
        self.assertEqual(result["xi_team_b"], ["Player Two"])
        self.assertEqual(result["confidence"], "medium")

    def test_predict_probable_xi_24h_gate_returns_early_without_transport(self):
        agent = OpenAIResearchAgent(api_key="test-key", transport=lambda payload: self.fail("transport called"))
        result = agent.predict_probable_xi(
            league="cpl",
            team_a="A",
            team_b="B",
            start_time_ist="",
            hours_to_start=48,
            squad_a=[],
            squad_b=[],
            analytics_a={},
            analytics_b={},
        )
        self.assertEqual(result["xi_team_a"], [])
        self.assertIn("not meaningful", result["rationale"])

    def test_predict_probable_xi_adds_reasoning_for_reasoning_models(self):
        squad = [{"name": "Player One", "role": "batter"}]
        prediction = {
            "xi_team_a": ["Player One"],
            "xi_team_b": ["Player One"],
            "rationale": "",
            "confidence": "low",
            "needs_human_review": True,
            "sources": [],
        }
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(prediction), "annotations": []}],
                }
            ]
        }

        def transport(payload):
            self.assertEqual(payload["reasoning"], {"effort": "medium"})
            return response

        agent = OpenAIResearchAgent(api_key="test-key", model="gpt-5.5", transport=transport)
        result = agent.predict_probable_xi(
            league="cpl",
            team_a="A",
            team_b="B",
            start_time_ist="",
            hours_to_start=6,
            squad_a=squad,
            squad_b=squad,
            analytics_a={},
            analytics_b={},
        )
        self.assertEqual(result["xi_team_a"], ["Player One"])

    def test_analyze_match_prediction_returns_interpreted_report(self):
        report = {
            "summary": "Close game; Team A slight favourites on venue form.",
            "predicted_team_a_score": "170-180",
            "predicted_team_b_score": "160-170",
            "chase_or_defend": "Chasing favoured.",
            "key_batters": ["Player One (in form, big venue record)"],
            "key_bowlers": ["Player Two (economical, takes wickets at this venue)"],
            "player_assessments": [
                {"name": "Player One", "team": "Team A", "role": "batsman",
                 "predicted_runs": "40-55", "predicted_wickets": "0",
                 "confidence": "medium", "notes": "Good recent form."},
            ],
            "risks": ["Possible rain."],
        }
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(report), "annotations": []}],
                }
            ]
        }

        def transport(payload):
            self.assertEqual(payload["model"], "gpt-4o")
            self.assertFalse(payload["store"])
            self.assertNotIn("reasoning", payload)  # gpt-4o must not get reasoning.effort
            return response

        agent = OpenAIResearchAgent(api_key="test-key", model="gpt-4o", transport=transport)
        result = agent.analyze_match_prediction(
            league="the100",
            team_a="Team A",
            team_b="Team B",
            venue="Lord's, London",
            start_time_ist="",
            hours_to_start=8,
            probable_xi_a=["Player One"],
            probable_xi_b=["Player Two"],
            player_analytics=[{"name": "Player One", "role": "batsman"}],
            venue_stats={"completed_matches": 10},
            team_h2h={"overall": {"completed": 4}},
        )
        self.assertEqual(result["summary"], report["summary"])
        self.assertEqual(result["player_assessments"][0]["name"], "Player One")
        self.assertEqual(result["key_batters"], ["Player One (in form, big venue record)"])

    def test_analyze_match_prediction_24h_gate_returns_early(self):
        agent = OpenAIResearchAgent(api_key="test-key", transport=lambda payload: self.fail("transport called"))
        result = agent.analyze_match_prediction(
            league="cpl",
            team_a="A",
            team_b="B",
            venue="",
            start_time_ist="",
            hours_to_start=48,
            probable_xi_a=[],
            probable_xi_b=[],
            player_analytics=[],
            venue_stats={},
            team_h2h={},
        )
        self.assertEqual(result["summary"], "Match starts in 48.0h — analysis not meaningful yet.")


if __name__ == "__main__":
    unittest.main()