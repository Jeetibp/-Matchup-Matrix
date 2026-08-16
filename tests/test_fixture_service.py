import json
import tempfile
import unittest
from pathlib import Path

from fixture_service import FixtureService, detect_supported_league


class FakeProvider:
    configured = True

    def __init__(self, fail=False, squad=None, has_squad=False):
        self.fail = fail
        self._squad = squad
        self._has_squad = has_squad

    def fetch_matches(self):
        if self.fail:
            raise RuntimeError("provider offline")
        return (
            [
                {"id": "supported", "series_id": "1", "teams": ["Team A", "Team B"], "date": "2026-08-15"},
                {"id": "international", "series_id": "2", "teams": ["India", "Australia"], "date": "2026-08-15"},
            ],
            {"1": "Caribbean Premier League 2026", "2": "India tour of Australia 2026"},
        )

    def fetch_match(self, match_id):
        return {
            "id": match_id,
            "teams": ["Team A", "Team B"],
            "status": "Team A 42/1",
            "seriesName": "Caribbean Premier League 2026",
            "hasSquad": self._has_squad,
        }

    def fetch_squad(self, match_id):
        if self.fail:
            return []
        return self._squad or []

    def fetch_series_squad(self, series_id):
        if self.fail:
            raise RuntimeError("provider offline")
        return [
            {"teamName": "Team A", "players": [{"name": "P1", "role": "batter"}]},
            {"teamName": "Team B", "players": [{"name": "P2", "role": "bowler"}]},
        ]


class FixtureServiceTests(unittest.TestCase):
    def test_detects_only_explicit_supported_leagues(self):
        self.assertEqual(detect_supported_league("Women's Big Bash League 2026"), "wbb")
        self.assertEqual(detect_supported_league("Indian Premier League 2026"), "ipl")
        self.assertEqual(detect_supported_league("Caribbean Premier League 2026"), "cpl")
        self.assertIsNone(detect_supported_league("Womens Caribbean Premier League 2026"))
        self.assertIsNone(detect_supported_league("India tour of Australia 2026"))

    def test_filters_provider_results_to_supported_leagues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_file = root / "fixtures.json"
            fixture_file.write_text('{"matches": []}', encoding="utf-8")
            service = FixtureService(
                provider=FakeProvider(),
                fixture_file=fixture_file,
                cache_file=root / "cache.json",
                fixture_ttl=0,
            )
            matches = service.list_matches("2026-08-15")
            self.assertEqual([match["id"] for match in matches], ["supported"])
            self.assertEqual(matches[0]["league"], "cpl")

    def test_filters_matches_by_selected_league(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_file = root / "fixtures.json"
            fixture_file.write_text(
                json.dumps({
                    "matches": [
                        {"fixture_id": "cpl-match", "league": "cpl", "team_a": "A", "team_b": "B", "match_date": "2026-08-16"},
                        {"fixture_id": "hundred-match", "league": "the100", "team_a": "C", "team_b": "D", "match_date": "2026-08-16"},
                    ]
                }),
                encoding="utf-8",
            )
            service = FixtureService(
                provider=FakeProvider(fail=True),
                fixture_file=fixture_file,
                cache_file=root / "cache.json",
                fixture_ttl=0,
            )
            matches = service.list_matches("2026-08-16", league="the100")
            self.assertEqual([match["id"] for match in matches], ["hundred-match"])

    def test_excludes_placeholder_teams(self):
        service = FixtureService(provider=FakeProvider())
        match = service.normalize(
            {
                "id": "future-final",
                "seriesName": "The Hundred 2026",
                "teams": ["Tbc", "Tbc"],
                "date": "2026-08-16",
            }
        )
        self.assertIsNone(match)

    def test_uses_stale_cache_when_provider_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_file = root / "cache.json"
            cache_file.write_text(
                json.dumps({"matches": [{"id": "cached", "league": "ipl", "team_a": "A", "team_b": "B", "date": "2026-08-15"}]}),
                encoding="utf-8",
            )
            service = FixtureService(
                provider=FakeProvider(fail=True),
                fixture_file=root / "fixtures.json",
                cache_file=cache_file,
                fixture_ttl=0,
            )
            self.assertEqual(service.list_matches("2026-08-15")[0]["id"], "cached")

    def test_get_series_squad_fetches_and_caches_for_24h(self):
        class SpyProvider(FakeProvider):
            squad_calls = 0
            def fetch_series_squad(self, series_id):
                self.squad_calls += 1
                return [
                    {"teamName": "Team A", "players": [{"name": "P1"}]},
                    {"teamName": "Team B", "players": [{"name": "P2"}]},
                ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spy = SpyProvider()
            service = FixtureService(
                provider=spy,
                fixture_file=root / "fixtures.json",
                cache_file=root / "cache.json",
                series_squad_ttl=86400,
            )
            first = service.get_series_squad("series-1")
            second = service.get_series_squad("series-1")
            self.assertEqual(len(first), 2)
            self.assertEqual(second, first)
            self.assertEqual(spy.squad_calls, 1)  # second call served from cache

    def test_get_series_squad_returns_empty_without_series_id(self):
        service = FixtureService(provider=FakeProvider())
        self.assertEqual(service.get_series_squad(""), [])
        self.assertEqual(service.get_series_squad(None), [])

    def test_squad_to_lineups_normalizes_match_squad_payload(self):
        service = FixtureService(provider=FakeProvider())
        squad = [
            {"team": "Team A", "players": [{"name": "Player One"}, {"name": "Player Two"}]},
            {"team": "Team B", "players": [{"name": "Player Three"}, {"name": "Player Four"}]},
        ]
        lineups = service._squad_to_lineups(squad)
        self.assertEqual(len(lineups), 2)
        self.assertEqual(lineups[0]["team"], "Team A")
        self.assertEqual(lineups[0]["players"], ["Player One", "Player Two"])
        self.assertTrue(lineups[0]["confirmed"])
        # Empty squad → empty lineups
        self.assertEqual(service._squad_to_lineups([]), [])
        # Bad shape → empty, no crash
        self.assertEqual(service._squad_to_lineups(None), [])
        self.assertEqual(service._squad_to_lineups([{"team": "T", "players": "not a list"}]), [])

    def test_live_match_fills_lineups_from_squad_when_hasSquad_true(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_file = root / "cache.json"
            cache_file.write_text(
                json.dumps({"matches": [{
                    "id": "squad-match", "league": "cpl", "team_a": "Team A",
                    "team_b": "Team B", "date": "2026-08-15",
                }]}),
                encoding="utf-8",
            )
            service = FixtureService(
                provider=FakeProvider(
                    has_squad=True,
                    squad=[{"team": "Team A", "players": [{"name": "P1"}, {"name": "P2"}]}],
                ),
                fixture_file=root / "fixtures.json",
                cache_file=cache_file,
                fixture_ttl=0,
                live_ttl=0,  # always treat live cache as expired so we exercise the fetch path
            )
            match = service.live_match("squad-match")
            self.assertIsNotNone(match)
            self.assertEqual(match["id"], "squad-match")
            self.assertEqual(len(match["lineups"]), 1)
            self.assertEqual(match["lineups"][0]["players"], ["P1", "P2"])

    def test_live_match_does_not_call_squad_when_already_have_lineups(self):
        # When fixture already carries lineups (e.g. from local fixtures file)
        # the server should not waste a quota call on fetch_squad.
        class SpyProvider(FakeProvider):
            squad_calls = 0
            def fetch_squad(self, match_id):
                self.squad_calls += 1
                return []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_file = root / "cache.json"
            cache_file.write_text(
                json.dumps({"matches": [{
                    "id": "lined-up", "league": "cpl", "team_a": "A", "team_b": "B",
                    "date": "2026-08-15",
                    "lineups": [{"team": "A", "players": ["X"], "confirmed": True}],
                }]}),
                encoding="utf-8",
            )
            spy = SpyProvider(has_squad=True)
            service = FixtureService(
                provider=spy,
                fixture_file=root / "fixtures.json",
                cache_file=cache_file,
                fixture_ttl=0,
                live_ttl=0,
            )
            match = service.live_match("lined-up")
            self.assertEqual(spy.squad_calls, 0)
            self.assertEqual(match["lineups"][0]["players"], ["X"])


if __name__ == "__main__":
    unittest.main()