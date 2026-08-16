from entity_resolution_agent import EntityResolutionAgent
from franchise_lineage_agent import FranchiseLineageAgent


def build_match_intelligence(analytics, fixture, resolver=None, lineage_agent=None, league=None):
    """Build deterministic match intelligence only from CricketAnalytics results."""
    resolver = resolver or EntityResolutionAgent()
    lineage_agent = lineage_agent or FranchiseLineageAgent()
    league = league or fixture.get("league")
    venues, teams = analytics.get_venue_team_options()
    team_a_resolution = resolver.resolve_team(fixture.get("team_a"), teams, league)
    team_b_resolution = resolver.resolve_team(fixture.get("team_b"), teams, league)
    venue_resolution = resolver.resolve_venue(fixture.get("venue"), venues, league)
    team_a = team_a_resolution["resolved"]
    team_b = team_b_resolution["resolved"]
    venue = venue_resolution["resolved"]

    coverage = {
        "team_a": team_a,
        "team_b": team_b,
        "venue": venue,
        "teams_ready": bool(team_a and team_b),
        "venue_ready": bool(venue),
        "team_a_resolution": team_a_resolution,
        "team_b_resolution": team_b_resolution,
        "venue_resolution": venue_resolution,
        "lineages": lineage_agent.provenance(league, [team_a, team_b]),
    }
    result = {
        "coverage": coverage,
        "head_to_head": None,
        "venue": None,
        "team_a_phases": None,
        "team_b_phases": None,
        "team_a_patterns": None,
        "team_b_patterns": None,
        "lineup_resolutions": [],
        "facts": [],
    }

    analytics_view = lineage_agent.analytics_view(analytics, league, [team_a, team_b])

    if fixture.get("lineups") and hasattr(analytics, "df"):
        available_players = sorted(set(
            analytics.df["batsman"].dropna().astype(str).tolist()
            + analytics.df["bowler"].dropna().astype(str).tolist()
        ))
        for lineup in fixture["lineups"]:
            result["lineup_resolutions"].append({
                "team": lineup.get("team"),
                "players": [
                    resolver.resolve_player(player, available_players, league)
                    for player in lineup.get("players", [])
                ],
            })

    if team_a and team_b:
        result["head_to_head"] = analytics_view.get_team_vs_team(team_a, team_b)
        result["team_a_phases"] = analytics_view.get_phase_analysis(team=team_a)
        result["team_b_phases"] = analytics_view.get_phase_analysis(team=team_b)
        result["team_a_patterns"] = analytics_view.get_winning_patterns(team_a, venue=venue)
        result["team_b_patterns"] = analytics_view.get_winning_patterns(team_b, venue=venue)

        overall = (result["head_to_head"] or {}).get("overall", {})
        if overall.get("completed"):
            result["facts"].append(
                {
                    "label": "Historical head-to-head",
                    "text": (
                        f"{team_a} won {overall.get('a_wins', 0)} and {team_b} won "
                        f"{overall.get('b_wins', 0)} of {overall['completed']} completed meetings."
                    ),
                    "sample_size": overall["completed"],
                    "source": "get_team_vs_team",
                    "scope": "verified_franchise_lineage",
                }
            )

    if venue:
        result["venue"] = analytics.get_venue_characteristics(venue)
        venue_result = result["venue"] or {}
        if venue_result.get("completed_matches"):
            result["facts"].append(
                {
                    "label": "Venue trend",
                    "text": (
                        f"Teams chasing won {venue_result.get('chase_win_pct', 0)}% of "
                        f"{venue_result['completed_matches']} completed matches at {venue}."
                    ),
                    "sample_size": venue_result["completed_matches"],
                    "source": "get_venue_characteristics",
                }
            )

    return result
