import json
import os
from urllib.request import Request, urlopen


class OpenAIResearchAgent:
    """Research entity histories, returning proposals that require human approval."""

    API_URL = "https://api.openai.com/v1/responses"

    def __init__(self, api_key=None, model=None, timeout=90, transport=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        # Prefer OPENAI_RESEARCH_MODEL env; default to gpt-4o (stable, supports web_search).
        # gpt-5.x ids are only available on higher-tier plans; using gpt-4o here avoids
        # the 200k-token mismatch you saw earlier.
        self.model = model or os.environ.get("OPENAI_RESEARCH_MODEL", "gpt-4o")
        self.timeout = timeout
        self.transport = transport or self._http_transport

    @property
    def configured(self):
        return bool(self.api_key)

    def _http_transport(self, payload):
        request = Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MatchupMatrix/1.0",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _schema():
        return {
            "type": "json_schema",
            "name": "franchise_lineage_proposal",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical_name": {"type": "string"},
                    "historical_names": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence_summary": {"type": "string"},
                    "conflicts": {"type": "array", "items": {"type": "string"}},
                    "needs_human_review": {"type": "boolean"},
                },
                "required": [
                    "canonical_name",
                    "historical_names",
                    "confidence",
                    "evidence_summary",
                    "conflicts",
                    "needs_human_review",
                ],
            },
        }

    @staticmethod
    def _probable_xi_schema():
        """JSON schema for a probable-XI prediction response."""
        return {
            "type": "json_schema",
            "name": "probable_xi",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "xi_team_a": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Predicted playing XI for team A (11 names).",
                    },
                    "xi_team_b": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Predicted playing XI for team B (11 names).",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief rationale — role balance, key form players, venue factors.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Overall confidence in the prediction.",
                    },
                    "needs_human_review": {"type": "boolean"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any URLs/web_search results referenced.",
                    },
                },
                "required": [
                    "xi_team_a",
                    "xi_team_b",
                    "rationale",
                    "confidence",
                    "needs_human_review",
                ],
            },
        }

    def build_payload(self, league, provider_name, csv_team_names):
        context = {
            "league": league,
            "provider_name": provider_name,
            "csv_team_names": sorted(set(str(name) for name in csv_team_names)),
        }
        return {
            "model": self.model,
            "store": False,
            "max_output_tokens": 1200,
            "max_tool_calls": 4,
            "reasoning": {"effort": "low"},
            "tools": [{"type": "web_search", "search_context_size": "low"}],
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "instructions": (
                "You research cricket franchise identity and name history. Determine only whether names "
                "represent the same continuing franchise in the specified league. Prefer official league "
                "and team sources. Do not infer continuity from a shared city alone. Return a proposal, "
                "never a final database decision. Every proposal requires human review."
            ),
            "input": json.dumps(context, ensure_ascii=True),
            "text": {"format": self._schema(), "verbosity": "low"},
        }

    @staticmethod
    def _extract(response):
        text = None
        sources = {}
        for item in response.get("output", []):
            action = item.get("action") or {}
            for source in action.get("sources") or []:
                url = source.get("url")
                if url:
                    sources[url] = {"url": url, "title": source.get("title") or url}
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    continue
                text = content.get("text")
                for annotation in content.get("annotations") or []:
                    if annotation.get("type") == "url_citation" and annotation.get("url"):
                        sources[annotation["url"]] = {
                            "url": annotation["url"],
                            "title": annotation.get("title") or annotation["url"],
                        }
        if not text:
            raise RuntimeError("OpenAI research response contained no structured output")
        proposal = json.loads(text)
        proposal["needs_human_review"] = True
        return proposal, list(sources.values())

    def research_franchise(self, league, provider_name, csv_team_names):
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        payload = self.build_payload(league, provider_name, csv_team_names)
        response = self.transport(payload)
        proposal, sources = self._extract(response)
        return {
            "status": "proposal_only",
            "proposal": proposal,
            "sources": sources,
            "model": response.get("model") or self.model,
            "usage": response.get("usage") or {},
            "applied": False,
        }

    def predict_probable_xi(
        self,
        league,
        team_a,
        team_b,
        start_time_ist,
        hours_to_start,
        squad_a,
        squad_b,
        analytics_a,
        analytics_b,
        enable_web_search: bool = True,
    ):
        """Predict the most likely XI for each team.

        Uses the gpt-4o model with a web_search tool to look up last-minute
        team news. The prediction is only meaningful when the match starts
        within 24 hours (the caller must enforce this).

        Args:
            league: league name string.
            team_a: full team name string.
            team_b: full team name string.
            start_time_ist: formatted IST string (e.g. "2026-08-16 19:30 IST").
            hours_to_start: float, predicted hours until first ball. Must be 0..24
                for a useful prediction.
            squad_a: list of player dicts from CricketData /v1/series_squad
                (or [] if unavailable). Each player has: name, role, battingStyle, etc.
            squad_b: same for team B.
            analytics_a: dict from analytics.get_player_form() for team A captain.
            analytics_b: dict from analytics.get_player_form() for team B captain.
            enable_web_search: if True, the model may use its built-in web_search
                tool to look up latest team news.

        Returns:
            dict with keys: xi_team_a, xi_team_b, rationale, confidence,
            needs_human_review, sources.
        """
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        # 24h gate — the caller should enforce this, but we double-check here.
        if hours_to_start is not None and hours_to_start > 24:
            return {
                "xi_team_a": [],
                "xi_team_b": [],
                "rationale": f"Match starts in {hours_to_start:.1f}h — prediction not meaningful yet.",
                "confidence": "low",
                "needs_human_review": True,
                "sources": [],
            }
        squad_a_txt = "; ".join(
            f"{p.get('name','')} ({p.get('role','')})" for p in squad_a[:15]
        )
        squad_b_txt = "; ".join(
            f"{p.get('name','')} ({p.get('role','')})" for p in squad_b[:15]
        )
        # Build the concise analytics blobs from your existing helpers
        a_form = analytics_a or {'aggregate': {}, 'batting': [], 'bowling': []}
        b_form = analytics_b or {'aggregate': {}, 'batting': [], 'bowling': []}
        a_agg = a_form.get('aggregate', {})
        b_agg = b_form.get('aggregate', {})
        a_bat = a_form.get('batting', [])[:5]
        b_bat = b_form.get('batting', [])[:5]
        # Reasoning models (gpt-5.x) accept `reasoning.effort`; gpt-4o does not
        # (it would reject the field). Only attach it for reasoning-capable ids.
        is_reasoning_model = str(self.model).startswith(("gpt-5", "o1", "o3", "o4"))
        payload = {
            "model": self.model,
            "store": False,
            "max_output_tokens": 8000,
            "max_tool_calls": 4,
            "tools": [
                {
                    "type": "web_search",
                    "search_context_size": "low",
                }
            ]
            if enable_web_search
            else [],
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "instructions": (
                "You are a cricket analyst. Predict the most likely playing XI "
                "(11 players) for each of the two T20 teams below. "
                + (
                    "The provider has NOT supplied squad lists, so you MUST use "
                    "the web_search tool to find each team's CURRENT 2026 squad "
                    "(including new signings and excluding players who left). "
                    "Only pick players you confirmed exist for the current "
                    "season via the search results — never invent names. "
                    if not (squad_a or squad_b)
                    else (
                        "Use the supplied squad lists as the ONLY allowed player "
                        "pool — do NOT invent names that are not in them. You may "
                        "use the web_search tool to verify availability (injuries, "
                        "rest, confirmed XIs). "
                    )
                )
                + "Balance roles: ~1 wicket-keeper, ~4-5 batters, ~2-3 "
                "allrounders, ~4-5 bowlers per team. Output a JSON object with "
                "keys: xi_team_a (array of 11 names), xi_team_b (array of 11 "
                "names), rationale (short string), confidence (high|medium|low), "
                "needs_human_review (always true), sources (list of any URLs)."
            ),
            "input": json.dumps(
                {
                    "league": league,
                    "team_a": team_a,
                    "team_b": team_b,
                    "start_time_ist": start_time_ist,
                    "hours_to_start": hours_to_start,
                    "squad_a_available": bool(squad_a),
                    "squad_b_available": bool(squad_b),
                    "squad_a": squad_a_txt,
                    "squad_b": squad_b_txt,
                    "analytics_a": {
                        "aggregate": a_agg,
                        "recent_batters": a_bat,
                    },
                    "analytics_b": {
                        "aggregate": b_agg,
                        "recent_batters": b_bat,
                    },
                },
                ensure_ascii=False,
            ),
            "text": {"format": self._probable_xi_schema(), "verbosity": "low"},
        }
        if is_reasoning_model:
            payload["reasoning"] = {"effort": "medium"}
        response = self.transport(payload)
        # Extract the single JSON-text output from the OpenAI response.
        try:
            text = None
            for item in response.get("output", []):
                if item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if content.get("type") != "output_text":
                        continue
                    text = content.get("text")
                    break
            if not text:
                raise RuntimeError("OpenAI response contained no output_text")
            result = json.loads(text)
            # Ensure the required keys exist
            result.setdefault("xi_team_a", [])
            result.setdefault("xi_team_b", [])
            result.setdefault("rationale", "")
            result.setdefault("confidence", "low")
            result.setdefault("needs_human_review", True)
            result.setdefault("sources", [])
            return result
        except Exception as e:
            print(f"predict_probable_xi transport/error: {e}")
            return {
                "failed": True,
                "xi_team_a": [],
                "xi_team_b": [],
                "rationale": "Prediction failed — will retry shortly.",
                "confidence": "low",
                "needs_human_review": True,
                "sources": [],
            }

    @staticmethod
    def _match_analysis_schema():
        """JSON schema for the interpreted match analysis report."""
        return {
            "type": "json_schema",
            "name": "match_analysis_report",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "2-3 sentence match prediction summary (likely winner, expected closeness).",
                    },
                    "predicted_team_a_score": {
                        "type": "string",
                        "description": "e.g. '165-175' or 'Not enough data'.",
                    },
                    "predicted_team_b_score": {
                        "type": "string",
                        "description": "e.g. '150-160' or 'Not enough data'.",
                    },
                    "chase_or_defend": {
                        "type": "string",
                        "description": "Whether the venue/history favors chasing or defending (if known).",
                    },
                    "key_batters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Player names most likely to score (with a short reason each).",
                    },
                    "key_bowlers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Player names most likely to take wickets (with a short reason each).",
                    },
                    "player_assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "team": {"type": "string"},
                                "role": {"type": "string"},
                                "predicted_runs": {"type": "string"},
                                "predicted_wickets": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                "notes": {"type": "string"},
                            },
                            "required": ["name", "team", "role"],
                        },
                        "description": "One entry per predicted player, grounded in the analytics.",
                    },
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Factors that could change the outcome (weather, injuries, unknown XIs).",
                    },
                },
                "required": [
                    "summary",
                    "predicted_team_a_score",
                    "predicted_team_b_score",
                    "chase_or_defend",
                    "key_batters",
                    "key_bowlers",
                    "player_assessments",
                    "risks",
                ],
            },
        }

    def analyze_match_prediction(
        self,
        league,
        team_a,
        team_b,
        venue,
        start_time_ist,
        hours_to_start,
        probable_xi_a,
        probable_xi_b,
        player_analytics,
        venue_stats,
        team_h2h,
        enable_web_search=True,
    ):
        """Interpret player/venue/h2h analytics into a written match prediction report.

        `player_analytics` is a list of the bundles produced by
        CricketAnalytics.get_player_match_analytics() (one per predicted player).
        `venue_stats` is the venue characteristics dict (or {}), `team_h2h` the
        team-vs-team dict (or {}). Returns a dict matching _match_analysis_schema.
        """
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if hours_to_start is not None and hours_to_start > 24:
            return {
                "summary": f"Match starts in {hours_to_start:.1f}h — analysis not meaningful yet.",
                "predicted_team_a_score": "Not available",
                "predicted_team_b_score": "Not available",
                "chase_or_defend": "Not available",
                "key_batters": [],
                "key_bowlers": [],
                "player_assessments": [],
                "risks": ["Match too far out for a useful analysis."],
            }
        is_reasoning_model = str(self.model).startswith(("gpt-5", "o1", "o3", "o4"))
        payload = {
            "model": self.model,
            "store": False,
            "max_output_tokens": 8000,
            "max_tool_calls": 3,
            "tools": [{"type": "web_search", "search_context_size": "low"}]
            if enable_web_search
            else [],
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "instructions": (
                "You are a cricket analyst producing a match prediction report. "
                "Use ONLY the provided structured data (probable XIs, per-player "
                "analytics, venue stats, head-to-head history). The player "
                "analytics come from historical ball-by-ball data; a player with "
                "no data simply has not featured in the available sample. "
                "Optionally use web_search for weather or last-minute news, but "
                "never invent statistics. For each player give a grounded "
                "predicted_runs / predicted_wickets estimate or 'Not enough data'. "
                "Keep every assessment short and factual."
            ),
            "input": json.dumps(
                {
                    "league": league,
                    "team_a": team_a,
                    "team_b": team_b,
                    "venue": venue,
                    "start_time_ist": start_time_ist,
                    "hours_to_start": hours_to_start,
                    "probable_xi_team_a": probable_xi_a,
                    "probable_xi_team_b": probable_xi_b,
                    "player_analytics": player_analytics,
                    "venue_stats": venue_stats or {},
                    "team_h2h": team_h2h or {},
                },
                ensure_ascii=False,
            ),
            "text": {"format": self._match_analysis_schema(), "verbosity": "low"},
        }
        if is_reasoning_model:
            payload["reasoning"] = {"effort": "medium"}
        response = self.transport(payload)
        try:
            text = None
            for item in response.get("output", []):
                if item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if content.get("type") != "output_text":
                        continue
                    text = content.get("text")
                    break
            if not text:
                raise RuntimeError("OpenAI response contained no output_text")
            result = json.loads(text)
            for key in (
                "summary", "predicted_team_a_score", "predicted_team_b_score",
                "chase_or_defend", "key_batters", "key_bowlers",
                "player_assessments", "risks",
            ):
                result.setdefault(key, [] if isinstance(result.get(key), list) else "Not available" if key not in ("key_batters", "key_bowlers", "player_assessments", "risks") else [])
            return result
        except Exception as e:
            print(f"analyze_match_prediction transport/error: {e}")
            return {
                "failed": True,
                "summary": "Analysis failed — will retry shortly.",
                "predicted_team_a_score": "Not available",
                "predicted_team_b_score": "Not available",
                "chase_or_defend": "Not available",
                "key_batters": [],
                "key_bowlers": [],
                "player_assessments": [],
                "risks": [],
            }
