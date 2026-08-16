import json
import os
import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SUPPORTED_LEAGUE_ALIASES = (
    ("the100women", ("the hundred women", "the hundred women's", "women's hundred")),
    ("the100", ("the hundred",)),
    ("wbb", ("women's big bash league", "womens big bash league", "wbbl")),
    ("wpl", ("women's premier league", "womens premier league", "wpl")),
    ("t20blast", ("vitality blast", "t20 blast")),
    ("sat20", ("sa20", "south africa t20")),
    ("ipl", ("indian premier league", "ipl")),
    ("bbl", ("big bash league", "bbl")),
    ("bpl", ("bangladesh premier league", "bpl")),
    ("cpl", ("caribbean premier league", "cpl")),
    ("ilt", ("international league t20", "ilt20")),
    ("lpl", ("lanka premier league", "lpl")),
    ("mlc", ("major league cricket", "mlc")),
    ("npl", ("nepal premier league", "npl")),
    ("psl", ("pakistan super league", "psl")),
)
SUPPORTED_LEAGUE_KEYS = {league for league, _ in SUPPORTED_LEAGUE_ALIASES}


def detect_supported_league(series_name):
    """Return the internal league key only for an explicit supported-league name."""
    value = " ".join(str(series_name or "").lower().replace("-", " ").split())
    if not value:
        return None
    is_womens_competition = "women" in value
    for league, aliases in SUPPORTED_LEAGUE_ALIASES:
        if is_womens_competition and league not in {"the100women", "wbb", "wpl"}:
            continue
        if not is_womens_competition and league in {"the100women", "wbb", "wpl"}:
            continue
        for alias in aliases:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", value):
                return league
    return None


class CricketDataProvider:
    """Small adapter for CricketData.org. The API key is never sent to clients."""

    BASE_URL = "https://api.cricapi.com/v1"

    def __init__(self, api_key=None, timeout=12):
        self.api_key = api_key or os.environ.get("CRICKETDATA_API_KEY", "")
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.api_key)

    def _get(self, endpoint, **params):
        if not self.configured:
            raise RuntimeError("CRICKETDATA_API_KEY is not configured")
        query = urlencode({"apikey": self.api_key, **params})
        request = Request(
            f"{self.BASE_URL}/{endpoint}?{query}",
            headers={"Accept": "application/json", "User-Agent": "MatchupMatrix/1.0"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if payload.get("status") == "failure":
            raise RuntimeError(payload.get("reason") or "CricketData request failed")
        return payload

    def fetch_matches(self):
        today = date.today()
        horizon = today.toordinal() + 14
        matches = []
        series_names = {}

        for league, aliases in SUPPORTED_LEAGUE_ALIASES:
            search = aliases[0]
            series_payload = self._get("series", offset=0, search=search)
            candidates = []
            for item in series_payload.get("data", []):
                if detect_supported_league(item.get("name")) != league:
                    continue
                window = self._series_window(item)
                if window and window[0].toordinal() <= horizon and window[1] >= today:
                    candidates.append((window[0], item))

            if not candidates:
                continue
            _, current_series = max(candidates, key=lambda candidate: candidate[0])
            series_id = str(current_series.get("id") or "")
            if not series_id:
                continue
            series_name = str(current_series.get("name") or search)
            series_names[series_id] = series_name
            info_payload = self._get("series_info", id=series_id, offset=0)
            series_data = info_payload.get("data") or {}
            for match in series_data.get("matchList", []):
                match = dict(match)
                match["series_id"] = series_id
                match["seriesName"] = series_name
                matches.append(match)

        return matches, series_names

    @staticmethod
    def _series_window(item):
        try:
            start = datetime.strptime(str(item.get("startDate")), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None
        raw_end = str(item.get("endDate") or "").strip()
        end = None
        for value, pattern in (
            (raw_end, "%Y-%m-%d"),
            (f"{raw_end} {start.year}", "%b %d %Y"),
        ):
            try:
                end = datetime.strptime(value, pattern).date()
                break
            except ValueError:
                continue
        if end is None:
            end = start
        if end < start:
            end = end.replace(year=end.year + 1)
        return start, end

    def fetch_match(self, match_id):
        return self._get("match_info", id=match_id, offset=0).get("data") or {}

    def fetch_squad(self, match_id):
        """Fetch Playing XI / full squad from the separate match_squad endpoint.

        CricketData's match_info response does not include Playing XI data;
        lineups (when published) live at /v1/match_squad as:
            data: [ { team: "...", players: [ {name, ...}, ... ] }, ... ]
        Returns the raw `data` array (may be empty when squad not yet published).
        """
        try:
            payload = self._get("match_squad", id=match_id, offset=0)
            data = payload.get("data") or []
            return data if isinstance(data, list) else []
        except RuntimeError:
            # Some plans/leagues do not expose match_squad; fail open with empty list.
            return []

    def fetch_series_squad(self, series_id):
        """Fetch the full squad for every team in a series (T20 franchise leagues).

        Endpoint: /v1/series_squad?id=<series_id>
        Returns:
            data: [
                { teamName: "...", shortname: "...", img: "...",
                  players: [ { id, name, role, battingStyle, bowlingStyle,
                               country, playerImg }, ... ]
                }, ...
            ]

        CricketData publishes full team squads for some leagues (IPL=10, PSL=8,
        MLC=6 verified) but NOT for others (CPL/WPL/LPL/ILT all return []). The
        `info.squads` field returned by /v1/series_info is the count, not the
        actual squad — this endpoint returns the actual player lists.

        May be empty [] when the provider hasn't published squads for this series.
        """
        try:
            payload = self._get("series_squad", id=series_id, offset=0)
            data = payload.get("data") or []
            return data if isinstance(data, list) else []
        except RuntimeError:
            return []


class FixtureService:
    """Normalizes supported fixtures and shields pages from provider failures."""

    def __init__(
        self,
        provider=None,
        fixture_file="data/upcoming_fixtures.json",
        cache_file="data/cache/fixture_provider_cache.json",
        fixture_ttl=86400,
        live_ttl=180,
        series_squad_ttl=86400,
    ):
        self.provider = provider or CricketDataProvider()
        self.fixture_file = Path(fixture_file)
        self.cache_file = Path(cache_file)
        self.fixture_ttl = fixture_ttl
        self.live_ttl = live_ttl
        self.series_squad_ttl = series_squad_ttl
        self._lock = threading.Lock()
        self._live_cache = {}
        # Per-series squad cache: { series_id: (timestamp, squad_list) }
        self._series_squad_cache = {}

    @staticmethod
    def _read_json(path, default):
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            return default

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        temporary.replace(path)

    def get_series_squad(self, series_id):
        """Return the full team squads for a series, cached for 24h in memory.

        CricketData publishes squads once before the tournament starts; once
        we have them there is no reason to re-fetch per match (would burn the
        quota and serve no new data). Returns the raw squad list (each item:
            { teamName, shortname, img, players: [{id, name, role, ...}] }
        ). Returns [] if the series has no published squad.
        """
        if not series_id:
            return []
        cached = self._series_squad_cache.get(series_id)
        if cached and time.time() - cached[0] < self.series_squad_ttl:
            return cached[1]
        if not self.provider.configured or not hasattr(self.provider, "fetch_series_squad"):
            return cached[1] if cached else []
        try:
            squad = self.provider.fetch_series_squad(series_id)
            self._series_squad_cache[series_id] = (time.time(), squad)
            return squad
        except Exception as exc:
            print(f"Series squad fetch failed for {series_id}: {exc}")
            return cached[1] if cached else []

    @staticmethod
    def _series_name(raw, series_names):
        value = raw.get("seriesName") or raw.get("series_name") or raw.get("league")
        if isinstance(value, dict):
            value = value.get("name")
        if value:
            return str(value)
        series_id = str(raw.get("series_id") or raw.get("seriesId") or "")
        return str(series_names.get(series_id, ""))

    @staticmethod
    def _team_names(raw):
        teams = raw.get("teams")
        if isinstance(teams, list) and len(teams) >= 2:
            return str(teams[0]), str(teams[1])
        team_info = raw.get("teamInfo") or raw.get("team_info")
        if isinstance(team_info, list) and len(team_info) >= 2:
            return str(team_info[0].get("name", "")), str(team_info[1].get("name", ""))
        return str(raw.get("team_a") or ""), str(raw.get("team_b") or "")

    @staticmethod
    def _is_placeholder_team(value):
        name = " ".join(str(value or "").lower().split())
        if name in {"tbc", "tbd", "to be confirmed", "to be decided"}:
            return True
        return bool(re.match(r"^(winner|loser|finalist|qualifier|[1-9](st|nd|rd|th) place)\b", name))

    @staticmethod
    def _lineups(raw):
        value = raw.get("playing_xi") or raw.get("playingXI") or raw.get("lineups") or []
        if not isinstance(value, list):
            return []
        lineups = []
        for item in value:
            if not isinstance(item, dict):
                continue
            players = item.get("players") or item.get("playing_xi") or []
            if not isinstance(players, list):
                continue
            names = [
                str(player.get("name") if isinstance(player, dict) else player).strip()
                for player in players
            ]
            names = [name for name in names if name and name != "None"]
            if names:
                lineups.append({
                    "team": str(item.get("team") or item.get("team_name") or "Team"),
                    "players": names,
                    "confirmed": bool(item.get("confirmed", True)),
                })
        return lineups

    @staticmethod
    def _squad_to_lineups(squad):
        """Normalize match_squad payload into the lineup structure the template expects.

        match_squad returns an array of: { team: str, players: [ {name, ...}, ... ] }
        Playing XI is treated as confirmed only when hasSquad is True at call time;
        the caller is responsible for that decision — here we mark every entry confirmed=True
        because the squad endpoint only populates after the official XI is published.
        """
        if not isinstance(squad, list):
            return []
        lineups = []
        for item in squad:
            if not isinstance(item, dict):
                continue
            players = item.get("players") or []
            if not isinstance(players, list):
                continue
            names = [
                str(player.get("name") if isinstance(player, dict) else player).strip()
                for player in players
            ]
            names = [name for name in names if name and name != "None"]
            if names:
                lineups.append({
                    "team": str(item.get("team") or item.get("team_name") or "Team"),
                    "players": names,
                    "confirmed": True,
                })
        return lineups

    def normalize(self, raw, series_names=None, trusted_league=None):
        series_names = series_names or {}
        series_name = self._series_name(raw, series_names)
        internal_league = str(raw.get("league") or "")
        league = trusted_league or (
            internal_league if internal_league in SUPPORTED_LEAGUE_KEYS else detect_supported_league(series_name)
        )
        if not league:
            return None
        team_a, team_b = self._team_names(raw)
        if (
            not team_a
            or not team_b
            or self._is_placeholder_team(team_a)
            or self._is_placeholder_team(team_b)
        ):
            return None
        starts_at = raw.get("dateTimeGMT") or raw.get("starts_at") or raw.get("start_time_utc") or ""
        match_date = raw.get("date") or raw.get("match_date") or str(starts_at)[:10]
        match_id = str(raw.get("id") or raw.get("fixture_id") or "").strip()
        if not match_id:
            return None
        starts_at_str = str(starts_at) if starts_at else ""
        starts_at_ist, hours_to_start = self._compute_start_context(starts_at_str)
        return {
            "id": match_id,
            "league": league,
            "series": series_name,
            "series_id": str(raw.get("series_id") or raw.get("seriesId") or ""),
            "name": str(raw.get("name") or f"{team_a} vs {team_b}"),
            "team_a": team_a,
            "team_b": team_b,
            "date": str(match_date),
            "starts_at": starts_at_str,
            "starts_at_ist": starts_at_ist,
            "hours_to_start": hours_to_start,
            "venue": str(raw.get("venue") or "Venue to be confirmed"),
            "status": str(raw.get("status") or "Scheduled"),
            "match_type": str(raw.get("matchType") or raw.get("match_type") or ""),
            "score": raw.get("score") if isinstance(raw.get("score"), list) else [],
            "toss_winner": raw.get("tossWinner") or raw.get("toss_winner"),
            "toss_choice": raw.get("tossChoice") or raw.get("toss_choice"),
            "lineups": self._lineups(raw),
            "source": str(raw.get("source") or "cricketdata"),
        }

    @staticmethod
    def _compute_start_context(starts_at_str):
        """Return (starts_at_ist_str, hours_to_start_float).

        starts_at_str is the provider's `dateTimeGMT` value (e.g. "2026-08-15T00:00:00").
        Returns ("", None) if the value is missing or unparseable.

        `hours_to_start` is negative once the match has started — callers should
        treat `0 <= hours_to_start <= 24` as the prediction window.
        """
        if not starts_at_str:
            return "", None
        try:
            # CricketData sends "YYYY-MM-DDTHH:MM:SS" without a Z; treat as UTC.
            s = starts_at_str.strip().replace("Z", "")
            dt_utc = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return "", None
        # IST = UTC + 5:30
        from datetime import timedelta as _td
        ist_tz = timezone(_td(hours=5, minutes=30))
        starts_at_ist = dt_utc.astimezone(ist_tz).strftime("%Y-%m-%d %H:%M IST")
        now_utc = datetime.now(tz=timezone.utc)
        hours_to_start = (dt_utc - now_utc).total_seconds() / 3600.0
        return starts_at_ist, round(hours_to_start, 2)

    def _local_matches(self):
        payload = self._read_json(self.fixture_file, {"matches": []})
        records = payload.get("matches", []) if isinstance(payload, dict) else []
        return [match for raw in records if (match := self.normalize(raw))]

    def _cached_provider_matches(self):
        payload = self._read_json(self.cache_file, {})
        records = payload.get("matches", []) if isinstance(payload, dict) else []
        return [
            match
            for raw in records
            if (match := self.normalize(raw, trusted_league=raw.get("league")))
        ]

    def _cache_is_fresh(self):
        try:
            return time.time() - self.cache_file.stat().st_mtime < self.fixture_ttl
        except OSError:
            return False

    def _provider_matches(self):
        cached = self._cached_provider_matches()
        if not self.provider.configured or self._cache_is_fresh():
            return cached
        with self._lock:
            if self._cache_is_fresh():
                return self._cached_provider_matches()
            try:
                raw_matches, series_names = self.provider.fetch_matches()
                matches = [
                    match
                    for raw in raw_matches
                    if (match := self.normalize(raw, series_names))
                ]
                self._write_json(
                    self.cache_file,
                    {"updated_at": datetime.now(timezone.utc).isoformat(), "matches": matches},
                )
                return matches
            except Exception as exc:
                print(f"Fixture provider unavailable; using stale cache: {exc}")
                return cached

    def list_matches(self, match_date=None, league=None):
        merged = {match["id"]: match for match in self._cached_provider_matches()}
        merged.update({match["id"]: match for match in self._provider_matches()})
        merged.update({match["id"]: match for match in self._local_matches()})
        matches = list(merged.values())
        if match_date:
            matches = [match for match in matches if match.get("date") == str(match_date)]
        if league:
            matches = [match for match in matches if match.get("league") == str(league)]
        return sorted(matches, key=lambda match: (match.get("starts_at") or match.get("date") or "", match["id"]))

    def today(self, league=None):
        return self.list_matches(date.today().isoformat(), league=league)

    def get_match(self, match_id):
        """Return the freshest known record: a live-fetched override, then the schedule cache."""
        listed = next((match for match in self.list_matches() if match["id"] == str(match_id)), None)
        cached_live = self._live_cache.get(str(match_id))
        if cached_live and time.time() - cached_live[0] < self.live_ttl:
            return cached_live[1] if listed else None
        return listed

    def live_match(self, match_id):
        """Refetch one match so a per-match team/venue/status correction is not masked by the schedule cache."""
        listed = next((match for match in self.list_matches() if match["id"] == str(match_id)), None)
        if not listed:
            return None
        cached = self._live_cache.get(str(match_id))
        if cached and time.time() - cached[0] < self.live_ttl:
            return cached[1]
        if not self.provider.configured:
            return listed
        try:
            raw = self.provider.fetch_match(str(match_id))
            current = self.normalize(raw, trusted_league=listed["league"]) or listed
            current["series"] = listed.get("series", current.get("series", ""))
            # Preserve previously-known lineups when the fresh match_info payload
            # does not include them (CricketData's match_info never carries XI).
            if not current.get("lineups") and listed.get("lineups"):
                current["lineups"] = listed["lineups"]

            # CricketData exposes Playing XI at /v1/match_squad, NOT in match_info.
            # Only fetch when we don't already have lineups — avoids burning quota
            # on every refresh of matches that already carry a lineup (e.g. from
            # the local fixtures file or a prior poll).
            if not current.get("lineups") and hasattr(self.provider, "fetch_squad"):
                squad = self.provider.fetch_squad(str(match_id))
                if squad:
                    current["lineups"] = self._squad_to_lineups(squad)
            self._live_cache[str(match_id)] = (time.time(), current)
            return current
        except Exception as exc:
            print(f"Live score provider unavailable; using fixture cache: {exc}")
            return cached[1] if cached else listed
