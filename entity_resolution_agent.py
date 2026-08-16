import re


class EntityResolutionAgent:
    """Resolve provider names to exact CSV entities without making statistical guesses."""

    def __init__(self, player_aliases=None, entity_aliases=None):
        self.player_aliases = player_aliases or {}
        self.entity_aliases = entity_aliases or {}

    @staticmethod
    def _tokens(value):
        text = str(value or "").lower().replace("&", " and ")
        tokens = re.findall(r"[a-z0-9]+", text)
        canonical = {"saint": "st"}
        return [canonical.get(token, token) for token in tokens]

    @classmethod
    def _key(cls, value):
        return "".join(cls._tokens(value))

    @staticmethod
    def _result(input_name, resolved_name=None, status="unresolved", confidence=0.0, method=None, candidates=None):
        return {
            "input": str(input_name or ""),
            "resolved": resolved_name,
            "status": status,
            "confidence": confidence,
            "method": method,
            "candidates": candidates or [],
        }

    @classmethod
    def _find_target(cls, target, available):
        target_key = cls._key(target)
        matches = [item for item in available if cls._key(item) == target_key]
        return matches[0] if len(matches) == 1 else None

    def _entity_aliases(self, entity_type, league):
        sections = self.entity_aliases.get(entity_type, {})
        aliases = dict(sections.get("global", {}))
        aliases.update(sections.get(league, {}))
        return aliases

    def _resolve_named_entity(self, value, available, entity_type, league=None):
        available = [str(item) for item in available if str(item).strip()]
        if not value or not available:
            return self._result(value)

        exact = [item for item in available if self._key(item) == self._key(value)]
        if len(exact) == 1:
            return self._result(value, exact[0], "resolved", 1.0, "canonical_exact")

        aliases = self._entity_aliases(entity_type, league)
        alias_target = aliases.get(str(value).strip().lower()) or aliases.get(self._key(value))
        if alias_target:
            resolved = self._find_target(alias_target, available)
            if resolved:
                return self._result(value, resolved, "resolved", 1.0, "verified_alias")

        target = self._key(value)
        partial = [item for item in available if target in self._key(item) or self._key(item) in target]
        if len(partial) == 1:
            return self._result(value, partial[0], "resolved", 0.85, "unique_partial")
        if len(partial) > 1:
            return self._result(value, status="ambiguous", method="unique_partial", candidates=partial)
        return self._result(value)

    def resolve_team(self, value, available, league=None):
        return self._resolve_named_entity(value, available, "teams", league)

    def resolve_venue(self, value, available, league=None):
        return self._resolve_named_entity(value, available, "venues", league)

    def resolve_player(self, value, available, league=None):
        available = [str(item) for item in available if str(item).strip()]
        if not value or not available:
            return self._result(value)

        exact = [item for item in available if self._key(item) == self._key(value)]
        if len(exact) == 1:
            return self._result(value, exact[0], "resolved", 1.0, "canonical_exact")

        aliases = dict(self.player_aliases.get("global", {}))
        aliases.update(self.player_aliases.get(league, {}))
        alias_target = aliases.get(str(value).strip().lower()) or aliases.get(self._key(value))
        if alias_target:
            resolved = self._find_target(alias_target, available)
            if resolved:
                return self._result(value, resolved, "resolved", 1.0, "verified_alias")

        requested = self._tokens(value)
        if not requested:
            return self._result(value)
        surname = requested[-1]
        surname_matches = [item for item in available if self._tokens(item) and self._tokens(item)[-1] == surname]

        if len(requested) == 1:
            if len(surname_matches) == 1:
                return self._result(value, surname_matches[0], "resolved", 0.75, "unique_surname")
            if len(surname_matches) > 1:
                return self._result(value, status="ambiguous", method="unique_surname", candidates=surname_matches)
            return self._result(value)

        first_initial = requested[0][0]
        initial_matches = [
            item
            for item in surname_matches
            if self._tokens(item)[0] and self._tokens(item)[0][0] == first_initial
        ]
        if len(initial_matches) == 1:
            return self._result(value, initial_matches[0], "resolved", 0.9, "initial_surname")
        if len(initial_matches) > 1:
            return self._result(value, status="ambiguous", method="initial_surname", candidates=initial_matches)
        return self._result(value)
