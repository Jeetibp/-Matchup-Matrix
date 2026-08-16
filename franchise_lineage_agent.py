class FranchiseLineageAgent:
    """Apply human-verified franchise histories to a temporary analytics view."""

    def __init__(self, lineages=None):
        self.lineages = lineages or {}

    def aliases_for(self, league, canonical_name):
        values = self.lineages.get(league, {}).get(canonical_name, [])
        aliases = [str(value) for value in values if str(value).strip()]
        if canonical_name and canonical_name not in aliases:
            aliases.insert(0, canonical_name)
        return aliases or ([canonical_name] if canonical_name else [])

    def canonical_map(self, league, canonical_names=None):
        requested = set(canonical_names or [])
        mapping = {}
        for canonical, aliases in self.lineages.get(league, {}).items():
            if requested and canonical not in requested:
                continue
            for alias in aliases:
                mapping[str(alias)] = str(canonical)
            mapping[str(canonical)] = str(canonical)
        return mapping

    def analytics_view(self, analytics, league, canonical_names):
        """Return an isolated CricketAnalytics-shaped object with canonical team names."""
        mapping = self.canonical_map(league, canonical_names)
        if not mapping:
            return analytics
        view = analytics.__class__.__new__(analytics.__class__)
        view.df = analytics.df.copy()
        for column in ("batting_team", "bowling_team"):
            if column in view.df.columns:
                view.df[column] = view.df[column].astype(str).replace(mapping)
        return view

    def provenance(self, league, canonical_names):
        return {
            name: self.aliases_for(league, name)
            for name in canonical_names
            if name
        }
