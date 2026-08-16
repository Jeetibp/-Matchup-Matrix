from flask import Flask, render_template, request, jsonify, session, send_from_directory
from cricket_analytics_core import CricketAnalytics
from entity_resolution_agent import EntityResolutionAgent
from fixture_service import FixtureService
from franchise_lineage_agent import FranchiseLineageAgent
from match_intelligence import build_match_intelligence
from report_cache import ReportCache
import os
import time
import warnings
import gc
import sys
import pickle
import threading
import queue
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional OpenAI imports — available when OPENAI_API_KEY is set in .env
try:
    from openai_research_agent import OpenAIResearchAgent
    _OAI_AGENT = OpenAIResearchAgent() if os.environ.get("OPENAI_API_KEY") else None
except Exception:
    _OAI_AGENT = None

# Suppress pandas warnings to reduce memory overhead
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 't20blast2025')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Enhanced production configuration for PythonAnywhere
APP_ENV = os.environ.get('APP_ENV', 'production')
if APP_ENV == 'production':
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    app.config['ENV'] = 'production'
else:
    app.config['DEBUG'] = True

LEAGUE_CSVS = {
    't20blast':    'data/all_matches_t20blast.csv',
    'mlc':         'data/all_matches_MLC.csv',
    'ipl':         'data/all_matches_ipl.csv',
    'the100':      'data/all_matches_the100.csv',
    'cpl':         'data/all_matches_cpl.csv',
    'the100women': 'data/all_matches_the100women.csv',
    'bbl':         'data/all_matches_bbl.csv',
    'bpl':         'data/all_matches_bpl.csv',
    'ilt':         'data/all_matches_ilt.csv',
    'lpl':         'data/all_matches_LPL.csv',
    'npl':         'data/all_matches_NPL.csv',
    'psl':         'data/all_matches_psl.csv',
    'sat20':       'data/all_matches_sat20.csv',
    'wbb':         'data/all_matches_wbb.csv',
    'wpl':         'data/all_matches_wpl.csv',
}

analytics_cache = {}
_probable_xi_cache = ReportCache("data/cache/reports/probable_xi")  # match_id -> report dict, TTL 6h
_match_analysis_cache = ReportCache("data/cache/reports/match_analysis")  # match_id -> report dict, TTL 6h
stats_cache = {}        # { (league, func, min_innings, innings_filter): DataFrame }
PICKLE_CACHE_DIR = Path('data/cache')
fixture_service = FixtureService()
match_intelligence_cache = {}

# ---------------------------------------------------------------------------
# Background AI report worker. Probable-XI + match-analysis are expensive OpenAI
# calls; running them inline makes the page block for 10-40s. Instead the page
# renders instantly from the file-backed cache and enqueues missing reports,
# which a single daemon worker computes off-request. Pre-warm fills the cache
# ahead of time so the first visit is usually already fast.
# ---------------------------------------------------------------------------
_report_queue = queue.Queue()
_report_in_flight = set()
_report_lock = threading.Lock()


def enqueue_report(match_id):
    match_id = str(match_id)
    with _report_lock:
        if match_id in _report_in_flight:
            return
        _report_in_flight.add(match_id)
    _report_queue.put(match_id)


def _report_worker_loop():
    while True:
        match_id = _report_queue.get()
        try:
            compute_match_report(match_id)
        except Exception as exc:
            print(f"Background report failed for {match_id}: {exc}")
        finally:
            with _report_lock:
                _report_in_flight.discard(match_id)


def _start_report_worker():
    thread = threading.Thread(target=_report_worker_loop, daemon=True, name="report-worker")
    thread.start()
    return thread


def report_gate_open(fixture):
    """True when AI prediction is applicable for this fixture right now."""
    if _OAI_AGENT is None:
        return False
    if not fixture:
        return False
    if fixture.get("lineups"):
        return False
    ha = fixture.get("hours_to_start")
    return ha is not None and 0 < ha <= 24

# --- Player alias map: league-wise { "global":{...}, "ipl":{...}, ... } ---
def _load_player_aliases():
    try:
        import json
        p = Path('data/player_aliases.json')
        if p.exists():
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Could not load player aliases: {e}")
    return {}

PLAYER_ALIASES = _load_player_aliases()

def _load_entity_aliases():
    try:
        import json
        path = Path('data/entity_aliases.json')
        if path.exists():
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
    except Exception as e:
        print(f"Could not load entity aliases: {e}")
    return {}

ENTITY_ALIASES = _load_entity_aliases()
entity_resolution_agent = EntityResolutionAgent(PLAYER_ALIASES, ENTITY_ALIASES)

def _load_franchise_lineages():
    try:
        import json
        path = Path('data/franchise_lineages.json')
        if path.exists():
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
    except Exception as e:
        print(f"Could not load franchise lineages: {e}")
    return {}

FRANCHISE_LINEAGES = _load_franchise_lineages()
franchise_lineage_agent = FranchiseLineageAgent(FRANCHISE_LINEAGES)

def get_aliases_for_league(league):
    """Merge global aliases + current league aliases (league overrides global)."""
    merged = dict(PLAYER_ALIASES.get('global', {}))
    merged.update(PLAYER_ALIASES.get(league, {}))
    return merged

# --- Venue mapping: normalise variant/old names to a canonical name ---
def _load_venue_mapping():
    try:
        import json
        p = Path('data/venue_mapping.json')
        if p.exists():
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Could not load venue mapping: {e}")
    return {}

VENUE_MAPPING = _load_venue_mapping()

def get_venue_map(league):
    """Return {variant: canonical} for the given league."""
    return dict(VENUE_MAPPING.get(league, {}))

def get_canonical_venue(venue, venue_map):
    """Return the canonical name for a venue (or the original if not mapped)."""
    return venue_map.get(venue, venue)

def get_venue_variants(canonical, venue_map):
    """Return all dataset names (including canonical) that resolve to the same canonical."""
    variants = {canonical}
    for variant, canon in venue_map.items():
        if canon == canonical:
            variants.add(variant)
    return variants

def _apply_venue_normalization(analytics_obj, league):
    """Normalize venue names in analytics.df in-place using venue_mapping.json."""
    try:
        vm = get_venue_map(league)
        if vm and analytics_obj is not None and hasattr(analytics_obj, 'df'):
            analytics_obj.df['venue'] = analytics_obj.df['venue'].map(lambda v: vm.get(v, v) if isinstance(v, str) else v)
    except Exception as e:
        print(f"Venue normalization failed for {league}: {e}")

# --- Pickle helpers ---
def _get_pickle_path(league):
    return PICKLE_CACHE_DIR / f'{league}.pkl'

def _pickle_is_valid(league, csv_path):
    try:
        pkl = _get_pickle_path(league)
        return pkl.exists() and pkl.stat().st_mtime > Path(csv_path).stat().st_mtime
    except Exception:
        return False

def _save_to_pickle(league, analytics):
    try:
        PICKLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_get_pickle_path(league), 'wb') as f:
            pickle.dump(analytics.df, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Pickle saved for {league}")
    except Exception as e:
        print(f"Pickle save failed for {league}: {e}")

def _load_from_pickle(league):
    try:
        with open(_get_pickle_path(league), 'rb') as f:
            df = pickle.load(f)
        analytics = CricketAnalytics.__new__(CricketAnalytics)
        analytics.df = df
        print(f"Loaded {league} from pickle (fast path)")
        return analytics
    except Exception as e:
        print(f"Pickle load failed for {league}: {e}")
        return None

# --- Stats result cache ---
def get_cached_stats(analytics, league, func, min_innings, innings_filter):
    key = (league, func, min_innings, innings_filter)
    if key not in stats_cache:
        if func == 'batting':
            stats_cache[key] = analytics.get_batting_stats(min_innings, innings_filter=innings_filter)
        else:
            stats_cache[key] = analytics.get_bowling_stats(min_innings, innings_filter=innings_filter)
        print(f"Stats cached: {key}")
    return stats_cache[key]

# --- Background warmup thread ---
def _warmup_all_leagues():
    import time
    time.sleep(2)  # let app fully start first
    print("Background warmup starting...")
    for league_key, csv_path in list(available_leagues().items()):
        if league_key not in analytics_cache:
            try:
                if _pickle_is_valid(league_key, csv_path):
                    a = _load_from_pickle(league_key)
                else:
                    print(f"Warmup full load: {league_key}")
                    a = CricketAnalytics(csv_path)
                    _save_to_pickle(league_key, a)
                if a:
                    _apply_venue_normalization(a, league_key)
                    analytics_cache[league_key] = a
                    print(f"Warmup done: {league_key}")
            except Exception as e:
                print(f"Warmup failed for {league_key}: {e}")
    print("All leagues warmed up.")

threading.Thread(target=_warmup_all_leagues, daemon=True).start()

# CRITICAL FIX: Optimize health checks to prevent unnecessary processing
@app.before_request
def optimize_health_checks():
    """Skip heavy processing for health checks and monitoring requests"""
    if request.path == '/' and request.method == 'HEAD':
        return jsonify({'status': 'ok'}), 200
    
    # Also optimize for monitoring requests
    if 'Go-http-client' in request.headers.get('User-Agent', ''):
        return jsonify({'status': 'healthy'}), 200

def available_leagues():
    """Get available cricket leagues based on existing CSV files"""
    return {k: v for k, v in LEAGUE_CSVS.items() if os.path.exists(v)}

def get_league():
    """Get current selected league from request or session"""
    avail = available_leagues()
    league = request.args.get('league') or session.get('league') or 'ipl'
    if league not in avail:
        league = list(avail.keys())[0] if avail else None
    session['league'] = league
    return league

def get_analytics():
    """Get cricket analytics instance with caching"""
    try:
        avail = available_leagues()
        league = get_league()
        if league and league in avail:
            if league in analytics_cache:
                analytics = analytics_cache[league]
            else:
                csv_path = avail[league]
                try:
                    if _pickle_is_valid(league, csv_path):
                        print(f"Loading {league} from pickle cache...")
                        analytics = _load_from_pickle(league)
                        if analytics is None:
                            raise Exception("Pickle load returned None, falling back to CSV")
                    else:
                        print(f"Loading cricket analytics for {league}...")
                        analytics = CricketAnalytics(csv_path)
                        _save_to_pickle(league, analytics)
                    _apply_venue_normalization(analytics, league)
                    analytics_cache[league] = analytics
                    gc.collect()
                    print(f"Successfully loaded {league} analytics")
                except Exception as e:
                    print(f"Error loading {league}: {e}")
                    return None, league, f"Error loading data for league: {league.upper()}<br>{e}"
            return analytics, league, None
        else:
            return None, league, f"No data available for selected league."
    except Exception as e:
        print(f"System error in get_analytics: {e}")
        return None, None, f"System error: {str(e)}"

def get_analytics_for_league(league):
    """Load one explicit league for fixture intelligence without changing the session."""
    avail = available_leagues()
    if league not in avail:
        return None
    if league in analytics_cache:
        return analytics_cache[league]
    csv_path = avail[league]
    analytics = _load_from_pickle(league) if _pickle_is_valid(league, csv_path) else None
    if analytics is None:
        analytics = CricketAnalytics(csv_path)
        _save_to_pickle(league, analytics)
    _apply_venue_normalization(analytics, league)
    analytics_cache[league] = analytics
    return analytics

@app.route('/')
def home():
    """Homepage with cricket analytics overview"""
    try:
        # Quick response for monitoring/health checks
        if 'Go-http-client' in request.headers.get('User-Agent', ''):
            return jsonify({'status': 'healthy', 'service': 'cricket-analytics'}), 200
            
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        
        if not analytics:
            return render_template(
                "home.html",
                total_players=0,
                total_bowlers=0,
                top_bat_all=None,
                top_bat_1=None,
                top_bat_2=None,
                top_bowl_all=None,
                top_bowl_1=None,
                top_bowl_2=None,
                league=league,
                leagues=leagues,
                error=error
            )
        
        # Use stats cache - compute once, reuse on every subsequent request
        try:
            print("Processing batting stats for homepage...")
            top_bat_all = get_cached_stats(analytics, league, 'batting', 1, None).head(1)
            top_bat_1   = get_cached_stats(analytics, league, 'batting', 1, 1).head(1)
            top_bat_2   = get_cached_stats(analytics, league, 'batting', 1, 2).head(1)

            print("Processing bowling stats for homepage...")
            top_bowl_all = get_cached_stats(analytics, league, 'bowling', 1, None).head(1)
            top_bowl_1   = get_cached_stats(analytics, league, 'bowling', 1, 1).head(1)
            top_bowl_2   = get_cached_stats(analytics, league, 'bowling', 1, 2).head(1)
            
            # FIXED: Get actual counts instead of DataFrame lengths
            total_players = analytics.df['batsman'].nunique() if hasattr(analytics, 'df') else 0
            total_bowlers = analytics.df['bowler'].nunique() if hasattr(analytics, 'df') else 0
            
            # Clear memory after processing
            gc.collect()
            print(f"Homepage stats processed successfully - {total_players} players, {total_bowlers} bowlers")
            
            return render_template(
                "home.html",
                total_players=total_players,
                total_bowlers=total_bowlers,
                top_bat_all=top_bat_all.to_dict("records")[0] if not top_bat_all.empty else None,
                top_bat_1=top_bat_1.to_dict("records")[0] if not top_bat_1.empty else None,
                top_bat_2=top_bat_2.to_dict("records")[0] if not top_bat_2.empty else None,
                top_bowl_all=top_bowl_all.to_dict("records")[0] if not top_bowl_all.empty else None,
                top_bowl_1=top_bowl_1.to_dict("records")[0] if not top_bowl_1.empty else None,
                top_bowl_2=top_bowl_2.to_dict("records")[0] if not top_bowl_2.empty else None,
                league=league,
                leagues=leagues,
                error=None
            )
        except Exception as stats_error:
            print(f"Error processing homepage stats: {stats_error}")
            # Fallback for memory issues
            return render_template(
                "home.html",
                total_players=0,
                total_bowlers=0,
                top_bat_all=None,
                top_bat_1=None,
                top_bat_2=None,
                top_bowl_all=None,
                top_bowl_1=None,
                top_bowl_2=None,
                league=league,
                leagues=leagues,
                error=f"Data loading in progress... Please refresh in a moment."
            )
    except Exception as e:
        print(f"Critical error in home route: {e}")
        return f"""
        <h1>🏏 Matchup Matrix - Cricket Analytics</h1>
        <p>Welcome to the Cricket Analytics Platform</p>
        <p>System is initializing... Please refresh in a moment.</p>
        <p style="color: #666; font-size: 12px;">Debug: {str(e)}</p>
        """

@app.route("/batting")
def batting():
    """Batting statistics page"""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        min_innings = request.args.get("min_innings", 5, type=int)
        innings_filter = request.args.get("innings_filter", 0, type=int)
        selected_seasons = [s for s in request.args.getlist("season") if s and s != 'all']
        filter_val = innings_filter if innings_filter in [1,2] else None
        seasons = []
        
        if analytics:
            seasons = sorted(analytics.df['season'].dropna().astype(str).unique().tolist(), reverse=True)
            if selected_seasons:
                filtered = CricketAnalytics.__new__(CricketAnalytics)
                filtered.df = analytics.df[analytics.df['season'].astype(str).isin(selected_seasons)].copy()
                stats = filtered.get_batting_stats(min_innings, innings_filter=filter_val)
            else:
                stats = get_cached_stats(analytics, league, 'batting', min_innings, filter_val)
        else:
            stats = []
        
        # Memory cleanup
        gc.collect()
        
        return render_template(
            "batting.html",
            stats=stats.to_dict("records") if analytics and hasattr(stats, 'to_dict') and not stats.empty else [],
            min_innings=min_innings,
            innings_filter=innings_filter,
            selected_seasons=selected_seasons,
            seasons=seasons,
            league=league,
            leagues=leagues,
            error=error
        )
    except Exception as e:
        print(f"Error in batting route: {e}")
        return render_template(
            "batting.html",
            stats=[],
            min_innings=5,
            innings_filter=0,
            selected_seasons=[],
            seasons=[],
            league=None,
            leagues=available_leagues(),
            error=f"Error loading batting stats: {str(e)}"
        )

@app.route("/bowling")
def bowling():
    """Bowling statistics page"""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        min_innings = request.args.get("min_innings", 3, type=int)
        innings_filter = request.args.get("innings_filter", 0, type=int)
        selected_seasons = [s for s in request.args.getlist("season") if s and s != 'all']
        filter_val = innings_filter if innings_filter in [1,2] else None
        seasons = []
        
        if analytics:
            seasons = sorted(analytics.df['season'].dropna().astype(str).unique().tolist(), reverse=True)
            if selected_seasons:
                filtered = CricketAnalytics.__new__(CricketAnalytics)
                filtered.df = analytics.df[analytics.df['season'].astype(str).isin(selected_seasons)].copy()
                stats = filtered.get_bowling_stats(min_innings, innings_filter=filter_val)
            else:
                stats = get_cached_stats(analytics, league, 'bowling', min_innings, filter_val)
        else:
            stats = []
        
        # Memory cleanup
        gc.collect()
        
        return render_template(
            "bowling.html",
            stats=stats.to_dict("records") if analytics and hasattr(stats, 'to_dict') and not stats.empty else [],
            min_innings=min_innings,
            innings_filter=innings_filter,
            selected_seasons=selected_seasons,
            seasons=seasons,
            league=league,
            leagues=leagues,
            error=error
        )
    except Exception as e:
        print(f"Error in bowling route: {e}")
        return render_template(
            "bowling.html",
            stats=[],
            min_innings=3,
            innings_filter=0,
            selected_seasons=[],
            seasons=[],
            league=None,
            leagues=available_leagues(),
            error=f"Error loading bowling stats: {str(e)}"
        )

@app.route("/headtohead", methods=["GET", "POST"])
def headtohead():
    """Head-to-head analysis page"""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        innings_filter = (
            request.form.get("innings_filter", request.args.get("innings_filter", 0))
        )
        try:
            innings_filter = int(innings_filter)
        except Exception:
            innings_filter = 0
        message = None
        matchup = None
        multiple = None

        if not analytics:
            return render_template(
                "headtohead.html",
                message=error or "No data available.",
                matchup=None,
                multiple_results=None,
                all_bowlers=[],
                all_batsmen=[],
                saved_inputs={'single_bowler':'','single_batsman':'','innings_filter':innings_filter, 'multiple_bowlers':[], 'multiple_batsmen':[]},
                innings_filter=innings_filter,
                league=league,
                leagues=leagues,
                error=error
            )

        if 'h2h_inputs' not in session:
            session['h2h_inputs'] = {'single_bowler':'','single_batsman':'','innings_filter':innings_filter,
                                     'multiple_bowlers':[], 'multiple_batsmen':[]}

        saved_inputs = session['h2h_inputs']
        saved_inputs["innings_filter"] = innings_filter

        try:
            innings_list = [innings_filter] if innings_filter in [1,2] else [1,2]
            all_bowlers = sorted(analytics.df[analytics.df["innings"].isin(innings_list)]["bowler"].dropna().unique())
            all_batsmen = sorted(analytics.df[analytics.df["innings"].isin(innings_list)]["batsman"].dropna().unique())
        except Exception as e:
            print(f"Error getting player lists: {e}")
            all_bowlers = []
            all_batsmen = []

        if request.method == "POST":
            atype = request.form.get("analysis_type", "single")
            league_aliases = get_aliases_for_league(league)

            # Pre-build frequency maps: player → number of rows in dataset
            _bat_freq = analytics.df['batsman'].value_counts().to_dict()
            _bowl_freq = analytics.df['bowler'].value_counts().to_dict()

            def resolve_player(name, player_list, freq_map):
                """Resolve via: exact match → alias map → partial match (most frequent wins)."""
                # 1. Exact match (case-insensitive)
                for p in player_list:
                    if p.lower() == name.lower():
                        return p
                # 2. Alias map (e.g. "bumrah" → "JJ Bumrah")
                alias = league_aliases.get(name.lower())
                if alias:
                    return alias
                # 3. Partial match — if multiple, pick the one with the most appearances
                matches = [p for p in player_list if name.lower() in p.lower()]
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    return max(matches, key=lambda p: freq_map.get(p, 0))
                return name

            if atype == "single":
                b = request.form.get("bowler", "").strip()
                bt = request.form.get("batsman", "").strip()
                session['h2h_inputs']['single_bowler'] = b
                session['h2h_inputs']['single_batsman'] = bt
                session['h2h_inputs']['innings_filter'] = innings_filter
                if b and bt:
                    b_resolved = resolve_player(b, all_bowlers, _bowl_freq)
                    bt_resolved = resolve_player(bt, all_batsmen, _bat_freq)
                    print(f"Processing H2H: {b_resolved} vs {bt_resolved}")
                    matchup = analytics.get_head_to_head(b_resolved, bt_resolved, innings_filter=innings_filter)
                    if not matchup:
                        message = f"No matchup found for {b} vs {bt} in {'All' if not innings_filter else str(innings_filter)+'st/2nd'} Innings"
                else:
                    message = "Select both bowler and batsman."
            elif atype == "multiple":
                bs = [x.strip() for x in request.form.getlist("bowlers[]") if x.strip()]
                bts = [x.strip() for x in request.form.getlist("batsmen[]") if x.strip()]
                session['h2h_inputs']['multiple_bowlers'] = bs
                session['h2h_inputs']['multiple_batsmen'] = bts
                session['h2h_inputs']['innings_filter'] = innings_filter
                if bs and bts:
                    bs_resolved = [resolve_player(b, all_bowlers, _bowl_freq) for b in bs]
                    bts_resolved = [resolve_player(bt, all_batsmen, _bat_freq) for bt in bts]
                    print(f"Processing multiple H2H: {len(bs_resolved)} bowlers vs {len(bts_resolved)} batsmen")
                    multiple = analytics.get_multiple_head_to_head(bs_resolved, bts_resolved, innings_filter=innings_filter)
                else:
                    message = "Select at least one bowler and batsman."
            elif atype == "swap_multiple":
                cb = session['h2h_inputs']['multiple_bowlers']
                cbt = session['h2h_inputs']['multiple_batsmen']
                session['h2h_inputs']['multiple_bowlers'] = cbt
                session['h2h_inputs']['multiple_batsmen'] = cb
                message = "Multiple players swapped!"
            elif atype == "reset":
                session['h2h_inputs'] = {'single_bowler':'','single_batsman':'','innings_filter':innings_filter,
                                        'multiple_bowlers':[], 'multiple_batsmen':[]}
                message = "All inputs cleared!"
        else:
            if saved_inputs["single_bowler"] and saved_inputs["single_batsman"]:
                _aliases = get_aliases_for_league(league)
                _bf = analytics.df['bowler'].value_counts().to_dict()
                _batf = analytics.df['batsman'].value_counts().to_dict()
                def _resolve(name, player_list, freq_map):
                    for p in player_list:
                        if p.lower() == name.lower(): return p
                    alias = _aliases.get(name.lower())
                    if alias: return alias
                    matches = [p for p in player_list if name.lower() in p.lower()]
                    if len(matches) == 1: return matches[0]
                    if len(matches) > 1: return max(matches, key=lambda p: freq_map.get(p, 0))
                    return name
                matchup = analytics.get_head_to_head(
                    _resolve(saved_inputs["single_bowler"], all_bowlers, _bf),
                    _resolve(saved_inputs["single_batsman"], all_batsmen, _batf),
                    innings_filter=saved_inputs.get("innings_filter")
                )

        # Memory cleanup
        gc.collect()

        return render_template(
            "headtohead.html",
            message=message,
            matchup=matchup,
            multiple_results=multiple,
            all_bowlers=all_bowlers,
            all_batsmen=all_batsmen,
            saved_inputs=session["h2h_inputs"],
            innings_filter=innings_filter,
            league=league,
            leagues=leagues,
            error=error
        )
    except Exception as e:
        print(f"Error in headtohead route: {e}")
        return render_template(
            "headtohead.html",
            message=f"Error loading head-to-head analysis: {str(e)}",
            matchup=None,
            multiple_results=None,
            all_bowlers=[],
            all_batsmen=[],
            saved_inputs={'single_bowler':'','single_batsman':'','innings_filter':0, 'multiple_bowlers':[], 'multiple_batsmen':[]},
            innings_filter=0,
            league=None,
            leagues=available_leagues(),
            error=f"Error: {str(e)}"
        )

# --- API for Fuzzy Player Suggestions ---
@app.route('/api/player_fuzzy')
def api_player_fuzzy():
    """API endpoint for player name suggestions"""
    try:
        analytics, league, error = get_analytics()
        if not analytics:
            return jsonify({'players': []})
        q = request.args.get('q', '').strip().lower()
        ptype = request.args.get('ptype', 'both')
        innings_filter = int(request.args.get('innings_filter', 0))
        
        if ptype == 'bowler':
            players = analytics.df['bowler'].dropna().astype(str)
            if innings_filter in [1,2]:
                players = analytics.df[analytics.df['innings']==innings_filter]['bowler'].dropna().astype(str)
        elif ptype == 'batsman':
            players = analytics.df['batsman'].dropna().astype(str)
            if innings_filter in [1,2]:
                players = analytics.df[analytics.df['innings']==innings_filter]['batsman'].dropna().astype(str)
        else:
            players = analytics.df['bowler'].dropna().astype(str).tolist() + analytics.df['batsman'].dropna().astype(str).tolist()
            if innings_filter in [1,2]:
                bowlers = analytics.df[analytics.df['innings']==innings_filter]['bowler'].dropna().astype(str).tolist()
                batsmen = analytics.df[analytics.df['innings']==innings_filter]['batsman'].dropna().astype(str).tolist()
                players = bowlers + batsmen
                
        players = sorted(set(players))

        # Build search terms: alias map + multi-word first-initial expansion
        terms = set()
        terms.add(q)

        # 1. Check alias map (global + league-specific): "virat" -> "V Kohli"
        league_aliases = get_aliases_for_league(league)
        alias_match = league_aliases.get(q)
        if alias_match:
            terms.add(alias_match.lower())

        # 2. Multi-word: "virat kohli" -> also try "v kohli" and "kohli"
        words = q.split()
        if len(words) >= 2:
            terms.add(words[0][0] + ' ' + ' '.join(words[1:]))
            terms.add(words[-1])
            # also check alias for the full multi-word query (already done above)

        seen = set()
        results = []
        for p in players:
            pl = p.lower()
            if any(t in pl for t in terms) and p not in seen:
                seen.add(p)
                results.append(p)

        return jsonify({'players': results[:20]})
    except Exception as e:
        print(f"Error in player_fuzzy API: {e}")
        return jsonify({'players': [], 'error': str(e)})

# --- API for Opponent Filtering for Dropdown (Smart Filter) ---
@app.route('/api/get_opponents', methods=["POST"])
def api_get_opponents():
    """API endpoint for getting player opponents"""
    try:
        data = request.get_json()
        analytics, league, error = get_analytics()
        if not analytics:
            return jsonify({'opponents': [], 'count': 0})
        player = data.get('player', '').strip()
        ptype = data.get('type')
        innings_filter = int(data.get('innings_filter', 0))
        if not player or not ptype:
            return jsonify({'opponents': [], 'count': 0})
        if innings_filter in [1,2]:
            df = analytics.df[analytics.df['innings']==innings_filter]
        else:
            df = analytics.df
        if ptype == 'bowler':
            subset = df[df['bowler'] == player]
            opponents = subset['batsman'].dropna().unique().tolist()
        else:
            subset = df[df['batsman'] == player]
            opponents = subset['bowler'].dropna().unique().tolist()
        return jsonify({'opponents': sorted(opponents), 'count': len(opponents)})
    except Exception as e:
        print(f"Error in get_opponents API: {e}")
        return jsonify({'opponents': [], 'count': 0, 'error': str(e)})

# --- API for Player Quick Stats ---
@app.route('/api/player_stats')
def api_player_stats():
    """API endpoint for player quick stats"""
    try:
        analytics, league, error = get_analytics()
        if not analytics:
            return jsonify({'error': 'No data loaded.'})
        name = request.args.get('name', '').strip()
        ptype = request.args.get('ptype', 'batsman')
        # Accept multiple seasons: either repeated ?season=X or comma-separated
        raw_seasons = request.args.getlist('season')
        seasons_list = []
        for s in raw_seasons:
            for part in s.split(','):
                part = part.strip()
                if part and part != 'all':
                    seasons_list.append(part)
        venue = request.args.get('venue', '').strip()
        if not name:
            return jsonify({'error': 'No player name specified.'})
        # Resolve alias: "Virat Kohli" -> "V Kohli" (global + league-specific)
        league_aliases = get_aliases_for_league(league)
        alias = league_aliases.get(name.lower())
        if alias:
            name = alias
        try:
            # Apply season + venue filters
            filtered_df = analytics.df
            if seasons_list:
                filtered_df = filtered_df[filtered_df['season'].astype(str).isin(seasons_list)]
            if venue and venue != 'all':
                # Expand canonical venue name to all its dataset variants
                venue_map = get_venue_map(league)
                venue_variants = get_venue_variants(venue, venue_map)
                filtered_df = filtered_df[filtered_df['venue'].isin(venue_variants)]
            if seasons_list or (venue and venue != 'all'):
                work = CricketAnalytics.__new__(CricketAnalytics)
                work.df = filtered_df.copy()
                work_df = work.df
            else:
                work = analytics
                work_df = analytics.df

            if ptype == 'batsman':
                stats = work.get_batting_stats(min_innings=0)
                stats['batsman'] = stats['batsman'].astype(str)
                player = stats[stats['batsman'].str.lower() == name.lower()]
                if player.empty:
                    return jsonify({'error': 'Batsman not found.'})
                rec = player.iloc[0]
                balls = int(rec['balls'])
                dismissals = work_df[(work_df['batsman'].str.lower() == name.lower()) & (work_df['player_dismissed'] == name)].shape[0]
                bpd = round(balls / dismissals, 2) if dismissals else "-"
                fours = int(work_df[work_df['batsman'].str.lower() == name.lower()]['isFour'].sum())
                sixes = int(work_df[work_df['batsman'].str.lower() == name.lower()]['isSix'].sum())
                bpb = round(balls / (fours + sixes), 2) if (fours + sixes) else "-"
                rpi_all = float(rec['RPI'])
                rpi_1 = float(rec.get('RPI_1', 0))
                rpi_2 = float(rec.get('RPI_2', 0))
                response = {
                    'resolved_name': name,
                    'matches': int(rec['innings']),
                    'balls': balls,
                    'runs': int(rec['runs']),
                    'avg': float(round(rec['runs']/rec['innings'],2)) if rec['innings']>0 else "-",
                    'sr': float(rec['SR']),
                    'hundreds': int(rec['hundreds']),
                    'fifties': int(rec['fifties']),
                    'hs': int(rec['hs']),
                    'rpi_all': rpi_all,
                    'rpi_1': rpi_1,
                    'rpi_2': rpi_2,
                    'dot_pct': float(rec.get('Dot%', 0)),
                    'bpd': bpd,
                    'bpb': bpb,
                }
            else:
                stats = work.get_bowling_stats(min_innings=0)
                stats['bowler'] = stats['bowler'].astype(str)
                player = stats[stats['bowler'].str.lower() == name.lower()]
                if player.empty:
                    return jsonify({'error': 'Bowler not found.'})
                rec = player.iloc[0]
                balls = int(rec['balls'])
                wickets = int(rec['wickets'])
                sr = round(balls / wickets, 2) if wickets else "-"
                df_player = work_df[work_df['bowler'].str.lower() == name.lower()]
                fours_conc = int(df_player['isFour'].sum())
                sixes_conc = int(df_player['isSix'].sum())
                bpb = round(balls / (fours_conc + sixes_conc), 2) if (fours_conc + sixes_conc) else "-"
                response = {
                    'resolved_name': name,
                    'matches': int(rec['innings']),
                    'balls': balls,
                    'wickets': wickets,
                    'avg': float(rec.get('AVG', 0)),
                    'eco': float(rec.get('ECO', 0)),
                    'sr': sr,
                    'wickets_1': int(rec.get('wickets_1', 0)),
                    'wickets_2': int(rec.get('wickets_2', 0)),
                    'best': int(rec.get('best', 0)),
                    'five_wkts': int(rec.get('five_wkts', 0)),
                    'dot_pct': float(rec.get('Dot%', 0)),
                    'bpb': bpb
                }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': f'Error getting player stats. ({str(e)})'})
    except Exception as e:
        print(f"Error in player_stats API: {e}")
        return jsonify({'error': f'System error: {str(e)}'})

@app.route('/api/get_seasons')
def api_get_seasons():
    """API endpoint to get available seasons for the selected league"""
    try:
        analytics, league, error = get_analytics()
        if not analytics:
            return jsonify({'seasons': []})
        seasons = sorted(
            analytics.df['season'].dropna().astype(str).unique().tolist(),
            reverse=True
        )
        return jsonify({'seasons': seasons})
    except Exception as e:
        print(f"Error in get_seasons API: {e}")
        return jsonify({'seasons': [], 'error': str(e)})

@app.route('/api/get_venues')
def api_get_venues():
    """API endpoint to get available venues for the selected league, normalised via venue_mapping."""
    try:
        analytics, league, error = get_analytics()
        if not analytics:
            return jsonify({'venues': []})
        venue_map = get_venue_map(league)
        raw_venues = analytics.df['venue'].dropna().astype(str).unique().tolist()
        # Map each raw venue to its canonical; deduplicate; sort
        canonical_set = sorted(set(get_canonical_venue(v, venue_map) for v in raw_venues))
        return jsonify({'venues': canonical_set})
    except Exception as e:
        print(f"Error in get_venues API: {e}")
        return jsonify({'venues': [], 'error': str(e)})

@app.route("/venuestats", methods=["GET"])
def venuestats():
    """Venue statistics page"""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        venues, teams = analytics.get_venue_team_options() if analytics else ([], [])
        selected_venue = request.args.get("venue", "")
        selected_team = request.args.get("team", "")
        # Accept multiple seasons; legacy single ?season=all still works
        selected_seasons = [s for s in request.args.getlist("season") if s and s != 'all']
        compare_teams = request.args.getlist("compare_teams")
        team_stats = None
        venue_characteristics = None
        team_comparison = None
        venue_records = None
        seasons = []

        if analytics:
            seasons = sorted(analytics.df['season'].dropna().astype(str).unique().tolist(), reverse=True)
            if selected_seasons:
                work = CricketAnalytics.__new__(CricketAnalytics)
                work.df = analytics.df[analytics.df['season'].astype(str).isin(selected_seasons)].copy()
            else:
                work = analytics
        else:
            work = None

        if work is not None and selected_venue:
            try:
                print(f"Processing venue stats for {selected_venue}")
                # Get venue characteristics
                venue_characteristics = work.get_venue_characteristics(selected_venue)

                # Get venue records
                venue_records = work.get_venue_records(selected_venue)

                # New: recent matches, all-teams summary, top batsmen/bowlers at venue
                recent_matches = work.get_venue_recent_matches(selected_venue, n=10)
                teams_summary = work.get_venue_all_teams_summary(selected_venue)
                top_batsmen = work.get_venue_top_batsmen(selected_venue, n=10)
                top_bowlers = work.get_venue_top_bowlers(selected_venue, n=10)

                # Single team analysis
                if selected_team:
                    print(f"Processing team performance: {selected_team} at {selected_venue}")
                    team_stats = work.get_venue_team_performance(selected_venue, selected_team)
                    if team_stats and team_stats.get('matches', 0) == 0:
                        team_stats = None

                # Multi-team comparison
                if compare_teams and len(compare_teams) >= 2:
                    print(f"Processing team comparison: {compare_teams}")
                    team_comparison = work.get_venue_team_comparison(selected_venue, compare_teams)

                # Memory cleanup
                gc.collect()

            except Exception as e:
                print(f"Error in venue analysis: {e}")
                error = f"Error analyzing venue performance: {str(e)}"
        else:
            recent_matches = []
            teams_summary = []
            top_batsmen = []
            top_bowlers = []

        return render_template(
            "venuestats.html",
            venues=venues,
            teams=teams,
            seasons=seasons,
            selected_venue=selected_venue,
            selected_team=selected_team,
            selected_seasons=selected_seasons,
            compare_teams=compare_teams,
            team_stats=team_stats,
            venue_characteristics=venue_characteristics,
            team_comparison=team_comparison,
            venue_records=venue_records,
            recent_matches=recent_matches,
            teams_summary=teams_summary,
            top_batsmen=top_batsmen,
            top_bowlers=top_bowlers,
            league=league,
            leagues=leagues,
            error=error
        )
    except Exception as e:
        print(f"Error in venuestats route: {e}")
        return render_template(
            "venuestats.html",
            venues=[],
            teams=[],
            seasons=[],
            selected_venue="",
            selected_team="",
            selected_seasons=[],
            compare_teams=[],
            team_stats=None,
            venue_characteristics=None,
            team_comparison=None,
            venue_records=None,
            recent_matches=[],
            teams_summary=[],
            top_batsmen=[],
            top_bowlers=[],
            league=None,
            leagues=available_leagues(),
            error=f"Error loading venue stats: {str(e)}"
        )

@app.route("/phasestats", methods=["GET"])
def phasestats():
    """Phase-wise analysis page (Powerplay / Middle1 / Middle2 / Death)."""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        venues, teams = analytics.get_venue_team_options() if analytics else ([], [])
        selected_venue = request.args.get("venue", "").strip()
        selected_team = request.args.get("team", "").strip()
        selected_seasons = [s for s in request.args.getlist("season") if s and s != 'all']
        seasons = []
        phase_result = None

        if analytics:
            seasons = sorted(analytics.df['season'].dropna().astype(str).unique().tolist(), reverse=True)
            if selected_seasons:
                work = CricketAnalytics.__new__(CricketAnalytics)
                work.df = analytics.df[analytics.df['season'].astype(str).isin(selected_seasons)].copy()
            else:
                work = analytics
        else:
            work = None

        if work is not None and (selected_venue or selected_team or selected_seasons):
            try:
                phase_result = work.get_phase_analysis(
                    venue=selected_venue or None,
                    team=selected_team or None,
                )
                gc.collect()
            except Exception as e:
                print(f"Error in phase analysis: {e}")
                error = f"Error analyzing phases: {str(e)}"

        return render_template(
            "phasestats.html",
            venues=venues,
            teams=teams,
            seasons=seasons,
            selected_venue=selected_venue,
            selected_team=selected_team,
            selected_seasons=selected_seasons,
            phase_result=phase_result,
            league=league,
            leagues=leagues,
            error=error,
        )
    except Exception as e:
        print(f"Error in phasestats route: {e}")
        return render_template(
            "phasestats.html",
            venues=[], teams=[], seasons=[],
            selected_venue="", selected_team="", selected_seasons=[],
            phase_result=None, league=None,
            leagues=available_leagues(),
            error=f"Error loading phase stats: {str(e)}",
        )

@app.route("/teamvsteam", methods=["GET"])
def teamvsteam():
    """Team vs Team head-to-head page."""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        venues, teams = analytics.get_venue_team_options() if analytics else ([], [])
        team_a = request.args.get("team_a", "").strip()
        team_b = request.args.get("team_b", "").strip()
        selected_venue = request.args.get("venue", "").strip()
        try:
            innings_filter = int(request.args.get("innings_filter", 0))
        except (ValueError, TypeError):
            innings_filter = 0
        selected_seasons = [s for s in request.args.getlist("season") if s and s != 'all']
        seasons = []
        result = None

        if analytics:
            seasons = sorted(analytics.df['season'].dropna().astype(str).unique().tolist(), reverse=True)
            if selected_seasons:
                work = CricketAnalytics.__new__(CricketAnalytics)
                work.df = analytics.df[analytics.df['season'].astype(str).isin(selected_seasons)].copy()
            else:
                work = analytics
        else:
            work = None

        if work is not None and team_a and team_b and team_a != team_b:
            try:
                result = work.get_team_vs_team(
                    team_a, team_b,
                    venue=selected_venue or None,
                    innings_filter=innings_filter if innings_filter in (1, 2) else None,
                )
                gc.collect()
            except Exception as e:
                print(f"Error in team vs team: {e}")
                error = f"Error: {str(e)}"

        return render_template(
            "teamvsteam.html",
            venues=venues, teams=teams, seasons=seasons,
            team_a=team_a, team_b=team_b,
            selected_venue=selected_venue,
            selected_seasons=selected_seasons,
            innings_filter=innings_filter,
            result=result,
            league=league, leagues=leagues, error=error,
        )
    except Exception as e:
        print(f"Error in teamvsteam route: {e}")
        return render_template(
            "teamvsteam.html",
            venues=[], teams=[], seasons=[],
            team_a="", team_b="", selected_venue="",
            selected_seasons=[], innings_filter=0, result=None,
            league=None, leagues=available_leagues(),
            error=f"Error: {str(e)}",
        )


@app.route("/winpatterns", methods=["GET"])
def winpatterns():
    """Winning pattern analysis page (score buckets)."""
    try:
        analytics, league, error = get_analytics()
        leagues = available_leagues()
        venues, teams = analytics.get_venue_team_options() if analytics else ([], [])
        selected_team = request.args.get("team", "").strip()
        selected_venue = request.args.get("venue", "").strip()
        selected_seasons = [s for s in request.args.getlist("season") if s and s != 'all']
        seasons = []
        result = None

        if analytics:
            seasons = sorted(analytics.df['season'].dropna().astype(str).unique().tolist(), reverse=True)
            if selected_seasons:
                work = CricketAnalytics.__new__(CricketAnalytics)
                work.df = analytics.df[analytics.df['season'].astype(str).isin(selected_seasons)].copy()
            else:
                work = analytics
        else:
            work = None

        if work is not None and selected_team:
            try:
                result = work.get_winning_patterns(selected_team, venue=selected_venue or None)
                gc.collect()
            except Exception as e:
                print(f"Error in winning patterns: {e}")
                error = f"Error: {str(e)}"

        return render_template(
            "winpatterns.html",
            venues=venues, teams=teams, seasons=seasons,
            selected_team=selected_team, selected_venue=selected_venue,
            selected_seasons=selected_seasons,
            result=result,
            league=league, leagues=leagues, error=error,
        )
    except Exception as e:
        print(f"Error in winpatterns route: {e}")
        return render_template(
            "winpatterns.html",
            venues=[], teams=[], seasons=[],
            selected_team="", selected_venue="", selected_seasons=[],
            result=None,
            league=None, leagues=available_leagues(),
            error=f"Error: {str(e)}",
        )


@app.route("/user_guide")
def user_guide():
    """User guide page"""
    try:
        return render_template("user_guide.html")
    except Exception as e:
        print(f"Error loading user guide: {e}")
        return f"""
        <h1>🏏 User Guide</h1>
        <p>User guide is currently being loaded...</p>
        <p>Please return to <a href="/">home page</a></p>
        <p style="color: #666; font-size: 12px;">Error: {str(e)}</p>
        """

@app.route("/matches")
def matches_page():
    """List only supported league fixtures for the selected date."""
    from datetime import date

    selected_date = request.args.get("date", date.today().isoformat())
    selected_league = get_league()
    return render_template(
        "matches.html",
        matches=fixture_service.list_matches(selected_date, league=selected_league),
        selected_date=selected_date,
        provider_configured=fixture_service.provider.configured,
        league=selected_league,
        leagues=available_leagues(),
    )

def _compute_probable_xi(match_id, fixture):
    """Predict the probable XI (OpenAI) and persist it to the file cache."""
    try:
        ha = fixture.get("hours_to_start")
        if not report_gate_open(fixture):
            return None
        series_id = fixture.get("series_id") or ""
        squads = fixture_service.get_series_squad(series_id)
        if series_id:
            squad_a = [p for p in squads if p.get("teamName") == fixture.get("team_a")]
            squad_b = [p for p in squads if p.get("teamName") == fixture.get("team_b")]
            # CricketData may not publish squad data for a series (e.g. CPL /
            # The Hundred 2026). In that case squad_a/squad_b stay empty and
            # the agent web-searches each team's current squad instead. When
            # provider squads exist they are the ONLY allowed player pool.
            if squad_a and squad_b:
                analytics = get_analytics_for_league(fixture["league"])
                a_analytics = analytics.get_player_form(fixture["team_a"]) if analytics else {}
                b_analytics = analytics.get_player_form(fixture["team_b"]) if analytics else {}
            else:
                analytics = get_analytics_for_league(fixture["league"])
                a_analytics = {}
                b_analytics = {}
            pi = _OAI_AGENT.predict_probable_xi(
                league=fixture["league"],
                team_a=fixture["team_a"],
                team_b=fixture["team_b"],
                start_time_ist=fixture.get("starts_at_ist", ""),
                hours_to_start=ha,
                squad_a=squad_a,
                squad_b=squad_b,
                analytics_a=a_analytics,
                analytics_b=b_analytics,
                enable_web_search=True,
            )
            # Cache successful predictions only. A failure (empty XIs,
            # transport error) must NOT be cached, otherwise the page is
            # stuck showing an empty section for the full 6h TTL.
            if pi and (pi.get("xi_team_a") or pi.get("xi_team_b")):
                _probable_xi_cache.set(match_id, pi)
                return pi
    except Exception as exc:
        print(f"Probable XI prediction failed for {match_id}: {exc}")
    return None


def _compute_match_analysis(match_id, fixture, probable_xi):
    """Compute the per-player analytics and OpenAI match report, persist it."""
    try:
        analytics = get_analytics_for_league(fixture["league"])
        if analytics is None or _OAI_AGENT is None:
            return None
        players = sorted(set(
            analytics.df["batsman"].dropna().astype(str).tolist()
            + analytics.df["bowler"].dropna().astype(str).tolist()
        ))
        # Resolve team names to CSV names so h2h/venue lookups work.
        resolver = entity_resolution_agent
        teams_csv = sorted(set(analytics.df["batting_team"].dropna().astype(str).tolist()))
        team_a_csv = resolver.resolve_team(fixture["team_a"], teams_csv, fixture["league"])["resolved"]
        team_b_csv = resolver.resolve_team(fixture["team_b"], teams_csv, fixture["league"])["resolved"]
        venues_csv = sorted(set(analytics.df["venue"].dropna().astype(str).tolist()))
        venue_csv = resolver.resolve_venue(fixture.get("venue"), venues_csv, fixture["league"])["resolved"]

        bundles = []
        for name in (probable_xi.get("xi_team_a") or []):
            resolved = resolver.resolve_player(name, players, fixture["league"])["resolved"] or name
            bundles.append(analytics.get_player_match_analytics(
                resolved, team=team_b_csv, venue=venue_csv))
            bundles[-1]["probable_name"] = name
            bundles[-1]["team"] = fixture["team_a"]
        for name in (probable_xi.get("xi_team_b") or []):
            resolved = resolver.resolve_player(name, players, fixture["league"])["resolved"] or name
            bundles.append(analytics.get_player_match_analytics(
                resolved, team=team_a_csv, venue=venue_csv))
            bundles[-1]["probable_name"] = name
            bundles[-1]["team"] = fixture["team_b"]

        venue_stats = {}
        if venue_csv:
            vc = analytics.get_venue_characteristics(venue_csv)
            if vc:
                venue_stats = vc
        team_h2h = {}
        if team_a_csv and team_b_csv:
            view = franchise_lineage_agent.analytics_view(
                analytics, fixture["league"], [team_a_csv, team_b_csv])
            team_h2h = view.get_team_vs_team(team_a_csv, team_b_csv) or {}

        match_analysis = _OAI_AGENT.analyze_match_prediction(
            league=fixture["league"],
            team_a=fixture["team_a"],
            team_b=fixture["team_b"],
            venue=fixture.get("venue", ""),
            start_time_ist=fixture.get("starts_at_ist", ""),
            hours_to_start=fixture.get("hours_to_start"),
            probable_xi_a=probable_xi.get("xi_team_a") or [],
            probable_xi_b=probable_xi.get("xi_team_b") or [],
            player_analytics=bundles,
            venue_stats=venue_stats,
            team_h2h=team_h2h,
            enable_web_search=True,
        )
        if match_analysis and not match_analysis.get("failed"):
            _match_analysis_cache.set(match_id, match_analysis)
            return match_analysis
        return match_analysis
    except Exception as exc:
        print(f"Match analysis failed for {match_id}: {exc}")
    return None


def compute_match_report(match_id):
    """Compute (or load from cache) the probable XI + match analysis for a match.

    Returns a dict {"probable_xi": ...|None, "match_analysis": ...|None}. Safe
    to call from the background worker or inline; results persist to disk so a
    later request renders instantly.
    """
    fixture = fixture_service.get_match(match_id)
    if not fixture:
        return {"probable_xi": None, "match_analysis": None}
    fixture = fixture_service.live_match(match_id) or fixture

    probable_xi = _probable_xi_cache.get(match_id)
    match_analysis = _match_analysis_cache.get(match_id)

    # Official Playing XI published -> prediction is no longer relevant.
    if fixture.get("lineups"):
        return {"probable_xi": None, "match_analysis": None}

    if probable_xi is None and report_gate_open(fixture):
        probable_xi = _compute_probable_xi(match_id, fixture)

    if probable_xi and (probable_xi.get("xi_team_a") or probable_xi.get("xi_team_b")):
        if match_analysis is None:
            match_analysis = _compute_match_analysis(match_id, fixture, probable_xi)

    return {"probable_xi": probable_xi, "match_analysis": match_analysis}


def prewarm_reports():
    """Enqueue every upcoming match that is inside the prediction gate and not
    already fully computed, so the report is ready before a user visits."""
    enqueued = 0
    try:
        for match in fixture_service.list_matches():
            if not report_gate_open(match):
                continue
            if _probable_xi_cache.get(match["id"]) and _match_analysis_cache.get(match["id"]):
                continue
            enqueue_report(match["id"])
            enqueued += 1
    except Exception as exc:
        print(f"Pre-warm scan failed: {exc}")
    if enqueued:
        print(f"Pre-warm: enqueued {enqueued} report job(s)")


def _prewarm_loop():
    while True:
        prewarm_reports()
        time.sleep(6 * 3600)  # refresh pre-warm ~4x per day


def _start_background_tasks():
    thread = threading.Thread(target=_prewarm_loop, daemon=True, name="prewarm-loop")
    thread.start()
    return _start_report_worker()


@app.route("/matches/<match_id>")
def match_intelligence_page(match_id):
    """Show cached live facts beside deterministic historical intelligence."""
    fixture = fixture_service.get_match(match_id)
    if not fixture:
        return render_template("404.html"), 404
    # The daily schedule cache can lag behind provider corrections (e.g. a franchise
    # name revised after discovery). Refetch this one match so the header and the
    # historical analysis always use the freshest known team/venue names.
    fixture = fixture_service.live_match(match_id) or fixture
    # CricketData exposes Playing XI on a separate endpoint (/v1/match_squad) that
    # is not covered by live_match's refresh window for pre-toss matches. Do one
    # direct squad lookup on first render so a confirmed XI shows without waiting
    # for the client-side 60s poll.
    if fixture and not fixture.get("lineups") and fixture_service.provider.configured:
        try:
            squad = fixture_service.provider.fetch_squad(str(match_id))
            if squad:
                fixture = dict(fixture)
                fixture["lineups"] = fixture_service._squad_to_lineups(squad)
                # Persist the refreshed lineups back into the live cache so the
                # subsequent /api/v1/matches/<id>/live poll doesn't drop them.
                fixture_service._live_cache[str(match_id)] = (time.time(), fixture)
        except Exception as exc:
            print(f"Squad fetch failed for {match_id}: {exc}")

    # ---- AI report (probable XI + match analysis), instant + background ----
    # Read from the file-backed cache; render immediately. If the report is not
    # cached yet and the match is inside the prediction gate, enqueue the work
    # for the background worker and let the client poll the report endpoint.
    probable_xi = _probable_xi_cache.get(match_id)
    match_analysis = _match_analysis_cache.get(match_id)
    report_pending = (
        report_gate_open(fixture)
        and (probable_xi is None or match_analysis is None)
    )
    if report_pending:
        enqueue_report(match_id)

    cache_key = (fixture.get("team_a"), fixture.get("team_b"), fixture.get("venue"))
    cache_entry = match_intelligence_cache.get(match_id)
    intelligence = None
    if cache_entry and cache_entry.get("key") == cache_key:
        intelligence = cache_entry.get("intelligence")
    if intelligence is None:
        try:
            analytics = get_analytics_for_league(fixture["league"])
            if analytics is not None:
                intelligence = build_match_intelligence(
                    analytics,
                    fixture,
                    resolver=entity_resolution_agent,
                    lineage_agent=franchise_lineage_agent,
                    league=fixture["league"],
                )
                match_intelligence_cache[match_id] = {"key": cache_key, "intelligence": intelligence}
        except Exception as exc:
            print(f"Match intelligence failed for {match_id}: {exc}")
    return render_template(
        "match_intelligence.html",
        fixture=fixture,
        intelligence=intelligence,
        probable_xi=probable_xi,
        match_analysis=match_analysis,
        report_pending=report_pending,
        league=fixture["league"],
        leagues=available_leagues(),
    )

@app.route("/api/v1/matches/today")
def api_matches_today():
    """Return today's supported fixtures from the shared provider cache."""
    league = request.args.get("league")
    matches = fixture_service.today(league=league)
    return jsonify({"matches": matches, "count": len(matches)})

@app.route("/api/v1/matches/<match_id>/live")
def api_match_live(match_id):
    """Return one supported match's cached live state."""
    match = fixture_service.live_match(match_id)
    if not match:
        return jsonify({"error": "Supported match not found."}), 404
    response = jsonify(match)
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/api/v1/matches/<match_id>/report")
def api_match_report(match_id):
    """Return the AI report status for a match, for client-side polling.

    status: "ready" (report fully cached), "computing" (work enqueued/running),
    "unavailable" (out of prediction gate or no OpenAI configured).
    """
    probable_xi = _probable_xi_cache.get(match_id)
    match_analysis = _match_analysis_cache.get(match_id)
    if probable_xi and match_analysis:
        return jsonify({
            "status": "ready",
            "probable_xi": probable_xi,
            "match_analysis": match_analysis,
        })
    fixture = fixture_service.get_match(match_id)
    if fixture and report_gate_open(fixture):
        enqueue_report(match_id)
        return jsonify({
            "status": "computing",
            "probable_xi": probable_xi,
            "match_analysis": match_analysis,
        })
    return jsonify({
        "status": "unavailable",
        "probable_xi": probable_xi,
        "match_analysis": match_analysis,
    })

# Health check endpoint for monitoring
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'matchup-matrix'}), 200

# --- PWA support (static file serving only — no analytics logic) ---
@app.route('/sw.js')
def service_worker():
    """Serve the service worker from root so it can control the whole app."""
    resp = send_from_directory('static/js', 'sw.js', mimetype='application/javascript')
    resp.headers['Service-Worker-Allowed'] = '/'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp

@app.route('/manifest.webmanifest')
def web_manifest():
    """Serve the PWA manifest."""
    return send_from_directory('static', 'manifest.webmanifest',
                               mimetype='application/manifest+json')

# Simple test route for debugging
@app.route('/test')
def test():
    """Simple test endpoint"""
    return "🎉 Flask app is working! All systems operational."

# Debug endpoint to check data loading
@app.route('/debug')
def debug():
    """Debug endpoint to inspect loaded data"""
    try:
        analytics, league, error = get_analytics()
        if analytics:
            summary = analytics.get_data_summary()
            sample_data = analytics.df.head(5).to_dict('records') if hasattr(analytics, 'df') else []
            columns = list(analytics.df.columns) if hasattr(analytics, 'df') else []
            return jsonify({
                'status': 'success',
                'summary': summary,
                'columns': columns,
                'sample_data': sample_data,
                'league': league,
                'data_shape': analytics.df.shape if hasattr(analytics, 'df') else [0, 0]
            })
        else:
            return jsonify({
                'status': 'error',
                'error': error,
                'league': league,
                'available_leagues': list(available_leagues().keys())
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'available_leagues': list(available_leagues().keys())
        })

# Status endpoint with memory info
@app.route('/status')
def status():
    """Status endpoint with system information"""
    try:
        # Try to import psutil for memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                'memory_percent': memory.percent,
                'memory_available': f"{memory.available / (1024**3):.2f} GB"
            }
        except ImportError:
            memory_info = {'memory_info': 'psutil not available'}
        
        return jsonify({
            'status': 'operational',
            'leagues_available': len(available_leagues()),
            'analytics_cached': len(analytics_cache),
            'python_version': sys.version,
            **memory_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# Start the background AI-report worker + pre-warm loop (daemon threads). They
# are harmless if the app is imported for tests or by a WSGI server.
_start_background_tasks()

# PythonAnywhere specific configuration
if __name__ == "__main__":
    # For local development only
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT, debug=True)
