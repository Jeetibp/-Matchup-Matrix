
from flask import Flask, render_template, request, jsonify, session
from cricket_analytics_core import CricketAnalytics
import os

app = Flask(__name__)
app.secret_key = 't20blast2025'
app.config['TEMPLATES_AUTO_RELOAD'] = True

LEAGUE_CSVS = {
    't20blast': 'data/all_matches_t20blast.csv',
    'mcl':     'data/all_matches_mcl.csv',
    'ipl':     'data/all_matches_ipl.csv',
    'the100':  'data/all_matches_the100.csv',
    'cpl':     'data/all_matches_cpl.csv',
    'the100women': 'data/all_matches_the100women.csv'
}

analytics_cache = {}

def available_leagues():
    return {k: v for k, v in LEAGUE_CSVS.items() if os.path.exists(v)}

def get_league():
    avail = available_leagues()
    league = request.args.get('league') or session.get('league') or 't20blast'
    if league not in avail:
        league = list(avail.keys())[0] if avail else None
    session['league'] = league
    return league

def get_analytics():
    avail = available_leagues()
    league = get_league()
    if league and league in avail:
        if league in analytics_cache:
            analytics = analytics_cache[league]
        else:
            try:
                analytics = CricketAnalytics(avail[league])
                analytics_cache[league] = analytics
            except Exception as e:
                return None, league, f"Error loading data for league: {league.upper()}<br>{e}"
        return analytics, league, None
    else:
        return None, league, f"No data available for selected league."

@app.route('/')
def home():
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
    top_bat_all = analytics.get_batting_stats().head(1)
    top_bat_1 = analytics.get_batting_stats(innings_filter=1).head(1)
    top_bat_2 = analytics.get_batting_stats(innings_filter=2).head(1)
    top_bowl_all = analytics.get_bowling_stats().head(1)
    top_bowl_1 = analytics.get_bowling_stats(innings_filter=1).head(1)
    top_bowl_2 = analytics.get_bowling_stats(innings_filter=2).head(1)
    return render_template(
        "home.html",
        total_players=len(analytics.get_batting_stats()),
        total_bowlers=len(analytics.get_bowling_stats()),
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

@app.route("/batting")
def batting():
    analytics, league, error = get_analytics()
    leagues = available_leagues()
    min_innings = request.args.get("min_innings", 5, type=int)
    innings_filter = request.args.get("innings_filter", 0, type=int)
    filter_val = innings_filter if innings_filter in [1,2] else None
    stats = analytics.get_batting_stats(min_innings, innings_filter=filter_val) if analytics else []
    return render_template(
        "batting.html",
        stats=stats.to_dict("records") if analytics else [],
        min_innings=min_innings,
        innings_filter=innings_filter,
        league=league,
        leagues=leagues,
        error=error
    )

@app.route("/bowling")
def bowling():
    analytics, league, error = get_analytics()
    leagues = available_leagues()
    min_innings = request.args.get("min_innings", 3, type=int)
    innings_filter = request.args.get("innings_filter", 0, type=int)
    filter_val = innings_filter if innings_filter in [1,2] else None
    stats = analytics.get_bowling_stats(min_innings, innings_filter=filter_val) if analytics else []
    return render_template(
        "bowling.html",
        stats=stats.to_dict("records") if analytics else [],
        min_innings=min_innings,
        innings_filter=innings_filter,
        league=league,
        leagues=leagues,
        error=error
    )

@app.route("/headtohead", methods=["GET", "POST"])
def headtohead():
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

    innings_list = [innings_filter] if innings_filter in [1,2] else [1,2]
    all_bowlers = sorted(analytics.df[analytics.df["innings"].isin(innings_list)]["bowler"].dropna().unique())
    all_batsmen = sorted(analytics.df[analytics.df["innings"].isin(innings_list)]["batsman"].dropna().unique())

    if request.method == "POST":
        atype = request.form.get("analysis_type", "single")
        if atype == "single":
            b = request.form.get("bowler", "").strip()
            bt = request.form.get("batsman", "").strip()
            session['h2h_inputs']['single_bowler'] = b
            session['h2h_inputs']['single_batsman'] = bt
            session['h2h_inputs']['innings_filter'] = innings_filter
            if b and bt:
                matchup = analytics.get_head_to_head(b, bt, innings_filter=innings_filter)
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
                multiple = analytics.get_multiple_head_to_head(bs, bts, innings_filter=innings_filter)
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
            matchup = analytics.get_head_to_head(
                saved_inputs["single_bowler"], saved_inputs["single_batsman"], innings_filter=saved_inputs.get("innings_filter")
            )

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

# --- API for Fuzzy Player Suggestions ---
@app.route('/api/player_fuzzy')
def api_player_fuzzy():
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
    results = [p for p in players if q in p.lower()]
    return jsonify({'players': results[:20]})

# --- API for Opponent Filtering for Dropdown (Smart Filter) ---
@app.route('/api/get_opponents', methods=["POST"])
def api_get_opponents():
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

# --- API for Player Quick Stats ---
@app.route('/api/player_stats')
def api_player_stats():
    analytics, league, error = get_analytics()
    if not analytics:
        return jsonify({'error': 'No data loaded.'})
    name = request.args.get('name', '').strip()
    ptype = request.args.get('ptype', 'batsman')
    if not name:
        return jsonify({'error': 'No player name specified.'})
    try:
        if ptype == 'batsman':
            stats = analytics.get_batting_stats(min_innings=0)
            stats['batsman'] = stats['batsman'].astype(str)
            player = stats[stats['batsman'].str.lower() == name.lower()]
            if player.empty:
                return jsonify({'error': 'Batsman not found.'})
            rec = player.iloc[0]
            balls = int(rec['balls'])
            dismissals = analytics.df[(analytics.df['batsman'].str.lower() == name.lower()) & (analytics.df['player_dismissed'] == name)].shape[0]
            bpd = round(balls / dismissals, 2) if dismissals else "-"
            fours = int(analytics.df[analytics.df['batsman'].str.lower() == name.lower()]['isFour'].sum())
            sixes = int(analytics.df[analytics.df['batsman'].str.lower() == name.lower()]['isSix'].sum())
            bpb = round(balls / (fours + sixes), 2) if (fours + sixes) else "-"
            rpi_all = float(rec['RPI'])
            rpi_1 = float(rec.get('RPI_1', 0))
            rpi_2 = float(rec.get('RPI_2', 0))
            response = {
                'matches': int(rec['innings']),
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
            stats = analytics.get_bowling_stats(min_innings=0)
            stats['bowler'] = stats['bowler'].astype(str)
            player = stats[stats['bowler'].str.lower() == name.lower()]
            if player.empty:
                return jsonify({'error': 'Bowler not found.'})
            rec = player.iloc[0]
            balls = int(rec['balls'])
            wickets = int(rec['wickets'])
            sr = round(balls / wickets, 2) if wickets else "-"
            df_player = analytics.df[analytics.df['bowler'].str.lower() == name.lower()]
            fours_conc = int(df_player['isFour'].sum())
            sixes_conc = int(df_player['isSix'].sum())
            bpb = round(balls / (fours_conc + sixes_conc), 2) if (fours_conc + sixes_conc) else "-"
            response = {
                'matches': int(rec['innings']),
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

@app.route("/venuestats", methods=["GET"])
def venuestats():
    analytics, league, error = get_analytics()
    leagues = available_leagues()
    venues, teams = analytics.get_venue_team_options() if analytics else ([], [])
    selected_venue = request.args.get("venue", "")
    selected_team = request.args.get("team", "")
    compare_teams = request.args.getlist("compare_teams")
    team_stats = None
    venue_characteristics = None
    team_comparison = None
    venue_records = None
    
    if analytics and selected_venue:
        try:
            # Get venue characteristics
            venue_characteristics = analytics.get_venue_characteristics(selected_venue)
            
            # Get venue records
            venue_records = analytics.get_venue_records(selected_venue)
            
            # Single team analysis
            if selected_team:
                team_stats = analytics.get_venue_team_performance(selected_venue, selected_team)
                if team_stats and team_stats.get('matches', 0) == 0:
                    team_stats = None
                    
            # Multi-team comparison
            if compare_teams and len(compare_teams) >= 2:
                team_comparison = analytics.get_venue_team_comparison(selected_venue, compare_teams)
                
        except Exception as e:
            error = f"Error analyzing venue performance: {str(e)}"

    return render_template(
        "venuestats.html",
        venues=venues,
        teams=teams,
        selected_venue=selected_venue,
        selected_team=selected_team,
        compare_teams=compare_teams,
        team_stats=team_stats,
        venue_characteristics=venue_characteristics,
        team_comparison=team_comparison,
        venue_records=venue_records,
        league=league,
        leagues=leagues,
        error=error
    )

# ----> ADD USER GUIDE ROUTE HERE <----
@app.route("/user_guide")
def user_guide():
    return render_template("user_guide.html")
    
if __name__ == "__main__":
    print("🟢 app.py loaded!")
    print("🏏 Flask server starting at http://localhost:5000/")
    app.run(port=5000, debug=True)
