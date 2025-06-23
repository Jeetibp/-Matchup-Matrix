import pandas as pd
import numpy as np
import gc

from datetime import datetime
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
pd.set_option('future.no_silent_downcasting', True)

class CricketAnalytics:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
        self.optimize_memory()

    def optimize_memory(self):
        """Optimize DataFrame memory usage"""
        df = self.df
        
        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Clear memory
        gc.collect()
        self.df = df

    def prepare_data(self):
        df = self.df
        df = df.rename(columns={
            'striker': 'batsman',
            'runs_off_bat': 'runs_of_bat',
            'ball': 'over',
            'wicket_type': 'dismissal_kind'
        })
        df['innings'] = df['innings'].astype(int)
        df['wides'] = df['wides'].fillna(0)
        df['noballs'] = df['noballs'].fillna(0)
        df['isDot'] = (df['runs_of_bat']==0).astype(int)
        df['isFour'] = (df['runs_of_bat']==4).astype(int)
        df['isSix'] = (df['runs_of_bat']==6).astype(int)
        df['total_run'] = df['runs_of_bat'] + df['wides'] + df['noballs']
        df['total_runs'] = df['runs_of_bat'] + df['extras']
        df['isBowlerWk'] = df.apply(
            lambda x: 1 if pd.notna(x['player_dismissed']) and x['dismissal_kind'] not in ['run out','retired hurt','retired out'] else 0,
            axis=1
        )
        self.df = df

    def get_batting_stats(self, min_innings=5, innings_filter=None):
        df = self.df
        if innings_filter in [1,2]:
            df = df[df['innings'] == innings_filter]
        # Per-match group for 100s, 50s, highest
        match_runs = df.groupby(['batsman', 'match_id'])['runs_of_bat'].sum().reset_index()
        batsman_match_scores = match_runs.groupby('batsman')['runs_of_bat'].agg(list)
        hundreds = batsman_match_scores.apply(lambda scores: sum(1 for s in scores if s >= 100))
        fifties = batsman_match_scores.apply(lambda scores: sum(1 for s in scores if 50 <= s < 100))
        highest_score = batsman_match_scores.apply(lambda scores: max(scores) if scores else 0)
        # Main stats
        runs = df.groupby('batsman')['runs_of_bat'].sum()
        balls = df.groupby('batsman').size()
        inns = df.groupby('batsman')['match_id'].nunique()
        fours = df.groupby('batsman')['isFour'].sum()
        sixes = df.groupby('batsman')['isSix'].sum()
        dot_pct = df.groupby('batsman')['isDot'].sum() / balls * 100
        boundary_pct = (fours + sixes) / balls * 100
        # Dismissals, BPD, BPB
        dismissals = df[df['player_dismissed'] == df['batsman']].groupby('batsman')['player_dismissed'].count()
        bpd = balls / dismissals.replace(0, pd.NA)
        bpb = balls / (fours + sixes).replace(0, pd.NA)
        # RPI
        rpi_all = runs / inns
        rpi_1 = df[df['innings']==1].groupby('batsman')['runs_of_bat'].sum() / df[df['innings']==1].groupby('batsman')['match_id'].nunique()
        rpi_2 = df[df['innings']==2].groupby('batsman')['runs_of_bat'].sum() / df[df['innings']==2].groupby('batsman')['match_id'].nunique()
        stats = pd.DataFrame({
            'batsman': runs.index,
            'runs': runs.values,
            'innings': inns.values,
            'balls': balls.values,
            'SR': (runs / balls * 100).round(2),
            'hundreds': hundreds,
            'fifties': fifties,
            'hs': highest_score,
            'RPI': rpi_all.round(2),
            'RPI_1': rpi_1.round(2).fillna(0),
            'RPI_2': rpi_2.round(2).fillna(0),
            'Dot%': dot_pct.round(2).fillna(0),
            'Boundary%': boundary_pct.round(2).fillna(0),
            'BPD': bpd.round(2).fillna(0),
            'BPB': bpb.round(2).fillna(0).infer_objects(copy=False).astype(int),
        })
        stats = stats[stats['innings']>=min_innings].fillna(0).sort_values('runs',ascending=False).reset_index(drop=True)
        # Clear memory after processing
        gc.collect()
        return stats

    def get_bowling_stats(self, min_innings=3, innings_filter=None):
        df = self.df
        if innings_filter in [1,2]:
            df = df[df['innings'] == innings_filter]
        # Best and 5W+
        match_wkts = df.groupby(['bowler', 'match_id'])['isBowlerWk'].sum().reset_index()
        best = match_wkts.groupby('bowler')['isBowlerWk'].max()
        five_wkts = match_wkts.groupby('bowler')['isBowlerWk'].apply(lambda x: sum(x >= 5))
        # Wickets split by innings
        wickets_1 = df[df['innings']==1].groupby('bowler')['isBowlerWk'].sum()
        wickets_2 = df[df['innings']==2].groupby('bowler')['isBowlerWk'].sum()
        # Main stats
        runs = df.groupby('bowler')['total_run'].sum()
        balls = df.groupby('bowler').size()
        inns = df.groupby('bowler')['match_id'].nunique()
        wickets = df.groupby('bowler')['isBowlerWk'].sum()
        dots = df.groupby('bowler')['isDot'].sum()
        eco = runs / (balls / 6)
        dot_pct = dots / balls * 100
        avg = (runs / wickets).replace([float('inf'), float('nan')], 0)
        # ----> Add Bowler SR: Balls/Wickets <----
        sr = (balls / wickets).replace([float('inf'), float('nan')], 0).round(2)
        stats = pd.DataFrame({
            'bowler': runs.index,
            'innings': inns.values,
            'balls': balls.values,
            'runs': runs.values,
            'wickets': wickets.values,
            'ECO': eco.round(2),
            'AVG': avg.round(2).fillna(0),
            'SR': sr.values,
            'Dot%': dot_pct.round(2).fillna(0),
            'wickets_1': wickets_1.reindex(runs.index, fill_value=0).astype(int),
            'wickets_2': wickets_2.reindex(runs.index, fill_value=0).astype(int),
            'best': best.reindex(runs.index, fill_value=0).astype(int),
            'five_wkts': five_wkts.reindex(runs.index, fill_value=0).astype(int),
        })
        stats = stats[stats['innings']>=min_innings].sort_values('wickets',ascending=False).reset_index(drop=True)
        # Clear memory after processing
        gc.collect()
        return stats

    def get_head_to_head(self, bowler, batsman, innings_filter=None):
        df = self.df[(self.df['bowler'] == bowler) & (self.df['batsman'] == batsman)]
        if innings_filter in [1,2]:
            df = df[df['innings'] == innings_filter]
        if df.empty: return None
        total_balls = len(df)
        total_runs  = int(df['runs_of_bat'].sum())
        wickets     = int(df['isBowlerWk'].sum())
        dot_balls   = int(df['isDot'].sum())
        strike_rate = round(100*total_runs/total_balls,2) if total_balls>0 else 0
        economy     = round(df['total_run'].sum()/(total_balls/6),2) if total_balls>0 else 0
        dot_pct     = round(100*dot_balls/total_balls,2) if total_balls>0 else 0
        matches     = df['match_id'].nunique()
        return {
            'bowler':bowler,'batsman':batsman,
            'balls':total_balls,'runs':total_runs,'wickets':wickets,
            'dot_balls':dot_balls,'strike_rate':strike_rate,
            'economy':economy,'dot_percentage':dot_pct,
            'matches':matches,'dismissed':'Yes' if wickets>0 else 'No'
        }

    def get_multiple_head_to_head(self, bowlers, batsmen, innings_filter=None):
        results = []
        for bowler in bowlers:
            for batsman in batsmen:
                matchup = self.get_head_to_head(bowler, batsman, innings_filter=innings_filter)
                if matchup is None:
                    matchup = {
                        'bowler': bowler,
                        'batsman': batsman,
                        'balls': None,
                        'runs': None,
                        'wickets': None,
                        'strike_rate': None,
                        'economy': None,
                        'matchup_found': False
                    }
                else:
                    matchup['matchup_found'] = True
                results.append(matchup)
        return results

    def get_player_opponents(self, player, ptype='bowler', innings_filter=None):
        df = self.df
        if innings_filter in [1,2]:
            df = df[df['innings'] == innings_filter]
        if ptype=='bowler':
            opps = df[df['bowler']==player]['batsman'].dropna().unique()
        else:
            opps = df[df['batsman']==player]['bowler'].dropna().unique()
        return sorted(opps.tolist())

    def search_players(self, query, ptype='both', limit=10, innings_filter=None):
        df = self.df
        if innings_filter in [1,2]:
            df = df[df['innings'] == innings_filter]
        q = query.lower()
        out = []
        if ptype in ['bowler','both']:
            bs = df['bowler'].dropna().unique()
            mb = [b for b in bs if q in b.lower()]
            out.extend([{'name':b,'type':'bowler','match_type':'exact' if b.lower().startswith(q) else 'contains'} for b in sorted(mb)[:limit]])
        if ptype in ['batsman','both']:
            bs = df['batsman'].dropna().unique()
            mb = [b for b in bs if q in b.lower()]
            out.extend([{'name':b,'type':'batsman','match_type':'exact' if b.lower().startswith(q) else 'contains'} for b in sorted(mb)[:limit]])
        return out[:limit]

    def get_venue_team_options(self):
        '''
        Get list of all venues and teams for dropdown options.
        Returns: (venues_list, teams_list)
        '''
        venues = sorted(self.df['venue'].dropna().unique().tolist())
        teams = sorted(self.df['batting_team'].dropna().unique().tolist())
        return venues, teams

    def get_venue_team_performance(self, venue_name, team_name):
        '''
        Comprehensive team performance analysis at a specific venue.
        Returns detailed statistics similar to the provided analysis code.
        '''
        # Filter dataset for matches played at the given venue
        venue_matches = self.df[self.df['venue'] == venue_name]
        
        # Filter for matches where the given team was the batting team
        team_matches_venue = venue_matches[venue_matches['batting_team'] == team_name]
        
        if team_matches_venue.empty:
            return {
                'venue': venue_name,
                'team': team_name,
                'matches': 0,
                'avg_innings_1': 0,
                'avg_innings_2': 0,
                'overall_avg': 0,
                'HS': 0,
                'LS': 0,
                'HC': 'N/A',
                'LD': 'N/A',
                'first_bat_wins': 0,
                'second_bat_wins': 0,
                'win_pct_1st': 0,
                'win_pct_2nd': 0
            }
        
        # Count the number of matches the team played at the venue
        team_match_count = team_matches_venue['match_id'].nunique()
        
        # Compute total runs per innings for the team
        team_innings_stats = (
            team_matches_venue.groupby(['match_id', 'innings'])['total_runs']
            .sum()
            .unstack(fill_value=0)
        )
        
        # Handle innings stats with proper Series handling
        team_total_innings_1 = team_innings_stats.get(1, pd.Series(dtype=float)).sum()
        team_total_innings_2 = team_innings_stats.get(2, pd.Series(dtype=float)).sum()
        
        # Count how many times the team batted first or second
        team_bat_1st_count = team_innings_stats.get(1, pd.Series(dtype=float)).astype(bool).sum()
        team_bat_2nd_count = team_innings_stats.get(2, pd.Series(dtype=float)).astype(bool).sum()
        
        # Compute average runs per innings
        team_avg_innings_1 = team_total_innings_1 / team_bat_1st_count if team_bat_1st_count > 0 else 0
        team_avg_innings_2 = team_total_innings_2 / team_bat_2nd_count if team_bat_2nd_count > 0 else 0
        team_total_runs = team_total_innings_1 + team_total_innings_2
        
        # Compute Highest & Lowest Score (HS & LS)
        if not team_innings_stats.empty:
            team_HS = team_innings_stats.max().max()
            team_LS = team_innings_stats.replace(0, np.inf).min().min()
            if team_LS == np.inf:
                team_LS = 0
        else:
            team_HS = 0
            team_LS = 0
        
        # Calculate wins and determine HC/LD based on match results
        team_match_results = []
        
        for match_id in team_matches_venue['match_id'].unique():
            match_data = venue_matches[venue_matches['match_id'] == match_id]
            
            # Get innings totals for this match
            innings_totals = match_data.groupby(['innings', 'batting_team'])['total_runs'].sum().reset_index()
            
            if len(innings_totals) >= 2:
                inn1_data = innings_totals[innings_totals['innings'] == 1]
                inn2_data = innings_totals[innings_totals['innings'] == 2]
                
                if not inn1_data.empty and not inn2_data.empty:
                    inn1_score = inn1_data['total_runs'].iloc[0]
                    inn2_score = inn2_data['total_runs'].iloc[0]
                    inn1_team = inn1_data['batting_team'].iloc[0]
                    inn2_team = inn2_data['batting_team'].iloc[0]
                    
                    # Determine winner
                    if inn1_score > inn2_score:
                        winner = inn1_team
                        result_type = "runs"
                    else:
                        winner = inn2_team
                        result_type = "wickets"
                    
                    # Check if our team was involved and won
                    if team_name == inn1_team:
                        team_score = inn1_score
                        team_innings = 1
                        team_won = (winner == team_name)
                    elif team_name == inn2_team:
                        team_score = inn2_score
                        team_innings = 2
                        team_won = (winner == team_name)
                    else:
                        continue
                    
                    team_match_results.append({
                        'match_id': match_id,
                        'team_score': team_score,
                        'team_innings': team_innings,
                        'team_won': team_won,
                        'result_type': result_type
                    })
        
        # Calculate HC and LD
        team_HC = "N/A"
        team_LD = "N/A"
        
        if team_match_results:
            # Highest Chase (when team batted 2nd and won)
            successful_chases = [r['team_score'] for r in team_match_results 
                               if r['team_innings'] == 2 and r['team_won']]
            if successful_chases:
                team_HC = max(successful_chases)
            
            # Lowest Defended (when team batted 1st and won)
            successful_defenses = [r['team_score'] for r in team_match_results 
                                 if r['team_innings'] == 1 and r['team_won']]
            if successful_defenses:
                team_LD = min(successful_defenses)
        
        # Calculate wins when batting first and second
        team_1st_bat_wins = len([r for r in team_match_results 
                               if r['team_innings'] == 1 and r['team_won']])
        team_2nd_bat_wins = len([r for r in team_match_results 
                               if r['team_innings'] == 2 and r['team_won']])
        
        # Calculate overall average
        team_overall_avg = team_total_runs / (team_bat_1st_count + team_bat_2nd_count) if (team_bat_1st_count + team_bat_2nd_count) > 0 else 0
        
        # Calculate win percentages
        team_1st_bat_win_percentage = (team_1st_bat_wins / team_bat_1st_count) * 100 if team_bat_1st_count > 0 else 0
        team_2nd_bat_win_percentage = (team_2nd_bat_wins / team_bat_2nd_count) * 100 if team_bat_2nd_count > 0 else 0
        
        result = {
            'venue': venue_name,
            'team': team_name,
            'matches': team_match_count,
            'avg_innings_1': round(team_avg_innings_1, 2),
            'avg_innings_2': round(team_avg_innings_2, 2),
            'overall_avg': round(team_overall_avg, 2),
            'HS': int(team_HS),
            'LS': int(team_LS),
            'HC': team_HC if team_HC == 'N/A' else int(team_HC),
            'LD': team_LD if team_LD == 'N/A' else int(team_LD),
            'first_bat_wins': team_1st_bat_wins,
            'second_bat_wins': team_2nd_bat_wins,
            'win_pct_1st': round(team_1st_bat_win_percentage, 2),
            'win_pct_2nd': round(team_2nd_bat_win_percentage, 2)
        }
        
        # Clear memory after processing
        gc.collect()
        return result

    def get_venue_characteristics(self, venue_name):
        '''
        Analyze venue characteristics and overall patterns
        '''
        venue_matches = self.df[self.df['venue'] == venue_name]
        
        if venue_matches.empty:
            return None
            
        # Calculate innings totals for each match
        match_innings_stats = venue_matches.groupby(['match_id', 'innings'])['total_runs'].sum().unstack(fill_value=0)
        
        # Venue characteristics
        total_matches = len(match_innings_stats)
        avg_1st_innings = match_innings_stats.get(1, pd.Series(dtype=float)).mean()
        avg_2nd_innings = match_innings_stats.get(2, pd.Series(dtype=float)).mean()
        
        # Chase success rate
        successful_chases = (match_innings_stats.get(2, pd.Series(dtype=float)) > match_innings_stats.get(1, pd.Series(dtype=float))).sum()
        chase_success_rate = (successful_chases / total_matches * 100) if total_matches > 0 else 0
        
        # Boundary analysis
        total_fours = venue_matches['isFour'].sum()
        total_sixes = venue_matches['isSix'].sum()
        total_balls = len(venue_matches)
        boundary_rate = ((total_fours + total_sixes) / total_balls * 100) if total_balls > 0 else 0
        
        # High scoring vs low scoring
        high_scores = (match_innings_stats >= 150).sum().sum()
        low_scores = (match_innings_stats <= 120).sum().sum()
        
        result = {
            'venue': venue_name,
            'total_matches': total_matches,
            'avg_1st_innings': round(avg_1st_innings, 2),
            'avg_2nd_innings': round(avg_2nd_innings, 2),
            'chase_success_rate': round(chase_success_rate, 2),
            'boundary_rate': round(boundary_rate, 2),
            'high_scores': int(high_scores),
            'low_scores': int(low_scores),
            'total_fours': int(total_fours),
            'total_sixes': int(total_sixes)
        }
        
        # Clear memory after processing
        gc.collect()
        return result

    def get_venue_team_comparison(self, venue_name, teams_list):
        '''
        Compare multiple teams performance at the same venue
        '''
        if len(teams_list) < 2:
            return []
            
        comparison_results = []
        for team in teams_list:
            team_stats = self.get_venue_team_performance(venue_name, team)
            if team_stats and team_stats.get('matches', 0) > 0:
                comparison_results.append(team_stats)
                
        return comparison_results

    def get_venue_records(self, venue_name):
        '''
        Get record holders at a specific venue
        '''
        venue_matches = self.df[self.df['venue'] == venue_name]
        
        if venue_matches.empty:
            return None
            
        # Highest individual score by batsman
        batsman_scores = venue_matches.groupby(['batsman', 'match_id'])['runs_of_bat'].sum()
        highest_individual = batsman_scores.max()
        highest_scorer = batsman_scores.idxmax()[0] if not batsman_scores.empty else "N/A"
        
        # Best bowling figures
        bowler_wickets = venue_matches.groupby(['bowler', 'match_id'])['isBowlerWk'].sum()
        best_bowling = bowler_wickets.max()
        best_bowler = bowler_wickets.idxmax()[0] if not bowler_wickets.empty else "N/A"
        
        # Most sixes in innings
        sixes_per_match = venue_matches.groupby(['batting_team', 'match_id', 'innings'])['isSix'].sum()
        most_sixes = sixes_per_match.max()
        
        result = {
            'venue': venue_name,
            'highest_individual_score': int(highest_individual) if highest_individual else 0,
            'highest_scorer': highest_scorer,
            'best_bowling_figures': int(best_bowling) if best_bowling else 0,
            'best_bowler': best_bowler,
            'most_sixes_innings': int(most_sixes) if most_sixes else 0
        }
        
        # Clear memory after processing
        gc.collect()
        return result
