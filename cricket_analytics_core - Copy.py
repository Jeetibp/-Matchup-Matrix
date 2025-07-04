import pandas as pd
import numpy as np
import gc
import os
import warnings
from datetime import datetime

# Suppress ALL pandas warnings for production
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('mode.chained_assignment', None)

# Try to import psutil, make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class CricketAnalytics:
    def __init__(self, csv_file):
        try:
            print("Starting Cricket Analytics initialization with FULL DATASET...")
            # Load complete dataset - NO RESTRICTIONS
            self.df = self._load_csv_optimized(csv_file)
            self.prepare_data()
            self.optimize_memory()
            
            matches = self.df['match_id'].nunique()
            players = self.df['batsman'].nunique()
            print(f"✅ Successfully loaded {len(self.df)} rows, {matches} matches, {players} players")
            self._monitor_memory("After initialization")
            
        except MemoryError:
            print("Memory constraint detected, using fallback loading...")
            self.df = self._load_csv_fallback(csv_file)
            self.prepare_data()
            self.optimize_memory()
        except Exception as e:
            print(f"Error loading cricket data: {e}")
            # Create minimal fallback dataset to prevent crashes
            self._create_fallback_data()

    def _create_fallback_data(self):
        """Create fallback data if loading fails"""
        print("Creating fallback dataset...")
        self.df = pd.DataFrame({
            'batsman': ['Sample Player 1', 'Sample Player 2', 'Sample Player 3'],
            'bowler': ['Sample Bowler 1', 'Sample Bowler 2', 'Sample Bowler 3'],
            'runs_of_bat': [25, 30, 15],
            'innings': [1, 2, 1],
            'match_id': ['sample_match_1', 'sample_match_2', 'sample_match_3'],
            'venue': ['Sample Venue'],
            'batting_team': ['Sample Team A', 'Sample Team B', 'Sample Team A'],
            'player_dismissed': [None, 'Sample Player 1', None],
            'dismissal_kind': [None, 'bowled', None],
            'wides': [0, 1, 0],
            'noballs': [0, 0, 1],
            'extras': [0, 1, 1],
            'isDot': [0, 0, 1],
            'isFour': [1, 1, 0],
            'isSix': [1, 0, 0],
            'total_run': [25, 31, 16],
            'total_runs': [25, 31, 16],
            'isBowlerWk': [0, 1, 0]
        })
        print("Fallback dataset created with sample cricket data")

    def _load_csv_optimized(self, csv_file):
        """Load COMPLETE CSV dataset - NO RESTRICTIONS for PythonAnywhere"""
        try:
            print(f"Loading FULL dataset from {csv_file}...")
            
            # COMPLETE DATASET LOADING - NO LIMITS!
            print("PythonAnywhere environment - loading COMPLETE dataset with NO restrictions")
            df = pd.read_csv(csv_file, low_memory=True)  # NO nrows parameter - load everything!
            
            # Print actual columns for debugging
            print(f"CSV loaded with columns: {list(df.columns)}")
            print(f"DataFrame shape: {df.shape}")
            print(f"✅ FULL DATASET LOADED: {len(df)} total rows!")
            
            # Force garbage collection after loading
            gc.collect()
            return df
            
        except Exception as e:
            print(f"CSV loading error: {e}")
            raise

    def _load_csv_fallback(self, csv_file):
        """Fallback still loads substantial data"""
        print("Using fallback loading with substantial data...")
        try:
            # Even fallback loads much more data
            df = pd.read_csv(csv_file, nrows=100000, low_memory=True)  # Much higher fallback limit
            print(f"Fallback loaded {len(df)} rows successfully")
            return df
        except Exception as e:
            print(f"Fallback error: {e}")
            # Final fallback
            return pd.read_csv(csv_file, nrows=20000, low_memory=True)

    def _monitor_memory(self, stage=""):
        """Monitor memory usage for debugging"""
        if not PSUTIL_AVAILABLE:
            return
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 85:  # If memory usage > 85%
                print(f"High memory usage detected {stage}: {memory.percent:.1f}%")
                gc.collect()  # Force garbage collection
        except:
            pass  # Fail silently if psutil not available

    def optimize_memory(self):
        """Enhanced memory optimization for full dataset"""
        df = self.df
        
        print(f"Optimizing memory for FULL DATASET: {len(df)} rows...")
        
        # More aggressive but safe downcasting
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert repeated strings to categories (saves significant memory)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < len(df) * 0.6:  # If less than 60% unique
                df[col] = df[col].astype('category')
        
        # Optimize boolean-like columns
        bool_cols = ['isDot', 'isFour', 'isSix', 'isBowlerWk']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype('int8')
        
        # Force cleanup
        gc.collect()
        self.df = df
        
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory optimization complete. Full dataset size: {memory_usage:.1f} MB")
        
        self._monitor_memory("After optimization")

    def prepare_data(self):
        """Robust data preparation with smart column detection"""
        try:
            df = self.df
            
            # Print actual column names for debugging
            print(f"Preparing FULL data with columns: {list(df.columns)}")
            
            # Smart column mapping - handle different possible column names
            column_mapping = {}
            
            # Map batsman column (try different possible names)
            batsman_cols = ['striker', 'batsman', 'batter', 'batting_player', 'player', 'batsman_name']
            for col in batsman_cols:
                if col in df.columns:
                    column_mapping[col] = 'batsman'
                    print(f"Found batsman column: {col}")
                    break
            
            # Map runs column
            runs_cols = ['runs_off_bat', 'runs_of_bat', 'runs', 'batsman_runs', 'striker_runs']
            for col in runs_cols:
                if col in df.columns:
                    column_mapping[col] = 'runs_of_bat'
                    print(f"Found runs column: {col}")
                    break
            
            # Map other common columns
            other_mappings = {
                'ball': 'over',
                'wicket_type': 'dismissal_kind',
                'non_striker': 'non_striker',
                'bowler': 'bowler',
                'match_id': 'match_id',
                'venue': 'venue',
                'batting_team': 'batting_team',
                'player_dismissed': 'player_dismissed'
            }
            
            for old_col, new_col in other_mappings.items():
                if old_col in df.columns:
                    column_mapping[old_col] = new_col
            
            print(f"Column mappings applied: {column_mapping}")
            
            # Apply column renaming
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist, create if missing
            required_columns = {
                'batsman': 'Unknown Player',
                'bowler': 'Unknown Bowler', 
                'runs_of_bat': 0,
                'innings': 1,
                'match_id': 'unknown_match',
                'venue': 'Unknown Venue',
                'batting_team': 'Unknown Team',
                'player_dismissed': None,
                'dismissal_kind': None,
                'wides': 0,
                'noballs': 0,
                'extras': 0
            }
            
            for col, default_val in required_columns.items():
                if col not in df.columns:
                    print(f"Creating missing column: {col} with default value: {default_val}")
                    df[col] = default_val
            
            # Data type conversions with error handling
            try:
                df['innings'] = pd.to_numeric(df['innings'], errors='coerce').fillna(1).astype('int8')
                df['runs_of_bat'] = pd.to_numeric(df['runs_of_bat'], errors='coerce').fillna(0).astype('int8')
                df['wides'] = pd.to_numeric(df['wides'], errors='coerce').fillna(0).astype('int8')
                df['noballs'] = pd.to_numeric(df['noballs'], errors='coerce').fillna(0).astype('int8')
                df['extras'] = pd.to_numeric(df['extras'], errors='coerce').fillna(0).astype('int8')
            except Exception as e:
                print(f"Data type conversion error: {e}")
                # Use default values if conversion fails
                df['innings'] = df.get('innings', 1)
                df['runs_of_bat'] = df.get('runs_of_bat', 0)
                df['wides'] = df.get('wides', 0)
                df['noballs'] = df.get('noballs', 0)
                df['extras'] = df.get('extras', 0)
            
            # Create derived columns safely
            df['isDot'] = (df['runs_of_bat']==0).astype('int8')
            df['isFour'] = (df['runs_of_bat']==4).astype('int8')
            df['isSix'] = (df['runs_of_bat']==6).astype('int8')
            df['total_run'] = (df['runs_of_bat'] + df['wides'] + df['noballs']).astype('int8')
            df['total_runs'] = (df['runs_of_bat'] + df['extras']).astype('int8')
            
            # Create isBowlerWk column safely
            try:
                df['isBowlerWk'] = df.apply(
                    lambda x: 1 if pd.notna(x['player_dismissed']) and x['dismissal_kind'] not in ['run out','retired hurt','retired out'] else 0,
                    axis=1
                ).astype('int8')
            except Exception as e:
                print(f"isBowlerWk creation error: {e}, using default values")
                df['isBowlerWk'] = 0
            
            print(f"✅ FULL Data preparation successful. Final shape: {df.shape}")
            print(f"✅ Unique batsmen: {df['batsman'].nunique()}")
            print(f"✅ Unique bowlers: {df['bowler'].nunique()}")
            print(f"✅ Total matches: {df['match_id'].nunique()}")
            
            # Memory cleanup after data preparation
            gc.collect()
            self.df = df
            
        except Exception as e:
            print(f"Critical error in prepare_data: {e}")
            print(f"Available columns: {list(self.df.columns) if hasattr(self, 'df') else 'No DataFrame'}")
            # Create emergency fallback data
            self._create_fallback_data()

    def get_batting_stats(self, min_innings=5, innings_filter=None):
        try:
            self._monitor_memory("Before batting stats")
            print(f"🏏 Processing batting stats: min_innings={min_innings}, filter={innings_filter}")
            
            # Work on a copy
            df = self.df.copy()
            if innings_filter in [1,2]:
                df = df[df['innings'] == innings_filter]
            
            print(f"📊 Processing {len(df)} rows for batting analysis")
            
            # CRITICAL FIX: Handle all data type issues
            # Convert categorical columns to strings
            for col in ['batsman', 'match_id', 'player_dismissed']:
                if col in df.columns:
                    if df[col].dtype.name == 'category':
                        df[col] = df[col].astype(str)
                    else:
                        df[col] = df[col].fillna('').astype(str)
            
            # Ensure numeric columns are proper numeric types
            df['runs_of_bat'] = pd.to_numeric(df['runs_of_bat'], errors='coerce').fillna(0)
            df['isFour'] = pd.to_numeric(df['isFour'], errors='coerce').fillna(0)
            df['isSix'] = pd.to_numeric(df['isSix'], errors='coerce').fillna(0)
            df['isDot'] = pd.to_numeric(df['isDot'], errors='coerce').fillna(0)
            
            print(f"✅ Data type conversion completed")
            
            # Calculate match-level scores for centuries/fifties
            try:
                match_runs = df.groupby(['batsman', 'match_id'])['runs_of_bat'].sum().reset_index()
                batsman_match_scores = match_runs.groupby('batsman')['runs_of_bat'].agg(list)
                
                hundreds = batsman_match_scores.apply(lambda scores: sum(1 for s in scores if s >= 100))
                fifties = batsman_match_scores.apply(lambda scores: sum(1 for s in scores if 50 <= s < 100))
                highest_score = batsman_match_scores.apply(lambda scores: max(scores) if scores else 0)
                
                print(f"✅ Match-level aggregation completed")
            except Exception as e:
                print(f"❌ Error in match aggregation: {e}")
                # Fallback
                hundreds = pd.Series(dtype=int)
                fifties = pd.Series(dtype=int)
                highest_score = pd.Series(dtype=int)
            
            # Main batting statistics
            try:
                runs = df.groupby('batsman')['runs_of_bat'].sum()
                balls = df.groupby('batsman').size()
                inns = df.groupby('batsman')['match_id'].nunique()
                fours = df.groupby('batsman')['isFour'].sum()
                sixes = df.groupby('batsman')['isSix'].sum()
                
                print(f"✅ Basic aggregation completed - {len(runs)} batsmen found")
                
                # Calculate percentages and derived stats
                dot_pct = df.groupby('batsman')['isDot'].sum() / balls * 100
                boundary_pct = (fours + sixes) / balls * 100
                
                # Dismissals calculation
                dismissed_df = df[df['player_dismissed'] == df['batsman']]
                dismissals = dismissed_df.groupby('batsman').size()
                
                # FIXED: BPD and BPB calculations with proper numeric conversion
                bpd_raw = balls / dismissals.reindex(balls.index, fill_value=1)  # Avoid division by zero
                bpd = pd.to_numeric(bpd_raw, errors='coerce').fillna(0)
                
                bpb_raw = balls / (fours + sixes).replace(0, 1)  # Avoid division by zero
                bpb = pd.to_numeric(bpb_raw, errors='coerce').fillna(0)
                
                # RPI calculations by innings
                try:
                    rpi_1_data = df[df['innings']==1].groupby('batsman').agg({
                        'runs_of_bat': 'sum',
                        'match_id': 'nunique'
                    })
                    rpi_1_raw = rpi_1_data['runs_of_bat'] / rpi_1_data['match_id']
                    rpi_1 = pd.to_numeric(rpi_1_raw, errors='coerce').fillna(0)
                    
                    rpi_2_data = df[df['innings']==2].groupby('batsman').agg({
                        'runs_of_bat': 'sum', 
                        'match_id': 'nunique'
                    })
                    rpi_2_raw = rpi_2_data['runs_of_bat'] / rpi_2_data['match_id']
                    rpi_2 = pd.to_numeric(rpi_2_raw, errors='coerce').fillna(0)
                    
                    rpi_all_raw = runs / inns
                    rpi_all = pd.to_numeric(rpi_all_raw, errors='coerce').fillna(0)
                except Exception as e:
                    print(f"⚠️ RPI calculation error: {e}")
                    rpi_1 = pd.Series(0, index=runs.index)
                    rpi_2 = pd.Series(0, index=runs.index)
                    rpi_all = pd.to_numeric(runs / inns, errors='coerce').fillna(0)
                
                print(f"✅ Advanced statistics calculated")
                
            except Exception as e:
                print(f"❌ Error in main statistics: {e}")
                return pd.DataFrame(columns=['batsman', 'runs', 'innings', 'balls', 'SR'])
            
            # Create the final DataFrame
            try:
                # FIXED: Ensure all numeric conversions before .round() operations
                sr_raw = (runs / balls * 100)
                sr = pd.to_numeric(sr_raw, errors='coerce').fillna(0)
                
                dot_pct_numeric = pd.to_numeric(dot_pct, errors='coerce').fillna(0)
                boundary_pct_numeric = pd.to_numeric(boundary_pct, errors='coerce').fillna(0)
                
                stats = pd.DataFrame({
                    'batsman': runs.index,
                    'runs': runs.values,
                    'innings': inns.values, 
                    'balls': balls.values,
                    'SR': sr.round(2),
                    'hundreds': hundreds.reindex(runs.index, fill_value=0),
                    'fifties': fifties.reindex(runs.index, fill_value=0),
                    'hs': highest_score.reindex(runs.index, fill_value=0),
                    'RPI': rpi_all.round(2),
                    'RPI_1': rpi_1.reindex(runs.index, fill_value=0).round(2),
                    'RPI_2': rpi_2.reindex(runs.index, fill_value=0).round(2),
                    'Dot%': dot_pct_numeric.round(2),
                    'Boundary%': boundary_pct_numeric.round(2),
                    'BPD': bpd.round(2),
                    'BPB': bpb.round(0).astype(int),  # FIXED: Convert to numeric first, then round, then int
                })
                
                # Apply minimum innings filter
                stats = stats[stats['innings'] >= min_innings]
                stats = stats.fillna(0).sort_values('runs', ascending=False).reset_index(drop=True)
                
                print(f"✅ Final batting stats: {len(stats)} players with {min_innings}+ innings")
                
                # Memory cleanup
                del df, runs, balls, inns, fours, sixes, dot_pct, boundary_pct
                gc.collect()
                
                return stats
                
            except Exception as e:
                print(f"❌ Error creating final DataFrame: {e}")
                return pd.DataFrame(columns=['batsman', 'runs', 'innings', 'balls', 'SR'])
                
        except Exception as e:
            print(f"❌ Critical error in batting stats: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['batsman', 'runs', 'innings', 'balls', 'SR'])

    def get_bowling_stats(self, min_innings=3, innings_filter=None):
        try:
            self._monitor_memory("Before bowling stats")
            df = self.df
            if innings_filter in [1,2]:
                df = df[df['innings'] == innings_filter]
                
            # CRITICAL FIX: Add observed=True to suppress FutureWarnings
            match_wkts = df.groupby(['bowler', 'match_id'], observed=True)['isBowlerWk'].sum().reset_index()
            best = match_wkts.groupby('bowler', observed=True)['isBowlerWk'].max()
            five_wkts = match_wkts.groupby('bowler', observed=True)['isBowlerWk'].apply(lambda x: sum(x >= 5))
            
            # Memory cleanup
            del match_wkts
            gc.collect()
            
            # Wickets split by innings with observed=True
            wickets_1 = df[df['innings']==1].groupby('bowler', observed=True)['isBowlerWk'].sum()
            wickets_2 = df[df['innings']==2].groupby('bowler', observed=True)['isBowlerWk'].sum()
            
            # Main stats with observed=True
            runs = df.groupby('bowler', observed=True)['total_run'].sum()
            balls = df.groupby('bowler', observed=True).size()
            inns = df.groupby('bowler', observed=True)['match_id'].nunique()
            wickets = df.groupby('bowler', observed=True)['isBowlerWk'].sum()
            dots = df.groupby('bowler', observed=True)['isDot'].sum()
            eco = runs / (balls / 6)
            dot_pct = dots / balls * 100
            avg = (runs / wickets).replace([float('inf'), float('nan')], 0)
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
            
            # Aggressive memory cleanup
            del runs, balls, inns, wickets, dots, eco, dot_pct, avg, sr, wickets_1, wickets_2, best, five_wkts
            gc.collect()
            
            self._monitor_memory("After bowling stats")
            return stats
            
        except MemoryError:
            print("Memory limit reached in bowling stats, returning limited data")
            return pd.DataFrame(columns=['bowler', 'innings', 'balls', 'runs', 'wickets', 'ECO'])
        except Exception as e:
            print(f"Error in bowling stats: {e}")
            return pd.DataFrame(columns=['bowler', 'innings', 'balls', 'runs', 'wickets', 'ECO'])

    def get_head_to_head(self, bowler, batsman, innings_filter=None):
        try:
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
            
            # Memory cleanup
            del df
            gc.collect()
            
            return {
                'bowler':bowler,'batsman':batsman,
                'balls':total_balls,'runs':total_runs,'wickets':wickets,
                'dot_balls':dot_balls,'strike_rate':strike_rate,
                'economy':economy,'dot_percentage':dot_pct,
                'matches':matches,'dismissed':'Yes' if wickets>0 else 'No'
            }
        except MemoryError:
            return None
        except Exception:
            return None

    def get_multiple_head_to_head(self, bowlers, batsmen, innings_filter=None):
        results = []
        # Process in smaller batches to avoid memory issues
        batch_size = 8  # Conservative batch size for stability
        
        for i in range(0, len(bowlers), batch_size):
            bowler_batch = bowlers[i:i+batch_size]
            for j in range(0, len(batsmen), batch_size):
                batsman_batch = batsmen[j:j+batch_size]
                
                for bowler in bowler_batch:
                    for batsman in batsman_batch:
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
                
                # Memory cleanup between batches
                gc.collect()
        
        return results

    def get_player_opponents(self, player, ptype='bowler', innings_filter=None):
        try:
            df = self.df
            if innings_filter in [1,2]:
                df = df[df['innings'] == innings_filter]
            if ptype=='bowler':
                opps = df[df['bowler']==player]['batsman'].dropna().unique()
            else:
                opps = df[df['batsman']==player]['bowler'].dropna().unique()
            
            # Memory cleanup
            del df
            gc.collect()
            
            return sorted(opps.tolist())
        except Exception:
            return []

    def search_players(self, query, ptype='both', limit=10, innings_filter=None):
        try:
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
            
            # Memory cleanup
            del df
            gc.collect()
            
            return out[:limit]
        except Exception:
            return []

    def get_venue_team_options(self):
        try:
            venues = sorted(self.df['venue'].dropna().unique().tolist())
            teams = sorted(self.df['batting_team'].dropna().unique().tolist())
            
            # Memory cleanup
            gc.collect()
            
            return venues, teams
        except Exception:
            return [], []

    def get_venue_team_performance(self, venue_name, team_name):
        try:
            self._monitor_memory("Before venue team performance")
            
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
            
            # Compute total runs per innings for the team with observed=True
            team_innings_stats = (
                team_matches_venue.groupby(['match_id', 'innings'], observed=True)['total_runs']
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
                
                # Get innings totals for this match with observed=True
                innings_totals = match_data.groupby(['innings', 'batting_team'], observed=True)['total_runs'].sum().reset_index()
                
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
                
                # Memory cleanup within loop
                del innings_totals
                gc.collect()
            
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
            
            # Aggressive memory cleanup
            del venue_matches, team_matches_venue, team_innings_stats, team_match_results
            gc.collect()
            self._monitor_memory("After venue team performance")
            
            return result
            
        except MemoryError:
            return {
                'venue': venue_name,
                'team': team_name,
                'matches': 0,
                'error': 'Memory limit reached'
            }
        except Exception as e:
            print(f"Error in venue team performance: {e}")
            return {
                'venue': venue_name,
                'team': team_name,
                'matches': 0,
                'error': str(e)
            }

    def get_venue_characteristics(self, venue_name):
        try:
            venue_matches = self.df[self.df['venue'] == venue_name]
            
            if venue_matches.empty:
                return None
                
            # Calculate innings totals for each match with observed=True
            match_innings_stats = venue_matches.groupby(['match_id', 'innings'], observed=True)['total_runs'].sum().unstack(fill_value=0)
            
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
            
            # Memory cleanup
            del venue_matches, match_innings_stats
            gc.collect()
            
            return result
            
        except MemoryError:
            return None
        except Exception as e:
            print(f"Error in venue characteristics: {e}")
            return None

    def get_venue_team_comparison(self, venue_name, teams_list):
        if len(teams_list) < 2:
            return []
            
        comparison_results = []
        for team in teams_list:
            team_stats = self.get_venue_team_performance(venue_name, team)
            if team_stats and team_stats.get('matches', 0) > 0:
                comparison_results.append(team_stats)
            
            # Memory cleanup between teams
            gc.collect()
                
        return comparison_results

    def get_venue_records(self, venue_name):
        try:
            venue_matches = self.df[self.df['venue'] == venue_name]
            
            if venue_matches.empty:
                return None
                
            # Highest individual score by batsman with observed=True
            batsman_scores = venue_matches.groupby(['batsman', 'match_id'], observed=True)['runs_of_bat'].sum()
            highest_individual = batsman_scores.max()
            highest_scorer = batsman_scores.idxmax()[0] if not batsman_scores.empty else "N/A"
            
            # Best bowling figures with observed=True
            bowler_wickets = venue_matches.groupby(['bowler', 'match_id'], observed=True)['isBowlerWk'].sum()
            best_bowling = bowler_wickets.max()
            best_bowler = bowler_wickets.idxmax()[0] if not bowler_wickets.empty else "N/A"
            
            # Most sixes in innings with observed=True
            sixes_per_match = venue_matches.groupby(['batting_team', 'match_id', 'innings'], observed=True)['isSix'].sum()
            most_sixes = sixes_per_match.max()
            
            result = {
                'venue': venue_name,
                'highest_individual_score': int(highest_individual) if highest_individual else 0,
                'highest_scorer': highest_scorer,
                'best_bowling_figures': int(best_bowling) if best_bowling else 0,
                'best_bowler': best_bowler,
                'most_sixes_innings': int(most_sixes) if most_sixes else 0
            }
            
            # Memory cleanup
            del venue_matches, batsman_scores, bowler_wickets, sixes_per_match
            gc.collect()
            
            return result
            
        except MemoryError:
            return None
        except Exception as e:
            print(f"Error in venue records: {e}")
            return None

    def get_data_summary(self):
        """Get summary of loaded data for verification"""
        try:
            total_matches = self.df['match_id'].nunique()
            total_players = self.df['batsman'].nunique()
            total_balls = len(self.df)
            
            player_match_counts = self.df.groupby('batsman')['match_id'].nunique().sort_values(ascending=False)
            
            return {
                'total_matches': total_matches,
                'total_players': total_players, 
                'total_balls': total_balls,
                'avg_matches_per_player': round(player_match_counts.mean(), 2),
                'max_matches_per_player': player_match_counts.max(),
                'top_10_players': player_match_counts.head(10).to_dict()
            }
        except Exception as e:
            return {
                'error': f"Error generating summary: {e}",
                'total_matches': 0,
                'total_players': 0,
                'total_balls': 0
            }
