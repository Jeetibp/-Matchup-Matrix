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
            print(f"🏏 Processing batting stats with FULL DATASET: min_innings={min_innings}, filter={innings_filter}")
            
            # CRITICAL FIX: Work on a copy and convert categorical columns to strings
            df = self.df.copy()
            if innings_filter in [1,2]:
                df = df[df['innings'] == innings_filter]
            
            print(f"📊 Processing {len(df)} rows for batting stats")
            
            # Convert categorical columns to strings to avoid groupby comparison issues
            categorical_cols = df.select_dtypes(include=['category']).columns
            for col in categorical_cols:
                if col in ['batsman', 'match_id', 'bowler']:  # Only convert columns used in groupby
                    df[col] = df[col].astype(str)
            
            # Ensure player_dismissed is also string for comparison
            if 'player_dismissed' in df.columns:
                df['player_dismissed'] = df['player_dismissed'].astype(str)
            
            # Per-match group for 100s, 50s, highest
            match_runs = df.groupby(['batsman', 'match_id'])['runs_of_bat'].sum().reset_index()
            batsman_match_scores = match_runs.groupby('batsman')['runs_of_bat'].agg(list)
            
            # Memory cleanup between operations
            del match_runs
            gc.collect()
            
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
            
            # Dismissals, BPD, BPB - Fixed to handle string comparison properly
            dismissed_df = df[df['player_dismissed'] == df['batsman']]
            dismissals = dismissed_df.groupby('batsman')['player_dismissed'].count()
            bpd = balls / dismissals.replace(0, pd.NA)
            bpb = balls / (fours + sixes).replace(0, pd.NA)
            
            # RPI calculations
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
                'BPB': bpb.round(2).fillna(0).astype('int64'),  # Use int64 to avoid buffer issues
            })
            
            stats = stats[stats['innings']>=min_innings].fillna(0).sort_values('runs',ascending=False).reset_index(drop=True)
            
            print(f"✅ Batting stats generated: {len(stats)} players with {min_innings}+ innings")
            
            # Aggressive memory cleanup
            del runs, balls, inns, fours, sixes, dot_pct, boundary_pct, dismissals, bpd, bpb, rpi_all, rpi_1, rpi_2, df, dismissed_df
            gc.collect()
            
            self._monitor_memory("After batting stats")
            return stats
            
        except Exception as e:
            print(f"❌ Error in batting stats: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['batsman', 'runs', 'innings', 'balls', 'SR', 'hundreds', 'fifties', 'hs', 'RPI', 'RPI_1', 'RPI_2', 'Dot%', 'Boundary%', 'BPD', 'BPB'])

    # [Rest of the methods remain the same - bowling_stats, head_to_head, venue methods, etc.]
    # Keep all your existing methods exactly as they are

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

    # [Include all your other methods: get_head_to_head, get_multiple_head_to_head, get_player_opponents, search_players, get_venue_team_options, get_venue_team_performance, get_venue_characteristics, get_venue_team_comparison, get_venue_records, get_data_summary]

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
