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

                # Average per innings (matches the player card "Average" field)
                avg_raw = runs / inns.replace(0, pd.NA)
                avg = pd.to_numeric(avg_raw, errors='coerce').fillna(0)
                
                # Avoid CategoricalIndex code-dtype mismatch on reindex by using object index
                target_idx = runs.index.astype(object)
                for _s in (hundreds, fifties, highest_score, rpi_1, rpi_2):
                    if hasattr(_s.index, 'categories'):
                        _s.index = _s.index.astype(object)

                stats = pd.DataFrame({
                    'batsman': runs.index,
                    'runs': runs.values,
                    'innings': inns.values, 
                    'balls': balls.values,
                    'AVG': avg.round(2),
                    'SR': sr.round(2),
                    'hundreds': hundreds.reindex(target_idx, fill_value=0).values,
                    'fifties': fifties.reindex(target_idx, fill_value=0).values,
                    'hs': highest_score.reindex(target_idx, fill_value=0).values,
                    'RPI': rpi_all.round(2),
                    'RPI_1': rpi_1.reindex(target_idx, fill_value=0).round(2).values,
                    'RPI_2': rpi_2.reindex(target_idx, fill_value=0).round(2).values,
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
            fours_c = df.groupby('bowler', observed=True)['isFour'].sum()
            sixes_c = df.groupby('bowler', observed=True)['isSix'].sum()
            eco = runs / (balls / 6)
            dot_pct = dots / balls * 100
            avg = (runs / wickets).replace([float('inf'), float('nan')], 0)
            sr = (balls / wickets).replace([float('inf'), float('nan')], 0).round(2)
            bpb = (balls / (fours_c + sixes_c).replace(0, pd.NA))
            bpb = pd.to_numeric(bpb, errors='coerce').fillna(0)
            
            # Avoid CategoricalIndex code-dtype mismatch on reindex by using object index
            target_idx = runs.index.astype(object)
            for _s in (wickets_1, wickets_2, best, five_wkts):
                if hasattr(_s.index, 'categories'):
                    _s.index = _s.index.astype(object)

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
                'BPB': bpb.round(2).values,
                'wickets_1': wickets_1.reindex(target_idx, fill_value=0).astype(int).values,
                'wickets_2': wickets_2.reindex(target_idx, fill_value=0).astype(int).values,
                'best': best.reindex(target_idx, fill_value=0).astype(int).values,
                'five_wkts': five_wkts.reindex(target_idx, fill_value=0).astype(int).values,
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

    def get_player_form(self, player, last_n=10):
        """Return a compact last-N-innings form summary for a player.

        Used to feed OpenAI probable-XI predictions. Returns a dict with:
          - role: 'batsman', 'bowler', or 'allrounder' (inferred from data volume)
          - batting: last N innings {date, runs, balls, sr, opp, venue, dismissed}
          - bowling: last N bowling innings {date, balls, runs, wkts, eco, opp, venue}
          - aggregate: { matches, total_runs, sr, total_wkts, eco }
        All values defensively defaulted to 0 / [] on any error.
        """
        result = {'role': 'batsman', 'batting': [], 'bowling': [], 'aggregate': {}}
        try:
            df = self.df
            # Resolve name via case-insensitive match in case provider uses different casing
            bat_match = None
            bowl_match = None
            try:
                bat_vals = df['batsman'].dropna().astype(str).unique()
                bowl_vals = df['bowler'].dropna().astype(str).unique()
                for v in bat_vals:
                    if v.lower() == player.lower():
                        bat_match = v; break
                for v in bowl_vals:
                    if v.lower() == player.lower():
                        bowl_match = v; break
            except Exception:
                pass
            bat_name = bat_match or player
            bowl_name = bowl_match or player

            # Batting innings
            try:
                bdf = df[df['batsman'] == bat_name]
                if not bdf.empty:
                    per_inn = bdf.groupby(['match_id', 'batting_team', 'bowling_team', 'venue'],
                                          observed=True).agg(
                        runs=('runs_of_bat', 'sum'),
                        balls=('runs_of_bat', 'size'),
                        fours=('isFour', 'sum'),
                        sixes=('isSix', 'sum'),
                        dismissed=('player_dismissed', lambda s: int((s == bat_name).sum())),
                    ).reset_index()
                    if 'start_date' in bdf.columns:
                        dates = bdf.groupby('match_id', observed=True)['start_date'].first()
                        per_inn['date'] = per_inn['match_id'].map(dates)
                    order_col = 'date' if 'date' in per_inn.columns else 'match_id'
                    per_inn = per_inn.sort_values(order_col, ascending=False).head(last_n)
                    batting_rows = []
                    for _, r in per_inn.iterrows():
                        sr = round(float(r['runs']) * 100.0 / r['balls'], 2) if r['balls'] else 0.0
                        batting_rows.append({
                            'date': str(r.get('date', ''))[:10] if pd.notna(r.get('date', '')) else '',
                            'runs': int(r['runs']),
                            'balls': int(r['balls']),
                            'sr': sr,
                            '4s': int(r['fours']),
                            '6s': int(r['sixes']),
                            'opp': str(r.get('bowling_team', '')),
                            'venue': str(r['venue']),
                            'out': bool(r['dismissed']),
                        })
                    result['batting'] = batting_rows
            except Exception as e:
                print(f"get_player_form batting error for {player}: {e}")

            # Bowling innings
            try:
                bwdf = df[df['bowler'] == bowl_name]
                if not bwdf.empty:
                    per_inn_b = bwdf.groupby(['match_id', 'bowling_team', 'batting_team', 'venue'],
                                             observed=True).agg(
                        balls=('isBowlerWk', 'size'),
                        runs=('total_run', 'sum'),
                        wkts=('isBowlerWk', 'sum'),
                    ).reset_index()
                    if 'start_date' in bwdf.columns:
                        dates_b = bwdf.groupby('match_id', observed=True)['start_date'].first()
                        per_inn_b['date'] = per_inn_b['match_id'].map(dates_b)
                    order_col = 'date' if 'date' in per_inn_b.columns else 'match_id'
                    per_inn_b = per_inn_b.sort_values(order_col, ascending=False).head(last_n)
                    bowling_rows = []
                    for _, r in per_inn_b.iterrows():
                        overs = r['balls'] / 6.0
                        eco = round(float(r['runs']) / overs, 2) if overs else 0.0
                        bowling_rows.append({
                            'date': str(r.get('date', ''))[:10] if pd.notna(r.get('date', '')) else '',
                            'balls': int(r['balls']),
                            'runs': int(r['runs']),
                            'wkts': int(r['wkts']),
                            'eco': eco,
                            'opp': str(r.get('batting_team', '')),
                            'venue': str(r['venue']),
                        })
                    result['bowling'] = bowling_rows
            except Exception as e:
                print(f"get_player_form bowling error for {player}: {e}")

            # Role inference
            bat_inn = len(result['batting'])
            bowl_inn = len(result['bowling'])
            if bat_inn > 0 and bowl_inn > 0:
                result['role'] = 'allrounder'
            elif bowl_inn > 0:
                result['role'] = 'bowler'
            else:
                result['role'] = 'batsman'

            # Aggregate
            agg = {'matches': max(bat_inn, bowl_inn)}
            if bat_inn:
                total_runs = sum(r['runs'] for r in result['batting'])
                total_balls = sum(r['balls'] for r in result['batting'])
                agg['runs'] = total_runs
                agg['sr'] = round(total_runs * 100.0 / total_balls, 2) if total_balls else 0.0
            if bowl_inn:
                total_wkts = sum(r['wkts'] for r in result['bowling'])
                total_balls_b = sum(r['balls'] for r in result['bowling'])
                total_runs_b = sum(r['runs'] for r in result['bowling'])
                overs_total = total_balls_b / 6.0
                agg['wkts'] = total_wkts
                agg['eco'] = round(total_runs_b / overs_total, 2) if overs_total else 0.0
            result['aggregate'] = agg
        except Exception as e:
            print(f"get_player_form critical error for {player}: {e}")
        return result

    def get_player_innings_split(self, player, ptype='batsman'):
        """Return a player's performance split by innings (1st vs 2nd).

        Returns a dict like:
            {
                'batting': {1: {matches, runs, balls, sr}, 2: {...}},
                'bowling': {1: {matches, balls, runs, wkts, eco}, 2: {...}},
            }
        Each innings map is keyed by int innings number. Missing data -> empty dict.
        """
        result = {'batting': {}, 'bowling': {}}
        try:
            df = self.df
            if ptype in ('batsman', 'both'):
                bdf = df[df['batsman'] == player]
                for innings in (1, 2):
                    sub = bdf[bdf['innings'] == innings]
                    if sub.empty:
                        continue
                    per = sub.groupby(['match_id'], observed=True).agg(
                        runs=('runs_of_bat', 'sum'),
                        balls=('runs_of_bat', 'size'),
                    )
                    if per.empty:
                        continue
                    total_runs = int(per['runs'].sum())
                    total_balls = int(per['balls'].sum())
                    result['batting'][innings] = {
                        'matches': int(per.shape[0]),
                        'runs': total_runs,
                        'balls': total_balls,
                        'sr': round(total_runs * 100.0 / total_balls, 2) if total_balls else 0.0,
                    }
            if ptype in ('bowler', 'both'):
                bwdf = df[df['bowler'] == player]
                for innings in (1, 2):
                    sub = bwdf[bwdf['innings'] == innings]
                    if sub.empty:
                        continue
                    per = sub.groupby(['match_id'], observed=True).agg(
                        balls=('isBowlerWk', 'size'),
                        runs=('total_run', 'sum'),
                        wkts=('isBowlerWk', 'sum'),
                    )
                    if per.empty:
                        continue
                    total_balls = int(per['balls'].sum())
                    total_runs = int(per['runs'].sum())
                    total_wkts = int(per['wkts'].sum())
                    overs = total_balls / 6.0
                    result['bowling'][innings] = {
                        'matches': int(per.shape[0]),
                        'balls': total_balls,
                        'runs': total_runs,
                        'wkts': total_wkts,
                        'eco': round(total_runs / overs, 2) if overs else 0.0,
                    }
        except Exception as e:
            print(f"get_player_innings_split error for {player}: {e}")
        return result

    def get_player_match_analytics(self, player, team=None, venue=None, last_n=10):
        """Bundle every analytics signal the AI analysis agent needs for one player.

        Uses the CSV name exactly as stored (caller should resolve the probable-XI
        name via EntityResolutionAgent first). Returns:
            {
                'name': player,
                'role': 'batsman' | 'bowler' | 'allrounder',
                'form': {...get_player_form()...},
                'h2h': {...get_player_vs_team()...} | None,
                'venue': {...get_player_at_venue()...} | None,
                'innings_split': {...get_player_innings_split()...},
            }
        Only the signals that could be computed are included; the agent sees
        'not available' for anything missing.
        """
        bundle = {'name': player}
        try:
            form = self.get_player_form(player, last_n=last_n)
            role = form.get('role', 'batsman')
            bundle['role'] = role
            bundle['form'] = form
            ptype = role if role in ('batsman', 'bowler') else 'both'
            bundle['innings_split'] = self.get_player_innings_split(player, ptype=ptype)
            if team:
                h2h = self.get_player_vs_team(player, team, ptype=ptype)
                if h2h.get('matches', 0) > 0:
                    bundle['h2h'] = {k: v for k, v in h2h.items()}
            if venue:
                venue_stats = self.get_player_at_venue(player, venue, ptype=ptype)
                if venue_stats.get('matches', 0) > 0:
                    bundle['venue'] = {k: v for k, v in venue_stats.items()}
        except Exception as e:
            print(f"get_player_match_analytics error for {player}: {e}")
        return bundle

    def get_player_vs_team(self, player, team, ptype='batsman'):
        """Return a player's record against a specific opponent team.

        Used for probable-XI prediction context. ptype='batsman' returns
        {runs, balls, sr, dismissals, matches} against this team's bowlers.
        ptype='bowler' returns {balls, runs, wkts, eco, matches} against this team's batters.
        """
        try:
            df = self.df
            if ptype == 'batsman':
                subset = df[(df['batsman'] == player) & (df['bowling_team'] == team)]
                if subset.empty:
                    return {'matches': 0}
                runs = int(subset['runs_of_bat'].sum())
                balls = int(len(subset))
                wkts_bowler = int(subset[subset['player_dismissed'] == player].shape[0])
                sr = round(runs * 100.0 / balls, 2) if balls else 0.0
                return {
                    'matches': int(subset['match_id'].nunique()),
                    'runs': runs, 'balls': balls, 'sr': sr, 'dismissals': wkts_bowler,
                }
            else:
                subset = df[(df['bowler'] == player) & (df['batting_team'] == team)]
                if subset.empty:
                    return {'matches': 0}
                balls = int(len(subset))
                runs = int(subset['total_run'].sum())
                wkts = int(subset['isBowlerWk'].sum())
                overs = balls / 6.0
                eco = round(runs / overs, 2) if overs else 0.0
                return {
                    'matches': int(subset['match_id'].nunique()),
                    'balls': balls, 'runs': runs, 'wkts': wkts, 'eco': eco,
                }
        except Exception as e:
            print(f"get_player_vs_team error for {player} vs {team}: {e}")
            return {'matches': 0}

    def get_player_at_venue(self, player, venue, ptype='batsman'):
        """Return a player's record at a specific venue.

        Used for probable-XI prediction context. Same return shape as
        get_player_vs_team but filtered by venue rather than opponent team.
        """
        try:
            df = self.df[self.df['venue'] == venue]
            if ptype == 'batsman':
                subset = df[df['batsman'] == player]
                if subset.empty:
                    return {'matches': 0}
                runs = int(subset['runs_of_bat'].sum())
                balls = int(len(subset))
                sr = round(runs * 100.0 / balls, 2) if balls else 0.0
                dismissals = int(subset[subset['player_dismissed'] == player].shape[0])
                return {
                    'matches': int(subset['match_id'].nunique()),
                    'runs': runs, 'balls': balls, 'sr': sr, 'dismissals': dismissals,
                }
            else:
                subset = df[df['bowler'] == player]
                if subset.empty:
                    return {'matches': 0}
                balls = int(len(subset))
                runs = int(subset['total_run'].sum())
                wkts = int(subset['isBowlerWk'].sum())
                overs = balls / 6.0
                eco = round(runs / overs, 2) if overs else 0.0
                return {
                    'matches': int(subset['match_id'].nunique()),
                    'balls': balls, 'runs': runs, 'wkts': wkts, 'eco': eco,
                }
        except Exception as e:
            print(f"get_player_at_venue error for {player} @ {venue}: {e}")
            return {'matches': 0}

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
            
            # High scoring vs low scoring — count matches (not innings), exclude fill_value=0 phantom entries
            high_scores = int(((match_innings_stats >= 180) & (match_innings_stats > 0)).any(axis=1).sum())
            low_scores = int(((match_innings_stats < 150) & (match_innings_stats > 0)).any(axis=1).sum())

            # Bat-first vs Chase win split (only matches with both innings present)
            both = match_innings_stats[(match_innings_stats.get(1, 0) > 0) & (match_innings_stats.get(2, 0) > 0)]
            completed = len(both)
            chase_wins = int((both[2] > both[1]).sum()) if completed else 0
            batfirst_wins = int((both[1] > both[2]).sum()) if completed else 0
            chase_win_pct = round(chase_wins * 100.0 / completed, 2) if completed else 0.0
            batfirst_win_pct = round(batfirst_wins * 100.0 / completed, 2) if completed else 0.0

            # Highest team total and lowest completed team total at venue
            inn_totals = venue_matches.groupby(['match_id', 'innings', 'batting_team'], observed=True)['total_runs'].sum().reset_index()
            highest_total = None
            lowest_total = None
            highest_chase = None
            if not inn_totals.empty:
                hi = inn_totals.loc[inn_totals['total_runs'].idxmax()]
                highest_total = {
                    'team': str(hi['batting_team']),
                    'runs': int(hi['total_runs']),
                }
                # Lowest: exclude tiny totals (likely abandoned matches) — require >= 50
                low_pool = inn_totals[inn_totals['total_runs'] >= 50]
                if not low_pool.empty:
                    lo = low_pool.loc[low_pool['total_runs'].idxmin()]
                    lowest_total = {
                        'team': str(lo['batting_team']),
                        'runs': int(lo['total_runs']),
                    }
                # Highest successful chase: largest 2nd-innings total that beat the 1st-innings
                if completed:
                    chase_pool = inn_totals[inn_totals['innings'] == 2].copy()
                    inn1_map = inn_totals[inn_totals['innings'] == 1].set_index('match_id')['total_runs']
                    chase_pool['inn1'] = chase_pool['match_id'].map(inn1_map)
                    successful = chase_pool[chase_pool['total_runs'] > chase_pool['inn1']]
                    if not successful.empty:
                        ch = successful.loc[successful['total_runs'].idxmax()]
                        highest_chase = {
                            'team': str(ch['batting_team']),
                            'runs': int(ch['total_runs']),
                            'target': int(ch['inn1']) + 1,
                        }

            result = {
                'venue': venue_name,
                'total_matches': total_matches,
                'avg_1st_innings': round(avg_1st_innings, 2),
                'avg_2nd_innings': round(avg_2nd_innings, 2),
                'chase_success_rate': round(chase_success_rate, 2),
                'batfirst_win_pct': batfirst_win_pct,
                'chase_win_pct': chase_win_pct,
                'completed_matches': completed,
                'boundary_rate': round(boundary_rate, 2),
                'high_scores': int(high_scores),
                'low_scores': int(low_scores),
                'total_fours': int(total_fours),
                'total_sixes': int(total_sixes),
                'highest_total': highest_total,
                'lowest_total': lowest_total,
                'highest_chase': highest_chase,
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

    def get_venue_recent_matches(self, venue_name, n=10):
        """Return the last `n` matches at the given venue as list of dicts with date/teams/scores/result."""
        try:
            vm = self.df[self.df['venue'] == venue_name]
            if vm.empty:
                return []
            # Per-innings totals + wickets
            inn = vm.groupby(['match_id', 'innings', 'batting_team'], observed=True).agg(
                runs=('total_runs', 'sum'),
                wickets=('player_dismissed', lambda s: s.notna().sum())
            ).reset_index()
            # Pivot to one row per match
            i1 = inn[inn['innings'] == 1].rename(columns={'batting_team': 'team1', 'runs': 'score1', 'wickets': 'wkts1'})[['match_id', 'team1', 'score1', 'wkts1']]
            i2 = inn[inn['innings'] == 2].rename(columns={'batting_team': 'team2', 'runs': 'score2', 'wickets': 'wkts2'})[['match_id', 'team2', 'score2', 'wkts2']]
            m = i1.merge(i2, on='match_id', how='inner')
            # Date (latest start_date per match) for sorting
            if 'start_date' in vm.columns:
                dates = vm.groupby('match_id', observed=True)['start_date'].first().reset_index()
                m = m.merge(dates, on='match_id', how='left')
                m['_sort'] = pd.to_datetime(m['start_date'], errors='coerce')
                m = m.sort_values('_sort', ascending=False, na_position='last')
            else:
                m = m.sort_values('match_id', ascending=False)
            m = m.head(n)
            out = []
            for _, r in m.iterrows():
                s1, s2, w1, w2 = int(r['score1']), int(r['score2']), int(r['wkts1']), int(r['wkts2'])
                if s1 > s2:
                    result = f"{r['team1']} won by {s1 - s2} runs"
                elif s2 > s1:
                    result = f"{r['team2']} won by {10 - w2} wickets"
                else:
                    result = "Tied"
                out.append({
                    'date': str(r['start_date']) if 'start_date' in m.columns and pd.notna(r.get('start_date')) else '',
                    'team1': str(r['team1']), 'score1': f"{s1}/{w1}",
                    'team2': str(r['team2']), 'score2': f"{s2}/{w2}",
                    'result': result,
                })
            del vm, inn, i1, i2, m
            gc.collect()
            return out
        except Exception as e:
            print(f"Error in venue recent matches: {e}")
            return []

    def get_venue_all_teams_summary(self, venue_name):
        """Return per-team summary at venue: matches, total_runs, avg_innings, HS, LS, wins (sorted by matches desc)."""
        try:
            vm = self.df[self.df['venue'] == venue_name]
            if vm.empty:
                return []
            # Per match-innings totals
            mi = vm.groupby(['match_id', 'innings', 'batting_team'], observed=True)['total_runs'].sum().reset_index()
            # Determine winner per match
            winners = {}
            for mid, g in mi.groupby('match_id', observed=True):
                if len(g) < 2:
                    continue
                g1 = g[g['innings'] == 1]
                g2 = g[g['innings'] == 2]
                if g1.empty or g2.empty:
                    continue
                s1, t1 = int(g1['total_runs'].iloc[0]), g1['batting_team'].iloc[0]
                s2, t2 = int(g2['total_runs'].iloc[0]), g2['batting_team'].iloc[0]
                winners[mid] = t1 if s1 > s2 else (t2 if s2 > s1 else None)
            # Aggregate per team
            out = []
            for team, tg in mi.groupby('batting_team', observed=True):
                team = str(team)
                matches = tg['match_id'].nunique()
                if matches == 0:
                    continue
                total = int(tg['total_runs'].sum())
                hs = int(tg['total_runs'].max())
                ls = int(tg['total_runs'].min())
                avg = round(total / matches, 2) if matches else 0
                wins = sum(1 for mid in tg['match_id'].unique() if winners.get(mid) == team)
                out.append({
                    'team': team, 'matches': matches, 'total_runs': total,
                    'avg': avg, 'hs': hs, 'ls': ls, 'wins': wins,
                    'win_pct': round(wins / matches * 100, 2) if matches else 0,
                })
            out.sort(key=lambda r: (-r['matches'], -r['avg']))
            del vm, mi, winners
            gc.collect()
            return out
        except Exception as e:
            print(f"Error in venue all-teams summary: {e}")
            return []

    def get_venue_top_batsmen(self, venue_name, n=10, min_balls=20):
        """Return top batsmen at the venue by runs scored."""
        try:
            vm = self.df[self.df['venue'] == venue_name]
            if vm.empty:
                return []
            g = vm.groupby('batsman', observed=True).agg(
                runs=('runs_of_bat', 'sum'),
                balls=('runs_of_bat', 'count'),
                fours=('isFour', 'sum'),
                sixes=('isSix', 'sum'),
            ).reset_index()
            g['matches'] = vm.groupby('batsman', observed=True)['match_id'].nunique().reindex(g['batsman']).values
            g = g[g['balls'] >= min_balls]
            g['sr'] = (g['runs'] / g['balls'] * 100).round(2)
            g = g.sort_values('runs', ascending=False).head(n)
            out = []
            for _, r in g.iterrows():
                out.append({
                    'batsman': str(r['batsman']),
                    'matches': int(r['matches']),
                    'runs': int(r['runs']),
                    'balls': int(r['balls']),
                    'sr': float(r['sr']),
                    'fours': int(r['fours']),
                    'sixes': int(r['sixes']),
                })
            del vm, g
            gc.collect()
            return out
        except Exception as e:
            print(f"Error in venue top batsmen: {e}")
            return []

    def get_venue_top_bowlers(self, venue_name, n=10, min_balls=24):
        """Return top bowlers at the venue by wickets."""
        try:
            vm = self.df[self.df['venue'] == venue_name]
            if vm.empty:
                return []
            g = vm.groupby('bowler', observed=True).agg(
                wickets=('isBowlerWk', 'sum'),
                runs=('total_runs', 'sum'),
                balls=('isBowlerWk', 'count'),
            ).reset_index()
            g['matches'] = vm.groupby('bowler', observed=True)['match_id'].nunique().reindex(g['bowler']).values
            g = g[g['balls'] >= min_balls]
            g['eco'] = (g['runs'] / (g['balls'] / 6)).round(2)
            g['avg'] = g.apply(lambda r: round(r['runs'] / r['wickets'], 2) if r['wickets'] else 0, axis=1)
            g['sr'] = g.apply(lambda r: round(r['balls'] / r['wickets'], 2) if r['wickets'] else 0, axis=1)
            g = g.sort_values('wickets', ascending=False).head(n)
            out = []
            for _, r in g.iterrows():
                out.append({
                    'bowler': str(r['bowler']),
                    'matches': int(r['matches']),
                    'wickets': int(r['wickets']),
                    'runs': int(r['runs']),
                    'balls': int(r['balls']),
                    'eco': float(r['eco']),
                    'avg': float(r['avg']),
                    'sr': float(r['sr']),
                })
            del vm, g
            gc.collect()
            return out
        except Exception as e:
            print(f"Error in venue top bowlers: {e}")
            return []

    def get_phase_analysis(self, venue=None, team=None, n_recent=15):
        """Phase-wise analysis (PP1: 1-6, PP2: 7-10, PP3: 11-15, PP4: 16-20).

        Filters by venue and/or team if provided. Returns:
          - phase_overall: dict of avg runs/wkts per phase across all innings considered
          - team_summary (only if team given): matches, wins, losses, win%, avg per phase when winning vs losing
          - team_breakdown (only if team is None): list of per-team avg phase runs
          - recent_matches: list of last N matches (full both-innings phase breakdown)
        """
        try:
            df = self.df
            if venue:
                df = df[df['venue'] == venue]
            if df.empty:
                return None

            # Restrict to matches involving the team (if filtering by team)
            if team:
                team_match_ids = df[df['batting_team'] == team]['match_id'].unique()
                if len(team_match_ids) == 0:
                    return None
                df = df[df['match_id'].isin(team_match_ids)]

            # Phase classifier from float over (e.g. 0.1..19.6)
            over_int = df['over'].astype(float).apply(lambda x: int(x))
            def _phase(o):
                if o <= 5: return 'PP1'
                if o <= 9: return 'PP2'
                if o <= 14: return 'PP3'
                return 'PP4'
            df = df.assign(phase=over_int.map(_phase))

            # Per match / per innings / per phase aggregation
            grp = df.groupby(['match_id', 'innings', 'batting_team', 'phase'], observed=True).agg(
                runs=('total_runs', 'sum'),
                wickets=('isBowlerWk', 'sum'),
                player_dismissed=('player_dismissed', lambda s: s.notna().sum()),
            ).reset_index()
            # use player_dismissed (includes run outs) as authoritative wicket count per phase
            grp['wickets'] = grp['player_dismissed'].astype(int)
            grp = grp.drop(columns=['player_dismissed'])

            # Pivot: per (match, innings) → columns PP1_runs, PP1_wkts...
            piv_runs = grp.pivot_table(index=['match_id', 'innings', 'batting_team'], columns='phase', values='runs', fill_value=0, observed=False)
            piv_wkts = grp.pivot_table(index=['match_id', 'innings', 'batting_team'], columns='phase', values='wickets', fill_value=0, observed=False)
            for p in ['PP1', 'PP2', 'PP3', 'PP4']:
                if p not in piv_runs.columns: piv_runs[p] = 0
                if p not in piv_wkts.columns: piv_wkts[p] = 0
            piv_runs = piv_runs[['PP1', 'PP2', 'PP3', 'PP4']]
            piv_wkts = piv_wkts[['PP1', 'PP2', 'PP3', 'PP4']]

            # Build per-innings rows
            inn = piv_runs.copy()
            inn.columns = [f'{c}_runs' for c in inn.columns]
            for p in ['PP1', 'PP2', 'PP3', 'PP4']:
                inn[f'{p}_wkts'] = piv_wkts[p]
            inn = inn.reset_index()
            inn['total_runs'] = inn[['PP1_runs', 'PP2_runs', 'PP3_runs', 'PP4_runs']].sum(axis=1)
            inn['total_wkts'] = inn[['PP1_wkts', 'PP2_wkts', 'PP3_wkts', 'PP4_wkts']].sum(axis=1)

            # Determine winner per match: team batting second wins if their total >= target (innings1 + 1)
            winners = {}
            for mid, g in inn.groupby('match_id'):
                if g['innings'].nunique() < 2:
                    continue
                try:
                    g1 = g[g['innings'] == 1].iloc[0]
                    g2 = g[g['innings'] == 2].iloc[0]
                    target = int(g1['total_runs']) + 1
                    winner = g2['batting_team'] if int(g2['total_runs']) >= target else g1['batting_team']
                    winners[mid] = winner
                except Exception:
                    continue
            inn['winner'] = inn['match_id'].map(winners)

            # Phase overall averages (across all innings considered)
            phase_overall = {}
            for p in ['PP1', 'PP2', 'PP3', 'PP4']:
                phase_overall[p] = {
                    'avg_runs': round(float(inn[f'{p}_runs'].mean()), 2),
                    'avg_wkts': round(float(inn[f'{p}_wkts'].mean()), 2),
                }
            phase_overall['innings_count'] = int(len(inn))
            phase_overall['matches'] = int(inn['match_id'].nunique())

            result = {
                'venue': venue,
                'team': team,
                'phase_overall': phase_overall,
            }

            # Team-specific summary (when team filter given)
            if team:
                team_inn = inn[inn['batting_team'] == team].copy()
                team_inn_with_winner = team_inn.dropna(subset=['winner'])
                wins = team_inn_with_winner[team_inn_with_winner['winner'] == team]
                losses = team_inn_with_winner[team_inn_with_winner['winner'] != team]
                def _avgs(frame):
                    if frame.empty:
                        return {p: {'avg_runs': 0.0, 'avg_wkts': 0.0} for p in ['PP1', 'PP2', 'PP3', 'PP4']}
                    return {p: {
                        'avg_runs': round(float(frame[f'{p}_runs'].mean()), 2),
                        'avg_wkts': round(float(frame[f'{p}_wkts'].mean()), 2),
                    } for p in ['PP1', 'PP2', 'PP3', 'PP4']}
                total = len(team_inn_with_winner)
                result['team_summary'] = {
                    'team': team,
                    'innings': int(len(team_inn)),
                    'matches_with_result': int(total),
                    'wins': int(len(wins)),
                    'losses': int(len(losses)),
                    'win_pct': round(len(wins) * 100.0 / total, 2) if total else 0.0,
                    'avg_when_winning': _avgs(wins),
                    'avg_when_losing': _avgs(losses),
                    'avg_batting_first': _avgs(team_inn[team_inn['innings'] == 1]),
                    'avg_batting_second': _avgs(team_inn[team_inn['innings'] == 2]),
                    'innings_first_count': int((team_inn['innings'] == 1).sum()),
                    'innings_second_count': int((team_inn['innings'] == 2).sum()),
                    'highest_score': int(team_inn['total_runs'].max()) if len(team_inn) else 0,
                    'lowest_score': int(team_inn['total_runs'].min()) if len(team_inn) else 0,
                    'avg_score': round(float(team_inn['total_runs'].mean()), 2) if len(team_inn) else 0.0,
                }
            else:
                # Team breakdown across all teams in this slice
                team_avgs = inn.groupby('batting_team', observed=True).agg(
                    innings=('match_id', 'count'),
                    PP1_runs=('PP1_runs', 'mean'), PP2_runs=('PP2_runs', 'mean'),
                    PP3_runs=('PP3_runs', 'mean'), PP4_runs=('PP4_runs', 'mean'),
                    total_runs=('total_runs', 'mean'),
                ).reset_index()
                team_avgs = team_avgs[team_avgs['innings'] > 0].sort_values('total_runs', ascending=False)
                breakdown = []
                for _, r in team_avgs.iterrows():
                    breakdown.append({
                        'team': str(r['batting_team']),
                        'innings': int(r['innings']),
                        'PP1': round(float(r['PP1_runs']), 2),
                        'PP2': round(float(r['PP2_runs']), 2),
                        'PP3': round(float(r['PP3_runs']), 2),
                        'PP4': round(float(r['PP4_runs']), 2),
                        'avg_total': round(float(r['total_runs']), 2),
                    })
                result['team_breakdown'] = breakdown

            # Recent matches (last N by start_date if available, else last N match_ids)
            order_col = None
            if 'start_date' in self.df.columns:
                date_map = self.df.groupby('match_id')['start_date'].first()
                inn['_order'] = inn['match_id'].map(date_map)
                order_col = '_order'
            recent_mids = []
            if order_col:
                ordered = inn[['match_id', order_col]].drop_duplicates('match_id').sort_values(order_col, ascending=False)
                recent_mids = ordered['match_id'].head(n_recent).tolist()
            else:
                recent_mids = list(inn['match_id'].drop_duplicates().tail(n_recent))

            recent = []
            for mid in recent_mids:
                g = inn[inn['match_id'] == mid]
                if g['innings'].nunique() < 2:
                    continue
                g1 = g[g['innings'] == 1].iloc[0]
                g2 = g[g['innings'] == 2].iloc[0]
                date_str = ''
                if order_col:
                    try:
                        date_str = str(g[order_col].iloc[0])[:10]
                    except Exception:
                        date_str = ''
                recent.append({
                    'match_id': str(mid),
                    'date': date_str,
                    'team1': str(g1['batting_team']),
                    'team2': str(g2['batting_team']),
                    'winner': str(g1.get('winner') or ''),
                    'team1_phases': {p: {'runs': int(g1[f'{p}_runs']), 'wkts': int(g1[f'{p}_wkts'])} for p in ['PP1', 'PP2', 'PP3', 'PP4']},
                    'team2_phases': {p: {'runs': int(g2[f'{p}_runs']), 'wkts': int(g2[f'{p}_wkts'])} for p in ['PP1', 'PP2', 'PP3', 'PP4']},
                    'team1_total': int(g1['total_runs']),
                    'team2_total': int(g2['total_runs']),
                    'team1_wkts': int(g1['total_wkts']),
                    'team2_wkts': int(g2['total_wkts']),
                })
            result['recent_matches'] = recent

            # ---- Phase specialists: top batters & bowlers per phase ----
            try:
                # Min thresholds keep cameos out of leaderboards
                MIN_BAT_BALLS = 15
                MIN_BOWL_BALLS = 30
                top_n = 10

                bat_grp = df.groupby(['batsman', 'phase'], observed=True).agg(
                    runs=('runs_of_bat', 'sum'),
                    balls=('runs_of_bat', 'size'),
                    fours=('isFour', 'sum'),
                    sixes=('isSix', 'sum'),
                    innings=('match_id', 'nunique'),
                ).reset_index()

                bowl_grp = df.groupby(['bowler', 'phase'], observed=True).agg(
                    runs=('total_run', 'sum'),
                    balls=('isBowlerWk', 'size'),
                    wickets=('isBowlerWk', 'sum'),
                    dots=('isDot', 'sum'),
                    innings=('match_id', 'nunique'),
                ).reset_index()

                specialists = {}
                for p in ['PP1', 'PP2', 'PP3', 'PP4']:
                    # Batters: rank by SR among those with enough balls; tie-break by runs
                    bb = bat_grp[(bat_grp['phase'] == p) & (bat_grp['balls'] >= MIN_BAT_BALLS)].copy()
                    bb['SR'] = (bb['runs'] / bb['balls'] * 100).round(2)
                    bb['boundary_pct'] = ((bb['fours'] + bb['sixes']) / bb['balls'] * 100).round(2)
                    bb = bb.sort_values(['SR', 'runs'], ascending=[False, False]).head(top_n)
                    bat_list = [{
                        'name': str(r['batsman']),
                        'innings': int(r['innings']),
                        'runs': int(r['runs']),
                        'balls': int(r['balls']),
                        'SR': float(r['SR']),
                        'boundary_pct': float(r['boundary_pct']),
                    } for _, r in bb.iterrows()]

                    # Bowlers: rank by economy among those with enough balls; tie-break by wickets desc
                    bw = bowl_grp[(bowl_grp['phase'] == p) & (bowl_grp['balls'] >= MIN_BOWL_BALLS)].copy()
                    bw['ECO'] = (bw['runs'] / (bw['balls'] / 6.0)).round(2)
                    bw['dot_pct'] = (bw['dots'] / bw['balls'] * 100).round(2)
                    bw = bw.sort_values(['ECO', 'wickets'], ascending=[True, False]).head(top_n)
                    bowl_list = [{
                        'name': str(r['bowler']),
                        'innings': int(r['innings']),
                        'balls': int(r['balls']),
                        'runs': int(r['runs']),
                        'wickets': int(r['wickets']),
                        'ECO': float(r['ECO']),
                        'dot_pct': float(r['dot_pct']),
                    } for _, r in bw.iterrows()]

                    specialists[p] = {'batters': bat_list, 'bowlers': bowl_list}
                result['specialists'] = specialists
            except Exception as e:
                print(f"Error computing phase specialists: {e}")
                result['specialists'] = None

            return result
        except Exception as e:
            print(f"Error in get_phase_analysis: {e}")
            import traceback; traceback.print_exc()
            return None

    def get_team_vs_team(self, team_a, team_b, venue=None, innings_filter=None, last_n=5, top_n=10):
        """Head-to-head between two teams: overall, per-venue, last N matches, top scorers/wicket-takers."""
        try:
            df = self.df
            pair_mask = (
                ((df['batting_team'] == team_a) & (df['bowling_team'] == team_b)) |
                ((df['batting_team'] == team_b) & (df['bowling_team'] == team_a))
            )
            df = df[pair_mask]
            if venue:
                df = df[df['venue'] == venue]
            if innings_filter in [1, 2]:
                df_inn = df[df['innings'] == innings_filter]
            else:
                df_inn = df

            if df.empty:
                return None

            # Per-match innings totals (use df, not df_inn, so winner can be derived from both innings)
            inn_totals = df.groupby(['match_id', 'innings', 'batting_team', 'bowling_team', 'venue', 'start_date'],
                                    observed=True)['total_runs'].sum().reset_index()

            # Build match-level records
            matches = {}
            for _, r in inn_totals.iterrows():
                mid = r['match_id']
                if mid not in matches:
                    matches[mid] = {'venue': str(r['venue']), 'date': str(r['start_date']), 'innings': {}}
                matches[mid]['innings'][int(r['innings'])] = {
                    'batting_team': str(r['batting_team']),
                    'bowling_team': str(r['bowling_team']),
                    'runs': int(r['total_runs'])
                }

            a_wins = b_wins = no_result = 0
            a_runs_total = a_innings_count = 0
            b_runs_total = b_innings_count = 0
            match_rows = []
            per_venue = {}
            for mid, m in matches.items():
                inn1 = m['innings'].get(1); inn2 = m['innings'].get(2)
                if not inn1 or not inn2:
                    no_result += 1
                    continue
                # Determine winner by score (assume completed: higher total wins)
                if inn1['runs'] > inn2['runs']:
                    winner = inn1['batting_team']; margin = f"{inn1['runs']-inn2['runs']} runs"
                elif inn2['runs'] > inn1['runs']:
                    winner = inn2['batting_team']; margin = "by chasing"
                else:
                    winner = 'Tie'; margin = 'Tie'
                # Aggregate runs by team
                for inn in (inn1, inn2):
                    if inn['batting_team'] == team_a:
                        a_runs_total += inn['runs']; a_innings_count += 1
                    elif inn['batting_team'] == team_b:
                        b_runs_total += inn['runs']; b_innings_count += 1
                if winner == team_a: a_wins += 1
                elif winner == team_b: b_wins += 1
                v = m['venue']
                per_venue.setdefault(v, {'matches': 0, 'a_wins': 0, 'b_wins': 0})
                per_venue[v]['matches'] += 1
                if winner == team_a: per_venue[v]['a_wins'] += 1
                elif winner == team_b: per_venue[v]['b_wins'] += 1
                match_rows.append({
                    'match_id': str(mid),
                    'date': str(m['date']),
                    'venue': m['venue'],
                    'team1': inn1['batting_team'],
                    'team1_score': inn1['runs'],
                    'team2': inn2['batting_team'],
                    'team2_score': inn2['runs'],
                    'winner': winner,
                    'margin': margin,
                })
            completed = a_wins + b_wins + (sum(1 for r in match_rows if r['winner'] == 'Tie'))

            match_rows.sort(key=lambda r: r['date'], reverse=True)
            recent = match_rows[:last_n]

            venue_breakdown = sorted([
                {'venue': v, **d} for v, d in per_venue.items()
            ], key=lambda r: -r['matches'])

            # Top run scorers (use df_inn so innings filter applies)
            runs_by_bat = df_inn.groupby(['batsman', 'batting_team'], observed=True).agg(
                runs=('runs_of_bat', 'sum'),
                balls=('runs_of_bat', 'size'),
                innings=('match_id', 'nunique'),
            ).reset_index().sort_values('runs', ascending=False).head(top_n)
            top_scorers = []
            for _, r in runs_by_bat.iterrows():
                balls = int(r['balls'])
                runs = int(r['runs'])
                top_scorers.append({
                    'name': str(r['batsman']),
                    'team': str(r['batting_team']),
                    'innings': int(r['innings']),
                    'runs': runs,
                    'balls': balls,
                    'sr': round(runs * 100.0 / balls, 2) if balls else 0.0,
                    'avg': round(runs / int(r['innings']), 2) if r['innings'] else 0.0,
                })

            # Top wicket takers
            wk_by_bowl = df_inn.groupby(['bowler', 'bowling_team'], observed=True).agg(
                wickets=('isBowlerWk', 'sum'),
                runs=('total_run', 'sum'),
                balls=('isBowlerWk', 'size'),
                innings=('match_id', 'nunique'),
            ).reset_index().sort_values('wickets', ascending=False).head(top_n)
            top_wickets = []
            for _, r in wk_by_bowl.iterrows():
                w = int(r['wickets']); b = int(r['balls']); rn = int(r['runs'])
                top_wickets.append({
                    'name': str(r['bowler']),
                    'team': str(r['bowling_team']),
                    'innings': int(r['innings']),
                    'wickets': w,
                    'balls': b,
                    'runs': rn,
                    'eco': round(rn / (b / 6.0), 2) if b else 0.0,
                    'avg': round(rn / w, 2) if w else 0.0,
                })

            return {
                'team_a': team_a,
                'team_b': team_b,
                'venue': venue,
                'innings_filter': innings_filter,
                'overall': {
                    'matches': len(match_rows) + no_result,
                    'completed': completed,
                    'a_wins': a_wins,
                    'b_wins': b_wins,
                    'no_result': no_result,
                    'a_win_pct': round(a_wins * 100.0 / completed, 2) if completed else 0.0,
                    'b_win_pct': round(b_wins * 100.0 / completed, 2) if completed else 0.0,
                    'a_avg_score': round(a_runs_total / a_innings_count, 2) if a_innings_count else 0.0,
                    'b_avg_score': round(b_runs_total / b_innings_count, 2) if b_innings_count else 0.0,
                },
                'venue_breakdown': venue_breakdown,
                'recent_matches': recent,
                'top_scorers': top_scorers,
                'top_wickets': top_wickets,
            }
        except Exception as e:
            print(f"Error in get_team_vs_team: {e}")
            import traceback; traceback.print_exc()
            return None

    def get_winning_patterns(self, team, venue=None):
        """Score-bucket win patterns for a team: batting first and chasing."""
        try:
            df = self.df
            if venue:
                df = df[df['venue'] == venue]
            # Only matches involving team
            team_match_ids = df[(df['batting_team'] == team) | (df['bowling_team'] == team)]['match_id'].unique()
            df = df[df['match_id'].isin(team_match_ids)]
            if df.empty:
                return None

            inn_totals = df.groupby(['match_id', 'innings', 'batting_team'], observed=True)['total_runs'].sum().reset_index()

            buckets = [(0, 119, '<120'), (120, 149, '120-149'), (150, 179, '150-179'),
                       (180, 199, '180-199'), (200, 10**9, '200+')]

            bat_first_rows = {b[2]: {'matches': 0, 'wins': 0, 'losses': 0} for b in buckets}
            chase_rows = {b[2]: {'matches': 0, 'wins': 0, 'losses': 0} for b in buckets}

            # Highlights
            hl_180_bat = {'matches': 0, 'wins': 0}
            hl_200_bat = {'matches': 0, 'wins': 0}
            hl_180_chase = {'matches': 0, 'wins': 0}

            matches = {}
            for _, r in inn_totals.iterrows():
                mid = r['match_id']
                matches.setdefault(mid, {})[int(r['innings'])] = {'team': str(r['batting_team']), 'runs': int(r['total_runs'])}

            total_played = wins_overall = 0
            for mid, m in matches.items():
                i1 = m.get(1); i2 = m.get(2)
                if not i1 or not i2:
                    continue
                if team not in (i1['team'], i2['team']):
                    continue
                total_played += 1
                # Winner = higher score
                if i1['runs'] > i2['runs']: winner = i1['team']
                elif i2['runs'] > i1['runs']: winner = i2['team']
                else: winner = None
                won = (winner == team)
                if won: wins_overall += 1

                # Team batting first?
                if i1['team'] == team:
                    score = i1['runs']
                    for lo, hi, label in buckets:
                        if lo <= score <= hi:
                            bat_first_rows[label]['matches'] += 1
                            if won: bat_first_rows[label]['wins'] += 1
                            else: bat_first_rows[label]['losses'] += 1
                            break
                    if score >= 180:
                        hl_180_bat['matches'] += 1
                        if won: hl_180_bat['wins'] += 1
                    if score >= 200:
                        hl_200_bat['matches'] += 1
                        if won: hl_200_bat['wins'] += 1
                else:
                    # Team chasing; target = i1.runs + 1
                    target = i1['runs'] + 1
                    for lo, hi, label in buckets:
                        if lo <= target <= hi:
                            chase_rows[label]['matches'] += 1
                            if won: chase_rows[label]['wins'] += 1
                            else: chase_rows[label]['losses'] += 1
                            break
                    if target >= 180:
                        hl_180_chase['matches'] += 1
                        if won: hl_180_chase['wins'] += 1

            def _finalize(rows):
                out = []
                for _, _, label in buckets:
                    d = rows[label]
                    m = d['matches']
                    d_out = dict(d)
                    d_out['bucket'] = label
                    d_out['win_pct'] = round(d['wins'] * 100.0 / m, 2) if m else 0.0
                    out.append(d_out)
                return out

            def _pct(d):
                return round(d['wins'] * 100.0 / d['matches'], 2) if d['matches'] else 0.0

            return {
                'team': team,
                'venue': venue,
                'matches_played': total_played,
                'wins_overall': wins_overall,
                'win_pct_overall': round(wins_overall * 100.0 / total_played, 2) if total_played else 0.0,
                'bat_first': _finalize(bat_first_rows),
                'chase': _finalize(chase_rows),
                'highlights': {
                    'bat_180_plus': {**hl_180_bat, 'win_pct': _pct(hl_180_bat)},
                    'bat_200_plus': {**hl_200_bat, 'win_pct': _pct(hl_200_bat)},
                    'chase_180_plus': {**hl_180_chase, 'win_pct': _pct(hl_180_chase)},
                },
            }
        except Exception as e:
            print(f"Error in get_winning_patterns: {e}")
            import traceback; traceback.print_exc()
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
