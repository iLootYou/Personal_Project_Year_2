import pandas as pd
import numpy as np
import glob
import pickle
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt

folder = "data/"
mapping = {'H':0, 'D':1, 'A': 2}

# load all the data
print("Loading data...")
all_files = glob.glob(f"{folder}/*.csv")
dfs = []

for file in all_files:
    # Encoding 'ISO-8859-1' because the files werent in UTF-8 which is the standard for Pandas
    df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1')

    # Parse DD/MM/YY
    date_2d = pd.to_datetime(
        df["Date"],
        format="%d/%m/%y",
        dayfirst=True,
        errors="coerce"
    )

    # Parse DD/MM/YYYY
    date_4d = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        dayfirst=True,
        errors="coerce"
    )
    df["Date"] = date_2d.fillna(date_4d)
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data = all_data.dropna(subset=["Date","FTR"])
all_data = all_data.sort_values(by="Date")

print(f"Matches loaded: {len(all_data)}")


def head2head_training(home_team,away_team, all_data_sorted, match_date, current_idx, h2h_matches):
    pair = tuple(sorted([home_team, away_team]))

    if pair not in h2h_matches:
        return[0] * 42

    # Get matches before current date using the indices
    h2h_indices = h2h_matches[pair]
    # Filter for indices before current match and before date
    valid_indices = [i for i in h2h_indices if i < current_idx and all_data_sorted.loc[i, 'Date'] < match_date]

    if not valid_indices:
        return[0] * 42
    
    last_5_indices = valid_indices[-5:] if len(valid_indices) >= 5 else valid_indices 
    last_5 = all_data_sorted.loc[last_5_indices]

    num_matches = len(last_5)

    # Shape[0] counts rows passing both filters, shape[1] gives columns and shape gives both
    # Win/Draw/Loss stats
    H2H_hometeam_wins_home = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'H')].shape[0]
    H2H_hometeam_wins_away = last_5[(last_5['AwayTeam'] == home_team) & (last_5['FTR'] == 'A')].shape[0]
    H2H_hometeam_wins = H2H_hometeam_wins_home + H2H_hometeam_wins_away

    H2H_awayteam_wins_home = last_5[(last_5['HomeTeam'] == away_team) & (last_5['FTR'] == 'H')].shape[0]
    H2H_awayteam_wins_away = last_5[(last_5['AwayTeam'] == away_team) & (last_5['FTR'] == 'A')].shape[0]
    H2H_awayteam_wins = H2H_awayteam_wins_home + H2H_awayteam_wins_away

    H2H_draws = (last_5['FTR'] == 'D').sum()
    
    # Win percentages
    H2H_hometeam_win_pct = H2H_hometeam_wins / num_matches  
    H2H_awayteam_win_pct = H2H_awayteam_wins / num_matches

    # Half time goals
    H2H_hometeam_halftime_goals_home = last_5[(last_5['HomeTeam'] == home_team)]['HTHG'].sum()
    H2H_hometeam_halftime_goals_away = last_5[(last_5['AwayTeam'] == home_team)]['HTAG'].sum()
    H2H_hometeam_halftime_goals = H2H_hometeam_halftime_goals_home + H2H_hometeam_halftime_goals_away

    H2H_awayteam_halftime_goals_home = last_5[(last_5['HomeTeam'] == away_team)]['HTHG'].sum()
    H2H_awayteam_halftime_goals_away = last_5[(last_5['AwayTeam'] == away_team)]['HTAG'].sum()
    H2H_awayteam_halftime_goals = H2H_awayteam_halftime_goals_home + H2H_awayteam_halftime_goals_away

    # Full time goals
    H2H_hometeam_fulltime_goals_home = last_5[(last_5['HomeTeam'] == home_team)]['FTHG'].sum()
    H2H_hometeam_fulltime_goals_away = last_5[(last_5['AwayTeam'] == home_team)]['FTAG'].sum()
    H2H_hometeam_fulltime_goals = H2H_hometeam_fulltime_goals_home + H2H_hometeam_fulltime_goals_away

    H2H_awayteam_fulltime_goals_home = last_5[(last_5['HomeTeam'] == away_team)]['FTHG'].sum()
    H2H_awayteam_fulltime_goals_away = last_5[(last_5['AwayTeam'] == away_team)]['FTAG'].sum()
    H2H_awayteam_fulltime_goals = H2H_awayteam_fulltime_goals_home + H2H_awayteam_fulltime_goals_away

    # Goal difference
    H2H_goal_diff = H2H_hometeam_fulltime_goals - H2H_awayteam_fulltime_goals

    # Average goals per match
    H2H_hometeam_avg_goals = H2H_hometeam_fulltime_goals / num_matches
    H2H_awayteam_avg_goals = H2H_awayteam_fulltime_goals / num_matches

    # Goals conceded
    H2H_hometeam_goals_conceded = H2H_awayteam_fulltime_goals
    H2H_awayteam_goals_conceded = H2H_hometeam_fulltime_goals

    # Half time results
    H2H_hometeam_halftime_wins_home = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'H')].shape[0]
    H2H_hometeam_halftime_wins_away = last_5[(last_5['AwayTeam'] == home_team) & (last_5['HTR'] == 'A')].shape[0]
    H2H_hometeam_halftime_wins = H2H_hometeam_halftime_wins_home + H2H_hometeam_halftime_wins_away

    H2H_awayteam_halftime_wins_home = last_5[(last_5['HomeTeam'] == away_team) & (last_5['HTR'] == 'H')].shape[0]
    H2H_awayteam_halftime_wins_away = last_5[(last_5['AwayTeam'] == away_team) & (last_5['HTR'] == 'A')].shape[0]
    H2H_awayteam_halftime_wins = H2H_awayteam_halftime_wins_home + H2H_awayteam_halftime_wins_away

    H2H_halftime_draws = (last_5['HTR'] == 'D').sum()


    # Total shots taken 
    H2H_hometeam_shots_home = last_5[(last_5['HomeTeam'] == home_team)]['HS'].sum()
    H2H_hometeam_shots_away = last_5[(last_5['AwayTeam'] == home_team)]['AS'].sum()
    H2H_hometeam_shots = H2H_hometeam_shots_home + H2H_hometeam_shots_away

    H2H_awayteam_shots_home = last_5[(last_5['HomeTeam'] == away_team)]['HS'].sum()
    H2H_awayteam_shots_away = last_5[(last_5['AwayTeam'] == away_team)]['AS'].sum()
    H2H_awayteam_shots = H2H_awayteam_shots_home + H2H_awayteam_shots_away

    # Total shots on target 
    H2H_hometeam_shots_on_target_home = last_5[(last_5['HomeTeam'] == home_team)]['HST'].sum()
    H2H_hometeam_shots_on_target_away = last_5[(last_5['AwayTeam'] == home_team)]['AST'].sum()
    H2H_hometeam_shots_on_target = H2H_hometeam_shots_on_target_home + H2H_hometeam_shots_on_target_away

    H2H_awayteam_shots_on_target_home = last_5[(last_5['HomeTeam'] == away_team)]['HST'].sum()
    H2H_awayteam_shots_on_target_away = last_5[(last_5['AwayTeam'] == away_team)]['AST'].sum()
    H2H_awayteam_shots_on_target = H2H_awayteam_shots_on_target_home + H2H_awayteam_shots_on_target_away

    # Shot accuracy and conversion
    H2H_hometeam_shot_accuracy = H2H_hometeam_shots_on_target / H2H_hometeam_shots if H2H_hometeam_shots > 0 else 0
    H2H_awayteam_shot_accuracy = H2H_awayteam_shots_on_target / H2H_awayteam_shots if H2H_awayteam_shots > 0 else 0
    
    H2H_hometeam_conversion = H2H_hometeam_fulltime_goals / H2H_hometeam_shots_on_target if H2H_hometeam_shots_on_target > 0 else 0
    H2H_awayteam_conversion = H2H_awayteam_fulltime_goals / H2H_awayteam_shots_on_target if H2H_awayteam_shots_on_target > 0 else 0

    # Total woodwork hits 
    H2H_hometeam_woodwork_home = last_5[(last_5['HomeTeam'] == home_team)]['HHW'].sum()
    H2H_hometeam_woodwork_away = last_5[(last_5['AwayTeam'] == home_team)]['AHW'].sum()
    H2H_hometeam_woodwork = H2H_hometeam_woodwork_home + H2H_hometeam_woodwork_away

    H2H_awayteam_woodwork_home = last_5[(last_5['HomeTeam'] == away_team)]['HHW'].sum()
    H2H_awayteam_woodwork_away = last_5[(last_5['AwayTeam'] == away_team)]['AHW'].sum()
    H2H_awayteam_woodwork = H2H_awayteam_woodwork_home + H2H_awayteam_woodwork_away

    # Total corners
    H2H_hometeam_corners_home = last_5[(last_5['HomeTeam'] == home_team)]['HC'].sum()
    H2H_hometeam_corners_away = last_5[(last_5['AwayTeam'] == home_team)]['AC'].sum()
    H2H_hometeam_corners = H2H_hometeam_corners_home + H2H_hometeam_corners_away

    H2H_awayteam_corners_home = last_5[(last_5['HomeTeam'] == away_team)]['HC'].sum()
    H2H_awayteam_corners_away = last_5[(last_5['AwayTeam'] == away_team)]['AC'].sum()
    H2H_awayteam_corners = H2H_awayteam_corners_home + H2H_awayteam_corners_away

    # Total fouls 
    H2H_hometeam_fouls_home = last_5[(last_5['HomeTeam'] == home_team)]['HF'].sum()
    H2H_hometeam_fouls_away = last_5[(last_5['AwayTeam'] == home_team)]['AF'].sum()
    H2H_hometeam_fouls = H2H_hometeam_fouls_home + H2H_hometeam_fouls_away

    H2H_awayteam_fouls_home = last_5[(last_5['HomeTeam'] == away_team)]['HF'].sum()
    H2H_awayteam_fouls_away = last_5[(last_5['AwayTeam'] == away_team)]['AF'].sum()
    H2H_awayteam_fouls = H2H_awayteam_fouls_home + H2H_awayteam_fouls_away
  
    # Total offsides 
    H2H_hometeam_offside_home = last_5[(last_5['HomeTeam'] == home_team)]['HO'].sum()
    H2H_hometeam_offside_away = last_5[(last_5['AwayTeam'] == home_team)]['AO'].sum()
    H2H_hometeam_offsides = H2H_hometeam_offside_home + H2H_hometeam_offside_away

    H2H_awayteam_offside_home = last_5[(last_5['HomeTeam'] == away_team)]['HO'].sum()
    H2H_awayteam_offside_away = last_5[(last_5['AwayTeam'] == away_team)]['AO'].sum()
    H2H_awayteam_offsides = H2H_awayteam_offside_home + H2H_awayteam_offside_away

    # Total yellow cards
    H2H_hometeam_yellow_card_home = last_5[(last_5['HomeTeam'] == home_team)]['HY'].sum()
    H2H_hometeam_yellow_card_away = last_5[(last_5['AwayTeam'] == home_team)]['AY'].sum()
    H2H_hometeam_yellow_card = H2H_hometeam_yellow_card_home + H2H_hometeam_yellow_card_away

    H2H_awayteam_yellow_card_home = last_5[(last_5['HomeTeam'] == away_team)]['HY'].sum()
    H2H_awayteam_yellow_card_away = last_5[(last_5['AwayTeam'] == away_team)]['AY'].sum()
    H2H_awayteam_yellow_card = H2H_awayteam_yellow_card_home + H2H_awayteam_yellow_card_away  

    # Total red cards
    H2H_hometeam_red_card_home = last_5[(last_5['HomeTeam'] == home_team)]['HR'].sum()
    H2H_hometeam_red_card_away = last_5[(last_5['AwayTeam'] == home_team)]['AR'].sum()
    H2H_hometeam_red_card = H2H_hometeam_red_card_home + H2H_hometeam_red_card_away 

    H2H_awayteam_red_card_home = last_5[(last_5['HomeTeam'] == away_team)]['HR'].sum()
    H2H_awayteam_red_card_away = last_5[(last_5['AwayTeam'] == away_team)]['AR'].sum()
    H2H_awayteam_red_card = H2H_awayteam_red_card_home + H2H_awayteam_red_card_away

    # For HomeTeam home odds
    subset_home = last_5[(last_5['HomeTeam'] == home_team)]['B365H']
    if subset_home.empty or subset_home.isnull().all():
        H2H_bet365_hometeam_probability_home = 0
    else:
        mean_val = subset_home.mean()
        H2H_bet365_hometeam_probability_home = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # For HomeTeam away odds
    subset_away = last_5[(last_5['AwayTeam'] == home_team)]['B365A']
    if subset_away.empty or subset_away.isnull().all():
        H2H_bet365_hometeam_probability_away = 0
    else:
        mean_val = subset_away.mean()
        H2H_bet365_hometeam_probability_away = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # For AwayTeam home odds
    subset_away_team_home = last_5[(last_5['HomeTeam'] == away_team)]['B365H']
    if subset_away_team_home.empty or subset_away_team_home.isnull().all():
        H2H_bet365_awayteam_probability_home = 0
    else:
        mean_val = subset_away_team_home.mean()
        H2H_bet365_awayteam_probability_home = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # For AwayTeam away odds
    subset_away_team_away = last_5[(last_5['AwayTeam'] == away_team)]['B365A']
    if subset_away_team_away.empty or subset_away_team_away.isnull().all():
        H2H_bet365_awayteam_probability_away = 0
    else:
        mean_val = subset_away_team_away.mean()
        H2H_bet365_awayteam_probability_away = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # For draws
    subset_draws = last_5['B365D']
    if subset_draws.empty or subset_draws.isnull().all():
        H2H_bet365_probability_draws = 0
    else:
        mean_val = subset_draws.mean()
        H2H_bet365_probability_draws = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    return [
        H2H_hometeam_wins, H2H_awayteam_wins, H2H_draws, 
        H2H_hometeam_win_pct, H2H_awayteam_win_pct,  
        H2H_hometeam_halftime_goals, H2H_awayteam_halftime_goals, 
        H2H_hometeam_fulltime_goals, H2H_awayteam_fulltime_goals, 
        H2H_goal_diff, H2H_hometeam_avg_goals, H2H_awayteam_avg_goals,  
        H2H_hometeam_goals_conceded, H2H_awayteam_goals_conceded,  
        H2H_hometeam_halftime_wins, H2H_awayteam_halftime_wins, H2H_halftime_draws, 
        H2H_hometeam_shots, H2H_awayteam_shots, 
        H2H_hometeam_shots_on_target, H2H_awayteam_shots_on_target, 
        H2H_hometeam_shot_accuracy, H2H_awayteam_shot_accuracy,  
        H2H_hometeam_conversion, H2H_awayteam_conversion,  
        H2H_hometeam_woodwork, H2H_awayteam_woodwork, 
        H2H_hometeam_corners, H2H_awayteam_corners, 
        H2H_hometeam_fouls, H2H_awayteam_fouls, 
        H2H_hometeam_offsides, H2H_awayteam_offsides, 
        H2H_hometeam_yellow_card, H2H_awayteam_yellow_card, 
        H2H_hometeam_red_card, H2H_awayteam_red_card, 
        H2H_bet365_hometeam_probability_home, H2H_bet365_hometeam_probability_away,
        H2H_bet365_awayteam_probability_home, H2H_bet365_awayteam_probability_away,
        H2H_bet365_probability_draws
    ]

def home_matches_training(home_team, all_data_sorted, match_date, current_idx, team_matches):
    if home_team not in team_matches:
        return[0] * 31

    # Get home match indices for team
    home_indices = team_matches[home_team]['home_indices']
    # Filter for indices before current match and before date
    valid_indices = [i for i in home_indices if i < current_idx and all_data_sorted.loc[i,'Date'] < match_date]
    
    if not valid_indices:
        return[0] * 31
    
    # Get last 5 matches
    last_5_indices = valid_indices[-5:] if len(valid_indices) >= 5 else valid_indices
    last_5 = all_data_sorted.loc[last_5_indices]

    num_matches = len(last_5)

    # Match results
    home_match_wins = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'H')].shape[0]
    home_win_rate = home_match_wins / 5  # Last 5 matches
    home_match_losses = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'A')].shape[0]
    home_match_draws = (last_5['FTR'] == 'D').sum()

    # Goals scored
    home_match_halftime_goals = last_5[(last_5['HomeTeam'] == home_team)]['HTHG'].sum()
    home_match_fulltime_goals = last_5[(last_5['HomeTeam'] == home_team)]['FTHG'].sum()

    # Goals conceded
    home_match_goals_conceded = last_5['FTAG'].sum()
    home_match_avg_goals_conceded = home_match_goals_conceded / num_matches

    # Clean sheets (zero goals conceded)
    home_match_clean_sheets = (last_5['FTAG'] == 0).sum()

    # Halftime results
    home_match_halftime_wins = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'H')].shape[0]
    home_match_halftime_losses = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'A')].shape[0]
    home_match_halftime_draws = (last_5['HTR'] == 'D').sum()

    # Shooting stats
    home_match_shots = last_5[(last_5['HomeTeam'] == home_team)]['HS'].sum()
    home_match_shots_on_target = last_5[(last_5['HomeTeam'] == home_team)]['HST'].sum()
    shot_accuracy = home_match_shots_on_target / home_match_shots if home_match_shots > 0 else 0
    conversion_rate = home_match_fulltime_goals / home_match_shots_on_target if home_match_shots_on_target > 0 else 0

    # Opponents shots (defensive pressure)
    home_match_shots_against = last_5['AS'].sum()
    home_match_shots_on_target_against = last_5['AST'].sum()
    home_match_woodwork_against = last_5['AHW'].sum()

    # Woodwork
    home_match_woodwork = last_5[(last_5['HomeTeam'] == home_team)]['HHW'].sum()

    # Corners
    home_match_corners = last_5[(last_5['HomeTeam'] == home_team)]['HC'].sum()

    # Corners against 
    home_match_corners_against = last_5['AC'].sum()

    # Discipline
    home_match_fouls = last_5[(last_5['HomeTeam'] == home_team)]['HF'].sum()
    home_match_offsides = last_5[(last_5['HomeTeam'] == home_team)]['HO'].sum()
    home_match_yellow_card = last_5[(last_5['HomeTeam'] == home_team)]['HY'].sum()
    home_match_red_card = last_5[(last_5['HomeTeam'] == home_team)]['HR'].sum()

    # Form indicator, recent points
    recent_points = (home_match_wins * 3) + home_match_draws

    # Momentum indicator (weighted recent results)
    if num_matches >= 3:
        recent_results = []
        # Index not needed so use "_" and row to get the data
        for _, row in last_5.iterrows():
            if row['FTR'] == 'H':
                recent_results.append(3)
            elif row['FTR'] == 'D':
                recent_results.append(1)
            else:
                recent_results.append(0)
        #Weight more recent matches higher, slice to get last num_matches
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][-num_matches:]
        # Zip pairs each result with corresponding weight, for each pair result is multiplied by weight
        # Dividing by the sum of weights normalizes the weights and then we get a weight averaged score
        home_match_momentum = sum(r * w for r, w in zip(recent_results, weights)) / sum(weights)
    else:
        # We normalize by multiplying num_matches by 3 because you can get a max of 3 points per game
        # Normalized score is always between 0 and 1.0
        home_match_momentum = recent_points / (num_matches * 3) if num_matches > 0 else 0

    # Probability Bet365 hometeam wins
    subset_wins = last_5[(last_5["HomeTeam"] == home_team)]['B365H']
    if subset_wins.empty or subset_wins.isnull().all():
        home_match_bet365_probability_wins = 0
    else:
        mean_val = subset_wins.mean()
        home_match_bet365_probability_wins = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # Probability Bet365 hometeam losses
    subset_losses = last_5[(last_5['HomeTeam'] == home_team)]['B365A']
    if subset_losses.empty or subset_losses.isnull().all():
        home_match_bet365_probability_losses = 0
    else:
        mean_val = subset_losses.mean()
        home_match_bet365_probability_losses = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # Probability Bet365 draw
    subset_draws = last_5['B365D']
    if subset_draws.empty or subset_draws.isnull().all():
        home_match_bet365_probability_draws = 0
    else:
        mean_val = subset_draws.mean()
        home_match_bet365_probability_draws = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0
    
    return [
        home_match_wins, home_win_rate, home_match_losses, home_match_draws, 
        home_match_halftime_goals, home_match_fulltime_goals, 
        home_match_goals_conceded, home_match_avg_goals_conceded,  
        home_match_clean_sheets,  
        home_match_halftime_wins, home_match_halftime_losses, home_match_halftime_draws, 
        home_match_shots, home_match_shots_on_target, shot_accuracy, conversion_rate, 
        home_match_shots_against, home_match_shots_on_target_against, home_match_woodwork_against,  
        home_match_woodwork, home_match_corners, home_match_corners_against,  
        home_match_fouls, home_match_offsides, 
        home_match_yellow_card, home_match_red_card, 
        recent_points, home_match_momentum,  
        home_match_bet365_probability_wins, home_match_bet365_probability_losses, 
        home_match_bet365_probability_draws
    ]

def away_matches_training(away_team, all_data_sorted, current_idx, match_date, team_matches):
    if away_team not in team_matches:
        return[0] * 31

    # Get home match indices for team
    away_indices = team_matches[away_team]['away_indices']

    dates = all_data_sorted['Date']
    # Filter for indices before current match and before date
    valid_indices = [i for i in away_indices if i < current_idx and all_data_sorted.loc[i,'Date'] < match_date]
    
    if not valid_indices:
        return[0] * 31
    
    # Get last 5 matches
    last_5_indices = valid_indices[-5:] if len(valid_indices) >= 5 else valid_indices
    last_5 = all_data_sorted.loc[last_5_indices]

    num_matches = len(last_5)

    
    num_matches = len(last_5)

    # Match results
    away_match_wins = last_5[(last_5['AwayTeam'] == away_team) & (last_5['FTR'] == 'A')].shape[0]
    away_win_rate = away_match_wins / num_matches
    away_match_losses = last_5[(last_5['AwayTeam'] == away_team) & (last_5['FTR'] == 'H')].shape[0]
    away_match_draws = (last_5['FTR'] == 'D').sum()

    # Goals scored
    away_match_halftime_goals = last_5[(last_5['AwayTeam'] == away_team)]['HTAG'].sum()
    away_match_fulltime_goals = last_5[(last_5['AwayTeam'] == away_team)]['FTAG'].sum()

    # Goals conceded
    away_match_goals_conceded = last_5['FTHG'].sum()
    away_match_avg_goals_conceded = away_match_goals_conceded / num_matches

    # Clean sheets (zero goals conceded)
    away_match_clean_sheets = (last_5['FTHG'] == 0).sum()

    # Half time results
    away_match_halftime_wins = last_5[(last_5['AwayTeam'] == away_team) & (last_5['HTR'] == 'A')].shape[0]
    away_match_halftime_losses = last_5[(last_5['AwayTeam'] == away_team) & (last_5['HTR'] == 'H')].shape[0]
    away_match_halftime_draws = (last_5['HTR'] == 'D').sum()

    # Shooting stats
    away_match_shots = last_5[(last_5['AwayTeam'] == away_team)]['AS'].sum()
    away_match_shots_on_target = last_5[(last_5['AwayTeam'] == away_team)]['AST'].sum()
    shot_accuracy = away_match_shots_on_target / away_match_shots if away_match_shots > 0 else 0
    conversion_rate = away_match_fulltime_goals / away_match_shots_on_target if away_match_shots_on_target > 0 else 0

    # Opponents shots (defensive pressure)
    away_match_shots_against = last_5['HS'].sum()
    away_match_shots_on_target_against = last_5['HST'].sum()
    away_match_woodwork_against = last_5['HHW'].sum()

    # Woodwork
    away_match_woodwork = last_5[(last_5['AwayTeam'] == away_team)]['AHW'].sum()

    # Corners
    away_match_corners = last_5[(last_5['AwayTeam'] == away_team)]['AC'].sum()

    # Corners against
    away_match_corners_against = last_5['HC'].sum()

    # Discipline
    away_match_fouls = last_5[(last_5['AwayTeam'] == away_team)]['AF'].sum()
    away_match_offsides = last_5[(last_5['AwayTeam'] == away_team)]['AO'].sum()
    away_match_yellow_card = last_5[(last_5['AwayTeam'] == away_team)]['AY'].sum()
    away_match_red_card = last_5[(last_5['AwayTeam'] == away_team)]['AR'].sum()

    # Form indicator, recent points
    recent_points = (away_match_wins * 3) + away_match_draws

    # Momentum indicator (weighted recent results)
    if num_matches >= 3:
        recent_results = []
        # Index not needed so use "_" and row to get the data
        for _, row in last_5.iterrows():
            if row['FTR'] == 'A':
                recent_results.append(3)
            elif row['FTR'] == 'D':
                recent_results.append(1)
            else:
                recent_results.append(0)
        #Weight more recent matches higher, slice to get last num_matches
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][-num_matches:]
        # Zip pairs each result with corresponding weight, for each pair result is multiplied by weight
        # Dividing by the sum of weights normalizes the weights and then we get a weight averaged score
        away_match_momentum = sum(r * w for r, w in zip(recent_results, weights)) / sum(weights)
    else:
        # We normalize by multiplying num_matches by 3 because you can get a max of 3 points per game
        # Normalized score is always between 0 and 1.0
        away_match_momentum = recent_points / (num_matches * 3) if num_matches > 0 else 0

    # Probability Bet365 awayteam wins
    subset_wins = last_5[(last_5["AwayTeam"] == away_team)]['B365A']
    if subset_wins.empty or subset_wins.isnull().all():
        away_match_bet365_probability_wins = 0
    else:
        mean_val = subset_wins.mean()
        away_match_bet365_probability_wins = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # Probability Bet365 awayteam losses
    subset_losses = last_5[(last_5['AwayTeam'] == away_team)]['B365H']
    if subset_losses.empty or subset_losses.isnull().all():
        away_match_bet365_probability_losses = 0
    else:
        mean_val = subset_losses.mean()
        away_match_bet365_probability_losses = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0

    # Probability Bet365 draw
    subset_draws = last_5['B365D']
    if subset_draws.empty or subset_draws.isnull().all():
        away_match_bet365_probability_draws = 0
    else:
        mean_val = subset_draws.mean()
        away_match_bet365_probability_draws = 1 / mean_val if mean_val and not np.isnan(mean_val) else 0
    
    return [
        away_match_wins, away_win_rate, away_match_losses, away_match_draws, 
        away_match_halftime_goals, away_match_fulltime_goals, 
        away_match_goals_conceded, away_match_avg_goals_conceded, 
        away_match_clean_sheets,
        away_match_halftime_wins, away_match_halftime_losses, away_match_halftime_draws, 
        away_match_shots, away_match_shots_on_target, shot_accuracy, conversion_rate, 
        away_match_shots_against, away_match_shots_on_target_against, away_match_woodwork_against,  
        away_match_woodwork, away_match_corners, away_match_corners_against,  
        away_match_fouls, away_match_offsides, 
        away_match_yellow_card, away_match_red_card, 
        recent_points, away_match_momentum,  
        away_match_bet365_probability_wins, away_match_bet365_probability_losses, 
        away_match_bet365_probability_draws
    ]

def feature_interactions(h2h_stats, home_stats, away_stats):
    # Unpack h2h_stats (42 features)
    (H2H_hometeam_wins, H2H_awayteam_wins, H2H_draws, 
     H2H_hometeam_win_pct, H2H_awayteam_win_pct,  
     H2H_hometeam_halftime_goals, H2H_awayteam_halftime_goals, 
     H2H_hometeam_fulltime_goals, H2H_awayteam_fulltime_goals, 
     H2H_goal_diff, H2H_hometeam_avg_goals, H2H_awayteam_avg_goals,  
     H2H_hometeam_goals_conceded, H2H_awayteam_goals_conceded,  
     H2H_hometeam_halftime_wins, H2H_awayteam_halftime_wins, H2H_halftime_draws, 
     H2H_hometeam_shots, H2H_awayteam_shots, 
     H2H_hometeam_shots_on_target, H2H_awayteam_shots_on_target, 
     H2H_hometeam_shot_accuracy, H2H_awayteam_shot_accuracy,  
     H2H_hometeam_conversion, H2H_awayteam_conversion,  
     H2H_hometeam_woodwork, H2H_awayteam_woodwork, 
     H2H_hometeam_corners, H2H_awayteam_corners, 
     H2H_hometeam_fouls, H2H_awayteam_fouls, 
     H2H_hometeam_offsides, H2H_awayteam_offsides, 
     H2H_hometeam_yellow_card, H2H_awayteam_yellow_card, 
     H2H_hometeam_red_card, H2H_awayteam_red_card, 
     H2H_bet365_hometeam_probability_home, H2H_bet365_hometeam_probability_away,
     H2H_bet365_awayteam_probability_home, H2H_bet365_awayteam_probability_away,
     H2H_bet365_probability_draws) = h2h_stats
    
    # Unpack home_stats (31 features)
    (home_match_wins, home_win_rate, home_match_losses, home_match_draws, 
     home_match_halftime_goals, home_match_fulltime_goals, 
     home_match_goals_conceded, home_match_avg_goals_conceded,  
     home_match_clean_sheets,  
     home_match_halftime_wins, home_match_halftime_losses, home_match_halftime_draws, 
     home_match_shots, home_match_shots_on_target, home_shot_accuracy, home_conversion_rate, 
     home_match_shots_against, home_match_shots_on_target_against, home_match_woodwork_against,  
     home_match_woodwork, home_match_corners, home_match_corners_against,  
     home_match_fouls, home_match_offsides, 
     home_match_yellow_card, home_match_red_card, 
     home_recent_points, home_match_momentum,  
     home_match_bet365_probability_wins, home_match_bet365_probability_losses, 
     home_match_bet365_probability_draws) = home_stats
    
    # Unpack away_stats (31 features)
    (away_match_wins, away_win_rate, away_match_losses, away_match_draws, 
     away_match_halftime_goals, away_match_fulltime_goals, 
     away_match_goals_conceded, away_match_avg_goals_conceded, 
     away_match_clean_sheets,
     away_match_halftime_wins, away_match_halftime_losses, away_match_halftime_draws, 
     away_match_shots, away_match_shots_on_target, away_shot_accuracy, away_conversion_rate, 
     away_match_shots_against, away_match_shots_on_target_against, away_match_woodwork_against,  
     away_match_woodwork, away_match_corners, away_match_corners_against,  
     away_match_fouls, away_match_offsides, 
     away_match_yellow_card, away_match_red_card, 
     away_recent_points, away_match_momentum,  
     away_match_bet365_probability_wins, away_match_bet365_probability_losses, 
     away_match_bet365_probability_draws) = away_stats

    # Difference in momentum
    momentum_difference = home_match_momentum - away_match_momentum

    # Combined momentum score
    combined_momentum = (home_match_momentum + away_match_momentum) / 2

    # Difference in points
    form_points_difference = home_recent_points - away_recent_points

    # Win rate difference 
    win_rate_difference = home_win_rate - away_win_rate

    # H2H win percantage difference
    H2H_win_pct_difference = H2H_hometeam_win_pct - H2H_awayteam_win_pct

    # Home attack vs away defense
    home_attack_vs_away_defense = home_match_fulltime_goals - away_match_goals_conceded

    # Away attack vs home defense
    away_attack_vs_home_defense = away_match_fulltime_goals - home_match_goals_conceded

    # Goal scoring difference
    goal_scoring_difference = home_match_fulltime_goals - away_match_fulltime_goals

    # Average goals concede difference
    goals_conceded_difference = home_match_avg_goals_conceded - away_match_avg_goals_conceded

    # Clean sheet difference
    clean_sheet_difference = home_match_clean_sheets - away_match_clean_sheets

    # Shot accuracy difference
    shot_accuracy_difference = home_shot_accuracy - away_shot_accuracy

    # Conversion rate difference
    conversion_rate_difference = home_conversion_rate - away_conversion_rate

    # Combined shooting efficiency
    home_shooting_efficiency = home_shot_accuracy * home_conversion_rate
    away_shooting_efficiency = away_shot_accuracy * home_conversion_rate
    shooting_efficiency_difference = home_shooting_efficiency - away_shooting_efficiency

    # Shot volume difference
    shots_difference = home_match_shots - away_match_shots

    # Shots on target difference
    shots_on_target_difference = home_match_shots_on_target - away_match_shots_on_target

    # Home defensive pressure
    home_defensive_pressure = home_match_shots_against - away_match_shots

    # Away defensive pressure
    away_defensive_pressure = away_match_shots_against - home_match_shots

    # Shots on target against difference
    shots_on_target_against_difference = home_match_shots_on_target_against - away_match_shots_on_target_against

    # Corner difference 
    corner_difference = home_match_corners - away_match_corners

    # Corner against difference
    corner_against_difference = home_match_corners_against - away_match_corners_against

    # Territory control
    home_territory_control = home_match_corners + home_match_shots
    away_territory_control = away_match_corners + away_match_shots
    territory_control_difference = home_territory_control - away_territory_control

    # Foul difference
    fouls_difference = home_match_fouls - away_match_fouls

    # Yellow card difference
    yellow_card_difference = home_match_yellow_card - away_match_yellow_card

    # Red card difference
    red_card_difference = home_match_red_card - away_match_red_card

    # Discipline score (lower is better)
    home_discipline_score = (home_match_fouls * 0.1) + (home_match_yellow_card * 1) + (home_match_red_card * 3)
    away_discipline_score = (away_match_fouls * 0.1) + (away_match_yellow_card * 1) + (away_match_red_card * 3)
    discipline_score_difference = away_discipline_score - home_discipline_score

    # Offside difference
    offside_difference = home_match_offsides - away_match_offsides

    # Half time goal difference
    halftime_goals_difference = home_match_halftime_goals - away_match_halftime_goals
    
    # Halftime wins difference
    halftime_wins_difference = home_match_halftime_wins - away_match_halftime_wins
    
    # First half strength (halftime wins / total wins)
    home_first_half_strength = home_match_halftime_wins / home_match_wins if home_match_wins > 0 else 0
    away_first_half_strength = away_match_halftime_wins / away_match_wins if away_match_wins > 0 else 0
    first_half_strength_difference = home_first_half_strength - away_first_half_strength
    
    # Bet365 probability difference
    home_bet365_probability = (home_match_bet365_probability_wins + H2H_bet365_hometeam_probability_home + 
                               H2H_bet365_hometeam_probability_away) / 3
    away_bet365_probability = (away_match_bet365_probability_wins + H2H_bet365_awayteam_probability_home + 
                               H2H_bet365_awayteam_probability_away) / 3
    bet365_probability_difference = home_bet365_probability - away_bet365_probability

    # Draw probability average
    draw_probability_average = (home_match_bet365_probability_draws + away_match_bet365_probability_draws + 
                                H2H_bet365_probability_draws) / 3
    
    # Market confidence (how decisive are the odds)
    market_confidence = abs(home_bet365_probability - away_bet365_probability)

    # H2H goal differece impact on form
    H2H_form_alignment = H2H_goal_diff * momentum_difference

    # H2H average goals vs current defensive form
    H2H_avg_goals_total = H2H_hometeam_avg_goals + H2H_awayteam_avg_goals
    current_avg_goals_conceded = (home_match_avg_goals_conceded + away_match_avg_goals_conceded) / 2
    H2H_goals_vs_defense = H2H_avg_goals_total - current_avg_goals_conceded

    # H2H shooting accuracy vs current form
    H2H_shot_accuracy_diff = H2H_hometeam_shot_accuracy - H2H_awayteam_shot_accuracy
    current_shot_accuracy_diff = home_shot_accuracy - away_shot_accuracy
    shot_accuracy_consitency = abs(H2H_shot_accuracy_diff - current_shot_accuracy_diff)

    # H2H discipline vs current discipline
    H2H_discipline_diff = (H2H_hometeam_yellow_card + H2H_hometeam_red_card * 3) - (H2H_awayteam_yellow_card + H2H_awayteam_red_card * 3)
    discipline_consistency = abs(H2H_discipline_diff - discipline_score_difference)

    # Overall attacking threat (goals + shots on target + corners)
    home_attacking_threat = home_match_fulltime_goals + (home_match_shots_on_target / 10) + (home_match_corners / 10)
    away_attacking_threat = away_match_fulltime_goals + (away_match_shots_on_target / 10) + (away_match_corners / 10)
    attacking_threat_differential = home_attacking_threat - away_attacking_threat

    # Overall defensive solidity (clean sheets + limited shots against)
    home_defensive_solidity = home_match_clean_sheets - (home_match_shots_on_target_against / 10)
    away_defensive_solidity = away_match_clean_sheets - (away_match_shots_on_target_against / 10)
    defensive_solidity_differential = home_defensive_solidity - away_defensive_solidity
   
    # Balanced team score (attack * defense)
    home_balance_score = home_attacking_threat * (home_defensive_solidity + 5)  # Add 5 to avoid negative multipliers
    away_balance_score = away_attacking_threat * (away_defensive_solidity + 5)
    balance_score_differential = home_balance_score - away_balance_score
    
    # Form * efficiency composite
    home_form_efficiency = home_match_momentum * home_shooting_efficiency * 10
    away_form_efficiency = away_match_momentum * away_shooting_efficiency * 10
    form_efficiency_differential = home_form_efficiency - away_form_efficiency
    
    # Pressure handling (performance when facing many shots)
    home_pressure_handling = home_match_goals_conceded / home_match_shots_against if home_match_shots_against > 0 else 0
    away_pressure_handling = away_match_goals_conceded / away_match_shots_against if away_match_shots_against > 0 else 0
    pressure_handling_differential = away_pressure_handling - home_pressure_handling  # Lower is better, so reversed
   
    # Finishing quality under pressure (conversion when facing defensive pressure)
    home_finishing_quality = (home_match_fulltime_goals / home_match_shots_on_target) * (1 - (home_match_shots_against / 100)) if home_match_shots_on_target > 0 else 0
    away_finishing_quality = (away_match_fulltime_goals / away_match_shots_on_target) * (1 - (away_match_shots_against / 100)) if away_match_shots_on_target > 0 else 0
    finishing_quality_differential = home_finishing_quality - away_finishing_quality
   
    # Woodwork luck factor
    woodwork_differential = home_match_woodwork - away_match_woodwork

    # Style clash indicator (high possession vs counter-attack)
    home_possession_indicator = (home_match_corners + home_match_shots) / (home_match_fouls + 1)
    away_possession_indicator = (away_match_corners + away_match_shots) / (away_match_fouls + 1)
    style_clash_indicator = abs(home_possession_indicator - away_possession_indicator)
  
    return [
        momentum_difference, combined_momentum, form_points_difference, win_rate_difference, 
        H2H_win_pct_difference, home_attack_vs_away_defense, away_attack_vs_home_defense, 
        goal_scoring_difference, goals_conceded_difference, shooting_efficiency_difference, 
        shots_difference, shots_on_target_difference, clean_sheet_difference, shot_accuracy_difference, 
        conversion_rate_difference, home_defensive_pressure, away_defensive_pressure, 
        shots_on_target_against_difference, corner_difference, corner_against_difference, 
        territory_control_difference, fouls_difference, yellow_card_difference, red_card_difference, 
        offside_difference, halftime_goals_difference, halftime_wins_difference, first_half_strength_difference, 
        bet365_probability_difference, draw_probability_average, market_confidence, H2H_form_alignment, 
        H2H_goals_vs_defense, shot_accuracy_consitency, discipline_consistency, attacking_threat_differential, 
        defensive_solidity_differential, balance_score_differential, form_efficiency_differential, 
        pressure_handling_differential, finishing_quality_differential, woodwork_differential, 
        style_clash_indicator
    ]

def feature_names():
    # H2H features (42)
    h2h_names = [
        'H2H_hometeam_wins', 'H2H_awayteam_wins', 'H2H_draws',
        'H2H_hometeam_win_pct', 'H2H_awayteam_win_pct',
        'H2H_hometeam_halftime_goals', 'H2H_awayteam_halftime_goals',
        'H2H_hometeam_fulltime_goals', 'H2H_awayteam_fulltime_goals',
        'H2H_goal_diff', 'H2H_hometeam_avg_goals', 'H2H_awayteam_avg_goals',
        'H2H_hometeam_goals_conceded', 'H2H_awayteam_goals_conceded',
        'H2H_hometeam_halftime_wins', 'H2H_awayteam_halftime_wins', 'H2H_halftime_draws',
        'H2H_hometeam_shots', 'H2H_awayteam_shots',
        'H2H_hometeam_shots_on_target', 'H2H_awayteam_shots_on_target',
        'H2H_hometeam_shot_accuracy', 'H2H_awayteam_shot_accuracy',
        'H2H_hometeam_conversion', 'H2H_awayteam_conversion',
        'H2H_hometeam_woodwork', 'H2H_awayteam_woodwork',
        'H2H_hometeam_corners', 'H2H_awayteam_corners',
        'H2H_hometeam_fouls', 'H2H_awayteam_fouls',
        'H2H_hometeam_offsides', 'H2H_awayteam_offsides',
        'H2H_hometeam_yellow_card', 'H2H_awayteam_yellow_card',
        'H2H_hometeam_red_card', 'H2H_awayteam_red_card',
        'H2H_bet365_hometeam_probability_home', 'H2H_bet365_hometeam_probability_away',
        'H2H_bet365_awayteam_probability_home', 'H2H_bet365_awayteam_probability_away',
        'H2H_bet365_probability_draws'
    ]
    
    # Home match features (31)
    home_names = [
        'home_match_wins', 'home_win_rate', 'home_match_losses', 'home_match_draws',
        'home_match_halftime_goals', 'home_match_fulltime_goals',
        'home_match_goals_conceded', 'home_match_avg_goals_conceded',
        'home_match_clean_sheets',
        'home_match_halftime_wins', 'home_match_halftime_losses', 'home_match_halftime_draws',
        'home_match_shots', 'home_match_shots_on_target', 'home_shot_accuracy', 'home_conversion_rate',
        'home_match_shots_against', 'home_match_shots_on_target_against', 'home_match_woodwork_against',
        'home_match_woodwork', 'home_match_corners', 'home_match_corners_against',
        'home_match_fouls', 'home_match_offsides',
        'home_match_yellow_card', 'home_match_red_card',
        'home_recent_points', 'home_match_momentum',
        'home_match_bet365_probability_wins', 'home_match_bet365_probability_losses',
        'home_match_bet365_probability_draws'
    ]
    
    # Away match features (31)
    away_names = [
        'away_match_wins', 'away_win_rate', 'away_match_losses', 'away_match_draws',
        'away_match_halftime_goals', 'away_match_fulltime_goals',
        'away_match_goals_conceded', 'away_match_avg_goals_conceded',
        'away_match_clean_sheets',
        'away_match_halftime_wins', 'away_match_halftime_losses', 'away_match_halftime_draws',
        'away_match_shots', 'away_match_shots_on_target', 'away_shot_accuracy', 'away_conversion_rate',
        'away_match_shots_against', 'away_match_shots_on_target_against', 'away_match_woodwork_against',
        'away_match_woodwork', 'away_match_corners', 'away_match_corners_against',
        'away_match_fouls', 'away_match_offsides',
        'away_match_yellow_card', 'away_match_red_card',
        'away_recent_points', 'away_match_momentum',
        'away_match_bet365_probability_wins', 'away_match_bet365_probability_losses',
        'away_match_bet365_probability_draws'
    ]
    
    # Interaction features (43)
    interaction_names = [
    'momentum_difference', 'combined_momentum', 'form_points_difference', 
    'win_rate_difference', 'H2H_win_pct_difference', 
    'home_attack_vs_away_defense', 'away_attack_vs_home_defense', 
    'goal_scoring_difference', 'goals_conceded_difference', 
    'shooting_efficiency_difference', 'shots_difference', 
    'shots_on_target_difference', 'clean_sheet_difference', 
    'shot_accuracy_difference', 'conversion_rate_difference', 
    'home_defensive_pressure', 'away_defensive_pressure', 
    'shots_on_target_against_difference', 'corner_difference', 
    'corner_against_difference', 'territory_control_difference', 
    'fouls_difference', 'yellow_card_difference', 'red_card_difference', 
    'offside_difference', 'halftime_goals_difference', 
    'halftime_wins_difference', 'first_half_strength_difference', 
    'bet365_probability_difference', 'draw_probability_average', 
    'market_confidence', 'H2H_form_alignment', 'H2H_goals_vs_defense', 
    'shot_accuracy_consistency', 'discipline_consistency', 
    'attacking_threat_differential', 'defensive_solidity_differential', 
    'balance_score_differential', 'form_efficiency_differential', 
    'pressure_handling_differential', 'finishing_quality_differential', 
    'woodwork_differential', 'style_clash_indicator'
]
    
    return h2h_names + home_names + away_names + interaction_names


def process_data():
 
    all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
    all_data_sorted = all_data.sort_values('Date').reset_index(drop=True)

    # Team match indices for faster look up
    print("Building team match indices")
    team_matches = {}
    for team in pd.concat([all_data_sorted['HomeTeam'], all_data_sorted['AwayTeam']]).unique():
        home_mask = all_data_sorted['HomeTeam'] == team
        away_mask = all_data_sorted['AwayTeam'] == team
        team_matches[team] = {
            'home_indices' : all_data_sorted[home_mask].index.tolist(),
            'away_indices' : all_data_sorted[away_mask].index.tolist(),
            'all_indices' : all_data_sorted[home_mask | away_mask].index.tolist() 
        }

    # H2H match indices
    print("Building H2H indices")
    h2h_matches = {}
    for idx, match in all_data_sorted.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        pair = tuple(sorted([home_team, away_team]))

        if pair not in h2h_matches:
            h2h_mask = (
                ((all_data_sorted['HomeTeam'] == home_team) & (all_data_sorted['AwayTeam'] == away_team)) |
                ((all_data_sorted['HomeTeam'] == away_team) & (all_data_sorted['AwayTeam'] == home_team))
            )
            h2h_matches[pair] = all_data_sorted[h2h_mask].index.tolist()

    # Training dataset
    X_train = []
    y_train = []

    # idx = the index row number, match is the actual row data and iterrows() loops through each row in the df
    for idx, match in all_data_sorted.iterrows():
        # To show that we are processing the data each 100 matches
        if idx % 100 == 0:
            print(f"Processing match data {idx}/{len(all_data_sorted)}")
        
        # Extracting the match data
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]
        match_date = match["Date"]
        outcome = match["FTR"]

        # Get the features using data before this match
        h2h_stats = head2head_training(home_team, away_team, all_data_sorted, match_date, idx, h2h_matches)
        home_stats = home_matches_training(home_team, all_data_sorted, match_date, idx, team_matches)
        away_stats = away_matches_training(away_team, all_data_sorted, idx, match_date, team_matches)
        feature_interaction = feature_interactions(h2h_stats, home_stats, away_stats)

        # Only include matches with historical data
        if sum(h2h_stats) > 0 or sum(home_stats) > 0 or sum(away_stats) > 0:
            features = h2h_stats + home_stats + away_stats + feature_interaction
            X_train.append(features)
            y_train.append(mapping[outcome])
        
    # Converting to dataframe with column names
    column_names = feature_names()
    X_train = pd.DataFrame(X_train, columns=column_names)
    y_train = pd.Series(y_train, name='outcome')

    print("Training dataset created")
    print("Original class distribution:", Counter(y_train))

    # Split into training and test data 
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train_split, X_test_split, y_train_split, y_test_split

def data_normalization(X_train_split, X_test_split):
    # Normalize data for models 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_test_scaled = scaler.transform(X_test_split)

    return X_train_scaled, X_test_scaled

# fit_transform on the train data to calculate the parameters and immediately apply the transformation
# To avoid data leakage we dont fit the test data but just apply the transformation, otherwise it would recalculate
# Now we ensure that the test data is scaled consistently.

# Training the model
print("Training the model..")

def objective(trial):
    tree_n_estimators = trial.suggest_int("tree_n_estimators", 50, 500) # Number of trees
    tree_max_depth = trial.suggest_int("tree_max_depth", 5, 50) # Maximum depth of each tree
    tree_min_samples_split = trial.suggest_int("tree_min_samples_split", 2, 20) # Minimum number of samples needed to split node
    tree_min_samples_leaf = trial.suggest_int("tree_min_samples_leaf", 1, 5) # Minimum number of samples needed at leaf node

    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200) # Number of boosting rounds
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 7) # Maximum depth of each tree
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2) # Step size shrinkage for update to prevent overfitting
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.7, 1.0) # Fraction of samples used per tree
    xgb_colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0) # Fraction of features used per tree

    cat_iterations = trial.suggest_int("cat__iterations", 100, 500) # Number of boosting iterations (trees)
    cat_depth = trial.suggest_int("cat__depth", 3, 8) # Maximum depth of each tree
    cat_learning_rate = trial.suggest_float("cat__learning_rate", 0.05, 0.1) # Step size for updating the model
    cat_random_state = trial.suggest_int("cat__random_state", 1, 50) # Internal randomness factor


    # Instantiate base models with suggested parameters
    tree_clf = RandomForestClassifier(
        n_estimators = tree_n_estimators,
        max_depth = tree_max_depth,
        min_samples_split = tree_min_samples_split,
        min_samples_leaf = tree_min_samples_leaf,
        random_state = 42,
        class_weight = 'balanced'
    )

    xgb_clf = XGBClassifier(
        n_estimators = xgb_n_estimators,
        max_depth = xgb_max_depth,
        learning_rate = xgb_learning_rate,
        subsample = xgb_subsample,
        colsample_bytree = xgb_colsample_bytree,
    )

    cat_clf = CatBoostClassifier(
        iterations = cat_iterations,
        depth = cat_depth,
        learning_rate = cat_learning_rate,
        random_state = cat_random_state,
        verbose = False,
    )

    base_models = [
        ('tree', tree_clf),
        ('xgb', xgb_clf),
        ('cat', cat_clf)
    ]

    voting_clf = VotingClassifier(
        estimators = base_models,
            voting = 'soft', 
            n_jobs = -1
    )

    score = cross_val_score(voting_clf, X_train_split, y_train_split, cv=5, scoring="accuracy").mean()
    return score

def study(X_train_split, y_train_split, X_test_split, y_test_split):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = study.best_params

    tree_clf = RandomForestClassifier(
        n_estimators=best_params["tree_n_estimators"],
        max_depth=best_params["tree_max_depth"],
        min_samples_split=best_params["tree_min_samples_split"],
        min_samples_leaf=best_params["tree_min_samples_leaf"],
        random_state=42,
        class_weight='balanced'
    )

    xgb_clf = XGBClassifier(
        n_estimators=best_params["xgb_n_estimators"],
        max_depth=best_params["xgb_max_depth"],
        learning_rate=best_params["xgb_learning_rate"],
        subsample=best_params["xgb_subsample"],
        colsample_bytree=best_params["xgb_colsample_bytree"],
    )

    cat_clf = CatBoostClassifier(
        iterations=best_params["cat_iterations"],
        depth=best_params["cat_depth"],
        learning_rate=best_params["cat_learning_rate"],
        random_state=best_params["cat_random_state"],
        verbose= False
    )

    ensemble = VotingClassifier(
        estimators=[
            ('tree', tree_clf),
            ('xgb', xgb_clf),
            ('cat', cat_clf)
        ],
        voting="soft",
        n_jobs=-1
    )

    # Fit ensemble with training data
    ensemble.fit(X_train_split, y_train_split)

    # Predict
    y_pred = ensemble.predict(X_test_split)
    print(f"Test Accuracy: {accuracy_score(y_test_split, y_pred):.4f}")

def prediction_model(y_train_split, X_train_scaled, X_test_scaled):
    # Define base models
    base_models = [
        ('tree', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('xgb', XGBClassifier())
    ] 

    voting_clf = VotingClassifier(
        estimators= base_models,
            voting= 'soft', 
            n_jobs= -1
    )

    param_grid = {
        'tree__n_estimators': randint(50, 500),      # Number of trees
        'tree__max_depth': randint(5, 50),           # Maximum depth of each tree
        'tree__min_samples_split': randint(2, 20),   # Minimum number of samples needed to split node
        'tree__min_samples_leaf': randint(1, 5),     # Minimum number of samples needed at leaf node
        
        'xgb__n_estimators': randint(50, 200),       # Number of boosting rounds
        'xgb__max_depth': randint(3, 7),             # Maximum depth of each tree
        'xgb__learning_rate': uniform(0.01, 0.2),    # Step size shrinkage for update to prevent overfitting
        'xgb__subsample': uniform(0.7, 0.3),         # Fraction of samples used per tree
        'xgb__colsample_bytree': uniform(0.7, 0.3)   # Fraction of features used per tree
    }

    # Grid search for hyperparameter tuning
    grid_search = RandomizedSearchCV(estimator=voting_clf,param_distributions=param_grid, cv=5,
                            scoring="accuracy")

    # Fitting the model
    grid_search.fit(X_train_scaled, y_train_split)

    print("Best parameters found:", grid_search.best_params_)
    # Best parameters found: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 30}
    # Model accuracy: 52.08%

    # Getting the best model from the search
    best_model = grid_search.best_estimator_

    # Predictions on the set
    y_pred = best_model.predict(X_test_scaled)

    return best_model, y_pred

def data_visualization(best_model, y_pred, X_test_scaled, y_test_split):
    # Evaluation
    accuracy = best_model.score(X_test_scaled, y_test_split)
    print(f"Model accuracy: {accuracy:.2%}")

    # Classification
    print("Classification Report")
    print(classification_report(y_test_split, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_split, y_pred)
    print(cm)

    # Vizualization of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home Win", "Draw", "Away Win"],
                yticklabels=["Home Win", "Draw", "Away Win"])
    plt.ylabel("ACTUAL")
    plt.xlabel("PREDICTED")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    X_train_split, X_test_split, y_train_split, y_test_split = process_data()
    X_train_scaled, X_test_scaled = data_normalization(X_train_split, X_test_split)
    study(X_train_split, y_train_split, X_test_split, y_test_split)
    #best_model, y_pred = prediction_model(y_train_split, X_train_scaled, X_test_scaled)
    #data_visualization(best_model, y_pred, X_test_scaled, y_test_split)


# Save the model
#with open('match_predictor.pkl', 'wb') as f:
#    pickle.dump(model, f)

#print(" Model saved as 'match_predictor.pkl'")