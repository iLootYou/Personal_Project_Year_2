import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

folder = "data/"
mapping = {'H':0, 'D':1, 'A': 2}

def data_frame():
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
    return all_data


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
        return[0] * 32

    # Get home match indices for team
    home_indices = team_matches[home_team]['home_indices']
    # Filter for indices before current match and before date
    valid_indices = [i for i in home_indices if i < current_idx and all_data_sorted.loc[i,'Date'] < match_date]
    
    if not valid_indices:
        return[0] * 32
    
    # Get last 5 matches
    last_5_indices = valid_indices[-5:] if len(valid_indices) >= 5 else valid_indices
    last_5 = all_data_sorted.loc[last_5_indices]

    num_matches = len(last_5)

    # Match results
    home_match_wins = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'H')].shape[0]
    home_win_rate = home_match_wins / num_matches
    home_match_losses = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'A')].shape[0]
    home_match_draws = (last_5['FTR'] == 'D').sum()

    # Goals scored
    home_match_halftime_goals = last_5[(last_5['HomeTeam'] == home_team)]['HTHG'].sum()
    home_match_fulltime_goals = last_5[(last_5['HomeTeam'] == home_team)]['FTHG'].sum()

    home_match_goals_per_match = home_match_fulltime_goals / num_matches

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
        home_match_halftime_goals, home_match_fulltime_goals, home_match_goals_per_match,
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
        return[0] * 32

    # Get home match indices for team
    away_indices = team_matches[away_team]['away_indices']

    dates = all_data_sorted['Date']
    # Filter for indices before current match and before date
    valid_indices = [i for i in away_indices if i < current_idx and all_data_sorted.loc[i,'Date'] < match_date]
    
    if not valid_indices:
        return[0] * 32
    
    # Get last 5 matches
    last_5_indices = valid_indices[-5:] if len(valid_indices) >= 5 else valid_indices
    last_5 = all_data_sorted.loc[last_5_indices]

    num_matches = len(last_5)

    # Match results
    away_match_wins = last_5[(last_5['AwayTeam'] == away_team) & (last_5['FTR'] == 'A')].shape[0]
    away_win_rate = away_match_wins / num_matches
    away_match_losses = last_5[(last_5['AwayTeam'] == away_team) & (last_5['FTR'] == 'H')].shape[0]
    away_match_draws = (last_5['FTR'] == 'D').sum()

    # Goals scored
    away_match_halftime_goals = last_5[(last_5['AwayTeam'] == away_team)]['HTAG'].sum()
    away_match_fulltime_goals = last_5[(last_5['AwayTeam'] == away_team)]['FTAG'].sum()

    away_match_goals_per_match = away_match_fulltime_goals / num_matches

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
        away_match_halftime_goals, away_match_fulltime_goals, away_match_goals_per_match,
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
    
    # Unpack home_stats (32 features)
    (home_match_wins, home_win_rate, home_match_losses, home_match_draws, 
     home_match_halftime_goals, home_match_fulltime_goals, home_match_goals_per_match,
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
    
    # Unpack away_stats (32 features)
    (away_match_wins, away_win_rate, away_match_losses, away_match_draws, 
     away_match_halftime_goals, away_match_fulltime_goals, away_match_goals_per_match,
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

def league_position(team, all_data_sorted, match_date, current_idx, season_start_month=8):
    current_year = match_date.year
    current_month = match_date.month

    if current_month >= season_start_month:
        season_year = current_year
    else:
        season_year = current_year - 1

    season_start = pd.Timestamp(year=season_year, month=season_start_month, day=1)

    # Get all matches for this team in the current season before this match
    team_season_matches = all_data_sorted[
        ((all_data_sorted['HomeTeam'] == team) | (all_data_sorted['AwayTeam'] == team)) &
        (all_data_sorted['Date'] >= season_start) &
        (all_data_sorted['Date'] < match_date) &
        (all_data_sorted.index < current_idx)
    ]

    if len(team_season_matches) == 0:
        return [0] * 10

    points = 0
    goals_for = 0
    goals_against = 0

    for _, match in team_season_matches.iterrows():
        if match['HomeTeam'] == team:
            goals_for += match['FTHG']
            goals_against += match['FTAG']
            if match['FTR'] == 'H':
                points += 3
            elif match['FTR'] == 'D':
                points += 1
        else:
            goals_for += match['FTAG']
            goals_against += match['FTHG']
            if match['FTR'] == 'A':
                points += 3
            elif match['FTR'] == 'D':
                points += 1
    
    matches_played = len(team_season_matches)

    points_per_game = points / matches_played if matches_played > 0 else 0
    goal_difference = goals_for - goals_against
    goal_difference_per_game = goal_difference / matches_played if matches_played > 0 else 0
    goals_for_per_game = goals_for / matches_played if matches_played > 0 else 0
    goals_against_per_game = goals_against / matches_played if matches_played > 0 else 0

    recent_5 = team_season_matches.tail(5)
    recent_points = 0

    for _, match in recent_5.iterrows():
        if match['HomeTeam'] == team:
            if match['FTR'] == 'H':
                recent_points += 3
            elif match['FTR'] == 'D':
                recent_points += 1
        else:
            if match['FTR'] == 'A':
                recent_points += 3
            elif match['FTR'] == 'D':
                recent_points += 1

    recent_points_per_game = recent_points / len(recent_5) if len(recent_5) > 0 else 0 

    form_trajectory = recent_points_per_game - points_per_game 

    wins = 0 

    for _, match in team_season_matches.iterrows():
        if ((match['HomeTeam'] == team and match['FTR'] == 'H') or 
           (match['AwayTeam'] == team and match['FTR'] == 'A')):
            wins += 1
    
    win_percentage = wins / matches_played if matches_played > 0 else 0

    estimated_position_score = points_per_game / 2.5 

    return [
        points,
        points_per_game,
        goal_difference,
        goal_difference_per_game,
        goals_for_per_game,
        goals_against_per_game,
        recent_points_per_game,
        form_trajectory,
        win_percentage,
        estimated_position_score
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
    
    # Home match features (32)
    home_names = [
        'home_match_wins', 'home_win_rate', 'home_match_losses', 'home_match_draws',
        'home_match_halftime_goals', 'home_match_fulltime_goals', 'home_match_goals_per_match',
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
    
    # Away match features (32)
    away_names = [
        'away_match_wins', 'away_win_rate', 'away_match_losses', 'away_match_draws',
        'away_match_halftime_goals', 'away_match_fulltime_goals', 'away_match_goals_per_match',
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

    # League features (10)
    league_position_names = [
        'points',
        'points_per_game',
        'goal_difference',
        'goal_difference_per_game',
        'goals_for_per_game',
        'goals_against_per_game',
        'recent_points_per_game',
        'form_trajectory',
        'win_percentage',
        'estimated_position_score'
    ]
    
    return h2h_names + home_names + away_names + interaction_names + league_position_names


def process_data(all_data):
 
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
        league_stats = league_position(team, all_data_sorted, match_date, idx, season_start_month=8)

        # Only include matches with historical data
        if sum(h2h_stats) > 0 or sum(home_stats) > 0 or sum(away_stats) > 0:
            features = h2h_stats + home_stats + away_stats + feature_interaction + league_stats
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