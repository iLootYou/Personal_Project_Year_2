import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
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
    # Coerce to convert invalid dates to NaT instead of errors, dayfirst because i cant seem to get the format
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data = all_data.dropna(subset=["Date","FTR"])
all_data = all_data.sort_values(by="Date")

print(f"Matches loaded: {len(all_data)}")


def head2head_training(home_team,away_team, all_data, before_date):
    # Filter for h2h matches before a certain date
    h2h_matches = all_data[
        (((all_data["HomeTeam"] == home_team) & (all_data["AwayTeam"] == away_team)) |
         ((all_data["HomeTeam"] == away_team) & (all_data["AwayTeam"] == home_team))) &
        (all_data["Date"] < before_date)]

    # Sort by the date and get the last 5 
    last_5 = h2h_matches.sort_values(by="Date").tail(5)

    if len(last_5) == 0:
        # Return a zeroed list matching the full feature count
        return [0] * 31  # Or however many features you always expect

    # Shape[0] counts rows passing both filters, shape[1] gives columns and shape gives both
    # Home team wins when playing at home
    H2H_hometeam_wins_home = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'H')].shape[0]

    # Home team wins when playing away
    H2H_hometeam_wins_away = last_5[(last_5['AwayTeam'] == home_team) & (last_5['FTR'] == 'A')].shape[0]

    # Total wins of the hometeam
    H2H_hometeam_wins = H2H_hometeam_wins_home + H2H_hometeam_wins_away

    # Away team wins when playing at home
    H2H_awayteam_wins_home = last_5[(last_5['HomeTeam'] == away_team) & (last_5['FTR'] == 'H')].shape[0]

    # Away team wins when playing away
    H2H_awayteam_wins_away = last_5[(last_5['AwayTeam'] == away_team) & (last_5['FTR'] == 'A')].shape[0]

    # Total wins of the awayteam
    H2H_awayteam_wins = H2H_awayteam_wins_home + H2H_awayteam_wins_away

    # Draws count 
    H2H_draws = (last_5['FTR'] == 'D').sum()
    #------------------------------------------------------------------------------------------------------------

    # Half time hometeam goals
    H2H_hometeam_halftime_goals_home = last_5[(last_5['HomeTeam'] == home_team)]['HTHG'].sum()
    H2H_hometeam_halftime_goals_away = last_5[(last_5['AwayTeam'] == home_team)]['HTAG'].sum()

    H2H_hometeam_halftime_goals = H2H_hometeam_halftime_goals_home + H2H_hometeam_halftime_goals_away

    # Half time awayteam goals
    H2H_awayteam_halftime_goals_home = last_5[(last_5['HomeTeam'] == away_team)]['HTHG'].sum()
    H2H_awayteam_halftime_goals_away = last_5[(last_5['AwayTeam'] == away_team)]['HTAG'].sum()

    H2H_awayteam_halftime_goals = H2H_awayteam_halftime_goals_home + H2H_awayteam_halftime_goals_away
    #------------------------------------------------------------------------------------------------------------

    # Full time hometeam goals
    H2H_hometeam_fulltime_goals_home = last_5[(last_5['HomeTeam'] == home_team)]['FTHG'].sum()
    H2H_hometeam_fulltime_goals_away = last_5[(last_5['AwayTeam'] == home_team)]['FTAG'].sum()

    H2H_hometeam_fulltime_goals = H2H_hometeam_fulltime_goals_home + H2H_hometeam_fulltime_goals_away

    # Full time awayteam goals
    H2H_awayteam_fulltime_goals_home = last_5[(last_5['HomeTeam'] == away_team)]['FTHG'].sum()
    H2H_awayteam_fulltime_goals_away = last_5[(last_5['AwayTeam'] == away_team)]['FTAG'].sum()

    H2H_awayteam_fulltime_goals = H2H_awayteam_fulltime_goals_home + H2H_awayteam_fulltime_goals_away
    #------------------------------------------------------------------------------------------------------------

    # Half time results
    H2H_hometeam_halftime_wins_home = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'H')].shape[0]
    H2H_hometeam_halftime_wins_away = last_5[(last_5['AwayTeam'] == home_team) & (last_5['HTR'] == 'A')].shape[0]

    H2H_hometeam_halftime_wins = H2H_hometeam_halftime_wins_home + H2H_hometeam_halftime_wins_away

    H2H_awayteam_halftime_wins_home = last_5[(last_5['HomeTeam'] == away_team) & (last_5['HTR'] == 'H')].shape[0]
    H2H_awayteam_halftime_wins_away = last_5[(last_5['AwayTeam'] == away_team) & (last_5['HTR'] == 'A')].shape[0]

    H2H_awayteam_halftime_wins = H2H_awayteam_halftime_wins_home + H2H_awayteam_halftime_wins_away

    H2H_halftime_draws = (last_5['HTR'] == 'D').sum()
    #------------------------------------------------------------------------------------------------------------

    # Total shots taken hometeam
    H2H_hometeam_shots_home = last_5[(last_5['HomeTeam'] == home_team)]['HS'].sum()
    H2H_hometeam_shots_away = last_5[(last_5['AwayTeam'] == home_team)]['AS'].sum()

    H2H_hometeam_shots = H2H_hometeam_shots_home + H2H_hometeam_shots_away

    # Total shots taken awayteam
    H2H_awayteam_shots_home = last_5[(last_5['HomeTeam'] == away_team)]['HS'].sum()
    H2H_awayteam_shots_away = last_5[(last_5['AwayTeam'] == away_team)]['AS'].sum()

    H2H_awayteam_shots = H2H_awayteam_shots_home + H2H_awayteam_shots_away
    #------------------------------------------------------------------------------------------------------------

    # Total shots on target hometeam
    H2H_hometeam_shots_on_target_home = last_5[(last_5['HomeTeam'] == home_team)]['HST'].sum()
    H2H_hometeam_shots_on_target_away = last_5[(last_5['AwayTeam'] == home_team)]['AST'].sum()

    H2H_hometeam_shots_on_target = H2H_hometeam_shots_on_target_home + H2H_hometeam_shots_on_target_away

    # Total shots on target awayteam
    H2H_awayteam_shots_on_target_home = last_5[(last_5['HomeTeam'] == away_team)]['HST'].sum()
    H2H_awayteam_shots_on_target_away = last_5[(last_5['AwayTeam'] == away_team)]['AST'].sum()

    H2H_awayteam_shots_on_target = H2H_awayteam_shots_on_target_home + H2H_awayteam_shots_on_target_away
    #------------------------------------------------------------------------------------------------------------

    # Total woodwork hits hometeam
    H2H_hometeam_woodwork_home = last_5[(last_5['HomeTeam'] == home_team)]['HHW'].sum()
    H2H_hometeam_woodwork_away = last_5[(last_5['AwayTeam'] == home_team)]['AHW'].sum()

    H2H_hometeam_woodwork = H2H_hometeam_woodwork_home + H2H_hometeam_woodwork_away

    # Total woodwork hits awayteam
    H2H_awayteam_woodwork_home = last_5[(last_5['HomeTeam'] == away_team)]['HHW'].sum()
    H2H_awayteam_woodwork_away = last_5[(last_5['AwayTeam'] == away_team)]['AHW'].sum()

    H2H_awayteam_woodwork = H2H_awayteam_woodwork_home + H2H_awayteam_woodwork_away
    #------------------------------------------------------------------------------------------------------------

    # Total corners for hometeam
    H2H_hometeam_corners_home = last_5[(last_5['HomeTeam'] == home_team)]['HC'].sum()
    H2H_hometeam_corners_away = last_5[(last_5['AwayTeam'] == home_team)]['AC'].sum()

    H2H_hometeam_corners = H2H_hometeam_corners_home + H2H_hometeam_corners_away

    # Total corners for awayteam
    H2H_awayteam_corners_home = last_5[(last_5['HomeTeam'] == away_team)]['HC'].sum()
    H2H_awayteam_corners_away = last_5[(last_5['AwayTeam'] == away_team)]['AC'].sum()

    H2H_awayteam_corners = H2H_awayteam_corners_home + H2H_awayteam_corners_away
    #------------------------------------------------------------------------------------------------------------

    # Total fouls for hometeam
    H2H_hometeam_fouls_home = last_5[(last_5['HomeTeam'] == home_team)]['HF'].sum()
    H2H_hometeam_fouls_away = last_5[(last_5['AwayTeam'] == home_team)]['AF'].sum()

    H2H_hometeam_fouls = H2H_hometeam_fouls_home + H2H_hometeam_fouls_away

    # Total fouls for awayteam
    H2H_awayteam_fouls_home = last_5[(last_5['HomeTeam'] == away_team)]['HF'].sum()
    H2H_awayteam_fouls_away = last_5[(last_5['AwayTeam'] == away_team)]['AF'].sum()

    H2H_awayteam_fouls = H2H_awayteam_fouls_home + H2H_awayteam_fouls_away
    #------------------------------------------------------------------------------------------------------------

    # Total offsides for hometeam
    H2H_hometeam_offside_home = last_5[(last_5['HomeTeam'] == home_team)]['HO'].sum()
    H2H_hometeam_offside_away = last_5[(last_5['AwayTeam'] == home_team)]['AO'].sum()

    H2H_hometeam_offsides = H2H_hometeam_offside_home + H2H_hometeam_offside_away

    # Total offsides for awayteam
    H2H_awayteam_offside_home = last_5[(last_5['HomeTeam'] == away_team)]['HO'].sum()
    H2H_awayteam_offside_away = last_5[(last_5['AwayTeam'] == away_team)]['AO'].sum()

    H2H_awayteam_offsides = H2H_awayteam_offside_home + H2H_awayteam_offside_away
    #------------------------------------------------------------------------------------------------------------

    # Total yellow cards for hometeam
    H2H_hometeam_yellow_card_home = last_5[(last_5['HomeTeam'] == home_team)]['HY'].sum()
    H2H_hometeam_yellow_card_away = last_5[(last_5['AwayTeam'] == home_team)]['AY'].sum()

    H2H_hometeam_yellow_card = H2H_hometeam_yellow_card_home + H2H_hometeam_yellow_card_away

    # Total yellow cards for awayteam
    H2H_awayteam_yellow_card_home = last_5[(last_5['HomeTeam'] == away_team)]['HY'].sum()
    H2H_awayteam_yellow_card_away = last_5[(last_5['AwayTeam'] == away_team)]['AY'].sum()

    H2H_awayteam_yellow_card = H2H_awayteam_yellow_card_home + H2H_awayteam_yellow_card_away  
    #------------------------------------------------------------------------------------------------------------ 

    # Total red cars for hometeam
    H2H_hometeam_red_card_home = last_5[(last_5['HomeTeam'] == home_team)]['HR'].sum()
    H2H_hometeam_red_card_away = last_5[(last_5['AwayTeam'] == home_team)]['AR'].sum()

    H2H_hometeam_red_card = H2H_hometeam_red_card_home + H2H_hometeam_red_card_away 

    # Total red cars for awayteam
    H2H_awayteam_red_card_home = last_5[(last_5['HomeTeam'] == away_team)]['HR'].sum()
    H2H_awayteam_red_card_away = last_5[(last_5['AwayTeam'] == away_team)]['AR'].sum()

    H2H_awayteam_red_card = H2H_awayteam_red_card_home + H2H_awayteam_red_card_away
    #------------------------------------------------------------------------------------------------------------ 

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


    return [H2H_hometeam_wins, H2H_awayteam_wins, H2H_draws, H2H_hometeam_halftime_goals, 
            H2H_awayteam_halftime_goals, H2H_hometeam_fulltime_goals, H2H_awayteam_fulltime_goals,  
            H2H_hometeam_halftime_wins, H2H_awayteam_halftime_wins, H2H_halftime_draws, H2H_hometeam_shots,
            H2H_awayteam_shots, H2H_hometeam_shots_on_target, H2H_awayteam_shots_on_target, H2H_hometeam_woodwork,
            H2H_awayteam_woodwork, H2H_hometeam_corners, H2H_awayteam_corners, H2H_hometeam_fouls, 
            H2H_awayteam_fouls, H2H_hometeam_offsides, H2H_awayteam_offsides, H2H_hometeam_yellow_card,
            H2H_awayteam_yellow_card, H2H_hometeam_red_card, H2H_awayteam_red_card, 
            H2H_bet365_hometeam_probability_home, H2H_bet365_hometeam_probability_away,
            H2H_bet365_awayteam_probability_home, H2H_bet365_awayteam_probability_away,
            H2H_bet365_probability_draws
            ]


def home_matches_training(home_team, all_data, before_date):
    # Append the matches to the array where the home team is in the column HomeTeam
    home_matches = all_data[(all_data["HomeTeam"] == home_team) & (all_data["Date"] < before_date)]

    # Sort by the date and get the last 5 
    last_5 = home_matches.sort_values(by="Date").tail(5)

    if len(last_5) == 0:
        # Return a zeroed list matching the full feature count
        return [0] * 17  # Or however many features you always expect

    # Amount of times they win when playing at home
    home_match_wins = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'H')].shape[0]

    # Amount of times they lose when playing at home
    home_match_losses = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'A')].shape[0]

    # Amount of they draw when playing at home 
    home_match_draws = (last_5['FTR'] == 'D').sum()

    # Amount of half time goals when playing at home
    home_match_halftime_goals = last_5[(last_5['HomeTeam'] == home_team)]['HTHG'].sum()

    # Amount of full time goals when playing at home
    home_match_fulltime_goals = last_5[(last_5['HomeTeam'] == home_team)]['FTHG'].sum()

    # Amount of times they are winning at half time when playing at home
    home_match_halftime_wins = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'H')].shape[0]

    # Amount of times they are losing at half time when playing at home
    home_match_halftime_losses = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'A')].shape[0]

    # Amount of times they are drawing at half time when playing at home
    home_match_halftime_draws = (last_5['HTR'] == 'D').sum()

    # Amount of times they shots they had when playing at home
    home_match_shots = last_5[(last_5['HomeTeam'] == home_team)]['HS'].sum()

    # Amount of times they had shots on target when playing at home
    home_match_shots_on_target = last_5[(last_5['HomeTeam'] == home_team)]['HST'].sum()

    # Amount of times they hit the woodwork when playing at home
    home_match_woodwork = last_5[(last_5['HomeTeam'] == home_team)]['HHW'].sum()

    # Amount of times they had a corner when playing at home
    home_match_corners = last_5[(last_5['HomeTeam'] == home_team)]['HC'].sum()

    # Amount of times they commited a foul when playing at home
    home_match_fouls = last_5[(last_5['HomeTeam'] == home_team)]['HF'].sum()

    # Amount of times they were offside when playing at home
    home_match_offsides = last_5[(last_5['HomeTeam'] == home_team)]['HO'].sum()

    # Amount of times they were given a yellow card when playing at home
    home_match_yellow_card = last_5[(last_5['HomeTeam'] == home_team)]['HY'].sum()

    # Amount of times they were given a red card when playing at home
    home_match_red_card = last_5[(last_5['HomeTeam'] == home_team)]['HR'].sum()

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
    

    return [home_match_wins, home_match_losses, home_match_draws, home_match_halftime_goals, 
            home_match_fulltime_goals,home_match_wins, home_match_halftime_wins, home_match_halftime_losses, 
            home_match_halftime_draws, home_match_shots, home_match_shots_on_target, home_match_woodwork, 
            home_match_corners, home_match_fouls, home_match_offsides, home_match_yellow_card, home_match_red_card,
            home_match_bet365_probability_wins, home_match_bet365_probability_losses, 
            home_match_bet365_probability_draws
            ]

# Training dataset
X_train = []
y_train = []

# idx = the index row number, match is the actual row data and iterrows() loops through each row in the df
for idx, match in all_data.iterrows():
    # To show that we are processing the data each 100 matches
    if idx % 100 == 0:
        print(f"Processing match data {idx}/{len(all_data)}")
    
    # Extracting the match data
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]
    match_date = match["Date"]
    outcome = match["FTR"]
    half_time_result = match["HTR"]
    half_time_hometeam_goals = match["HTHG"]
    half_time_awayteam_goals = match["HTAG"]
    full_time_hometeam_goals = match["FTHG"]
    full_time_awayteam_goals = match["FTAG"]
    hometeam_shots_on_target = match["HST"] 
    awayteam_shots_on_target = match["AST"]
    hometeam_hit_woodwork = match["HHW"]
    awayteam_hit_woodwork = match["AHW"]
    hometeam_corners = match["HC"]
    awayteam_corners = match["AC"]
    hometeam_fouls_committed = match["HF"]
    awayteam_fouls_committed = match["AF"]
    hometeam_offsides = match["HO"]
    awayteam_offsides = match["AO"]
    hometeam_yellow_cards = match["HY"]
    awayteam_yellow_cards = match["AY"]
    hometeam_red_cards = match["HR"]
    awayteam_red_cards = match["AR"]
    bet365_hometeam_win_odds = match['B365H']
    bet365_awayteam_win_odds = match['B365A']
    bet365_draw_odds = match['B365D']

    # Get the features using data before this match
    h2h_stats = head2head_training(home_team, away_team, all_data, match_date)
    home_stats = home_matches_training(home_team, all_data, match_date)

    # Only include matches with historical data
    if sum(h2h_stats) > 0 or sum(home_stats) > 0:
        features = h2h_stats + home_stats
        X_train.append(features)
        y_train.append(mapping[outcome])
    
# Converting to NumPy arrays, because scikit learn requires it 
X_train = np.array(X_train)
y_train = np.array(y_train)

print("Training dataset created")

# Split into training and test data 
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Normalize data for models like SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_test_scaled = scaler.transform(X_test_split)

# Synthetic minority oversampeling technique
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_split, y_train_split)

# fit_transform on the train data to calculate the parameters and immediately apply the transformation
# To avoid data leakage we dont fit the test data but just apply the transformation, otherwise it would recalculate
# Now we ensure that the test data is scaled consistently.


# Training the model
print("Training the model..")

# RandomForest
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': randint(50, 500),      # Number of trees
    'max_depth': randint(5, 50),           # Maximum depth of each tree
    'min_samples_split': randint(2, 20),   # Minimum number of samples needed to split node
    'min_samples_leaf': randint(1, 5)      # Minimum number of samples needed at leaf node
}

# Grid search for hyperparameter tuning
grid_search = RandomizedSearchCV(estimator=model,param_distributions=param_grid, cv=5,
                           scoring="neg_mean_absolute_error")

# Fitting the model
grid_search.fit(X_train_split, y_train_split)

print("Best parameters found:", grid_search.best_params_)
# Best parameters found: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 30}
# Model accuracy: 52.08%

# Getting the best model from the search
best_model = grid_search.best_estimator_

# Predictions on the set
y_pred = best_model.predict(X_test_split)

"""
# Support Vector Machine (SVM)

model = SVC(class_weight='balanced')

# Grid search for hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],         # Control tradeoff between maximizing and minimizing classification error
    'gamma': [0.01, 0.1, 1, 10, 100],     # Determines the influence of single training examples
    'kernel': ['linear', 'rbf']           # Defines function used to transform data into higher dimensions
}

grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5,
                           scoring="accuracy")

# Fitting the model
grid_search.fit(X_train_scaled, y_train_split)

print("Best parameters found:", grid_search.best_params_)

# Getting the best model from the search
best_model = grid_search.best_estimator_

# Predictions on the set
y_pred = best_model.predict(X_test_scaled)
"""
"""
# Gradient boost

model = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [None, 1, 2, 4, 6, 8],     # complexity of each tree
    'learning_rate': [0.01, 0.1, 1]         # shrinkage step               
}

grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5,
                           scoring="accuracy")

# Fitting the model
grid_search.fit(X_train_split, y_train_split)

print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# Predictions on the set
y_pred = best_model.predict(X_test_split)
"""

"""
Accuracy: Overall percentage of correct predictions.
    If accuracy is 50%, the model is right half the time.
Precision: Of all matches predicted as "Home Win", how many were actually Home Wins?
    High precision = few false alarms.
Recall: Of all actual "Home Wins", how many did we predict correctly?
    High recall = we don't miss many.
F1-Score: Balance between precision and recall.
    Good overall measure for each class.
Confusion Matrix: Shows where the model gets confused
"""

# Evaluation
accuracy = best_model.score(X_test_split, y_test_split)
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

# Save the model
#with open('match_predictor.pkl', 'wb') as f:
#    pickle.dump(model, f)

#print(" Model saved as 'match_predictor.pkl'")