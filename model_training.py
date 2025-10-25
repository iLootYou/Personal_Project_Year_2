import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        return [0,0,0]

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

    # Half time hometeam goals
    H2H_hometeam_halftime_goals_home = last_5[(last_5['HomeTeam'] == home_team)]['HTHG'].sum()
    H2H_hometeam_halftime_goals_away = last_5[(last_5['AwayTeam'] == home_team)]['HTHG'].sum()

    H2H_hometeam_halftime_goals = H2H_hometeam_halftime_goals_home + H2H_hometeam_halftime_goals_away

    # Half time awayteam goals
    H2H_awayteam_halftime_goals_home = last_5[(last_5['HomeTeam'] == away_team)]['HTAG'].sum()
    H2H_awayteam_halftime_goals_away = last_5[(last_5['AwayTeam'] == away_team)]['HTAG'].sum()

    H2H_awayteam_halftime_goals = H2H_awayteam_halftime_goals_home + H2H_awayteam_halftime_goals_away

    # Half time results
    H2H_hometeam_halftime_wins_home = last_5[(last_5['HomeTeam'] == home_team) & (last_5['HTR'] == 'H')].shape[0]
    H2H_hometeam_halftime_wins_away = last_5[(last_5['AwayTeam'] == home_team) & (last_5['HTR'] == 'A')].shape[0]

    H2H_hometeam_halftime_wins = H2H_hometeam_halftime_wins_home + H2H_hometeam_halftime_wins_away

    H2H_awayteam_halftime_wins_home = last_5[(last_5['HomeTeam'] == away_team) & (last_5['HTR'] == 'H')].shape[0]
    H2H_awayteam_halftime_wins_away = last_5[(last_5['AwayTeam'] == away_team) & (last_5['HTR'] == 'A')].shape[0]

    H2H_awayteam_halftime_wins = H2H_awayteam_halftime_wins_home + H2H_awayteam_halftime_wins_away

    H2H_halftime_draws = (last_5['HTR'] == 'D').sum()


    return [H2H_hometeam_wins, H2H_awayteam_wins, H2H_draws, H2H_hometeam_halftime_goals, 
            H2H_awayteam_halftime_goals, H2H_hometeam_halftime_wins, H2H_awayteam_halftime_wins, H2H_halftime_draws
            ]


def home_matches_training(home_team, all_data, before_date):
    # Append the matches to the array where the home team is in the column HomeTeam
    home_matches = all_data[(all_data["HomeTeam"] == home_team) & (all_data["Date"] < before_date)]

    # Sort by the date and get the last 5 
    last_5 = home_matches.sort_values(by="Date").tail(5)

    if len(last_5) == 0:
        return [0,0,0]

    # Amount of times they win when playing at home
    home_match_wins = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'H')].shape[0]

    # Amount of times they lose when playing at home
    home_match_losses = last_5[(last_5['HomeTeam'] == home_team) & (last_5['FTR'] == 'A')].shape[0]

    # Amount of they draw when playing at home 
    home_match_draws = (last_5['FTR'] == 'D').sum()

    return [home_match_wins, home_match_losses, home_match_draws]

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

# fit_transform on the train data to calculate the parameters and immediately apply the transformation
# To avoid data leakage we dont fit the test data but just apply the transformation, otherwise it would recalculate
# Now we ensure that the test data is scaled consistently.


# Training the model
print("Training the model..")
"""
# RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],    # Number of trees
    'max_depth': [None, 10, 20, 30],   # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples needed to split node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples needed at leaf node
}

# Grid search for hyperparameter tuning
grid_search = RandomizedSearchCV(estimator=model,param_distributions=param_grid, cv=5,
                           scoring="neg_mean_absolute_error")

# Fitting the model
grid_search.fit(X_train_split, y_train_split)

print("Best parameters found:", grid_search.best_params_)
# Best parameters found: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200}

# Getting the best model from the search
best_model = grid_search.best_estimator_

# Predictions on the set
y_pred = best_model.predict(X_test_split)
"""
"""
# Support Vector Machine (SVM)

model = SVC()

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