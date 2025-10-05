import pandas as pd
import numpy as np
import glob
from main import home_matches, head2head, home_team, away_team
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

folder = "data/"
mapping = {'H':0, 'D':1, 'A': 2}

# Get stats from functions in main.py
h2h_stats = head2head(home_team, away_team)
home_stats = home_matches(home_team)
features = h2h_stats + home_stats


# The '*' is to get all the data files that end in .csv so we dont have to specify
for file in glob.glob(f"{folder}/*.csv"):
    # Encoding 'ISO-8859-1' because the files werent in UTF-8 which is the standard for Pandas
    df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1')

    # Getting the data ready
    X = np.array([features])
    y = df["FTR"].map(mapping)