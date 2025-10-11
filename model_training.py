import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
        ((all_data["HomeTeam"] == home_team) | (all_data["AwayTeam"] == home_team)) & 
        ((all_data["HomeTeam"] == away_team) | (all_data["AwayTeam"] == away_team))
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

    return [H2H_hometeam_wins, H2H_awayteam_wins, H2H_draws]


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

