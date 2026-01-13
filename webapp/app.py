from flask import Flask, render_template, request
import pickle
import pandas as pd
import sys
import glob
from pathlib import Path

# Project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all feature engineering functions
import fe.feature_engineering as fe  # Module with data_frame, process_data, etc.


app = Flask(__name__)

# Load the model at startup
print("Loading model and top features")
with open('match_predictor.pkl', 'rb') as f:
    model = pickle.load(f)
# Load top features
with open('top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

# Load ALL historical data
print("Loading historical data for feature engineering...")
all_data = fe.data_frame()  # Same as your training!
print(f"Loaded {len(all_data)} historical matches")

app.jinja_env.globals['all_data'] = all_data # Make it available to functions

def build_features_for_match(hometeam, awayteam):
    # Get the last date in your dataset as "current date"
    latest_date = all_data['Date'].max()

    # Find indices for this hypothetical future match
    mock_idx = len(all_data)

    print(f"Predicting {hometeam} vs {awayteam} on {latest_date}")

    # Build team match indices (same as process_data)
    team_matches = {}
    all_teams = pd.concat([all_data['HomeTeam'], all_data['AwayTeam']]).unique()
    for team in all_teams:
        home_mask = all_data['HomeTeam'] == team
        away_mask = all_data['AwayTeam'] == team
        team_matches[team] = {
            'home_indices': all_data[home_mask].index.tolist(),
            'away_indices': all_data[away_mask].index.tolist(),
            'all_indices': all_data[home_mask | away_mask].index.tolist()
        }
    
    # Build H2H indices
    h2h_matches = {}
    for _, match in all_data.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        pair = tuple(sorted([home_team, away_team]))
        if pair not in h2h_matches:
            h2h_mask = (
                ((all_data['HomeTeam'] == home_team) & (all_data['AwayTeam'] == away_team)) |
                ((all_data['HomeTeam'] == away_team) & (all_data['AwayTeam'] == home_team))
            )
            h2h_matches[pair] = all_data[h2h_mask].index.tolist()
    
    # Generate features EXACTLY like training (using data BEFORE this match)
    h2h_stats = fe.head2head_training(hometeam, awayteam, all_data, latest_date, mock_idx, h2h_matches)
    home_stats = fe.home_matches_training(hometeam, all_data, latest_date, mock_idx, team_matches)
    away_stats = fe.away_matches_training(awayteam, all_data, mock_idx, latest_date, team_matches)
    feature_interaction = fe.feature_interactions(h2h_stats, home_stats, away_stats)
    
    # League stats - fix 'team' variable (you had undefined 'team')
    league_stats = fe.league_position(hometeam, all_data, latest_date, mock_idx)  # Use hometeam
    
    # Combine ALL features
    all_features = h2h_stats + home_stats + away_stats + feature_interaction + league_stats

    # Create dataframe with all column names
    column_names = fe.feature_names()
    X_full = pd.DataFrame([all_features], columns=column_names)

    # Select only top features
    X_selected = X_full[top_features].fillna(0)

    print(f"Feature shape: {X_selected.shape} (expected: 1x{len(top_features)})")

    return X_selected

def H2H(hometeam, awayteam):
    """Get head-to-head stats between two teams"""
    
    # Filter matches where these two teams played each other
    h2h_matches = all_data[
        ((all_data['HomeTeam'] == hometeam) & (all_data['AwayTeam'] == awayteam)) |
        ((all_data['HomeTeam'] == awayteam) & (all_data['AwayTeam'] == hometeam))
    ].copy()
    
    # Return empty if no matches found
    if len(h2h_matches) == 0:
        return pd.DataFrame(), [0, 0, 0]
    
    # Convert dates properly
    h2h_matches["Date"] = pd.to_datetime(h2h_matches["Date"], dayfirst=True, errors='coerce')
    
    # Sort by date and get the last 5 matches
    last_5 = h2h_matches.sort_values(by="Date", ascending=False).head(5).sort_values(by="Date")
    
    # Select columns to display
    subset = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
    display_table = last_5[subset]
    display_table = display_table.rename(columns={'FTR': 'Outcome'})

    outcome_map = {'H': 'Home', 'D': 'Draw', 'A': 'Away'}
    display_table['Outcome'] = display_table['Outcome'].map(outcome_map)
    
    # Calculate stats
    H2H_hometeam_wins_home = last_5[(last_5['HomeTeam'] == hometeam) & (last_5['FTR'] == 'H')].shape[0]
    H2H_hometeam_wins_away = last_5[(last_5['AwayTeam'] == hometeam) & (last_5['FTR'] == 'A')].shape[0]
    H2H_hometeam_wins = H2H_hometeam_wins_home + H2H_hometeam_wins_away
    
    H2H_awayteam_wins_home = last_5[(last_5['HomeTeam'] == awayteam) & (last_5['FTR'] == 'H')].shape[0]
    H2H_awayteam_wins_away = last_5[(last_5['AwayTeam'] == awayteam) & (last_5['FTR'] == 'A')].shape[0]
    H2H_awayteam_wins = H2H_awayteam_wins_home + H2H_awayteam_wins_away
    
    H2H_draws = (last_5['FTR'] == 'D').sum()
    
    stats_text = f"{hometeam}: {H2H_hometeam_wins} wins | Draws: {H2H_draws} | {awayteam}: {H2H_awayteam_wins} wins"
    
    return display_table, stats_text

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    hometeam = request.form.get('hometeam', '').strip().title()
    awayteam = request.form.get('awayteam', '').strip().title()

    if not hometeam or not awayteam:
        return render_template('index.html', error="Please enter both teams"), 400
    
    if hometeam == awayteam:
        return render_template('index.html', error="Teams must be different"), 400
    
    if (hometeam not in all_data['HomeTeam'].values and hometeam not in all_data['AwayTeam'].values) or \
    (awayteam not in all_data['HomeTeam'].values and awayteam not in all_data['AwayTeam'].values):
        return render_template('index.html', error="One or both teams not found in database"), 400

    try:
        X = build_features_for_match(hometeam, awayteam)
        pred_class = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]

        label_map = {0: "Home win", 1: "Draw", 2: "Away win"}
        prediction = label_map[pred_class]
        css_class = {0: "home-win", 1: "draw", 2: "away-win"}[pred_class]

        # Get H2H data
        h2h_table, h2h_stats = H2H(hometeam, awayteam)

        return render_template('index.html',
                               prediction=prediction,
                               css_class=css_class,
                               hometeam=hometeam,
                               awayteam=awayteam,
                               proba_home=f"{pred_proba[0]:.1%}",
                               proba_draw=f"{pred_proba[1]:.1%}",
                               proba_away=f"{pred_proba[2]:.1%}",
                               h2h_table=h2h_table.to_html(classes='table table-striped', index=False),
                               h2h_stats=h2h_stats
        )
    
    except Exception as e:
        return render_template('index.html', error=str(e))
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
