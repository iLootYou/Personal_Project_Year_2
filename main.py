import requests 
import os
import pandas as pd

url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
response = requests.get(url)
folder = "data/"
file = "Prem25-26.csv"

# Full path to where the file will be saved
file_path = os.path.join(folder, file)

if response.status_code == 200:
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"Download successful to {file_path}")
else:
    print("Failed to download file")

print("To predict the winning team please enter the home team and away team")
home_team = input("Enter the home team: ").strip().title()
away_team = input("Enter the away team: ").strip().title()

df = pd.read_csv(file_path)

home_away_condition = (df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)
if home_away_condition.any():
    print("yes")
else:
    print("The filled in teams have yet to play eachother or are invalid")
