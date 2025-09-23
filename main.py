import requests 
import os
import pandas as pd
import glob

url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
response = requests.get(url)
folder = "data/"
file = "Prem25-26.csv"

# Full path to where the file will be saved
file_path = os.path.join(folder, file)

def current_season_update():
    # "wb" for write mode and binary mode so we can write raw bytes which is essential when downloading
    # from web request
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Download successful to {file_path}")
    else:
        print("Failed to download file")

def outcome():
    print("To predict the winning team please enter the home team and away team")

    home_team = input("Enter the home team: ").strip().title()
    away_team = input("Enter the away team: ").strip().title()

    df = pd.read_csv(file_path)

    # .values gives us an array of the column so we can see if the teams are in that column
    home_team_exists = (home_team in df["HomeTeam"].values) or (home_team in df["AwayTeam"].values)
    away_team_exists = (away_team in df["HomeTeam"].values) or (away_team in df["AwayTeam"].values)

    if home_team_exists and away_team_exists:
        print("yes")
        # We have to look the last 5 matches played (win, tie, loss), match history between the two clubs
        # Look at who is playing at home, and if they have been winning or losing their home matches
    else:
        print("The filled in teams are invalid")

def last_5_matches():
    print("")

def main():
    #current_season_update()
    last_5_matches()

if __name__ == "__main__":
    main()