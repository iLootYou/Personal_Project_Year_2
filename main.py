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

print("To predict the winning team please enter the home team and away team")

home_team = input("Enter the home team: ").strip().title()
away_team = input("Enter the away team: ").strip().title()


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
    df = pd.read_csv(file_path)

    # .values gives us an array of the column so we can see if the teams are in that column
    home_team_exists = (home_team in df["HomeTeam"].values) or (home_team in df["AwayTeam"].values)
    away_team_exists = (away_team in df["HomeTeam"].values) or (away_team in df["AwayTeam"].values)

    if home_team_exists and away_team_exists:
        # Look the last 5 matches played (win, tie, loss), match history between the two clubs
        head2head()

        # Look at who is playing at home, and if they have been winning or losing their home matches
        home_matches()
    else:
        print("The filled in teams are invalid")


def head2head():
    matches = []

    # The '*' is to get all the data files that end in .csv so we dont have to specify
    for file in glob.glob(f"{folder}/*.csv"):
        # Encoding 'ISO-8859-1' because the files werent in UTF-8 which is the standard for Pandas
        df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1')

        # Append the matches to the array
        matches.append(df[
            ((df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team)) & 
            ((df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team))
            ])
    
    # Merge the dataframes
    all_matches = pd.concat(matches)

    # Coerce to convert invalid dates to NaT instead of errors, dayfirst because i cant seem to get the format
    all_matches["Date"] = pd.to_datetime(all_matches["Date"], dayfirst=True, errors='coerce')

    # Sort by the date and get the last 5 
    last_5 = all_matches.sort_values(by="Date").tail(5)
  
    # So we can print out the columns we need to see
    subset = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']

    print(last_5[subset])


def home_matches():
    matches = []

    # The '*' is to get all the data files that end in .csv so we dont have to specify
    for file in glob.glob(f"{folder}/*.csv"):
        # Encoding 'ISO-8859-1' because the files werent in UTF-8 which is the standard for Pandas
        df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1')

        # Append the matches to the array where the home team is in the column HomeTeam
        matches.append(df[df["HomeTeam"] == home_team])

    # Merge the dataframes
    all_home_matches = pd.concat(matches)

    # Coerce to convert invalid dates to NaT instead of errors, dayfirst because i cant seem to get the format
    all_home_matches["Date"] = pd.to_datetime(all_home_matches["Date"], dayfirst=True, errors='coerce')

    # Sort by the date and get the last 5 
    last_5 = all_home_matches.sort_values(by="Date").tail(5)

        # So we can print out the columns we need to see
    subset = ['Date', 'HomeTeam','AwayTeam','FTR']
        
    print(last_5[subset])


def last_5_matches():
    home_matches = []
    away_matches = []

    # The '*' is to get all the data files that end in .csv so we dont have to specify
    for file in glob.glob(f"{folder}/*.csv"):
        # Encoding 'ISO-8859-1' because the files werent in UTF-8 which is the standard for Pandas
        df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1')

        # Append the matches to the array where the team is in either the home or away column
        home_matches.append(df[(df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team)])
        away_matches.append(df[(df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team)])
    
    # Merge the dataframes
    all_home_matches = pd.concat(home_matches)
    all_away_matches = pd.concat(away_matches)

    # Coerce to convert invalid dates to NaT instead of errors, dayfirst because i cant seem to get the format
    all_home_matches["Date"] = pd.to_datetime(all_home_matches["Date"], dayfirst=True, errors='coerce')
    all_away_matches["Date"] = pd.to_datetime(all_away_matches["Date"], dayfirst=True, errors='coerce')

    # Sort by the date and get the last 5 
    last_5_home = all_home_matches.sort_values(by="Date").tail(5)
    last_5_away = all_away_matches.sort_values(by="Date").tail(5)
    
    # So we can print out the columns we need to see
    subset = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']

    print(last_5_home[subset])
    print()
    print(last_5_away[subset])


def main():
    #current_season_update()
    #last_5_matches()
    outcome()


if __name__ == "__main__":
    main()