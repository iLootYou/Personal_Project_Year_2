import requests 
import os

url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
response = requests.get(url)
folder = "data/"
file = "Prem25-26.csv"

# Full path to where the file will be saved
file_path = os.path.join(folder, file)

if response.status_code == 200:
    with open(file_path, "wb") as f:
        f.write(response.content)
    print("Download successful")
else:
    print("Failed to download file")
