import requests 

url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
response = requests.get(url)

if response.status_code == 200:
    with open("Prem25-26.csv", "wb") as f:
        f.write(response.content)
    print("Download successful")
else:
    print("Failed to download file")