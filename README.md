# Football Match Predictor
A Flask web application that predicts Premier League match outcomes using machine learning. Enter a home team and away team to get win/draw/loss probabilities.

## Features
* Premier League predictions using historical match data

* Ensemble ML model (Random Forest + XGBoost + CatBoost)

* Feature engineering with H2H stats, form, league position

* Docker deployment ready

* Responsive UI with live probability display

## Tech Stack
* Backend: Flask, Python 3.11

* ML: scikit-learn, XGBoost, CatBoost, Optuna

* Data: Pandas, historical Premier League matches

* Deployment: Docker

* Frontend: HTML/CSS, responsive design

## Model Accuracy
* Test Accuracy: ~53% 

* Ensemble voting: Combines 3 models for robustness

* Features: 80 top-ranked (H2H, form, league position)

## How to test and use
### Requirements
Make sure you have the following software installed:

* Python 3.11 or higher

* Docker (optional, but recommended)

* Git

### Installation without Docker
1. Clone the Repository

    git clone https://github.com/iLootYou/Personal_Project_Year_2.git
   
    cd Personal_Project_Year_2
   
2. Create a Virtual Environment
   
   python -m venv venv

   #### On Windows:
   
   venv\Scripts\activate

   #### On macOS/Linux:
   
   source venv/bin/activate

3. Install Dependencies
   
   pip install -r requirements.txt

4. Download the Data
The project automatically fetches the most recent Premier League data from the current season 25-26 from football-data.co.uk. This happens when you start the application via the current_season_update() function.
5. Start the Webapp
   
   python app.py
   
The application is now available at: http://localhost:5000

6. Make a Prediction

Open your browser and go to http://localhost:5000
Enter the home team (e.g., "Manchester United")
Enter the away team (e.g., "Liverpool")
Click the predict button
View the predicted result and probabilities


### Installation with Docker
1. Clone the Repository
   
  git clone https://github.com/iLootYou/Personal_Project_Year_2.git
  
  cd Personal_Project_Year_2

2. Build the Docker Image
   
  docker build -t match-predictor .

3. Run the Container
   
   docker run -p 5000:5000 match-predictor

The application is now available at: http://localhost:5000

## Project Structure

    Personal_Project_Year_2/ 
    Webapp
    ├── data/ │ 
        └── Prem00-26.csv # Premier League 
    ├── fe/ │ 
        ├── __init__.py # Empty file used for paths
        └── feature_engineering.py # Feature engineering 
    ├──static/ 
        └── style.css # CSS styling
    ├──templates/ 
        └── index.html # HTML template 
    ├── app.py # Flask applicatie 
    ├── Dockerfile # Docker configuratie 
    ├── match_predictor.pkl # Getraind ensemble 
    ├── requirements.txt # Python dependencies 
    ├── top_features.pkl # Top 80 features
    model_training.py # Model training file
    README.md # The readme file


## Training the Model (Optional)

If you want to retrain the model with your own data:

python train_model.py

This will:

* Load and preprocess the data

* Perform feature engineering

* Train the ensemble model

* Save the model as ensemble_model.pkl

Note: Training can take a few minutes, depending on your hardware.

## API Endpoints

The webapp provides the following endpoint:

#### POST /predict

Predict the outcome of a match.

Request:
    
    {
      "home_team": "Manchester United",
      "away_team": "Liverpool"
    }
    
Response:

    {
      "prediction": "Home Win",
      "probabilities": {
        "home_win": 0.62,
        "draw": 0.18,
        "away_win": 0.20
       }
    }    

## Further Information

* GitHub Repository: iLootYou/Personal_Project_Year_2

* Data Source: Football Data UK

* Trained Model: Ensemble of RandomForest, XGBoost, and CatBoost

* Accuracy: ~53% (see project report for details)

You can also find this information in my portfolio.

## Legal Disclaimer & Gambling Warning
* This tool is for educational/entertainment purposes only.

* Creator takes NO responsibility for any financial losses

* Gamble responsibly - only use money you can afford to lose

* Must be of legal gambling age in your jurisdiction

* Predictions are NOT guarantees - football is unpredictable

Use at your own risk, if you have a gambling problem, seek help immediately.
