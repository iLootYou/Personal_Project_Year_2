import pandas as pd
import numpy as np
import glob
import pickle
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt

import feature_engineering as fe

all_data = fe.data_frame()
X_train, X_test, y_train, y_test = fe.process_data(all_data)
X_train_scaled, X_test_scaled = fe.data_normalization(X_train, X_test)


# fit_transform on the train data to calculate the parameters and immediately apply the transformation
# To avoid data leakage we dont fit the test data but just apply the transformation, otherwise it would recalculate
# Now we ensure that the test data is scaled consistently.

# Training the model
print("Training the model..")

def objective(trial):
    tree_n_estimators = trial.suggest_int("tree_n_estimators", 50, 500) # Number of trees
    tree_max_depth = trial.suggest_int("tree_max_depth", 5, 50) # Maximum depth of each tree
    tree_min_samples_split = trial.suggest_int("tree_min_samples_split", 2, 20) # Minimum number of samples needed to split node
    tree_min_samples_leaf = trial.suggest_int("tree_min_samples_leaf", 1, 5) # Minimum number of samples needed at leaf node

    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200) # Number of boosting rounds
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 7) # Maximum depth of each tree
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2) # Step size shrinkage for update to prevent overfitting
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.7, 1.0) # Fraction of samples used per tree
    xgb_colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0) # Fraction of features used per tree

    cat_iterations = trial.suggest_int("cat__iterations", 100, 500) # Number of boosting iterations (trees)
    cat_depth = trial.suggest_int("cat__depth", 3, 8) # Maximum depth of each tree
    cat_learning_rate = trial.suggest_float("cat__learning_rate", 0.05, 0.1) # Step size for updating the model
    cat_random_state = trial.suggest_int("cat__random_state", 1, 50) # Internal randomness factor


    # Instantiate base models with suggested parameters
    tree_clf = RandomForestClassifier(
        n_estimators = tree_n_estimators,
        max_depth = tree_max_depth,
        min_samples_split = tree_min_samples_split,
        min_samples_leaf = tree_min_samples_leaf,
        random_state = 42,
        class_weight = 'balanced'
    )

    xgb_clf = XGBClassifier(
        n_estimators = xgb_n_estimators,
        max_depth = xgb_max_depth,
        learning_rate = xgb_learning_rate,
        subsample = xgb_subsample,
        colsample_bytree = xgb_colsample_bytree,
    )

    cat_clf = CatBoostClassifier(
        iterations = cat_iterations,
        depth = cat_depth,
        learning_rate = cat_learning_rate,
        random_state = cat_random_state,
        verbose = False,
    )

    base_models = [
        ('tree', tree_clf),
        ('xgb', xgb_clf),
        ('cat', cat_clf)
    ]

    #voting_clf = VotingClassifier(
    #    estimators = base_models,
    #        voting = 'soft', 
    #        n_jobs = -1
    #)

    stacking = StackingClassifier(
        estimators= base_models,
        final_estimator= LogisticRegression(),
        n_jobs = -1
    )

    score = cross_val_score(stacking, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

def study(X_train_selected, y_train_split, X_test_split, y_test_split):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = study.best_params

    tree_clf = RandomForestClassifier(
        n_estimators=best_params["tree_n_estimators"],
        max_depth=best_params["tree_max_depth"],
        min_samples_split=best_params["tree_min_samples_split"],
        min_samples_leaf=best_params["tree_min_samples_leaf"],
        random_state=42,
        class_weight='balanced'
    )

    xgb_clf = XGBClassifier(
        n_estimators=best_params["xgb_n_estimators"],
        max_depth=best_params["xgb_max_depth"],
        learning_rate=best_params["xgb_learning_rate"],
        subsample=best_params["xgb_subsample"],
        colsample_bytree=best_params["xgb_colsample_bytree"],
    )

    cat_clf = CatBoostClassifier(
        iterations=best_params["cat_iterations"],
        depth=best_params["cat_depth"],
        learning_rate=best_params["cat_learning_rate"],
        random_state=best_params["cat_random_state"],
        verbose= False
    )

    ensemble = StackingClassifier(
        estimators=[
            ('tree', tree_clf),
            ('xgb', xgb_clf),
            ('cat', cat_clf)
        ],
        n_jobs=-1
    )

    # Fit ensemble with training data
    ensemble.fit(X_train_selected, y_train_split)

    # Predict
    y_pred = ensemble.predict(X_test_split)
    print(f"Test Accuracy: {accuracy_score(y_test_split, y_pred):.4f}")

def select_top_features(X_train_split, y_train_split, top_k=80):
    # Select most important features to reduce noise

    feature_names_list = fe.feature_names()

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_split, y_train_split)

    importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature' : feature_names_list,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance_df.head(20))

    top_features = feature_importance_df.head(top_k)['feature'].tolist()
    top_feature_indices = [feature_names_list.index(f) for f in top_features]

    X_train_selected = X_train_split.iloc[:, top_feature_indices]
    
    print(f"\nReduced features from {len(feature_names_list)} to {top_k}")
    
    return X_train_selected, top_feature_indices, top_features

def prediction_model(y_train_split, X_train_scaled, X_test_scaled):
    # Define base models
    base_models = [
        ('tree', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('xgb', XGBClassifier())
    ] 

    voting_clf = VotingClassifier(
        estimators= base_models,
            voting= 'soft', 
            n_jobs= -1
    )

    param_grid = {
        'tree__n_estimators': randint(50, 500),      # Number of trees
        'tree__max_depth': randint(5, 50),           # Maximum depth of each tree
        'tree__min_samples_split': randint(2, 20),   # Minimum number of samples needed to split node
        'tree__min_samples_leaf': randint(1, 5),     # Minimum number of samples needed at leaf node
        
        'xgb__n_estimators': randint(50, 200),       # Number of boosting rounds
        'xgb__max_depth': randint(3, 7),             # Maximum depth of each tree
        'xgb__learning_rate': uniform(0.01, 0.2),    # Step size shrinkage for update to prevent overfitting
        'xgb__subsample': uniform(0.7, 0.3),         # Fraction of samples used per tree
        'xgb__colsample_bytree': uniform(0.7, 0.3)   # Fraction of features used per tree
    }

    # Grid search for hyperparameter tuning
    grid_search = RandomizedSearchCV(estimator=voting_clf,param_distributions=param_grid, cv=5,
                            scoring="accuracy")

    # Fitting the model
    grid_search.fit(X_train_scaled, y_train_split)

    print("Best parameters found:", grid_search.best_params_)
    # Best parameters found: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 30}
    # Model accuracy: 52.08%

    # Getting the best model from the search
    best_model = grid_search.best_estimator_

    # Predictions on the set
    y_pred = best_model.predict(X_test_scaled)

    return best_model, y_pred

def data_visualization(best_model, y_pred, X_test_scaled, y_test_split):
    # Evaluation
    accuracy = best_model.score(X_test_scaled, y_test_split)
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


if __name__ == "__main__":
    X_train_selected = select_top_features(X_train, y_train, top_k=80)
    study(X_train_selected, y_train, X_test, y_test)
    #best_model, y_pred = prediction_model(y_train, X_train_scaled, X_test_scaled)
    #data_visualization(best_model, y_pred, X_test_scaled, y_test)


# Save the model
#with open('match_predictor.pkl', 'wb') as f:
#    pickle.dump(model, f)

#print(" Model saved as 'match_predictor.pkl'")