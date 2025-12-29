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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import feature_engineering as fe

# Training the model
print("Training the model..")


def select_top_features(X_train_split, y_train_split, X_test_split, top_k=80):
    """Select top k features based on training data and apply to both train and test"""
    
    feature_names_list = list(X_train_split.columns)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_split, y_train_split)

    importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names_list,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance_df.head(20))

    # Get top k features
    top_features = feature_importance_df.head(top_k)['feature'].tolist()
   
    # Apply same features to both train and test
    X_train_selected = X_train_split[top_features]
    X_test_selected = X_test_split[top_features]
   
    print(f"\nReduced features from {len(feature_names_list)} to {top_k}")
   
    return X_train_selected, X_test_selected, top_features


def objective(trial, X_train_selected, y_train_split):
    """Objective function for Optuna optimization"""
    tree_n_estimators = trial.suggest_int("tree_n_estimators", 50, 500)
    tree_max_depth = trial.suggest_int("tree_max_depth", 5, 50)
    tree_min_samples_split = trial.suggest_int("tree_min_samples_split", 2, 20)
    tree_min_samples_leaf = trial.suggest_int("tree_min_samples_leaf", 1, 5)

    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 7)
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.7, 1.0)
    xgb_colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0)

    cat_iterations = trial.suggest_int("cat_iterations", 100, 500)
    cat_depth = trial.suggest_int("cat_depth", 3, 8)
    cat_learning_rate = trial.suggest_float("cat_learning_rate", 0.05, 0.1)

    tree_clf = RandomForestClassifier(
        n_estimators=tree_n_estimators,
        max_depth=tree_max_depth,
        min_samples_split=tree_min_samples_split,
        min_samples_leaf=tree_min_samples_leaf,
        random_state=42,
        class_weight='balanced'
    )

    xgb_clf = XGBClassifier(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        random_state=42,
        verbosity=0,
        objective='multi:softprob'  # Multi-class classification
    )

    cat_clf = CatBoostClassifier(
        iterations=cat_iterations,
        depth=cat_depth,
        learning_rate=cat_learning_rate,
        random_state=42,
        verbose=False,
        loss_function='MultiClass'  # CRITICAL: Specify multi-class loss
    )

    base_models = [
        ('tree', tree_clf),
        ('xgb', xgb_clf),
        ('cat', cat_clf)
    ]

    voting_clf = VotingClassifier(
        estimators=base_models,
        voting='soft',
        n_jobs=-1
    )

    # Use cross-validation on training data to evaluate
    score = cross_val_score(voting_clf, X_train_selected, y_train_split, cv=5, scoring="accuracy").mean()
    return score


def optimize_ensemble(X_train_selected, y_train_split, X_test_selected, y_test_split):
    """Run Optuna optimization and train final model"""
    
    # Create study with lambda to pass data to objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train_selected, y_train_split),
        n_trials=50
    )

    print("\n" + "="*50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Cross-Val Accuracy: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50 + "\n")

    best_params = study.best_params

    # Recreate best model with best parameters
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
        random_state=42,
        verbosity=0,
        objective='multi:softprob'  # Multi-class classification
    )

    cat_clf = CatBoostClassifier(
        iterations=best_params["cat_iterations"],
        depth=best_params["cat_depth"],
        learning_rate=best_params["cat_learning_rate"],
        random_state=42,
        verbose=False,
        loss_function='MultiClass'  # CRITICAL: Specify multi-class loss
    )

    ensemble = VotingClassifier(
        estimators=[
            ('tree', tree_clf),
            ('xgb', xgb_clf),
            ('cat', cat_clf)
        ],
        voting='soft',
        n_jobs=-1
    )

    # Fit ensemble with training data
    ensemble.fit(X_train_selected, y_train_split)

    # Predict on test data
    y_pred = ensemble.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test_split, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    return ensemble, study


def data_visualization(best_model, y_pred, X_test_selected, y_test_split):
    """Evaluation and visualization of model performance"""
    accuracy = best_model.score(X_test_selected, y_test_split)
    print(f"Model accuracy: {accuracy:.2%}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_split, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_split, y_pred)
    print(cm)

    # Visualization of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home Win", "Draw", "Away Win"],
                yticklabels=["Home Win", "Draw", "Away Win"])
    plt.ylabel("ACTUAL")
    plt.xlabel("PREDICTED")
    plt.title("Confusion Matrix")
    plt.show()


# Usage:
if __name__ == "__main__":
    # Process your data (your existing function)
    all_data = fe.data_frame()
    X_train_split, X_test_split, y_train_split, y_test_split = fe.process_data(all_data)
    #X_train_scaled, X_test_scaled = fe.data_normalization(X_train_split, X_test_split)
    
    print(f"Train set shape: {X_train_split.shape}")
    print(f"Test set shape: {X_test_split.shape}")
    print(f"Unique classes in y_train: {y_train_split.unique()}")
    
    # Select top features from training data, apply to both
    X_train_selected, X_test_selected, top_features = select_top_features(
        X_train_split, y_train_split, X_test_split, top_k=80
    )

    # Save top features
    with open('top_features.pkl', 'wb') as f:
        pickle.dump(top_features, f)

    print("Top features saved as 'top_features.pkl'")
    
    print(f"Selected train shape: {X_train_selected.shape}")
    print(f"Selected test shape: {X_test_selected.shape}")
    
    #Run optimization
    best_model, optuna_study = optimize_ensemble(
        X_train_selected, y_train_split, X_test_selected, y_test_split
    )
    
    #Get predictions for visualization
    y_pred = best_model.predict(X_test_selected)
    
    #Visualize results
    data_visualization(best_model, y_pred, X_test_selected, y_test_split)
    
    #Save the model
    with open('match_predictor.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("Model saved as 'match_predictor.pkl'")