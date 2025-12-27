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

all_data = fe.data_frame()
X_train, X_test, y_train, y_test = fe.process_data(all_data)
X_train_scaled, X_test_scaled = fe.data_normalization(X_train, X_test)

# Training the model
print("Training the model..")


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


def create_objective_function(X_train, y_train, outcome_name):
    """
    Creates an objective function for Optuna to optimize
    outcome_name: 'home', 'draw', or 'away'
    """
    def objective(trial):
        # Suggest hyperparameters for RandomForest
        tree_n_estimators = trial.suggest_int("tree_n_estimators", 50, 500)
        tree_max_depth = trial.suggest_int("tree_max_depth", 5, 50)
        tree_min_samples_split = trial.suggest_int("tree_min_samples_split", 2, 20)
        tree_min_samples_leaf = trial.suggest_int("tree_min_samples_leaf", 1, 5)

        # Suggest hyperparameters for XGBoost
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 7)
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2)
        xgb_subsample = trial.suggest_float("xgb_subsample", 0.7, 1.0)
        xgb_colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0)

        # Suggest hyperparameters for CatBoost
        cat_iterations = trial.suggest_int("cat_iterations", 100, 500)
        cat_depth = trial.suggest_int("cat_depth", 3, 8)
        cat_learning_rate = trial.suggest_float("cat_learning_rate", 0.05, 0.1)
        cat_random_state = trial.suggest_int("cat_random_state", 1, 50) 

        # Moderation factor for the weights
        moderation_factor = trial.suggest_float("moderation_factor", 0.1, 1.0)

        # Create models with suggested parameters
        tree_clf = RandomForestClassifier(
            n_estimators=tree_n_estimators,
            max_depth=tree_max_depth,
            min_samples_split=tree_min_samples_split,
            min_samples_leaf=tree_min_samples_leaf,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        xgb_clf = XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        cat_clf = CatBoostClassifier(
            iterations=cat_iterations,
            depth=cat_depth,
            learning_rate=cat_learning_rate,
            random_state=cat_random_state,
            verbose=False
        )

        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('tree', tree_clf),
                ('xgb', xgb_clf),
                ('cat', cat_clf)
            ],
            voting='soft',
            n_jobs=-1
        )

        # stacking_clf = StackingClassifier(
        #   estimators=[
        #         ('tree', tree_clf),
        #         ('xgb', xgb_clf),
        #         ('cat', cat_clf)
        #     ],
        # final_estimator= LogisticRegression(),
        # n_jobs = -1
        # )

        # Use SMOTE inside a pipeline and stratified CV; optimize with F1 for imbalanced binaries
        from sklearn.model_selection import StratifiedKFold
        sampler = SMOTE(random_state=42)
        pipeline = Pipeline([('sampler', sampler), ('clf', voting_clf)])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = 'f1'  # prioritize F1 for the positive class (draw or wins)
        score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
        
        return score
    
    return objective

def multiclass_model_with_optuna(X_train_input, y_train, X_test_input, y_test, n_trials=50):
    """
    Multi-class Stacking model—use ALL features, optimize f1_macro for draws
    """
    # Use ALL features (no selection) like old.py
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_input, y_train, test_size=0.20, stratify=y_train, random_state=42
    )

    def objective_multiclass(trial):
        tree_n_estimators = trial.suggest_int("tree_n_estimators", 50, 500)
        tree_max_depth = trial.suggest_int("tree_max_depth", 5, 50)
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 7)
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2)
        cat_iterations = trial.suggest_int("cat_iterations", 100, 500)
        cat_depth = trial.suggest_int("cat_depth", 3, 8)
        cat_learning_rate = trial.suggest_float("cat_learning_rate", 0.05, 0.1)

        tree_clf = RandomForestClassifier(
            n_estimators=tree_n_estimators,
            max_depth=tree_max_depth,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        xgb_clf = XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )

        cat_clf = CatBoostClassifier(
            iterations=cat_iterations,
            depth=cat_depth,
            learning_rate=cat_learning_rate,
            random_state=42,
            verbose=False
        )

        # Use Stacking with Logistic Regression final estimator
        stacking_clf = StackingClassifier(
            estimators=[
                ('tree', tree_clf),
                ('xgb', xgb_clf),
                ('cat', cat_clf)
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            n_jobs=-1
        )

        # Use f1_macro to equally weight all classes (penalizes missing draws)
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(stacking_clf, X_train_fit, y_train_fit, cv=cv, scoring='f1_macro', n_jobs=-1).mean()
        return score

    print("=" * 60)
    print("Optimizing Multi-Class Stacking Model (ALL FEATURES, f1_macro)...")
    print("=" * 60)
    
    study = optuna.create_study(direction="maximize", study_name="multiclass_stacking")
    study.optimize(objective_multiclass, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    
    # Build final stacking ensemble
    multiclass_clf = StackingClassifier(
        estimators=[
            ('tree', RandomForestClassifier(
                n_estimators=best_params['tree_n_estimators'],
                max_depth=best_params['tree_max_depth'],
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=best_params['xgb_n_estimators'],
                max_depth=best_params['xgb_max_depth'],
                learning_rate=best_params['xgb_learning_rate'],
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=best_params['cat_iterations'],
                depth=best_params['cat_depth'],
                learning_rate=best_params['cat_learning_rate'],
                random_state=42,
                verbose=False
            ))
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        n_jobs=-1
    )

    multiclass_clf.fit(X_train_fit, y_train_fit)
    predictions = multiclass_clf.predict(X_test_input)
    final_accuracy = accuracy_score(y_test, predictions)

    print(f"\n{'='*60}")
    print("FINAL STACKING MODEL RESULTS (ALL FEATURES)")
    print(f"{'='*60}")
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Home', 'Draw', 'Away']))
    print(f"\nBest Optuna Trial Value: {study.best_value:.4f}")

    return {
        'clf': multiclass_clf,
        'accuracy': final_accuracy,
        'study': study
    }

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
    # Use ALL features (no feature selection)
    print("Multi-Class Stacking (All Features, f1_macro):")
    multiclass_model_with_optuna(X_train, y_train, X_test, y_test, n_trials=50)
    #binary_models(X_train, y_train, X_test, y_test)
    #study(X_train_selected, y_train, X_test, y_test)
    #best_model, y_pred = prediction_model(y_train, X_train_scaled, X_test_scaled)
    #data_visualization(best_model, y_pred, X_test_scaled, y_test)


# Save the model
#with open('match_predictor.pkl', 'wb') as f:
#    pickle.dump(model, f)

#print(" Model saved as 'match_predictor.pkl'")