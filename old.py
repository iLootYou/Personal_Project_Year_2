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


def prediction_model(y_train_split, X_train_scaled, X_test_scaled):
    # Define base models
    base_models = [
        ('tree', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('xgb', XGBClassifier()),
        ('cat_clf', CatBoostClassifier())
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
        'xgb__colsample_bytree': uniform(0.7, 0.3),  # Fraction of features used per tree

        'cat__iterations': randint(100, 500),        # Number of boosting iterations (trees)
        'cat__depth': randint(3, 8),                 # Maximum depth of each tree
        'cat__learning_rate': uniform(0.05, 0.1),    # Step size for updating the model
        'cat__random_state': randint( 1, 50)         # Internal randomness factor
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

# def binary_models(X_train, y_train, X_test, y_test):
#     y_train_home = (y_train == 0).astype(int)
#     y_train_draw = (y_train == 1).astype(int)
#     y_train_away = (y_train == 2).astype(int)

#     y_test_home = (y_test == 0).astype(int)
#     y_test_draw = (y_test == 1).astype(int)
#     y_test_away = (y_test == 2).astype(int)

#     # Define base models (will be cloned for each classifier)
#     base_models = [
#         ('tree', RandomForestClassifier(random_state=42, class_weight='balanced')),
#         ('xgb', XGBClassifier(random_state=42)),
#         ('cat_clf', CatBoostClassifier(verbose=False, random_state=42))
#     ] 

#     param_grid = {
#         'tree__n_estimators': randint(50, 500),
#         'tree__max_depth': randint(5, 50),
#         'tree__min_samples_split': randint(2, 20),
#         'tree__min_samples_leaf': randint(1, 5),
        
#         'xgb__n_estimators': randint(50, 200),
#         'xgb__max_depth': randint(3, 7),
#         'xgb__learning_rate': uniform(0.01, 0.19),  # uniform(low, width) not (low, high)
#         'xgb__subsample': uniform(0.7, 0.3),
#         'xgb__colsample_bytree': uniform(0.7, 0.3),

#         'cat_clf__iterations': randint(100, 500),  # Fixed: cat_clf not cat
#         'cat_clf__depth': randint(3, 8),
#         'cat_clf__learning_rate': uniform(0.05, 0.05),  # uniform(0.05, 0.05) = range 0.05-0.10
#         'cat_clf__random_state': randint(1, 50)
#     }

#     # Train Home Win classifier
#     print("Training Home Win classifier...")
#     home_clf = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
#     home_clf_grid_search = RandomizedSearchCV(
#         estimator=home_clf, 
#         param_distributions=param_grid, 
#         cv=5,
#         scoring="accuracy",
#         n_iter=30,  # Add this!
#         random_state=42,
#         verbose=1
#     )
#     home_clf_grid_search.fit(X_train, y_train_home)
#     home_pred = home_clf_grid_search.predict(X_test)
#     home_acc = accuracy_score(y_test_home, home_pred)
#     print(f"  Home Win Binary Accuracy: {home_acc:.4f}")
    
#     # Train Draw classifier
#     print("\nTraining Draw classifier...")
#     draw_clf = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
#     draw_clf_grid_search = RandomizedSearchCV(
#         estimator=draw_clf,
#         param_distributions=param_grid, 
#         cv=5,
#         scoring="accuracy",
#         n_iter=30,
#         random_state=42,
#         verbose=1
#     )
#     draw_clf_grid_search.fit(X_train, y_train_draw)
#     draw_pred = draw_clf_grid_search.predict(X_test)
#     draw_acc = accuracy_score(y_test_draw, draw_pred)
#     print(f"  Draw Binary Accuracy: {draw_acc:.4f}")
    
#     # Train Away Win classifier
#     print("\nTraining Away Win classifier...")
#     away_clf = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
#     away_clf_grid_search = RandomizedSearchCV(
#         estimator=away_clf,
#         param_distributions=param_grid, 
#         cv=5,
#         scoring="accuracy",
#         n_iter=30,
#         random_state=42,
#         verbose=1
#     )
#     away_clf_grid_search.fit(X_train, y_train_away)
#     away_pred = away_clf_grid_search.predict(X_test)
#     away_acc = accuracy_score(y_test_away, away_pred)
#     print(f"  Away Win Binary Accuracy: {away_acc:.4f}")
    
#     # Get probabilities
#     home_probs = home_clf_grid_search.predict_proba(X_test)[:, 1]
#     draw_probs = draw_clf_grid_search.predict_proba(X_test)[:, 1]
#     away_probs = away_clf_grid_search.predict_proba(X_test)[:, 1]

#     # Normalize probabilities
#     total_probs = home_probs + draw_probs + away_probs
#     home_probs_norm = home_probs / total_probs 
#     draw_probs_norm = draw_probs / total_probs
#     away_probs_norm = away_probs / total_probs

#     # Make final predictions
#     predictions = []
#     for h, d, a in zip(home_probs_norm, draw_probs_norm, away_probs_norm):
#         if h > d and h > a:
#             predictions.append(0)
#         elif d > h and d > a:
#             predictions.append(1)
#         else:
#             predictions.append(2)

#     predictions = np.array(predictions)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"\n=== Binary Cascade Model Performance ===")
#     print(f"Final 3-Way Accuracy: {accuracy:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, predictions, target_names=['Home', 'Draw', 'Away']))

#     # Show probability distributions
#     print(f"\nAverage Probabilities:")
#     print(f"  Home: {home_probs_norm.mean():.3f}")
#     print(f"  Draw: {draw_probs_norm.mean():.3f}")
#     print(f"  Away: {away_probs_norm.mean():.3f}")
    
#     return {
#         'home_clf': home_clf_grid_search,
#         'draw_clf': draw_clf_grid_search,
#         'away_clf': away_clf_grid_search,
#         'accuracy': accuracy,
#         'binary_accuracies': {
#             'home': home_acc,
#             'draw': draw_acc,
#             'away': away_acc
#         }
#     }