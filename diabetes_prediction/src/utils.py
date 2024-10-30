import os
import sys
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_model = None
        best_score = 0
        selected_features = {}


        # Iterate through each model in the dictionary
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            params = param[model_name]

            # Check if model supports RFE (i.e., if it has 'coef_' or 'feature_importances_' attribute)
            if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
                rfe = RFE(estimator=model, n_features_to_select=15)
                X_train_selected = rfe.fit_transform(X_train, y_train)
                X_test_selected = rfe.transform(X_test)
            else:
                X_train_selected = X_train
                X_test_selected = X_test
                print(f"Skipping RFE for {model_name} as it does not support feature importance.")

            selected_features[model_name] = X_train_selected

            # Perform GridSearchCV to find the best hyperparameters
            gs = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train_selected, y_train)

            # Update the model with the best found parameters and fit it
            best_model_for_current = gs.best_estimator_
            best_model_for_current.fit(X_train_selected, y_train)

            # Predict on test data and calculate accuracy
            y_test_pred = best_model_for_current.predict(X_test_selected)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Store the test accuracy in the report dictionary
            report[model_name] = test_model_score

            # Update the best model if the current model is better
            if test_model_score > best_score:
                best_score = test_model_score
                best_model = best_model_for_current

            print(f"Model: {model_name}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Test Accuracy: {test_model_score}\n")
            
        # Ensemble model using VotingClassifier
        voting_clf = VotingClassifier(estimators=[
            #('lr', LogisticRegression(max_iter=1000)),
            ('best_model', best_model),
            #('gb', GradientBoostingClassifier())
        ], voting='soft')

        # Fit the voting classifier on the resampled training data
        voting_clf.fit(X_train, y_train)

        # Make predictions on the test data using the ensemble model
        y_pred_ensemble = voting_clf.predict(X_test)

        # Calculate and log the accuracy of the ensemble model
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        report['Voting Ensemble'] = ensemble_accuracy

        print(f"Ensemble Model Test Accuracy: {ensemble_accuracy}\n")

        return report, voting_clf, selected_features

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
