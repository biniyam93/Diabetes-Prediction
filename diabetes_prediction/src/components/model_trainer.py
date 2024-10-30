import os
import sys
from dataclasses import dataclass
from sklearn.metrics import roc_curve, RocCurveDisplay

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

warnings.filterwarnings("ignore", category=ConvergenceWarning)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Log the shapes of the arrays
            logging.info(f"Shape of train_array: {train_array.shape}")
            logging.info(f"Shape of test_array: {test_array.shape}")

            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
                'Extra Trees': ExtraTreesClassifier(random_state=42),
                #'SVC': SVC(probability=True, random_state=42),
                'KNeighbors': KNeighborsClassifier(),
                'GaussianNB': GaussianNB(),
                'MLP': MLPClassifier(random_state=42, max_iter=1000),
            }

            param_grids = {
                'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
                'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
                'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
                'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1]},
                'CatBoost': {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.2], 'iterations': [100, 200]},
                'Extra Trees': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
                #'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
                'KNeighbors': {'n_neighbors': [3, 5, 7]},
                'GaussianNB': {},
                'MLP': {'hidden_layer_sizes': [(100,), (100, 50)], 'alpha': [0.0001, 0.001]},
            }

            # Evaluate models and get the best model
            model_report, best_model, selected_features = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=param_grids
            )
            best_model_score = max(model_report.values())

            if best_model_score < 0.6:
                raise CustomException("No best model found with an acceptable score", sys)

            logging.info(f"Best model found with Test Accuracy: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions on the test set using the best model
            X_test_selected = selected_features.get(best_model.__class__.__name__, X_test)
            predicted = best_model.predict(X_test_selected)
            test_accuracy = accuracy_score(y_test, predicted)
            logging.info(f"Accuracy on test data: {test_accuracy}")


            return test_accuracy

        except Exception as e:
            raise CustomException(e, sys)
