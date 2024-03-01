from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from src.CarPricePrediction.components.data_transformation import data_transform
import pandas as pd
from src.CarPricePrediction.logger import logging

class ModelTraining:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "ElasticNet": ElasticNet(),
            "Random Forest Regression": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN Regression": KNeighborsRegressor(),
        }

    # Inside model_train.py

    def load_data(self):
        data = pd.read_csv(self.data_file_path, dtype={"Selling_Price": float})
        return data

    def data_preprocessing(self, data):
        transformer = data_transform(data)
        transformed_data = transformer.data_encoding()
        logging.info("No need to transform the data")
        return transformed_data

    def train_models(self, x_train, y_train):
        for name, model in self.models.items():
            model.fit(x_train, y_train)

    def evaluate_models(self, x_test, y_test):
        r2_scores = {}
        for model_name, model in self.models.items():
            predictions = model.predict(x_test)
            r2 = r2_score(y_test, predictions)
            r2_scores[model_name] = r2
            logging.info(f"Model: {model_name}, R2 Score: {r2}")
        logging.info("------------ Model Train Completed ------------\n")
        return r2_scores

    def get_best_model(self, r2_scores):
        best_model_name = max(r2_scores, key=r2_scores.get)
        best_model_score = r2_scores[best_model_name]
        logging.info("Initializing the best Model from R2 Score")
        return best_model_name, best_model_score