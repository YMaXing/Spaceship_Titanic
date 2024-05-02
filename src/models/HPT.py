import pandas as pd
import numpy as np
from src.utils.HPT_utils import read_data, save_data, cv_training
from joblib import dump
from src.utils.config_utils import get_config
import logging
from src.config_schemas.models.HPT_config_schema import HPT_Config
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


@get_config(config_path="../configs/models", config_name="HPT_config")
def HPT(config: HPT_Config, args) -> None:

    logging.info("Hyperparameter tuning started")
    # Load training data
    train = pd.read_csv(config.local_data_dir + "/train.csv")

    mlflow.set_experiment("Hyperparameter Tuning for " + config.model_name)
    with mlflow.start_run():
        # Split training data
        X_train = train.drop(config.label, axis=1)
        y_train = train[config.label]

        # Hyperparameter tuning
        model = config.model(**config.model_params)
        model.fit(X_train, y_train)
        logging.info(f"Best hyperparameters: {model.best_params_}")
        logging.info(f"Best score: {model.best_score_}")
        logging.info("Hyperparameter tuning ended")

        # Save best model
        mlflow.log_params("model", config.model_name)
        mlflow.log_params("best params", model.best_params_)
        mlflow.log_params("best score", model.best_score_)


    logging.info("Model training started")
    # Model training
    model = config.model(**model.best_params_)
    model.fit(X_train, y_train)
    logging.info("Model training ended")

if __name__ == "__main__":
    HPT(parse_args())  # type: ignore
