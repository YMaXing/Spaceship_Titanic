import pandas as pd
import numpy as np
from src.utils.HPT_utils import read_data, save_data, cv_training
from joblib import dump
from pathlib import Path
from src.utils.config_utils import get_config
import logging
from src.config_schemas.models.HPT_config_schema import HPT_Config
from src.utils.HPT_utils import get_cat_features, plot_confusion_matrix, get_fit_cat_params, convert_object_to_category
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import pyfunc


@get_config(config_path="../configs/models", config_name="HPT_config")
def Baseline_models(config: HPT_Config) -> None:
    df_train_NE, _ = read_data(config.local_data_dir + "/unencoded")
    df_train_ME, _ = read_data(config.local_data_dir + "/MEstimate")
    df_train_Mixed, _ = read_data(config.local_data_dir + "/Mixed")
    
    df_train_NE = convert_object_to_category(df_train_NE)
    df_train_ME = convert_object_to_category(df_train_ME)
    df_train_Mixed = convert_object_to_category(df_train_Mixed)

    training_sets = {"NE": df_train_NE, "ME": df_train_ME, "Mixed": df_train_Mixed}

    mlflow.set_experiment("Baseline_models")
    for model, model_name in config.models:
        for encoding in config.encodings:
            if model_name == "ExtraTrees" and encoding == "NE":
                continue
            logging.info(f"Training {model_name} with {encoding} encoding")
            with mlflow.start_run(run_name=f"Base_{model_name}_{encoding}"):
                base_params = config.base_params_dict[model_name]
                base_fit_kwargs = get_fit_cat_params(model.__name__, cat_col_list=get_cat_features(training_sets[encoding], 'Transported'))
                # The initialization of the model is different for HistGradientBoostingClassifier
                # as it takes the list of categorical features directly as an argument when define an object of it, instead of an argument for its fit method.
                if model_name == "Hist":
                    base_params = {**base_params, **base_fit_kwargs}
                    base_model = cv_training(estimator=model, params=base_params, n_splits=10)
                    base_model.fit(
                        training_sets[encoding],
                        label="Transported",
                        metric_list=["accuracy", "roc_auc", "f1", "confusion_matrix"],
                        metric_opt_dir_list=["max", "max", "max", "compr"],
                    )
                else:
                    base_model = cv_training(estimator=model, params=base_params, n_splits=10)
                    base_model.fit(training_sets[encoding], label="Transported", fit_kwargs=base_fit_kwargs, metric_list=["accuracy", "roc_auc", "f1", "confusion_matrix"], metric_opt_dir_list=["max", "max", "max", "compr"])
                logging.info(f"Model trained with {base_model.metrics_stats['accuracy']['final']} accuracy")

                mlflow.log_params(base_model.estimators[0].get_params())
                for metric in [m for m in base_model.metrics.keys() if m != "confusion_matrix"]:
                    for stat in ["mean", "median", "std", "final"]:
                        mlflow.log_metric(f"{metric}_{stat}", base_model.metrics_stats[metric][stat])

                # Generate and save the confusion matrix figure
                fig = plot_confusion_matrix(base_model.metrics_stats["confusion_matrix"]["final"], class_labels=["False", "True"])
                figure_path = Path(config.directory_base) / f"confusion_matrix_{model_name}_{encoding}.html"  # Full path
                fig.write_html(figure_path)  # Save the figure to HTML file
                mlflow.log_artifact(figure_path)  # Log the HTML file as an artifact

                mlflow.sklearn.log_model(model, f"{model_name}_{encoding}")
                logging.info(f"Model saved as {model_name}_{encoding}")

@get_config(config_path="../configs/models", config_name="HPT_config")
def Hyperparameters_tuning(config: HPT_Config) -> None:
    pass


if __name__ == "__main__":
    Baseline_models()  # type: ignore
