import pandas as pd
import numpy as np
from src.utils.HPT_utils import read_data, save_data, cv_training
from joblib import dump
from pathlib import Path
from src.utils.config_utils import get_config
import logging
from src.config_schemas.models.HPT_config_schema import HPT_Config
from src.utils.HPT_utils import get_cat_features, plot_confusion_matrix, get_fit_cat_params, convert_object_to_category, get_model_class
from src.utils.HPT_utils import HPT_Optuna_CV

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import pyfunc
from optuna import pruners, samplers
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_intermediate_values, plot_parallel_coordinate, plot_slice, plot_edf


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

                mlflow.sklearn.log_model(base_model, f"{model_name}_{encoding}")
                logging.info(f"Model saved as {model_name}_{encoding}")


@get_config(config_path="../configs/models", config_name="HPT_config")
def Hyperparameters_tuning(config: HPT_Config) -> None:
    """
    Performs hyperparameter tuning for a specified machine learning model using Optuna, with
    extensive logging and visualization via MLflow.

    This function reads training data, configures and executes an Optuna study to find optimal
    hyperparameters, and logs the study results and visualizations in MLflow. It assumes a specific
    configuration setup and requires an accompanying configuration file.

    Args:
        config (HPT_Config): An object containing all the necessary configuration settings. This includes
                             paths, model parameters, fitting parameters, and Optuna configuration like
                             pruners, samplers, and the number of trials.

    The function executes the following major steps:
    - Reads and preprocesses the data.
    - Sets up and runs the Optuna hyperparameter tuning study.
    - Logs the results of the study to MLflow.
    - Generates and saves several Optuna visualization plots to HTML files, which are then logged to MLflow.

    Visualizations include:
    - Optimization history
    - Parameter importances
    - Contour plots of the hyperparameters
    - Intermediate values across trials
    - Parallel coordinate plot of the trials
    - Slice plots of the hyperparameters
    - Empirical distribution function (EDF) of the trials

    Notes:
    - This function is highly dependent on the structure of the `HPT_Config` class, which needs to be predefined
      with all required attributes.
    - Proper error handling should be implemented within the configuration reading and writing processes to
      ensure robustness.

    Raises:
        FileNotFoundError: If specified data or configuration files are not found.
        ValueError: If any configurations are invalid or if the hyperparameter tuning fails due to configuration issues.
        Exception: General exception catch for unexpected errors, with a log output for diagnostics.
    """
    logging.info("Reading data...")
    df_train, _ = read_data(config.local_data_dir + f"/{config.HPT_encoding}")
    df_train = convert_object_to_category(df_train)
    logging.info("Data read successfully")

    logging.info("Setting up the hyperparameter tuning...")
    HPT = HPT_Optuna_CV(cfg=config)
    mlflow.set_experiment(f"{config.HPT_model_name}_{config.HPT_encoding}")
    study = HPT.launch_study(
        study_name=f"{config.HPT_model_name}_{config.HPT_encoding}",
        pruner=config.pruner,
        pruner_kwargs=config.pruner_kwargs,
        n_rungs=config.n_rungs,
        sampler=config.sampler,
        sampler_kwargs=config.sampler_kwargs,
        model=get_model_class(config.HPT_model_name),
        model_name=config.HPT_model_name,
        encoding=config.HPT_encoding,
        fit_kwargs=config.fit_kwargs,
        predict_kwargs=config.predict_kwargs,
        metric_name=config.metric_name,
        metric_list=config.metric_list,
        metric_opt_dir_list=config.metric_opt_dir_list,
        metric_kwargs=config.metric_kwargs,
        df=df_train,
        label=config.label,
        n_trials=config.n_trials,
        artifact_directory=config.artifact_directory,
        if_callback=config.if_callback,
    )
    logging.info("Hyperparameter tuning completed successfully")

    artifact_path = Path(config.artifact_directory) / f"{config.HPT_model_name}_{config.HPT_encoding}"
    logging.info("Start visualization...")
    plot_optimization_history(study).write_html(artifact_path / "optimization_history.html")
    mlflow.log_artifact(artifact_path / "optimization_history.html")
    plot_param_importances(study).write_html(artifact_path / "param_importances.html")
    mlflow.log_artifact(artifact_path / "param_importances.html")
    plot_contour(study).write_html(artifact_path / "contour.html")
    mlflow.log_artifact(artifact_path / "contour.html")
    plot_intermediate_values(study).write_html(artifact_path / "intermediate_values.html")
    mlflow.log_artifact(artifact_path / "intermediate_values.html")
    plot_parallel_coordinate(study).write_html(artifact_path / "parallel_coordinate.html")
    mlflow.log_artifact(artifact_path / "parallel_coordinate.html")
    plot_slice(study).write_html(artifact_path / "slice.html")
    mlflow.log_artifact(artifact_path / "slice.html")
    plot_edf(study).write_html(artifact_path / "edf.html")
    mlflow.log_artifact(artifact_path / "edf.html")

    logging.info("Completed visualization...")


if __name__ == "__main__":
    Hyperparameters_tuning()  # type: ignore
