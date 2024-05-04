import pandas as pd
import numpy as np
import logging
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    hamming_loss,
    log_loss,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    mean_absolute_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from src.config_schemas.models.HPT_config_schema import HPT_Config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

import optuna
from optuna import Trial
from optuna import Study
from optuna import pruners
from optuna import create_study
from optuna.pruners import SuccessiveHalvingPruner
from optuna import samplers
from optuna.samplers import TPESampler

import os
import mlflow
from pathlib import Path


def read_data(local_data_dir: str) -> pd.DataFrame:
    df_train = pd.read_csv(local_data_dir + "/train.csv")
    df_test = pd.read_csv(local_data_dir + "/test.csv")
    return df_train, df_test


def save_data(train: pd.DataFrame, test: pd.DataFrame, local_save_dir: str) -> None:
    train.to_csv(local_save_dir + "/train.csv", index=False)
    test.to_csv(local_save_dir + "/test.csv", index=False)


def get_cat_features(df: pd.DataFrame, label: str) -> list[str]:
    cat_features = [
        feature
        for feature in df.columns
        if feature != label
        and (df[feature].dtype == "O" or df[feature].dtype == "bool" or df[feature].dtype == "category")
    ]
    return cat_features


def get_fit_cat_params(estimator_class: str, cat_col_list=[]) -> dict:
    if cat_col_list is None:
        cat_col_list = []  # Default to an empty list if no categories are provided

    estimator_classes = {
        "CatBoostClassifier": {"cat_features": cat_col_list},
        "CatBoostRegressor": {"cat_features": cat_col_list},
        "LGBMClassifier": {"categorical_feature": cat_col_list},
        "LGBMRegressor": {"categorical_feature": cat_col_list},
        "HistGradientBoostingClassifier": {"categorical_features": cat_col_list},
        "HistGradientBoostingRegressor": {"categorical_features": cat_col_list},
    }
    # Directly return the appropriate dictionary or an empty dictionary if the class is not found
    return estimator_classes.get(estimator_class, {})


def get_single_metric(metric_name: str):
    """
    Get metric by its name
    :param metric_name: Name of desired encoder
    :param metric_kwargs: kwrags for metric object
    :return: metric object
    """
    metric_classes = {
        "accuracy": accuracy_score,
        "roc_auc": roc_auc_score,
        "f1": f1_score,
        "confusion_matrix": confusion_matrix,
        "precision": precision_score,
        "recall": recall_score,
        "precision_recall_curve": precision_recall_curve,
        "average_precision": average_precision_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "MAE": mean_absolute_error,
    }

    metric_class = metric_classes.get(metric_name)
    if metric_class:
        return metric_class
    else:
        raise ValueError(f"Metric name '{metric_name}' is not supported.")


class cv_training(BaseEstimator, ClassifierMixin):
    """
    This class performs cross-validated training for a given estimator on a provided dataset. It integrates
    scikit-learn's BaseEstimator and TransformerMixin to support pipelining and consistent interface with
    scikit-learn tools.

    Attributes:
        estimator (object): The machine learning estimator object that adheres to scikit-learn's estimator interface.
        params (dict): Parameters for initializing the estimator.
        random_state (int): Random state to ensure reproducibility.
        cv (object): Cross-validation strategy object.
        n_splits (int): Number of splits for the cross-validation.
        estimators (list): List of trained estimator objects from each fold.
        features (list): List of feature names used for training.
        fit_kwargs (dict): Additional keyword arguments for the `fit` method of the estimator.
        predict_kwargs (dict): Additional keyword arguments for the `predict` method of the estimator.
        metrics (dict): Dictionary storing the scores for each metric across folds.
        metrics_stats (dict): Dictionary storing the statistical measures (mean, median, std, final) for each metric.

    Args:
        n_splits (int): Number of splits for cross-validation.
        estimator (object): Estimator to be used for training.
        params (dict, optional): Dictionary of parameters to initialize the estimator.
        random_state (int, optional): Seed for the random number generator used in cross-validation.

    Raises:
        ValueError: If the label column, metric list, or metric optimization direction list is not properly defined.
    """

    def __init__(self, n_splits: int = 10, estimator=None, params: dict = {}, random_state: int = 42):
        self.estimator = estimator
        self.params = params
        self.random_state = random_state
        self.cv = None
        self.n_splits = n_splits
        self.estimators = []
        self.features = []
        self.conf_matrices = []  # Store confusion matrices for each fold

    def fit(
        self,
        df: pd.DataFrame,
        label: str = None,
        fit_kwargs: dict = {},
        predict_kwargs: dict = {},
        metric_list: list[str] = [],
        metric_opt_dir_list: list[str] = [],
        metric_kwargs: dict = {},
    ):
        """
        Fits the estimator to the data using specified cross-validation strategy and computes metrics for each fold.

        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            label (str): Name of the target variable column.
            fit_kwargs (dict): Additional keyword arguments for the `fit` method of the estimator.
            predict_kwargs (dict): Additional keyword arguments for the `predict` method of the estimator.
            metric_list (list of str): List of metric names to evaluate.
            metric_opt_dir_list (list of str): List specifying the direction ('min' for minimization, 'max' for maximization or 'compr' for comprehensive information) for each metric's optimization.
            metric_kwargs (dict): Additional keyword arguments for each metric computation.

        Returns:
            self: Returns an instance of self.
        """
        logging.info("Starting the fitting process.")
        print("Starting the fitting process.")

        if not label:
            raise ValueError("Label column must be specified.")
        if not metric_list:
            raise ValueError("Metric list must not be empty.")
        if not metric_opt_dir_list:
            raise ValueError("Metric optimization direction list must not be empty.")

        # First, we prepare the features X and the label Y for the training
        self.label = label
        Y = df.loc[:, self.label]
        X = df.drop(columns=self.label)
        self.features = X.columns

        # Second, we initialize the estimator's fit and predict method and the metrics, as well as the dictionaries for metrics and metric stats
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs

        if metric_kwargs == {}:
            self.metric_kwargs = {metric_name: {} for metric_name in metric_list}
        else:
            self.metric_kwargs = metric_kwargs

        self.metrics = {metric_name: [] for metric_name in metric_list}
        self.metrics_stats = {
            metric_name: {"mean": 0.0, "median": 0.0, "std": 0.0, "final": 0.0} for metric_name in metric_list
        }

        self.cv = (
            StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
            if (Y.dtype == "O" or Y.dtype == bool or Y.dtype == "category")
            else KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        )
        # Then, we start cross-validated training process
        for n_fold, (train_index, val_index) in enumerate(self.cv.split(X, Y)):

            logging.info(f"Starting training for fold {n_fold+1}")
            print(f"Starting training for fold {n_fold+1}")

            # Define X_train, Y_train, X_val, Y_val
            X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
            X_val, Y_val = X.iloc[val_index], Y.iloc[val_index]

            # Fit the estimator and append it to the list "self.estimators"
            estimator = self.estimator(**self.params)
            estimator.fit(X_train, Y_train, **fit_kwargs)
            self.estimators.append(estimator)

            # Make predictions and get scores for each metric
            for metric_name in metric_list:
                metric = get_single_metric(metric_name)
                if metric_name in ["roc_auc", "average_precision", "precision_recall_curve"]:
                    y_pred = estimator.predict_proba(X_val, **predict_kwargs)[:, 1]
                else:
                    y_pred = estimator.predict(X_val, **predict_kwargs)
                result = metric(Y_val, y_pred, **self.metric_kwargs[metric_name])
                self.metrics[metric_name].append(result)
            logging.info(f"Completed training for fold {n_fold+1}")
            print(f"Completed training for fold {n_fold+1}")

        # Finally, print some stats for training
        for metric_name, metric_opt_dir in zip(metric_list, metric_opt_dir_list):

            mean_score = np.mean(self.metrics[metric_name])
            self.metrics_stats[metric_name]["mean"] = mean_score

            median_score = np.median(self.metrics[metric_name])
            self.metrics_stats[metric_name]["median"] = median_score

            std_score = np.std(self.metrics[metric_name])
            self.metrics_stats[metric_name]["std"] = std_score

            if metric_opt_dir == "max":
                final_score = np.min([mean_score, median_score])
            elif metric_opt_dir == "min":
                final_score = np.max([mean_score, median_score])
            elif metric_opt_dir == "compr":
                final_score = np.stack(self.metrics[metric_name])
                final_score = np.sum(final_score, axis=0)
                plot_confusion_matrix(final_score, class_labels=["False", "True"])
            else:
                raise ValueError(
                    "metric_opt_dir as the direction of the metric optimization can either be 'min' for minimize and 'max' for maximize"
                )
            self.metrics_stats[metric_name]["final"] = final_score

            print("%" * 100)
            logging.info(
                f"The metric scores in all cv folds for {metric_name} are {self.metrics[metric_name]}. \n The final score is {final_score}, and the standard deviation is {std_score}"
            )
            print(
                f"The metric scores in all cv folds for {metric_name} are {self.metrics[metric_name]}. \n The final score is {final_score}, and the standard deviation is {std_score}"
            )
            print("%" * 100)

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Applies the trained model to predict the target variable on a new dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the new data on which predictions are to be made.

        Returns:
            df (pd.DataFrame): DataFrame with the predictions added as a new column corresponding to the label attribute.

        Raises:
            ValueError: If the model has not been trained before calling this method.
        """
        if not self.estimators:
            logging.error("Please first train the model using fit before making predictions")
            raise ValueError("Please first train the model using fit before making predictions")
        predictions = self.estimator.predict(df, **self.predict_kwargs)
        logging.info("Prediction completed for the test set")
        print("Prediction completed for the test set")

        return predictions


def plot_confusion_matrix(cm, class_labels=None):
    # Default labels to 0 and 1 if none are provided
    if class_labels is None:
        class_labels = ["0", "1"]

    # Define the confusion matrix data and annotations
    z = cm
    x = class_labels
    y = class_labels
    z_text = [[str(y) for y in x] for x in z]

    # Create the confusion matrix as a heatmap
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale="Blues")

    # Add title and axis labels
    fig.update_layout(title="Confusion Matrix", xaxis=dict(title="Predicted value"), yaxis=dict(title="Actual value"))

    # Reverse the y-axis to put '0' at the top
    fig["layout"]["yaxis"]["autorange"] = "reversed"

    return fig


class HPT_Optuna_CV:
    """
    A class for conducting hyperparameter tuning using Optuna with a focus on cross-validation.

    This class provides functionality to perform both non-cross-validated and cross-validated
    model evaluations within Optuna trials to efficiently use computational resources. It first
    evaluates simpler models and proceeds to more computationally expensive cross-validated models
    if initial results are promising.

    Attributes:
        random_state (int): Controls the randomness of the trial sampling and other stochastic elements, defaults to 42.
        prune (bool): Flag to determine whether to use pruning, initialized as True.
        direction (str or None): The direction of optimization ('minimize' or 'maximize'), not set initially.

    """

    def __init__(self, cfg: HPT_Config, random_state: int = 42, prune: bool = False, verbose: int = 0):
        """Initializes the hyperparameter tuning class with the specified random state."""
        self.random_state = random_state
        self.cfg = cfg
        self.prune = prune
        self.direction = None
        self.verbose = verbose

    def instantiate_cv_model(self, trial: Trial, model=None, params: dict = {}):
        """
        Instantiates a cross-validation model using specified parameters.

        Args:
            trial (optuna.trial._trial.Trial): The trial instance from Optuna.
            model (class, optional): The model class to be instantiated for CV.
            params (dict, optional): Parameters to initialize the model.

        Returns:
            object: The instantiated model ready for cross-validation.
        """
        return cv_training(estimator=model, params=params, random_state=self.random_state)

    def obj_function(
        self,
        trial: Trial,
        model=None,
        fit_kwargs: dict = {},
        predict_kwargs: dict = {},
        metric_name: str = None,
        metric_list: list[str] = [],
        metric_opt_dir_list: list[str] = [],
        metric_kwargs: dict = {},
        df: pd.DataFrame = None,
        label: str = None,
    ) -> float:
        """
        Objective function for Optuna optimization, evaluating model performance using both non-CV and CV methods.
        It first evaluates the model on a split dataset and if the results are promising, proceeds to a cross-validated evaluation.

        Args:
            trial (optuna.trial._trial.Trial): The trial instance from Optuna.
            model (class): The model class to be used for the trials.
            fit_kwargs (dict): Keyword arguments for the model's fit method.
            predict_kwargs (dict): Keyword arguments for the model's predict method.
            metric_name (str): The name of the metric to optimize.
            metric_list (list of str): List of metrics to be evaluated.
            metric_opt_dir_list (list of str): Directions ('minimize' or 'maximize') for each metric.
            metric_kwargs (dict): Additional keyword arguments for the metric function.
            df (pd.DataFrame): The dataframe containing features and target.
            label (str): The column name of the target variable in `df`.

        Returns:
            float: The evaluation metric value for the cross-validated model.

        Raises:
            ValueError: If required parameters are not provided or if initial non-CV evaluation fails to meet expectations.
        """
        if df is None or df.empty:
            raise ValueError("Training set must be provided.")
        if not label:
            raise ValueError("The name of thelabel(target) must be specified.")
        if not metric_name:
            raise ValueError("The name of the metric to be optimized must be specified.")
        if metric_name not in metric_list:
            raise ValueError(f"The metric name '{metric_name}' is not in the list of metrics to be evaluated.")
        if not model:
            raise ValueError("The class of the model must be specified.")

        # Define the search space for the hyperparameters
        params = {
            k: getattr(trial, f'suggest_{v["type"]}')(name=v["name"], **{key: val for key, val in v.items() if key not in ['type', 'name']})
            for k, v in self.cfg.hyperparameters.items()
        }
        if self.cfg.verbose_obj_def:
            params["verbose"] = self.verbose
        logging.info(f"Hyperparameters for the trial are {params}")
        # Initialize features and label
        X = df.drop(columns=label)
        Y = df[label]
        cat_features = get_cat_features(X, label)
        cat_fit_kwargs = get_fit_cat_params(model.__name__, cat_col_list=cat_features)
        # Pruning a trial based on the score of the model without cv. If the non-cv score is promising, the trial won't be killed and we proceed to train the model with cv.
        if self.prune is True:
            # Intialize the model, the training and the validation set without cv to be evaluated for pruning
            model_prune = model(**params, random_state=self.random_state)
            X_train, X_val, Y_train, Y_val = train_test_split(
                X, Y, test_size=0.2, random_state=self.random_state, stratify=df[label]
            )
            # If a boosting tree model has native support warm_start, we prune the model by adding a single tree in each iteration.

            if params.get("warm_start") is True and params.get("n_estimator", 0) > 100:
                n_estimators = model.get_params().get("n_estimators")
                min_estimators = 100

                for num_estimator in range(min_estimators, n_estimators):
                    model_prune.set_params(n_estimators=num_estimator)
                    model_prune.fit(X_train, Y_train, **fit_kwargs, **cat_fit_kwargs)
                    y_pred = model_prune.predict(X_val, **predict_kwargs)
                    metric = get_single_metric(metric_name)
                    if (
                        metric_name == "roc_auc"
                        or metric_name == "average_precision"
                        or metric_name == "precision_recall_curve"
                    ):
                        y_pred = model_prune.predict_proba(X_val, **predict_kwargs)[:, 1]
                    score = metric(Y_val, y_pred, **metric_kwargs)
                    trial.report(score, num_estimator)

                    if trial.should_prune():
                        raise optuna.TrialPruned()
            # In other cases, if the pruner is still SuccessiveHalvingPruner, we run an Asynchronous SHA (ASHA) implementation to prune a trial
            elif self.pruner == SuccessiveHalvingPruner:
                n_samples_list = generate_sample_numbers(Y_train, self.base, self.n_rungs)
                for n_samples in n_samples_list:
                    _, X_train_sample, _, Y_train_sample = train_test_split(
                        X_train, Y_train, test_size=n_samples, random_state=self.random_state, stratify=Y_train
                    )
                    model_prune.fit(X_train_sample, Y_train_sample.values.ravel(), **fit_kwargs, **cat_fit_kwargs)
                    y_pred = model_prune.predict(X_val, **predict_kwargs)
                    metric = get_single_metric(metric_name)
                    if (
                        metric_name == "roc_auc"
                        or metric_name == "average_precision"
                        or metric_name == "precision_recall_curve"
                    ):
                        y_pred = model_prune.predict_proba(X_val, **predict_kwargs)[:, 1]
                    score = metric(Y_val, y_pred, **metric_kwargs)
                    trial.report(score, n_samples)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

        model_cv = self.instantiate_cv_model(trial, model=model, params=params)
        model_cv.fit(
            df,
            label,
            fit_kwargs={**fit_kwargs, **cat_fit_kwargs},
            predict_kwargs=predict_kwargs,
            metric_list=metric_list,
            metric_opt_dir_list=metric_opt_dir_list,
            metric_kwargs=metric_kwargs,
        )
        score_cv = model_cv.metrics_stats[metric_name]["final"]
        return score_cv

    def launch_study(
        self,
        study_name: str = None,
        pruner: pruners.BasePruner = SuccessiveHalvingPruner,
        pruner_kwargs: dict = {"reduction_factor": 2},
        n_rungs: int = 4,
        sampler: samplers.BaseSampler = TPESampler,
        sampler_kwargs: dict = {},
        model=None,
        model_name: str = None,
        encoding: str = None,
        fit_kwargs: dict = {},
        predict_kwargs: dict = {},
        metric_name: str = None,
        metric_list: list[str] = [],
        metric_opt_dir_list: list[str] = [],
        metric_kwargs: dict = {},
        df: pd.DataFrame = None,
        label: str = None,
        n_trials: int = 100,
        artifact_directory: str = None,
        if_callback: bool = True,
    ):
        """
        Launches an Optuna optimization study.

        Args:
            study_name (str, optional): Name of the study.
            pruner (optuna.pruners.BasePruner, optional): The pruning strategy to use.
            pruner_kwargs (dict, optional): Keyword arguments to initialize the pruner.
            model (class, optional): The model class to be used.
            model_name (str, optional): Name of the model.
            encoding (str, optional): Encoding type to use.
            fit_kwargs (dict, optional): Additional keyword arguments for fitting the model.
            predict_kwargs (dict, optional): Additional keyword arguments for model prediction.
            metric_name (str, optional): Name of the primary metric for optimization.
            metric_list (list of str, optional): List of all metrics to consider.
            metric_opt_dir_list (list of str, optional): Directions of optimization for each metric.
            metric_kwargs (dict, optional): Additional keyword arguments for metrics.
            df (pd.DataFrame, optional): DataFrame containing the training data.
            label (str, optional): Name of the label column in `df`.
            n_trials (int, optional): Number of trials to conduct, defaults to 100.
            artifact_directory (str, optional): Directory to save artifacts.
            if_callback (bool, optional): Flag to determine whether to use the callback function.

        Returns:
            optuna.study.Study: The completed Optuna study.
        """
        # Check if the reduction factor (base) is an integer greater than or equal to 2 if the pruner is Successive Halving
        self.pruner = pruner
        if pruner == SuccessiveHalvingPruner and ("reduction_factor" in pruner_kwargs):
            if pruner_kwargs["reduction_factor"] < 2 or pruner_kwargs["reduction_factor"] % 1 != 0:
                raise ValueError(
                    "The reduction factor for the Successive Halving Pruner must be an integer greater than or equal to 2"
                )
            else:
                self.base = pruner_kwargs["reduction_factor"]
                self.n_rungs = n_rungs

        self.direction = metric_opt_dir_list[metric_list.index(metric_name)]
        # Create the study according to the direction of the metric optimization
        if self.direction == "min":
            study = create_study(
                direction="minimize",
                pruner=pruner(**pruner_kwargs),
                sampler=sampler(seed=self.random_state, **sampler_kwargs),
            )
        elif self.direction == "max":
            study = create_study(
                direction="maximize",
                pruner=pruner(**pruner_kwargs),
                sampler=sampler(seed=self.random_state, **sampler_kwargs),
            )
        else:
            raise ValueError(
                "The direction of the metric optimization can either be 'min' for minimize and 'max' for maximize"
            )

        # Create call back to log the best trial
        artifact_directory = Path(artifact_directory) / f"{model_name}_{encoding}"
        if if_callback:
            def callback(study, trial):
                log_best_trial(
                    study=study,
                    trial=trial,
                    df=df,
                    label=label,
                    model=model,
                    model_name=model_name,
                    encoding=encoding,
                    fit_kwargs=fit_kwargs,
                    predict_kwargs=predict_kwargs,
                    metric_list=metric_list,
                    metric_opt_dir_list=metric_opt_dir_list,
                    metric_kwargs=metric_kwargs,
                    metric_obj=metric_name,
                    random_state=self.random_state,
                    directory=artifact_directory,
                )
        else:
            callback = None
        # Optimize the objective function
        study.optimize(
            lambda trial: self.obj_function(
                trial=trial,
                model=model,
                fit_kwargs=fit_kwargs,
                predict_kwargs=predict_kwargs,
                metric_name=metric_name,
                metric_list=metric_list,
                metric_opt_dir_list=metric_opt_dir_list,
                metric_kwargs=metric_kwargs,
                df=df,
                label=label
            ),
            callbacks=[callback],
            n_trials=n_trials,
        )
        return study


def log_int(x, base: int = 2):
    return np.floor(np.log(x) / np.log(base)).astype(int)


def generate_sample_numbers(y: pd.DataFrame, base: int, n_rungs: int) -> list[int]:
    """
    Generates a list of sample numbers based on the total number of samples in the DataFrame,
    scaled logarithmically according to a specified base and number of rungs. This function is
    typically used to determine the sizes of samples for successive halving in hyperparameter tuning.

    Args:
        y (pd.DataFrame): DataFrame containing the target variable, used to determine the total number of samples.
        base (int): The base of the logarithm used for calculating scales of sample sizes.
        n_rungs (int): The number of divisions or "rungs" to use in the scaling, affecting the granularity of sample sizes.

    Returns:
        list[int]: A list of integers representing sample sizes for each rung, scaled logarithmically from the minimum scale to the original scale of the data.

    Example:
        # Assuming 'y' is a DataFrame with 1000 entries, base is 2, and n_rungs is 3
        >>> sample_sizes = generate_sample_numbers(y, 2, 3)
        >>> print(sample_sizes)
        [125, 250, 500, 1000]

    This example would output sample sizes starting from approximately 1/8th of the total data size up to the full size,
    doubling at each rung if the base is 2 and there are 3 rungs.
    """

    data_size = len(y)
    data_scale = log_int(data_size, base)
    min_scale = data_scale - n_rungs

    return [*map(lambda scale: base**scale, range(min_scale, data_scale + 1))]


def convert_object_to_category(df):
    """
    Convert all object type columns in a DataFrame to category type.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        pd.DataFrame: A DataFrame with object type columns converted to category type.
    """
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')
    return df


def get_serial_number(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            serial_number = int(file.read().strip()) + 1
    else:
        serial_number = 1
    with open(file_path, "w") as file:
        file.write(str(serial_number))
    return serial_number


def log_best_trial(
    study: Study,
    trial: Trial,
    df: pd.DataFrame,
    label: str,
    encoding: str = None,
    model=None,
    model_name: str = None,
    fit_kwargs: dict = {},
    predict_kwargs: dict = {},
    metric_list: list = [],
    metric_opt_dir_list: list = [],
    metric_kwargs: dict = {},
    metric_obj: str = None,
    random_state: int = None,
    directory=None
) -> None:
    """Logs the best trial from an Optuna study into MLflow, including retraining the model, logging metrics, and saving a confusion matrix as an HTML file.

    This function is designed to be used as a callback during the optimization process in Optuna. It is triggered after each trial, and if the trial is identified as the best so far, it performs extensive logging including parameters, metrics, and model artifacts. Additionally, it saves a confusion matrix visualization to an HTML file.

    Args:
        study (Study): The study object from Optuna containing all trials.
        trial (Trial): The current trial object with access to the trial's parameters and results.
        df (pd.DataFrame): DataFrame containing the data used for training the model.
        label (str): The name of the target variable column in the dataframe.
        encoding (str, optional): Encoding type if applicable, affects serialization filename. Defaults to None.
        model (object, optional): The model class or instance to be used. Defaults to None.
        model_name (str, optional): The base name for the saved model and logs.
        fit_kwargs (dict, optional): Additional keyword arguments to pass to the model's fit method.
        predict_kwargs (dict, optional): Additional keyword arguments to pass to the model's predict method.
        metric_list (list, optional): List of metric names to evaluate.
        metric_opt_dir_list (list, optional): List indicating the optimization direction ('maximize' or 'minimize') for each metric in `metric_list`.
        metric_kwargs (dict, optional): Additional keyword arguments for metric calculation.
        metric_obj (str, optional): The name of the metric used as the optimization objective. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
        directory (str, optional): Base directory for saving files related to this run.

    Raises:
        IOError: If the function fails to read or write the serial number file.
    """
    if trial.number == study.best_trial.number:
        # Get the serial number for the model
        try:
            serial_number = get_serial_number(directory / f"serial_number_{model_name}_{encoding}.txt")
        except IOError as e:
            logging.error("Failed to read/write serial number file: %s", e)
            return

        with mlflow.start_run(run_name=f"{model_name}_{encoding}_best_trial_{serial_number}"):
            mlflow.log_params(trial.params)
            print(f"the parameters in trial are {trial.params}")
            # Assuming the model can be retrained here or retrieved somehow
            model_cv = cv_training(estimator=model, params=trial.params, random_state=random_state)
            cat_fit_kwargs = get_fit_cat_params(model.__name__, cat_col_list=get_cat_features(df, label))
            model_cv.fit(
                df,
                label,
                fit_kwargs={"verbose": False, **fit_kwargs, **cat_fit_kwargs},
                predict_kwargs=predict_kwargs,
                metric_list=metric_list,
                metric_opt_dir_list=metric_opt_dir_list,
                metric_kwargs=metric_kwargs,
            )
            for metric in [m for m in model_cv.metrics.keys() if m != "confusion_matrix"]:
                for stat in ["mean", "median", "std", "final"]:
                    mlflow.log_metric(f"{metric}_{stat}", model_cv.metrics_stats[metric][stat])
                    # Generate and save the confusion matrix figure
            fig = plot_confusion_matrix(
                model_cv.metrics_stats["confusion_matrix"]["final"], class_labels=["False", "True"]
            )
            figure_path = directory / f"confusion_matrix_{model_name}_{encoding}_{serial_number}.html"
            fig.write_html(figure_path)  # Save the figure to HTML file
            mlflow.log_artifact(figure_path)  # Log the HTML file as an artifact
            mlflow.sklearn.log_model(model_cv, f"{model_name}_{encoding}_best_trial_{serial_number}")
            logging.info(f"Model saved as {model_name}_{encoding}_best_trial_{serial_number}")


def get_model_class(model_name: str):
    models = {
        "CatBoost": CatBoostClassifier,
        "XGBoost": xgb.XGBClassifier,
        "LGBM": lgb.LGBMClassifier,
        "ExtraTrees": ExtraTreesClassifier,
        "Hist": HistGradientBoostingClassifier,
    }
    return models.get(model_name)
