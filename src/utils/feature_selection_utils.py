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
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging
from shap import TreeExplainer
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import random


def get_cat_features(df: pd.DataFrame, label: str) -> list[str]:
    cat_features = [feature for feature in df.columns if feature != label and (df[feature].dtype == "O" or df[feature].dtype == "bool" or df[feature].dtype == "category")]
    return cat_features


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


class cv_training(BaseEstimator, TransformerMixin):
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

    def __init__(self, n_splits: int = 5, estimator=None, params: dict = {}, random_state: int = 42):
        self.estimator = estimator
        self.params = params
        self.random_state = random_state
        self.cv = None
        self.n_splits = n_splits
        self.estimators = []
        self.features = []

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
            metric_opt_dir_list (list of str): List specifying the direction ('min' or 'max') for each metric's optimization.
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
            y_pred = estimator.predict(X_val, **predict_kwargs)
            for metric_name in metric_list:
                metric = get_single_metric(metric_name)
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

    def transform(self, df: pd.DataFrame):
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
        df[self.label] = predictions
        logging.info("Prediction completed for the test set")
        print("Prediction completed for the test set")

        return df


def get_feature_contributions(y_true, y_pred, shap_values):
    """Compute prediction contribution and error contribution for each feature."""

    prediction_contribution = shap_values.abs().mean().rename("prediction_contribution")

    abs_error = (y_true - y_pred).abs()
    y_pred_wo_feature = shap_values.apply(lambda feature: y_pred - feature)
    abs_error_wo_feature = y_pred_wo_feature.apply(lambda feature: (y_true - feature).abs())
    error_contribution = (
        abs_error_wo_feature.apply(lambda feature: abs_error - feature).mean().rename("error_contribution")
    )

    return prediction_contribution, error_contribution


class shap_tree_cv:
    """
    A class designed to compute SHAP values for machine learning models trained with cross-validation
    specifically for tree-based models. It also calculates mean absolute error (MAE) and provides visualization
    for the contributions of each feature towards predictions and errors.

    Attributes:
        trained_cv (cv_training): An instance of the cv_training class containing trained estimators.
        shap_explainers (list): A list of SHAP explainer objects for each fold.
        contributions (list): A list of dataframes with contributions for predictions and errors for each fold.
        MAEs (list): A list of mean absolute errors computed for each fold.

    Args:
        trained_cv (cv_training): A pre-trained cv_training instance containing tree-based estimators.
    """

    def __init__(self, trained_cv: cv_training):

        self.trained_cv = trained_cv
        self.shap_explainers = []
        self.contributions = []
        self.MAEs = []

    def get_MAE_and_contrib(self, df: pd.DataFrame):
        """
        Computes the Mean Absolute Error (MAE) and SHAP values for the features used in the trained models
        against a provided dataframe. The dataframe must contain the same features and the label as the one
        used for training the models in trained_cv.

        Args:
            df (pd.DataFrame): The dataframe containing the same structure (features and label) as used in training.

        Returns:
            float: The average MAE across all folds.
            pd.DataFrame: The mean of the contributions of predictions and errors for all features across all folds.

        Raises:
            ValueError: If the input dataframe does not contain the same features or the target label as the training dataframe.
        """

        # Check if the imput dataframe has the same features and label as the dataframe trained_cv was trained on
        if self.trained_cv.label not in df.columns:
            raise ValueError(f"Error: The target column '{self.trained_cv.label}' is missing.")
        if set(self.trained_cv.features) != set(feature for feature in df.columns if feature != self.trained_cv.label):
            raise ValueError("Error: The imput dataframe do NOT have the same features as trained_cv was trained on.")

        X = df[self.trained_cv.features]
        Y = df[self.trained_cv.label]

        for i_fold, (_, val_index) in enumerate(self.trained_cv.cv.split(X, Y)):
            X_val, y_val = X.iloc[val_index], Y.iloc[val_index]
            estimator = self.trained_cv.estimators[i_fold]
            y_pred = pd.Series(estimator.predict(X_val, **self.trained_cv.predict_kwargs), index=X_val.index)

            # Get shap values for each training fold
            logging.info(f"Starting acquiring shap values for fold {i_fold+1}")
            print(f"Starting acquiring shap values for fold {i_fold+1}")
            shap_explainer = TreeExplainer(estimator)
            self.shap_explainers.append(shap_explainer)
            shaps = pd.DataFrame(
                data=shap_explainer.shap_values(X_val), index=X_val.index, columns=self.trained_cv.features
            )
            logging.info(f"Finished acquiring shap values for fold {i_fold+1}")
            print(f"Finished acquiring shap values for fold {i_fold+1}")

            # Get the MAE, and then get the both prediction and error contributions for each feature in each training fold
            MAE = mean_absolute_error(y_val, y_pred)
            self.MAEs.append(MAE)
            prediction_contribution, error_contribution = get_feature_contributions(y_val, y_pred, shaps)
            contribution = pd.concat([prediction_contribution, error_contribution], axis=1)
            self.contributions.append(contribution)

        return (
            np.mean(self.MAEs),
            pd.concat(self.contributions, keys=range(len(self.contributions))).groupby(level=1).mean(),
        )

    def plot_contributions(self) -> None:
        """
        Generates scatter plots of the contributions of each feature towards the predictions and errors.
        Each plot corresponds to a different fold from the cross-validation training. The method checks if
        contributions are available before plotting.

        Raises:
            ValueError: If contributions have not been calculated before calling this method.
        """
        if not self.contributions:
            raise ValueError("Please first calculate the contributions using get_contributions method!")
        for i_fold in range(0, self.trained_cv.n_splits):
            # Create the scatter plot using Plotly Express
            fig = px.scatter(
                self.contributions[i_fold],
                x="prediction_contribution",
                y="error_contribution",
                text=self.contributions[i_fold].index,
                title=f"Training fold {i_fold}",
                labels={
                    "prediction_contribution": "Prediction Contribution",
                    "error_contribution": "Error Contribution",
                },
                width=1200,
                height=800,
            )

            # Adding a horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="black")

            # Customize the plot layout
            fig.update_layout(
                xaxis_title="Prediction Contribution",
                yaxis_title="Error Contribution",
                title_font_size=16,
                font_size=12,
                xaxis_rangeslider_visible=True,  # Optional: add a range slider for the x-axis
            )

            # Show the plot
            fig.show()


def get_fit_cat_params(estimator_class, cat_col_list=[]) -> dict:
    estimator_classes = {
        "CatBoostClassifier": {"cat_features": cat_col_list},
        "CatBoostRegressor": {"cat_features": cat_col_list},
        "LGBMClassifier": {"categorical_feature": cat_col_list},
        "LGBMRegressor": {"categorical_feature": cat_col_list},
        "HistGradientBoostingClassifier": {"categorical_features": cat_col_list},
        "HistGradientBoostingRegressor": {"categorical_features": cat_col_list},
    }
    cat_key = estimator_classes.get(estimator_class)
    if estimator_class:
        return cat_key
    else:
        raise ValueError(f"Estimator class '{estimator_class}' is not supported.")


class rfe_shap_cv:
    """
    This class implements Recursive Feature Elimination (RFE) using SHAP values to guide the feature elimination
    process. It integrates cross-validated model training and computes SHAP values to determine the least
    important features which are subsequently dropped in each iteration until a specified number of features remains.

    Attributes:
        contribution (str): Specifies whether 'error_contribution' or 'prediction_contribution' is used to
                            evaluate feature importance.
        n_feat_final (int): The final number of features to retain.
        rfe_record (pd.DataFrame): Records the metrics and features eliminated in each iteration of RFE.

    Args:
        contribution (str): Specifies the type of SHAP value contribution to use for RFE ('Error' for error_contribution
                            and anything else defaults to prediction_contribution).
        n_feat_final (int): The desired number of features to retain after the elimination process.
    """

    def __init__(self, contribution: str = "Error", n_feat_final: int = 1):
        self.contribution = "error_contribution" if contribution == "Error" else "prediction_contribution"
        self.n_feat_final = n_feat_final
        self.rfe_record = pd.DataFrame(dtype=float)

    def RFE(
        self,
        df: pd.DataFrame,
        label: str = None,
        cv_trainer_params: dict = {},
        fit_kwargs: dict = {},
        predict_kwargs: dict = {},
        metric_list: list[str] = [],
        metric_opt_dir_list: list[str] = [],
        metric_kwargs: dict = {},
    ):
        """
        Executes the Recursive Feature Elimination (RFE) process using cross-validated training and SHAP value analysis.

        Args:
            df (pd.DataFrame): The dataframe containing the features and label for model training.
            label (str): The name of the label column in the dataframe.
            cv_trainer_params (dict): Parameters to initialize the cv_training class.
            fit_kwargs (dict): Keyword arguments for the fit method of the model.
            predict_kwargs (dict): Keyword arguments for the predict method of the model.
            metric_list (list[str]): List of metrics to evaluate during model training.
            metric_opt_dir_list (list[str]): List specifying the direction ('min' or 'max') for each metric's optimization.
            metric_kwargs (dict): Additional keyword arguments for metric computations.

        Returns:
            pd.DataFrame: A dataframe containing the record of number of features, MAE, worst contributor, and the feature
                          dropped in each iteration of the RFE process.

        Raises:
            ValueError: If the label is not provided or the dataframe does not contain the necessary structure.
        """
        if not label:
            raise ValueError("Error: Please first input the label before starting RFE.")

        # First, we get the features and the label of the dataset
        self.label = label
        self.features = [feature for feature in df.columns if feature != self.label]
        features_curr = self.features
        # Then, we start RFE process with cv
        for iteration in tqdm(range(len(self.features) - self.n_feat_final + 1)):
            print("%" * 100)
            logging.info(f"Starting to eliminate feature number {iteration+1}")
            print(f"Starting to eliminate feature number {iteration+1}")
            print("%" * 100)
            # First, we initialize the cv_trainer and fit it with the subset of the dataset with the remaining best features
            cv_trainer = cv_training(**cv_trainer_params)
            df_curr = df[features_curr + [self.label]]
            fit_kwargs_curr = {
                **get_fit_cat_params(cv_trainer.estimator.__name__, cat_col_list=get_cat_features(df_curr, self.label)),
                **fit_kwargs,
            }

            cv_trainer.fit(
                df_curr,
                self.label,
                fit_kwargs=fit_kwargs_curr,
                predict_kwargs=predict_kwargs,
                metric_list=metric_list,
                metric_opt_dir_list=metric_opt_dir_list,
                metric_kwargs=metric_kwargs,
            )
            # Then, we initialize the shap_tree_cv and get the contributions of each feature in each cv fold
            shap_cv = shap_tree_cv(trained_cv=cv_trainer)
            MAE, shap_cv_contributions = shap_cv.get_MAE_and_contrib(df_curr)
            print(shap_cv_contributions)

            # We drop the worst feature, i.e. the feature with the smallest prediction contribution or the biggest error contribution and then record the contributions in this iteration
            self.rfe_record.loc[iteration, "n_features"] = len(features_curr)
            self.rfe_record.loc[iteration, "mae"] = MAE
            if self.contribution == "error_contribution":
                self.rfe_record.loc[iteration, "worst_contrib"] = shap_cv_contributions[self.contribution].max()
                feature_drop = shap_cv_contributions[self.contribution].idxmax()
            else:
                self.rfe_record.loc[iteration, "worst_contrib"] = shap_cv_contributions[self.contribution].min()
                feature_drop = shap_cv_contributions[self.contribution].idxmin()

            self.rfe_record.loc[iteration, "feature_drop"] = feature_drop
            features_curr.remove(feature_drop)
            print("%" * 100)
            logging.info(f"Eliminated feature number {iteration+1}: {feature_drop}")
            print(f"Eliminated feature number {iteration+1}: {feature_drop}")
            print("%" * 100)

        return self.rfe_record


def plot_RFE(rfe_prediction, rfe_error, max_iter_show: int):
    """
    Generates a line plot to visualize the Recursive Feature Elimination (RFE) process for both prediction and error
    contributions across specified iterations. It highlights the iteration where the mean absolute error (MAE) is minimized.

    Args:
        rfe_prediction (pd.DataFrame): A DataFrame containing RFE results where prediction contribution was used
                                       for feature elimination. Must include 'mae' and 'n_features' columns.
        rfe_error (pd.DataFrame): A DataFrame containing RFE results where error contribution was used for feature
                                  elimination. Must include 'mae' and 'n_features' columns.
        max_iter_show (int): The maximum number of iterations to display on the plot.

    Plot Elements:
        - Two line plots representing the MAE over iterations for both prediction and error contributions.
        - Scatter points indicating the position of minimum MAE for both types of contributions.
        - Customized axes with inverted feature counts to indicate the progression of feature elimination.

    Note:
        This function uses Plotly's `go.Figure` to create and display the plot, which includes interactive elements
        such as zooming and panning to explore the data. Ensure Plotly and its dependencies are properly installed
        and configured in your environment.

    Example:
        Assuming `rfe_pred` and `rfe_err` are DataFrames with the necessary columns,
        call the function as follows:

        plot_RFE(rfe_pred, rfe_err, 50)
    """
    idxmin_prediction = rfe_prediction["mae"].idxmin()
    idxmin_error = rfe_error["mae"].idxmin()

    fig = go.Figure()

    # Adding line plots
    fig.add_trace(go.Scatter(
        x=-rfe_prediction.head(max_iter_show)["n_features"],
        y=rfe_prediction.head(max_iter_show)["mae"],
        mode='lines',
        line=dict(color="blue", width=3),
        name="RFE - Prediction\nContribution"
    ))

    fig.add_trace(go.Scatter(
        x=-rfe_error.head(max_iter_show)["n_features"],
        y=rfe_error.head(max_iter_show)["mae"],
        mode='lines',
        line=dict(color="orange", width=3),
        name="RFE - Error\nContribution"
    ))

    # Adding scatter points for min values
    fig.add_trace(go.Scatter(
        x=[-rfe_prediction.loc[idxmin_prediction, "n_features"]],
        y=[rfe_prediction.loc[idxmin_prediction, "mae"]],
        mode='markers',
        marker=dict(color='red', size=12, line=dict(color='red', width=3)),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[-rfe_error.loc[idxmin_error, "n_features"]],
        y=[rfe_error.loc[idxmin_error, "mae"]],
        mode='markers',
        marker=dict(color='red', size=12, line=dict(color='red', width=3)),
        showlegend=False
    ))

    # Setting titles and labels
    fig.update_layout(
        title="Validation sets averaged over all CV folds",
        xaxis_title="n_Features",
        yaxis_title="Mean Absolute Error",
        legend=dict(x=1.05, y=0.5, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.1)'),
        font=dict(size=12),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Setting plot background to transparent
        xaxis=dict(
            tickmode='array',
            tickvals=rfe_prediction.head(max_iter_show)["n_features"],
            ticktext=[-int(x) for x in rfe_prediction.head(max_iter_show)["n_features"]]
        )
    )

    # Adding grid lines manually since Plotly's grid is on by default
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Show the plot
    fig.show()


def annealing_iter_decision(
    metric_new, metric_curr, subset_new, subset_curr, metric_best, subset_best, T, beta, metric_opt_dir
):
    """
    Decides whether to accept or reject a new subset based on its performance metric, comparing it with the current
    subset's metric, and updates the best known metric and subset if necessary. It uses a simulated annealing approach
    to potentially accept worse solutions to escape local minima.

    Args:
        metric_new (float): The performance metric of the new subset.
        metric_curr (float): The current performance metric.
        subset_new (set): The new set of features.
        subset_curr (set): The current set of features.
        metric_best (float): The best known performance metric.
        subset_best (set): The best known set of features.
        T (float): The current temperature in simulated annealing.
        beta (float): The control parameter for the acceptance probability.
        metric_opt_dir (str): The optimization direction, 'max' for maximization and 'min' for minimization.

    Returns:
        tuple: A tuple containing updated values for metric_curr, subset_curr, metric_best, subset_best, and a status string
               indicating if the new subset was 'Improved', 'Accept'ed despite being worse, or 'Reject'ed. Additionally,
               returns the acceptance probability and random number generated during the decision process, if applicable.

    Nested Functions:
        accept_change(diff, T, beta):
            Evaluates whether a change resulting in a worse metric should be accepted, based on the temperature and a random factor.

            Args:
                diff (float): The difference between the current and new metrics.
                T (float): The current temperature in the annealing process.
                beta (float): The control parameter for scaling the acceptance probability.

            Returns:
                tuple: A tuple indicating if the change is accepted, the acceptance probability, and the generated random number.
    """

    def accept_change(diff, T, beta):
        rnd = np.random.uniform()
        accept_prob = np.exp(-beta * abs(diff) / T)
        if rnd < accept_prob:
            print("%" * 150)
            print(
                f"New subset has worse performance but still accept. Metric change:{diff:8.4f}, Acceptance probability:{accept_prob:6.4f}, Random number:{rnd:6.4f}"
            )
            print("%" * 150)
            return True, accept_prob, rnd
        else:
            print("%" * 150)
            print(
                f"New subset has worse performance, therefore reject. Metric change:{diff:8.4f}, Acceptance probability:{accept_prob:6.4f}, Random number:{rnd:6.4f}"
            )
            print("%" * 150)
            return False, accept_prob, rnd

    improvement = (metric_new > metric_curr) if metric_opt_dir == "max" else (metric_new < metric_curr)
    if improvement:
        print("%" * 150)
        print(f"Local improvement in metric from {metric_curr:8.4f} to {metric_new:8.4f} - New subset accepted")
        print("%" * 150)
        metric_curr = metric_new
        subset_curr = subset_new.copy()
        global_improvement = (metric_new > metric_best) if metric_opt_dir == "max" else (metric_new < metric_best)
        if global_improvement:
            print("%" * 150)
            print(f"Global improvement in metric from {metric_best:8.4f} to {metric_new:8.4f} - Best subset updated")
            print("%" * 150)
            metric_best = metric_new
            subset_best = subset_new.copy()
        return metric_curr, subset_curr, metric_best, subset_best, "Improved", "-", "-"
    else:
        diff = metric_curr - metric_new
        accept, accept_prob, rnd = accept_change(diff, T, beta)
        if accept:
            metric_curr = metric_new
            subset_curr = subset_new.copy()
            return metric_curr, subset_curr, metric_best, subset_best, "Accept", accept_prob, rnd
        else:
            return metric_curr, subset_curr, metric_best, subset_best, "Reject", accept_prob, rnd


class simulated_annealing_cv:
    """
    Implements a simulated annealing algorithm for feature selection aimed at optimizing a given metric over a specified number
    of iterations. The class can dynamically adjust the feature subset based on the performance of a cross-validated model.

    Attributes:
        maxiters (int): Maximum number of iterations for the annealing process.
        min_n_feat_final (int): Minimum number of features to retain in the final selected subset.
        sub_pct_init (float): Initial percentage of features to start with.
        metric_name (str): The performance metric to be optimized during feature selection.
        alpha (float): The cooling rate used in the temperature reduction schedule.
        beta (float): The scaling factor used in calculating the acceptance probability for new subsets.
        T_0 (float): Initial temperature for the simulated annealing process.
        update_iters (int): Number of iterations after which the temperature is updated.
        temp_reduction (str): The method of temperature reduction, options include 'geometric', 'linear', or 'slow decrease'.
        b (float): Parameter used in the 'slow decrease' temperature reduction method to adjust the rate of decrease.
        hash_values (set): A set used to keep track of already visited subsets to avoid revisiting.

    Args:
        maxiters (int): Maximum number of iterations for the annealing process.
        min_n_feat_final (int): Minimum number of features to retain in the final selected subset.
        sub_pct_init (float): Initial percentage of features to start with.
        metric_name (str): The performance metric to be optimized during feature selection.
        alpha (float): The cooling rate used in the temperature reduction schedule.
        beta (float): The scaling factor used in calculating the acceptance probability for new subsets.
        T_0 (float): Initial temperature for the simulated annealing process.
        update_iters (int): Number of iterations after which the temperature is updated.
        temp_reduction (str): The method of temperature reduction, options include 'geometric', 'linear', or 'slow decrease'.
        b (float): Parameter used in the 'slow decrease' temperature reduction method to adjust the rate of decrease.
    """

    def __init__(
        self,
        maxiters: int = 50,
        min_n_feat_final: int = 2,
        sub_pct_init: float = 0.66,
        metric_name: str = "accuracy",
        alpha: float = 0.95,
        beta: float = 1,
        T_0: float = 1,
        update_iters: int = 1,
        temp_reduction: str = "geometric",
        b: float = 5.0,
    ):
        self.maxiters = maxiters
        self.min_n_feat_final = min_n_feat_final
        self.sub_pct_init = sub_pct_init
        self.metric_name = metric_name
        self.alpha = alpha
        self.beta = beta
        self.T_0 = T_0
        self.update_iters = update_iters
        self.temp_reduction = temp_reduction
        self.b = b
        self.hash_values = set()

    def anneal(
        self,
        df: pd.DataFrame,
        label: str = None,
        cv_trainer_params: dict = {},
        fit_kwargs: dict = {},
        predict_kwargs: dict = {},
        metric_list: list[str] = [],
        metric_opt_dir_list: list[str] = [],
        metric_kwargs: dict = {},
    ):
        """
        Performs the annealing process to select an optimal subset of features based on the specified metric.

        Args:
            df (pd.DataFrame): The dataset containing features and a label.
            label (str): Column name of the label in the dataframe.
            cv_trainer_params (dict): Parameters to initialize the cross-validation trainer.
            fit_kwargs (dict): Keyword arguments for the fit method of the model.
            predict_kwargs (dict): Keyword arguments for the predict method of the model.
            metric_list (list[str]): List of metrics to evaluate during the annealing process.
            metric_opt_dir_list (list[str]): List specifying the direction ('min' or 'max') for each metric's optimization.
            metric_kwargs (dict): Additional keyword arguments for metric computations.

        Returns:
            tuple: A tuple containing the dataframe of results from each iteration, the best metric achieved, and the list
                   of best features selected.
        """
        T = self.T_0
        columns = [
            "Iteration",
            "Feature Count",
            "Feature Set",
            "Metric",
            "Best Metric",
            "Acceptance Probability",
            "Random Number",
            "Outcome",
        ]
        results = pd.DataFrame(index=range(self.maxiters), columns=columns)

        X_train = df.copy().drop(columns=label)
        Y_train = df.copy()[label]
        full_set = set(np.arange(len(df.columns) - 1))

        # Generate initial random subset based on ~(self.sub_pct_init)% of columns
        subset_curr = set(random.sample(list(full_set), round(self.sub_pct_init * len(full_set))))
        subset_best = subset_curr
        X_curr = X_train.iloc[:, list(subset_curr)]

        print("%" * 150)
        logging.info(f"The initial set of features are {X_curr.columns}")
        print(f"The initial set of features are {X_curr.columns}")
        print("%" * 150)

        df_curr = pd.concat([X_curr, Y_train], axis=1)
        cv_trainer = cv_training(**cv_trainer_params)
        fit_kwargs_curr = {
            **get_fit_cat_params(cv_trainer.estimator.__name__, cat_col_list=get_cat_features(df_curr, label)),
            **fit_kwargs,
        }
        cv_trainer.fit(
            df_curr,
            label=label,
            fit_kwargs=fit_kwargs_curr,
            predict_kwargs=predict_kwargs,
            metric_list=metric_list,
            metric_opt_dir_list=metric_opt_dir_list,
            metric_kwargs=metric_kwargs,
        )
        metric_curr = cv_trainer.metrics_stats[self.metric_name]["final"]
        metric_best = metric_curr

        for i in tqdm(range(self.maxiters)):
            print("%" * 150)
            logging.info(f"Starting iteration {i+1}")
            print(f"Starting iteration {i+1}")
            print("%" * 150)

            if T < 0.01:
                print(f"Temperature {T} below threshold. Termination condition met")
                break

            while True:
                if len(subset_curr) == len(full_set):
                    move = "Remove"
                elif len(subset_curr) == self.min_n_feat_final:  # Not to go below (self.min_n_feat_final) features
                    move = random.choice(["Add", "Replace"])
                else:
                    move = random.choice(["Add", "Replace", "Remove"])

                pending_cols = full_set.difference(subset_curr)
                subset_new = subset_curr.copy()

                if move == "Add":
                    subset_new.add(random.choice(list(pending_cols)))
                elif move == "Replace":
                    subset_new.remove(random.choice(list(subset_curr)))
                    subset_new.add(random.choice(list(pending_cols)))
                else:
                    subset_new.remove(random.choice(list(subset_curr)))

                if subset_new in self.hash_values:
                    print("Subset already visited, trying to get a new subset of features for this iteration.")
                else:
                    self.hash_values.add(frozenset(subset_new))
                    break

            X_new = X_train.iloc[:, list(subset_new)]
            df_new = pd.concat([X_new, Y_train], axis=1)
            cv_trainer = cv_training(**cv_trainer_params)
            fit_kwargs_new = {
                **get_fit_cat_params(cv_trainer.estimator.__name__, cat_col_list=get_cat_features(df_new, label)),
                **fit_kwargs,
            }
            cv_trainer.fit(
                df_new,
                label=label,
                fit_kwargs=fit_kwargs_new,
                predict_kwargs=predict_kwargs,
                metric_list=metric_list,
                metric_opt_dir_list=metric_opt_dir_list,
                metric_kwargs=metric_kwargs,
            )

            print("%" * 150)
            logging.info(f"The new set of features are {X_new.columns}")
            print(f"The new set of features are {X_new.columns}")
            print("%" * 150)

            metric_new = cv_trainer.metrics_stats[self.metric_name]["final"]
            # Make decision based on metric_new and metric_curr
            metric_curr, subset_curr, metric_best, subset_best, outcome, accept_prob, rnd = annealing_iter_decision(
                metric_new,
                metric_curr,
                subset_new,
                subset_curr,
                metric_best,
                subset_best,
                T,
                self.beta,
                metric_opt_dir_list[0],
            )
            results.loc[i, "Iteration"] = i + 1
            results.loc[i, "Feature Count"] = len(subset_curr)
            results.loc[i, "Feature Set"] = sorted(subset_curr)
            results.loc[i, "Metric"] = metric_curr
            results.loc[i, "Best Metric"] = metric_best
            results.loc[i, "Acceptance Probability"] = accept_prob
            results.loc[i, "Random Number"] = rnd
            results.loc[i, "Outcome"] = outcome

            # Temperature cooling schedule
            if i % self.update_iters == 0:
                if self.temp_reduction == "geometric":
                    T = self.alpha * T
                elif self.temp_reduction == "linear":
                    T -= self.alpha
                elif self.temp_reduction == "slow decrease":
                    T = T / (1 + self.b * T)
                else:
                    raise Exception("Temperature reduction strategy not recognized")
            print("%" * 150)
            logging.info(f"Finished iteration {i+1}")
            print(f"Finished iteration {i+1}")
            print("%" * 150)

        best_subset_cols = [list(X_train.columns)[i] for i in list(subset_best)]
        results = results.dropna(axis=0, how="all")

        return results, metric_best, best_subset_cols
