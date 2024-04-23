from abc import abstractmethod
import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier, CatBoostRegressor
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import hamming_loss, mean_squared_error

# Visualization
import matplotlib.pyplot as plt

from src.utils.imputation_utils import missing_index, get_features, fill_placeholder


class imputer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def __init__() -> None:
        pass

    @abstractmethod
    def fit(self, train: pd.DataFrame):
        pass
        return self

    @abstractmethod
    def transform(self, test: pd.DataFrame):
        pass
        return test


class iter_cv_catboost_imputer(imputer):
    """
    Note that dataset used to train the imputer and the dataset to be imputed must have exactly the same features.

    Always first train the imputer before imputing another dataset using "transform" method

    """

    def __init__(
        self,
        label: str = None,
        max_iter: int = 10,
        n_splits: int = 5,
    ):
        self.label = label
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.cat_features = None
        self.num_features = None
        self.imp_features = None
        self.cat_imp_features = None
        self.num_imp_features = None
        self.estimators = None
        self.label_encoders = None
        self.errors = None

    def fit(self, train: pd.DataFrame):
        logging.info("Starting the fitting process.")
        # Get all the categorical and numerical features respectively in the training set
        self.cat_features = [
            feature for feature in train.columns if train[feature].dtype == "O" and feature != self.label
        ]
        self.num_features = [
            feature for feature in train.columns if train[feature].dtype != "O" and feature != self.label
        ]
        # Get features to be imputed
        self.cat_imp_features = get_features(train, missing_type="cat", label=self.label)
        self.num_imp_features = get_features(train, missing_type="num", label=self.label)
        self.imp_features = self.cat_imp_features + self.num_imp_features
        logging.info(f"Identified categorical features to impute: {self.cat_imp_features}")
        logging.info(f"Identified numerical features to impute: {self.num_imp_features}")
        # Get the indices of the rows with missing values for every feature to be imputed
        missing_rows = missing_index(train, self.imp_features)

        # First, fill the missing values with the appropriate placeholder
        train = fill_placeholder(df=train, features=self.imp_features)

        # Second, we drop the label (the label for our ML task, not the label in each imputation iteration) from the dataset
        train = train.drop(columns=self.label)

        # Initialize the estimator, error and label encoder dictionaries to save the error/difference in each iteration to monitor our training process
        estimators = {feature: [] for feature in self.imp_features}
        errors = {feature: [] for feature in self.imp_features}
        label_encoders = {feature: [] for feature in self.cat_imp_features}

        # Third, we start training and then impute the missing values in the training set iteratively
        for i in tqdm(range(self.max_iter), desc="Iterations"):
            for feature in self.imp_features:

                logging.info(f"Starting iteration {i+1} for feature: {feature}")
                # Set up imputation training and test sets specifically for the current feature
                test_curr = train.iloc[missing_rows[feature]]
                train_curr = train.drop(index=missing_rows[feature])
                # Define the imputation label as the current feature to be imputed and the features (especially the categorical features) to be the remaining features
                label_curr = feature
                features_curr = [feature_curr for feature_curr in self.imp_features if feature_curr != label_curr]
                cat_features_curr = [
                    feature_curr for feature_curr in self.cat_imp_features if feature_curr != label_curr
                ]
                # Define X and Y in train_curr for the current feature, apply label encoding to Y if label_curr is categorical
                X_curr = train_curr.loc[:, features_curr]
                Y_curr = train_curr.loc[:, label_curr]
                if label_curr in self.cat_imp_features:
                    label_encoder = LabelEncoder()
                    Y_curr = pd.Series(label_encoder.fit_transform(Y_curr), index=Y_curr.index)
                # Define X in test_curr for the current feature which will be used in prediction
                X_test_curr = test_curr.loc[:, features_curr]

                # Select the appropriate cross-validation technique
                if feature in self.cat_imp_features:
                    cv = StratifiedKFold(n_splits=self.n_splits)
                else:
                    cv = KFold(n_splits=self.n_splits)
                # Initialize the list of the fitted estimator from different cv folds for the current feature in the current iteration
                estimators_curr = []
                # Start cross-validation training
                for train_index, _ in cv.split(X_curr, Y_curr):

                    logging.info(f"Starting new cv fold for feature: {label_curr}")
                    # Define the X and Y for both training folds (the test fold is useless here)
                    X_train, Y_train = X_curr.iloc[train_index], Y_curr.iloc[train_index]
                    # Define Catboost estimator based on the type of the current feature to be imputed)
                    estimator_curr = (
                        CatBoostClassifier()
                        if label_curr in self.cat_imp_features
                        else CatBoostRegressor()
                    )
                    # Fit the Catboost estimator and append it to the list "estimators_curr"
                    estimator_curr.fit(X_train, Y_train, cat_features=cat_features_curr, verbose=False)
                    estimators_curr.append(estimator_curr)
                    logging.info(f"Completed new cv fold for feature: {label_curr}")
                # Predict the missing values of the current feature to be impute by first averaging the prediction from different folds
                Y_pred = np.mean(
                    np.array(
                        [
                            (
                                estimator.predict_proba(X_test_curr)
                                if label_curr in self.cat_imp_features
                                else estimator.predict(X_test_curr)
                            )
                            for estimator in estimators_curr
                        ]
                    ),
                    axis=0,
                )
                # If the feature to be imputed is categorical, we need to pick the category with the highest average probability as our prediction
                if label_curr in self.cat_imp_features:
                    Y_pred = pd.Series(
                        label_encoder.inverse_transform(np.argmax(Y_pred, axis=1)), index=test_curr.index
                    )
                # Fill the prediction into the training set
                Y_pred_prev = test_curr.loc[:, label_curr]
                train.loc[missing_rows[label_curr], label_curr] = Y_pred
                logging.info(f"Updated feature '{feature}' with imputed values.")
                # Calculate the difference between the imputed values of label_curr from two consecutive iterations
                if i > 0:
                    (
                        errors[label_curr].append(
                            hamming_loss(Y_pred_prev.astype("category"), Y_pred.astype("category"))
                        )
                        if label_curr in self.cat_imp_features
                        else errors[label_curr].append(np.sqrt(mean_squared_error(Y_pred_prev, Y_pred)))
                    )
                # Finally, we need to save all the label encoders and the estimators in the final iteration so that we can use it to impute the other dataset, say the validation and the test set of ML task
                if i == self.max_iter - 1:
                    estimators[label_curr] = estimators_curr
                    if label_curr in self.cat_imp_features:
                        label_encoders[label_curr] = label_encoder
                logging.info("Completed the fitting process.")

        self.errors = errors
        self.estimators = estimators
        self.label_encoders = label_encoders

        return self

    def plot_training_error(self):
        for feature, values in self.errors.items():
            iterations = range(1, len(values) + 1)  # x-axis values (iterations)
            plt.plot(iterations, values, label=feature)  # plot the values
            plt.xlabel("Iterations")
            plt.ylabel("Errors")
            plt.title(
                "Minimization of Error (Hamming loss for categorical features and RMSE for numerical features) with iterations"
            )
            plt.legend()
            plt.show()

    def transform(self, test: pd.DataFrame):
        # First, ensure the test set has the exact same categorical and numerical features as the training set
        test_cat_features = [
            feature for feature in test.columns if test[feature].dtype == "O" and feature != self.label
        ]
        test_num_features = [
            feature for feature in test.columns if test[feature].dtype != "O" and feature != self.label
        ]

        if set(test_cat_features) != set(self.cat_features) or set(test_num_features) != set(self.num_features):
            raise ValueError(
                "The features in the test set do NOT match those in the training set. Check both categorical and numerical features."
            )

        # Second, get the indices of the rows with missing values for every feature to be imputed
        missing_rows = missing_index(test, self.imp_features)

        # Third, fill the missing values with the appropriate placeholder
        test = fill_placeholder(df=test, features=self.imp_features)

        # Finally, we start to fill the missing values with the saved estimators we obtained from training
        for feature in self.imp_features:
            if missing_rows[feature].any():
                X_curr = test.loc[missing_rows[feature], [f for f in self.imp_features if f != feature]]
                if feature in self.cat_imp_features:
                    predictions = np.array([estimator.predict_proba(X_curr) for estimator in self.estimators[feature]])
                    Y_curr_pred = np.mean(predictions, axis=0)
                    Y_curr_pred = self.label_encoders[feature].inverse_transform(np.argmax(Y_curr_pred, axis=1))
                else:
                    predictions = np.array([estimator.predict(X_curr) for estimator in self.estimators[feature]])
                    Y_curr_pred = np.mean(predictions, axis=0)

            test.loc[missing_rows[feature], feature] = Y_curr_pred

        return test

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.fit(train)
        return train
