import numpy as np
import pandas as pd

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin


class DoubleValidationEncoderNumerical(BaseEstimator, TransformerMixin):
    """
    Encoder with validation within
    """

    def __init__(self, encoder_name: str):
        """
        :param encoder_name: Name of encoder
        """
        self.cat_cols = None
        self.encoder_name = encoder_name

        self.n_folds = 5
        self.model_validation = StratifiedKFold(n_splits=self.n_folds, random_state=42)
        self.encoders_dict = []

    def fit(self, X: pd.DataFrame, y: np.array):
        self.cat_cols = [col for col in X.columns if X[col].dtype == "O"]
        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
            encoder = get_single_encoder(self.encoder_name, self.cols)
            X_train, y_train = X.loc[train_idx], y[train_idx]
            encoder.fit(X_train, y_train)
            self.encoders_dict[self.encoder_name].append(encoder)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.encoders_list:
            raise RuntimeError("The encoder has not been fitted yet.")
        # Initialize an empty DataFrame to accumulate weighted averages
        X_encoded_sum = pd.DataFrame(index=X.index, columns=self.cat_cols).fillna(0.0)

        # Apply each fold's encoder and update the cumulative average
        fold_count = 0
        for encoder in self.encoders_list:
            X_encoded = encoder.transform(X)[self.cat_cols]
            # Cumulative moving average update
            X_encoded_sum += (X_encoded - X_encoded_sum) / (fold_count + 1)
            fold_count += 1

        # Replace original categorical columns with their encoded values
        X[self.cat_cols] = X_encoded_sum
        return X


class MultipleEncoder(BaseEstimator, TransformerMixin):
    """
    Multiple encoder for categorical columns
    """

    def __init__(self, encoder_name: str):
        """
        :param encoder_name: Name of encoder. Possible values are:
        "WOEEncoder", "TargetEncoder", "SumEncoder", "MEstimateEncoder", "LeaveOneOutEncoder",
        "HelmertEncoder", "BackwardDifferenceEncoder", "JamesSteinEncoder", "OrdinalEncoder""CatBoostEncoder"
        """

        self.cat_cols = None
        self.encoder_name = encoder_name
        self.encoder = None

    def fit(self, X: pd.DataFrame, y: np.array) -> None:
        self.cat_cols = [col for col in X.columns if X[col].dtype == "O"]
        encoder = get_single_encoder(encoder_name=self.encoder_name, cat_cols=self.cat_cols)
        encoder.fit(X, y)
        self.encoder = encoder

        return self

    def transform(self, X) -> pd.DataFrame:
        X_encoded = self.encoder.transform(X)
        return X_encoded


def get_single_encoder(encoder_name: str, cat_cols: list):
    """
    Get encoder by its name
    :param encoder_name: Name of desired encoder
    :param cat_cols: Cat columns for encoding
    :return: Categorical encoder
    """
    encoder_classes = {
        "WOEEncoder": WOEEncoder,
        "TargetEncoder": TargetEncoder,
        "SumEncoder": SumEncoder,
        "MEstimateEncoder": MEstimateEncoder,
        "LeaveOneOutEncoder": LeaveOneOutEncoder,
        "HelmertEncoder": HelmertEncoder,
        "BackwardDifferenceEncoder": BackwardDifferenceEncoder,
        "JamesSteinEncoder": JamesSteinEncoder,
        "OrdinalEncoder": OrdinalEncoder,
        "CatBoostEncoder": CatBoostEncoder,
        "OneHotEncoder": OneHotEncoder,
    }

    encoder_class = encoder_classes.get(encoder_name)
    if encoder_class:
        return encoder_class(cols=cat_cols)
    else:
        raise ValueError(f"Encoder name '{encoder_name}' is not supported.")


def read_data(local_data_dir: str, label: str) -> pd.DataFrame:
    df_train = pd.read_csv(local_data_dir + "/train.csv")
    X = df_train.drop(columns=label)
    Y = df_train[label]
    df_test = pd.read_csv(local_data_dir + "/test.csv")
    return X, Y, df_test


def save_data(train: pd.DataFrame, test: pd.DataFrame, local_save_dir: str) -> None:
    train.to_csv(local_save_dir + "/train.csv", index=False)
    test.to_csv(local_save_dir + "/test.csv", index=False)
