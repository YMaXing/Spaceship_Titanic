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
from category_encoders.frequency import FrequencyEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin


class DoubleValidationEncoderNumerical(BaseEstimator, TransformerMixin):
    """
    Encoder with validation within
    """

    def __init__(self, cols, encoders_names_tuple=()):
        """
        :param cols: Categorical columns
        :param encoders_names_tuple: Tuple of str with encoders
        """
        self.cols, self.num_cols = cols, None
        self.encoders_names_tuple = encoders_names_tuple

        self.n_folds = 5
        self.model_validation = StratifiedKFold(n_splits=self.n_folds, random_state=42)
        self.encoders_dict = {}

        self.storage = None

    def fit_transform(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.num_cols = [col for col in X.columns if col not in self.cols]
        self.storage = []

        for encoder_name in self.encoders_names_tuple:
            for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
                encoder = get_single_encoder(encoder_name, self.cols)

                X_train, y_train = X.loc[train_idx], y[train_idx]
                encoder.fit(X_train, y_train)

                if encoder_name not in self.encoders_dict.keys():
                    self.encoders_dict[encoder_name] = [encoder]
                else:
                    self.encoders_dict[encoder_name].append(encoder)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for encoder_name in self.encoders_names_tuple:
            for encoder in self.encoders_dict[encoder_name]:
                test_tr = encoder.transform(X)
                test_tr = test_tr[[col for col in test_tr.columns if col not in self.num_cols]].values

                if cols_representation is None:
                    cols_representation = np.zeros(test_tr.shape)

                cols_representation = cols_representation + test_tr / self.n_folds / self.n_repeats

            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [f"encoded_{encoder_name}_{i}" for i in range(cols_representation.shape[1])]
            self.storage.append(cols_representation)

        for df in self.storage:
            X = pd.concat([X, df], axis=1)

        X.drop(self.cols, axis=1, inplace=True)
        return X


class MultipleEncoder(BaseEstimator, TransformerMixin):
    """
    Multiple encoder for categorical columns
    """

    def __init__(self, cols: list[str], encoders_names_tuple=()):
        """
        :param cols: List of categorical columns
        :param encoders_names_tuple: Tuple of categorical encoders names. Possible values in tuple are:
        "WOEEncoder", "TargetEncoder", "SumEncoder", "MEstimateEncoder", "LeaveOneOutEncoder",
        "HelmertEncoder", "BackwardDifferenceEncoder", "JamesSteinEncoder", "OrdinalEncoder""CatBoostEncoder"
        """

        self.cols = cols
        self.num_cols = None
        self.encoders_names_tuple = encoders_names_tuple
        self.encoders_dict = {}

        # list for storing results of transformation from each encoder
        self.storage = None

    def fit_transform(self, X: pd.DataFrame, y: np.array) -> None:
        self.num_cols = [col for col in X.columns if col not in self.cols]
        self.storage = []
        for encoder_name in self.encoders_names_tuple:
            encoder = get_single_encoder(encoder_name=encoder_name, cat_cols=self.cols)

            cols_representation = encoder.fit_transform(X, y)
            self.encoders_dict[encoder_name] = encoder
            cols_representation = cols_representation[
                [col for col in cols_representation.columns if col not in self.num_cols]
            ].values
            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [f"encoded_{encoder_name}_{i}" for i in range(cols_representation.shape[1])]
            self.storage.append(cols_representation)

        # concat cat cols representations with initial dataframe
        for df in self.storage:
            print(df.shape)
            X = pd.concat([X, df], axis=1)

        # remove all columns as far as we have their representations
        X.drop(self.cols, axis=1, inplace=True)
        return X

    def transform(self, X) -> pd.DataFrame:
        self.storage = []
        for encoder_name in self.encoders_names_tuple:
            # get representation of cat columns and form a pd.DataFrame for it
            cols_representation = self.encoders_dict[encoder_name].transform(X)
            cols_representation = cols_representation[
                [col for col in cols_representation.columns if col not in self.num_cols]
            ].values
            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [f"encoded_{encoder_name}_{i}" for i in range(cols_representation.shape[1])]
            self.storage.append(cols_representation)

        # concat cat cols representations with initial dataframe
        for df in self.storage:
            print(df.shape)
            X = pd.concat([X, df], axis=1)

        # remove all columns as far as we have their representations
        X.drop(self.cols, axis=1, inplace=True)
        return X


def get_single_encoder(encoder_name: str, cat_cols: list):
    """
    Get encoder by its name
    :param encoder_name: Name of desired encoder
    :param cat_cols: Cat columns for encoding
    :return: Categorical encoder
    """
    if encoder_name == "FrequencyEncoder":
        encoder = FrequencyEncoder(cols=cat_cols)

    if encoder_name == "WOEEncoder":
        encoder = WOEEncoder(cols=cat_cols)

    if encoder_name == "TargetEncoder":
        encoder = TargetEncoder(cols=cat_cols)

    if encoder_name == "SumEncoder":
        encoder = SumEncoder(cols=cat_cols)

    if encoder_name == "MEstimateEncoder":
        encoder = MEstimateEncoder(cols=cat_cols)

    if encoder_name == "LeaveOneOutEncoder":
        encoder = LeaveOneOutEncoder(cols=cat_cols)

    if encoder_name == "HelmertEncoder":
        encoder = HelmertEncoder(cols=cat_cols)

    if encoder_name == "BackwardDifferenceEncoder":
        encoder = BackwardDifferenceEncoder(cols=cat_cols)

    if encoder_name == "JamesSteinEncoder":
        encoder = JamesSteinEncoder(cols=cat_cols)

    if encoder_name == "OrdinalEncoder":
        encoder = OrdinalEncoder(cols=cat_cols)

    if encoder_name == "CatBoostEncoder":
        encoder = CatBoostEncoder(cols=cat_cols)

    if encoder_name == "MEstimateEncoder":
        encoder = MEstimateEncoder(cols=cat_cols)
    return encoder


def read_data(local_data_dir: str, label: str) -> pd.DataFrame:
    df_train = pd.read_csv(local_data_dir + "/train.csv")
    X = df_train.drop(columns=label)
    Y = df_train[label]
    df_test = pd.read_csv(local_data_dir + "/test.csv")
    return X, Y, df_test


def save_data(train: pd.DataFrame, test: pd.DataFrame, local_save_dir: str) -> None:
    train.to_csv(local_save_dir + "/train.csv")
    test.to_csv(local_save_dir + "/test.csv")