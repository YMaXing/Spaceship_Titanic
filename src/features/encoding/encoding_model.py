import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

from src.utils.encoding_utils import MultipleEncoder, DoubleValidationEncoderNumerical
from sklearn.base import BaseEstimator, TransformerMixin


class Model(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cat_validation="Double",
        encoders_name=None,
        cat_cols=None,
        model_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ):
        self.cat_validation = cat_validation
        self.encoders_name = encoders_name
        self.cat_cols = cat_cols
        self.model_validation = model_validation

        self.encoders_list = []

    def fit_transform(self, X: pd.DataFrame, y: np.array) -> None:
        # process cat cols
        X_val_list = []
        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train = y[train_idx]
            print(f"shapes before encoder: {X_train.shape}, {X_val.shape}")

            if self.cat_validation == "Single":
                encoder = MultipleEncoder(cols=self.cat_cols, encoders_name=self.encoders_name)
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val)
            if self.cat_validation == "Double":
                encoder = DoubleValidationEncoderNumerical(cols=self.cat_cols, encoders_name=self.encoders_name)
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val)
                pass
            X_val_list.append(X_val)
            self.encoders_list.append(encoder)

        X = pd.concat(X_val_list, axis=0).sort_index()
        return X

    def predict(self, X: pd.DataFrame) -> np.array:
        y_hat = np.zeros(X.shape[0])
        for encoder, model in zip(self.encoders_list, self.models_list):
            X_test = X.copy()
            X_test = encoder.transform(X_test)

            # check for OrdinalEncoder encoding
            for col in [col for col in X_test.columns if "OrdinalEncoder" in col]:
                X_test[col] = X_test[col].astype("category")

            unranked_preds = model.predict_proba(X_test)[:, 1]
            y_hat += rankdata(unranked_preds)
        return y_hat, X_test.shape[1]
