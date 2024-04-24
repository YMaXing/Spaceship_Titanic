import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.encoding_utils import MultipleEncoder, DoubleValidationEncoderNumerical


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cat_validation=None,
        encoder_name=None,
        model_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ):
        self.cat_validation = cat_validation
        self.encoder_name = encoder_name
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
                encoder = MultipleEncoder(encoder_name=self.encoder_name)
                X_train = encoder.fit(X_train, y_train)
                X_val = encoder.transform(X_val)
            if self.cat_validation == "Double":
                encoder = DoubleValidationEncoderNumerical(encoder_name=self.encoder_name)
                X_train = encoder(X_train, y_train)
                X_val = encoder.transform(X_val)

            X_val_list.append(X_val)
            self.encoders_list.append(encoder)

        X = pd.concat(X_val_list, axis=0).sort_index()
        return X

    def predict(self, X: pd.DataFrame) -> np.array:
        if not self.encoders_list:
            raise RuntimeError("The encoder has not been fitted yet.")
        # Initialize an empty DataFrame to accumulate weighted averages
        X_encoded_sum = pd.DataFrame(index=X.index, columns=self.cat_cols).fillna(0.0)

        # Apply each fold's encoder and update the cumulative average
        fold_count = 0
        for encoder in self.encoders_list:
            X_encoded = encoder.transform(X)[encoder.cat_cols]
            # Cumulative moving average update
            X_encoded_sum += (X_encoded - X_encoded_sum) / (fold_count + 1)
            fold_count += 1

        # Replace original categorical columns with their encoded values
        X[encoder.cat_cols] = X_encoded_sum
        return X
