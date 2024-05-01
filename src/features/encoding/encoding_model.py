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
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y[train_idx]
            print(f"shapes before encoder: {X_train.shape}, {X_val.shape}")

            if self.cat_validation == "Single":
                encoder = MultipleEncoder(encoder_name=self.encoder_name)
                X_train = encoder.fit(X_train, y_train)
                X_val = encoder.transform(X_val)
            if self.cat_validation == "Double":
                encoder = DoubleValidationEncoderNumerical(encoder_name=self.encoder_name)
                X_train = encoder.fit(X_train, y_train)
                X_val = encoder.transform(X_val)

            X_val_list.append(X_val)
            self.encoders_list.append(encoder)

        X = pd.concat(X_val_list, axis=0).sort_index()
        self.features_after_encoder = X.columns.tolist()
        print(self.features_after_encoder)
        return X

    def transform(self, X: pd.DataFrame) -> np.array:
        if not self.encoders_list:
            raise RuntimeError("The encoder has not been fitted yet.")

        # Apply each fold's encoder and update the cumulative average
        fold_count = 0
        for encoder in self.encoders_list:
            X_encoded = encoder.transform(X).drop(columns=encoder.num_cols)
            # Initialize an empty DataFrame to accumulate weighted averages
            if fold_count == 0:
                X_encoded_sum = pd.DataFrame(index=X.index, columns=X_encoded.columns).fillna(0.0)
            # Cumulative moving average update
            X_encoded_sum += (X_encoded - X_encoded_sum) / (fold_count + 1)
            fold_count += 1

        return pd.concat([X_encoded_sum, X.drop(columns=encoder.cat_cols)], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return self.features_after_encoder
