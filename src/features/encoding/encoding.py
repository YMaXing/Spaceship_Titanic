from json import encoder
import pandas as pd
import logging
from pathlib import Path

from src.utils.encoding_utils import read_data, save_data
from src.utils.config_utils import get_config
from src.config_schemas.features.encoding_config_schema import encoding_Config
from src.features.encoding.encoding_model import Encoder
from sklearn.compose import ColumnTransformer


@get_config(config_path="../configs/features", config_name="encoding_config")
def preliminary_encoding(config: encoding_Config) -> None:
    """
    Supported encoder names: "WOEEncoder", "CatBoostEncoder", "MEstimateEncoder", "OneHotEncoder"
    """
    df_train_X, df_train_Y, df_test = read_data(config.local_data_dir, config.label)
    logging.info("Data read successfully.")

    encoder = Encoder(encoder_name=config.encoder_name, cat_validation=config.cat_validation)

    logging.info("Start encoding the training set.")
    df_train_X = encoder.fit_transform(df_train_X, df_train_Y)
    logging.info("Finished encoding the training set.")
    df_train = pd.concat([df_train_X, pd.Series(df_train_Y, name=config.label)], axis=1)
    logging.info("Start encoding the test set.")
    df_test = encoder.transform(df_test)
    logging.info("Finished encoding the test set.")

    if config.encoder_name == "CatBoostEncoder":
        save_dir = config.local_save_dir + "/CatBoost"
    elif config.encoder_name == "WOEEncoder":
        save_dir = config.local_save_dir + "/WOE"
    elif config.encoder_name == "MEstimateEncoder":
        save_dir = config.local_save_dir + "/MEstimate"
    elif config.encoder_name == "OneHotEncoder":
        save_dir = config.local_save_dir + "/One-Hot"
    else:
        ValueError("Encoder not supported")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Successfully created directory {save_dir}")
    else:
        logging.info(f"Directory {save_dir} already exists")
    save_data(df_train, df_test, save_dir)
    logging.info("Encoded data saved successfully.")


@get_config(config_path="../configs/features", config_name="encoding_config")
def encoding(config: encoding_Config) -> None:
    df_train_X, df_train_Y, df_test = read_data(config.local_data_dir, config.label)
    logging.info("Data read successfully.")

    encoders = {
        name: Encoder(encoder_name=name, cat_validation=config.cat_validation) for name in config.supported_encoders
    }

    # Creating a list of tuples for ColumnTransformer
    transformers = [(name, encoders[name], getattr(config, f"{name}_cols")) for name in config.supported_encoders]

    # Using ColumnTransformer to apply the encoders
    mixed_encoder = ColumnTransformer(transformers=transformers, remainder="passthrough")

    logging.info("Start encoding the training set.")
    df_train_X_transformed = mixed_encoder.fit_transform(df_train_X, df_train_Y)
    print(mixed_encoder.get_feature_names_out())
    df_train_X = pd.DataFrame(
        df_train_X_transformed, columns=[name.split("__")[-1] for name in mixed_encoder.get_feature_names_out()]
    )

    df_train = pd.concat([df_train_X, pd.Series(df_train_Y, name=config.label)], axis=1)
    logging.info("Finished encoding the training set.")

    logging.info("Start encoding the test set.")
    df_test_transformed = mixed_encoder.transform(df_test)
    df_test = pd.DataFrame(
        df_test_transformed, columns=[name.split("__")[-1] for name in mixed_encoder.get_feature_names_out()]
    )
    logging.info("Finished encoding the test set.")

    save_dir = Path(config.local_save_dir + "/Mixed")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Successfully created directory {save_dir}")
    else:
        logging.info(f"Directory {save_dir} already exists")
    save_data(df_train, df_test, save_dir)
    logging.info("Encoded data saved successfully.")


# Ensure proper logging setup:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preliminary_encoding()
