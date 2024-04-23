import pandas as pd
import numpy as np
import logging

from src.utils.encoding_utils import read_data, save_data
from src.utils.config_utils import get_config
from src.config_schemas.features.encoding_config_schema import encoding_Config
from src.features.encoding.encoding import Model


@get_config(config_path="../configs/features", config_name="encoding_config")
def encoding(config: encoding_Config) -> None:
    df_train_X, df_train_Y, df_test = read_data(config.local_data_dir, config.label)
    logging.info("Data read successfully.")

    model = Model(encoder_name=config.encoder_name,
                  cat_cols=config.cat_cols,
                  cat_validation=config.cat_validation)

    logging.info("Start encoding the training set.")
    df_train_X = model.fit_transform(df_train_X, df_train_Y)
    logging.info("Finished encoding the training set.")
    df_train = pd.concat([df_train_X, df_train_Y], axis=1)
    logging.info("Start encoding the test set.")
    df_test = model.transform(df_test)
    logging.info("Finished encoding the test set.")

    if config.encoder_name == "CatBoostEncoder":
        save_dir = config.local_save_dir + "/CatBoost"
    elif config.encoder_name == "WOEEnconder":
        save_dir = config.local_save_dir + "/WOE"
    else:
        ValueError("Encoder not supported")

    save_data(df_train, df_test, save_dir)
    logging.info("Encoded data saved successfully.")


if __name__ == "__main__":
    encoding()  # type: ignore
