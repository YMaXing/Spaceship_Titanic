import pandas as pd
from src.utils.config_utils import get_config
from src.config_schemas.features.feature_selection_config_schema import feature_selection_Config
import logging
from src.utils.feature_selection_utils import read_data, save_data, select_feature


@get_config(config_path="../configs/features", config_name="feature_selection_config")
def feature_selection(config: feature_selection_Config) -> None:

    logging.info("Feature selection started for unencoded data")
    # Load unencoded data
    unencoded_train, unencoded_test = read_data(config.local_unencoded_dir)
    # Feature selection for unencoded data
    selected_features_unencoded_train = select_feature(unencoded_train, config.unencoded_features + config.label)
    selected_features_unencoded_test = select_feature(unencoded_test, config.unencoded_features)
    # Save unencoded data
    save_data(selected_features_unencoded_train, selected_features_unencoded_test, config.local_save_dir)
    logging.info("Feature selection ended for unencoded data")

    logging.info("Feature selection started for encoded data")
    for encoding in config.encoded:
        # Load encoded data
        encoded_train, encoded_test = read_data(config.local_encoded_dir + f"/{encoding}")
        # Feature selection for encoded data
        selected_features_encoded_train = select_feature(encoded_train, getattr(config, f"{encoding}_features") + config.label)
        selected_features_encoded_test = select_feature(encoded_test, getattr(config, f"{encoding}_features"))
        # Save encoded data
        save_data(selected_features_encoded_train, selected_features_encoded_test, config.local_save_dir + f"/{encoding}")
    logging.info("Feature selection ended for encoded data")

if __name__ == "__main__":
    feature_selection()  # type: ignore
