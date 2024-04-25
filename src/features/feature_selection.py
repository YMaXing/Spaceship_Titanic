import pandas as pd
from src.utils.config_utils import get_config
from src.config_schemas.features.feature_selection_config_schema import feature_selection_Config
import logging


@get_config(config_path="../configs/features", config_name="feature_selection_config")
def feature_selection(config: feature_selection_Config) -> None:
    pass


if __name__ == "__main__":
    feature_selection()  # type: ignore
