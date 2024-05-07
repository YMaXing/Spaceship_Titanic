from typing import Any, Optional
import yaml
import logging
import logging.config

import hydra
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf

from src.config_schemas import config_schema
from src.config_schemas.data import get_raw_data_config_schema
from src.config_schemas.features import imputation_config_schema, encoding_config_schema, outlier_config_schema, feature_engineering_config_schema, feature_selection_config_schema
from src.config_schemas.models import HPT_config_schema


def get_config(config_path: str, config_name: str) -> TaskFunction:
    setup_config()
    setup_logger()

    def main_decorator(task_function: TaskFunction) -> Any:
        @hydra.main(config_path=config_path, config_name=config_name, version_base=None)
        def decorated_main(dict_config: Optional[DictConfig] = None) -> Any:
            config = OmegaConf.to_object(dict_config)
            return task_function(config)

        return decorated_main

    return main_decorator


def setup_config() -> None:
    get_raw_data_config_schema.setup_config()
    imputation_config_schema.setup_config()
    encoding_config_schema.setup_config()
    outlier_config_schema.setup_config()
    feature_engineering_config_schema.setup_config()
    feature_selection_config_schema.setup_config()
    HPT_config_schema.setup_config()
    config_schema.setup_config()


def setup_logger() -> None:
    with open("./src/configs/hydra/job_logging/custom.yaml", "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)
