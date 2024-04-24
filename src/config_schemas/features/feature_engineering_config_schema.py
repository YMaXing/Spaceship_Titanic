from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class feature_engineering_Config:
    local_data_dir = "data/imputed"
    local_save_dir = "data/engineered"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="feature_engineering_config_schema", node=feature_engineering_Config)
