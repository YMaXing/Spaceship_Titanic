from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class feature_selection_Config:
    local_data_dir = "data/outlier_removed"
    local_save_dir = "data/engineered"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="feature_selection_config_schema", node=feature_selection_Config)
