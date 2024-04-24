from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class encoding_Config:
    local_data_dir = "data/engineered"
    local_save_dir = "data/encoded"
    encoder_name: str = MISSING
    cat_validation: str = "Double"
    label: str = "Transported"


def setup_config() -> None:

    cs = ConfigStore.instance()
    cs.store(name="encoding_config_schema", node=encoding_Config)
